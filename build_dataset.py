#!/usr/bin/env python3
"""
build_dataset.py - builds the course project dataset (1,520 clips, 4 s, 16 kHz mono)
from the three public corpora named in the project brief, and writes train.csv.

    ASVspoof 5 (dev set, attacks A09-A16) : 300 bonafide + 300 spoof
    CodecFake  (VCTK re-synthesised by neural codecs) : 300 spoof
    VCTK 0.92  (the real recordings CodecFake was made from) : 300 bonafide
    M-AILABS   (en_US audiobooks, real only) : 320 bonafide
    -> if the M-AILABS servers are unreachable (they were down in Nov 2026), the same
       role is filled by LibriSpeech dev-clean (OpenSLR 12): also real LibriVox
       audiobooks, 320 bonafide, source "LibriSpeech". Use --only librispeech.

Only the clips that are kept are stored. Archives are read as streams or with
HTTP range requests, so the full corpora (100+ GB) are never saved to disk.

Usage (on your own computer, which has normal internet access):
    pip install numpy pandas soundfile soxr requests pyarrow huggingface_hub remotezip
    python build_dataset.py --out data

Every source can also be given as a local file instead of a URL (see --help),
e.g. if you already downloaded an archive in the browser.
Re-running is safe: a finished source is skipped (delete its folder to redo it).
"""
import argparse, io, json, os, random, re, sys, tarfile, time, zipfile
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import soundfile as sf

SR = 16000
CLIP_S = 4.0
MIN_DUR = 3.0   # shorter source clips are skipped: long zero-padding would be an easy shortcut
SEED = 42

ASV_PROTOCOLS = "https://zenodo.org/records/14498691/files/ASVspoof5_protocols.tar.gz?download=1"
ASV_DEV_TARS = [f"https://zenodo.org/records/14498691/files/flac_D_a{c}.tar?download=1" for c in "abc"]
CODECFAKE_REPO = "datasets/rogertseng/CodecFake"
VCTK_ZIP = "https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip"
MAILABS_TGZ = ["https://data.solak.de/data/Training/stt_tts/en_US.tgz",
               "http://www.caito.de/data/Training/stt_tts/en_US.tgz"]
# LibriSpeech dev-clean: 40 speakers, ~340 MB, 16 kHz FLAC, CC BY 4.0. Official site + mirrors.
LIBRI_TGZ = ["https://www.openslr.org/resources/12/dev-clean.tar.gz",
             "https://us.openslr.org/resources/12/dev-clean.tar.gz",
             "https://openslr.elda.org/resources/12/dev-clean.tar.gz"]


# --------------------------------------------------------------------------- helpers
def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def is_url(p):
    return str(p).startswith(("http://", "https://"))


def open_stream(src):
    """A readable, non-seekable stream for a URL or a local path."""
    if is_url(src):
        import requests
        r = requests.get(src, stream=True, timeout=60)
        r.raise_for_status()
        r.raw.decode_content = True
        return r.raw
    return open(src, "rb")


def to_clip(y, sr, rng):
    """mono -> 16 kHz -> exactly 4 s (random window if longer, zero-pad if shorter)."""
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        import soxr
        y = soxr.resample(y, sr, SR)
    n = int(SR * CLIP_S)
    if len(y) >= n:
        start = rng.integers(0, len(y) - n + 1)
        y = y[start:start + n]
    else:
        y = np.pad(y, (0, n - len(y)))
    return y.astype(np.float32)


def with_retry(fn, tries=4, wait=5):
    """Network reads sometimes drop; try again a few times before giving up."""
    for k in range(tries):
        try:
            return fn()
        except Exception as e:
            if k == tries - 1:
                raise
            log(f"  read failed ({e}); retrying in {wait * (k + 1)} s")
            time.sleep(wait * (k + 1))


def decode(data_bytes):
    y, sr = sf.read(io.BytesIO(data_bytes), dtype="float32", always_2d=False)
    return y, sr


class Writer:
    """Writes clips of one source into out/wav/<source>/ and remembers the rows."""

    def __init__(self, out, source):
        self.dir = os.path.join(out, "wav", source)
        os.makedirs(self.dir, exist_ok=True)
        self.source, self.rows, self.rng = source, [], np.random.default_rng(SEED)

    def add(self, y, sr, label, speaker, system, orig):
        cid = f"{self.source}_{len(self.rows):04d}"
        clip = to_clip(y, sr, self.rng)
        sf.write(os.path.join(self.dir, cid + ".wav"), clip, SR, subtype="PCM_16")
        self.rows.append(dict(id=cid, file=f"wav/{self.source}/{cid}.wav", label=label,
                              source=self.source, speaker=str(speaker), system=system,
                              orig_file=orig, orig_sr=sr, orig_dur=round(len(y) / sr, 2)))

    def done(self, out):
        pd.DataFrame(self.rows).to_csv(os.path.join(out, f"part_{self.source}.csv"), index=False)
        log(f"{self.source}: wrote {len(self.rows)} clips")


def finished(out, source):
    return os.path.exists(os.path.join(out, f"part_{source}.csv"))


def fair_quota(groups, total, rng):
    """Split `total` as evenly as possible over the keys in `groups`."""
    keys = sorted(groups)
    base, extra = divmod(total, len(keys))
    order = list(keys)
    rng.shuffle(order)
    return {k: base + (1 if i < extra else 0) for i, k in enumerate(order)}


# --------------------------------------------------------------------------- ASVspoof 5
def parse_asv_protocol(text):
    """Rows of the ASVspoof 5 dev protocol -> dict file_id -> (speaker, label, attack)."""
    meta = {}
    for line in text.splitlines():
        tok = line.split()
        if len(tok) < 3:
            continue
        key = next((t for t in tok if t in ("bonafide", "spoof")), None)
        if key is None:
            continue
        attack = next((t for t in tok if re.fullmatch(r"A\d\d", t)), "-")
        meta[tok[1]] = (tok[0], key, attack if key == "spoof" else "-")
    return meta


def load_asv_protocol(src):
    """The dev-set protocol (ASVspoof5.dev.track_1.tsv) from the protocols archive."""
    found = {}
    with tarfile.open(fileobj=open_stream(src), mode="r|*") as tf:
        for m in tf:
            base = os.path.basename(m.name).lower()
            if m.isfile() and "dev" in base and base.endswith((".tsv", ".txt")):
                found[base] = tf.extractfile(m).read().decode()
    for base in sorted(found, key=lambda b: ("track_1" not in b, "trial" in b, b)):
        meta = parse_asv_protocol(found[base])
        if meta:
            log(f"ASVspoof5: using protocol {base}")
            return meta
    raise RuntimeError(f"ASVspoof5 dev protocol not found in {src} (saw: {sorted(found)})")


def build_asvspoof(out, protocols, tars, n_bona=300, n_spoof=300, max_per_spk=3):
    if finished(out, "ASVspoof5"):
        return log("ASVspoof5: already done, skipping")
    meta = load_asv_protocol(protocols)
    attacks = sorted({a for _, k, a in meta.values() if k == "spoof"})
    log(f"ASVspoof5 protocol: {len(meta)} dev files, attacks {attacks}")
    rng = np.random.default_rng(SEED)
    quota = fair_quota(attacks, n_spoof, rng)
    got, per_spk, w = Counter(), Counter(), Writer(out, "ASVspoof5")
    need = lambda: got["bonafide"] < n_bona or any(got[a] < q for a, q in quota.items())
    for src in tars:
        if not need():
            break
        log(f"ASVspoof5: streaming {src}")
        with tarfile.open(fileobj=open_stream(src), mode="r|*") as tf:
            for m in tf:
                if not need():
                    break
                fid = os.path.splitext(os.path.basename(m.name))[0]
                if not m.isfile() or fid not in meta:
                    continue
                spk, key, att = meta[fid]
                slot = "bonafide" if key == "bonafide" else att
                cap = n_bona if slot == "bonafide" else quota.get(slot, 0)
                if got[slot] >= cap or per_spk[(spk, key)] >= max_per_spk:
                    continue
                y, sr = decode(tf.extractfile(m).read())
                if len(y) / sr < MIN_DUR:
                    continue
                w.add(y, sr, key, spk, att, m.name)
                got[slot] += 1
                per_spk[(spk, key)] += 1
                if len(w.rows) % 100 == 0:
                    log(f"ASVspoof5: {len(w.rows)} clips {dict(got)}")
    if need():
        log(f"WARNING ASVspoof5 quotas not met: {dict(got)}")
    w.done(out)


# --------------------------------------------------------------------------- CodecFake
def build_codecfake(out, repo, n_spoof=300, n_codecs=10, max_per_spk=8, local_parquets=None):
    if finished(out, "CodecFake"):
        return log("CodecFake: already done, skipping")
    import pyarrow.parquet as pq
    if local_parquets:
        files, opener = sorted(local_parquets), (lambda p: open(p, "rb"))
    else:
        from huggingface_hub import HfFileSystem
        fs = HfFileSystem()
        files = sorted(fs.glob(f"{repo}/data/train-*.parquet"))
        opener = lambda p: fs.open(p, "rb", block_size=4 << 20)
    log(f"CodecFake: {len(files)} parquet shards")
    # 1) find which codec each shard holds - reads only one small column per shard
    rng = random.Random(SEED)
    probe = files if len(files) <= 40 else files[::max(1, len(files) // 40)]
    shard_codec = {}
    for p in probe:
        pf = pq.ParquetFile(opener(p))
        names = pf.read_row_group(0, columns=["codec_name"]).column(0).to_pylist()
        shard_codec[p] = Counter(names).most_common(1)[0][0]
    by_codec = defaultdict(list)
    for p, c in shard_codec.items():
        by_codec[c].append(p)
    codecs = sorted(by_codec)
    rng.shuffle(codecs)
    codecs = sorted(codecs[:n_codecs])
    log(f"CodecFake: using codecs {codecs}")
    quota = fair_quota(codecs, n_spoof, np.random.default_rng(SEED))
    w, per_spk = Writer(out, "CodecFake"), Counter()
    for c in codecs:
        got = 0
        pf = pq.ParquetFile(opener(rng.choice(by_codec[c])))
        groups = list(range(pf.num_row_groups))
        rng.shuffle(groups)
        for g in groups:
            if got >= quota[c]:
                break
            t = pf.read_row_group(g, columns=["audio", "label", "speaker_id", "codec_name"]).to_pylist()
            rng.shuffle(t)
            for r in t:
                if got >= quota[c]:
                    break
                if r["codec_name"] != c or per_spk[(r["speaker_id"], c)] >= max_per_spk:
                    continue
                a = r["audio"]
                y, sr = decode(a["bytes"])
                if len(y) / sr < MIN_DUR:
                    continue
                w.add(y, sr, "spoof", r["speaker_id"], c, a.get("path") or "")
                got += 1
                per_spk[(r["speaker_id"], c)] += 1
        log(f"CodecFake: {c}: {got}/{quota[c]}")
    w.done(out)


# --------------------------------------------------------------------------- VCTK (real side of CodecFake)
def build_vctk(out, src, speakers, n_bona=300, max_per_spk=4):
    if finished(out, "VCTK"):
        return log("VCTK: already done, skipping")
    if is_url(src):
        from remotezip import RemoteZip
        zf = with_retry(lambda: RemoteZip(src))
    else:
        zf = zipfile.ZipFile(src)
    names = [n for n in zf.namelist() if re.search(r"wav48_silence_trimmed/p\d+/p\d+_\d+_mic1\.flac$", n)]
    by_spk = defaultdict(list)
    for n in names:
        by_spk[n.split("/")[-2]].append(n)
    pool = sorted(s for s in by_spk if not speakers or s in speakers) or sorted(by_spk)
    log(f"VCTK: {len(names)} mic1 files, {len(pool)} speakers in pool")
    rng = random.Random(SEED)
    quota = fair_quota(pool, n_bona, np.random.default_rng(SEED))
    w = Writer(out, "VCTK")
    for s in pool:
        got = 0
        for n in rng.sample(by_spk[s], len(by_spk[s])):
            if got >= quota[s]:
                break
            try:
                y, sr = decode(with_retry(lambda: zf.read(n)))
            except Exception as e:
                log(f"VCTK: skipping {n}: {e}")
                continue
            if len(y) / sr < MIN_DUR:
                continue
            w.add(y, sr, "bonafide", s, "-", n)
            got += 1
            if len(w.rows) % 25 == 0:
                log(f"VCTK: {len(w.rows)}/{n_bona} clips")
    w.done(out)


# --------------------------------------------------------------------------- M-AILABS
def build_mailabs(out, sources, n_bona=320, max_per_book=10):
    if finished(out, "M-AILABS"):
        return log("M-AILABS: already done, skipping")
    w, per_book, last_err = Writer(out, "M-AILABS"), Counter(), None
    for src in sources:
        try:
            log(f"M-AILABS: streaming {src}")
            with tarfile.open(fileobj=open_stream(src), mode="r|*") as tf:
                for m in tf:
                    if len(w.rows) >= n_bona:
                        break
                    if not (m.isfile() and m.name.endswith(".wav") and "/wavs/" in m.name):
                        continue
                    parts = m.name.split("/")
                    book, speaker = parts[-3], parts[-4]
                    if per_book[(speaker, book)] >= max_per_book:
                        continue
                    y, sr = decode(tf.extractfile(m).read())
                    if len(y) / sr < MIN_DUR:
                        continue
                    w.add(y, sr, "bonafide", speaker, "-", m.name)
                    per_book[(speaker, book)] += 1
                    if len(w.rows) % 40 == 0:
                        log(f"M-AILABS: {len(w.rows)}/{n_bona} clips")
            break
        except Exception as e:  # try the next mirror
            last_err = e
            log(f"M-AILABS: {src} failed: {e}")
    if not w.rows:
        raise RuntimeError(f"M-AILABS: no source worked ({last_err})")
    w.done(out)


def build_librispeech(out, sources, n_bona=320, max_per_spk=8, max_per_chapter=4):
    """Real audiobook speech in place of M-AILABS (same origin: LibriVox recordings).
    dev-clean has 40 speakers; at most 8 clips per speaker and 4 per chapter, so the
    320 clips cover all speakers and many chapters instead of a few long readings."""
    if finished(out, "LibriSpeech"):
        return log("LibriSpeech: already done, skipping")
    last_err = None
    for src in sources:
        w, per_spk, per_ch = Writer(out, "LibriSpeech"), Counter(), Counter()
        try:
            log(f"LibriSpeech: streaming {src}")
            with tarfile.open(fileobj=open_stream(src), mode="r|*") as tf:
                for m in tf:
                    if len(w.rows) >= n_bona:
                        break
                    if not (m.isfile() and m.name.endswith(".flac")):
                        continue
                    parts = m.name.split("/")          # LibriSpeech/dev-clean/<spk>/<chapter>/<utt>.flac
                    speaker, chapter = parts[-3], parts[-2]
                    if per_spk[speaker] >= max_per_spk or per_ch[(speaker, chapter)] >= max_per_chapter:
                        continue
                    y, sr = decode(tf.extractfile(m).read())
                    if len(y) / sr < MIN_DUR:
                        continue
                    w.add(y, sr, "bonafide", speaker, "-", m.name)
                    per_spk[speaker] += 1
                    per_ch[(speaker, chapter)] += 1
                    if len(w.rows) % 40 == 0:
                        log(f"LibriSpeech: {len(w.rows)}/{n_bona} clips")
            if w.rows:
                w.done(out)
                return
        except Exception as e:  # try the next mirror from scratch
            last_err = e
            log(f"LibriSpeech: {src} failed: {e}")
    raise RuntimeError(f"LibriSpeech: no source worked ({last_err})")


# --------------------------------------------------------------------------- main
def merge(out):
    parts = [pd.read_csv(os.path.join(out, f)) for f in sorted(os.listdir(out)) if f.startswith("part_")]
    df = pd.concat(parts, ignore_index=True)
    # The brief treats VCTK as the real half of CodecFake: same "source" for both.
    df["corpus"] = df["source"]
    df.loc[df["source"] == "VCTK", "source"] = "CodecFake"
    df["speaker"] = df["corpus"].replace({"VCTK": "CodecFake"}) + ":" + df["speaker"].astype(str)
    df = df[["id", "file", "label", "source", "corpus", "speaker", "system", "orig_file", "orig_sr", "orig_dur"]]
    df.to_csv(os.path.join(out, "train.csv"), index=False)
    print("\n" + pd.crosstab(df["source"], df["label"], margins=True).to_string())
    json.dump(dict(built=time.strftime("%Y-%m-%d %H:%M"), n=len(df), sources=dict(
        ASVspoof5=ASV_DEV_TARS[0], CodecFake=CODECFAKE_REPO, VCTK=VCTK_ZIP, MAILABS=MAILABS_TGZ[0], LibriSpeech=LIBRI_TGZ[0])),
        open(os.path.join(out, "build_info.json"), "w"), indent=1)
    log(f"train.csv: {len(df)} rows -> {os.path.join(out, 'train.csv')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data")
    ap.add_argument("--only", nargs="*", choices=["asv", "codecfake", "vctk", "mailabs", "librispeech"],
                    help="build only these (default: all except librispeech)")
    ap.add_argument("--asv-protocols", default=ASV_PROTOCOLS)
    ap.add_argument("--asv-tars", nargs="*", default=ASV_DEV_TARS)
    ap.add_argument("--codecfake-parquets", nargs="*", help="local CodecFake .parquet files instead of Hugging Face")
    ap.add_argument("--vctk", default=VCTK_ZIP)
    ap.add_argument("--mailabs", nargs="*", default=MAILABS_TGZ)
    ap.add_argument("--librispeech", nargs="*", default=LIBRI_TGZ)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    todo = set(a.only or ["asv", "codecfake", "vctk", "mailabs"])
    if "asv" in todo:
        build_asvspoof(a.out, a.asv_protocols, a.asv_tars)
    if "codecfake" in todo:
        build_codecfake(a.out, CODECFAKE_REPO, local_parquets=a.codecfake_parquets)
    if "vctk" in todo:
        spk = set()
        p = os.path.join(a.out, "part_CodecFake.csv")
        if os.path.exists(p):
            spk = set(pd.read_csv(p)["speaker"].astype(str))
        build_vctk(a.out, a.vctk, spk)
    if "mailabs" in todo:
        build_mailabs(a.out, a.mailabs)
    if "librispeech" in todo:
        build_librispeech(a.out, a.librispeech)
    merge(a.out)


if __name__ == "__main__":
    main()
