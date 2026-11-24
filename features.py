"""Feature extraction for the deepfake-voice project. Results are cached in <data>/features/."""
import os
import numpy as np
import librosa
import scipy.fftpack

SR = 16000
N_FFT, HOP = 512, 160


def load(path, trim=False):
    y, _ = librosa.load(path, sr=SR, mono=True)
    if trim:
        yt, _ = librosa.effects.trim(y, top_db=35)
        y = yt if len(yt) > SR // 4 else y
    return y


def _stats(f):
    """frames (d x T) + deltas -> mean and std over time."""
    f = np.vstack([f, librosa.feature.delta(f), librosa.feature.delta(f, order=2)])
    return np.concatenate([f.mean(axis=1), f.std(axis=1)])


def mfcc(y, n=20):
    return _stats(librosa.feature.mfcc(y=y, sr=SR, n_mfcc=n, n_fft=N_FFT, hop_length=HOP, n_mels=40))


def _linear_fbank(n_filters=40):
    freqs = np.linspace(0, SR / 2, N_FFT // 2 + 1)
    edges = np.linspace(0, SR / 2, n_filters + 2)
    fb = np.zeros((n_filters, len(freqs)))
    for i in range(n_filters):
        lo, c, hi = edges[i], edges[i + 1], edges[i + 2]
        fb[i] = np.clip(np.minimum((freqs - lo) / (c - lo), (hi - freqs) / (hi - c)), 0, None)
    return fb


_FB = _linear_fbank()


def lfcc(y, n=20):
    """Like MFCC but with linearly spaced filters: high frequencies keep their resolution."""
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP)) ** 2
    E = np.log(_FB @ S + 1e-10)
    return _stats(scipy.fftpack.dct(E, type=2, axis=0, norm="ortho")[:n])


def logspec(y):
    """Mean and std of the log power spectrum per frequency bin (257 bins, linear scale)."""
    S = np.log(np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP)) ** 2 + 1e-10)
    return np.concatenate([S.mean(axis=1), S.std(axis=1)])


def silence_stats(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    _, (a, b) = librosa.effects.trim(y, top_db=35)
    return np.array([a / SR, (len(y) - b) / SR])


HANDCRAFTED = {"mfcc": mfcc, "lfcc": lfcc, "logspec": logspec}


def extract(df, data_dir, name, trim=False):
    """Matrix of features `name` for every row of df (cached)."""
    cache = os.path.join(data_dir, "features", f"{name}{'_trim' if trim else ''}.npy")
    if os.path.exists(cache):
        X = np.load(cache)
        if len(X) == len(df):
            return X
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    if name == "silence":
        X = np.stack([silence_stats(os.path.join(data_dir, f)) for f in df["file"]])
    else:
        fn = HANDCRAFTED[name]
        X = np.stack([fn(load(os.path.join(data_dir, f), trim)) for f in df["file"]])
    np.save(cache, X)
    return X


def extract_ssl(df, data_dir, model_dir, tag, deadline=None):
    """Mean-pooled hidden states of a frozen wav2vec2/WavLM model: array (n_layers+1, n_clips, dim).
    With a deadline (time.time() value) it saves progress and returns None when time is up;
    the next call continues from the saved clips."""
    cache = os.path.join(data_dir, "features", f"ssl_{tag}.npy")
    if os.path.exists(cache):
        X = np.load(cache)
        if X.shape[1] == len(df):
            return X
    import torch
    from transformers import AutoFeatureExtractor, AutoModel
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    fe = AutoFeatureExtractor.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir).eval()
    import time
    partial = os.path.join(data_dir, "features", f"ssl_{tag}_partial.npy")
    os.makedirs(os.path.dirname(partial), exist_ok=True)
    out = list(np.load(partial)) if os.path.exists(partial) else []
    if out:
        print(f"  ssl {tag}: resuming at {len(out)}/{len(df)}", flush=True)
    with torch.no_grad():
        for i, f in enumerate(df["file"]):
            if i < len(out):
                continue
            if deadline and time.time() > deadline:
                np.save(partial, np.stack(out).astype(np.float32))
                print(f"  ssl {tag}: paused at {len(out)}/{len(df)}", flush=True)
                return None
            y = load(os.path.join(data_dir, f))
            inp = fe(y, sampling_rate=SR, return_tensors="pt")
            hs = model(**inp, output_hidden_states=True).hidden_states
            out.append(torch.stack([h[0].mean(0) for h in hs]).numpy())
            if (i + 1) % 100 == 0:
                print(f"  ssl {tag}: {i + 1}/{len(df)}", flush=True)
                np.save(partial, np.stack(out).astype(np.float32))
    X = np.stack(out, axis=1).astype(np.float32)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.save(cache, X)
    return X
