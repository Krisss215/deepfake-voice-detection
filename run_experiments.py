#!/usr/bin/env python3
"""
run_experiments.py - every experiment of the project plan (stage 2) on one dataset.

    python run_experiments.py --data data [--ssl-models data/models/wav2vec2-base data/models/wavlm-base-plus]

Writes results/results.json and CSV tables. All scores are out-of-fold: a clip is
scored only by a model that never saw its speaker. Hyper-parameters (C, SSL layer)
are chosen inside each outer fold (nested CV), so reported numbers are not tuned on
the data they are measured on.
"""
import argparse, json, os, time, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import features as F

warnings.filterwarnings("ignore")
SEED, OUTER, INNER = 42, 5, 3
C_GRID = [0.001, 0.01, 0.1, 1.0]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# --------------------------------------------------------------------------- metrics
def eer(y, s):
    y, s = np.asarray(y), np.asarray(s)
    if len(np.unique(y)) < 2:
        return np.nan
    fpr, tpr, thr = roc_curve(y, s)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[i] + fnr[i]) / 2)


def eer_threshold(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    return float(thr[np.nanargmin(np.abs((1 - tpr) - fpr))])


def bootstrap_ci(y, s, groups, n=500, seed=SEED):
    """95% CI of the EER, resampling whole speakers."""
    rng = np.random.default_rng(seed)
    g = pd.Series(groups)
    idx_by = {k: np.flatnonzero(g.values == k) for k in g.unique()}
    keys = list(idx_by)
    vals = []
    for _ in range(n):
        pick = np.concatenate([idx_by[k] for k in rng.choice(keys, len(keys))])
        vals.append(eer(y[pick], s[pick]))
    return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]


# --------------------------------------------------------------------------- models
def make_model(C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))


def pick_C(X, y, g):
    """Inner group CV: the C with the lowest mean EER."""
    cv = StratifiedGroupKFold(INNER, shuffle=True, random_state=SEED)
    best, best_c = np.inf, C_GRID[-1]
    for C in C_GRID:
        e = []
        for tr, te in cv.split(X, y, g):
            if len(np.unique(y[te])) < 2:
                continue
            e.append(eer(y[te], make_model(C).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]))
        if e and np.mean(e) < best:
            best, best_c = np.mean(e), C
    return best_c


def fit_predict(Xtr, ytr, gtr, Xte, layers=False):
    """Fit with an inner-CV choice of C (and of the SSL layer if X is 3-D); score Xte."""
    if not layers:
        C = pick_C(Xtr, ytr, gtr)
        return make_model(C).fit(Xtr, ytr).predict_proba(Xte)[:, 1], dict(C=C)
    best = (np.inf, None, None)
    cv = StratifiedGroupKFold(INNER, shuffle=True, random_state=SEED)
    for L in range(Xtr.shape[0]):
        e = [eer(ytr[te], make_model(0.01).fit(Xtr[L][tr], ytr[tr]).predict_proba(Xtr[L][te])[:, 1])
             for tr, te in cv.split(Xtr[L], ytr, gtr)]
        if np.mean(e) < best[0]:
            best = (np.mean(e), L, None)
    L = best[1]
    C = pick_C(Xtr[L], ytr, gtr)
    return make_model(C).fit(Xtr[L], ytr).predict_proba(Xte[L])[:, 1], dict(C=C, layer=int(L))


def oof(X, y, g, mask_train=None, layers=False, random_split=False):
    """Out-of-fold scores with speaker-disjoint outer folds. mask_train limits what may be trained on."""
    n = len(y)
    s = np.full(n, np.nan)
    cv = (StratifiedKFold(OUTER, shuffle=True, random_state=SEED) if random_split
          else StratifiedGroupKFold(OUTER, shuffle=True, random_state=SEED))
    params = []
    Xi = X[0] if layers else X
    for tr, te in cv.split(Xi, y, g):
        if mask_train is not None:
            tr = tr[mask_train[tr]]
        Xtr = X[:, tr] if layers else X[tr]
        Xte = X[:, te] if layers else X[te]
        s[te], p = fit_predict(Xtr, y[tr], g[tr], Xte, layers)
        params.append(p)
    return s, params


# --------------------------------------------------------------------------- reports
def breakdown(df, y, s, name):
    """EER overall, within each source, per spoofing system (vs bonafide of the same source)."""
    row = {"model": name, "EER_all": eer(y, s)}
    two = [src for src in df.source.unique() if df.loc[df.source == src, "label"].nunique() == 2]
    for src in sorted(df.source.unique()):
        m = (df.source == src).values
        row[f"EER_{src}"] = eer(y[m], s[m])
    m = df.source.isin(two).values
    row["EER_excl_real_only"] = eer(y[m], s[m])
    thr = eer_threshold(y, s)
    pred = s >= thr
    row["miss_rate"] = float(np.mean(~pred[y == 1]))
    row["false_alarm"] = float(np.mean(pred[y == 0]))
    per_sys = []
    for (src, sysname), grp in df[df.label == "spoof"].groupby(["source", "system"]):
        mm = ((df.source == src) & ((df.label == "bonafide") | (df.system == sysname))).values
        per_sys.append(dict(model=name, source=src, system=sysname, n_spoof=len(grp), EER=eer(y[mm], s[mm])))
    return row, per_sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--ssl-models", nargs="*", default=[])
    ap.add_argument("--quick", action="store_true", help="skip the slow per-system leave-one-out runs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df = pd.read_csv(os.path.join(a.data, "train.csv"))
    y = (df.label == "spoof").astype(int).values
    g = df.speaker.astype(str).values
    log(f"{len(df)} clips, {df.speaker.nunique()} speakers")
    print(pd.crosstab(df.source, df.label, margins=True))

    feats = {k: F.extract(df, a.data, k) for k in F.HANDCRAFTED}
    feats_trim = {k: F.extract(df, a.data, k, trim=True) for k in ["mfcc", "lfcc"]}
    ssl = {}
    for md in a.ssl_models:
        tag = os.path.basename(md.rstrip("/"))
        log(f"SSL features: {tag}")
        ssl[tag] = F.extract_ssl(df, a.data, md, tag)

    summary, per_system, notes = [], [], {}

    def run(name, X, layers=False, **kw):
        log(f"run {name}")
        s, params = oof(X, y, g, layers=layers, **kw)
        row, ps = breakdown(df, y, s, name)
        row["CI95_all"] = bootstrap_ci(y, s, g)
        row["params"] = params
        summary.append(row)
        per_system.extend(ps)
        return s

    # 1. baseline + approach A + approach B, speaker-disjoint
    names = {"mfcc": "Baseline: MFCC+LR", "lfcc": "A: LFCC+LR", "logspec": "A: log-spectrogram+LR"}
    names.update({tag: f"B: {tag}+LR" for tag in ssl})
    for k in ["mfcc", "lfcc", "logspec"]:
        run(names[k], feats[k])
    for tag, X in ssl.items():
        run(names[tag], X, layers=True)

    # 2. checks for leakage and shortcuts
    run("Check: MFCC, random split (mixes speakers)", feats["mfcc"], random_split=True)
    src_onehot = pd.get_dummies(df.source).values.astype(float)
    run("Check: source only", src_onehot)
    run("Check: silence only", F.extract(df, a.data, "silence"))
    run("Ablation: MFCC after silence trim", feats_trim["mfcc"])
    run("Ablation: LFCC after silence trim", feats_trim["lfcc"])
    real_only = [src for src in df.source.unique() if df.loc[df.source == src, "label"].nunique() == 1]
    if real_only:  # M-AILABS, or LibriSpeech when it stands in for M-AILABS
        run(f"Ablation: MFCC, trained without {'+'.join(real_only)}", feats["mfcc"],
            mask_train=(~df.source.isin(real_only)).values)
    cvs = StratifiedGroupKFold(OUTER, shuffle=True, random_state=SEED)
    pred = np.empty(len(df), dtype=object)
    for tr, te in cvs.split(feats["mfcc"], df.source, g):
        pred[te] = make_model(1.0).fit(feats["mfcc"][tr], df.source.values[tr]).predict(feats["mfcc"][te])
    notes["source_classifier_accuracy_mfcc"] = float(accuracy_score(df.source, pred))
    notes["source_majority_rate"] = float(df.source.value_counts(normalize=True).max())

    # 3. cross-dataset: train on one source with spoofs, test on the other
    cross = []
    two = [s for s in sorted(df.source.unique()) if df.loc[df.source == s, "label"].nunique() == 2]
    allfeats = dict(feats, **ssl)
    for fname, X in allfeats.items():
        layers = X.ndim == 3
        for tr_src in two:
            for te_src in two:
                if tr_src == te_src:
                    continue
                tr = (df.source == tr_src).values
                te = (df.source == te_src).values
                Xtr = X[:, tr] if layers else X[tr]
                Xte = X[:, te] if layers else X[te]
                s, p = fit_predict(Xtr, y[tr], g[tr], Xte, layers)
                within = next(r[f"EER_{te_src}"] for r in summary if r["model"] == names[fname])
                cross.append(dict(features=fname, train=tr_src, test=te_src,
                                  EER_cross=eer(y[te], s), EER_within_test_source=within, params=p))
        log(f"cross-dataset done: {fname}")

    # 4. leave-one-system-out inside each source (unseen attack)
    loso = []
    if not a.quick:
        for fname, X in allfeats.items():
            layers = X.ndim == 3
            for (src, sysname), _ in df[df.label == "spoof"].groupby(["source", "system"]):
                keep = ~((df.source == src) & (df.system == sysname)).values  # never train on this system
                evalm = ((df.source == src) & ((df.label == "bonafide") | (df.system == sysname))).values
                s, _ = oof(X, y, g, mask_train=keep, layers=layers)
                loso.append(dict(features=fname, source=src, system=sysname,
                                 n_spoof=int(((df.source == src) & (df.system == sysname)).sum()),
                                 EER_unseen=eer(y[evalm], s[evalm])))
            log(f"leave-one-system-out done: {fname}")

    pd.DataFrame(summary).drop(columns=["params"]).to_csv(os.path.join(a.out, "summary.csv"), index=False)
    pd.DataFrame(per_system).to_csv(os.path.join(a.out, "per_system.csv"), index=False)
    pd.DataFrame(cross).drop(columns=["params"]).to_csv(os.path.join(a.out, "cross_dataset.csv"), index=False)
    pd.DataFrame(loso).to_csv(os.path.join(a.out, "leave_one_system_out.csv"), index=False)
    json.dump(dict(summary=summary, per_system=per_system, cross_dataset=cross, loso=loso, notes=notes,
                   n=len(df), speakers=int(df.speaker.nunique())),
              open(os.path.join(a.out, "results.json"), "w"), indent=1, default=float)
    pd.set_option("display.width", 200)
    print(pd.DataFrame(summary).drop(columns=["params", "CI95_all"]).round(3).to_string())
    print(pd.DataFrame(cross).drop(columns=["params"]).round(3).to_string())
    print(notes)
    log("done")


if __name__ == "__main__":
    main()
