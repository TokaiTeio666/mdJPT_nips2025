"""LOSO readout probes + spectral simple baseline for the four-source experiment.

For every feature set in overnight/features/*/ (random seeds, ISA/CDA/ISA+CDA seeds)
and for the log-bandpower simple baseline (computed from the built sliced windows),
fits leave-one-subject-out probes with per-fold standardization:

  regression    Ridge(alpha=10)            DASPS sam_valence/sam_arousal,
                                           DEAP valence/arousal, SAM40 stress
  classification LogisticRegression(C=1)   SAM40 task (4-way), EEGMAT condition (2-way)

Baselines: constant train-mean (reg) / majority class (clf).
Writes overnight/results.json and prints a compact table.
"""
import argparse
import glob
import importlib.util
import json
import os
import sys

import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = 'D:/大创/mdjpt_data_v4'
BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]

REG_TARGETS = {'DASPS': ['sam_valence', 'sam_arousal'],
               'DEAP': ['valence', 'arousal'],
               'SAM40': ['stress']}
CLF_TARGETS = {'SAM40': ('task', 4), 'EEGMAT': ('condition', 2)}


def loso_eval(X, y, groups, kind):
    """Pooled-LOSO predictions + trivial baseline predictions."""
    preds = np.empty(len(y))
    trivial = np.empty(len(y))
    for g in np.unique(groups):
        te = groups == g
        tr = ~te
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        if kind == 'reg':
            m = Ridge(alpha=10.0).fit(Xtr, y[tr])
            preds[te] = m.predict(Xte)
            trivial[te] = y[tr].mean()
        else:
            m = LogisticRegression(max_iter=500, C=1.0).fit(Xtr, y[tr])
            preds[te] = m.predict(Xte)
            vals, cnts = np.unique(y[tr], return_counts=True)
            trivial[te] = vals[np.argmax(cnts)]
    return preds, trivial


def metrics(y, preds, trivial, kind):
    if kind == 'reg':
        return {'mse': float(np.mean((y - preds) ** 2)),
                'trivial_mse': float(np.mean((y - trivial) ** 2)),
                'pearson_r': float(np.corrcoef(y, preds)[0, 1])
                if np.std(preds) > 1e-9 else 0.0}
    return {'acc': float(accuracy_score(y, preds)),
            'trivial_acc': float(accuracy_score(y, trivial)),
            'balanced_acc': float(balanced_accuracy_score(y, preds))}


def eval_feature_matrix(X, targets, subject_idx, out, label):
    for tname, y in targets.items():
        if tname in ('task', 'condition'):
            kind = 'clf'
        else:
            y = np.asarray(y, float)
            ok = np.isfinite(y)
            if ok.sum() < len(y):  # e.g. SAM40 stress with missing ratings
                X2, y2, g2 = X[ok], y[ok], subject_idx[ok]
            else:
                X2, y2, g2 = X, y, subject_idx
            kind = 'reg'
        if kind == 'clf':
            X2, y2, g2 = X, np.asarray(y, int), subject_idx
        preds, trivial = loso_eval(X2, y2, g2, kind)
        m = metrics(y2, preds, trivial, kind)
        out.append({'features': label, 'target': tname, 'kind': kind, **m})
        print(f"  {label:14s} {tname:12s} "
              f"{ {k: round(v, 4) for k, v in m.items()} }")


def bandpower_features(src, n_windows):
    """Log band powers (5 bands x 60 ch = 300-dim) from the built sliced windows,
    in the same canonical file order as the extracted features."""
    d = os.path.join(DATA_ROOT, src, 'sliced_data', 'sliced_len5_step2', 'data')
    freqs, P = welch(np.load(os.path.join(d, 'data_sample_0.npy')), fs=125, nperseg=625)
    bands = []
    for lo, hi in BANDS:
        bands.append((freqs >= lo) & (freqs < hi))
    X = np.empty((n_windows, 60 * len(BANDS)), np.float32)
    for i in range(n_windows):
        w = np.load(os.path.join(d, f'data_sample_{i}.npy'))
        _, P = welch(w, fs=125, nperseg=625)
        feats = [np.log10(P[:, b].sum(axis=1) + 1e-12) for b in bands]
        X[i] = np.concatenate(feats)
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features-dir', default=os.path.join(REPO, 'overnight', 'features'))
    ap.add_argument('--out', default=os.path.join(REPO, 'overnight', 'results.json'))
    ap.add_argument('--skip-simple', action='store_true')
    args = ap.parse_args()

    sets = sorted(glob.glob(os.path.join(args.features_dir, '*')))
    sets = [s for s in sets if os.path.isdir(s)]
    assert sets, 'no feature sets found'
    results = []

    for src in ['DASPS', 'DEAP', 'SAM40', 'EEGMAT']:
        ref = np.load(os.path.join(sets[0], f'{src}.npz'))
        n = len(ref['subject_idx'])
        targets = {k: ref[k] for k in ref.files if k not in ('fea', 'subject_idx', 'group_idx')}
        subj = ref['subject_idx']
        print(f'== {src} (N={n}) ==')
        for s in sets:
            name = os.path.basename(s)
            f = np.load(os.path.join(s, f'{src}.npz'))
            assert len(f['subject_idx']) == n
            eval_feature_matrix(f['fea'].astype(np.float64), targets, subj, results, name)
        if not args.skip_simple:
            print(f'  computing spectral baseline ({n} windows)...')
            X = bandpower_features(src, n).astype(np.float64)
            eval_feature_matrix(X, targets, subj, results, 'spectral_simple')

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f'-> {args.out} ({len(results)} rows)')


if __name__ == '__main__':
    main()
