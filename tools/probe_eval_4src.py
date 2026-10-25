"""LOSO readout probes + spectral simple baseline for the four-source experiment.

For every feature set in overnight/features/*/ (random seeds, ISA/CDA/ISA+CDA seeds)
and for the log-bandpower simple baseline (computed once from the built sliced
windows, cached in overnight/bandpower_cache/), fits leave-one-subject-out probes
with per-fold standardization:

  regression    Ridge(alpha=10)            DASPS sam_valence/sam_arousal,
                                           DEAP valence/arousal, SAM40 stress
  classification LogisticRegression(C=1)   SAM40 task (4-way), EEGMAT condition (2-way)

Baselines: constant train-mean (reg) / majority class (clf).

Error policy: a broken/missing/mismatched feature set or a failing single probe is
recorded as an {'error': ...} row and skipped; the remaining rows are still written.
Exit code is 0 if at least one metric row was produced, 2 otherwise.
"""
import argparse
import glob
import json
import os
import sys
import traceback

import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
DATA_ROOT = 'D:/大创/mdjpt_data_v4'
CACHE_DIR = os.path.join(REPO, 'overnight', 'bandpower_cache')
BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]

REG_TARGETS = {'DASPS': ['sam_valence', 'sam_arousal'],
               'DEAP': ['valence', 'arousal'],
               'SAM40': ['stress']}
CLF_TARGETS = {'SAM40': 'task', 'EEGMAT': 'condition'}


def safe_corr(y, p):
    try:
        if np.std(p) < 1e-9 or np.std(y) < 1e-9:
            return None
        return float(np.corrcoef(y, p)[0, 1])
    except Exception:
        return None


def loso_eval(X, y, groups, kind):
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
                'pearson_r': safe_corr(y, preds)}
    return {'acc': float(accuracy_score(y, preds)),
            'trivial_acc': float(accuracy_score(y, trivial)),
            'balanced_acc': float(balanced_accuracy_score(y, preds))}


def eval_feature_matrix(X, targets, subject_idx, out, label):
    for tname, y in targets.items():
        kind = 'clf' if tname in ('task', 'condition') else 'reg'
        try:
            if kind == 'reg':
                y = np.asarray(y, float)
                ok = np.isfinite(y)
                if not (ok.sum() >= 100 and np.std(y[ok]) > 1e-6):
                    out.append({'features': label, 'target': tname, 'kind': kind,
                                'error': f'insufficient valid samples ({int(ok.sum())})'})
                    continue
                X2, y2, g2 = X[ok] if ok.sum() < len(y) else X, y[ok] if ok.sum() < len(y) else y, \
                    subject_idx[ok] if ok.sum() < len(y) else subject_idx
            else:
                y2 = np.asarray(y, int)
                if len(np.unique(y2)) < 2:
                    out.append({'features': label, 'target': tname, 'kind': kind,
                                'error': 'single class'})
                    continue
                X2, g2 = X, subject_idx
            preds, trivial = loso_eval(X2.astype(np.float64), y2, g2, kind)
            m = metrics(y2, preds, trivial, kind)
        except Exception as e:
            out.append({'features': label, 'target': tname, 'kind': kind, 'error': str(e)})
            traceback.print_exc()
            continue
        out.append({'features': label, 'target': tname, 'kind': kind, **m})
        extra = f"r={m['pearson_r']:.3f}" if kind == 'reg' and m['pearson_r'] is not None \
            else (f"acc={m['acc']:.3f}" if kind == 'clf' else '')
        print(f"  {label:14s} {tname:12s} "
              f"{ {k: (round(v, 4) if v is not None else None) for k, v in m.items()} } {extra}")


def bandpower_features(src, n_windows):
    """Log band powers (5 bands x 60 ch), canonical file order, disk-cached."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f'{src}.npy')
    if os.path.exists(cache):
        X = np.load(cache)
        if X.shape[0] == n_windows:
            return X
        print(f'  [cache] {src} stale ({X.shape[0]} != {n_windows}), recomputing')
    freqs, P = welch(np.load(os.path.join(DATA_ROOT, src, 'sliced_data', 'sliced_len5_step2',
                                          'data', 'data_sample_0.npy')), fs=125, nperseg=625)
    bands = [(freqs >= lo) & (freqs < hi) for lo, hi in BANDS]
    d = os.path.join(DATA_ROOT, src, 'sliced_data', 'sliced_len5_step2', 'data')
    X = np.empty((n_windows, 60 * len(BANDS)), np.float32)
    for i in range(n_windows):
        _, P = welch(np.load(os.path.join(d, f'data_sample_{i}.npy')), fs=125, nperseg=625)
        X[i] = np.concatenate([np.log10(P[:, b].sum(axis=1) + 1e-12) for b in bands])
    np.save(cache, X)
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features-dir', default=os.path.join(REPO, 'overnight', 'features'))
    ap.add_argument('--out', default=os.path.join(REPO, 'overnight', 'results.json'))
    ap.add_argument('--skip-simple', action='store_true')
    args = ap.parse_args()

    sets = sorted(s for s in glob.glob(os.path.join(args.features_dir, '*')) if os.path.isdir(s))
    assert sets, 'no feature sets found'
    results = []

    for src in ['DASPS', 'DEAP', 'SAM40', 'EEGMAT']:
        ref_path = os.path.join(sets[0], f'{src}.npz')
        if not os.path.exists(ref_path):
            results.append({'features': os.path.basename(sets[0]), 'target': src,
                            'error': 'reference feature set missing this source'})
            continue
        ref = np.load(ref_path)
        n = len(ref['subject_idx'])
        targets = {k: ref[k] for k in ref.files if k not in ('fea', 'subject_idx', 'group_idx')}
        subj = ref['subject_idx']
        print(f'== {src} (N={n}) ==')
        for s in sets:
            name = os.path.basename(s)
            path = os.path.join(s, f'{src}.npz')
            try:
                if not os.path.exists(path):
                    raise FileNotFoundError(f'{path} missing (extraction failed?)')
                f = np.load(path)
                if len(f['subject_idx']) != n:
                    raise ValueError(f'window count {len(f["subject_idx"])} != reference {n} '
                                     '(stale build? rerun extraction)')
                eval_feature_matrix(f['fea'].astype(np.float64), targets, subj, results, name)
            except Exception as e:
                results.append({'features': name, 'target': src, 'error': str(e)})
                print(f'  {name:14s} ERROR: {e}')
        if not args.skip_simple:
            try:
                print(f'  computing spectral baseline ({n} windows)...')
                X = bandpower_features(src, n).astype(np.float64)
                eval_feature_matrix(X, targets, subj, results, 'spectral_simple')
            except Exception as e:
                results.append({'features': 'spectral_simple', 'target': src, 'error': str(e)})
                traceback.print_exc()

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    ok_rows = sum(1 for r in results if 'error' not in r)
    err_rows = len(results) - ok_rows
    print(f'-> {args.out} ({ok_rows} metric rows, {err_rows} error rows)')
    sys.exit(0 if ok_rows else 2)


if __name__ == '__main__':
    main()
