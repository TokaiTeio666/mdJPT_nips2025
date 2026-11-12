"""LOSO readout probes + spectral simple baseline for the six-source v6 experiment.

Extends probe_eval_4src.py with the anxiety targets that motivated the new sources:
  DS007609  stai_state, stai_trait   (regression, 51 subjects, per-subject scores)
  DS004902  SAI_NS, SAI_SD, SAI_delta (regression; delta = SD - NS, NaN where absent)

Bandpower simple baseline: DEAP reads the v6 K=4 build (mdjpt_data_v6/DEAP);
all other sources read mdjpt_data_v4. Cache lives in overnight6/bandpower_cache.

Usage: python tools/probe_eval_6src.py [--skip-simple]
"""
import argparse
import glob
import importlib.util
import json
import os
import sys
import traceback

import numpy as np
from scipy.signal import welch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)

_spec = importlib.util.spec_from_file_location(
    'pe4', os.path.join(REPO, 'tools', 'probe_eval_4src.py'))
pe4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pe4)

CACHE_DIR = os.path.join(REPO, 'overnight6', 'bandpower_cache')
ROOT_MAP = {'DEAP': 'D:/大创/mdjpt_data_v6/DEAP'}
SOURCES = ['DASPS', 'DEAP', 'SAM40', 'EEGMAT', 'DS007609', 'DS004902']


def bandpower_features(src, n_windows):
    """Log band powers (5 bands x 60 ch), canonical file order, disk-cached."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f'{src}.npy')
    if os.path.exists(cache):
        X = np.load(cache)
        if X.shape[0] == n_windows:
            return X
        print(f'  [cache] {src} stale ({X.shape[0]} != {n_windows}), recomputing')
    root = ROOT_MAP.get(src, f'D:/大创/mdjpt_data_v4/{src}')
    d = os.path.join(root, 'sliced_data', 'sliced_len5_step2', 'data')
    freqs, P = welch(np.load(os.path.join(d, 'data_sample_0.npy')), fs=125, nperseg=625)
    bands = [(freqs >= lo) & (freqs < hi) for lo, hi in pe4.BANDS]
    X = np.empty((n_windows, 60 * len(pe4.BANDS)), np.float32)
    for i in range(n_windows):
        _, P = welch(np.load(os.path.join(d, f'data_sample_{i}.npy')), fs=125, nperseg=625)
        X[i] = np.concatenate([np.log10(P[:, b].sum(axis=1) + 1e-12) for b in bands])
    np.save(cache, X)
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features-dir', default=os.path.join(REPO, 'overnight6', 'features'))
    ap.add_argument('--out', default=os.path.join(REPO, 'overnight6', 'results.json'))
    ap.add_argument('--skip-simple', action='store_true')
    args = ap.parse_args()

    sets = sorted(s for s in glob.glob(os.path.join(args.features_dir, '*')) if os.path.isdir(s))
    assert sets, 'no feature sets found'
    results = []

    for src in SOURCES:
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
                pe4.eval_feature_matrix(f['fea'].astype(np.float64), targets, subj, results, name)
            except Exception as e:
                results.append({'features': name, 'target': src, 'error': str(e)})
                print(f'  {name:14s} ERROR: {e}')
        if not args.skip_simple:
            try:
                print(f'  computing spectral baseline ({n} windows)...')
                X = bandpower_features(src, n).astype(np.float64)
                pe4.eval_feature_matrix(X, targets, subj, results, 'spectral_simple')
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
