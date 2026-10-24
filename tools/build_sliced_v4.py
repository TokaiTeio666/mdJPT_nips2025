"""Build mdJPT `sliced_len5_step2` contract data from `processed_clean_v4`.

Implements 修改方案 §3.1 (docs/four_source_v4/四源接入mdJPT联合预训练_修改方案_2026-10-23.md,
§9 增补: DASPS ISA upgraded to weak). Zero changes to src/: the output layout matches
io_utils.save_sliced_data exactly, so EEG_Dataset's slicing branch is skipped via saved.npy.

Per-source grouping (K = n_samples_sessions columns, n_vids = per-source batch vids):
  DASPS  : trial_01..12        -> 4 groups x 3 trials  (18 windows/group)  K=4 n_vids=4
  DEAP   : video:01..40        -> 8 groups x 5 videos  (140 windows/group) K=8 n_vids=8
  SAM40  : 4 tasks x 3 repeats -> 4 groups x 1 task    (33 windows/group)  K=4 n_vids=4
  EEGMAT : rest(first 29) + arithmetic(29)             (29 windows/group)  K=2 n_vids=3

Normalization: per-subject robust z-score over the whole cleaned recording,
identical to the original loaders (thr = 30*median|x|; mean/std over |x|<thr).

Side products: metadata/trial_alignment.json (ISA whitelist evidence) and
build_audit.json at OUT_ROOT (acceptance gates 1 & 2 of the plan).

Usage: python tools/build_sliced_v4.py [--force]
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np

V4_ROOT = 'D:/大创/processed_clean_v4'
OUT_ROOT = 'D:/大创/mdjpt_data_v4'
REPO = 'D:/大创/mdJPT_nips2025'
TIMELEN, TIMESTEP, FS = 5, 2, 125
SEG = TIMELEN * FS  # 625

SOURCES = {
    'DASPS':  dict(n_subs=23, n_vids=4, isa='weak(2026-10-23 evidence; source meta: unverified)'),
    'DEAP':   dict(n_subs=32, n_vids=8, isa='exact'),
    'SAM40':  dict(n_subs=40, n_vids=4, isa='weak'),
    'EEGMAT': dict(n_subs=36, n_vids=3, isa='none(CDA-only; no time-locked stimulus)'),
}


def robust_z(x):
    # identical to src/data/io_utils.py load_processed_*_NEW_data convention
    a = x.astype(np.float64)
    thr = 30.0 * np.median(np.abs(a))
    m = np.abs(a) < thr
    mu = a[m].mean()
    sd = a[m].std()
    return ((a - mu) / sd).astype(np.float32), float(mu), float(sd), float(m.mean())


def load_windows(sub_dir):
    """Return {key: [(offset, strength, flagged)] sorted by offset}."""
    by_key = {}
    with open(os.path.join(sub_dir, 'windows.jsonl'), encoding='utf-8') as f:
        for line in f:
            w = json.loads(line)
            key = w['alignment_key'] if w['alignment_key'] else w['trial_id']
            assert w['n_samples'] == SEG, (key, w['n_samples'])
            by_key.setdefault(key, []).append(
                (int(w['offset']), w['alignment_strength'], bool(w.get('quality_flags'))))
    for k in by_key:
        by_key[k].sort(key=lambda t: t[0])
    return by_key


def group_plan(src, by_key_first):
    """Return ordered [(group_name, [key, ...])]. Same mapping for every subject."""
    if src == 'DASPS':
        keys = [f'trial_{i:02d}' for i in range(1, 13)]
        return [(f'g{g}_trials{1+3*g:02d}-{3+3*g:02d}', keys[3 * g:3 * g + 3]) for g in range(4)]
    if src == 'DEAP':
        keys = sorted(by_key_first, key=lambda k: int(k.split(':')[-1]))
        assert len(keys) == 40, keys
        return [(f'g{g}_videos{1+5*g:02d}-{5+5*g:02d}', keys[5 * g:5 * g + 5]) for g in range(8)]
    if src == 'SAM40':
        tasks = sorted({k.split(':')[1] for k in by_key_first})
        assert len(tasks) == 4, tasks
        return [(f'g{g}_{task}', [f'SAM40:{task}:{r}' for r in (1, 2, 3)])
                for g, task in enumerate(tasks)]
    if src == 'EEGMAT':
        return [('g0_rest_first29', ['EEGMAT:rest']), ('g1_arithmetic', ['EEGMAT:arithmetic'])]
    raise ValueError(src)


def truncated_keys(src, key, wins):
    return wins[:29] if (src == 'EEGMAT' and key == 'EEGMAT:rest') else wins


def build_source(src, spec, force):
    out_dir = os.path.join(OUT_ROOT, src, 'sliced_data', f'sliced_len{TIMELEN}_step{TIMESTEP}')
    saved_flag = os.path.join(out_dir, 'saved.npy')
    audit_path = os.path.join(out_dir, 'metadata', 'trial_alignment.json')
    if os.path.exists(saved_flag) and os.path.exists(audit_path) and not force:
        print(f'[{src}] saved.npy exists, skip (--force to rebuild)')
        return None

    os.makedirs(os.path.join(out_dir, 'data'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'metadata'), exist_ok=True)

    sub_dirs = sorted(os.listdir(os.path.join(V4_ROOT, src)))
    assert len(sub_dirs) == spec['n_subs'], (src, len(sub_dirs))

    first = load_windows(os.path.join(V4_ROOT, src, sub_dirs[0]))
    plan = group_plan(src, first)
    # per-key expected window counts (uniform across subjects; EEGMAT rest truncated to 29)
    expect = {}
    for _, keys in plan:
        for k in keys:
            expect[k] = len(truncated_keys(src, k, first[k]))

    channels = json.load(open(os.path.join(V4_ROOT, src, sub_dirs[0], 'summary.json'),
                              encoding='utf-8'))['target_channels']
    assert len(channels) == 60

    onesub = sum(expect.values())
    labels = None            # group index per window (same for every subject)
    group_counts = None      # windows per group
    audit = {'source': src, 'subjects': [], 'channels_verified': True}
    cov_sum = np.zeros((60, 60), np.float64)
    ch_sum = np.zeros(60, np.float64)
    ch_sq = np.zeros(60, np.float64)
    ch_n = 0
    cov_n = 0
    flagged_selected = 0
    picked_total = 0
    t0 = time.time()

    for s_pos, sub in enumerate(sub_dirs):
        sub_dir = os.path.join(V4_ROOT, src, sub)
        by_key = load_windows(sub_dir)
        for k, cnt in expect.items():
            assert len(truncated_keys(src, k, by_key[k])) == cnt, (src, sub, k)
        ch_sub = json.load(open(os.path.join(sub_dir, 'summary.json'), encoding='utf-8'))['target_channels']
        if ch_sub != channels:
            audit['channels_verified'] = False

        signals = np.load(os.path.join(sub_dir, 'signals.npy'))
        assert signals.shape[0] == 60 and signals.dtype == np.float32
        z, mu, sd, kept = robust_z(signals)

        sub_labels = []
        win_offsets = []     # flat (group_idx, offset, flagged) in write order
        for g_idx, (_, keys) in enumerate(plan):
            for k in keys:
                for off, _, flagged in truncated_keys(src, k, by_key[k]):
                    seg = z[:, off:off + SEG]
                    assert seg.shape == (60, SEG)
                    np.save(os.path.join(out_dir, 'data',
                                         f'data_sample_{s_pos * onesub + len(sub_labels)}.npy'),
                            np.ascontiguousarray(seg, dtype=np.float32))
                    win_offsets.append((g_idx, off, flagged))
                    sub_labels.append(g_idx)

        counts = [sub_labels.count(g) for g in range(len(plan))]
        if s_pos == 0:
            labels, group_counts = sub_labels, counts
        else:
            assert sub_labels == labels and counts == group_counts, (src, sub)

        picks = np.unique(np.linspace(0, len(sub_labels) - 1, min(6, len(sub_labels))).astype(int))
        for p in picks:
            g_idx, off, flagged = win_offsets[p]
            seg = z[:, off:off + SEG].astype(np.float64)
            cov_sum += seg @ seg.T / (SEG - 1)
            cov_n += 1
            ch_sum += seg.sum(axis=1)
            ch_sq += (seg ** 2).sum(axis=1)
            ch_n += SEG
            flagged_selected += int(flagged)
        picked_total += len(picks)
        audit['subjects'].append({'subject': sub, 'z_mu': round(mu, 4), 'z_sd': round(sd, 4),
                                  'robust_kept_frac': round(kept, 4), 'n_windows': len(sub_labels)})

    np.save(os.path.join(out_dir, 'metadata', 'onesub_labels.npy'), np.array(labels, np.int64))
    np.save(os.path.join(out_dir, 'metadata', 'n_samples_onesub.npy'), np.array(group_counts, np.int64))
    np.save(os.path.join(out_dir, 'metadata', 'n_samples_sessions.npy'),
            np.array([group_counts], np.int64))
    np.save(saved_flag, [True])

    strengths = sorted({w[1] for wins in first.values() for w in wins})
    align = {
        'source': src, 'isa_tier': spec['isa'], 'n_vids': spec['n_vids'],
        'alignment_strength_values': strengths,
        'groups': [{'group': g_name, 'index': g, 'alignment_keys': keys,
                    'windows_per_subject': group_counts[g]}
                   for g, (g_name, keys) in enumerate(plan)],
        'pairing_semantics': 'positive pair = same window index across two subjects '
                             '= same group(trial/task/condition) + same key + same offset',
    }
    with open(os.path.join(out_dir, 'metadata', 'trial_alignment.json'), 'w', encoding='utf-8') as f:
        json.dump(align, f, ensure_ascii=False, indent=1)

    cov_mean = cov_sum / cov_n
    np.fill_diagonal(cov_mean, 0)
    per_ch_std = np.sqrt(np.maximum(ch_sq / ch_n - (ch_sum / ch_n) ** 2, 0))
    audit.update({
        'n_subs': len(sub_dirs), 'windows_per_sub': onesub,
        'group_counts': group_counts, 'n_samples_sessions': [group_counts],
        'cov_frobenius_offdiag': float(np.linalg.norm(cov_mean, 'fro')),
        'per_channel_std_min': float(per_ch_std.min()), 'per_channel_std_max': float(per_ch_std.max()),
        'per_channel_std_median': float(np.median(per_ch_std)),
        'per_channel_std_outliers_lt0.3': int((per_ch_std < 0.3).sum()),
        'per_channel_std_outliers_gt3': int((per_ch_std > 3).sum()),
        'flagged_frac_sampled': round(flagged_selected / max(1, picked_total), 4),
        'build_seconds': round(time.time() - t0, 1),
        'channels': channels,
    })
    print(f"[{src}] {len(sub_dirs)} subs x {onesub} win = {len(sub_dirs) * onesub} files, "
          f"cov_fro={audit['cov_frobenius_offdiag']:.2f}, "
          f"ch_std[{audit['per_channel_std_min']:.3f},{audit['per_channel_std_max']:.3f}], "
          f"{audit['build_seconds']}s")
    return audit


def write_yaml(src, spec, channels):
    body = [
        f"dataset_name: '{src}'",
        f"data_dir: 'D:/大创/mdjpt_data_v4/{src}'",
        f"fs: {FS}", 'n_channs: 60', 'n_session: 1',
        f"n_subs: {spec['n_subs']}", f"n_vids: {spec['n_vids']}",
        'n_class: 2  # unused during pretraining',
        f'timeLen: {TIMELEN}', f'timeStep: {TIMESTEP}',
        f'timeLen2: {TIMELEN}', f'timeStep2: {TIMESTEP}',
        'channels: [' + ', '.join(f'"{c}"' for c in channels) + ']',
    ]
    with open(os.path.join(REPO, 'cfgs_multi', 'data', f'{src}_v4.yaml'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(body) + '\n')
    print(f"wrote cfgs_multi/data/{src}_v4.yaml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    audits, yamls_pending = {}, []
    for src, spec in SOURCES.items():
        audit = build_source(src, spec, args.force)
        if audit is None:
            continue
        audits[src] = audit
        yamls_pending.append((src, audit['channels']))

    if not audits:
        print('nothing built (all skipped); delete OUT_ROOT or use --force to rebuild')
        return

    fro = {s: audits[s]['cov_frobenius_offdiag'] for s in audits}
    ratio = max(fro.values()) / min(fro.values())
    gate1 = all(audits[s]['per_channel_std_min'] >= 0.3 and audits[s]['per_channel_std_max'] <= 3.0
                for s in audits)
    summary = {
        'generated': time.strftime('%Y-%m-%d %H:%M:%S'),
        'script_sha256': hashlib.sha256(open(__file__, 'rb').read()).hexdigest()[:16],
        'v4_root': V4_ROOT, 'out_root': OUT_ROOT,
        'cov_frobenius': fro, 'cda_ratio_max_min': ratio,
        'gate1_per_channel_std_0.3_3': 'PASS' if gate1 else 'FAIL',
        'gate2_cda_ratio_lt_3': 'PASS' if ratio < 3 else 'FAIL',
        'sources': audits,
    }
    with open(os.path.join(OUT_ROOT, 'build_audit.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"GATE1 per-channel std in [0.3,3]: {summary['gate1_per_channel_std_0.3_3']}")
    print(f"GATE2 CDA ratio max/min = {ratio:.3f} < 3: {summary['gate2_cda_ratio_lt_3']}")
    print(f"audit -> {os.path.join(OUT_ROOT, 'build_audit.json')}")

    for src, channels in yamls_pending:
        write_yaml(src, SOURCES[src], channels)


if __name__ == '__main__':
    main()
