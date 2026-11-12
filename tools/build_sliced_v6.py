"""Build mdJPT sliced contract data for the two new sources (6-source extension).

Reuses build_sliced_v4 helpers (robust_z, load_windows) and io layout. The old four
sources are already built in D:/大创/mdjpt_data_v4 and are NOT touched.

Grouping (K=2 groups per source, n_vids=2 -> 4 windows/step each; budget keeps the
6-source total at 2*(4+8+4+3+2+2)=38 windows/step ~= 6.5 GiB, the measured 4-source
budget, by pooling EO/EC blocks instead of trimming DEAP):
  DS007609: EO1+EO2 blocks -> g0_EO (56 windows), EC1+EC2 -> g1_EC (58)
            window index = (block within condition, offset) fixed across subjects
  DS004902: NS-EO+SD-EO -> g0_EO (296), NS-EC+SD-EC -> g1_EC (296)
            window index = (session, condition, offset) fixed across subjects

ISA tier weak for both: protocol-locked uniform layout (EO/EC blocks of fixed order
and duration), same-index windows share condition + offset across subjects.

Usage: python tools/build_sliced_v6.py [--force]
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np

import importlib.util
_spec = importlib.util.spec_from_file_location(
    'bsv4', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'build_sliced_v4.py'))
bsv4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bsv4)

V6_ROOT = 'D:/大创/processed_clean_v6'
OUT_ROOT = bsv4.OUT_ROOT          # D:/大创/mdjpt_data_v4 (same root as the four sources)
REPO = bsv4.REPO
TIMELEN, TIMESTEP, FS, SEG = bsv4.TIMELEN, bsv4.TIMESTEP, bsv4.FS, bsv4.SEG

SOURCES = {
    'DS007609': dict(n_subs=51, n_vids=2,
                     isa='weak(protocol-locked EO/EC blocks; uniform 59/63 s blocks)'),
    'DS004902': dict(n_subs=35, n_vids=2,
                     isa='weak(condition+session-locked; uniform 300 s recordings)'),
}

GROUP_PLANS = {
    'DS007609': [
        ('g0_EO', ['DS007609:EO:1', 'DS007609:EO:2']),
        ('g1_EC', ['DS007609:EC:1', 'DS007609:EC:2']),
    ],
    'DS004902': [
        ('g0_EO', ['DS004902:NS-EO', 'DS004902:SD-EO']),
        ('g1_EC', ['DS004902:NS-EC', 'DS004902:SD-EC']),
    ],
}

# DEAP-v6: rebuild with K=4 groups (10 videos each) so n_vids=4 samples one window
# per group without the sampler's last-group degeneration (n_vids < K would pin
# every batch to the final group). Separate out root so the published 4src K=8
# build stays byte-identical.
DEAP_V6 = dict(n_subs=32, n_vids=4, isa='exact(v4 grouping, K=4 regroup)',
               root='D:/大创/processed_clean_v4/DEAP',
               out='D:/大创/mdjpt_data_v6/DEAP')


def build_deap_v6(force):
    src = 'DEAP'
    out_dir = os.path.join(DEAP_V6['out'], 'sliced_data', f'sliced_len{TIMELEN}_step{TIMESTEP}')
    saved_flag = os.path.join(out_dir, 'saved.npy')
    audit_path = os.path.join(out_dir, 'metadata', 'trial_alignment.json')
    if os.path.exists(saved_flag) and os.path.exists(audit_path) and not force:
        print('[DEAP-v6] saved.npy exists, skip (--force to rebuild)')
        return None
    os.makedirs(os.path.join(out_dir, 'data'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'metadata'), exist_ok=True)

    sub_dirs = sorted(os.listdir(DEAP_V6['root']))
    assert len(sub_dirs) == DEAP_V6['n_subs']
    first = bsv4.load_windows(os.path.join(DEAP_V6['root'], sub_dirs[0]))
    keys = sorted(first, key=lambda k: int(k.split(':')[-1]))
    assert len(keys) == 40, keys
    plan = [(f'g{g}_videos{1 + 10 * g:02d}-{10 + 10 * g:02d}', keys[10 * g:10 * g + 10])
            for g in range(4)]
    expect = {k: len(first[k]) for _, ks in plan for k in ks}
    onesub = sum(expect.values())
    assert onesub == 1120
    channels = json.load(open(os.path.join(DEAP_V6['root'], sub_dirs[0], 'summary.json'),
                              encoding='utf-8'))['target_channels']

    labels, group_counts = None, None
    cov_sum = np.zeros((60, 60), np.float64)
    ch_sum = np.zeros(60, np.float64)
    ch_sq = np.zeros(60, np.float64)
    ch_n = cov_n = 0
    subjects_meta = []
    t0 = time.time()
    for s_pos, sub in enumerate(sub_dirs):
        by_key = bsv4.load_windows(os.path.join(DEAP_V6['root'], sub))
        for k, cnt in expect.items():
            assert len(by_key[k]) == cnt, (sub, k)
        signals = np.load(os.path.join(DEAP_V6['root'], sub, 'signals.npy'))
        z, mu, sd, kept = bsv4.robust_z(signals)
        sub_labels = []
        for g_idx, (_, ks) in enumerate(plan):
            for k in ks:
                for off, _, _f in by_key[k]:
                    seg = z[:, off:off + SEG]
                    np.save(os.path.join(out_dir, 'data',
                                         f'data_sample_{s_pos * onesub + len(sub_labels)}.npy'),
                            np.ascontiguousarray(seg, dtype=np.float32))
                    sub_labels.append(g_idx)
        counts = [sub_labels.count(g) for g in range(4)]
        if s_pos == 0:
            labels, group_counts = sub_labels, counts
        else:
            assert sub_labels == labels and counts == group_counts
        picks = np.unique(np.linspace(0, len(sub_labels) - 1, 6).astype(int))
        segs = []
        # keep z in memory per subject for the audit picks (recompute from offsets)
        for p in picks:
            g_idx = labels[p]
            off = None
            idx = 0
            for g_idx_p, (_, ks) in enumerate(plan):
                for k in ks:
                    for o, _, _f in by_key[k]:
                        if idx == p:
                            off = o
                        idx += 1
            seg = z[:, off:off + SEG].astype(np.float64)
            segs.append(seg)
            cov_sum += seg @ seg.T / (SEG - 1)
            cov_n += 1
            ch_sum += seg.sum(axis=1)
            ch_sq += (seg ** 2).sum(axis=1)
            ch_n += SEG
        subjects_meta.append({'subject': sub, 'z_mu': round(mu, 4), 'z_sd': round(sd, 4),
                              'robust_kept_frac': round(kept, 4), 'n_windows': len(sub_labels)})
        if s_pos % 8 == 0:
            print(f'  [DEAP-v6 {s_pos + 1}/{len(sub_dirs)}] {sub}')

    np.save(os.path.join(out_dir, 'metadata', 'onesub_labels.npy'), np.array(labels, np.int64))
    np.save(os.path.join(out_dir, 'metadata', 'n_samples_onesub.npy'), np.array(group_counts, np.int64))
    np.save(os.path.join(out_dir, 'metadata', 'n_samples_sessions.npy'),
            np.array([group_counts], np.int64))
    np.save(saved_flag, [True])
    align = {'source': src, 'isa_tier': DEAP_V6['isa'], 'n_vids': DEAP_V6['n_vids'],
             'regroup_note': 'K=4 x 10 videos (v4 published build used K=8 x 5 videos; '
                             'same windows, same exact pairing semantics)',
             'groups': [{'group': g, 'index': i, 'alignment_keys': ks,
                         'windows_per_subject': group_counts[i]}
                        for i, (g, ks) in enumerate(plan)],
             'pairing_semantics': 'positive pair = same window index across two subjects '
                                  '= same video group + same video + same offset'}
    with open(audit_path, 'w', encoding='utf-8') as f:
        json.dump(align, f, ensure_ascii=False, indent=1)
    cov_mean = cov_sum / cov_n
    np.fill_diagonal(cov_mean, 0)
    per_ch_std = np.sqrt(np.maximum(ch_sq / ch_n - (ch_sum / ch_n) ** 2, 0))
    audit = {'source': 'DEAP', 'variant': 'v6-K4', 'n_subs': len(sub_dirs),
             'windows_per_sub': onesub, 'group_counts': group_counts,
             'cov_frobenius_offdiag': float(np.linalg.norm(cov_mean, 'fro')),
             'per_channel_std_min': float(per_ch_std.min()),
             'per_channel_std_max': float(per_ch_std.max()),
             'subjects': subjects_meta, 'channels': channels,
             'build_seconds': round(time.time() - t0, 1)}
    print(f"[DEAP-v6] {len(sub_dirs)} x {onesub} win, groups {group_counts}, "
          f"cov_fro={audit['cov_frobenius_offdiag']:.2f}")
    return audit


def build_source(src, spec, force):
    out_dir = os.path.join(OUT_ROOT, src, 'sliced_data', f'sliced_len{TIMELEN}_step{TIMESTEP}')
    saved_flag = os.path.join(out_dir, 'saved.npy')
    audit_path = os.path.join(out_dir, 'metadata', 'trial_alignment.json')
    if os.path.exists(saved_flag) and os.path.exists(audit_path) and not force:
        print(f'[{src}] saved.npy exists, skip (--force to rebuild)')
        return None

    os.makedirs(os.path.join(out_dir, 'data'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'metadata'), exist_ok=True)

    plan = GROUP_PLANS[src]
    sub_dirs = sorted(os.listdir(os.path.join(V6_ROOT, src)))
    assert len(sub_dirs) == spec['n_subs'], (src, len(sub_dirs))

    first = bsv4.load_windows(os.path.join(V6_ROOT, src, sub_dirs[0]))
    expect = {k: len(first[k]) for _, keys in plan for k in keys}
    onesub = sum(expect.values())
    channels = json.load(open(os.path.join(V6_ROOT, src, sub_dirs[0], 'summary.json'),
                              encoding='utf-8'))['target_channels']
    assert len(channels) == 60

    labels, group_counts = None, None
    audit = {'source': src, 'subjects': [], 'channels_verified': True}
    cov_sum = np.zeros((60, 60), np.float64)
    ch_sum = np.zeros(60, np.float64)
    ch_sq = np.zeros(60, np.float64)
    ch_n = cov_n = 0
    flagged_selected = picked_total = 0
    t0 = time.time()

    for s_pos, sub in enumerate(sub_dirs):
        sub_dir = os.path.join(V6_ROOT, src, sub)
        by_key = bsv4.load_windows(sub_dir)
        for k, cnt in expect.items():
            assert len(by_key[k]) == cnt, (src, sub, k, len(by_key[k]), cnt)
        ch_sub = json.load(open(os.path.join(sub_dir, 'summary.json'),
                                encoding='utf-8'))['target_channels']
        if ch_sub != channels:
            audit['channels_verified'] = False

        signals = np.load(os.path.join(sub_dir, 'signals.npy'))
        assert signals.shape[0] == 60 and signals.dtype == np.float32
        z, mu, sd, kept = bsv4.robust_z(signals)

        sub_labels, win_offsets = [], []
        for g_idx, (_, keys) in enumerate(plan):
            for k in keys:
                for off, _, flagged in by_key[k]:
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

    align = {
        'source': src, 'isa_tier': spec['isa'], 'n_vids': spec['n_vids'],
        'groups': [{'group': g, 'index': i, 'alignment_keys': keys,
                    'windows_per_subject': group_counts[i]}
                   for i, (g, keys) in enumerate(plan)],
        'pairing_semantics': 'positive pair = same window index across two subjects '
                             '= same condition group + same key + same offset',
    }
    with open(audit_path, 'w', encoding='utf-8') as f:
        json.dump(align, f, ensure_ascii=False, indent=1)

    cov_mean = cov_sum / cov_n
    np.fill_diagonal(cov_mean, 0)
    per_ch_std = np.sqrt(np.maximum(ch_sq / ch_n - (ch_sum / ch_n) ** 2, 0))
    audit.update({
        'n_subs': len(sub_dirs), 'windows_per_sub': onesub, 'group_counts': group_counts,
        'cov_frobenius_offdiag': float(np.linalg.norm(cov_mean, 'fro')),
        'per_channel_std_min': float(per_ch_std.min()),
        'per_channel_std_max': float(per_ch_std.max()),
        'per_channel_std_median': float(np.median(per_ch_std)),
        'flagged_frac_sampled': round(flagged_selected / max(1, picked_total), 4),
        'build_seconds': round(time.time() - t0, 1), 'channels': channels,
    })
    print(f"[{src}] {len(sub_dirs)} subs x {onesub} win, groups {group_counts}, "
          f"cov_fro={audit['cov_frobenius_offdiag']:.2f}, "
          f"ch_std[{audit['per_channel_std_min']:.2f},{audit['per_channel_std_max']:.2f}], "
          f"{audit['build_seconds']}s")
    return audit


def write_yaml(src, spec, channels):
    body = [
        f"dataset_name: '{src}'",
        f"data_dir: '{spec.get('out', OUT_ROOT + '/' + src)}'",
        f"fs: {FS}", 'n_channs: 60', 'n_session: 1',
        f"n_subs: {spec['n_subs']}", f"n_vids: {spec['n_vids']}",
        'n_class: 2  # unused during pretraining',
        f'timeLen: {TIMELEN}', f'timeStep: {TIMESTEP}',
        f'timeLen2: {TIMELEN}', f'timeStep2: {TIMESTEP}',
        'channels: [' + ', '.join(f'"{c}"' for c in channels) + ']',
    ]
    suffix = spec.get('suffix', '_v6')
    with open(os.path.join(REPO, 'cfgs_multi', 'data', f'{src}{suffix}.yaml'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(body) + '\n')
    print(f"wrote cfgs_multi/data/{src}{suffix}.yaml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--new-only', action='store_true',
                    help='force-rebuild only DS007609/DS004902 (restores full audit)')
    args = ap.parse_args()

    audits = {}
    for src, spec in SOURCES.items():
        a = build_source(src, spec, args.force or args.new_only)
        if a is not None:
            audits[src] = a
            write_yaml(src, spec, a['channels'])

    da = build_deap_v6(args.force and not args.new_only)
    if da is not None:
        audits['DEAP'] = da
        write_yaml('DEAP', DEAP_V6, da['channels'])

    if not audits:
        print('nothing built (all skipped)')
        return

    # CDA balance gate across all six sources (old four from the v4 build audit,
    # skipped sources from the previous v6 audit)
    old = json.load(open(os.path.join(OUT_ROOT, 'build_audit.json'), encoding='utf-8'))
    fro = {s: old['cov_frobenius'][s] for s in old['cov_frobenius']}
    prev_path = os.path.join(OUT_ROOT, 'build_audit_v6.json')
    if os.path.exists(prev_path) and not args.force:
        prev = json.load(open(prev_path, encoding='utf-8'))
        for s, a in prev.get('sources', {}).items():
            if s not in audits and 'cov_frobenius_offdiag' in a:
                audits[s] = a
    fro.update({s: audits[s]['cov_frobenius_offdiag'] for s in audits})
    ratio = max(fro.values()) / min(fro.values())
    gate1 = all(audits[s]['per_channel_std_min'] >= 0.3 and audits[s]['per_channel_std_max'] <= 3.0
                for s in audits)
    summary = {
        'generated': time.strftime('%Y-%m-%d %H:%M:%S'),
        'script_sha256': hashlib.sha256(open(__file__, 'rb').read()).hexdigest()[:16],
        'v6_root': V6_ROOT, 'out_root': OUT_ROOT,
        'cov_frobenius_all6': fro, 'cda_ratio_max_min_all6': ratio,
        'gate1_per_channel_std_0.3_3_new2': 'PASS' if gate1 else 'FAIL',
        'gate2_cda_ratio_lt_3_all6': 'PASS' if ratio < 3 else 'FAIL',
        'sources': audits,
    }
    with open(os.path.join(OUT_ROOT, 'build_audit_v6.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"GATE1 (new two) per-channel std in [0.3,3]: {summary['gate1_per_channel_std_0.3_3_new2']}")
    print(f"GATE2 all-6 CDA ratio max/min = {ratio:.3f} < 3: {summary['gate2_cda_ratio_lt_3_all6']}")
    print('cov_frobenius all 6:', {k: round(v, 2) for k, v in fro.items()})


if __name__ == '__main__':
    main()
