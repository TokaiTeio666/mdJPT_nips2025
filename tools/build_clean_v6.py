"""Clean ds007609 + ds004902 into the v4 intermediate contract (processed_clean_v6).

v4 contract per subject dir (see tools/build_sliced_v4.py):
  signals.npy   (60, N) float32 uV @ 125 Hz, canonical condition blocks concatenated
  windows.jsonl  one record per 5 s window: trial_id, alignment_key,
                 alignment_strength, offset (samples @125 Hz), n_samples, quality_flags
  summary.json   target_channels (60, canonical v4 order), fs, source notes

Sources
  DS007609  derivatives/preprocessed/*.set (EEGLAB, HCGSN 256 ch, average ref, 500 Hz,
            0.1-50 Hz + CleanLine + ICA already applied by the dataset authors).
            256 -> 60 spherical-spline interpolation (Perrin via MNE), positions from
            per-subject electrodes.tsv; target positions from MNE standard_1005.
            4 rest blocks (EO,EC,EO,EC ~60 s each, events.tsv) -> windows.
  DS004902  raw BrainVision .vhdr, 61 ch 10-20 ext, FCz online ref, 500 Hz, uV.
            Session order is counterbalanced (SessionOrder NS->SD / SD->NS):
            sessions are re-labeled to NS/SD before canonical ordering.
            Missing standard channels (FCz, PO5, PO6) spherical-spline interpolated
            from the 55 recorded ones + montage positions; AF7/AF8/TP9/TP10 dropped.
            Minimal cleaning: 0.1-45 Hz FIR + 125 Hz polyphase resample (raw data;
            robust z-score at slice-build time is the artifact backstop; per-window
            amplitude flags recorded, never dropped -- v4 policy).

Usage: .venv/Scripts/python.exe tools/build_clean_v6.py [--sources DS007609,DS004902]
"""
import argparse
import csv
import hashlib
import json
import os
import time

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly

REPO = 'D:/大创/mdJPT_nips2025'
DS007609 = 'D:/大创/datasets/ds007609'
DS004902 = 'D:/大创/datasets/ds004902'
OUT_ROOT = 'D:/大创/processed_clean_v6'
V4_ROOT = 'D:/大创/processed_clean_v4'
FS, TIMELEN, TIMESTEP = 125, 5, 2
SEG = TIMELEN * FS
STEP = TIMESTEP * FS
AMP_FLAG_UV = 150.0

DS007609_ORDER = [('EO', 'eyes_open', 1), ('EC', 'eyes_closed', 1),
                  ('EO', 'eyes_open', 2), ('EC', 'eyes_closed', 2)]
DS004902_ORDER = [('NS', 'EO'), ('NS', 'EC'), ('SD', 'EO'), ('SD', 'EC')]


def canonical_channels():
    """60 channel names, exact v4 order (from a built v4 source summary)."""
    s = json.load(open(os.path.join(V4_ROOT, 'DEAP', 'sub-01' if os.path.isdir(
        os.path.join(V4_ROOT, 'DEAP', 'sub-01')) else sorted(os.listdir(os.path.join(V4_ROOT, 'DEAP')))[0],
        'summary.json'), encoding='utf-8'))
    ch = s['target_channels']
    assert len(ch) == 60, len(ch)
    return ch


def montage_positions(names):
    """{name: xyz} from the MNE standard_1005 montage (10-05 system)."""
    import mne
    mont = mne.channels.make_standard_montage('standard_1005')
    pos = {ch_name.upper(): np.asarray(p, float) for ch_name, p in mont.get_positions()['ch_pos'].items()}
    out = {}
    for n in names:
        key = n.upper()
        if key not in pos:
            raise KeyError(f'channel {n} not in standard_1005 montage')
        out[n] = pos[key]
    return out


def fit_sphere(pts):
    """Coope least-spheres fit: return (center, radius)."""
    A = np.c_[2 * pts, np.ones(len(pts))]
    b = (pts ** 2).sum(axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:3]
    r = float(np.sqrt(sol[3] + (c ** 2).sum()))
    return c, r


def unit_positions(pts):
    """Center on fitted sphere and project to the unit sphere."""
    c, r = fit_sphere(pts)
    d = pts - c
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def spline_matrix(pos_from_unit, pos_to_unit):
    """(n_to, n_from) Perrin spherical-spline gain via MNE's internal helper."""
    from mne.channels.interpolation import _make_interpolation_matrix
    G = _make_interpolation_matrix(np.asarray(pos_from_unit, float),
                                   np.asarray(pos_to_unit, float), alpha=1e-5)
    G = np.asarray(G, np.float64)
    if G.shape != (len(pos_to_unit), len(pos_from_unit)):
        G = G.T
    assert G.shape == (len(pos_to_unit), len(pos_from_unit)), G.shape
    return G


def write_subject(src, sub, signals, win_records, channels, notes):
    out = os.path.join(OUT_ROOT, src, sub)
    os.makedirs(out, exist_ok=True)
    assert signals.dtype == np.float32 and signals.shape[0] == 60, (signals.shape, signals.dtype)
    np.save(os.path.join(out, 'signals.npy'), signals)
    with open(os.path.join(out, 'windows.jsonl'), 'w', encoding='utf-8') as f:
        for w in win_records:
            f.write(json.dumps(w) + '\n')
    json.dump({'target_channels': channels, 'fs': FS, 'timeLen': TIMELEN, 'timeStep': TIMESTEP,
               'notes': notes}, open(os.path.join(out, 'summary.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)


def flag_window(seg):
    return ['amp>150uV'] if float(np.abs(seg).max()) > AMP_FLAG_UV else []


def make_windows(n_samples_total, keys_with_dur, blocks_are_contiguous):
    """5 s windows, 2 s step, inside each block. keys_with_dur: [(key, n_samples)].
    Returns (win_records, signals_len). Blocks are concatenated back-to-back."""
    recs, cursor = [], 0
    for key, dur in keys_with_dur:
        assert dur >= SEG, (key, dur)
        last = ((dur - SEG) // STEP) * STEP
        for off in range(0, last + 1, STEP):
            recs.append({'trial_id': f'{key}@{off}', 'alignment_key': key,
                         'alignment_strength': 'weak(condition/session-locked, uniform layout)',
                         'offset': cursor + off, 'n_samples': SEG, 'quality_flags': []})
        cursor += dur
    return recs, cursor


def build_ds007609(channels, tgt_pos, limit=0):
    import mne
    mne.set_log_level('ERROR')
    subs = sorted(d for d in os.listdir(DS007609) if d.startswith('sub-'))
    t0 = time.time()

    # per-subject events + durations (for the uniform window plan)
    plans = {}
    for sub in subs:
        ev = list(csv.DictReader(open(os.path.join(
            DS007609, sub, 'eeg', f'{sub}_task-rest_events.tsv'), encoding='utf-8'), delimiter='	'))
        blocks = [(r['trial_type'], float(r['onset']), float(r['duration']))
                  for r in ev if r['trial_type'] in ('eyes_open', 'eyes_closed')]
        assert len(blocks) == 4, (sub, len(blocks))
        # canonical order EO1, EC1, EO2, EC2 by onset
        eo = sorted((b for b in blocks if b[0] == 'eyes_open'), key=lambda b: b[1])
        ec = sorted((b for b in blocks if b[0] == 'eyes_closed'), key=lambda b: b[1])
        plans[sub] = [('DS007609:EO:1', eo[0]), ('DS007609:EC:1', ec[0]),
                      ('DS007609:EO:2', eo[1]), ('DS007609:EC:2', ec[1])]
    # uniform per-block usable length across subjects (min duration, 2 s grid)
    min_dur = {}
    for sub in subs:
        for (key, (_, _, dur)) in plans[sub]:
            min_dur[key] = min(min_dur.get(key, 1e9), dur)
    block_plan = [(k, int(((min_dur[k] - TIMELEN) // TIMESTEP) * TIMESTEP * FS) + SEG)
                  for k in ('DS007609:EO:1', 'DS007609:EC:1', 'DS007609:EO:2', 'DS007609:EC:2')]
    print(f'[DS007609] block plan (min-dur across {len(subs)} subs): '
          + ', '.join(f'{k}={v / FS:.0f}s' for k, v in block_plan))

    # electrode positions: verify identical across subjects, build G once
    ele_hashes = {}
    for sub in subs:
        p = os.path.join(DS007609, sub, 'eeg', f'{sub}_electrodes.tsv')
        ele_hashes[sub] = hashlib.md5(open(p, 'rb').read()).hexdigest()
    pos_const = len(set(ele_hashes.values())) == 1
    print(f'[DS007609] electrodes.tsv identical across subjects: {pos_const}')

    first_sub = subs[0]
    ele = list(csv.DictReader(open(os.path.join(
        DS007609, first_sub, 'eeg', f'{first_sub}_electrodes.tsv'), encoding='utf-8'), delimiter='	'))
    ele_names = [r['name'] for r in ele]
    ele_pos = np.array([[float(r['x']), float(r['y']), float(r['z'])] for r in ele])
    assert len(ele_names) == 256, len(ele_names)
    G = spline_matrix(unit_positions(ele_pos), unit_positions(np.array([tgt_pos[c] for c in channels])))
    rowsum_dev = float(np.abs(G.sum(axis=1) - 1).max())
    print(f'[DS007609] G {G.shape}, max |rowsum-1| = {rowsum_dev:.4f} (constant-field check)')
    assert rowsum_dev < 0.05, rowsum_dev

    if limit:
        subs = subs[:limit]
    audit = {'source': 'DS007609', 'n_subs': len(subs), 'block_plan': dict(block_plan),
             'G_rowsum_max_dev': rowsum_dev, 'electrodes_const': pos_const, 'subjects': []}
    for s_pos, sub in enumerate(subs):
        x, sf, _ = read_eeglab_fdt(os.path.join(
            DS007609, 'derivatives', 'preprocessed', sub, 'eeg',
            f'{sub}_task-rest_desc-preproc_eeg.set'))
        assert x.shape[0] == 256 and sf == 500, (sub, x.shape, sf)
        if not pos_const:
            e = list(csv.DictReader(open(os.path.join(
                DS007609, sub, 'eeg', f'{sub}_electrodes.tsv'), encoding='utf-8'), delimiter='	'))
            G = spline_matrix(unit_positions(np.array([[float(r['x']), float(r['y']), float(r['z'])]
                                                       for r in e])),
                              unit_positions(np.array([tgt_pos[c] for c in channels])))
        x60 = G @ x  # uV, (60, N)
        y = resample_poly(x60, 1, 4, axis=1).astype(np.float32)  # 500 -> 125 Hz
        recs, total = make_windows(y.shape[1], block_plan, True)
        assert total <= y.shape[1], (sub, total, y.shape[1])
        for w in recs:
            seg = y[:, w['offset']:w['offset'] + SEG]
            w['quality_flags'] = flag_window(seg)
        write_subject('DS007609', sub, y[:, :total], recs, channels,
                      'ds007609 derivatives/preprocessed; 256->60 spherical spline; avg ref; 125 Hz')
        nf = sum(bool(w['quality_flags']) for w in recs)
        audit['subjects'].append({'subject': sub, 'n_windows': len(recs), 'flagged': nf})
        if s_pos % 10 == 0:
            print(f'  [{s_pos + 1}/{len(subs)}] {sub}: {len(recs)} windows, {nf} flagged')
    print(f'[DS007609] done in {time.time() - t0:.0f}s')
    return audit


def _fld(E, name):
    return E[name] if isinstance(E, dict) else getattr(E, name)


def read_eeglab_fdt(set_path):
    """(n_ch, n_samples) float64 uV + srate, reading the MATLAB header and .fdt
    directly (channel-major float32; MNE's EEGLAB reader rejects these files).
    Handles both flat (ds007609) and EEG-nested (ds004902) .set layouts."""
    e = loadmat(set_path, squeeze_me=True, struct_as_record=False)
    E = e['EEG'] if 'EEG' in e else e
    nbc, pnts = int(_fld(E, 'nbchan')), int(_fld(E, 'pnts'))
    sf = int(float(_fld(E, 'srate')))
    fdt = os.path.join(os.path.dirname(set_path), str(_fld(E, 'datfile')))
    x = np.fromfile(fdt, dtype='<f4')
    assert x.size == nbc * pnts, (set_path, x.size, nbc, pnts)
    x = x.reshape(nbc, pnts).astype(np.float64)
    if np.abs(x).max() < 1e-3:  # volts -> uV safety
        x = x * 1e6
    return x, sf, {'ref': str(_fld(E, 'ref'))}


def tsv_names(path):
    return [r['name'] for r in csv.DictReader(open(path, encoding='utf-8'), delimiter='\t')]


def build_ds004902(channels, tgt_pos, limit=0):
    from scipy.signal import butter, sosfiltfilt
    # 38 complete subjects (all four recordings), SessionOrder normalization
    session_order = {}
    parts = list(csv.DictReader(open(os.path.join(DS004902, 'participants.tsv'),
                                     encoding='utf-8'), delimiter='\t'))
    for r in parts:
        session_order[r['participant_id']] = r['SessionOrder']
    sos = butter(4, [0.1, 45.0], btype='band', fs=500, output='sos')

    # canonical recording order with session-order normalization
    def rec_path(sub, ses_label, cond):
        ses = 'ses-1' if ((session_order[sub] == 'NS->SD' and ses_label == 'NS') or
                          (session_order[sub] == 'SD->NS' and ses_label == 'SD')) else 'ses-2'
        task = 'eyesopen' if cond == 'EO' else 'eyesclosed'
        return os.path.join(DS004902, sub, ses, 'eeg', f'{sub}_{ses}_task-{task}_eeg.set')

    subs, pnts_map, excluded = [], {}, {}
    for sub in sorted(session_order):
        ok, why = True, ''
        p_map = {}
        for ses_label, cond in DS004902_ORDER:
            p = rec_path(sub, ses_label, cond)
            if not os.path.exists(p):
                ok, why = False, f'missing {os.path.basename(p)}'
                break
            E = loadmat(p, squeeze_me=True, struct_as_record=False)
            E = E['EEG'] if 'EEG' in E else E
            nbc, pnts = int(_fld(E, 'nbchan')), int(_fld(E, 'pnts'))
            fdt = os.path.join(os.path.dirname(p), str(_fld(E, 'datfile')))
            need = nbc * pnts * 4
            if not os.path.exists(fdt) or os.path.getsize(fdt) < need:
                ok, why = False, (f'short fdt {os.path.basename(fdt)}: '
                                  f'{os.path.getsize(fdt) if os.path.exists(fdt) else 0} < {need}')
                break
            p_map[(ses_label, cond)] = (nbc, pnts)
        if ok:
            subs.append(sub)
            pnts_map[sub] = p_map
        else:
            excluded[sub] = why
    for s, w in excluded.items():
        print(f'[DS004902] exclude {s}: {w}')
    print(f'[DS004902] valid subjects: {len(subs)} / {len(session_order)} '
          f'(excluded {len(excluded)})')
    # uniform window plan: per condition, min usable length across subjects (125 Hz)
    plan_dur = {}
    for ses_label, cond in DS004902_ORDER:
        pnts_min = min(pnts_map[s][(ses_label, cond)][1] for s in subs)
        assert all(pnts_map[s][(ses_label, cond)][0] == 61 for s in subs)
        plan_dur[(ses_label, cond)] = pnts_min // 4  # 500 -> 125 Hz samples

    # channel mapping from the first recording (EEGLAB set/fdt, like ds007609)
    set0 = os.path.join(DS004902, subs[0], 'ses-1', 'eeg',
                        f'{subs[0]}_ses-1_task-eyesopen_eeg.set')
    rec_names = tsv_names(set0.replace('_task-eyesopen_eeg.set', '_task-eyesopen_channels.tsv'))
    have = {n.upper(): n for n in rec_names}
    missing = [c for c in channels if c.upper() not in have]
    extra = [n for n in rec_names if n.upper() not in {c.upper() for c in channels}]
    print(f'[DS004902] recorded channels: {len(rec_names)}; missing standard: {missing}; '
          f'dropped: {extra}')
    assert len(rec_names) - len(extra) + len(missing) == 60

    rec_pos = montage_positions(rec_names)
    from_pos = unit_positions(np.array([rec_pos[n] for n in rec_names]))
    to_pos = unit_positions(np.array([tgt_pos[c] for c in channels]))
    name_to_idx = {n.upper(): i for i, n in enumerate(rec_names)}
    # mixed map: identity rows for recorded channels, spline rows for missing ones
    G = np.zeros((60, len(rec_names)), np.float64)
    G_spline = spline_matrix(from_pos, to_pos)
    for i, c in enumerate(channels):
        if c.upper() in name_to_idx:
            G[i, name_to_idx[c.upper()]] = 1.0
        else:
            G[i] = G_spline[i]
    rowsum_dev = float(np.abs(G.sum(axis=1) - 1).max())
    assert rowsum_dev < 0.05, rowsum_dev
    print(f'[DS004902] mixed map (identity x{60 - len(missing)}, spline x{len(missing)}), '
          f'max |rowsum-1| = {rowsum_dev:.4f}')

    if limit:
        subs = subs[:limit]
    audit = {'source': 'DS004902', 'n_subs': len(subs), 'missing_interp': missing,
             'dropped': extra, 'session_order_norm': True, 'plan_dur_125Hz':
             {f'{s}-{c}': int(d) for (s, c), d in plan_dur.items()}, 'subjects': []}
    t0 = time.time()
    for s_pos, sub in enumerate(subs):
        parts60, plan = [], []
        for ses_label, cond in DS004902_ORDER:
            x, sf, _ = read_eeglab_fdt(rec_path(sub, ses_label, cond))
            assert sf == 500 and x.shape[0] == len(rec_names), (sub, ses_label, cond)
            x = sosfiltfilt(sos, x, axis=1)          # 0.1-45 Hz, zero-phase
            x = G @ x                                 # (60, N) uV
            y500 = resample_poly(x, 1, 4, axis=1)     # 500 -> 125 Hz
            parts60.append(y500.astype(np.float32))
            plan.append((f'DS004902:{ses_label}-{cond}', plan_dur[(ses_label, cond)]))
        y = np.concatenate(parts60, axis=1)
        recs, total = make_windows(y.shape[1], plan, True)
        assert total <= y.shape[1]
        flagged = 0
        for w in recs:
            seg = y[:, w['offset']:w['offset'] + SEG]
            w['quality_flags'] = flag_window(seg)
            flagged += bool(w['quality_flags'])
        write_subject('DS004902', sub, y[:, :total], recs, channels,
                      'ds004902 raw set/fdt; 0.1-45 Hz butterworth zero-phase; '
                      f'{len(missing)} ch spherical-spline interp; '
                      'session order normalized to NS/SD; 125 Hz')
        audit['subjects'].append({'subject': sub, 'n_windows': len(recs), 'flagged': flagged})
        if s_pos % 10 == 0:
            print(f'  [{s_pos + 1}/{len(subs)}] {sub}: {len(recs)} windows, {flagged} flagged')
    print(f'[DS004902] done in {time.time() - t0:.0f}s')
    return audit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sources', default='DS007609,DS004902')
    ap.add_argument('--limit', type=int, default=0, help='debug: only first N subjects')
    args = ap.parse_args()
    which = {s.strip() for s in args.sources.split(',')}

    channels = canonical_channels()
    tgt_pos = montage_positions(channels)
    audits = {}
    if 'DS007609' in which:
        audits['DS007609'] = build_ds007609(channels, tgt_pos, args.limit)
    if 'DS004902' in which:
        audits['DS004902'] = build_ds004902(channels, tgt_pos, args.limit)

    os.makedirs(OUT_ROOT, exist_ok=True)
    out = {
        'generated': time.strftime('%Y-%m-%d %H:%M:%S'),
        'script_sha256': hashlib.sha256(open(__file__, 'rb').read()).hexdigest()[:16],
        'fs': FS, 'timelen': TIMELEN, 'timestep': TIMESTEP,
        'n_channels': 60, 'channels': channels, 'sources': audits,
    }
    json.dump(out, open(os.path.join(OUT_ROOT, 'cleaning_manifest.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print(f"manifest -> {os.path.join(OUT_ROOT, 'cleaning_manifest.json')}")


if __name__ == '__main__':
    main()
