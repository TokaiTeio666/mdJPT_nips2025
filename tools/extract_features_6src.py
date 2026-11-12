"""Extract frozen encoder features for the six-source v6 experiment.

Extends extract_features_4src.py with:
  DEAP-v6   K=4 regroup (build_deap_v6 order), windows from processed_clean_v4,
            per-window DEAP valence/arousal labels (unchanged semantics)
  DS007609  v6 windows; per-subject targets stai_state / stai_trait
            (phenotype/stai.tsv, 51/51 complete)
  DS004902  v6 windows; per-subject targets SAI_NS / SAI_SD / SAI_delta
            (participants.tsv; SAI_delta = SD - NS where both present, else NaN)

Canonical window order mirrors tools/build_sliced_v6.py exactly (subject-major,
group-major, key order, offset order), so feature row i corresponds to
data_sample_{i}.npy.

Usage:
  python tools/extract_features_6src.py --name random_s32 --seed 19260832
  python tools/extract_features_6src.py --name isacda_s32 --ckpt log/6src_isacda_s32/ckpt/epoch=02.ckpt
"""
import argparse
import csv
import importlib.util
import json
import os
import sys
import traceback

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

_spec = importlib.util.spec_from_file_location(
    'ef4', os.path.join(REPO, 'tools', 'extract_features_4src.py'))
ef4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ef4)
bsv = ef4.bsv
_spec6 = importlib.util.spec_from_file_location(
    'bsv6', os.path.join(REPO, 'tools', 'build_sliced_v6.py'))
bsv6 = importlib.util.module_from_spec(_spec6)
_spec6.loader.exec_module(bsv6)

from hydra import compose, initialize_config_dir
from src.model.MultiModel_PL import MultiModel_PL

V6_ROOT = bsv6.V6_ROOT
OUT_DIR = os.path.join(REPO, 'overnight6', 'features')

DS007609_ROOT = os.path.join(V6_ROOT, 'DS007609')
DS004902_ROOT = os.path.join(V6_ROOT, 'DS004902')
DEAP_V6_ROOT = bsv6.DEAP_V6['root']
DEAP_V6_PLAN = None  # built lazily, mirrors build_deap_v6


def load_stai():
    out = {}
    for r in csv.DictReader(open('D:/大创/datasets/ds007609/phenotype/stai.tsv',
                                 encoding='utf-8'), delimiter='\t'):
        out[r['participant_id']] = (float(r['stai_state']), float(r['stai_trait']))
    return out


def load_sai():
    out = {}
    for r in csv.DictReader(open('D:/大创/datasets/ds004902/participants.tsv',
                                 encoding='utf-8'), delimiter='\t'):
        ns = float(r['SAI_NS']) if r['SAI_NS'] not in ('n/a', '', 'NaN') else float('nan')
        sd = float(r['SAI_SD']) if r['SAI_SD'] not in ('n/a', '', 'NaN') else float('nan')
        delta = sd - ns if (np.isfinite(ns) and np.isfinite(sd)) else float('nan')
        out[r['participant_id']] = (ns, sd, delta)
    return out


def deap_v6_plan(by_key_first):
    keys = sorted(by_key_first, key=lambda k: int(k.split(':')[-1]))
    assert len(keys) == 40, keys
    return [(f'g{g}_videos{1 + 10 * g:02d}-{10 + 10 * g:02d}', keys[10 * g:10 * g + 10])
            for g in range(4)]


@torch.no_grad()
def extract_source_v6(model, src, out_root, device, batch, stai=None, sai=None):
    out_path = os.path.join(out_root, f'{src}.npz')
    if os.path.exists(out_path):
        print(f'[{src}] exists, skip')
        return
    if src in ('DASPS', 'SAM40', 'EEGMAT'):
        return ef4.extract_source(model, src, out_root, device, batch)
    if src == 'DEAP':
        root, plan_fn = DEAP_V6_ROOT, deap_v6_plan
        target_keys = ['valence', 'arousal']
    elif src == 'DS007609':
        root, plan_fn = DS007609_ROOT, lambda _: bsv6.GROUP_PLANS[src]
        target_keys = ['stai_state', 'stai_trait']
    else:
        root, plan_fn = DS004902_ROOT, lambda _: bsv6.GROUP_PLANS[src]
        target_keys = ['SAI_NS', 'SAI_SD', 'SAI_delta']

    sub_dirs = sorted(os.listdir(root))
    first = ef4.load_records(os.path.join(root, sub_dirs[0]))
    plan = plan_fn(first)
    channels = json.load(open(os.path.join(root, sub_dirs[0], 'summary.json'),
                              encoding='utf-8'))['target_channels']
    assert len(channels) == 60

    feas, subj, grp = [], [], []
    targets = {k: [] for k in target_keys}
    for s_pos, sub in enumerate(sub_dirs):
        by_key = ef4.load_records(os.path.join(root, sub))
        signals = np.load(os.path.join(root, sub, 'signals.npy'))
        z, _, _, _ = bsv.robust_z(signals)
        segs = []
        for g_idx, (_, keys) in enumerate(plan):
            for k in keys:
                for w in by_key[k]:
                    off = w['offset']
                    if off + bsv.SEG > z.shape[1]:
                        raise RuntimeError(f'{src}/{sub}/{k}: window exceeds signal')
                    segs.append(z[:, off:off + bsv.SEG].astype(np.float32))
                    subj.append(s_pos)
                    grp.append(g_idx)
                    if src == 'DEAP':
                        lab = w['labels']
                        targets['valence'].append(float(lab['valence']))
                        targets['arousal'].append(float(lab['arousal']))
                    elif src == 'DS007609':
                        st, tr = stai[sub]
                        targets['stai_state'].append(st)
                        targets['stai_trait'].append(tr)
                    else:
                        ns, sd, d = sai[sub]
                        targets['SAI_NS'].append(ns)
                        targets['SAI_SD'].append(sd)
                        targets['SAI_delta'].append(d)
        for i0 in range(0, len(segs), batch):
            feas.append(ef4.forward_windows(model, segs[i0:i0 + batch], channels, device))
        if s_pos % 10 == 0:
            print(f'  [{src} {s_pos + 1}/{len(sub_dirs)}]')
    fea = np.concatenate(feas)
    np.savez_compressed(out_path, fea=fea,
                        subject_idx=np.array(subj, np.int32),
                        group_idx=np.array(grp, np.int32),
                        **{k: np.array(v, np.float32) for k, v in targets.items()})
    print(f'[{src}] fea {fea.shape} -> {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--seed', type=int, default=19260832)
    ap.add_argument('--batch', type=int, default=32)
    args = ap.parse_args()

    free = 0
    if torch.cuda.is_available():
        try:
            q = os.popen('nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits').read()
            free = int(q.strip().splitlines()[0])
        except Exception:
            free = 0
    device = torch.device('cuda') if free >= 2500 else torch.device('cpu')
    if device.type == 'cpu':
        ef4.patch_to_patch_for_cpu()
    print(f'device={device} (free {free} MiB)')

    with initialize_config_dir(config_dir=os.path.join(REPO, 'cfgs_multi'), version_base='1.3'):
        cfg = compose(config_name='config_6src_v6')
    cfg.data_cfg_list = [cfg.data_0, cfg.data_1, cfg.data_2, cfg.data_3, cfg.data_4, cfg.data_5]
    cfg.data_cfg_list = [c for c in cfg.data_cfg_list if c.dataset_name != 'None']

    torch.manual_seed(args.seed)
    model = MultiModel_PL(cfg)
    if args.ckpt:
        if not os.path.exists(args.ckpt):
            print(f'ERROR: ckpt not found: {args.ckpt}')
            sys.exit(2)
        try:
            model.load_state_dict(ef4.load_ckpt_state(args.ckpt))
            print(f'loaded ckpt {args.ckpt}')
        except Exception as e:
            print(f'ERROR: ckpt load failed ({e}); corrupt or partial checkpoint?')
            sys.exit(2)
    else:
        print(f'random init (seed={args.seed})')
    model.eval().to(device)

    out_root = os.path.join(OUT_DIR, args.name)
    os.makedirs(out_root, exist_ok=True)
    stai = load_stai()
    sai = load_sai()
    failures = []
    for src in ('DASPS', 'DEAP', 'SAM40', 'EEGMAT', 'DS007609', 'DS004902'):
        try:
            extract_source_v6(model, src, out_root, device, args.batch, stai=stai, sai=sai)
        except RuntimeError as e:
            if device.type == 'cuda' and 'out of memory' in str(e).lower():
                print(f'[{src}] CUDA OOM ({e}); falling back to CPU for remaining sources')
                torch.cuda.empty_cache()
                device = torch.device('cpu')
                ef4.patch_to_patch_for_cpu()
                model.to(device)
                try:
                    extract_source_v6(model, src, out_root, device, args.batch, stai=stai, sai=sai)
                except Exception as e2:
                    failures.append((src, str(e2)))
                    traceback.print_exc()
            else:
                failures.append((src, str(e)))
                traceback.print_exc()
    if failures:
        print('FAILED sources: ' + '; '.join(f'{s}: {e}' for s, e in failures))
        sys.exit(1)
    print(f'extraction {args.name} complete')


if __name__ == '__main__':
    main()
