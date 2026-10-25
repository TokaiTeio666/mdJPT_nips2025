"""Extract frozen encoder features for the four sources, in canonical build order.

Mirrors tools/build_sliced_v4.py window ordering exactly (subject-major, group-major,
key order, offset order, EEGMAT rest truncated to 29), so feature row i corresponds to
data_sample_{i}.npy and to the label table collected here from v4 windows.jsonl.

Per source saves overnight/features/{name}/{SRC}.npz:
  fea (N, D) float32      encoder output (fea_clisa, L2-normalized by cnn_MLLA)
  subject_idx (N,)        subject position (0..n_subs-1), for LOSO grouping
  group_idx (N,)          build group index
  targets dict            native labels:
    DASPS  sam_valence, sam_arousal        (regression)
    DEAP   valence, arousal                (regression)
    SAM40  task (int), stress (float, nan where unavailable)
    EEGMAT condition (int: 0=rest, 1=arithmetic)

Usage:
  python tools/extract_features_4src.py --name random_s32 --seed 19260832
  python tools/extract_features_4src.py --name isacda_s32 --ckpt log/4src_isacda_s32/ckpt/epoch=02.ckpt
"""
import argparse
import glob
import importlib.util
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)  # channel_interpolate.npy + relative paths

_spec = importlib.util.spec_from_file_location('bsv', os.path.join(REPO, 'tools', 'build_sliced_v4.py'))
bsv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bsv)

from hydra import compose, initialize_config_dir
from src.model.MultiModel_PL import MultiModel_PL

V4_ROOT = bsv.V4_ROOT
OUT_DIR = os.path.join(REPO, 'overnight', 'features')


def load_records(sub_dir):
    """{key: [window json records sorted by offset]}"""
    by_key = {}
    with open(os.path.join(sub_dir, 'windows.jsonl'), encoding='utf-8') as f:
        for line in f:
            w = json.loads(line)
            key = w['alignment_key'] if w['alignment_key'] else w['trial_id']
            by_key.setdefault(key, []).append(w)
    for k in by_key:
        by_key[k].sort(key=lambda w: w['offset'])
    return by_key


import json


def build_model(cfg, ckpt, seed, device):
    torch.manual_seed(seed)
    model = MultiModel_PL(cfg)
    if ckpt:
        sd = torch.load(ckpt, map_location='cpu')['state_dict']
        model.load_state_dict(sd)
        print(f'loaded ckpt {ckpt}')
    else:
        print(f'random init (seed={seed})')
    model.eval().to(device)
    return model


@torch.no_grad()
def forward_windows(model, segs, channels, device):
    x = torch.from_numpy(np.stack(segs)).unsqueeze(1).to(device)  # (B,1,60,625)
    x = model.channel_project(x, channels)
    fea, _ = model.forward(x, 0)
    return fea.float().cpu().numpy()


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
        # upstream to_patch hardcodes cuda (MLLA_new.py:65); neutral device version for CPU
        import src.model.MLLA_new as m
        def _tp(data, patch_size=50, stride=25):
            b, t, d = data.shape
            n = (t - patch_size) // stride + 1
            p = torch.zeros((b, n, patch_size), device=data.device)
            for i in range(n):
                p[:, i, :] = data[:, i * stride:i * stride + patch_size, 0]
            return p
        m.to_patch = _tp
    print(f'device={device} (free {free} MiB)')

    with initialize_config_dir(config_dir=os.path.join(REPO, 'cfgs_multi'), version_base='1.3'):
        cfg = compose(config_name='config_4src_v4')
    cfg.data_cfg_list = [cfg.data_0, cfg.data_1, cfg.data_2, cfg.data_3, cfg.data_4]
    cfg.data_cfg_list = [c for c in cfg.data_cfg_list if c.dataset_name != 'None']

    model = build_model(cfg, args.ckpt, args.seed, device)

    out_root = os.path.join(OUT_DIR, args.name)
    os.makedirs(out_root, exist_ok=True)
    for src, spec in bsv.SOURCES.items():
        sub_dirs = sorted(os.listdir(os.path.join(V4_ROOT, src)))
        first = load_records(os.path.join(V4_ROOT, src, sub_dirs[0]))
        plan = bsv.group_plan(src, first)
        channels = json.load(open(os.path.join(V4_ROOT, src, sub_dirs[0], 'summary.json'),
                                  encoding='utf-8'))['target_channels']
        sam_tasks = sorted({k.split(':')[1] for k in first}) if src == 'SAM40' else []

        feas, subj, grp, targets = [], [], [], {k: [] for k in (
            ['sam_valence', 'sam_arousal'] if src == 'DASPS' else
            ['valence', 'arousal'] if src == 'DEAP' else
            ['task', 'stress'] if src == 'SAM40' else
            ['condition'])}
        for s_pos, sub in enumerate(sub_dirs):
            by_key = load_records(os.path.join(V4_ROOT, src, sub))
            signals = np.load(os.path.join(V4_ROOT, src, sub, 'signals.npy'))
            z, _, _, _ = bsv.robust_z(signals)
            segs = []
            for g_idx, (_, keys) in enumerate(plan):
                for k in keys:
                    recs = bsv.truncated_keys(src, k, by_key[k])
                    for w in recs:
                        segs.append(z[:, w['offset']:w['offset'] + bsv.SEG].astype(np.float32))
                        subj.append(s_pos)
                        grp.append(g_idx)
                        lab = w['labels']
                        if src == 'DASPS':
                            targets['sam_valence'].append(float(lab['sam'][0]))
                            targets['sam_arousal'].append(float(lab['sam'][1]))
                        elif src == 'DEAP':
                            targets['valence'].append(float(lab['valence']))
                            targets['arousal'].append(float(lab['arousal']))
                        elif src == 'SAM40':
                            targets['task'].append(sam_tasks.index(lab['task']))
                            targets['stress'].append(float(lab['stress_rating'])
                                                     if lab.get('rating_available') else float('nan'))
                        else:
                            targets['condition'].append(g_idx)
            for i0 in range(0, len(segs), args.batch):
                feas.append(forward_windows(model, segs[i0:i0 + args.batch], channels, device))
        fea = np.concatenate(feas)
        np.savez_compressed(os.path.join(out_root, f'{src}.npz'), fea=fea,
                            subject_idx=np.array(subj, np.int32),
                            group_idx=np.array(grp, np.int32),
                            **{k: np.array(v, np.float32) for k, v in targets.items()})
        print(f'[{src}] fea {fea.shape} -> {out_root}/{src}.npz')


if __name__ == '__main__':
    main()
