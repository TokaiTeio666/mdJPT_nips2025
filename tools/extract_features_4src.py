"""Extract frozen encoder features for the four sources, in canonical build order.

Mirrors tools/build_sliced_v4.py window ordering exactly (subject-major, group-major,
key order, offset order, EEGMAT rest truncated to 29), so feature row i corresponds to
data_sample_{i}.npy and to the label table collected here from v4 windows.jsonl.

Per source saves overnight/features/{name}/{SRC}.npz (per-source resume: existing
files are skipped):
  fea (N, D) float32      encoder output (fea_clisa, L2-normalized by cnn_MLLA)
  subject_idx (N,)        subject position (0..n_subs-1), for LOSO grouping
  group_idx (N,)          build group index
  targets dict            native labels:
    DASPS  sam_valence, sam_arousal        (regression)
    DEAP   valence, arousal                (regression)
    SAM40  task (int), stress (float, nan where unavailable)
    EEGMAT condition (int: 0=rest, 1=arithmetic)

Error policy: CUDA OOM (e.g. another process grabs the GPU mid-run) falls back to
CPU automatically for the remaining sources; corrupt/missing inputs fail the source
with a clear message and nonzero exit.

Usage:
  python tools/extract_features_4src.py --name random_s32 --seed 19260832
  python tools/extract_features_4src.py --name isacda_s32 --ckpt log/4src_isacda_s32/ckpt/epoch=02.ckpt
"""
import argparse
import importlib.util
import json
import os
import sys
import traceback

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


def load_ckpt_state(ckpt):
    try:
        return torch.load(ckpt, map_location='cpu')['state_dict']
    except Exception:
        # torch>=2.6 defaults to weights_only=True; Lightning ckpts may carry
        # non-tensor fields -> explicit opt-out for files we produced ourselves
        return torch.load(ckpt, map_location='cpu', weights_only=False)['state_dict']


def patch_to_patch_for_cpu():
    import src.model.MLLA_new as m

    def _tp(data, patch_size=50, stride=25):
        b, t, d = data.shape
        n = (t - patch_size) // stride + 1
        p = torch.zeros((b, n, patch_size), device=data.device)
        for i in range(n):
            p[:, i, :] = data[:, i * stride:i * stride + patch_size, 0]
        return p
    m.to_patch = _tp


@torch.no_grad()
def forward_windows(model, segs, channels, device):
    x = torch.from_numpy(np.stack(segs)).unsqueeze(1).to(device)  # (B,1,60,625)
    x = model.channel_project(x, channels)
    fea, _ = model.forward(x, 0)
    return fea.float().cpu().numpy()


def extract_source(model, src, out_root, device, batch):
    out_path = os.path.join(out_root, f'{src}.npz')
    if os.path.exists(out_path):
        print(f'[{src}] exists, skip')
        return
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
                    off = w['offset']
                    if off + bsv.SEG > z.shape[1]:
                        raise RuntimeError(f'{src}/{sub}/{k}: window exceeds signal')
                    segs.append(z[:, off:off + bsv.SEG].astype(np.float32))
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
        for i0 in range(0, len(segs), batch):
            feas.append(forward_windows(model, segs[i0:i0 + batch], channels, device))
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
        patch_to_patch_for_cpu()
    print(f'device={device} (free {free} MiB)')

    with initialize_config_dir(config_dir=os.path.join(REPO, 'cfgs_multi'), version_base='1.3'):
        cfg = compose(config_name='config_4src_v4')
    cfg.data_cfg_list = [cfg.data_0, cfg.data_1, cfg.data_2, cfg.data_3, cfg.data_4]
    cfg.data_cfg_list = [c for c in cfg.data_cfg_list if c.dataset_name != 'None']

    torch.manual_seed(args.seed)
    model = MultiModel_PL(cfg)
    if args.ckpt:
        if not os.path.exists(args.ckpt):
            print(f'ERROR: ckpt not found: {args.ckpt}')
            sys.exit(2)
        try:
            model.load_state_dict(load_ckpt_state(args.ckpt))
            print(f'loaded ckpt {args.ckpt}')
        except Exception as e:
            print(f'ERROR: ckpt load failed ({e}); corrupt or partial checkpoint?')
            sys.exit(2)
    else:
        print(f'random init (seed={args.seed})')
    model.eval().to(device)

    out_root = os.path.join(OUT_DIR, args.name)
    os.makedirs(out_root, exist_ok=True)
    failures = []
    for src in bsv.SOURCES:
        try:
            extract_source(model, src, out_root, device, args.batch)
        except RuntimeError as e:
            if device.type == 'cuda' and 'out of memory' in str(e).lower():
                print(f'[{src}] CUDA OOM ({e}); falling back to CPU for remaining sources')
                torch.cuda.empty_cache()
                device = torch.device('cpu')
                patch_to_patch_for_cpu()
                model.to(device)
                try:
                    extract_source(model, src, out_root, device, args.batch)
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
