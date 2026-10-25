"""Overnight orchestrator for the four-source v4 experiment.

Stages (resumable; state in overnight/status.json):
  preflight  gates PASS + GPU free + disk
  smoke      8-step GPU smoke (VRAM gate)
  pretrain   3 conditions (isa / cda / isacda) x 3 seeds = 9 runs, ~35 min each
  extract    9 ckpt feature extractions + 3 random-seed extractions
  probe      LOSO readout probes + spectral simple baseline
  report     overnight/FINAL_REPORT.md + console summary

Error policy: per-run failures are recorded and skipped, never abort the night;
only preflight violations and missing prerequisites abort (state is kept, rerun
resumes). Ctrl-C saves state and exits cleanly.

Usage:
  .venv/Scripts/python.exe tools/run_overnight_4src.py            # everything
  .venv/Scripts/python.exe tools/run_overnight_4src.py --only probe
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
import traceback

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
OVER = os.path.join(REPO, 'overnight')
PY = sys.executable
SEEDS = [19260832, 19260833, 19260834]
CONDITIONS = {  # hydra overrides relative to config_4src_v4
    'isa': ['train.loss.CDA_loss=false'],
    'cda': ['train.loss.clisa_loss=false'],
    'isacda': [],
}
TRAIN_TIMEOUT = 5400       # 90 min per training run (~35 min nominal + throttle margin)
GPU_WAIT_TIMEOUT = 1800    # max minutes-scale wait for GPU between runs: 30 min
STATUS = os.path.join(OVER, 'status.json')


def load_status():
    if os.path.exists(STATUS):
        st = json.load(open(STATUS, encoding='utf-8'))
    else:
        st = {}
    # status files written by older versions may lack newer keys
    st.setdefault('runs', {})
    st.setdefault('stages', {})
    st.setdefault('extracts', {})
    return st


def save_status(st):
    os.makedirs(OVER, exist_ok=True)
    st['updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
    json.dump(st, open(STATUS, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


def gpu(query):
    """-1 = query failed (driver hiccup); callers treat as unknown, not as free."""
    try:
        return int(os.popen(f'nvidia-smi --query-gpu={query} --format=csv,noheader,nounits'
                            ).read().strip().splitlines()[0])
    except Exception:
        return -1


def wait_gpu_free(need_free=7000, timeout=GPU_WAIT_TIMEOUT, why='next run'):
    t0 = time.time()
    announced = False
    while time.time() - t0 < timeout:
        free = gpu('memory.free')
        if free >= need_free:
            return True
        if not announced:
            print(f'[gpu-wait] need {need_free} MiB free for {why}, currently {free} MiB; '
                  f'waiting up to {timeout // 60} min...')
            announced = True
        time.sleep(30)
    return False


def cooldown():
    t0 = time.time()
    while time.time() - t0 < 300:
        t, used = gpu('temperature.gpu'), gpu('memory.used')
        if (t < 82 or t == -1) and (used < 1500 or used == -1):
            return
        time.sleep(20)


def log_tail(name, n=2500):
    try:
        return open(os.path.join(OVER, 'logs', name), encoding='utf-8',
                    errors='replace').read()[-n:]
    except Exception:
        return '<log unreadable>'


def run_cmd(cmd, log_name, timeout):
    """Never raises. Returns (rc, timed_out). rc 124 = timeout, 125 = spawn error."""
    os.makedirs(os.path.join(OVER, 'logs'), exist_ok=True)
    log = open(os.path.join(OVER, 'logs', log_name), 'w', encoding='utf-8')
    try:
        p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        return p.returncode, False
    except subprocess.TimeoutExpired:
        log.write(f'\n[orchestrator] TIMEOUT after {timeout}s, process killed\n')
        return 124, True
    except Exception as e:
        log.write(f'\n[orchestrator] spawn error: {e}\n')
        return 125, False
    finally:
        log.close()


def tb_trend(run_name):
    """first-vs-last epoch value for every clisa/cda scalar; True = decreasing."""
    events = glob.glob(os.path.join(REPO, 'log', run_name, '**', 'events.out.tfevents*'),
                       recursive=True)
    if not events:
        return {'trend': 'no_event_files'}
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(os.path.dirname(sorted(events)[-1]),
                          size_guidance={'scalars': 0})
    ea.Reload()
    trends = {}
    single = []
    for tag in ea.Tags().get('scalars', []):
        if 'clisa' not in tag and 'cda' not in tag:
            continue
        v = ea.Scalars(tag)
        if len(v) >= 2:
            trends[tag] = bool(v[-1].value < v[0].value)
        elif len(v) == 1:
            single.append(tag)
    if not trends:
        return {'trend': 'single_epoch_undefined' if single else 'no_loss_scalars',
                'single_point_tags': single}
    return {'trends': trends, 'trend': all(trends.values())}


def latest_ckpt(run_name):
    cks = glob.glob(os.path.join(REPO, 'log', run_name, 'ckpt', '*.ckpt'))
    if not cks:
        return None
    return sorted(cks, key=lambda p: int(''.join(filter(str.isdigit, os.path.basename(p))) or 0))[-1]


def stage_preflight(st):
    try:
        audit = json.load(open('D:/大创/mdjpt_data_v4/build_audit.json', encoding='utf-8'))
    except Exception as e:
        raise RuntimeError(f'cannot read build_audit.json ({e}); run tools/build_sliced_v4.py first')
    assert audit.get('gate1_per_channel_std_0.3_3') == 'PASS' and \
        audit.get('gate2_cda_ratio_lt_3') == 'PASS', 'build gates not PASS'
    for src in ['DASPS', 'DEAP', 'SAM40', 'EEGMAT']:
        if not os.path.exists(f'D:/大创/mdjpt_data_v4/{src}/sliced_data/sliced_len5_step2/saved.npy'):
            raise RuntimeError(f'{src}: sliced data missing; run tools/build_sliced_v4.py')
    free = gpu('memory.free')
    if free != -1 and free < 1500:
        raise RuntimeError(f'GPU busy ({gpu("memory.used")} MiB used; ollama?). '
                           'Free the GPU (unload models / quit Ollama) and rerun.')
    try:
        import shutil
        free_gb = shutil.disk_usage('D:/').free / 1e9
        assert free_gb > 20, f'low disk on D: {free_gb:.1f} GB'
    except (OSError, AssertionError) as e:
        if isinstance(e, AssertionError):
            raise RuntimeError(str(e))
        free_gb = -1
    st['stages']['preflight'] = {'status': 'PASS', 'gpu_free_mib': free,
                                 'disk_free_gb': round(free_gb, 1) if free_gb > 0 else 'unknown'}
    save_status(st)
    print(f"[preflight] PASS (gpu free {free} MiB)")


def stage_smoke(st):
    if st['stages'].get('smoke', {}).get('status') == 'PASS':
        print('[smoke] already PASS')
        return
    name = 'smoke_overnight'
    rc, timed_out = run_cmd([PY, 'train_multi.py', '--config-name=config_4src_v4', 'hydra.run.dir=.',
                             f'log.run_name={name}', 'train.n_pairs=8', 'train.max_epochs=1'],
                            f'{name}.log', 1800)
    ok = rc == 0 and bool(glob.glob(os.path.join(REPO, 'log', name, 'ckpt', '*.ckpt')))
    st['stages']['smoke'] = {'status': 'PASS' if ok else 'FAIL', 'rc': rc}
    save_status(st)
    if not ok:
        raise RuntimeError(f'smoke failed (rc={rc}, timeout={timed_out}):\n{log_tail(f"{name}.log")}')


def stage_pretrain(st):
    plan = [(c, s) for c in CONDITIONS for s in SEEDS]
    for i, (cond, seed) in enumerate(plan):
        name = f'4src_{cond}_s{seed % 100}'
        rec = st['runs'].get(name, {})
        if rec.get('status') == 'done' and latest_ckpt(name):
            print(f'[{i+1}/9] {name} already done')
            continue
        if not wait_gpu_free(7000, GPU_WAIT_TIMEOUT, why=name):
            rec.update(status='gpu_busy',
                       error=f'GPU not free within {GPU_WAIT_TIMEOUT//60} min; '
                             'free it and rerun to resume')
            st['runs'][name] = rec
            save_status(st)
            print(f'[{i+1}/9] {name}: GPU BUSY -> aborted remaining pretrains '
                  '(resume by rerunning)')
            break
        t0 = time.time()
        rc, timed_out = run_cmd([PY, 'train_multi.py', '--config-name=config_4src_v4',
                                 'hydra.run.dir=.', f'log.run_name={name}',
                                 f'train.seed={seed}'] + CONDITIONS[cond],
                                f'{name}.log', TRAIN_TIMEOUT)
        info = {'cond': cond, 'seed': seed, 'rc': rc, 'timed_out': timed_out,
                'seconds': round(time.time() - t0)}
        try:
            info.update(tb_trend(name))
        except Exception as e:
            info['trend'] = f'parse_error: {e}'
        info['status'] = 'done' if rc == 0 and latest_ckpt(name) else 'fail'
        st['runs'][name] = info
        save_status(st)
        print(f"[{i+1}/9] {name}: {info['status']} ({info['seconds']}s, "
              f"trend={info.get('trend')})")
        if info['status'] == 'fail':
            print(f'  -> recorded and continuing; see overnight/logs/{name}.log')
        cooldown()
    done = [r for r in st['runs'].values() if r.get('status') == 'done']
    trends_ok = sum(1 for r in done if r.get('trend') is True)
    print(f'[pretrain] {len(done)}/9 complete; loss trends PASS {trends_ok}/{len(done)}')
    if not done:
        raise RuntimeError('no pretraining run succeeded; check overnight/logs/')


def stage_extract(st):
    random_seeds = {f'random_s{s % 100}': s for s in SEEDS}
    jobs = [(f'4src_{c}_s{s % 100}', latest_ckpt(f'4src_{c}_s{s % 100}'))
            for c in CONDITIONS for s in SEEDS]
    jobs += [(name, None) for name in random_seeds]
    for name, ckpt in jobs:
        out = os.path.join(OVER, 'features', name)
        if os.path.exists(os.path.join(out, 'EEGMAT.npz')):
            print(f'[extract] {name} exists')
            st['extracts'][name] = {'status': 'done', 'cached': True}
            continue
        if name.startswith('4src') and not ckpt:
            print(f'[extract] SKIP {name}: no ckpt (failed pretrain)')
            st['extracts'][name] = {'status': 'skipped', 'reason': 'no ckpt'}
            continue
        cmd = [PY, 'tools/extract_features_4src.py', f'--name={name}']
        if ckpt:
            cmd += [f'--ckpt={ckpt}']
        else:
            cmd += [f'--seed={random_seeds[name]}']
        rc, timed_out = run_cmd(cmd, f'extract_{name}.log', 3600)
        ok = os.path.exists(os.path.join(out, 'EEGMAT.npz'))
        st['extracts'][name] = {'status': 'done' if ok else 'fail', 'rc': rc,
                                'timed_out': timed_out, 'ckpt': ckpt}
        save_status(st)
        print(f"[extract] {name}: {st['extracts'][name]['status']} (rc={rc})")
        if not ok:
            print(f'  -> continuing; see overnight/logs/extract_{name}.log')
    ok_n = sum(1 for v in st['extracts'].values() if v.get('status') == 'done')
    if ok_n == 0:
        raise RuntimeError('all extractions failed; check overnight/logs/extract_*.log')
    print(f'[extract] {ok_n}/{len(jobs)} feature sets ready')


def stage_probe(st):
    rc, timed_out = run_cmd([PY, 'tools/probe_eval_4src.py',
                             f'--features-dir={os.path.join(OVER, "features")}',
                             f'--out={os.path.join(OVER, "results.json")}'],
                            'probe.log', 7200)
    ok = rc == 0 and os.path.exists(os.path.join(OVER, 'results.json'))
    st['stages']['probe'] = {'status': 'PASS' if ok else 'FAIL', 'rc': rc}
    save_status(st)
    print(f'[probe] rc={rc}')
    if not ok:
        raise RuntimeError('probe failed; see overnight/logs/probe.log')


def stage_report(st):
    path = os.path.join(OVER, 'results.json')
    if not os.path.exists(path):
        raise RuntimeError('no results.json; run --only probe first')
    rows = json.load(open(path, encoding='utf-8'))
    rows = [r for r in rows if 'error' not in r]
    errs = [r for r in json.load(open(path, encoding='utf-8')) if 'error' in r]
    lines = ['# 四源 v4 通宵实验报告', '',
             f'生成时间: {time.strftime("%Y-%m-%d %H:%M:%S")}', '',
             '## 1. 预训练完成情况', '',
             '| run | condition | seed | wall time | rc | loss trend |', '|---|---|---|---|---|---|']
    for name in sorted(st.get('runs', {})):
        r = st['runs'][name]
        lines.append(f"| {name} | {r.get('cond')} | {r.get('seed')} | {r.get('seconds', 0)}s "
                     f"| {r.get('rc')} | {r.get('trend')} |")
    done = [r for r in st.get('runs', {}).values() if r.get('status') == 'done']
    trends_ok = sum(1 for r in done if r.get('trend') is True)
    lines += ['', f"完成 {len(done)}/9；损失趋势（首 epoch vs 末 epoch 下降）通过 "
              f"{trends_ok}/{len(done)}。", '']

    lines += ['## 2. LOSO 读出探针（被试留一，池化预测）', '',
              '| features | target | kind | 主指标 | 平凡对照 |', '|---|---|---|---|---|']
    for r in rows:
        if r['kind'] == 'reg':
            main_m = f"{r.get('mse', float('nan')):.4f} (r={r.get('pearson_r') if r.get('pearson_r') is not None else float('nan'):.3f})"
            base_m = f"{r.get('trivial_mse', float('nan')):.4f}"
        else:
            main_m = f"bacc={r.get('balanced_acc', float('nan')):.4f} (acc={r.get('acc', float('nan')):.3f})"
            base_m = f"acc={r.get('trivial_acc', float('nan')):.4f}"
        lines.append(f"| {r['features']} | {r['target']} | {r['kind']} | {main_m} | {base_m} |")
    if errs:
        lines += ['', '### 评价中出错被跳过的组合', ''] + \
                 [f"- {e['features']} / {e['target']}: {e['error']}" for e in errs]

    def agg(sub, key):
        vals = [r[key] for r in sub if key in r and r.get(key) is not None]
        return (float(np.mean(vals)), float(np.std(vals))) if vals else (None, None)

    names = sorted({r['features'] for r in rows})
    lines += ['', '## 3. 聚合（seed 均值 ± 标准差）与判定', '']
    for target in sorted({r['target'] for r in rows}):
        kind = next(r['kind'] for r in rows if r['target'] == target)
        key = 'mse' if kind == 'reg' else 'balanced_acc'
        lines.append(f'### {target} ({kind}, {"lower better" if kind == "reg" else "higher better"})')
        table = []
        for n in names:
            m, s = agg([r for r in rows if r['features'] == n and r['target'] == target], key)
            if m is not None:
                table.append((n, m, s))
        if not table:
            lines.append('(no data)')
            continue
        lines.append(f'| features | {key} (mean±std) |')
        lines.append('|---|---|')
        for n, m, s in table:
            lines.append(f'| {n} | {m:.4f} ± {s:.4f} |')
        sim = next((m for n, m, _ in table if n == 'spectral_simple'), None)
        rand_vals = [m for n, m, _ in table if n.startswith('random')]
        rand = float(np.mean(rand_vals)) if rand_vals else None
        pre_vals = [m for n, m, _ in table if n.startswith('4src')]
        best_pre = (min(pre_vals) if kind == 'reg' else max(pre_vals)) if pre_vals else None
        verdict = []
        if best_pre is not None and rand is not None:
            beat_rand = best_pre < rand if kind == 'reg' else best_pre > rand
            verdict.append(f"预训练最好 vs random: {'优' if beat_rand else '不优'}"
                           f"（{best_pre:.4f} vs {rand:.4f}）")
        else:
            verdict.append('random 或预训练特征缺失，无法归因判定')
        if sim is not None and best_pre is not None:
            beat_sim = best_pre < sim if kind == 'reg' else best_pre > sim
            verdict.append(f"vs 简单谱基线: {'超过' if beat_sim else '未超过'}"
                           f"（{best_pre:.4f} vs {sim:.4f}）")
        lines.append('')
        lines.append('判定: ' + '；'.join(verdict))
        lines.append('')
    lines += ['## 4. 边界（报告必读）', '',
              '- 预训练使用各源全部被试，LOSO 只保证被试级读出泛化，属结构内评价，不是严格归纳设置。',
              '- 探针为线性/岭读出（弱探针）；负结果不能否定表征价值，正结果也不能归因于预训练'
              '除非同时优于 random 与 spectral_simple。',
              '- ISA 仅 DEAP/SAM40/DASPS（对齐证据见 docs/four_source_v4/alignment_evidence/），EEGMAT 仅 CDA。']

    out = os.path.join(OVER, 'FINAL_REPORT.md')
    open(out, 'w', encoding='utf-8').write('\n'.join(lines))
    print(f'[report] -> {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', choices=['preflight', 'smoke', 'pretrain', 'extract', 'probe', 'report'])
    ap.add_argument('--skip-smoke', action='store_true')
    args = ap.parse_args()
    st = load_status()
    stages = [('preflight', stage_preflight), ('smoke', stage_smoke),
              ('pretrain', stage_pretrain), ('extract', stage_extract),
              ('probe', stage_probe), ('report', stage_report)]
    if args.skip_smoke:
        stages = [s for s in stages if s[0] != 'smoke']
    if args.only:
        stages = [s for s in stages if s[0] == args.only]
    t0 = time.time()
    for name, fn in stages:
        try:
            fn(st)
        except KeyboardInterrupt:
            st['stages'][name] = {'status': 'interrupted'}
            save_status(st)
            print(f'[{name}] interrupted by user; state saved, rerun to resume')
            sys.exit(130)
        except Exception as e:
            traceback.print_exc()
            st['stages'][name] = {'status': 'aborted', 'error': str(e)}
            save_status(st)
            print(f'[{name}] ABORTED: {e}')
            print(f'state kept in {STATUS}; fix and rerun (completed stages are skipped).')
            sys.exit(1)
    print(f'overnight pipeline finished in {(time.time()-t0)/3600:.2f} h; '
          f'report: {os.path.join(OVER, "FINAL_REPORT.md")}')


if __name__ == '__main__':
    main()
