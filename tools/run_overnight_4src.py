"""Overnight orchestrator for the four-source v4 experiment.

Stages (resumable; state in overnight/status.json):
  preflight  gates PASS + GPU free + disk
  smoke      8-step GPU smoke (VRAM gate)
  pretrain   3 conditions (isa / cda / isacda) x 3 seeds = 9 runs, ~35 min each
  extract    9 ckpt feature extractions + 3 random-seed extractions
  probe      LOSO readout probes + spectral simple baseline
  report     overnight/FINAL_REPORT.md + console summary

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
TRAIN_TIMEOUT = 5400  # 90 min per run (~35 min nominal + throttle margin)
STATUS = os.path.join(OVER, 'status.json')


def load_status():
    return json.load(open(STATUS, encoding='utf-8')) if os.path.exists(STATUS) \
        else {'runs': {}, 'stages': {}}


def save_status(st):
    os.makedirs(OVER, exist_ok=True)
    json.dump(st, open(STATUS, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


def gpu(query):
    try:
        return int(os.popen(f'nvidia-smi --query-gpu={query} --format=csv,noheader,nounits'
                            ).read().strip().splitlines()[0])
    except Exception:
        return -1


def wait_gpu_free(need_free=7000, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if gpu('memory.free') >= need_free:
            return True
        time.sleep(30)
    return False


def cooldown():
    t0 = time.time()
    while time.time() - t0 < 300:
        if gpu('temperature.gpu') < 82 and gpu('memory.used') < 1500:
            return
        time.sleep(20)


def run_cmd(cmd, log_name, timeout):
    os.makedirs(os.path.join(OVER, 'logs'), exist_ok=True)
    log = open(os.path.join(OVER, 'logs', log_name), 'w', encoding='utf-8')
    p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    return p.returncode


def tb_trend(run_name):
    """first-vs-last epoch value for every clisa/cda scalar; True = decreasing."""
    events = glob.glob(os.path.join(REPO, 'log', run_name, '**', 'events.out.tfevents*'),
                       recursive=True)
    if not events:
        return {'trend': 'no_event_files'}
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(os.path.dirname(events[0]))
    ea.Reload()
    trends = {}
    for tag in ea.Tags().get('scalars', []):
        if 'clisa' not in tag and 'cda' not in tag:
            continue
        v = ea.Scalars(tag)
        if len(v) >= 2:
            trends[tag] = bool(v[-1].value < v[0].value)
    if not trends:
        return {'trend': 'no_loss_scalars'}
    return {'trends': trends, 'trend': all(trends.values())}


def latest_ckpt(run_name):
    cks = glob.glob(os.path.join(REPO, 'log', run_name, 'ckpt', '*.ckpt'))
    if not cks:
        return None
    return sorted(cks, key=lambda p: int(''.join(filter(str.isdigit, os.path.basename(p))) or 0))[-1]


def stage_preflight(st):
    audit = json.load(open('D:/大创/mdjpt_data_v4/build_audit.json', encoding='utf-8'))
    assert audit['gate1_per_channel_std_0.3_3'] == 'PASS' and \
        audit['gate2_cda_ratio_lt_3'] == 'PASS', 'build gates not PASS'
    free = gpu('memory.free')
    if free < 1500:
        raise RuntimeError(f'GPU busy ({gpu("memory.used")} MiB used; ollama?). '
                           'Free the GPU (unload models / quit Ollama) and rerun.')
    du = os.popen('df -k D:').read()
    free_gb = int(du.splitlines()[1].split()[3]) / 1e6
    assert free_gb > 20, f'low disk on D: {free_gb:.1f} GB'
    st['stages']['preflight'] = {'status': 'PASS', 'gpu_free_mib': free, 'disk_free_gb': round(free_gb, 1)}
    save_status(st)
    print(f"[preflight] PASS (gpu free {free} MiB, disk {free_gb:.0f} GB)")


def stage_smoke(st):
    if st['stages'].get('smoke', {}).get('status') == 'PASS':
        print('[smoke] already PASS'); return
    name = 'smoke_overnight'
    rc = run_cmd([PY, 'train_multi.py', '--config-name=config_4src_v4', 'hydra.run.dir=.',
                  f'log.run_name={name}', 'train.n_pairs=8', 'train.max_epochs=1'],
                 f'{name}.log', 1800)
    ok = rc == 0 and bool(glob.glob(os.path.join(REPO, 'log', name, 'ckpt', '*.ckpt')))
    st['stages']['smoke'] = {'status': 'PASS' if ok else 'FAIL', 'rc': rc}
    save_status(st)
    if not ok:
        tail = open(os.path.join(OVER, 'logs', f'{name}.log'), encoding='utf-8').read()[-2000:]
        raise RuntimeError(f'smoke failed (rc={rc}):\n{tail}')


def stage_pretrain(st):
    plan = [(c, s) for c in CONDITIONS for s in SEEDS]
    for i, (cond, seed) in enumerate(plan):
        name = f'4src_{cond}_s{seed % 100}'
        rec = st['runs'].get(name, {})
        if rec.get('status') == 'done' and latest_ckpt(name):
            print(f'[{i+1}/9] {name} already done'); continue
        if not wait_gpu_free(7000, 600):
            rec.update(status='gpu_busy', error='GPU not free for 10 min'); save_status(st)
            raise RuntimeError('GPU stayed busy; aborting overnight run')
        t0 = time.time()
        rc = run_cmd([PY, 'train_multi.py', '--config-name=config_4src_v4', 'hydra.run.dir=.',
                      f'log.run_name={name}', f'train.seed={seed}'] + CONDITIONS[cond],
                     f'{name}.log', TRAIN_TIMEOUT)
        info = {'cond': cond, 'seed': seed, 'rc': rc, 'seconds': round(time.time() - t0)}
        try:
            info.update(tb_trend(name))
        except Exception as e:
            info['trend_parse_error'] = str(e)
        info['status'] = 'done' if rc == 0 and latest_ckpt(name) else 'fail'
        st['runs'][name] = info
        save_status(st)
        print(f"[{i+1}/9] {name}: {info['status']} ({info['seconds']}s, trend={info.get('trend')})")
        if info['status'] == 'fail':
            print(f'  -> see overnight/logs/{name}.log')
        cooldown()
    done = [r for r in st['runs'].values() if r.get('status') == 'done']
    print(f'[pretrain] {len(done)}/9 complete; trends PASS '
          f"{sum(1 for r in done if r.get('trend') is True)}/{len(done)}")


def stage_extract(st):
    jobs = [(f'4src_{c}_s{s % 100}', latest_ckpt(f'4src_{c}_s{s % 100}'))
            for c in CONDITIONS for s in SEEDS]
    jobs += [(f'random_s{s % 100}', None) for s in SEEDS]
    for name, ckpt in jobs:
        out = os.path.join(OVER, 'features', name)
        if os.path.exists(os.path.join(out, 'EEGMAT.npz')):
            print(f'[extract] {name} exists'); continue
        if name.startswith('4src') and not ckpt:
            print(f'[extract] SKIP {name}: no ckpt (failed run)'); continue
        cmd = [PY, 'tools/extract_features_4src.py', f'--name={name}']
        if ckpt:
            cmd += [f'--ckpt={ckpt}']
        else:
            cmd += [f'--seed={SEEDS[int(name[-2:]) - 32]}']
        rc = run_cmd(cmd, f'extract_{name}.log', 3600)
        print(f'[extract] {name}: rc={rc} (ckpt={bool(ckpt)})')
    st['stages']['extract'] = {'status': 'done'}
    save_status(st)


def stage_probe(st):
    rc = run_cmd([PY, 'tools/probe_eval_4src.py',
                  f'--features-dir={os.path.join(OVER, "features")}',
                  f'--out={os.path.join(OVER, "results.json")}'], 'probe.log', 7200)
    st['stages']['probe'] = {'status': 'PASS' if rc == 0 else 'FAIL'}
    save_status(st)
    print(f'[probe] rc={rc}')


def stage_report(st):
    rows = json.load(open(os.path.join(OVER, 'results.json'), encoding='utf-8'))
    lines = ['# 四源 v4 通宵实验报告', '',
             f'生成时间: {time.strftime("%Y-%m-%d %H:%M:%S")}', '',
             '## 1. 预训练完成情况', '',
             '| run | condition | seed | wall time | loss trend |', '|---|---|---|---|---|']
    for name, r in st['runs'].items():
        lines.append(f"| {name} | {r.get('cond')} | {r.get('seed')} | "
                     f"{r.get('seconds', 0)}s | {r.get('trend')} |")
    done = [r for r in st['runs'].values() if r.get('status') == 'done']
    lines += ['', f"完成 {len(done)}/9；损失趋势（首 epoch vs 末 epoch 下降）通过 "
              f"{sum(1 for r in done if r.get('trend') is True)}/{len(done)}。", '']

    lines += ['## 2. LOSO 读出探针（被试留一，池化预测）', '',
              '| features | target | kind | 主指标 | 对照 |', '|---|---|---|---|---|']
    for r in rows:
        main_m = r.get('mse', r.get('balanced_acc'))
        base_m = r.get('trivial_mse', r.get('trivial_acc'))
        extra = f"r={r['pearson_r']:.3f}" if r['kind'] == 'reg' else f"acc={r['acc']:.3f}"
        lines.append(f"| {r['features']} | {r['target']} | {r['kind']} | "
                     f"{main_m:.4f} ({extra}) | {base_m:.4f} |")

    def agg(sub, key):
        vals = [r[key] for r in sub if key in r]
        return (float(np.mean(vals)), float(np.std(vals))) if vals else (None, None)

    names = sorted({r['features'] for r in rows})
    lines += ['', '## 3. 聚合（seed 均值 ± 标准差）与判定', '']
    for target in sorted({r['target'] for r in rows}):
        kind = next(r['kind'] for r in rows if r['target'] == target)
        key = 'mse' if kind == 'reg' else 'balanced_acc'
        better = np.mean if kind == 'reg' else np.mean
        lines.append(f'### {target} ({kind}, {"lower better" if kind == "reg" else "higher better"})')
        table = []
        for n in names:
            m, s = agg([r for r in rows if r['features'] == n and r['target'] == target], key)
            if m is not None:
                table.append((n, m, s))
        if not table:
            continue
        lines.append('| features | %s (mean±std) |' % key + '|---|---|')
        for n, m, s in table:
            lines.append(f'| {n} | {m:.4f} ± {s:.4f} |')
        sim = next((m for n, m, _ in table if n == 'spectral_simple'), None)
        rand = np.mean([m for n, m, _ in table if n.startswith('random')])
        best_pre = min([m for n, m, _ in table if n.startswith('4src')], default=None) \
            if kind == 'reg' else max([m for n, m, _ in table if n.startswith('4src')], default=None)
        verdict = []
        if best_pre is not None and rand is not None:
            verdict.append(f"预训练最好 vs random: {'优' if (best_pre < rand if kind=='reg' else best_pre > rand) else '不优'}"
                           f"（{best_pre:.4f} vs {rand:.4f}）")
        if sim is not None and best_pre is not None:
            beat = best_pre < sim if kind == 'reg' else best_pre > sim
            verdict.append(f"vs 简单谱基线: {'超过' if beat else '未超过'}（{best_pre:.4f} vs {sim:.4f}）")
        lines.append('')
        lines.append('判定: ' + '；'.join(verdict) if verdict else '')
        lines.append('')
    lines += ['## 4. 边界（报告必读）', '',
              '- 预训练使用各源全部被试，LOSO 只保证被试级读出泛化，属结构内评价，不是严格归纳设置。',
              '- 探针为线性/岭读出（弱探针）；负结果不能否定表征价值，正结果也不能归因于预训练'
              '除非同时优于 random 与 spectral_simple。',
              '- ISA 仅 DEAP/SAM40/DASPS（对齐证据见 docs/four_source_v4/alignment_evidence/），EEGMAT 仅 CDA。']

    path = os.path.join(OVER, 'FINAL_REPORT.md')
    open(path, 'w', encoding='utf-8').write('\n'.join(lines))
    print(f'[report] -> {path}')


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
        except Exception as e:
            traceback.print_exc()
            print(f'[{name}] ABORTED: {e}')
            print(f'status kept in {STATUS}; fix and rerun (completed stages are skipped).')
            sys.exit(1)
    print(f'overnight pipeline finished in {(time.time()-t0)/3600:.2f} h; '
          f'report: {os.path.join(OVER, "FINAL_REPORT.md")}')


if __name__ == '__main__':
    main()
