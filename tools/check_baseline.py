"""Preflight by default; explicit --gpu-smoke checks unmodified upstream losses."""
import argparse
from datetime import datetime, timezone
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from resource_gate import sample, violation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def worker(directory):
    if os.environ.get("MDJPT_GUARDED_WORKER") != "1":
        raise RuntimeError("Use --gpu-smoke so the resource guard remains active")
    import torch
    from hydra import compose, initialize_config_dir
    from src.model.MultiModel_PL import MultiModel_PL

    torch.set_num_threads(2)
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)
    with initialize_config_dir(config_dir=str(ROOT / "cfgs_multi"), version_base="1.3"):
        cfg = compose(config_name="config_restart")
    cfg.data_cfg_list = [cfg.data_0] * 4
    assert cfg.train.loss.clisa_loss and cfg.train.loss.CDA_loss
    model = MultiModel_PL(cfg).to("cuda").train()
    parameters = [p for p in model.parameters() if p.requires_grad]
    embeddings, covariance_features = [], []
    # Synthetic domains validate execution, not cross-subject correspondence.
    for source in range(4):
        x = torch.randn(6, 1, 60, 625, device="cuda") * (1 + source * 0.2)
        embedding, covariance = model(x)
        embeddings.append(embedding)
        covariance_features.append(covariance)
    isa_parts = [model.clisa_loss(embedding)[0] for embedding in embeddings]
    isa = torch.stack(isa_parts).sum()  # Same sum convention as training_step.
    cda = model.cda_loss(covariance_features)
    total = isa + cfg.train.loss.CDA_factor * cda

    def gradient_norm(loss):
        gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        used = [g for g in gradients if g is not None]
        if not used or not all(torch.isfinite(g).all() for g in used):
            raise RuntimeError("Missing or non-finite objective gradient")
        norm = sum(float(g.detach().float().square().sum()) for g in used) ** 0.5
        if norm <= 0:
            raise RuntimeError("Objective produces no gradient")
        return norm

    isa_norm, cda_norm = gradient_norm(isa), gradient_norm(cda)
    if not torch.isfinite(total):
        raise RuntimeError("Non-finite joint loss")
    optimizer = torch.optim.Adam(parameters, lr=cfg.train.lr, weight_decay=cfg.train.wd)
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    if not all(torch.isfinite(p.grad).all() for p in parameters if p.grad is not None):
        raise RuntimeError("Non-finite combined gradient")
    before = {name: p.detach().clone() for name, p in model.named_parameters()
              if p.grad is not None}
    optimizer.step()
    changed = any(not torch.equal(before[name], p.detach())
                  for name, p in model.named_parameters() if name in before)
    if not changed:
        raise RuntimeError("Optimizer did not update any active parameter")
    torch.cuda.synchronize()
    dump(directory / "smoke.json", {
        "status": "passed", "scope": "synthetic execution only; not scientific acceptance",
        "source": "unmodified upstream MultiModel_PL, SimCLRLoss, CDALoss",
        "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "parameters": sum(p.numel() for p in model.parameters()),
        "isa_per_domain": [float(x.detach()) for x in isa_parts],
        "isa_sum": float(isa.detach()), "cda_raw": float(cda.detach()),
        "cda_weight": cfg.train.loss.CDA_factor, "total": float(total.detach()),
        "isa_gradient_norm": isa_norm, "cda_gradient_norm": cda_norm,
        "optimizer_changed_parameters": changed,
        "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-smoke", action="store_true")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.chdir(ROOT)  # Upstream loads channel_interpolate.npy relative to CWD.
    if args.worker:
        worker(args.worker)
        return 0
    packages = ("torch", "torchvision", "numpy", "scipy", "timm", "pytorch-lightning",
                "hydra-core", "mne", "hdf5storage", "tensorboard", "transformers", "wandb")
    versions = {name: metadata.version(name) for name in packages}
    resources = sample()
    reason = violation(**resources, startup=True)
    print(json.dumps({"versions": versions, "resources": resources,
                      "gpu_start_blocker": reason}, indent=2))
    if not args.gpu_smoke:
        return 0
    if reason:
        print("BLOCKED: " + reason)
        return 2
    directory = ROOT / "outputs" / datetime.now(timezone.utc).strftime("smoke_%Y%m%dT%H%M%S_%fZ")
    directory.mkdir(parents=True, exist_ok=False)
    import msvcrt
    lock_path = ROOT / "outputs" / "gpu.lock"
    with lock_path.open("a+b") as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        with (directory / "stdout.log").open("w") as out, (directory / "stderr.log").open("w") as err:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", MDJPT_GUARDED_WORKER="1")
            process = subprocess.Popen([sys.executable, __file__, "--worker", str(directory)],
                                       cwd=ROOT, env=env, stdout=out, stderr=err,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
            samples = []
            failure = None
            try:
                while process.poll() is None:
                    resources = sample(process)
                    samples.append(resources)
                    failure = violation(**resources)
                    if failure:
                        break
                    time.sleep(0.5)
            except BaseException as exc:
                failure = str(exc) or type(exc).__name__
            finally:
                if process.poll() is None:
                    # Terminate our worker's process tree, including venv proxy children.
                    import psutil
                    root = psutil.Process(process.pid)
                    children = root.children(recursive=True)
                    for child in reversed(children):
                        try:
                            child.terminate()
                        except psutil.NoSuchProcess:
                            pass
                    process.terminate()
                code = process.wait()
            dump(directory / "resources.json", {
                "returncode": code, "resource_failure": failure, "samples": samples})
        print(str(directory))
        if failure or code != 0 or not (directory / "smoke.json").is_file():
            print("TECHNICAL_FAILURE: " + str(failure or f"worker exit {code}; see stderr.log"))
            return 3
        print((directory / "smoke.json").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
