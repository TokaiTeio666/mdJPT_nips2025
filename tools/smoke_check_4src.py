"""CPU smoke check for the four-source v4 pretraining path (no GPU required).

Verifies, on one real joint batch (38 windows: DEAP16+SAM40 8+DASPS 8+EEGMAT 6):
  1. build -> MultiDataModule -> EEGSampler -> EEG_Dataset contract holds end to end
  2. MultiModel_PL.training_step returns a finite loss
  3. ISA whitelist: clisa_loss is invoked for exactly the whitelisted datasets
     (DEAP/SAM40/DASPS) and never for EEGMAT  -> gate 3 of the plan
  4. loss.backward() produces finite, non-zero grads; one optimizer step succeeds

GPU smoke (VRAM gate) remains: python train_multi.py --config-name=config_4src_v4
  train.n_pairs=8 train.max_epochs=1 log.run_name=smoke_4src
"""
import sys

sys.path.insert(0, 'D:/大创/mdJPT_nips2025')

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

# upstream to_patch hardcodes .to('cuda') (MLLA_new.py:65); swap in a device-neutral
# equivalent for this CPU check only -- src/ stays untouched
import src.model.MLLA_new as mlla_mod
def _to_patch_cpu(data, patch_size=50, stride=25):
    batchsize, timelen, dim = data.shape
    num_patches = (timelen - patch_size) // stride + 1
    patches = torch.zeros((batchsize, num_patches, patch_size), device=data.device)
    for i in range(num_patches):
        s, e = i * stride, i * stride + patch_size
        patches[:, i, :] = data[:, s:e, 0]
    return patches
mlla_mod.to_patch = _to_patch_cpu

from src.data.multi_dataloader import MultiDataModule
from src.model.MultiModel_PL import MultiModel_PL

with initialize_config_dir(config_dir='D:/大创/mdJPT_nips2025/cfgs_multi', version_base='1.3'):
    cfg = compose(config_name='config_4src_v4')

cfg.data_cfg_list = [cfg.data_0, cfg.data_1, cfg.data_2, cfg.data_3, cfg.data_4]
cfg.data_cfg_list = [c for c in cfg.data_cfg_list if c.dataset_name != 'None']
print('datasets:', [c.dataset_name for c in cfg.data_cfg_list])

dm = MultiDataModule(cfg.data_cfg_list, 0, 1, num_workers=0, n_pairs=4)
dm.setup('fit')
loader = dm.train_dataloader()
batch = next(iter(loader))
x_list, y_list = batch
# DataLoader wraps the whole per-source group in a leading dim (batch_size=1 default);
# training_step indexes it away the same way
x_list = [x[0] for x in x_list]
print('batch windows per source:', [tuple(x.shape) for x in x_list])
assert sum(x.shape[0] for x in x_list) == 38, 'expected 38 windows/step'

model = MultiModel_PL(cfg)
print('params:', sum(p.numel() for p in model.parameters()))

calls = []
def hook(_module, inputs, _output):
    calls.append(inputs[0].shape[0])
model.clisa_loss.register_forward_hook(hook)

loss = model.training_step(batch, 0)
assert torch.isfinite(loss), loss
print(f'training_step loss = {loss.item():.4f} (finite)')

# whitelist: batch sizes are 2*n_vids in data_cfg_list order
expected = [2 * c.n_vids for c in cfg.data_cfg_list]
allowed = [b for c, b in zip(cfg.data_cfg_list, expected)
           if c.dataset_name in list(cfg.train.loss.clisa_datasets)]
assert sorted(calls) == sorted(allowed), (calls, allowed)
print(f'GATE3 clisa invoked on {len(calls)}/{len(cfg.data_cfg_list)} sources '
      f'(batch sizes {calls}); EEGMAT skipped: '
      f'{all(b != 2 * cfg.data_3.n_vids for b in calls)}')

loss.backward()
finite_grads = 0
nonzero_grads = 0
for p in model.parameters():
    if p.grad is not None:
        assert torch.isfinite(p.grad).all()
        finite_grads += 1
        if p.grad.abs().sum() > 0:
            nonzero_grads += 1
print(f'grads finite: {finite_grads} tensors, non-zero: {nonzero_grads}')
assert nonzero_grads > 0

opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
step_before = torch.cat([p.detach().flatten() for p in model.parameters()]).clone()
loss2 = model.training_step(batch, 1)
loss2.backward()
opt.step()
delta = (torch.cat([p.detach().flatten() for p in model.parameters()]) - step_before).abs().max()
print(f'optimizer step ok, max param delta = {delta.item():.2e}')
assert delta > 0
print('CPU SMOKE: ALL CHECKS PASSED')
