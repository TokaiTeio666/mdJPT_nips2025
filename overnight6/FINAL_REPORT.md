六源

生成时间: 2026-10-30 17:22:13

## 1. 预训练完成情况

| run | condition | seed | wall time | rc | loss trend |
|---|---|---|---|---|---|
| 6src_cda_s32 | cda | 19260832 | 2724s | 0 | True |
| 6src_cda_s33 | cda | 19260833 | 2823s | 0 | True |
| 6src_cda_s34 | cda | 19260834 | 2723s | 0 | True |
| 6src_isa_s32 | isa | 19260832 | 1641s | 0 | True |
| 6src_isa_s33 | isa | 19260833 | 1425s | 0 | True |
| 6src_isa_s34 | isa | 19260834 | 1334s | 0 | True |
| 6src_isacda_s32 | isacda | 19260832 | 2946s | 0 | False |
| 6src_isacda_s33 | isacda | 19260833 | 3680s | 0 | False |
| 6src_isacda_s34 | isacda | 19260834 | 3115s | 0 | False |

完成 9/9；损失趋势（首 epoch vs 末 epoch 下降）通过 6/9。

## 2. LOSO 读出探针（被试留一，池化预测）

| features | target | kind | 主指标 | 平凡对照 |
|---|---|---|---|---|
| 6src_cda_s32 | sam_valence | reg | 485.3150 (r=0.016) | 2.9539 |
| 6src_cda_s32 | sam_arousal | reg | 720.3561 (r=-0.058) | 5.8573 |
| 6src_cda_s33 | sam_valence | reg | 16037.6957 (r=-0.001) | 2.9539 |
| 6src_cda_s33 | sam_arousal | reg | 6178.6466 (r=-0.046) | 5.8573 |
| 6src_cda_s34 | sam_valence | reg | 11.7961 (r=-0.053) | 2.9539 |
| 6src_cda_s34 | sam_arousal | reg | 35.8053 (r=-0.139) | 5.8573 |
| 6src_isa_s32 | sam_valence | reg | 646165.6799 (r=-0.030) | 2.9539 |
| 6src_isa_s32 | sam_arousal | reg | 5093435.3948 (r=-0.011) | 5.8573 |
| 6src_isa_s33 | sam_valence | reg | 9167.8397 (r=-0.004) | 2.9539 |
| 6src_isa_s33 | sam_arousal | reg | 208674.3103 (r=-0.010) | 5.8573 |
| 6src_isa_s34 | sam_valence | reg | 8.2058 (r=0.032) | 2.9539 |
| 6src_isa_s34 | sam_arousal | reg | 20.3068 (r=-0.098) | 5.8573 |
| 6src_isacda_s32 | sam_valence | reg | 12241.1623 (r=-0.048) | 2.9539 |
| 6src_isacda_s32 | sam_arousal | reg | 24504.8729 (r=-0.016) | 5.8573 |
| 6src_isacda_s33 | sam_valence | reg | 3042558.4535 (r=0.006) | 2.9539 |
| 6src_isacda_s33 | sam_arousal | reg | 979720.5264 (r=0.002) | 5.8573 |
| 6src_isacda_s34 | sam_valence | reg | 290.3612 (r=-0.004) | 2.9539 |
| 6src_isacda_s34 | sam_arousal | reg | 1642.2011 (r=-0.001) | 5.8573 |
| random_s32 | sam_valence | reg | 39.9754 (r=-0.028) | 2.9539 |
| random_s32 | sam_arousal | reg | 25.8467 (r=0.003) | 5.8573 |
| random_s33 | sam_valence | reg | 3367.4198 (r=0.044) | 2.9539 |
| random_s33 | sam_arousal | reg | 5244.5771 (r=0.039) | 5.8573 |
| random_s34 | sam_valence | reg | 884.3224 (r=0.033) | 2.9539 |
| random_s34 | sam_arousal | reg | 1397.3943 (r=0.054) | 5.8573 |
| spectral_simple | sam_valence | reg | 4.1757 (r=-0.072) | 2.9539 |
| spectral_simple | sam_arousal | reg | 8.3756 (r=-0.076) | 5.8573 |
| 6src_cda_s32 | valence | reg | 5.7892 (r=-0.006) | 4.5509 |
| 6src_cda_s32 | arousal | reg | 5.9344 (r=0.002) | 4.1105 |
| 6src_cda_s33 | valence | reg | 3589.4335 (r=-0.009) | 4.5509 |
| 6src_cda_s33 | arousal | reg | 48498.3268 (r=0.000) | 4.1105 |
| 6src_cda_s34 | valence | reg | 10.8404 (r=0.007) | 4.5509 |
| 6src_cda_s34 | arousal | reg | 12.5608 (r=-0.012) | 4.1105 |
| 6src_isa_s32 | valence | reg | 4.6548 (r=0.002) | 4.5509 |
| 6src_isa_s32 | arousal | reg | 4.2119 (r=-0.052) | 4.1105 |
| 6src_isa_s33 | valence | reg | 257055.2400 (r=-0.003) | 4.5509 |
| 6src_isa_s33 | arousal | reg | 89074.8472 (r=0.007) | 4.1105 |
| 6src_isa_s34 | valence | reg | 4.9832 (r=0.028) | 4.5509 |
| 6src_isa_s34 | arousal | reg | 4.5133 (r=0.011) | 4.1105 |
| 6src_isacda_s32 | valence | reg | 4.6853 (r=-0.046) | 4.5509 |
| 6src_isacda_s32 | arousal | reg | 4.2233 (r=-0.036) | 4.1105 |
| 6src_isacda_s33 | valence | reg | 4.6417 (r=0.030) | 4.5509 |
| 6src_isacda_s33 | arousal | reg | 4.3352 (r=-0.078) | 4.1105 |
| 6src_isacda_s34 | valence | reg | 524759.4403 (r=-0.002) | 4.5509 |
| 6src_isacda_s34 | arousal | reg | 86061.0430 (r=-0.001) | 4.1105 |
| random_s32 | valence | reg | 82.7177 (r=0.003) | 4.5509 |
| random_s32 | arousal | reg | 80.3621 (r=-0.003) | 4.1105 |
| random_s33 | valence | reg | 6.0624 (r=-0.009) | 4.5509 |
| random_s33 | arousal | reg | 18.3440 (r=0.001) | 4.1105 |
| random_s34 | valence | reg | 5.5074 (r=0.005) | 4.5509 |
| random_s34 | arousal | reg | 4.7152 (r=-0.016) | 4.1105 |
| spectral_simple | valence | reg | 7.0962 (r=0.007) | 4.5509 |
| spectral_simple | arousal | reg | 5.7573 (r=0.098) | 4.1105 |
| 6src_cda_s32 | task | clf | bacc=0.3036 (acc=0.304) | acc=0.2500 |
| 6src_cda_s32 | stress | reg | 1767.2038 (r=0.036) | 4.1585 |
| 6src_cda_s33 | task | clf | bacc=0.2775 (acc=0.277) | acc=0.2500 |
| 6src_cda_s33 | stress | reg | 4.6889 (r=-0.032) | 4.1585 |
| 6src_cda_s34 | task | clf | bacc=0.3013 (acc=0.301) | acc=0.2500 |
| 6src_cda_s34 | stress | reg | 11.5976 (r=-0.015) | 4.1585 |
| 6src_isa_s32 | task | clf | bacc=0.2811 (acc=0.281) | acc=0.2500 |
| 6src_isa_s32 | stress | reg | 9.3530 (r=-0.023) | 4.1585 |
| 6src_isa_s33 | task | clf | bacc=0.2854 (acc=0.285) | acc=0.2500 |
| 6src_isa_s33 | stress | reg | 10.7393 (r=0.129) | 4.1585 |
| 6src_isa_s34 | task | clf | bacc=0.2661 (acc=0.266) | acc=0.2500 |
| 6src_isa_s34 | stress | reg | 24065.3601 (r=0.002) | 4.1585 |
| 6src_isacda_s32 | task | clf | bacc=0.3028 (acc=0.303) | acc=0.2500 |
| 6src_isacda_s32 | stress | reg | 3699.9574 (r=0.010) | 4.1585 |
| 6src_isacda_s33 | task | clf | bacc=0.2972 (acc=0.297) | acc=0.2500 |
| 6src_isacda_s33 | stress | reg | 15845.0475 (r=0.009) | 4.1585 |
| 6src_isacda_s34 | task | clf | bacc=0.3055 (acc=0.305) | acc=0.2500 |
| 6src_isacda_s34 | stress | reg | 4.9881 (r=-0.098) | 4.1585 |
| random_s32 | task | clf | bacc=0.2775 (acc=0.277) | acc=0.2500 |
| random_s32 | stress | reg | 22.4802 (r=-0.025) | 4.1585 |
| random_s33 | task | clf | bacc=0.2778 (acc=0.278) | acc=0.2500 |
| random_s33 | stress | reg | 17.8627 (r=0.036) | 4.1585 |
| random_s34 | task | clf | bacc=0.2989 (acc=0.299) | acc=0.2500 |
| random_s34 | stress | reg | 3439.6216 (r=0.010) | 4.1585 |
| spectral_simple | task | clf | bacc=0.4680 (acc=0.468) | acc=0.2500 |
| spectral_simple | stress | reg | 5.3578 (r=0.115) | 4.1585 |
| 6src_cda_s32 | condition | clf | bacc=0.9392 (acc=0.939) | acc=0.5000 |
| 6src_cda_s33 | condition | clf | bacc=0.9440 (acc=0.944) | acc=0.5000 |
| 6src_cda_s34 | condition | clf | bacc=0.9186 (acc=0.919) | acc=0.5000 |
| 6src_isa_s32 | condition | clf | bacc=0.9100 (acc=0.910) | acc=0.5000 |
| 6src_isa_s33 | condition | clf | bacc=0.9128 (acc=0.913) | acc=0.5000 |
| 6src_isa_s34 | condition | clf | bacc=0.9109 (acc=0.911) | acc=0.5000 |
| 6src_isacda_s32 | condition | clf | bacc=0.9109 (acc=0.911) | acc=0.5000 |
| 6src_isacda_s33 | condition | clf | bacc=0.9272 (acc=0.927) | acc=0.5000 |
| 6src_isacda_s34 | condition | clf | bacc=0.9128 (acc=0.913) | acc=0.5000 |
| random_s32 | condition | clf | bacc=0.9215 (acc=0.921) | acc=0.5000 |
| random_s33 | condition | clf | bacc=0.8673 (acc=0.867) | acc=0.5000 |
| random_s34 | condition | clf | bacc=0.9205 (acc=0.920) | acc=0.5000 |
| spectral_simple | condition | clf | bacc=0.6446 (acc=0.645) | acc=0.5000 |
| 6src_cda_s32 | stai_state | reg | 679.7244 (r=-0.017) | 110.9064 |
| 6src_cda_s32 | stai_trait | reg | 367.5350 (r=0.007) | 74.1864 |
| 6src_cda_s33 | stai_state | reg | 1486.8311 (r=0.014) | 110.9064 |
| 6src_cda_s33 | stai_trait | reg | 1308.1197 (r=-0.001) | 74.1864 |
| 6src_cda_s34 | stai_state | reg | 158.2506 (r=0.083) | 110.9064 |
| 6src_cda_s34 | stai_trait | reg | 114.7679 (r=0.071) | 74.1864 |
| 6src_isa_s32 | stai_state | reg | 127.5062 (r=-0.031) | 110.9064 |
| 6src_isa_s32 | stai_trait | reg | 84.7821 (r=-0.015) | 74.1864 |
| 6src_isa_s33 | stai_state | reg | 128.0531 (r=-0.040) | 110.9064 |
| 6src_isa_s33 | stai_trait | reg | 84.4515 (r=0.024) | 74.1864 |
| 6src_isa_s34 | stai_state | reg | 148.2478 (r=-0.080) | 110.9064 |
| 6src_isa_s34 | stai_trait | reg | 89.1479 (r=0.033) | 74.1864 |
| 6src_isacda_s32 | stai_state | reg | 85057508.7790 (r=-0.000) | 110.9064 |
| 6src_isacda_s32 | stai_trait | reg | 25515183.8996 (r=-0.000) | 74.1864 |
| 6src_isacda_s33 | stai_state | reg | 164.9576 (r=0.057) | 110.9064 |
| 6src_isacda_s33 | stai_trait | reg | 93.5006 (r=-0.018) | 74.1864 |
| 6src_isacda_s34 | stai_state | reg | 7389554.1360 (r=0.000) | 110.9064 |
| 6src_isacda_s34 | stai_trait | reg | 1922025.8076 (r=-0.000) | 74.1864 |
| random_s32 | stai_state | reg | 704.4048 (r=-0.012) | 110.9064 |
| random_s32 | stai_trait | reg | 154.7290 (r=-0.040) | 74.1864 |
| random_s33 | stai_state | reg | 531.1523 (r=-0.025) | 110.9064 |
| random_s33 | stai_trait | reg | 264.4397 (r=-0.025) | 74.1864 |
| random_s34 | stai_state | reg | 5575.9893 (r=-0.002) | 110.9064 |
| random_s34 | stai_trait | reg | 1870.5350 (r=-0.005) | 74.1864 |
| spectral_simple | stai_state | reg | 190.2755 (r=-0.023) | 110.9064 |
| spectral_simple | stai_trait | reg | 108.5156 (r=0.082) | 74.1864 |

### 评价中出错被跳过的组合

- 6src_cda_s32 / SAI_NS: insufficient valid samples (0)
- 6src_cda_s32 / SAI_SD: insufficient valid samples (0)
- 6src_cda_s32 / SAI_delta: insufficient valid samples (0)
- 6src_cda_s33 / SAI_NS: insufficient valid samples (0)
- 6src_cda_s33 / SAI_SD: insufficient valid samples (0)
- 6src_cda_s33 / SAI_delta: insufficient valid samples (0)
- 6src_cda_s34 / SAI_NS: insufficient valid samples (0)
- 6src_cda_s34 / SAI_SD: insufficient valid samples (0)
- 6src_cda_s34 / SAI_delta: insufficient valid samples (0)
- 6src_isa_s32 / SAI_NS: insufficient valid samples (0)
- 6src_isa_s32 / SAI_SD: insufficient valid samples (0)
- 6src_isa_s32 / SAI_delta: insufficient valid samples (0)
- 6src_isa_s33 / SAI_NS: insufficient valid samples (0)
- 6src_isa_s33 / SAI_SD: insufficient valid samples (0)
- 6src_isa_s33 / SAI_delta: insufficient valid samples (0)
- 6src_isa_s34 / SAI_NS: insufficient valid samples (0)
- 6src_isa_s34 / SAI_SD: insufficient valid samples (0)
- 6src_isa_s34 / SAI_delta: insufficient valid samples (0)
- 6src_isacda_s32 / SAI_NS: insufficient valid samples (0)
- 6src_isacda_s32 / SAI_SD: insufficient valid samples (0)
- 6src_isacda_s32 / SAI_delta: insufficient valid samples (0)
- 6src_isacda_s33 / SAI_NS: insufficient valid samples (0)
- 6src_isacda_s33 / SAI_SD: insufficient valid samples (0)
- 6src_isacda_s33 / SAI_delta: insufficient valid samples (0)
- 6src_isacda_s34 / SAI_NS: insufficient valid samples (0)
- 6src_isacda_s34 / SAI_SD: insufficient valid samples (0)
- 6src_isacda_s34 / SAI_delta: insufficient valid samples (0)
- random_s32 / SAI_NS: insufficient valid samples (0)
- random_s32 / SAI_SD: insufficient valid samples (0)
- random_s32 / SAI_delta: insufficient valid samples (0)
- random_s33 / SAI_NS: insufficient valid samples (0)
- random_s33 / SAI_SD: insufficient valid samples (0)
- random_s33 / SAI_delta: insufficient valid samples (0)
- random_s34 / SAI_NS: insufficient valid samples (0)
- random_s34 / SAI_SD: insufficient valid samples (0)
- random_s34 / SAI_delta: insufficient valid samples (0)
- spectral_simple / SAI_NS: insufficient valid samples (0)
- spectral_simple / SAI_SD: insufficient valid samples (0)
- spectral_simple / SAI_delta: insufficient valid samples (0)

## 3. 聚合（seed 均值 ± 标准差）与判定

### arousal (reg, lower better)
| features | mse (mean±std) |
|---|---|
| 6src_cda_s32 | 5.9344 ± 0.0000 |
| 6src_cda_s33 | 48498.3268 ± 0.0000 |
| 6src_cda_s34 | 12.5608 ± 0.0000 |
| 6src_isa_s32 | 4.2119 ± 0.0000 |
| 6src_isa_s33 | 89074.8472 ± 0.0000 |
| 6src_isa_s34 | 4.5133 ± 0.0000 |
| 6src_isacda_s32 | 4.2233 ± 0.0000 |
| 6src_isacda_s33 | 4.3352 ± 0.0000 |
| 6src_isacda_s34 | 86061.0430 ± 0.0000 |
| random_s32 | 80.3621 ± 0.0000 |
| random_s33 | 18.3440 ± 0.0000 |
| random_s34 | 4.7152 ± 0.0000 |
| spectral_simple | 5.7573 ± 0.0000 |

判定: random 或预训练特征缺失，无法归因判定

### condition (clf, higher better)
| features | balanced_acc (mean±std) |
|---|---|
| 6src_cda_s32 | 0.9392 ± 0.0000 |
| 6src_cda_s33 | 0.9440 ± 0.0000 |
| 6src_cda_s34 | 0.9186 ± 0.0000 |
| 6src_isa_s32 | 0.9100 ± 0.0000 |
| 6src_isa_s33 | 0.9128 ± 0.0000 |
| 6src_isa_s34 | 0.9109 ± 0.0000 |
| 6src_isacda_s32 | 0.9109 ± 0.0000 |
| 6src_isacda_s33 | 0.9272 ± 0.0000 |
| 6src_isacda_s34 | 0.9128 ± 0.0000 |
| random_s32 | 0.9215 ± 0.0000 |
| random_s33 | 0.8673 ± 0.0000 |
| random_s34 | 0.9205 ± 0.0000 |
| spectral_simple | 0.6446 ± 0.0000 |

判定: random 或预训练特征缺失，无法归因判定

### sam_arousal (reg, lower better)
| features | mse (mean±std) |
|---|---|
| 6src_cda_s32 | 720.3561 ± 0.0000 |
| 6src_cda_s33 | 6178.6466 ± 0.0000 |
| 6src_cda_s34 | 35.8053 ± 0.0000 |
| 6src_isa_s32 | 5093435.3948 ± 0.0000 |
| 6src_isa_s33 | 208674.3103 ± 0.0000 |
| 6src_isa_s34 | 20.3068 ± 0.0000 |
| 6src_isacda_s32 | 24504.8729 ± 0.0000 |
| 6src_isacda_s33 | 979720.5264 ± 0.0000 |
| 6src_isacda_s34 | 1642.2011 ± 0.0000 |
| random_s32 | 25.8467 ± 0.0000 |
| random_s33 | 5244.5771 ± 0.0000 |
| random_s34 | 1397.3943 ± 0.0000 |
| spectral_simple | 8.3756 ± 0.0000 |

判定: random 或预训练特征缺失，无法归因判定

### sam_valence (reg, lower better)
| features | mse (mean±std) |
|---|---|
| 6src_cda_s32 | 485.3150 ± 0.0000 |
| 6src_cda_s33 | 16037.6957 ± 0.0000 |
| 6src_cda_s34 | 11.7961 ± 0.0000 |
| 6src_isa_s32 | 646165.6799 ± 0.0000 |
| 6src_isa_s33 | 9167.8397 ± 0.0000 |
| 6src_isa_s34 | 8.2058 ± 0.0000 |
| 6src_isacda_s32 | 12241.1623 ± 0.0000 |
| 6src_isacda_s33 | 3042558.4535 ± 0.0000 |
| 6src_isacda_s34 | 290.3612 ± 0.0000 |
| random_s32 | 39.9754 ± 0.0000 |
| random_s33 | 3367.4198 ± 0.0000 |
| random_s34 | 884.3224 ± 0.0000 |
| spectral_simple | 4.1757 ± 0.0000 |

判定: random 或预训练特征缺失，无法归因判定

### stai_state (reg, lower better)
| features | mse (mean±std) |
|---|---|
| 6src_cda_s32 | 679.7244 ± 0.0000 |
| 6src_cda_s33 | 1486.8311 ± 0.0000 |
| 6src_cda_s34 | 158.2506 ± 0.0000 |
| 6src_isa_s32 | 127.5062 ± 0.0000 |
| 6src_isa_s33 | 128.0531 ± 0.0000 |
| 6src_isa_s34 | 148.2478 ± 0.0000 |
| 6src_isacda_s32 | 85057508.7790 ± 0.0000 |
| 6src_isacda_s33 | 164.9576 ± 0.0000 |
| 6src_isacda_s34 | 7389554.1360 ± 0.0000 |
| random_s32 | 704.4048 ± 0.0000 |
| random_s33 | 531.1523 ± 0.0000 |
| random_s34 | 5575.9893 ± 0.0000 |
| spectral_simple | 190.2755 ± 0.0000 |

判定: random 或预训练特征缺失，无法归因判定

### stai_trait (reg, lower better)
| features | mse (mean±std) |
|---|---|
| 6src_cda_s32 | 367.5350 ± 0.0000 |
| 6src_cda_s33 | 1308.1197 ± 0.0000 |
| 6src_cda_s34 | 114.7679 ± 0.0000 |
| 6src_isa_s32 | 84.7821 ± 0.0000 |
| 6src_isa_s33 | 84.4515 ± 0.0000 |
| 6src_isa_s34 | 89.1479 ± 0.0000 |
| 6src_isacda_s32 | 25515183.8996 ± 0.0000 |
| 6src_isacda_s33 | 93.5006 ± 0.0000 |
| 6src_isacda_s34 | 1922025.8076 ± 0.0000 |
| random_s32 | 154.7290 ± 0.0000 |
| random_s33 | 264.4397 ± 0.0000 |
| random_s34 | 1870.5350 ± 0.0000 |
| spectral_simple | 108.5156 ± 0.0000 |

判定: random 或预训练特征缺失，无法归因判定

### stress (reg, lower better)
| features | mse (mean±std) |
|---|---|
| 6src_cda_s32 | 1767.2038 ± 0.0000 |
| 6src_cda_s33 | 4.6889 ± 0.0000 |
| 6src_cda_s34 | 11.5976 ± 0.0000 |
| 6src_isa_s32 | 9.3530 ± 0.0000 |
| 6src_isa_s33 | 10.7393 ± 0.0000 |
| 6src_isa_s34 | 24065.3601 ± 0.0000 |
| 6src_isacda_s32 | 3699.9574 ± 0.0000 |
| 6src_isacda_s33 | 15845.0475 ± 0.0000 |
| 6src_isacda_s34 | 4.9881 ± 0.0000 |
| random_s32 | 22.4802 ± 0.0000 |
| random_s33 | 17.8627 ± 0.0000 |
| random_s34 | 3439.6216 ± 0.0000 |
| spectral_simple | 5.3578 ± 0.0000 |

判定: random 或预训练特征缺失，无法归因判定

### task (clf, higher better)
| features | balanced_acc (mean±std) |
|---|---|
| 6src_cda_s32 | 0.3036 ± 0.0000 |
| 6src_cda_s33 | 0.2775 ± 0.0000 |
| 6src_cda_s34 | 0.3013 ± 0.0000 |
| 6src_isa_s32 | 0.2811 ± 0.0000 |
| 6src_isa_s33 | 0.2854 ± 0.0000 |
| 6src_isa_s34 | 0.2661 ± 0.0000 |
| 6src_isacda_s32 | 0.3028 ± 0.0000 |
| 6src_isacda_s33 | 0.2972 ± 0.0000 |
| 6src_isacda_s34 | 0.3055 ± 0.0000 |
| random_s32 | 0.2775 ± 0.0000 |
| random_s33 | 0.2778 ± 0.0000 |
| random_s34 | 0.2989 ± 0.0000 |
| spectral_simple | 0.4680 ± 0.0000 |

判定: random 或预训练特征缺失，无法归因判定

### valence (reg, lower better)
| features | mse (mean±std) |
|---|---|
| 6src_cda_s32 | 5.7892 ± 0.0000 |
| 6src_cda_s33 | 3589.4335 ± 0.0000 |
| 6src_cda_s34 | 10.8404 ± 0.0000 |
| 6src_isa_s32 | 4.6548 ± 0.0000 |
| 6src_isa_s33 | 257055.2400 ± 0.0000 |
| 6src_isa_s34 | 4.9832 ± 0.0000 |
| 6src_isacda_s32 | 4.6853 ± 0.0000 |
| 6src_isacda_s33 | 4.6417 ± 0.0000 |
| 6src_isacda_s34 | 524759.4403 ± 0.0000 |
| random_s32 | 82.7177 ± 0.0000 |
| random_s33 | 6.0624 ± 0.0000 |
| random_s34 | 5.5074 ± 0.0000 |
| spectral_simple | 7.0962 ± 0.0000 |

判定: random 或预训练特征缺失，无法归因判定

## 4. 边界（报告必读）

- 预训练使用各源全部被试，LOSO 只保证被试级读出泛化，属结构内评价，不是严格归纳设置。
- 探针为线性/岭读出（弱探针）；负结果不能否定表征价值，正结果也不能归因于预训练除非同时优于 random 与 spectral_simple。
- ISA 仅 DEAP/SAM40/DASPS（对齐证据见 docs/four_source_v4/alignment_evidence/），EEGMAT 仅 CDA。