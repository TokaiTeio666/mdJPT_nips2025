# 四源 v4 通宵实验报告

生成时间: 2026-10-25 22:05:45

## 1. 预训练完成情况

| run | condition | seed | wall time | rc | loss trend |
|---|---|---|---|---|---|
| 4src_cda_s32 | cda | 19260832 | 2070s | 0 | True |
| 4src_cda_s33 | cda | 19260833 | 1942s | 0 | True |
| 4src_cda_s34 | cda | 19260834 | 2470s | 0 | True |
| 4src_isa_s32 | isa | 19260832 | 1370s | 0 | True |
| 4src_isa_s33 | isa | 19260833 | 1133s | 0 | True |
| 4src_isa_s34 | isa | 19260834 | 1209s | 0 | True |
| 4src_isacda_s32 | isacda | 19260832 | 2657s | 0 | False |
| 4src_isacda_s33 | isacda | 19260833 | 2236s | 0 | False |
| 4src_isacda_s34 | isacda | 19260834 | 2187s | 0 | False |

完成 9/9；损失趋势（首 epoch vs 末 epoch 下降）通过 6/9。

## 2. LOSO 读出探针（被试留一，池化预测）

| features | target | kind | 主指标 | 平凡对照 |
|---|---|---|---|---|
| 4src_cda_s32 | sam_valence | reg | 38.9098 (r=0.089) | 2.9539 |
| 4src_cda_s32 | sam_arousal | reg | 102.0962 (r=0.064) | 5.8573 |
| 4src_cda_s33 | sam_valence | reg | 674.7159 (r=-0.103) | 2.9539 |
| 4src_cda_s33 | sam_arousal | reg | 790.8795 (r=-0.113) | 5.8573 |
| 4src_cda_s34 | sam_valence | reg | 10.0348 (r=0.030) | 2.9539 |
| 4src_cda_s34 | sam_arousal | reg | 27.7035 (r=0.097) | 5.8573 |
| 4src_isa_s32 | sam_valence | reg | 3822049.9338 (r=-0.004) | 2.9539 |
| 4src_isa_s32 | sam_arousal | reg | 3613393.5576 (r=0.001) | 5.8573 |
| 4src_isa_s33 | sam_valence | reg | 4.9081 (r=-0.012) | 2.9539 |
| 4src_isa_s33 | sam_arousal | reg | 10.4477 (r=-0.040) | 5.8573 |
| 4src_isa_s34 | sam_valence | reg | 1034.0926 (r=0.007) | 2.9539 |
| 4src_isa_s34 | sam_arousal | reg | 6542.6641 (r=0.006) | 5.8573 |
| 4src_isacda_s32 | sam_valence | reg | 14500425.4672 (r=0.017) | 2.9539 |
| 4src_isacda_s32 | sam_arousal | reg | 46814912.3015 (r=0.015) | 5.8573 |
| 4src_isacda_s33 | sam_valence | reg | 9.4095 (r=0.171) | 2.9539 |
| 4src_isacda_s33 | sam_arousal | reg | 19.4952 (r=0.043) | 5.8573 |
| 4src_isacda_s34 | sam_valence | reg | 577.6527 (r=-0.055) | 2.9539 |
| 4src_isacda_s34 | sam_arousal | reg | 8168.3201 (r=0.048) | 5.8573 |
| random_s32 | sam_valence | reg | 39.9754 (r=-0.028) | 2.9539 |
| random_s32 | sam_arousal | reg | 25.8467 (r=0.003) | 5.8573 |
| random_s33 | sam_valence | reg | 3367.4198 (r=0.044) | 2.9539 |
| random_s33 | sam_arousal | reg | 5244.5771 (r=0.039) | 5.8573 |
| random_s34 | sam_valence | reg | 884.3224 (r=0.033) | 2.9539 |
| random_s34 | sam_arousal | reg | 1397.3943 (r=0.054) | 5.8573 |
| spectral_simple | sam_valence | reg | 4.1757 (r=-0.072) | 2.9539 |
| spectral_simple | sam_arousal | reg | 8.3756 (r=-0.076) | 5.8573 |
| 4src_cda_s32 | valence | reg | 5.6952 (r=0.000) | 4.5509 |
| 4src_cda_s32 | arousal | reg | 13.4777 (r=-0.011) | 4.1105 |
| 4src_cda_s33 | valence | reg | 25.8189 (r=-0.011) | 4.5509 |
| 4src_cda_s33 | arousal | reg | 87.8578 (r=-0.004) | 4.1105 |
| 4src_cda_s34 | valence | reg | 22.7794 (r=0.003) | 4.5509 |
| 4src_cda_s34 | arousal | reg | 18.5248 (r=-0.001) | 4.1105 |
| 4src_isa_s32 | valence | reg | 5.0061 (r=-0.004) | 4.5509 |
| 4src_isa_s32 | arousal | reg | 4.3757 (r=-0.015) | 4.1105 |
| 4src_isa_s33 | valence | reg | 4.8058 (r=-0.031) | 4.5509 |
| 4src_isa_s33 | arousal | reg | 4.3124 (r=0.021) | 4.1105 |
| 4src_isa_s34 | valence | reg | 4.6481 (r=-0.005) | 4.5509 |
| 4src_isa_s34 | arousal | reg | 4.2184 (r=-0.004) | 4.1105 |
| 4src_isacda_s32 | valence | reg | 4.8237 (r=0.000) | 4.5509 |
| 4src_isacda_s32 | arousal | reg | 4.2611 (r=-0.046) | 4.1105 |
| 4src_isacda_s33 | valence | reg | 4.6790 (r=0.009) | 4.5509 |
| 4src_isacda_s33 | arousal | reg | 4.2653 (r=-0.044) | 4.1105 |
| 4src_isacda_s34 | valence | reg | 1093917419.3552 (r=0.000) | 4.5509 |
| 4src_isacda_s34 | arousal | reg | 2249065822.4502 (r=0.000) | 4.1105 |
| random_s32 | valence | reg | 82.7177 (r=0.003) | 4.5509 |
| random_s32 | arousal | reg | 80.3621 (r=-0.003) | 4.1105 |
| random_s33 | valence | reg | 6.0624 (r=-0.009) | 4.5509 |
| random_s33 | arousal | reg | 18.3440 (r=0.001) | 4.1105 |
| random_s34 | valence | reg | 5.5074 (r=0.005) | 4.5509 |
| random_s34 | arousal | reg | 4.7152 (r=-0.016) | 4.1105 |
| spectral_simple | valence | reg | 7.0962 (r=0.007) | 4.5509 |
| spectral_simple | arousal | reg | 5.7573 (r=0.098) | 4.1105 |
| 4src_cda_s32 | task | clf | bacc=0.2598 (acc=0.260) | acc=0.2500 |
| 4src_cda_s32 | stress | reg | 19.2702 (r=0.016) | 4.1585 |
| 4src_cda_s33 | task | clf | bacc=0.2722 (acc=0.272) | acc=0.2500 |
| 4src_cda_s33 | stress | reg | 38.7885 (r=-0.019) | 4.1585 |
| 4src_cda_s34 | task | clf | bacc=0.2564 (acc=0.256) | acc=0.2500 |
| 4src_cda_s34 | stress | reg | 53.2490 (r=0.010) | 4.1585 |
| 4src_isa_s32 | task | clf | bacc=0.2860 (acc=0.286) | acc=0.2500 |
| 4src_isa_s32 | stress | reg | 4.8956 (r=-0.041) | 4.1585 |
| 4src_isa_s33 | task | clf | bacc=0.2953 (acc=0.295) | acc=0.2500 |
| 4src_isa_s33 | stress | reg | 11.2266 (r=-0.007) | 4.1585 |
| 4src_isa_s34 | task | clf | bacc=0.2877 (acc=0.288) | acc=0.2500 |
| 4src_isa_s34 | stress | reg | 4.8681 (r=-0.014) | 4.1585 |
| 4src_isacda_s32 | task | clf | bacc=0.2845 (acc=0.284) | acc=0.2500 |
| 4src_isacda_s32 | stress | reg | 379.2735 (r=0.020) | 4.1585 |
| 4src_isacda_s33 | task | clf | bacc=0.3019 (acc=0.302) | acc=0.2500 |
| 4src_isacda_s33 | stress | reg | 4.6613 (r=0.005) | 4.1585 |
| 4src_isacda_s34 | task | clf | bacc=0.3019 (acc=0.302) | acc=0.2500 |
| 4src_isacda_s34 | stress | reg | 8.1045 (r=-0.034) | 4.1585 |
| random_s32 | task | clf | bacc=0.2775 (acc=0.277) | acc=0.2500 |
| random_s32 | stress | reg | 22.4802 (r=-0.025) | 4.1585 |
| random_s33 | task | clf | bacc=0.2778 (acc=0.278) | acc=0.2500 |
| random_s33 | stress | reg | 17.8627 (r=0.036) | 4.1585 |
| random_s34 | task | clf | bacc=0.2989 (acc=0.299) | acc=0.2500 |
| random_s34 | stress | reg | 3439.6216 (r=0.010) | 4.1585 |
| spectral_simple | task | clf | bacc=0.4680 (acc=0.468) | acc=0.2500 |
| spectral_simple | stress | reg | 5.3578 (r=0.115) | 4.1585 |
| 4src_cda_s32 | condition | clf | bacc=0.9033 (acc=0.903) | acc=0.5000 |
| 4src_cda_s33 | condition | clf | bacc=0.9042 (acc=0.904) | acc=0.5000 |
| 4src_cda_s34 | condition | clf | bacc=0.9229 (acc=0.923) | acc=0.5000 |
| 4src_isa_s32 | condition | clf | bacc=0.9152 (acc=0.915) | acc=0.5000 |
| 4src_isa_s33 | condition | clf | bacc=0.9167 (acc=0.917) | acc=0.5000 |
| 4src_isa_s34 | condition | clf | bacc=0.9128 (acc=0.913) | acc=0.5000 |
| 4src_isacda_s32 | condition | clf | bacc=0.9291 (acc=0.929) | acc=0.5000 |
| 4src_isacda_s33 | condition | clf | bacc=0.9210 (acc=0.921) | acc=0.5000 |
| 4src_isacda_s34 | condition | clf | bacc=0.9267 (acc=0.927) | acc=0.5000 |
| random_s32 | condition | clf | bacc=0.9215 (acc=0.921) | acc=0.5000 |
| random_s33 | condition | clf | bacc=0.8673 (acc=0.867) | acc=0.5000 |
| random_s34 | condition | clf | bacc=0.9205 (acc=0.920) | acc=0.5000 |
| spectral_simple | condition | clf | bacc=0.6446 (acc=0.645) | acc=0.5000 |

## 3. 聚合（seed 均值 ± 标准差）与判定

### arousal (reg, lower better)
| features | mse (mean±std) |
|---|---|
| 4src_cda_s32 | 13.4777 ± 0.0000 |
| 4src_cda_s33 | 87.8578 ± 0.0000 |
| 4src_cda_s34 | 18.5248 ± 0.0000 |
| 4src_isa_s32 | 4.3757 ± 0.0000 |
| 4src_isa_s33 | 4.3124 ± 0.0000 |
| 4src_isa_s34 | 4.2184 ± 0.0000 |
| 4src_isacda_s32 | 4.2611 ± 0.0000 |
| 4src_isacda_s33 | 4.2653 ± 0.0000 |
| 4src_isacda_s34 | 2249065822.4502 ± 0.0000 |
| random_s32 | 80.3621 ± 0.0000 |
| random_s33 | 18.3440 ± 0.0000 |
| random_s34 | 4.7152 ± 0.0000 |
| spectral_simple | 5.7573 ± 0.0000 |

判定: 预训练最好 vs random: 优（4.2184 vs 34.4738）；vs 简单谱基线: 超过（4.2184 vs 5.7573）

### condition (clf, higher better)
| features | balanced_acc (mean±std) |
|---|---|
| 4src_cda_s32 | 0.9033 ± 0.0000 |
| 4src_cda_s33 | 0.9042 ± 0.0000 |
| 4src_cda_s34 | 0.9229 ± 0.0000 |
| 4src_isa_s32 | 0.9152 ± 0.0000 |
| 4src_isa_s33 | 0.9167 ± 0.0000 |
| 4src_isa_s34 | 0.9128 ± 0.0000 |
| 4src_isacda_s32 | 0.9291 ± 0.0000 |
| 4src_isacda_s33 | 0.9210 ± 0.0000 |
| 4src_isacda_s34 | 0.9267 ± 0.0000 |
| random_s32 | 0.9215 ± 0.0000 |
| random_s33 | 0.8673 ± 0.0000 |
| random_s34 | 0.9205 ± 0.0000 |
| spectral_simple | 0.6446 ± 0.0000 |

判定: 预训练最好 vs random: 优（0.9291 vs 0.9031）；vs 简单谱基线: 超过（0.9291 vs 0.6446）

### sam_arousal (reg, lower better)
| features | mse (mean±std) |
|---|---|
| 4src_cda_s32 | 102.0962 ± 0.0000 |
| 4src_cda_s33 | 790.8795 ± 0.0000 |
| 4src_cda_s34 | 27.7035 ± 0.0000 |
| 4src_isa_s32 | 3613393.5576 ± 0.0000 |
| 4src_isa_s33 | 10.4477 ± 0.0000 |
| 4src_isa_s34 | 6542.6641 ± 0.0000 |
| 4src_isacda_s32 | 46814912.3015 ± 0.0000 |
| 4src_isacda_s33 | 19.4952 ± 0.0000 |
| 4src_isacda_s34 | 8168.3201 ± 0.0000 |
| random_s32 | 25.8467 ± 0.0000 |
| random_s33 | 5244.5771 ± 0.0000 |
| random_s34 | 1397.3943 ± 0.0000 |
| spectral_simple | 8.3756 ± 0.0000 |

判定: 预训练最好 vs random: 优（10.4477 vs 2222.6060）；vs 简单谱基线: 未超过（10.4477 vs 8.3756）

### sam_valence (reg, lower better)
| features | mse (mean±std) |
|---|---|
| 4src_cda_s32 | 38.9098 ± 0.0000 |
| 4src_cda_s33 | 674.7159 ± 0.0000 |
| 4src_cda_s34 | 10.0348 ± 0.0000 |
| 4src_isa_s32 | 3822049.9338 ± 0.0000 |
| 4src_isa_s33 | 4.9081 ± 0.0000 |
| 4src_isa_s34 | 1034.0926 ± 0.0000 |
| 4src_isacda_s32 | 14500425.4672 ± 0.0000 |
| 4src_isacda_s33 | 9.4095 ± 0.0000 |
| 4src_isacda_s34 | 577.6527 ± 0.0000 |
| random_s32 | 39.9754 ± 0.0000 |
| random_s33 | 3367.4198 ± 0.0000 |
| random_s34 | 884.3224 ± 0.0000 |
| spectral_simple | 4.1757 ± 0.0000 |

判定: 预训练最好 vs random: 优（4.9081 vs 1430.5725）；vs 简单谱基线: 未超过（4.9081 vs 4.1757）

### stress (reg, lower better)
| features | mse (mean±std) |
|---|---|
| 4src_cda_s32 | 19.2702 ± 0.0000 |
| 4src_cda_s33 | 38.7885 ± 0.0000 |
| 4src_cda_s34 | 53.2490 ± 0.0000 |
| 4src_isa_s32 | 4.8956 ± 0.0000 |
| 4src_isa_s33 | 11.2266 ± 0.0000 |
| 4src_isa_s34 | 4.8681 ± 0.0000 |
| 4src_isacda_s32 | 379.2735 ± 0.0000 |
| 4src_isacda_s33 | 4.6613 ± 0.0000 |
| 4src_isacda_s34 | 8.1045 ± 0.0000 |
| random_s32 | 22.4802 ± 0.0000 |
| random_s33 | 17.8627 ± 0.0000 |
| random_s34 | 3439.6216 ± 0.0000 |
| spectral_simple | 5.3578 ± 0.0000 |

判定: 预训练最好 vs random: 优（4.6613 vs 1159.9882）；vs 简单谱基线: 超过（4.6613 vs 5.3578）

### task (clf, higher better)
| features | balanced_acc (mean±std) |
|---|---|
| 4src_cda_s32 | 0.2598 ± 0.0000 |
| 4src_cda_s33 | 0.2722 ± 0.0000 |
| 4src_cda_s34 | 0.2564 ± 0.0000 |
| 4src_isa_s32 | 0.2860 ± 0.0000 |
| 4src_isa_s33 | 0.2953 ± 0.0000 |
| 4src_isa_s34 | 0.2877 ± 0.0000 |
| 4src_isacda_s32 | 0.2845 ± 0.0000 |
| 4src_isacda_s33 | 0.3019 ± 0.0000 |
| 4src_isacda_s34 | 0.3019 ± 0.0000 |
| random_s32 | 0.2775 ± 0.0000 |
| random_s33 | 0.2778 ± 0.0000 |
| random_s34 | 0.2989 ± 0.0000 |
| spectral_simple | 0.4680 ± 0.0000 |

判定: 预训练最好 vs random: 优（0.3019 vs 0.2847）；vs 简单谱基线: 未超过（0.3019 vs 0.4680）

### valence (reg, lower better)
| features | mse (mean±std) |
|---|---|
| 4src_cda_s32 | 5.6952 ± 0.0000 |
| 4src_cda_s33 | 25.8189 ± 0.0000 |
| 4src_cda_s34 | 22.7794 ± 0.0000 |
| 4src_isa_s32 | 5.0061 ± 0.0000 |
| 4src_isa_s33 | 4.8058 ± 0.0000 |
| 4src_isa_s34 | 4.6481 ± 0.0000 |
| 4src_isacda_s32 | 4.8237 ± 0.0000 |
| 4src_isacda_s33 | 4.6790 ± 0.0000 |
| 4src_isacda_s34 | 1093917419.3552 ± 0.0000 |
| random_s32 | 82.7177 ± 0.0000 |
| random_s33 | 6.0624 ± 0.0000 |
| random_s34 | 5.5074 ± 0.0000 |
| spectral_simple | 7.0962 ± 0.0000 |

判定: 预训练最好 vs random: 优（4.6481 vs 31.4291）；vs 简单谱基线: 超过（4.6481 vs 7.0962）

## 4. 边界（报告必读）

- 预训练使用各源全部被试，LOSO 只保证被试级读出泛化，属结构内评价，不是严格归纳设置。
- 探针为线性/岭读出（弱探针）；负结果不能否定表征价值，正结果也不能归因于预训练除非同时优于 random 与 spectral_simple。
- ISA 仅 DEAP/SAM40/DASPS（对齐证据见 docs/four_source_v4/alignment_evidence/），EEGMAT 仅 CDA。
---

## 5. §3 的修正（2026-10-25 22:30 附，统计缺陷更正）

§3 的自动聚合存在两处缺陷：把 9 个单 run 当作独立组显示（±0.0000），且"预训练最好"取的是 9 个 run 的最优值对比 random 均值（挑选最优，非公平对照）。以下为**按条件跨 seed 取中位数**（对岭回归爆炸值稳健）的修正表，结论以本表为准：

| 目标 | isa | cda | isacda | random | 谱基线 | 平凡 | 判定 |
|---|---|---|---|---|---|---|---|
| sam_valence (MSE↓) | 1034.1 | 38.9 | 577.7 | 884.3 | 4.18 | **2.95** | 全败于常数 |
| sam_arousal (MSE↓) | 6542.7 | 102.1 | 8168.3 | 1397.4 | 8.38 | **5.86** | 全败于常数 |
| valence (MSE↓) | 4.81 | 22.8 | 4.82 | 6.06 | 7.10 | **4.55** | 最好仍劣于常数 |
| arousal (MSE↓) | 4.31 | 18.5 | 4.27 | 18.3 | 5.76 | **4.11** | 最好仍劣于常数 |
| stress (MSE↓) | 4.90 | 38.8 | 8.10 | 22.5 | 5.36 | **4.16** | 最好仍劣于常数 |
| task (bacc↑) | 0.288 | 0.260 | 0.302 | 0.278 | **0.468** | 0.250 | 谱基线碾压全部深度特征 |
| condition (bacc↑) | 0.915 | 0.904 | 0.927 | 0.920 | 0.645 | 0.500 | 与 random 无差异 |

**修正后总结论：没有任何预训练条件在任何目标上同时打穿 random 与谱基线两道对照；全部回归读出劣于常数基线；SAM40 任务上谱基线大幅领先；EEGMAT 条件分类 random 即达 0.92，无归因增益。** 与失败案例摘要（2026-10-22）的负结果模式同构——本实验是其"ISA 白名单合法化 + 单位统一 + CDA 四源均衡"后的干净复现。

## 6. 过程事件记录

- isacda 三种子一致出现 clisa 降 / cda 升：两目标竞争，值得单独立项分析。
- 运行中出现一次双实例竞争（第二个终端以 uv Python 重复启动流水线，状态文件被陈旧写入污染），已终止重复实例并干净重跑探针收尾；`stages.extract` 的 'aborted' 为该事件残留，实际提取 12/12 完成（features/ 全部在盘）。
