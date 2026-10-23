# DASPS 原始数据对齐证据报告

日期：2026-10-23（第二轮更新：博士论文取证 + Table 3.2 数值对账，类别顺序已证实固定，DASPS 升级 weak 档）
执行：本机取证（原始 .mat 逐文件解析 + SHA256 比对）+ 官方论文协议核对
配套机器可读文件（同目录）：
- `DASPS_raw_verification_2026-10-23.json`（23 人哈希/形状/成对性逐人记录）
- `DASPS_trial_alignment_2026-10-23.json`（结论层证据）
- `DASPS_subjects_trials_2026-10-23.json`（每人 12 trial 的情节/阶段/SAM 逐 trial 明细）

---

## 1. 原始数据已拿到并验证

原始 23 个 `SXX.mat` 位于 `E:\mdJPT_nips2025-main\DASPS\`（E 盘已恢复在线）。
**23/23 个文件的 SHA256 与 `processed_clean_v4/DASPS/<sub>/source_files.json` 的溯源记录完全一致**——确认这就是 v4 清洗链当时使用的同一份原始数据，本轮证据对 v4 有效。

原始文件 schema（v7.3 HDF5）：

| 字段 | 形状 | 含义 |
|---|---|---|
| `data` | (12, 1920, 14) | 12 trial × 1920 采样点（128 Hz × 15 s）× EPOC+ 14 通道 |
| `labels` | (2, 12) | [valence; arousal] SAM 评分（1–9），每 trial 一组 |
| `hamilton` | (2, 1) | HAM-A 前/后测（被试级） |
| `situation` | (6, 1) uint16 | 字符码点，每被试**一个**法语词 |

## 2. 关键结论一：12 trial = 6 情境情节 × 2 阶段（已证明）

**数据内证据（决定性）**：全部 23 个被试的 `labels` 均严格成对——`labels[:,2k] == labels[:,2k+1]` 对全部 k=0..5 成立（valence 与 arousal 同时相等）。12 列恰好 6 种取值各出现连续两次。按协议 SAM 是每个情境刺激结束后评一次分，成对相等说明相邻两个 trial 属于同一情境情节。

**协议证据**：
- 原始论文（arXiv:1901.02942v2，Baghdadi et al.）：每个 trial 30 秒刺激 = "recitation by the psychotherapist for the first 15 sec and Recall by the subject for the last 15 sec"，共 6 个情境，"This trial is repeated until the sixth situation"。
- PLOS ONE 2022（PMC8932601，独立复述）："Both the recital and the imagination phase are composed of 15 seconds each"，"the sequence of anxiety provocation and SAM rating is executed six times"。

因此：**奇数 trial = 朗读阶段（0–15 s，治疗师口述），偶数 trial = 回忆阶段（15–30 s，被试自行想象回忆）**；trial (2k−1, 2k) 为同一情境情节的两个阶段，共享该情节的 SAM 评分。

## 3. 关键结论二：`situation` 字段是被试级属性，不是 trial 元数据

23 个被试各解码出一个词（每人仅一个）：proche×9、accident×7、maltraitance×2、finance×2、famille×2、deadline×1。六个词与论文从人群调查选出的 6 个官方情境类别一一对应（proche≈丧失挚亲、famille≈家庭问题、finance≈经济问题、deadline、accident≈目睹死亡事故、maltraitance≈虐待经历）。

按论文协议"each participant identifying the most anxious situation"，该词是**被试自认的最焦虑情境**。v4 曾把它复制进每个被试全部 trial 的 `labels.situation`——应更正理解为被试级属性。它不提供任何 trial 级类别信息。

## 4. 关键结论三（第二轮）：类别呈现顺序跨被试固定 —— 已证实

来源：**Asma Baghdadi 博士论文**（2021，[HAL tel-04163090](https://theses.hal.science/tel-04163090)，本地副本 `Baghdadi_thesis_2021.pdf`）第 3 章，DASPS 协议的完整第一手描述：

1. **固定协议**：论文 p.47 原文 "**The fixed protocol** is as depicted in Figure 3.4"——知情同意 → HAM-A 前测 → 6×（朗读 15 s + 回忆 15 s + SAM 评分）→ HAM-A 后测。全文无任何随机化表述（网络搜索摘要曾声称"顺序随机化"，在两篇论文与博士论文原文中均无出处，不采信）。
2. **Table 3.2 按位置给出跨被试统计**：论文对 "Situation 1..6" 六个位置分别给出全部被试的均值±标准差。**若顺序跨被试随机，该表在数学上无意义**（各位置均值会收敛到总均值）；其存在本身就是顺序固定的陈述。
3. **数值对账（决定性）**：用本地 23 个原始文件的 `labels` 按情节位置 1..6 计算跨被试统计，与 Table 3.2 逐位比较：
   - **12/12 个标准差完全一致**（ddof=1，两位小数全等，如 S1 valence 1.687 vs 论文 1.68）；
   - **12/12 个均值差 ≤ 0.01**（论文 §3.2.6 声明发表前剔除了评分异常数据，可解释残差）；
   - 平均 CV：本地 0.574/0.413 vs 论文 0.58/0.42。
   - 明细见 `DASPS_table3_2_verification_2026-10-23.json`。

结论：**trial 2k−1 = 情境 k（跨被试同一类别）朗读阶段，trial 2k = 情境 k 回忆阶段，23 人同构**。仍未知的两点降级为：①位置 1..6 与类别英文名的对应表未发表（只影响命名，不影响 ISA 合法性）；②口述脚本的逐字标准化程度未说明（但波形级差异本来就存在，见第 5 节封顶）。

## 5. 对 ISA 白名单的结论（第二轮修订）

对齐能力分级（最终）：

| 层级 | 内容 | 状态 |
|---|---|---|
| ✅ | trial 序号 k 跨被试 = 会话内同一序位 | 本轮 |
| ✅ | 奇/偶 trial = 朗读/回忆阶段 | 本轮 |
| ✅ | (2k−1, 2k) 同属一个情境情节 | SAM 成对 |
| ✅ | 情境类别顺序跨被试固定 | 博士论文 fixed protocol + Table 3.2 对账 |
| ❌ | 位置→类别名称对应表、脚本逐字标准化 | 未发表（不影响 ISA） |
| 封顶 | 现场口述刺激，波形跨被试永不同 | **weak 为上限，exact 不可达** |

**DASPS 升级为 `weak` 档，与 SAM40 同级**（SAM40 的任务内容同样非逐位相同，同为具名条件级对齐）。

- **主四源联合预训练：`clisa_datasets` 升级为 `[DEAP, SAM40, DASPS]`**。数据管线**无需重切**：ISA 配对在窗口级（同窗位 = 同 trial + 同 offset，布局 23/23 一致已核验），修改方案 §3.1 的 4 组×3 trial 分组不影响窗口级对齐；`DASPS_v4.yaml` 的 `n_vids=4` 不变，总窗数仍 38/步，显存预算不变。修改方案的其余条目不受影响（见该文档 §9 增补）。
- EEGMAT 维持排除（rest/arithmetic 无时间锁定，与配对语义不兼容）。
- 报告注记：DASPS ISA 配对语义 = 同序位情境类别 + 同阶段 + 情节内同时刻（weak）；刺激为现场口述；论文 p.47 记录刺激阶段被试**闭眼**、尽量减少动作和言语（跨数据集对比时与 DEAP 睁眼观看视频的形态差异需知悉）。
- `exact` 档永远不可达——这是刺激性质决定的，不是证据缺口。官方 `Documents and Code.zip`（IEEE DataPort，需账号）如日后取得，可进一步核对分段脚本与位置→类别命名，属锦上添花。

## 6. 证据来源

- 本机原始文件：`E:\mdJPT_nips2025-main\DASPS\S01.mat … S23.mat`（23/23 SHA256 与 v4 溯源一致）
- DASPS 原始论文：[arXiv:1901.02942v2](https://arxiv.org/html/1901.02942v2)
- IEEE DataPort：[DASPS Database](https://ieee-dataport.org/open-access/dasps-database)（128 Hz EPOC+；评论区证实分段脚本在 Documents and Code.zip，且此前无人答复过 trial 维度顺序问题）
- PLOS ONE 2022：[PMC8932601](https://pmc.ncbi.nlm.nih.gov/articles/PMC8932601/)
- **Asma Baghdadi 博士论文 2021**：[HAL tel-04163090](https://theses.hal.science/tel-04163090)（本地副本 `Baghdadi_thesis_2021.pdf`，第 3 章：fixed protocol、Table 3.2、HAM-A/SAM 细节）
