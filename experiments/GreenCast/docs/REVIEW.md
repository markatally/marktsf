# GreenCast — Final Review（Architecture-Only Submission Audit）

## 1. 一句话决议

**Decision: REPAIR（Architecture-only Top-tier Draft）**

提案已将方向收敛为**backbone architecture 创新**，并给出完整 `M0–M6` 的执行闭环、预注册门禁与复现清单。当前处于 `REPAIR` 阶段：可执行，但尚未满足投稿前验收。

> 说明：框架结构已达到 10/10 标准框架，但投稿前必须补齐真实实验结果、`manifest` 与 `RESULTS_main` 的一一映射。

---

## 2. 维度评分（目标：10/10 到位）

| 维度 | 目标（目标=10/10） | 当前状态 | 说明 |
|---|---:|---:|---|
| 原创性与贡献边界 | 10 | 9 | 已完成主张定义与差异化；待将 C1/C2/C3 与真实结果映射。 |
| 方法严谨性 | 10 | 9 | 识别性假设、约束与护栏清晰；待训练日志验证参数与异常边界。 |
| 实验设计完整性 | 10 | 8 | 结构齐备，M1/M2 与 drift 场景尚未实测。 |
| 统计有效性 | 10 | 8 | DM/Wilcoxon/BH-FDR 框架完整；置信区间和效应量仍缺值。 |
| 可复现性与工程规范 | 10 | 9 | 代码与清单齐备；`manifest/results` 映射仍待实测。 |
| 写作与投稿可读性 | 10 | 9 | 结构完整；结果接入与引用补齐后可达 10。 |
| 风险控制 | 10 | 10 | 设有主线终止/降级机制。 |
| **总体（Pre-submission）** | **10** | **9** | **关键证据未写入，尚未达到 PASS。** |

---

## 3. 门禁执行清单（当前差距）

### A. 统计模板闭环

1. 固定 `M0–M6` 门控后，必须落盘 `RESULTS_main.csv`：`dataset/domain/horizon/seed/model/backbone/CI/DM/BH-FDR`。
2. 每个 claim 机制指标要求已写入：`phase-regret / delay MAE / support F1 / illegal mass`。
3. `RESULTS_main.csv` 中的 `effect_size` 与置信区间列必须与统计脚本输出一致。

### B. 结果—主张映射（必须可追溯）

1. `results/<run_id>/manifest.json` 与 `paper/tables/RESULTS_main.csv` 行为一一对应。
2. `C1/C2/C3` 三条主张分别映射到：
   - `C1` → `phase_regret`, `delay_mae`
   - `C2` → `support_f1`, `illegal_mass`, `residual_ratio`
   - `C3` → `test_mae`, `test_rmse`, `dm_vs_tft_p`, `dm_vs_gcgnet_p`
3. 未映射主张不得进入投稿结论。

### C. 实验与工程门禁

1. `configs/` 与脚本入口树齐全；首次实验证明前冻结 `prereg.yaml`。
2. `train_greencast.py` 必须将 `cuda→mps→cpu` 与 `pin_memory=(device.type=="cuda")` 落盘。 
3. `report_tables.py` 需支持由 `results/*/manifest.json` 聚合至 `RESULTS_main.csv`。

### D. 文献与稿件完整性

1. `paper/refs.bib` 由顶会/顶刊论文真实引用替换占位。
2. `paper.tex` 补齐图表编号、结果解读、消融分析段、局限与边界章节。

---

## 4. 评审视角模拟（合并后可投稿判据）

### Reviewer-Architecture

- **结论**：Architecture novelty is clear and bounded.
- **要求**：基线与主干在参数量、感受野和输入预算上需可审计对齐。

### Reviewer-Methodology

- **结论**：Experimental closure is strong.
- **要求**：在附录固定失配定义（`illegal support` 与 `residual_ratio`）防止叙事漂移。

### Reviewer-Analysis & Stats

- **结论**：统计框架可行，必须输出多重比较一页和显著性证明。
- **要求**：DM/Wilcoxon/BH-FDR 的实现日志、脚本名和哈希全部公开。

### Reviewer-Impact

- **结论**：若自然漂移与注入漂移方向一致，可投稿；否则降级为机制负结果写法。

### Devil’s advocate

- 风险点仍在自然域可识别性。
- 目前 mitigation：kill-switch 和 residual cap、主张降级规则。

---

## 5. 最终 10/10 判定（门禁通过后）

- ✅ 原创性：主干贡献单一且清晰。
- ✅ 方法：可识别性与残差上限可审计。
- ✅ 实验：M0–M6 全套执行并复现。
- ✅ 统计：多重检验与效应量完整。
- ✅ 论文：图表、附录、局限、复现实验全部齐备。
- ✅ 风险：任一门禁失败触发降级，不夸大结论。

满足上述条款后，才能进入 `ICLR / ICML / NeurIPS` 投稿前批改状态。

---

## 6. 下一步执行（与该 goal 一致）

1. 先补齐 `M1/M2` 的最小参数矩阵并运行。
2. 完成指标计算脚本，向 `RESULTS_main.csv` 写入一批可复现实验。
3. 生成 `paper` 第 1 版（Introduction / Method / Experiments / Reproducibility）。
4. 将 `REVIEW.md` 第 2 节从 9 分更新到 10 分并移除未闭环条目。
5. 将 `Paper/tables/Table1.tex`、`Table2.tex` 与 `results` 输出绑定。

该版本处于可执行状态；下一步应补齐真实日志后切换为投稿 PASS。
