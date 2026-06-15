# GreenCast Proposal（v1.4）
## 题目（候选）
**GreenCast: Stable Green Kernels as a Backbone for Exogenous MISO Forecasting**

## 0. 任务目标与论文定位（强制边界）

- **研究方向**：`backbone architecture / 模型架构创新`。
- **禁止边界**：不作为 benchmark-only、not a theory-only。实验与统计服务于主架构主张。
- **投稿目标**：ICLR / ICML / NeurIPS 2027 main track。
- **提交标准**：所有主张可被直接验证、可被攻击、可复现；任何关键门禁未达标则降级或终止。

---

## 1. 一句话 thesis（全文锚点）

在外生 MISO 预测中，将协变量效应建模为**稳定、可约束、可解释的响应核主干**，比仅对融合权重进行再参数化能在时滞漂移与协变量稀疏条件下提供更好的泛化归纳偏置，并保留可复现的解释路径。

### 1.1 核心可投主张（必须逐条闭环）

1. `C1`：稳定 Green 核主干对可见性受限的外生滞后关系具有更好的可归纳偏置与解释一致性。
2. `C2`：当残差分支被明确上限约束后，预测与机制指标（`phase-regret`、`delay`、`support`）会同步提升。
3. `C3`：在自然漂移与注入漂移场景下，GreenCast 的主干贡献保持跨任务一致优势。

---

## 2. 问题定义与机制表达

在时刻 t，目标历史与协变量定义为：

- `y_{1:t}`：目标历史
- `x^-_{1:t}`：历史可见协变量（past-only）
- `x^+_{t+1:t+H}`：known-future 协变量

GreenCast 的预测形式为：

`\hat y_{t+h}=b_h(H_t)+\sum_j \sum_\tau G_{j,h}(\tau\mid q_t)\,u_{j,t+\tau}+r_h(H_t),\quad h\in[1,H]`

- `u_{j}`：协变量创新源项（去趋势与去可预测成分）
- `q_t`：上下文（自相关、缺失率、协变量可见性、Horizon 信息）
- `b_h`：自回归目标分支
- `r_h`：残差修正分支（受控）

### 可执行硬约束

1. **可见性约束**：`G_{j,h}(τ)` 的支持域仅允许合法时间范围。
2. **稳定性约束**：核幅值随 `|τ|` 衰减。
3. **归因约束**：目标分解主干与残差能量比例受显式上限限制。

---

## 3. 核心架构创新

### 3.1 主干公式

主干响应核参数化为：

`G_{j,h}(τ\mid q_t)=M_{j,h}(τ)\cdot \sum_{m=1}^{R} a_{jhm}(q_t)\,\exp(-\rho_{jhm}|τ|)\cos(\omega_{jhm}τ+\phi_{jhm})`

其中：
- `ρ_{jhm}=softplus(ρ̃)+ε`（稳定）
- `M_{j,h}` 为合法支撑掩码（past-only / known-future）
- `a_{jhm}` 为稀疏门控权重

### 3.2 架构模块

1. **Innovation Extractor**：
   - 去趋势/去季节项，输出 `u_j`
   - 附带缺失、可预见性、可靠性元特征
2. **Context Encoder**：
   - 输入谱熵、相关峰位、可见性比例、历史波动性
   - 输出 `q_t`
3. **Stable Green Bank（核心骨干）**：
   - 为每协变量生成 `R` 阶共享字典响应
   - 通过 `mask` 限制时间支撑
4. **Residual Head（受限）**：
   - 小规模 MLP/attention
   - 仅补充非主干可解释以外的交互

---

## 4. 为什么不是 benchmark/theory

- **不是 benchmark 论文**：不以数据榜单为论文目标；基准、消融只是用于验证 `Stable Green Bank` 的归因与泛化价值。
- **不是纯理论论文**：不追求新定理框架。理论成分限定为：
  - 识别性假设与其适用边界
  - 可计算机制指标
  - 与实证结果之间的一一映射。

---

## 5. 相关工作边界与差异（修订后）

### 必答边界

- 与 **TFT / NBEATSx / TimeXer / GCGNet / xCPD**：这些方法偏向选择与关系建模；GreenCast 将响应核本身作为主干。
- 与 **Koopman / Local Projections / 状态空间(S4/Mamba)**：共享“系统响应”语言，但不覆盖我们在 **外生协变量主干 + 约束支撑 + 可解释归因占比** 的组合。
- 与 **ARIMAX / Dynamic Regression**：经典线性基线，作为主线保守对照。

### 差异表述（简化）

`系统建模方法`：解释动态关系

`GreenCast`：在可见性、稳定性和残差占比上硬约束 `response-kernel backbone` 的可训练主干。

---

## 6. 可识别性与可证伪框架（Reviewer-hardening）

### 6.1 识别性假设（实验前固定）

- A1: `u_j` 与目标自回归项在统计上可分解；
- A2: 真正有效协变量响应具有有限支撑，响应平稳且可被参数化。
- A3: 残差分支仅用于高阶交互，不承担主响应建模责任。
- A4: 至少存在一个协变量方向性先验（如传播/生理/物理时滞）。

### 6.2 识别性保护

- 残差占比上限：`Rratio = E||r||_1 / E||ŷ||_1 <= β_max`（先验值固定）
- 主干去除对照：`GreenCast - GreenBank`
- 残差去除对照：`GreenCast - residual`
- Shuffle source / oracle kernel 对照
- 参数预算与输入管线对齐控制，避免归因混淆

### 6.3 核心可证伪条件（未满足则终止主线）

- 若 `Rratio > β_max` 且性能优势主要由残差分支贡献 => **终止主架构主张**。
- 若自然漂移与注入漂移下，主干优势不一致 => **降级为机制说明稿**。
- 若 `kernel recovery` 与 `support recovery` 显著失败 => **降级为失败分析案例**。

---

## 7. 实验系统（端到端）

我们采用 `M0` 到 `M6` 阶段，必须按顺序执行。

### M0. Prior-art sweep（1 周内）
- 搜索范围：2023-2026arXiv + NeurIPS/ICLR/ICML/AAAI。
- 判定规则：若已有顶会主会场已发表/等价同构工作，**转为诊断或附录方法，不主推主张**。

### M1. 真实先导（Pilot）—**先于 M2 实施**
- 数据域：Weather / ECL / PEMS（任一）
- 任务：保持边际相关性，同时改变量滞后/相位结构。
- 输出：
  - rank-instability
  - phase-response regret
  - DM 统计差异（种子平均）
- 通过规则：至少 1 个域显著通过（`p < 0.05`，FDR修正后），否则直接降级。

### M2. 合成与半合成机制检验
- 生成器：ARMA + event + 非线性交互 + noise，支持可控延迟/阻尼/符号。
- 评估变量：`R_{kernel}`, `support`, `delay`
- 目标：检验可恢复能力，排除单纯参数量解释。
- Greenlight：
  1. phase-regret 优势达到门槛
  2. delay MAE 达标（预注册）
  3. illegal support mass 明显更低

### M3. 自然 MISO 实验
- 数据：Weather、ECL、PEMS、Exchange（至少 3 家族）
- 设置：
  - chrono split + purge/embargo ≥ H
  - target rotation（每域 4~8 个）
  - 5 seeds
  - 固定 horizon：24/48/96/192（按域变更）

### M4. 天然漂移与对照
- 选择自然时滞变化片段（而非仅人工插值）
- 与注入漂移并行报告
- 负控与上界对照：`shuffled`, `oracle-kernel`

### M5. 对照与消融（硬性）
- 强基线：ARIMAX、Dynamic Regression、DLinear/N-Linear、TFT、NBEATSx、TimeXer、GCGNet、xCPD。
- 架构基线：Mamba/S4、TCN（受限感受野覆盖到 max lag）、attention（等 receptive field）。
- 为保证可比性，**每个基线与 GreenCast 对齐预算约束**：参数量差异控制在 `±10%`，单步推理延迟差异控制在 `1.5x` 内，输入预处理完全等价。
- 消融套件：
  - G0 target-only
  - G1 static-lag
  - G2 free-conv
  - G3 phase-context only
  - G4 GreenBank only
  - G5 GreenCast full
  - G6 residual-only
  - G7 oracle-kernel

### M6. 统计与复现审计（投稿前）
- 显著性：DM test + paired sign-rank + Wilcoxon + BH-FDR（全表）
- 置信区间：按 bootstrap/重采样给出
- 结果与脚本：统一 `manifest.json`

---

## 8. 训练与实现红线（必须落盘）

### 8.1 训练默认配置（最小集合）

- device 选择必须按顺序执行：

```python
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

pin_memory = (device.type == "cuda")
```

- 统一优化：AdamW，学习率余弦退火
- Early stop：验证集 MSE/MAE 结合
- 每次试验写入：seed / split / config / git hash / 环境依赖

### 8.2 必要文件结构（必须创建）

- `experiments/GreenCast/configs/`：每模型与每数据域独立 yaml
- `experiments/GreenCast/scripts/train_greencast.py`
- `experiments/GreenCast/scripts/eval_greencast.py`
- `experiments/GreenCast/scripts/report_tables.py`
- `experiments/GreenCast/results/<run_id>/manifest.json`
- `experiments/GreenCast/paper/figures/*`
- `experiments/GreenCast/paper/tables/*`

---

## 9. 指标与统计协议（论文主指标）

### 9.1 主要指标

- Forecast quality: MAE, RMSE, MASE/sMAPE
- Mechanism: phase-response regret, delay MAE, support F1, illegal mass
- Attribution: residual ratio, active-source precision/recall
- Cost: 参数量, FLOPs, 延迟, 可扩展性

### 9.2 显著性与纠偏

- 统一显著性水平 `α=0.05`
- 所有多重比较走 BH-FDR (`q<=0.05`)
- 报告效应量与置信区间，不报单点 p 值叙事。

---

## 10. 论文交付计划（投稿就绪）

### 10.1 论文章节

1. Introduction（问题与可证伪主张）
2. Related Work（与 Koopman / LP / SSM / TSF 统一对齐）
3. Problem & Model Formulation
4. GreenCast Architecture (核心机制)
5. Identifiability & limits（假设、边界）
6. Experimental protocol（M0-M6）
7. Results & analysis（机制/消融/负控）
8. Reproducibility & discussion（开源/限制/伦理）

### 10.2 关键图表（投稿版）

- Fig.1: M1 先导图（真实数据下响应漂移失败/恢复）
- Fig.2: 模型架构图
- Fig.3: 合成核恢复与 phase-regret 曲线
- Fig.4: 自然数据主结果表（含统计）
- Fig.5: 响应图（delay / damping / gain）
- Fig.6: 消融与鲁棒性分解图

### 10.3 交付物门禁（任一未满足则不提交）

- 关键 claim 与机制指标不一一匹配
- 统计与基线公平性缺失
- 复现 manifest 不完整
- 残差占比未达上限控制

### 10.4 结果与清单模板（落盘要求）

#### 10.4.1 结果表模板（`paper/tables/RESULTS_main.csv`）

每行字段：

`dataset,domain,horizon,seed,model,backbone,val_mae,val_rmse,test_mae,test_rmse,phase_regret,delay_mae,support_f1,illegal_mass,residual_ratio,dm_vs_tft_p,dm_vs_gcgnet_p,wilcoxon_p,bh_fdr_q,effect_size,run_id`

字段数量为 20。

#### 10.4.2 实验 manifest 模板（`results/<run_id>/manifest.json`）

最少字段：

```
{
  "run_id": "run_YYYYMMDD_xxx",
  "experiment_stage": "M1|M2|M3|M4|M5",
  "dataset": "PEMS|Weather|ECL",
  "horizons": [24,48,96,192],
  "seeds": [2021,2022,2023,2024,2025],
  "beta_max": 0.25,
  "model_config": "configs/greencast_g5.yaml",
  "git_commit": "<sha>",
  "device": "cuda|mps|cpu",
  "pin_memory": true,
  "dataloader_num_workers": 4,
  "stat_test": ["dm","wilcoxon","bh_fdr"],
  "notes": "pre-registered",
  "status": "pass|fail|reject"
}
```

#### 10.4.3 预注册记录模板（`configs/prereg.yaml`）

至少包含：

`beta_max, phase_regret_gate, delay_mae_gate, fdr_q, target_gain_gate, m0_kill_rule, m1_min_effect_size`


---

## 11. 风险控制与退出策略

1. **基线压倒主干**：若强基线系统性优于 GreenCast，收敛为负向发现论文（机制解释边界），不提交主架构。 
2. **残差主导**：若残差分量成为解释主渠道，则主张转为 residual-enhanced forecaster，去除核心主张。 
3. **自然漂移不稳定**：若自然/注入漂移结论方向不同，报告失败条件并收缩结论域。 
4. **复现断层**：脚本、配置、seed 任一缺失则禁止进入 M3。

---

## 12. 最终可投判据（10/10 目标门禁）

### 门禁映射（全部满足则进入投稿）

- Originality: 明确区分已工作并给出“主干差分”机制。
- Rigor: 识别性假设固定 + 反例对照 + 预注册阈值。
- Evidence: M1/M2/M3/M4 全部通过。
- Writing: 章节、图表、局限、复现附录齐全。
- Reproducibility: manifest 与脚本可一键重现。

### 结果判定

- **PASS**：全部门禁通过 → 直接投稿。
- **REPAIR**：部分门禁失效 → 执行局部实验补齐。
- **FAIL**：核心门禁失效 → 降级为诊断型方法稿。

附注（实操上界）：

- 结果总行数要求为：`datasets × domains × models × horizons × seeds`（当前每个 `run` 以 1 个固定 `model/backbone` 计，若开启多模型对比则乘以模型数）。

### 13.1 论文投稿交付物总清单

- `paper/paper.tex` 与 `paper/refs.bib`
- `paper/tables/RESULTS_main.csv`
- `paper/tables/Table1.tex` / `Table2.tex`
- `paper/figures/`：Fig.1 ~ Fig.6（png/pdf + caption.json）
- `results/<run_id>/manifest.json`（每 run 一份）
- `results/<run_id>/train.log`
- `results/<run_id>/metrics_*.json`（可复现实验指标）
- `paper/reproducibility_checklist.md`

论文在 `REVIEW.md` 维度评分中只在以下条件均满足且 evidence 已在 `results/*/manifest.json + paper/tables/RESULTS_main.csv` 中落地时，才可切换到投递状态。

### 12.1 证据映射（Claim Matrix）

| 主张 | 需验收证据 | 必要实验 | 最低达标条件 |
|---|---|---|---|
| C1 | `phase_regret` 下降 + `delay_mae` 降低 | M2、M3、M4 | 效果量 `>= m1_min_effect_size`，至少 3/4 域通过 |
| C2 | `support_f1` 上升 + `illegal_mass` 减少 + `residual_ratio <= beta_max` | M2、M3、M5 | FDR 校正后 `q <= fdr_q`，并且 `residual_ratio` 不超过上限 |
| C3 | `test_mae/test_rmse` 与机制指标在漂移场景中同向改善 | M4、M5 | 自然漂移与注入漂移方向一致；DM 与 Wilcoxon 两者显著（或同等幅度支持负向结果） |

### 12.2 复现实验落盘命名与覆盖规则

每个 run 必须建立文件夹 `results/<run_id>/`，并包含：

- `manifest.json`
- `config_snapshot.yaml`
- `train.log`
- `metrics_<seed>.json`（可选）或 `metrics.csv`
- `predictions.parquet`（可选）
- `failures.json`（失败运行可选但必须存在空数组或错误记录）

`RESULTS_main.csv` 的每一行必须映射到一个 `run_id`，且 `run_id` 在 `manifest.json` 与结果汇总中保持全局唯一。

### 12.3 投递前停止条件（10/10 门禁）

- 任一主张未完成映射证据：暂停投稿，退回 M2 或 M5 补实验；
- 任一统计检验列为 `n/a`：暂停投稿，补齐全量统计并更新；
- 任一 manifest 漏字段（`device/pin_memory/model/gate settings`）：暂停投稿，重跑失败 run；
- 一旦所有上界门禁满足，才允许进入 `FINAL` 标记并执行版式定稿。

---

## 13. 端到端投稿执行清单（最终交付）

### 13.1 论文与复现实验落盘

- `experiments/GreenCast/paper/paper.tex`：主文稿源文件
- `experiments/GreenCast/paper/figures/`：主图（Fig.1~Fig.6）
- `experiments/GreenCast/paper/tables/RESULTS_main.csv`：主结果表（实验汇总）
- `experiments/GreenCast/paper/tables/Table1.tex`：主表格脚本输出
- `experiments/GreenCast/scripts/train_greencast.py`：训练入口（含设备红线）
- `experiments/GreenCast/scripts/eval_greencast.py`：评估入口（含 DM/Wilcoxon/BH-FDR）
- `experiments/GreenCast/scripts/report_tables.py`：结果聚合入口
- `experiments/GreenCast/results/<run_id>/manifest.json`：每次实验的主证据链
- `experiments/GreenCast/configs/*.yaml`：预注册与模型/数据域配置

### 13.2 论文交付执行命令（附录公开）

1. `python scripts/train_greencast.py --config configs/greencast_g5.yaml --dataset Weather --domain finance --stage M1 --run_id gc_w01`
2. `python scripts/eval_greencast.py --run-id gc_w01 --manifest results/gc_w01/manifest.json --output paper/tables/RESULTS_main.csv`
3. `python scripts/report_tables.py --results-root results --out-csv paper/tables/RESULTS_main.csv`
4. `git rev-parse HEAD > paper/build/git.txt`（记录版本）
5. 将 10.4 模板字段与 10.4.1/10.4.2 输出对齐到附录 `A.2`
