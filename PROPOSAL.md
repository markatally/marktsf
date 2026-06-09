# SPECTRE — 提案与实验框架设计

> **一句话定位**：把"变量当 token"（iTransformer）从**对称多变量预测**改造为**非对称协变量预测（MISO）**，
> 并在其中引入两项新机制——**频带分辨的「驱动→目标」耦合** 与 **协变量预测作为辅助正则任务（Version A）**——
> 在通用 / 金融收益率 / 零售销量三类场景上统一刷 SOTA。
>
> 本文是后续所有 coding 的"单一事实来源（single source of truth）"。代码围绕第 6–8 节的接口契约、实验矩阵、仓库结构展开。

---

## 0. TL;DR（给实现者）

- **范式**：MISO（multiple-input, single-output）/ 带外生协变量的单目标预测。区分两类协变量：`past-only`（仅历史可见，如订单流代理）与 `known-future`（未来已知，如促销日历、计划采购）。
- **方法（Version A）**：共享 encoder 产出目标 latent `z_y` → **主头只从 `z_y` 预测目标**；**辅助头预测协变量未来，仅作正则**（绝不前喂给主头，规避数据处理不等式瓶颈）。
- **两项新机制**：① 频带分辨的非对称耦合（每个频带学一套"哪些协变量在该尺度上驱动目标"，且随机制漂移）；② 辅助多任务正则。
- **退化关系（= 天然消融）**：关掉全部新机制 ≈ iTransformer/MISO；硬级联（把 `x_hat` 前喂）= 被支配的对照；逐步加机制 = A0→A5。
- **三战场**：S0 通用（ETT/Weather/Electricity/…，证明普适 SOTA）；S1 金融（Crypto/AU888 → 未来 N 期收益率，IC/RankIC/方向命中/回测）；S2 零售（M5/Favorita，待采集 → 销量，WMAPE/RMSSE/分位损失）。

---

## 1. 问题形式化

在时刻 `t`，给定回看长度 `L`、预测步长 `H`，定义一个样本：

| 张量 | 形状 | 含义 |
|---|---|---|
| `x_target` | `[L]` | 目标变量历史 |
| `x_past_cov` | `[L, C_p]` | **past-only** 协变量历史（如 OFI 代理、成交量、波动率） |
| `x_known_past` | `[L, C_f]` | **known-future** 协变量的历史段（如日历、促销） |
| `x_known_fut` | `[H, C_f]` | **known-future** 协变量的未来段（**外生新信息**，可合法前喂） |
| `static` | `[C_s]` | 静态协变量（资产 id / 门店 / 品类，可选） |
| `y` | `[H]` | 预测目标（label） |

预测目标 `\hat y_{1:H} = F(x_target, x_past_cov, x_known_*, static)`。

**金融特化**：目标定义为前向 N 期对数收益 `r_{t+k} = log(p_{t+k}) - log(p_{t+k-1})`（或累计 `log(p_{t+H}/p_t)`），由价格在数据层即时计算，避免把"预测价格"和"预测收益"混淆。

**核心不对称性**：目标 `y` 与协变量 `X` 角色不同——`X` 是预测器、不是预测产物。这是与对称 MTSF（iTransformer 预测全部通道）的本质区别，也是本工作的立论支点。

---

## 2. 立论与贡献（motivation）

1. **范式纠偏**：主流通道策略文献（Channel Survey、iTransformer、Crossformer、Channel-Masks、Bias 实证）几乎都在**对称 MTSF**（如 ETT 预测全列）上 benchmark；而真实部署（量化、供应链）几乎都是**非对称 MISO**。我们给出"非对称协变量预测"这条被忽视的坐标轴，并证明对称 benchmark 上的最优通道策略未必迁移到非对称场景。
2. **频带分辨的耦合**：协变量对目标的驱动是**频率/时间尺度相关**的（高频：订单流→秒级收益；低频：库存/采购→季节性销量），且**随机制漂移**（危机期低频耦合骤增）。现有方法要么时域学一张静态全局通道图（塌缩所有频率），要么频域逐通道滤波（丢弃通道耦合）。我们在 `(通道 × 频带)` 单元上建模**单向、漂移感知**的耦合。
3. **辅助任务式融合（Version A）**：理论上证明"先预测协变量再前喂"的硬级联被端到端模型支配（数据处理不等式，见 §5）；我们将协变量预测降为**辅助正则任务**，在不引入信息瓶颈的前提下获得"表征必须丰富到能预测驱动变量"的归纳偏置。

> 贡献可证伪、有退化关系、有合成实验可先验证现象（§7.6），符合"提出问题 → 机制验证 → benchmark SOTA"的闭环。

---

## 3. 方法：SPECTRE（Version A）

```
                 ┌─────────────────────────── Shared Encoder Φ ──────────────────────────┐
 x_target ─┐     │  RevIN ─► 变量级 token 嵌入(asymmetric) ─► 频带分解(B bands)            │
 x_past_cov├────►│        ─► 每频带 [驱动→目标] 交叉注意力(漂移感知) ─► 跨频带聚合 ─► z_y    │
 x_known_* ┘     └───────────────────────────────┬───────────────────────────┬───────────┘
                                                 │ z_y (目标 latent)          │ 每协变量 latent
                                                 ▼                            ▼
                              主头 g: [z_y ⊕ emb(x_known_fut)] ─► ŷ[H]   辅助头 a ─► x̂[H, C_p]  (仅正则)
```

### 3.1 归一化
逐通道 RevIN（可逆实例归一化），处理非平稳的边缘分布漂移；统计量在反归一化时还原。

### 3.2 非对称变量级 token 嵌入
沿用 iTransformer "变量当 token"：每个通道的 `L` 长历史 → 一个 token `e_c ∈ R^d`。但**角色非对称**：目标 token `e_y` 作为后续注意力的 **query**，协变量 token 作为 **key/value**。

### 3.3 频带分解（novel-1）
对每个通道历史做 `B` 个频带的分解，得到 `e_{c,b} ∈ R^d`：
- 默认实现：rFFT → 按频率分桶为 `B` 段（low→high）→ 每段 iFFT 回时域得带限分量 → 嵌入；或直接在频域对每段做线性嵌入（FITS 风格）。
- 可选实现：可学习带通滤波器组（参数化截止频率），便于端到端优化。
- `B` 为超参（默认 4：trend / low / mid / high）。`B=1` 退化为无频带分辨。

### 3.4 频带内「驱动→目标」交叉注意力（novel-1 续）
对每个频带 `b`：
```
z_y^(b) = Attention(query = e_{y,b}, keys/values = {e_{c,b} : c ∈ covariates})
```
即每个频带独立学习"哪些协变量在该时间尺度上对目标有用"。**单向**（covariates→target），不强行对称重建协变量。

**漂移感知（novel-1 再续）**：注意力 logits 由当前窗口的频带统计量（如各带功率、带内波动）经小型 hypernet 调制：
`logit_{b,c} += MLP_b(bandpower_window)`，使耦合图随机制变化（同一对通道在不同窗口耦合强度不同）。

跨频带聚合：`z_y = Σ_b α_b · z_y^(b)`，`α_b` 为可学习权重或对 `{z_y^(b)}` 的注意力。

### 3.5 双头（Version A 核心）
- **主头** `g`：`ŷ = g([z_y ⊕ emb(x_known_fut)])`。
  `x_known_fut` 是**外生新信息**（未来已知协变量），可合法注入；`z_y` 是目标 latent。
  **主头不接收 `x̂`**（辅助头输出），这是 Version A 与硬级联的根本区别。
- **辅助头** `a`：`x̂ = a({per-covariate latent})`，预测 past-only 协变量的未来 `[H, C_p]`。**只进损失、不进主头**。

### 3.6 损失
```
L = L_tgt(ŷ, y)  +  λ_aux · L_aux(x̂, x_future_cov)  +  λ_freq · L_freq(ŷ, y)
```
- `L_tgt`：通用/零售用 MSE（或 Huber / 分位损失）；金融可加 IC/秩相关感知项（§6.4）。
- `L_aux`：协变量预测 MSE（正则），`λ_aux` 退火（前期大、后期衰减），防止压过主任务。
- `L_freq`：FreDF 风格频域损失（对 `ŷ`、`y` 做 rFFT，复数幅度+相位对齐），`λ_freq` 可为 0（消融）。

### 3.7 退化关系（= 消融配置）
| 配置 | freq 模块 | 漂移感知 | aux 头 | known-fut 注入 | x̂ 前喂主头 | 等价于 |
|---|---|---|---|---|---|---|
| **A0** | ✗(B=1) | ✗ | ✗ | ✗ | ✗ | iTransformer/MISO 下界 |
| **A1** | ✗ | ✗ | ✓ | ✗ | **✓** | 硬级联（被支配对照） |
| **A2** | ✗ | ✗ | ✓ | ✗ | ✗ | Version A 核心 |
| **A3** | ✓ | ✗ | ✓ | ✗ | ✗ | + 频带耦合 |
| **A4** | ✓ | ✓ | ✓ | ✗ | ✗ | + 漂移感知 |
| **A5 = SPECTRE** | ✓ | ✓ | ✓ | ✓ | ✗ | 完整方法 |

---

## 4. 为什么一个方法吃下两个场景

| 维度 | S1 金融（收益率） | S2 零售（销量） |
|---|---|---|
| 目标 | 前向 N 期收益 | 未来销量 |
| past-only 协变量 | 成交量、波动率(高-低)、OFI 代理、动量、持仓量(AU888) | 历史订单、历史库存 |
| known-future 协变量 | （几乎无）→ 退化为纯历史驱动 | 计划采购单、促销/节假日日历、价格计划 |
| 主导耦合频带 | 高频 / 短 horizon，机制漂移剧烈 | 低频 / 季节性，缓慢漂移 |
| `x_known_fut` 注入 | 关闭或弱 | 强 |
| `λ_aux` 行为（预期） | 协变量难预测→辅助任务损失高→自适应小贡献 | 协变量较可预测→辅助正则收益大 |

**同一架构、配置驱动**：通过 `covariate_spec`（哪些列 past-only / known-future / static）与超参，自动适配两类 regime。"同一机制在两 regime 涌现不同行为"本身是论文卖点。

---

## 5. 理论支撑：为什么是辅助任务而非硬级联

设历史 `H = (x_target, x_past_cov, x_known_past)`。硬级联：`x̂ = f(H)`，`ŷ = g(x̂, H)`。
因 `x̂` 是 `H` 的确定函数，`g(x̂,H)` 仍是 `H` 的函数，故 `I(y; x̂) ≤ I(y; H)`：

> **先预测协变量再前喂，不向目标注入任何 `H` 之外的信息，反而插入"瓶颈+噪声"。在理想极限下硬级联 ≤ 端到端。** 唯一例外是注入 `H` 之外的新信息——即 `x_known_fut`（未来已知协变量），这正是我们唯一前喂的量。

Version A 让 `y` 直接由 `z_y` 预测（不过 `x̂`），把协变量预测降为**塑形表征的正则**：拿到归纳偏置，不付信息瓶颈。§7.6 合成实验给出经验证据。

---

## 6. 实验框架

### 6.1 三战场与数据映射（基于 `input/` 现状）

**S0 通用（reviewer 必需，证明普适 SOTA）**
- 数据：`ETT{h1,h2,m1,m2}`、`Weather`、`Electricity`、`Traffic`、`Exchange`、`Solar`。
- MISO 化：指定单目标列（ETT→`OT`；其余→约定末列或数据集既定目标），其余列为 past-only 协变量；日历特征作 known-future。
- 同时报告**对称全列**结果与 iTransformer 公平对齐。
- horizon `H ∈ {96,192,336,720}`，`L ∈ {96,336,512}`。指标 MSE / MAE。

**S1 金融（收益率预测）**
- 数据：`Crypto/*`（1m OHLCV，14 资产）、`Finance/AU888_5m`（含 `open_interest`）、`Finance/CSI300_60m`；可扩展 `CSI500/SP500/NASDAQ`。
- 目标：前向 N 期对数收益（数据层计算，`N` 对应 `H`，如 1m 数据 `H∈{1,5,15,60}`）。
- 协变量（**OHLCV 派生，透明声明为代理**）：对数收益、log(high/low) 已实现波动、成交量及其 z-score、Amihud 非流动性、动量、（AU888）持仓量变化。**注意：仓库无真实 LOB，OFI 为 bar 级代理**；如需强微观结构主张，行动项见 §9 引入 FI-2010 / LOBSTER。
- 指标：IC、RankIC、方向命中率(DA)、收益 MSE；附简单多空回测 Sharpe/最大回撤（可选，作落地说明非主指标）。
- 切分：**严格按时间顺序** + embargo/purge（防泄漏，见 6.5）。

**S2 零售（销量预测）**
- **数据缺口**：`input/` 无库存/采购/销量数据集。**行动项**：采集 M5 (Walmart) 与 Favorita（含价格、促销、SNAP/节假日 = 天然 known-future 协变量）。
- 临时替身：`NN5`（ATM 取现，需求型）、`Wiki/Wike2000`（网页流量）+ 派生日历 known-future 协变量，用于先跑通 S2 管线。
- 指标：WMAPE、RMSSE（M5 官方）、MASE、分位损失（P50/P90）。

### 6.2 统一样本契约（所有代码围绕它）
见 §1 表。批处理后加前导 `B` 维。`Dataset.__getitem__ → WindowSample`；`collate → batch dict`；`Model.forward(batch) → {y_hat:[B,H], x_hat:[B,H,C_p], aux:{...}}`。
**任何数据集 adapter 的唯一职责**：把原始文件映射到该契约（指定 target / past-only / known-future / static）。

### 6.3 Baselines（必须实现或移植）
- 通道/骨干类：**iTransformer**、PatchTST、DLinear、RLinear、TiDE、FITS/FreTS、Crossformer、（可选 TimeMixer/Autoformer/FEDformer）。
- **协变量感知类（与我们同 regime，必比）**：TFT、TiDE(with covariates)、（可选 TimeXer / NBEATSx）。
- 朴素统计：Naive/Seasonal-Naive、线性回归（金融额外加：历史均值、AR(1)）。

### 6.4 指标实现
- 通用：`MSE`、`MAE`。
- 金融：`IC`(Pearson)、`RankIC`(Spearman)、`DA`(sign match)、`ICIR`、可选回测 `Sharpe`、`MDD`。
- 零售：`WMAPE`、`RMSSE`、`MASE`、`QuantileLoss`。
- 金融可选损失项 `L_ic = 1 - corr(ŷ, y)`（batch 内），与 MSE 加权。

### 6.5 协议（防泄漏 / 复现）
- 切分：S0 用各数据集标准比例（ETT 12/4/4 月）；S1/S2 **chronological**，train→val→test 时间不重叠，窗口跨界处 **purge + embargo**（embargo ≥ H）。
- 归一化：仅用 train 段统计；RevIN 实例级在 forward 内。
- 随机种子 `{2021,2022,2023}`，报告 `mean ± std`。
- 统一 early stopping（val 指标）、统一 optimizer（AdamW）、统一 lr 调度，记录全部配置（hash 入 run 目录）。

### 6.6 消融与机制验证矩阵
- **主消融**：A0→A5（§3.7）逐项加机制，三战场各跑。
- **支配性验证**：A1(硬级联) vs A2(辅助) ——证明硬级联被支配。
- **超参敏感度**：`λ_aux`、`λ_freq`、频带数 `B`、`L`。
- **§7.6 合成机制实验**：生成"频率相关 + 机制切换"耦合的数据，且分两档协变量可预测性（可预测 / 纯噪声）：
  - 验证 1：协变量不可预测时 A1 崩、A2 稳（支撑 §5）。
  - 验证 2：A3 能恢复真实频带耦合结构（注意力权重 vs 真值）。
  - 验证 3：漂移切换时 A4 跟随耦合演化优于 A2/A3。

---

## 7. 代码架构（后续 coding 蓝图）

> 技术栈：Python 3.12 + PyTorch。配置驱动（YAML + dataclass），强类型接口，many small files。

### 7.1 仓库结构
```
spectre/
  __init__.py
  configs/                  # YAML 实验配置（每个 run 一份，可继承 base）
    base.yaml
    s0_ett.yaml  s1_crypto.yaml  s2_m5.yaml  ablation_*.yaml
  data/
    contract.py             # WindowSample dataclass + 张量形状校验
    registry.py             # name -> adapter 注册表
    windowing.py            # 滑窗、chronological split、purge/embargo、collate
    normalization.py        # train-only 统计、RevIN 辅助
    adapters/
      base.py               # DatasetAdapter 接口
      ett.py  crypto.py  au888.py  m5.py  nn5.py  ...
    features/
      finance.py            # 收益/波动/Amihud/OFI 代理/持仓量
      calendar.py           # 日历 known-future 特征
  models/
    base.py                 # ForecastModel 接口 forward(batch)->outputs
    spectre/
      model.py              # 组装：encoder + 双头
      encoder.py            # 非对称变量 token 嵌入
      freq_coupling.py      # 频带分解 + 频带内交叉注意力 + 漂移感知
      heads.py              # 主头 / 辅助头
    baselines/
      itransformer.py patchtst.py dlinear.py rlinear.py tide.py
      fits.py frets.py crossformer.py tft.py naive.py
  losses/
    mse.py huber.py quantile.py freq.py ic.py   # 可组合 LossSpec
  metrics/
    general.py finance.py retail.py             # 统一 Metric 接口
  engine/
    trainer.py evaluator.py callbacks.py earlystop.py
    seed.py registry.py                         # 通用注册/工厂
  experiments/
    run.py                  # 入口：python -m spectre.experiments.run --config ...
    ablation.py sweep.py    # 批量 A0–A5 / 超参扫描
  synthetic/
    generate.py             # §7.6 频率相关+机制切换合成数据
  utils/
    logging.py io.py config.py
tests/
  test_contract.py test_windowing_noleak.py test_shapes.py
  test_revin_invertible.py test_metrics.py test_synthetic_dominance.py
```

### 7.2 关键接口契约
```python
# data/contract.py
@dataclass(frozen=True)
class WindowSample:
    x_target:    Tensor  # [L]
    x_past_cov:  Tensor  # [L, C_p]
    x_known_past:Tensor  # [L, C_f]
    x_known_fut: Tensor  # [H, C_f]
    static:      Tensor  # [C_s]
    y:           Tensor  # [H]
    meta:        dict

# models/base.py
class ForecastModel(Protocol):
    def forward(self, batch: dict) -> dict:  # -> {"y_hat":[B,H], "x_hat":[B,H,C_p]|None}
        ...

# 不可变原则：adapter/transform 返回新对象，绝不原地修改原始 DataFrame/张量
```
配置项：`dataset`、`target_spec`、`covariate_spec{past_only:[...], known_future:[...], static:[...]}`、`L`、`H`、`model`、`loss{lambda_aux, lambda_freq, type}`、`bands B`、`train{...}`、`seed`。

### 7.3 设计约束（遵循全局规则）
- **不可变**：变换返回新对象；窗口化不原地改原数据。
- **小文件**：每文件 200–400 行，单一职责。
- **边界校验**：`WindowSample` 构造即校验形状；adapter 输出过 schema 校验。
- **错误显式**：数据缺列、形状不符、泄漏检测失败 → 立即 raise，附清晰信息。
- **TDD**：先写测试（形状、无泄漏、RevIN 可逆、合成支配性）再实现。

---

## 8. 实施路线（阶段化，每阶段可验收）

| 阶段 | 目标 | 验收标准 |
|---|---|---|
| **P0 脚手架** | 契约 + ETT adapter + 窗口化 + RevIN + DLinear + Trainer 跑通 | ETT 端到端绿管线；无泄漏测试通过 |
| **P1 基线复现** | iTransformer/PatchTST/TiDE/TFT + S0 全数据 | 复现公开 MSE 在容差内（建立 harness 可信度） |
| **P2 SPECTRE 核心** | 非对称 encoder + 辅助头（A0→A2） | S0/S1 上 A2 ≥ A0；A1 被 A2 支配 |
| **P3 频带耦合** | freq_coupling + 漂移感知 + FreDF 损失（A3→A5） | 完整 SPECTRE；通用 benchmark 出 SOTA 信号 |
| **P4 场景落地** | S1 金融特征+IC/回测；S2 采集 M5 + 销量指标 | S1 IC 显著>基线；S2 WMAPE/RMSSE>基线 |
| **P5 消融+合成+成稿** | A0–A5 全矩阵 + §7.6 合成 + 敏感度 | 机制验证三条结论成立；论文级表格/图 |

---

## 9. 风险与行动项

| 风险 / 缺口 | 影响 | 缓解 |
|---|---|---|
| **零售数据缺失**（无 M5/Favorita） | S2 无法做真实 known-future 协变量实验 | **行动项 A**：采集 M5 + Favorita；过渡用 NN5/Wiki + 日历特征跑通管线 |
| **Crypto 仅 OHLCV，无真实 LOB/OFI** | 微观结构主张偏弱 | 透明声明协变量为 bar 级代理；**行动项 B**：引入 FI-2010 / LOBSTER 强化微观结构故事（可选） |
| 金融数据泄漏 | 结果不可信 | chronological + purge/embargo + 单元测试 `test_windowing_noleak` |
| 辅助任务压过主任务 | 主指标下降 | `λ_aux` 退火 + 敏感度扫描；A0/A2 对照守门 |
| 频带分解不稳定（相位 unwrap） | 训练震荡 | 默认幅度域 + 可学习滤波器；FreDF 损失渐进引入 |
| 基线复现不达标 | SOTA 主张被质疑 | P1 设"复现容差门"，未过不进 P2 |

---

## 10. 已确认的决定（2026-06-09）

1. **命名**：沿用 `SPECTRE` 作为方法名。
2. **主战场权重**：**三战场对等**（S0 通用 / S1 金融 / S2 零售平均用力，强调方法普适性）。指标、消融、数据采集三线并进。
3. **零售数据**：**立即采集 M5 / Favorita**（行动项 A 升为正式任务，含价格/促销/SNAP 等真实 known-future 协变量）。
4. **金融 LOB**：**引入 FI-2010 / LOBSTER**（行动项 B 升为正式任务，支撑强微观结构主张）；Crypto/AU888 OHLCV 代理保留为补充。

### 并行任务线
- **代码线（进行中）**：从 **P0 脚手架**开始 —— `data/contract.py` + `ETT adapter` + `windowing`（含无泄漏测试）+ `RevIN` + `DLinear` baseline + `Trainer`，先立绿管线。P0 仅依赖 ETT，不被数据采集阻塞。
- **数据线（并行）**：行动项 A（M5/Favorita）+ 行动项 B（FI-2010/LOBSTER）采集与入库，落地到 `input/` 并更新 `DATASET_SOURCES.md`。
