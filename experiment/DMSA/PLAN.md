# DMSA — Phase 1 执行计划

> DMSA = Delayed Mean–Scale Adaptation，课题工作代号。版本 v0.2 · 2026-09-14。
> 科学问题、贡献边界、统计原则与投稿要求以 [RESEARCH.md](../RESEARCH.md) 为准；本文负责执行细节。改变假设或协议时同步更新两份文档及合同版本，不沿用旧验收。
> 当前 F1.01—F1.08 全部 TODO。工作树存在计划与旧 `src/r1/`，不存在 `input/`、`src/dmsa/`、`tests/dmsa/` 或新实验产物；旧数据盘点不代表当前文件可用。本次更新不启动下载、训练或付费计算。

## 1. 阶段问题、范围与时间

在异方差与标签延迟下，利用成熟的前瞻比较证据，能否更可靠地接受有益均值更新，并改善之后的实际预测？不预设精确识别不可见状态，也不把残差方差直接等同真实噪声方差。

Phase 1 以小模型、受控过程及一个合格金融协议检验是否值得继续。通用数据、多骨干和理论发展属于后续工作，不要求两周内完成全部投稿证据。

T0 是实际启动并登记资源预算的日期，当前尚未登记。下列为预算通过后的目标；2026-10-31 是形成可信结果与稿件的努力目标。数据恢复、对手复现或功效不足时据实重排。

| Task | 里程碑 / 依赖 | 退出产物 |
|---|---|---|
| F1.01 协议与对手资格 | T0+2；起点 | 协议、数据来源、原生入口、指标资格 |
| F1.02 数据与标签时钟 | T0+3；正式处理依赖 01 | 数据与切分 manifest、反馈审计 |
| F1.03 计分、回放与预算 | T0+4；人工 fixture 可并行 | 正确性测试、账本、成本外推 |
| F1.04 强基线 | T0+6；真实运行依赖 01—03 | 开发比较池与冻结 B* |
| F1.05 机制与功效 pilot | T0+6；与 04 并行 | 识别边界、受控对照、功效设计 |
| F1.06 唯一候选 | T0+10；依赖 04—05 | 候选、成本与确认合同 |
| F1.07 隔离确认 | T0+13；06 已冻结 | 后续时间证据、消融和区间 |
| F1.08 投入决策 | T0+14；汇总实际结果 | GO / COMPONENT / NO_GO / INCONCLUSIVE / BLOCKED |

Phase 1 不读取 final 目标或分数。时钟、计分、必要对照与隔离验证不能为赶日期而删减。

## 2. 协议、访问与最小模型

| 对象 | 角色与主要终点 | 使用边界 |
|---|---|---|
| Qlib CSI300 × Alpha360 | 首选金融主协议；日横截面 IC 时间均值 | 以 DoubleAdapt 可复算公开版本为锚；冻结数据、标签、评分空间及更新规则；当前未取得数据 |
| G-Research 14 资产 | 条件成立后的分钟开发/迁移；官方 weighted Pearson | 本地留出不等于 private leaderboard；Target 离线评分与因果反馈分别核定；当前文件不存在 |
| 合成过程 | 条件均值损失、更新效用及失效边界 | 独立路径为重复；已知状态仅作诊断 |
| 非金融协议 / CSI500 / 第二骨干 | Phase 2 按贡献选择扩展 | 通用主张拟覆盖三个领域，先核验资格、成本，再冻结矩阵 |

统一 `train → validation_select → validation_confirm → final_test`：初始化、预处理、episode 校准只用许可历史；select 开发调参；confirm 一次冻结批次；final 只登记边界。已用于方法设计的 confirm 永久成为开发资料，不能再次称独立确认。

Qlib 暂定作者外层 valid 按交易日历前后各半分 select/confirm，奇数日归前半；先估计长度与功效再冻结。所有 early stopping/HPO 仅用 select，不传默认整段 valid。原生复现与共同开发协议分别编号，不能拼接不同协议分数。

G-Research 数据恢复且反馈合格后，候选 train=2018—2019、select=2020 上半年、confirm=2020 下半年、final=2021 起至原 train 末尾，均为待核定默认值。UTC 左闭右开端点，按真实依赖 purge；supplemental 先只核验重复/冲突。若旧运行接触过拟定 final，登记历史，改用未用保留段或新时期，不能自动称全新留出。

各配置从相同许可前缀重置回放，不继承其他试验 optimizer、gate、待成熟队列。正式 final 可按预定在线规则在预测后消费成熟标签，但整体分数待冻结批次完成才释放，不反馈方法开发。

模型是训练期拟合后冻结的两层 MLP，hidden=64、ReLU，加相同预处理的原始输入 skip；线性 μθ 与独立线性 log-scale。尺度正值下限训练期固定；μ默认 MSE，σ默认 `log σ + stop_gradient(y−μ)²/(2σ²)`，不据此声明真实高斯或创新。

四动作为 00/10/01/11。**同一起始状态、同一周期内**，00=01、10=11 点预测相同；仅σ更新不改μ、表示或反标准化。σ可能改变将来证据估计，因此整条后续轨迹不必相同。TSFM 仅在小模型通过独立增量门后测试一次接口与价值，无增量则保留一般在线方法定位。

## 3. 目录、交付与运行账本

下列除 PLAN.md 外均为待交付目标，存在、可运行、通过均需后续证据。

```text
experiment/DMSA/
  PLAN.md / PROTOCOL.md / DECISION.md
  configs/
    phase1.yaml / baselines.yaml / synthetic.yaml / candidate.yaml / confirm.yaml
  manifests/
    protocol.json / baseline_registry.json / data_manifest.json
    split_manifest.json / runtime_budget.json / run_registry.jsonl
    access_log.jsonl / power_plan.json / novelty_matrix.json / claim_evidence_map.json
  reports/
    F1.01_QUALIFICATION.md / F1.02_DATA_AUDIT.md / F1.03_RUNTIME.md
    F1.04_BASELINES.md / F1.05_MECHANISM.md / F1.06_CANDIDATE.md
    F1.07_VALIDATION.md / IDENTIFIABILITY.md
  phase2_matrix.json
src/dmsa/
  data.py / metrics.py / replay.py / baselines.py
  synthetic.py / selector.py / models.py / run.py
tests/dmsa/
```

大数据、权重及逐样本预测放仓库外缓存，manifest 保存解析后的绝对路径、来源、哈希。运行登记 `run_id, task_id, protocol_hash, data_hash, code_hash, config_hash, seed, device, start/end, status, artifact_path, error_reason`，失败保留。

拟定 CLI：`PYTHONPATH=src python -m dmsa.run <stage> --config experiment/DMSA/configs/phase1.yaml`，尚未实现。stage=`qualify/prepare/preflight/baselines/mechanism/candidate/confirm/decision`。缺文件、上游未通过、目标访问被禁均显式失败。

## 4. F1.01 — 协议、近邻与研究合同

输入为 RESEARCH 的任务、协议、近邻、成功判定内容及官方论文/代码，固定实际版本，不把旧审阅摘要当实现证明。

1. 登记硬件、预算与数据来源。取得 Qlib 合格版本后用作者最小入口在开发前缀预测并独立复算 IC；先检查入口不会自动打开 final。
2. 抄录标签公式、标准化评分、股票池、处理器、外层日期、update step 与成熟规则；原生/移植不同输入和协议分表。
3. 核定 G-Research 评分与 Target 全部依赖；分别登记 `offline_score_eligible/causal_feedback_eligible`。网页不可读、数据未取得或反馈不明均不通过。
4. 近邻矩阵列问题、输入反馈、更新对象、目标、延迟、保证、实现与差异。OMPB 长预测不自动等于相同真实延迟，其保证不能转述为无界原始 MSE 安全认证。
5. “普通配对损失/方差门+age penalty”为必需简单对手；候选与其不可区分时独立贡献失败。近邻选择/省略依据先于结果。
6. 冻结主终点、B*选型、搜索、开发 seeds=`2021,2022,2023`、功效目标与实用效应；金融ΔIC=0.002 不是所有数据的通用阈值。

`protocol.json`：`protocol_id, version, source_url, source_commit, config_path, config_sha256, data_release, feature_set, label_expr, scoring_space, outer_splits, prediction_clock, label_availability_rule, update_timing, update_step, primary_metric, score_mask_rule, seeds, search_budget, offline_score_eligible, causal_feedback_eligible, claim_scope`。

`baseline_registry.json`：`method_id, source_url, commit, native_entrypoint, native_protocol, ported_protocol, information_budget, search_space, required_phase, status, blocker, omission_reason`。每家族最多 8 格为上限，不要求跑满；保留作者默认，按适用参数构建表。

交付合同、资格报告、近邻矩阵和初版主张证据表。T0+2 不合格则 BLOCKED 并重排；合成可继续，但不能称金融标准协议已可落地。

交付定位：`PROTOCOL.md`、`configs/phase1.yaml`、`manifests/protocol.json`、`baseline_registry.json`、`novelty_matrix.json`、`claim_evidence_map.json`、`reports/F1.01_QUALIFICATION.md`。缺失关键资格项必须能定位具体来源与阻碍。

## 5. F1.02 — 数据、标签与访问审计

1. `input/`当前缺失，先定位合法来源和版本。历史字节/行数仅作线索；恢复后重新计算流式 SHA256、schema、行数、时间和资产覆盖。只恢复所需数据。
2. 检查 `(timestamp, Asset_ID)` 重复、train/supplemental 冲突；相同重复、不同值、缺失分别计数。Qlib 保留发行版交易日历、历史股票池及处理器。
3. 索引真实标签依赖与 available_at，含平滑、残差化、跨资产参数；“15 分钟”不直接等于成熟时间。可借更早上下文，不保留跨界未成熟训练标签。
4. scaler、填补、特征筛选和目标变换保存 fit 截止及状态 hash。标签不插补，评分 mask 由协议固定，不能随方法变化。
5. 统计只读许可区间；记录目标、分数、图表与调参的访问日志。登记 final 边界不允许读取其目标分布或事后事件切片。

样本字段：`sample_id, source_row_id, asset_id, prediction_at, feature_max_available_at, label_dependency_start/end, label_available_at, split, score_eligible, exclusion_reason, input_hash, target_hash`。

Qlib 第一版 Alpha360 不混 Alpha158。G-Research 暂用过去 64 分钟 log-return、High/Low−1、Close/Open−1、log1p(Volume/Count)，附 mask、时间间隔、资产 ID；非正价格无效。peer 仅已完成分钟历史收益均值/标准差与有效资产数，所有方法相同输入。窗口/特征变更计入搜索。

验收为计数守恒、可追溯 hash、特征合法、标签成熟、final 无开发访问。Target 含全期不可得参数则禁止严格因果反馈；保留离线任务或另建因果协议并重跑对手。未解决冲突禁止真实训练。

交付定位：`manifests/data_manifest.json`、`split_manifest.json`、`access_log.jsonl`、缓存样本索引及 `reports/F1.02_DATA_AUDIT.md`；报告按资产×阶段列剔除、缺失与冲突。

## 6. F1.03 — 回放、计分和预算

实现 `predict(features,state) → release_labels(clock) → update(matured,state)`，独立评分。默认同事件先为全部资产保存预测，再释放标签；原生合法顺序不同则另编号，资产遍历不得增加反馈。

预测不可覆盖；live 用于主评分，影子仅动作选择。保存模型、optimizer、gate、RNG、pending 队列支持恢复。

账本：`run_id, protocol_hash, data_hash, config_hash, seed, sample_id, prediction_at, asset_id, y_hat, scale_hat, model_state_id, feature_cutoff, max_consumed_label_available_at, action, cycle_id, evidence_age, candidate_fit_age, oldest_label_age, fit_ids_hash, selection_ids_hash, fit_seconds, infer_seconds, update_seconds, shadow_seconds`。完整标签 ID 集合存缓存关联 hash。

| fixture | 必须断言 |
|---|---|
| 独立计分 | float64 差≤1e-10；不等权、缺失、并列秩、每日样本量不等 |
| 指标口径 | 区分全行 weighted Pearson 与资产内相关均值；Qlib 先日 IC；常数未定义不填 0 |
| 无效预测 | NaN/Inf/缺失令运行无效，不删行提分 |
| 成熟边界 | available_at 早于/等于/晚于事件，消费集合精确符合合同 |
| 未来篡改 | 改未成熟标签/未来特征/未来资产，之前预测、动作、消费不变 |
| 同步资产 | 重排同时间资产，按键还原后相同 |
| 阶段隔离 | Phase1 打开 final 目标/分数立即失败并留日志 |
| 影子隔离 | F 拟合与 E 选择标签不重用，仅后续 live 衡量已选动作 |
| 分支/过期 | 同周期 00=01、10=11；仅σ不改μ；过期/样本不足不提交 |
| 重启恢复 | 连续与保存恢复预测/动作/状态在冻结容差内一致 |
| 设备 | CUDA→MPS→CPU；仅 CUDA pin_memory=True；CPU/实际设备容差预定 |

测冷启动、缓存、训练、推理、普通更新、影子及选择开销，按全量样本/周期外推。80%预算计划、20%重跑；先删辅助数据/骨干/无必要搜索，核心不可负担则 BLOCKED。

拟定 `PYTHONPATH=src python -m pytest tests/dmsa -q` 尚不可运行。旧 `tests/test_oracle_drift.py` 导入缺失的 `experiments.PRISM.oracle_drift`，单独登记；新定向通过不等于全仓库通过。

交付定位：`src/dmsa/metrics.py`、`replay.py`、`tests/dmsa/`、`manifests/runtime_budget.json`、`reports/F1.03_RUNTIME.md`。真实小样本必须能从账本独立复算指标与消费标签集合。

## 7. F1.04 — 强基线与冻结 B*

| 比较臂 | 用途与要求 |
|---|---|
| 零收益率、Ridge、LightGBM、冻结 MLP | 零收益率仅 MSE 诊断；同输入廉价强锚点 |
| 普通 Adam/SGD、小 LR、clip、Huber | 排除稳定化；同初始化与成熟标签池 |
| 尺度归一化梯度、标准化残差阈值门 | 排除稳健降权和简单触发 |
| 配对损失/方差门+age penalty | 必跑最近简单对手；同 F/E、等待、预算 |
| 原始配对损失门、旧分位数阈值四影子规则 | 分离方差/时效修正；旧规则仅 baseline |
| 因果 adaptive RLS/Kalman 线性适应 | 排除已有递归估计/遗忘可解释收益 |
| 日 IC 直接效用门 | Qlib 指标对齐比较，MSE 不保证 IC |
| 原生 DoubleAdapt | 合格 Qlib 金融强对手；保留核心和标签评分区别 |
| 最近通用适应对手 | 先按机制资格指定一个并 smoke；按预算完整开发，欠缺限制结论 |

select 预定短段 smoke 后跑完整许可开发期，每家族≤8 格，全部失败与成本入账。选中规格三个固定 seed 复跑，按单模型指标均值选型，不平均预测制造未声明集成。

B*是 select 必跑可比方法中主终点最佳冻结配置，所有对手保留确认表，不在 confirm 改选弱 B*。输入/骨干不同分原生与匹配表，移植版不能冒充完整作者复现。

正式冻结前对 ADAPT-Z、Proceed、D3A、OMPB 及新直接近邻审核适用性，必要者完成同真实延迟比较。不可运行/任务不同注明理由并缩小主张，不静默省略；Phase1 GO 不等于胜过全部方法。

交付定位：方法包装器、`configs/baselines.yaml`、全部 select 预测/成本、`reports/F1.04_BASELINES.md`。表内逐家族列调参次数、B*、原生/移植差异和未完成对手，import 成功不是复现完成。

## 8. F1.05 — 机制 pilot、识别与功效

14 资产、4 维 AR(1)：`x_t=0.5*x_(t−1)+sqrt(0.75)*η_t`，η公共/资产噪声方差各半，burn-in256。`y=β_rᵀx+σ_r ε`；ε默认独立标准正态；β从(1,0,0,0)到(0,1,0,0)，σ1=3σ0。基准长度 4096、初始化 1024、变化点 2048，训练 episode 用不同变化点与随机流。

主开发矩阵：均值×尺度 2×2、SNR=0.1/0.01、无延迟/主协议真实延迟，共 16 格。预定小子集含状态短于/长于反馈与等待、有限方差厚尾、纯 covariate shift、稳定正确预测器、共同截距/比例变化 IC 负对照。无先兆不要求首批变化标签成熟前识别隐藏变化。

20 个独立 path_id/环境仅 pilot 和方差估计，同 path 不同干预共享随机数，不是独立重复。训练/select/正式机制确认随机流分开；方法共享路径配对，所有环境保留。

输出 live 相对已知μ的 MSE、预测 y 的原指标、接受/拒绝效用、错误更新率、恢复、等待与尺度诊断。E 择优分数不能代替之后预测；oracle 只作诊断，不作可达表现或立项正证据。

IDENTIFIABILITY.md 明确条件零均值、固定候选、成熟样本、输入可激发变化方向、有限适用矩、状态持续等条件。固定候选令 `u=μ1−μ0`、`d=(y−μ0)²−(y−μ1)²`，则在 `E[ε|X,H_s]=0, Var(ε|X,H_s)=1` 下 `Var(d|X,H_s)=4u²σ²`，H_s 为构造候选的完整历史，不能仅假设对 X 条件零均值；这是单点恒等式，不是依赖样本均值方差或未来安全保证。残差尺度含偏差另作错设诊断。

先功效后新确认：用独立 pilot 配对差 sd 与最小效应，可用 `n≈((z.975+z.8)*sdΔ/δmin)²` 近似规划，再模拟检验 80%功效。真实市场用 select 相关结构、配对时间块、预定ΔIC=0.002 估计可检测效应；seed 不当市场重复。冻结 fresh 路径数/随机流、块长规则和上限后才确认。

机制须超越普通稳健更新与配对损失门，并报告低信号/短状态失败区间。低功效不明确记 INCONCLUSIVE；充分证据表明简单门解释增益则贡献 FAIL，停止扩张，不用更多相关时间点伪造样本量。

交付定位：`synthetic.py`、最小 `selector.py`、`configs/synthetic.yaml`、`manifests/power_plan.json`、`reports/IDENTIFIABILITY.md`、`F1.05_MECHANISM.md`。功效文件明确 pilot 已见数据与 fresh 确认数据、最小效应、路径数、统计单位、停止上限。

## 9. F1.06 — F 拟合 → E 前瞻选择 → U 后续效用

这是待证伪经验规则，尚无新颖性或安全性证明。

1. 周期 s 从 live 状态 S 取得最近 K 个成熟完整块 F，复制拟合μ1，独立拟合σ1，保留μ0/σ0。尺度拟合用固定均值锚点并 stop-gradient；不足最小样本不启动。
2. 锚点、候选、live 在周期内冻结，只一个在途周期。对之后 M 块 E 保存μ0/μ1/σ0/σ1 真实前瞻预测；F/E 标签不重用，依赖重叠按合同留间隔。
3. 等 E 成熟；pending/observed/terminal_missing 只在合法到达或 expiry 揭示，不预读缺失 mask。覆盖、最小块、expiry 事前固定，不足/过期关闭，不补选有利块。
4. μ效用为**未经尺度除权的原始**配对 MSE 差 d，固定资产权重和块聚合得 D_b。σ仅参与不确定性。配置明确 K/M、最小样本、块长、HAC lag、尺度上下限、最大 evidence_age、c>0/κ≥0/τμ≥0/γ≥0、τσ=0 及非有限回退。
5. 一版估计：s²=D_b 样本方差；Ω_HAC=固定 lag/Bartlett 权重 Newey–West 长程方差；`G=mean_b[4(Σ_i |w_bi*u_bi*σ0_bi|)²]`，每块内所有计分观测的资产/时间权重总和为 1。γ≥0（初始探针 1，消融 0）、v_min>0，`vhat=max(s²,γG,v_min)`、`Ω=max(Ω_HAC,vhat)`、`n_eff=n*vhat/Ω`，下界式=`mean(D)−c*sqrt(vhat/n_eff)−κ*age`。γG 是经验方差正则，σ0 为 E 预测时冻结的旧尺度，可能含均值偏差；它不构成真实噪声方差界，无覆盖率保证。n 为有效日历块数，n≥2 且大于 HAC lag，预设更强 n_min 由开发校准决定；不足不执行门。lag/clip/校准规则仅 train/select 冻结。
6. 下界式>τμ且未过期、样本足够才接受μ1，否则μ0。age=evidence_age，为 E 预测事件到部署的预定加权平均间隔；门与 age_max 均使用它。candidate_fit_age 记录 F 拟合截止到部署，oldest_label_age 记录最旧证据预测到部署，单位随协议固定。c/κ/τμ仅训练 episode 与 select 校准；最多 8 格只扫少数关键参数，其余固定，不隐式交叉扩大。
7. 先确定已选μ（μ0 或μ1），σ用 E 上**同一已选μ**比较冻结σ0/σ1 的配对 NLL，按相同预定权重计算 mean(NLL 旧−NLL 新)>τσ（第一版τσ=0）才更新。σ1 仍是 F 上以固定μ0 残差拟合的唯一候选，不因选择结果在 E 重拟合；它对新μ1 可能失配，保留 joint 诊断。σ不批准/否决μ，两步选择不预设独立最优，也不声称 NLL 门有显著性保证。
8. 提交后仅新预测 U 证明效用，不用 E 重训后冒充原候选；下周期 F 可按预定规则含成熟 E。等待期、冷启动、过期、计算全部计主表。

普通配对损失/方差门同 F/E/U、age、预算，检验 scale-aware floor 额外信息。方差估计或 age penalty 本身不是贡献；不能超过该门时停止独立顶会方法叙事。

先 smoke 后完整 select 选一个规格，小模型无增量不接 TSFM。通过后第一骨干可尝试 Chronos-2-Synth，固定 revision、映射、缓存边界并沿用规则；接口能跑与预训练增量分别验收。

交付定位：`models.py`、`selector.py`、runner、分支/时钟测试、`configs/candidate.yaml`、`reports/F1.06_CANDIDATE.md`。候选 hash 必须覆盖估计器定义及全部超参数，不能只保存网络权重。

## 10. F1.07 — 冻结确认与归因

confirm 前冻结代码/数据/协议、候选/B*、全部对手、seed 和报告清单 hash。一次完整批次，所有资产、warm/cold 计入；下列是归因覆盖，不默认无预算全组合。

| 比较 | 问题 |
|---|---|
| 完整候选 vs B*和必要对手 | 主收益与是否被其他强方法超过 |
| 同周期始终更新、select 预定周期/随机更新 | 等待/低频解释；不能看 confirm 次数再反向匹配 |
| 固定σ、仅σ、训练固定方差 floor | 学习尺度是否影响后续μ接受及预测 |
| 普通配对损失/方差门+同 age | 是否超出普通稳健选择 |
| 去 age penalty / 仅 age cutoff | 时效增量与过期边界 |
| 日 IC 效用门、同信息同总计算更新 | 指标对齐与多算力解释 |

主比较对 select 冻结 B*；每 seed 独立计分再平均单模型指标，另报 seed 离散度。2000 次配对日历块 bootstrap，所有方法同块、全资产一起；Qlib 重算日 IC 均值，G-Research 重算整条抽样数据 weighted Pearson，不平均块相关。

块长按 train/select 依赖、标签跨度和功效冻结；预定半/两倍敏感性，最低不短于标签依赖跨度。confirm 等分三连续段作描述；不要求每段独立显著。对多数据/指标/消融的显著性主张预定 Holm，其余清楚描述性。

金融开发门目标ΔIC≥0.002、95%配对区间支持正改善、三段至少两段同向，且简单门/成本不能解释全部。有效块少时记录有限样本局限并按功效计划处理，不把固定 20 块当统计定理。区间仍包含有意义改善但功效不足时记 INCONCLUSIVE；若上界已低于预定最小实用效应，按预注册规则记实用目标未达并可 NO_GO，不等于证明零效应。跨零不一律归为证据不足。

确认不用于继续改方法。确证 bug 保留原结果/影响/修复，统一作废受影响运行；若分数参与选型，修复重跑不能恢复独立性，需要新确认设计。

交付定位：`configs/confirm.yaml`、完整预测/动作、`reports/F1.07_VALIDATION.md`，包括主表、消融、三时期、成本、区间与复算命令。确认配置及批次清单的 hash 在首次读取目标前入账。

## 11. F1.08 — 投入决策与投稿准备

从账本复算协议、基线、机制、确认与成本，不训练新模型；主张证据表记录 PASS/FAIL/INCONCLUSIVE/BLOCKED、图表及未证范围。

| 决策 | 条件与后续 |
|---|---|
| GO | 正确性、协议、机制、确认与成本支持继续；冻结 Phase2，尚非 SOTA 或投稿就绪 |
| COMPONENT | 可重复收益被普通门/频率/估计器解释；保存组件与负结果，重新寻找贡献 |
| NO_GO | C2 被简单方法解释，或确认区间上界低于预定最小实用效应；区分创新不足与实用目标未达，不宣称零效应已证 |
| INCONCLUSIVE | 区间仍包含有意义改善但功效不足；打开新确认前决定独立追加或停止并记录上限 |
| BLOCKED | 数据、反馈、必要实现或核心算力不足；列依赖和重排。因预算停止另记 resource_stop，不当成科学反证 |

投稿另需可辩护近邻差异、主张匹配的多域/骨干、适用强对手、统计/成本复算、失效边界、匿名产物及合理理论/统计解释。没有“必须定理才可投稿”或“金融赢一次即顶会”的规则。

`phase2_matrix.json`：`candidate_hash, protocol_hash, primary_metric, comparison_pool, seeds, remaining_baselines, generalization_protocols, compute_budget, power_plan_hash, final_test_access_policy, claim_scope`。三个非金融领域为通用性目标，具体数据/原生 horizon/骨干通过资源资格再冻结，避免无预算全组合。换主线重新立项，不能在已看 confirm 上挑协变量备选直到通过。

交付定位：`DECISION.md`、`phase2_matrix.json`、报告索引和一页贡献草稿。所有 PASS 链接真实产物与命令，回填实际完成日；失败、跳过和证据不足不得勾成完成。

## 12. 当前状态与复用

| Task | 状态 | 完成日 | 结果 |
|---|---|---|---|
| F1.01 | TODO | — | 尚无 |
| F1.02 | TODO | — | 尚无 |
| F1.03 | TODO | — | 尚无 |
| F1.04 | TODO | — | 尚无 |
| F1.05 | TODO | — | 尚无 |
| F1.06 | TODO | — | 尚无 |
| F1.07 | TODO | — | 尚无 |
| F1.08 | TODO | — | 尚无 |

先协议/资源资格、人工 fixture 与合成设计，真实训练等数据/时钟/计分通过。可复用 [R1 负结果](../R1/R1.md)、[设备选择](../../src/r1/models.py)、[哈希与 worker](../../src/r1/native_workflow.py)、[成熟标签过滤](../../src/r1/calibration_probe.py)，旧实现仍按新合同核验。旧 BOOM 聚合、任务 bootstrap、oracle gap 或价格实验不作金融成功证据。

官方来源统一维护于 [研究总纲](../RESEARCH.md)，运行时固定实际版本。另行下载论文遵守 AGENTS.md 命名与 PAPER.md 登记要求；本次未下载论文。
