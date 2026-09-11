**金融 MISO 时序基础模型：Benchmark、研究方向与 ADAPT-Z 核验（2026-09-11）**

检索及本地核验日期：2026-09-11。工作目录：`/Users/mark/Git/hub/mark-tsf`。

本报告使用用户指定的 timesfm-forecasting 与 academic-research-suite 工作流，结合官方论文、会议收录页、作者代码库、数据主办方说明，以及本地文件检查。属于有针对性的研究决策调查，不是穷尽性系统综述。本文提出的机制是待验证假设，没有运行模型对比，也没有得到新的 SOTA 结果。未下载外部数据集、模型权重或论文 PDF。

**1. 立项结论**

建议主线为：**面向低信噪比、标签延迟和预测关系变化的金融收益率基础模型适应框架**。

- 收益率主实验：先使用本地 G-Research Crypto，再以 Qlib CSI300、CSI500 的固定公开协议作为股票基准候选。
- 波动率泛化：优先考虑 VOLARE；需要分钟窗口预测时，再考虑 Optiver。
- 期货：本地黄金连续合约用于先导实验；当前检索未确认与用户要求同时匹配的、公开且被广泛复用的分钟级传统期货收益率标准 benchmark。不能将自行构造的 AU888 实验称为已有公认 benchmark 的 SOTA。
- 模型研究优先级：①区分噪声与预测关系变化的选择性在线适应；②面向目标资产的增量预测信息筛选。

这里的优先级是基于任务匹配、强基线成熟度、改动规模和可证伪性作出的研究判断，没有足够证据给出“超过 SOTA 的成功概率”。两条路线都需要先通过小规模对照实验。

用户的成功条件是超过某个公认 benchmark 的预测指标，因此无需先证明覆盖全部市场。跨市场、跨时期、突变稳定性和扣费经济价值放入辅助实验；不能以这些指标替代主预测指标。

**2. 先固定任务，避免改变题目后比较分数**

MISO 表示每次查询只预测一个目标变量，可以同时输入该资产多个特征与其他资产的已知历史。共享模型可依次处理不同资产；未来多个 horizon 仍属于同一目标变量的多步预测。是否共享表示或批量处理，不改变原 benchmark 的评估资产集合。

一般研究中的累计对数收益率可定义为 `r(t,h) = log(P[t+h]/P[t])`。但主表应保留 benchmark 原始定义：G-Research 是 15 分钟 residualized return；Qlib 的收益率起止时点由具体配置确定。它们不能统一改成上述公式后继续直接比较原论文数值。

波动率要区分未来 h 期累计实现方差、平均实现方差、以及第 t+h 期的实现方差。价格预测增加随机游走基线 `P_hat[t+h] = P[t]`；收益率为零的基线可用于 MSE，但常数预测的 Pearson 相关系数未定义，不能随意报告成 0。

价格近似随机游走、收益率线性自相关弱和波动率持续性可以同时存在；波动率不需要也像价格一样随机游走。条件异方差也不自动意味着无条件非平稳。应在数据组合中覆盖金融现象，逐项检验，不能预设每个序列同时满足所有性质。金融收益率经验事实可参考 [Cont 的原始研究](https://doi.org/10.1080/713665670)。

| 关心的现象 | 在实验中如何定义或观察 | 不能据此直接推断什么 |
|---|---|---|
| 波动率聚集、厚尾 | 收益率绝对值/平方的相关结构、尾部分位数、超额峰度；检查数据错误和换月 | 不能由高峰度证明所有极端值都是真实跳跃 |
| 随机游走式价格、低 SNR | 与随机游走、零收益率、线性及树模型比较；分时期报告样本外预测增益 | 收益率自相关小不等于独立，也不是精确 SNR 估计 |
| Temporal / covariate shift | 顺序训练和测试，观察输入分布、尺度与缺失机制变化 | `P(X)` 变化不必然等于预测关系变化 |
| Concept / regime shift | 检验条件预测关系和成熟标签上的误差结构变化；使用可控机制实验 | 不能用全样本 HMM 平滑状态作为当时可见输入 |
| Jump 和状态切换 | 预先定义事件检测规则；区分金融事件、停牌、缺口和连续合约换月 | 极端时期切片表现好不能代替完整测试期表现 |

**3. 主实验与辅助 benchmark 选择**

| 数据 / 协议 | 市场与频率 | 原始任务和主要指标 | 本地状态 | 建议位置 |
|---|---|---|---|---|
| G-Research Crypto Forecasting | 14 种加密资产，分钟 | 15 分钟残差收益率；资产加权 Pearson，越高越好 | 原始结构的 CSV 分片及 supplemental 已存在；主 train.csv 尚未拼接 | 最先启动的收益率实验；公共竞赛 benchmark |
| Qlib CSI300 × Alpha360 | 中国股票，日频 | 配置指定的前瞻收益率；IC / RankIC，越高越好 | `input/` 未发现标准数据包 | 股票主 benchmark 候选；锁定数据发行版和配置 |
| Qlib CSI500 × Alpha360 | 中国股票，日频 | 同上 | 本地 CSI500 收盘价矩阵不能替代 Qlib 包 | 第二个股票市场规模 / 股票池实验；不是独立资产大类 |
| VOLARE + 2026 年 TSFM 对比协议 | 股票、外汇、期货；日频实现波动测度 | h=1/5/22 的 RV 预测；QLIKE 为重要主指标，越低越好 | 未发现 | 波动率和跨资产泛化；新数据基础设施，尚非成熟的通用 TSF 排行榜 |
| Optiver Realized Volatility Prediction | 股票订单簿及成交；10 分钟预测窗口 | 后续窗口实现波动率；RMSPE，越低越好 | 未发现 | 分钟窗口波动率辅助；与完整连续分钟 K 线不同 |
| FI-2010 | 5 只股票，事件级订单簿 | 未来中间价运动三分类；常用 macro-F1 | 4 个 DecPre TXT 已存在 | 低优先级微观结构验证；不是连续收益率回归主任务 |

G-Research 的官方数据说明将 `Target` 定义为 15 分钟残差收益率；保留其资产权重、缺失标签计分规则及标签生成方式。[官方数据说明](https://www.kaggle.com/competitions/g-research-crypto-forecasting/data)、[官方评价入口](https://www.kaggle.com/competitions/g-research-crypto-forecasting/overview/evaluation)。它是金融机器学习公共竞赛，不能描述成与 ETT 一样被各类通用 TSFM 普遍采用的统一学术协议。

Qlib 有 CSI300/CSI500、Alpha158/Alpha360 的公开模型比较表。Alpha360 更适合作为从价格与成交历史学习表示的起点，Alpha158 适合作为强手工因子对照；二者不能混用信息预算。官方表格包含多次运行统计，复现时应遵循所选配置的次数与统计方式。表格中的最高分也不等于截至今日所有文献的全局 SOTA。[Qlib 官方 benchmark](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/README.md)。

VOLARE 公开提供股票、外汇及期货的实现波动相关测度。[项目官网](https://volare.unime.it/)、[数据论文，2026 年预印本](https://arxiv.org/abs/2602.19732)。2026 年 7 月的实证论文提供 50 个资产、三个 horizon 的 TSFM 与计量模型对比；其主任务是预测 t+h 时点的 RV，不能直接改成未来 h 天累计 RV。论文发现基础模型并不普遍胜过 Log-HAR，简单 TTM 与 Log-HAR 组合也是必要强基线。该比较仍是预印本，精确复现代码的可得性本次未确认。[对比研究](https://arxiv.org/abs/2607.05291)。

Optiver 有明确的 RMSPE 目标，适合补充波动率预测；匿名时间标识使全局时间顺序和真实在线反馈不能按普通连续时间序列处理，需先核验原始协议。[竞赛主办方说明](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction)、[时间标识讨论](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/249752)。FI-2010 的原始任务是订单簿中间价预测，样本只覆盖较短交易区间，不适合承担多年金融状态切换的核心证据。[原始数据论文](https://arxiv.org/abs/1705.03233)。

**4. 期货优先级与待引入名单**

金融优先级与公共 benchmark 可得性存在冲突。本次检索没有确认“分钟级传统期货收益率 + 完整公共原始数据 + 固定预测评估协议 + 多篇顶会强基线”同时满足的首选对象。这是此次检索的证据边界，不是证明此类资源不存在。

| 候选资源 | 可以解决的问题 | 暂不放入主表的原因 | 本地状态 |
|---|---|---|---|
| CN-Future / AU888 | 本地黄金期货 5 分钟和 60 分钟收益率实验、较长历史迁移 | 未确认通用公共标准协议；连续合约换月与数据来源需核验；单品种不代表全部期货 | 已有 |
| Artur Sepp 的 84 futures 公共归档 | 日频跨股指、利率、外汇、商品期货覆盖 | 是公开研究数据及趋势跟踪资源，未确认其为普遍使用的 TSF 预测 benchmark；日频也低于用户分钟偏好 | 未发现 |
| Jane Street Real-Time Market Data Forecasting | 高维匿名金融目标预测，官方 weighted zero-mean R² | 目标 responder_6 的经济含义和时间尺度不能直接等同于未来 N 期收益率；不能确认真实资产大类覆盖 | 未发现 |
| DRW Crypto Market Prediction | 分钟级匿名加密预测任务 | 目标经济定义、隐藏测试的时序可用性需核验；不替代清晰的原始收益率任务 | 未发现 |
| FinTSB | 金融状态分组、股票预测基准设计 | 当前作者库仍有数据获取与完整预处理发布限制；不能只凭论文标题认定即下即跑 | 未发现 |
| FinTSBridge | MISO 金融预测设计，包括不同金融任务 | 检索到 ICLR 2025 workshop 论文，未确认完整公开数据下载；不属于 ICLR 主会论文 | 未发现 |
| FinVerse | 新的金融基础模型综合评估候选 | 2026 年 8 月预印本；本次未确认完整数据发布和成熟复用，也需核验发布时间对齐 | 未发现 |

资源依据：[84 futures 作者仓库](https://github.com/ArturSepp/TrendFollowingSystems)、[Jane Street 主办方](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting)、[DRW 主办方](https://www.kaggle.com/competitions/drw-crypto-market-prediction)、[FinTSB 作者库](https://github.com/TongjiFinLab/FinTSB)、[FinTSBridge 论文](https://arxiv.org/abs/2503.06928)、[FinVerse 论文](https://arxiv.org/abs/2608.03259)。此表供后续决定是否引入，不代表已下载或已接受其 benchmark 资格。

MASTER 的公开版本也可作为股票候选，但要特别处理版本差异：作者明确说明原论文数据源不能完整公开，后来公开数据及验证处理发生过调整。因此只能比较同一个公开版本重新运行的结果，不能把其结果直接与原论文表格拼接。[MASTER 作者说明](https://github.com/SJTU-DMTai/MASTER)。

**5. 本地数据核验结果**

以下数值来自本次实际文件读取；G-Research 全量行数例外，使用已有 `VERIFY.json` 清单，并检查分片字节数及实际 CSV 内容。未对 G-Research 重新逐行计数，也未与主办方做全量哈希同一性证明。

| 路径 | 本次确认的内容 |
|---|---|
| `input/G-Research-Crypto/chunks/train.csv.part-000` | 1,572,864,000 bytes |
| `input/G-Research-Crypto/chunks/train.csv.part-001` | 1,246,422,393 bytes；两片合计 2,819,286,393 bytes |
| `input/G-Research-Crypto/supplemental_train.csv` | 300,931,304 bytes |
| `input/G-Research-Crypto/VERIFY.json` | 清单记载训练 24,236,806 行、补充 2,518,278 行、14 个资产；训练实际内容时间范围端点为 2018-01-01 至 2021-09-21，补充文件末尾为 2022-01-24，日期按 UTC |
| `input/CN-Future/AU888m5.csv` | 342,018 行 × 8 列，2011-01-04 至 2025-07-10；黄金连续合约 |
| `input/CN-Future/AU888m60.csv` | 34,298 行 × 8 列，同为黄金连续合约 |
| `input/CN-Future/CSI300m60.csv` | 8,000 行 × 8 列，`order_book_id=000300.SH`；是沪深 300 指数，不能因目录名称而认定为股指期货 |
| `input/CSI500/CSI500.csv` | 1,457 行 × 481 列，日期 + 480 序列；不是完整 Qlib 特征数据 |
| `input/SP500/SP500.csv` | 1,509 行 × 476 列，日期 + 475 序列；与部分旧文档列数不同，不能直接认为与文献的股票池一致 |
| `input/FI2010/Train_Dst_NoAuction_DecPre_CF_7.txt` | 矩阵 149 × 254,750；列是样本 |
| `input/FI2010/Test_Dst_NoAuction_DecPre_CF_7.txt` | 矩阵 149 × 55,478 |
| `input/FI2010/Test_Dst_NoAuction_DecPre_CF_8.txt` | 矩阵 149 × 52,172 |
| `input/FI2010/Test_Dst_NoAuction_DecPre_CF_9.txt` | 矩阵 149 × 31,937 |

FI-2010 的 149 行是 144 特征加 5 个标签行；本地属于 DecPre 版本，与其他论文使用的 Z-score 版本不能直接互换。`input/Crypto/` 的 Binance OHLCV 文件与 G-Research 比赛资产及标签不同，也不能替代。`NASDAQ.csv`、`NYSE.csv` 的名称不能证明它们就是其他论文同名股票池的数据版本。

先导检查中，未清洗 AU888 5 分钟收盘价对数收益率的 lag-1 相关约为 -0.019，绝对收益率相关约为 0.147，超额峰度约 509.5。它们提示原始序列值得研究，但换月、跨交易时段缺口和异常记录均可能影响这些数字；尚不能当作真实金融跳跃或低 SNR 的完整统计证明。

**6. 基础模型和既有工作的定位**

| 模型 / 方法 | 已核验的身份 | 在本课题中的作用 |
|---|---|---|
| Chronos-2-Synth | 官方模型卡声明只使用合成单变量、多变量数据预训练，约 0.1B 参数 | 优先评估的可修改起点，降低旧金融测试数据直接进入预训练的风险 |
| TimesFM-3 | Google 于 2026-08-31 发布；原生多变量和协变量能力 | 当前强通用对照及跨 backbone 验证候选；不能继续把 2.5 称为最新版 |
| Kronos | AAAI 2026 金融基础模型；OHLCV 类金融表示 | 金融专用强对照；旧 benchmark 必须检查预训练重叠 |
| ADAPT-Z | ICLR 2026 | 冻结基础模型的在线适应参考与必要对照；不是已证明金融收益率最优的模型 |
| DoubleAdapt | KDD 2023 | 金融增量学习强对照，尤其适合 Qlib；不能只与通用 TSF 方法比较 |
| UniCA | ICLR 2026 | 协变量适应的最近重要对照；“基础模型加协变量”已不构成独立新意 |
| TimeFilter | ICML 2025 | 通道筛选强对照；简单 top-k、稀疏图和跨变量去噪不足以独立宣称新颖 |

模型来源：[Chronos-2-Synth 官方模型卡](https://huggingface.co/autogluon/chronos-2-synth)、[TimesFM-3 官方发布](https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/)、[Kronos AAAI 论文](https://ojs.aaai.org/index.php/AAAI/article/download/39730/43691)、[ADAPT-Z ICLR 页面](https://iclr.cc/virtual/2026/poster/10007091)、[DoubleAdapt 作者代码](https://github.com/SJTU-DMTai/DoubleAdapt)、[UniCA ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/0b5eb45a22ff33956c043dd271f244ea-Abstract-Conference.html)、[TimeFilter 作者库](https://github.com/TROUBADOUR000/TimeFilter)。

ADAPT-Z 在 ETT、PEMS 等通用数据上的结果不能推导为金融收益率或波动率 SOTA；即使其中包含 Exchange，也不是当前 G-Research 或 Qlib 的任务协议。[用户指定的 ADAPT-Z v2](https://arxiv.org/abs/2509.03810v2)。

Kronos 论文描述的金融预训练时间覆盖到 2024 年 6 月，而本地 G-Research 与股票矩阵大部分早于该时点。这只能说明存在需要排查的重叠风险，不能断言这些具体测试样本一定被使用过。合成预训练模型也不能免除下游调参泄漏。对未知预训练语料模型单独标注，不能把同 checkpoint 的消融改善升级成严格的无污染零样本结论。[Kronos 训练与评估说明](https://arxiv.org/abs/2508.02739)。

在已有 foundation model 上做 adapter，论文可定位为“foundation model adaptation framework”。如果没有新的通用预训练与迁移证据，不宜把单任务微调模型描述为自主训练的新 foundation model。

**7. 主攻方向一：区分噪声冲击与预测关系变化的选择性在线适应**

研究问题：**在标签延迟、弱收益率信号和波动聚集下，能否判断当前应更新条件均值、风险尺度、还是暂不更新，从而提高收益率预测指标？**

要质疑的假设是“预测误差变大就应该加快更新”。用 `y = mu(X) + sigma(X) * epsilon` 描述时，误差增大可能来自 `sigma`，未必说明 `mu` 已经变化；直接追逐大残差可能使均值预测器学习噪声。这是机制假设，不能仅凭金融直觉证明。

建议最小方案：保留预训练 backbone，增加收益率均值 adapter、尺度分支以及轻量更新选择器。尺度分支处理波动变化；选择器根据已成熟标签上的标准化误差、梯度一致性和短期可预测性证据，决定更新模块与更新强度。用历史时间段顺序构成元训练 episode，在线只使用实际已到达的信息。

接近的先行工作包括 DoubleAdapt 的金融增量适应、ADAPT-Z 的在线适应，以及将预测均值与噪声状态分开的 DeRegiME。DynaME 也已研究动态专家应对变化。因此，“两个分支”“Student-t”“检测漂移”“混合专家”本身都不是足够的新颖点。需要证明的新贡献是：**在延迟反馈下识别更新对象的机制，以及它相对同预算更新方法的收益和适用边界**。[DeRegiME，2026 年预印本](https://arxiv.org/abs/2605.19231)、[DynaME 作者代码，WWW 2026](https://github.com/shhong97/DynaME)。

| 需要检验的论点 | 必做对照 / 消融 | 否定该论点的信号 |
|---|---|---|
| 尺度突变不应该自动触发均值大幅更新 | 仅改变噪声尺度、仅改变预测系数、二者同时变化的合成机制；算法不知道真实状态，oracle 只作诊断上界 | 无法区分两类变化，或收益只来自减小整体学习率 |
| 选择更新比始终更新更适合收益率 | Frozen、普通在线 SGD、滚动重训、相同平均更新次数的定期更新、ADAPT-Z、DoubleAdapt；相同信息与调参预算 | 相同预算下普通更新已达到同样效果 |
| 增益来自均值信息而不只是波动校准 | 只校准尺度、只更新均值、移除选择器、移除成熟标签屏蔽 | 只改善 QLIKE/NLL，主收益率 Pearson/IC 未改善 |
| 机制有真实金融适用性 | G-Research 主协议与 Qlib 公开协议；完整期 + 预先定义的时期切片 | 只在一个事后挑选的极端窗口成立 |

相对优先理由：能直接对准在线学习在金融弱信号条件下的失效方式；所需新增参数和实验规模可控；对照方法和可证伪机制明确。最大困难是条件均值变化本来就很难从噪声中识别，不能预设选择器能做到。

**8. 主攻方向二：面向目标资产的增量预测信息筛选**

研究问题：**哪些其他资产与特征，在当前状态和预测 horizon 下，真正增加目标收益率的样本外可预测信息？**

要质疑的假设是“跨资产相关性高、attention 权重大，意味着该变量有预测价值”。共同市场波动会造成很强的同期相关；对未来收益率的增量信息可能很小，而且 lead-lag 关系可能变化。

建议最小方案：目标资产和 horizon 作为查询，只在已知历史上生成协变量表示；在基础模型的跨变量交互前加入轻量门控。门控的学习目标与增量预测收益及跨时间稳定性关联，而不是只按相关系数或注意力大小筛选。保留缺失标识和资产身份；不能把其他资产未来价格或成交量当成已知外生变量。

UniCA、TimeFilter、MASTER，以及已有多变量基础模型，已覆盖协变量融合、通道筛选和跨股票建模。拟议空间是**目标条件下的增量信息判定如何随状态变化而调整，并避免把共同噪声当成可迁移预测关系**。这仍需要更细的相邻文献审查与机制实验，不能预先认定没有同类工作。[MASTER 作者库](https://github.com/SJTU-DMTai/MASTER)、[UniCA](https://proceedings.iclr.cc/paper_files/paper/2026/hash/0b5eb45a22ff33956c043dd271f244ea-Abstract-Conference.html)、[TimeFilter](https://arxiv.org/abs/2501.13041)。

| 需要检验的论点 | 必做对照 / 消融 | 否定该论点的信号 |
|---|---|---|
| 筛选利用预测关系，不只压缩维度 | 单资产、全部协变量、随机同数量协变量、固定相关性 top-k、TimeFilter / UniCA；输入完全相同 | 随机选择或简单 top-k 即可取得同等收益 |
| 选择器能抑制共同噪声和失效关系 | 增加无关但尺度相似的通道；仅在诊断副本中打乱跨资产对齐；构造 lead-lag 改变 | 只要加更多变量就有相同收益，不能解释机制 |
| 预训练表示对弱信号有用 | 相同门控搭配线性、MLP、LightGBM 特征和从头初始化小网络 | Foundation backbone 没有可重复的增量价值 |
| 状态条件化是必要机制 | 静态门控、去除目标查询、去除 horizon、去除状态输入 | 改善仅来自参数量或额外特征 |

相对优先理由：收益率 MISO 任务天然适配，且本地 G-Research 有跨资产分钟数据；模块改造较小。主要风险是领域已拥挤，必须清楚超过现有协变量适应与筛选方法，不能仅重命名一个 attention 模块。

**9. 作为基础配置，而非单独创新点的内容**

- 直接预测目标收益率；未来价格和 RV 是辅助任务。原始累计收益率、残差收益率和横截面排序应分别命名与计分。
- 使用过去信息估计尺度，但保留可用的波动状态特征，避免一边标准化掉风险信息、一边宣称建模波动状态。
- 先比较小型收益率 head、常规 LoRA、MSE 与相关性目标等基础适应方法。收益率、RV 多任务训练要比较是否产生负迁移。
- 不把“LoRA + RevIN + 多头输出 + Student-t”拼接本身作为论文贡献。也不优先用深度 RL 优化交易收益，因为当前首要成功指标是预测 benchmark。
- 暂不从零训练大型金融基础模型；先用可修改预训练骨干建立有效模块和严格协议。不能用更大模型和更大搜索预算替代机制证据。

**10. 公平比较和成功判定**

1. 固定 benchmark 版本、文件哈希、资产池、特征、标签、训练/验证/测试边界和缺失处理。每个 horizon 预先指定主指标。股票按同一时点的资产横截面计算 IC；G-Research 保留其官方加权相关聚合，二者不可混用。
2. 将静态与在线两条赛道分开。保持原始静态协议用于既有分数对照；在线赛道让所有允许更新的方法获得相同的成熟标签、重训机会与预算。不能仅允许新方法使用测试期标签后声称打破原静态榜。
3. 对标签设定实际可得时间，而非只看行时间。G-Research 最近历史行的 `Target` 仍可能依赖未来价格，不能把这些值作为当时已知的模型历史。Qlib 也须按收益率标签终点延迟反馈。
4. 训练/验证和验证/测试边界清除跨界标签；purge 长度依据真正标签终点。跨资产随机拆行、全样本归一化、全样本状态平滑都不能用于因果主实验。
5. 同预算强基线包括线性/MLP/LightGBM、所选 benchmark 的既有优胜方法及简单集成；再比较 TSFM 冻结、普通微调和创新模块。RV 必须包含 HAR/Log-HAR 及简单组合；价格必须有随机游走。
6. 审查预训练重叠；对未知语料单列结果与局限。冻结同 checkpoint 的消融可以解释模块增量，但不能自动解决基础模型对历史测试样本的接触问题。
7. 原官方测试集与评分服务可用时，按其协议评价；若只能使用训练集自建时间留出，必须将强基线在同一划分重跑，只声称该公开复现协议下的领先，不能拿分数直接比较 Kaggle private leaderboard。
8. 报告按协议重复运行的均值和不确定性。对重叠 horizon、跨资产共同冲击采用适当时间块重采样或相关性稳健检验；多次搜索只用验证集，不用最终测试集决定模块和参数。
9. 在至少一个预先指定的主 benchmark / 指标上，超过同协议下充分调优的最强可复现对照，且改善不是随机种子偶然性，才认定该实验达到用户定义的目标。跨市场和经济价值作为辅助，不扩大已证实的结论范围。

DoubleAdapt 作者特别指出当前接口存在标签标准化评价和 horizon/update-step 的适用限制；移植时需要记录原设置与修订设置，不能悄悄改变强基线。[实现说明](https://github.com/SJTU-DMTai/DoubleAdapt)。

**11. 建议实施顺序与停止条件**

| 阶段 | 具体产物 | 继续投入的条件 |
|---|---|---|
| 基线和协议 | G-Research 分片恢复、数据签名、固定时间协议、标签成熟规则；树模型/MLP/冻结及常规微调 TSFM 对比 | 数据和基线可复现；找得到稳定失败方式；暂不宣称 SOTA |
| 方向一最小实验 | 仅均值 adapter + 尺度分支 + 更新选择器，先做合成机制和一个历史验证段 | 在相同预算下胜过简单在线更新，且收益率主指标改善 |
| 方向二独立实验 | 单独增加协变量选择，暂不叠加方向一 | 胜过全部通道与简单筛选；增益不能被树模型新增特征完全解释 |
| 正式主实验 | 验证期冻结方案后评估完整留出期；若引入 Qlib，则使用确定的公开发行版重跑强基线 | 至少一个预先指定的收益率 benchmark 有可靠改善 |
| 泛化证据 | 用户决定引入后，再做 VOLARE/Optiver；本地黄金先做迁移；增加价格辅助头 | 主预测增益与机制解释保持一致，清楚标明新协议和辅助任务 |

如果方向一只有波动率指标改善，不能把它包装成收益率预测成功；可以作为独立波动率课题候选，但不能替代本次主目标。如果方向二只是常规门控带来的小幅调参收益，应停止围绕它构建顶会级新颖性叙事。

**12. 检索边界与尚待核验项**

主要检索词包括 `financial time series foundation model benchmark 2026 returns`、`G-Research residual return evaluation`、`Qlib benchmark DoubleAdapt MASTER`、`futures minute forecasting public benchmark`、`VOLARE foundation models realized volatility`、`ADAPT-Z ICLR 2026`、`covariate adaptation TimeFilter UniCA`、`mean variance regime adaptation DeRegiME`。优先采用会议主办方、作者论文与仓库、数据主办方；没有将 workshop 或 arXiv 标为顶会主会收录。

仍待后续立项核验：缺失数据的实际下载权限与完整性、Qlib 具体发行版及其最强同协议近期对照、G-Research 官方历史测试评分的当前可用性、VOLARE 对比代码与确切版本、各模型预训练资产清单及可微调接口、期货连续合约换月规范，以及拟议两条机制的更细粒度新颖性检索。

本次本地状态已重新检查；既有记忆只用于确定检查范围，不作为当前数据存在性的唯一证据。

**13. ADAPT-Z 的 SOTA 口径及 13 个 benchmark 本地覆盖**

对用户指定的 v2，准确结论是：**Table 2 在 13 个数据集上，跨三个 backbone 和三个 horizon 平均后的 MSE 均为表内最优**。这不等于每个 backbone × horizon 的单项均最优；例如 Table 8 的 TimesNet / PEMS03 / H=1，DSOF 为 0.0505，ADAPT-Z 为 0.0582。表中 IMP 相对的是原模型，不是相对第二名。[原文 Table 2 与 Table 8](https://arxiv.org/html/2509.03810v2)。

下表的平均 MSE 来自论文 Table 2；路径和形状为本次对 `mark-tsf/input/` 的实际核验，CSV 形状包括日期列，TXT/NPZ 按原始数组表示。

| Benchmark | ADAPT-Z 平均 MSE | 本地文件 | 本地形状 |
|---|---:|---|---|
| ETTh1 | 0.2657 | `input/ETT/ETTh1.csv` | 17,420 × 8 |
| ETTh2 | 0.1604 | `input/ETT/ETTh2.csv` | 17,420 × 8 |
| ETTm1 | 0.1937 | `input/ETT/ETTm1.csv` | 69,680 × 8 |
| ETTm2 | 0.0937 | `input/ETT/ETTm2.csv` | 69,680 × 8 |
| PEMS03 | 0.0959 | `input/PEMS03/PEMS03.npz` | 26,208 × 358 × 1 |
| PEMS04 | 0.1223 | `input/PEMS04/PEMS04.npz` | 16,992 × 307 × 3 |
| PEMS07 | 0.0892 | `input/PEMS07/PEMS07.npz` | 28,224 × 883 × 1 |
| PEMS08 | 0.1426 | `input/PEMS08/PEMS08.npz` | 17,856 × 170 × 3 |
| Electricity | 0.1096 | `input/Electricity/electricity.csv` | 26,304 × 322 |
| Exchange | 0.0429 | `input/Exchange/exchange_rate.csv` | 7,588 × 9 |
| Solar | 0.0948 | `input/Solar/solar_AL.txt` | 52,560 × 137 |
| Traffic | 0.3689 | `input/Traffic/traffic.csv` | 17,544 × 863 |
| Weather | 0.1481 | `input/Weather/weather.csv` | 52,696 × 22 |

因此本仓库为 **13/13 数据文件存在**，但尚未证明所有文件与原作者版本逐字节一致或已经完成复现。v2 正文写 horizon 12/24/48，详细表格写 1/24/48，存在不一致；正式复现需以作者代码配置核定并记录。论文使用 lookback 96、60%/10%/30% 顺序划分，不能直接与其他划分的 MSE 比较。

对于最早提出的“2026 年最高质量的一篇”，本次没有足够证据给出跨任务的绝对最高排名。面向金融预训练骨干，Kronos 是已核验的 AAAI 2026 参考；面向在线适应，ADAPT-Z 是已核验的 ICLR 2026 参考。二者角色不同，均不能凭原文结果直接推导为本课题收益率 benchmark 的最优起点。
