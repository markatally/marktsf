# M1 / M2 最小化脚本参数表（草案）

## M1 真实先导（Pilot）

| 参数 | 值 | 说明 |
|---|---:|---|
| dataset | Weather | 至少 1 个先导域 |
| stage | M1 | 先导闭环 |
| seeds | 2021,2022,2023 | 小规模复现基准 |
| horizons | 24,48,96 | 先导覆盖短中期 |
| target_gain_gate | 预注册中 `target_gain_gate` | 统一门控 |
| metrics | phase-regret, DM, rank-instability | 机制 + 统计 |

## M2 合成/半合成机制检验

| 参数 | 值 | 说明 |
|---|---:|---|
| dataset | synthetic | 受控生成器 |
| stage | M2 | 机制回收阶段 |
| seeds | 2021,2022,2023,2024,2025 | 提升功效 |
| horizons | 24,48,96,192 | 统一测试 |
| target_gain_gate | 预注册中 `target_gain_gate` | 与 M1 同步 |
| metrics | support F1, delay MAE, R_ratio, phase-regret | 机制保真 + 残差占比 |
