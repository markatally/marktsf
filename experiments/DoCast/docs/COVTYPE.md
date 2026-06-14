# COVTYPE — Covariate Typing for DoCast Battlefields

Version: v1.0 (M0 deliverable)

Each covariate is typed as:
- **a**: controllable known-future — chosen by a decision-maker in response to (anticipated) target
- **c**: exogenous known-future — known in advance and NOT chosen in response to the target
- **x**: past-only — observable in history but not available as a known-future input

Plan-ahead vs adaptive sub-typing for `a`:
- **a-plan**: decided at forecast origin for the entire horizon (e.g., a promotion calendar)
- **a-adaptive**: may change in response to realized outcomes within the horizon (e.g., markdown triggered by slow sales)

---

## M5 Dataset

| Covariate | Source file | Type | Sub-type | Justification |
|---|---|---|---|---|
| `sell_price` | `sell_prices.csv` | **a** | mostly a-plan; partially a-adaptive | Prices are set weekly per store-item. Markdown events (price drops mid-week) may react to realized inventory/sales → mixed. Competition standard uses weekly prices decided in advance → treated as **a-plan** with V2 diagnostic flag on markdown-heavy segments. |
| `snap_CA/TX/WI` | `calendar.csv` | **c** | — | SNAP (Supplemental Nutrition Assistance Program) eligibility calendars are set by the USDA and state agencies on a multi-year basis. No store can influence the SNAP schedule. Exogenous natural experiment. |
| `event_name_1/2`, `event_type_1/2` | `calendar.csv` | **c** | — | National/sporting/cultural events (Super Bowl, Easter, Christmas) are not chosen in response to store-level demand. |
| `wday`, `month`, `year`, `wm_yr_wk` | `calendar.csv` | **c** | — | Calendar time features; fully determined by the Gregorian calendar. |
| Past sales `d_{1:t}` | `sales_train_evaluation.csv` | **x** | — | Only observable up to forecast origin; not available as known future. |
| `dept_id`, `cat_id`, `store_id`, `state_id`, `item_id` | `sales_train_evaluation.csv` | **static** | — | Non-time-varying identifiers; treated as static features. |

**Identification note (M5)**: The primary V1 confounders are sell_prices reacting to anticipated seasonal demand. The V2 risk is markdown decisions within a week triggered by realized excess inventory. Our treatment of sell_prices as **a-plan** is a modeling choice that will be stress-tested in the V2 diagnostic arm (§5.5 of PROPOSAL).

---

## Favorita Dataset

| Covariate | Source file | Type | Sub-type | Justification |
|---|---|---|---|---|
| `onpromotion` | `train.csv` (chunk) | **a** | a-plan with adaptive risk | Promotion flags are included in the Favorita data as known-future. However, promotions are typically planned seasonally (holiday tie-ins, chain-wide campaigns) → a-plan primary. Risk: ad-hoc promotions reacting to slow recent sales are a-adaptive. Identification diagnostic (§7.3) will quantify the policy-predictability R² to assess overlap. |
| `holiday_type`, `transferred`, `locale` | `holidays_events.csv` | **c** | — | National/regional/local holidays are government-set or fixed-calendar; not chosen in response to store demand. Note: Ecuador earthquake 2016-04-16 → exogenous shock used as natural experiment (§7.5). |
| `dcoilwtico` | `oil.csv` | **x** | (use as x) | Daily WTI oil price is known in arrears at forecast time in most practical settings; Ecuador's economy is oil-dependent but the store-level planner does not set oil prices. Treated as past-only covariate `x`; its future value is not in any planning contract. |
| `transactions` | `transactions.csv` | **x** | — | Number of daily transactions is a realized outcome observable only in history. |
| Store / item static attributes | `stores.csv`, `items.csv` | **static** | — | City, state, cluster, family, class, perishable — non-time-varying identifiers. |

**Identification note (Favorita)**: Favorita's promotion process is confounding-heavy (holiday clusters, inventory cycles). Oil price is past-only (not a known-future controllable). The 2016-04-16 earthquake window is an exogenous supply shock useful for a robustness holdout.

---

## Release Note

This typing is a research artifact, not a data-processing prescription. Users of DoCast should re-examine the plan-ahead assumption for their own business context: if promotions are decided rolling-3-days rather than 28-days in advance, the V2 diagnostic becomes more load-bearing.
