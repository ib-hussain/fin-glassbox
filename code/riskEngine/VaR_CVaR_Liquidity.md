# Historical VaR, CVaR, and Liquidity Risk 

## 1. Document Purpose

This document is the final documentation for the three non-parametric / rule-based risk modules:

1. **Historical VaR**
2. **CVaR / Expected Shortfall**
3. **Liquidity Risk**

These modules are implemented in:

```text
code/riskEngine/var_cvar_liquidity.py
```

They belong to the **Risk Engine** of the `fin-glassbox` project:

**An Explainable Distributed Deep Learning Framework for Financial Risk Management**

This document replaces the older VaR/CVaR/Liquidity notes and should be treated as the current reference for module purpose, methodology, outputs, XAI, integration, and validation.

---

## 2. Role in the Full Architecture

The Risk Engine contains both trained neural modules and classical financial risk modules.

VaR, CVaR, and Liquidity sit in the rule/statistical part of the Risk Engine:

```text
Time-Series Market Data
├── returns_long.csv
│   └── Historical VaR / CVaR
│
└── liquidity_features.csv
    └── Liquidity Risk

Outputs
└── Position Sizing Engine
    └── Quantitative Analyst
        └── Fusion Engine
```

These modules do not produce Buy/Hold/Sell signals. They estimate risk constraints used by downstream modules, especially **Position Sizing**.

---

## 3. Why These Modules Are Not Trained Models

Historical VaR, CVaR, and Liquidity Risk are intentionally implemented as non-parametric or rule-based modules.

This is a design strength, not a weakness.

| Module | Why no training is needed |
|---|---|
| Historical VaR | Direct empirical quantile of past returns |
| CVaR / Expected Shortfall | Direct empirical average of tail-loss observations |
| Liquidity Risk | Rule-based score from dollar volume, volume ratio, turnover proxy, slippage proxy |

These modules provide finance-grounded, interpretable anchors inside a system that also contains neural models. They make the Risk Engine more defensible because not every risk estimate is outsourced to a black-box model.

---

## 4. Input Data

### 4.1 VaR/CVaR input

Historical VaR and CVaR consume:

```text
data/yFinance/processed/returns_long.csv
```

Expected essential columns:

```text
date
ticker
simple_return
```

The code expects simple returns for the actual VaR/CVaR calculation. Negative return values represent losses.

### 4.2 Liquidity input

Liquidity Risk consumes:

```text
data/yFinance/processed/liquidity_features.csv
```

Expected feature columns include:

```text
date
ticker
dollar_volume
volume_ratio
volume_zscore
turnover_proxy
```

The module outputs a liquidity score and practical tradability proxies.

---

## 5. Historical VaR

### 5.1 Financial meaning

Value at Risk answers:

```text
What loss threshold should not be exceeded most of the time?
```

For example:

```text
95% VaR = -0.04
```

means that historically, the daily loss was worse than 4% only around 5% of the time, based on the selected rolling window.

### 5.2 Confidence levels

The module computes:

```text
var_95
var_99
```

These correspond to 95% and 99% VaR.

### 5.3 Formula

For confidence level α:

```text
VaR_α = empirical quantile of returns at (1 - α)
```

For 95% VaR:

```text
VaR_95 = 5th percentile of historical returns
```

For 99% VaR:

```text
VaR_99 = 1st percentile of historical returns
```

The module reports VaR as a negative return value.

---

## 6. CVaR / Expected Shortfall

### 6.1 Financial meaning

CVaR answers:

```text
When returns are worse than VaR, how bad is the average tail loss?
```

VaR gives the threshold. CVaR measures the severity beyond that threshold.

For example:

```text
VaR_95  = -0.04
CVaR_95 = -0.07
```

This means the 5% worst outcomes start at -4%, but the average loss inside that tail is -7%.

### 6.2 Formula

For confidence level α:

```text
CVaR_α = mean(return values where return <= VaR_α)
```

The module computes:

```text
cvar_95
cvar_99
```

CVaR is usually more informative than VaR because it measures tail severity, not just the boundary of the tail.

---

## 7. Tail Ratio

The module also computes:

```text
tail_ratio_95
tail_ratio_99
```

The formula is:

```text
tail_ratio = abs(CVaR / VaR)
```

Interpretation:

| Tail ratio | Meaning |
|---:|---|
| ≈ 1.0 | Tail losses are close to the VaR threshold |
| > 1.0 | Tail losses are worse than VaR alone suggests |
| > 1.5 | Fat-tail warning |

Tail ratio is useful because two assets can have the same VaR but very different tail behaviour.

---

## 8. Rolling Window Design

The module uses:

```text
VAR_WINDOW = 504 trading days
```

This is approximately two trading years.

For each ticker and date:

```text
window = returns from max(0, current_index - 503) to current_index
```

This means the module only uses information available up to the current date. It does not use future returns.

### 8.1 Minimum data handling

If fewer than 100 valid observations exist, the module returns missing values for VaR/CVaR. This prevents unstable estimates from very small samples.

### 8.2 No-leakage property

The calculation is point-in-time safe because every day’s risk estimate is calculated only from the historical returns available up to that day.

---

## 9. Liquidity Risk

### 9.1 Financial meaning

Liquidity risk answers:

```text
Can this stock actually be traded without excessive slippage or execution difficulty?
```

A stock can look attractive from a return perspective but still be dangerous if there is not enough trading volume to enter or exit efficiently.

### 9.2 Output interpretation

The module outputs:

```text
liquidity_score
```

where:

```text
0 = very illiquid / difficult to trade
1 = highly liquid / easy to trade
```

Downstream modules typically convert this into risk as:

```text
liquidity_risk = 1 - liquidity_score
```

### 9.3 Liquidity components

The module calculates three component scores:

| Component | Output column | Meaning |
|---|---|---|
| Dollar volume score | `dv_score` | How much money trades daily |
| Volume ratio score | `vr_score` | Today’s volume relative to recent average |
| Turnover score | `to_score` | Volume relative to longer-term activity |

### 9.4 Liquidity score formula

The composite score is:

```text
liquidity_score = 0.4 × dv_score + 0.3 × vr_score + 0.3 × to_score
```

The heavier dollar-volume weight is intentional because actual tradeability depends strongly on how much capital changes hands daily.

---

## 10. Liquidity Component Calculations

### 10.1 Dollar volume score

Dollar volume is converted onto a log scale:

```text
dv_score = clip(log10(dollar_volume) / 9, 0, 1)
```

This avoids allowing extremely large stocks to dominate linearly while still separating illiquid stocks from liquid ones.

### 10.2 Volume ratio score

```text
vr_score = clip(volume_ratio / 2, 0, 1)
```

A volume ratio of around 2 or more is considered highly active.

### 10.3 Turnover score

```text
to_score = clip(turnover_proxy / 2, 0, 1)
```

This captures whether current volume is healthy relative to longer-term activity.

### 10.4 Slippage estimate

The slippage proxy is based on a square-root impact-style relationship:

```text
slippage_estimate_pct = 0.01 / sqrt(dollar_volume / 1,000,000)
```

If dollar volume is zero, a high default slippage value is assigned.

### 10.5 Days to liquidate

The module estimates how long it would take to trade a $1M position:

```text
days_to_liquidate_1M = 1,000,000 / dollar_volume
```

This is a practical execution-risk proxy.

### 10.6 Tradable flag

```text
tradable = 1 if liquidity_score >= 0.3 else 0
```

The Position Sizing Engine can use this to cap or block exposure.

---

## 11. Output Files

### 11.1 Global outputs

The primary outputs are:

```text
outputs/results/risk/var_cvar.csv
outputs/results/risk/liquidity.csv
```

### 11.2 VaR/CVaR columns

```text
var_95
var_99
cvar_95
cvar_99
tail_ratio_95
tail_ratio_99
date
ticker
window_size
```

### 11.3 Liquidity columns

```text
date
ticker
liquidity_score
slippage_estimate_pct
days_to_liquidate_1M
tradable
dv_score
vr_score
to_score
```

### 11.4 XAI outputs

```text
outputs/results/risk/xai/var_cvar_explanations.json
outputs/results/risk/xai/liquidity_explanations.json
```

### 11.5 Chunk-split outputs

For integration with the rest of the project, global outputs are split into chronological chunk files:

```text
outputs/results/risk/chunks/var_cvar_chunk1_train.csv
outputs/results/risk/chunks/var_cvar_chunk1_val.csv
outputs/results/risk/chunks/var_cvar_chunk1_test.csv
...
outputs/results/risk/chunks/liquidity_chunk3_test.csv
```

These files are consumed by the Position Sizing Engine.

---

## 12. Final Production Output Scale

The global production run produced:

| File | Rows |
|---|---:|
| `outputs/results/risk/var_cvar.csv` | 15,712,457 |
| `outputs/results/risk/liquidity.csv` | 15,715,000 |

Observed summary statistics from production VaR/CVaR output:

| Metric | Mean |
|---|---:|
| VaR 95% | -0.0432 |
| CVaR 95% | -0.0649 |
| VaR 99% | -0.0759 |
| CVaR 99% | -0.1015 |

Observed liquidity distribution from production output:

| Threshold | Share of rows above threshold |
|---|---:|
| liquidity score ≥ 0.9 | 2.5% |
| liquidity score ≥ 0.7 | 17.6% |
| liquidity score ≥ 0.5 | 66.9% |
| liquidity score ≥ 0.3 | 95.2% |
| liquidity score ≥ 0.1 | 99.9% |

Tradable rows were approximately:

```text
95.2%
```

Median slippage estimate was approximately:

```text
0.005889
```

---

## 13. Chunk-Aligned Split Outputs

After splitting the global outputs into project chunks, the generated row counts were:

### 13.1 VaR/CVaR chunk outputs

| File | Rows |
|---|---:|
| `var_cvar_chunk1_train.csv` | 3,137,499 |
| `var_cvar_chunk1_val.csv` | 630,000 |
| `var_cvar_chunk1_test.csv` | 627,500 |
| `var_cvar_chunk2_train.csv` | 5,034,958 |
| `var_cvar_chunk2_val.csv` | 630,000 |
| `var_cvar_chunk2_test.csv` | 630,000 |
| `var_cvar_chunk3_train.csv` | 3,775,000 |
| `var_cvar_chunk3_val.csv` | 625,000 |
| `var_cvar_chunk3_test.csv` | 622,500 |

### 13.2 Liquidity chunk outputs

| File | Rows |
|---|---:|
| `liquidity_chunk1_train.csv` | 1,546,136 |
| `liquidity_chunk1_val.csv` | 310,212 |
| `liquidity_chunk1_test.csv` | 308,981 |
| `liquidity_chunk2_train.csv` | 2,479,234 |
| `liquidity_chunk2_val.csv` | 310,212 |
| `liquidity_chunk2_test.csv` | 310,212 |
| `liquidity_chunk3_train.csv` | 1,858,810 |
| `liquidity_chunk3_val.csv` | 307,706 |
| `liquidity_chunk3_test.csv` | 306,270 |

These chunk files make the outputs compatible with the same chronological split structure used by the trained models.

---

## 14. XAI Design

These modules are highly explainable because they are rule/statistical modules. Their XAI is not a post-hoc approximation; it directly traces the calculation.

### 14.1 VaR/CVaR XAI

For each ticker, the XAI JSON includes:

```text
module_name
ticker
date
primary_score
raw_value
percentile_vs_history
trend
thresholds_exceeded
top_positive_factors
top_negative_factors
historical_range
metadata
```

The VaR/CVaR explanation answers:

- What is the latest VaR and CVaR?
- How severe is it relative to that ticker’s own history?
- Is the current VaR elevated or severe?
- Are tail ratios warning of fat-tail risk?
- Is the risk trend increasing, decreasing, or stable?

### 14.2 Liquidity XAI

For each sampled/latest ticker explanation, the XAI JSON includes:

```text
module_name
ticker
date
primary_score
confidence
raw_value
rule_trace
percentile_vs_history
thresholds_exceeded
top_positive_factors
top_negative_factors
trend
component_breakdown
metadata
```

The Liquidity explanation answers:

- What is the current liquidity score?
- Is the stock tradable?
- Which component caused low liquidity?
- Is dollar volume too low?
- Would a $1M position take too many days to liquidate?
- Is liquidity improving or worsening?

---

## 15. Integration with Position Sizing

The Position Sizing Engine consumes these columns:

### 15.1 VaR/CVaR inputs

```text
var_95
var_99
cvar_95
cvar_99
tail_ratio_95
tail_ratio_99
```

It converts them into:

```text
var_cvar_risk_score
```

A typical conversion uses absolute tail severity and tail ratios. Larger loss magnitude and larger tail ratio increase risk.

### 15.2 Liquidity inputs

```text
liquidity_score
slippage_estimate_pct
days_to_liquidate_1M
tradable
```

Position sizing interprets:

```text
liquidity_risk_score = 1 - liquidity_score
```

Poor liquidity can apply a hard cap or prevent a trade entirely if `tradable = 0`.

---

## 16. No-Leakage and Point-in-Time Correctness

### 16.1 VaR/CVaR

VaR and CVaR are computed using historical returns only up to the current date.

This is point-in-time safe:

```text
risk estimate at date t uses returns <= t
risk estimate at date t does not use returns > t
```

### 16.2 Liquidity

Liquidity is calculated from current and historical volume-derived features. It should not use future volume information.

Any rolling inputs inside `liquidity_features.csv` must be constructed with trailing windows only.

---

## 17. CLI Commands

### 17.1 Run the global module

```bash
cd ~/fin-glassbox && python code/riskEngine/var_cvar_liquidity.py --workers 6
```

### 17.2 Run with 4 workers

```bash
cd ~/fin-glassbox && python code/riskEngine/var_cvar_liquidity.py --workers 4
```

### 17.3 Historical note on `--chunk`

The script accepts:

```text
--chunk
```

but in the current implementation, the main calculation writes global outputs to the same files. Chunk-aligned outputs are created by a separate splitting step. Therefore, repeated chunk runs may recompute and overwrite the same global output files unless the code is modified to write chunk-specific outputs directly.

---

## 18. Chunk-Splitting Command

After the global files are produced, use this split logic to generate chunk-specific files:

```bash
cd ~/fin-glassbox && python - <<'PY'
import pandas as pd
from pathlib import Path
out = Path('outputs/results/risk/chunks')
out.mkdir(parents=True, exist_ok=True)
splits = {'chunk1_train': ('2000-01-01', '2004-12-31'), 'chunk1_val': ('2005-01-01', '2005-12-31'), 'chunk1_test': ('2006-01-01', '2006-12-31'), 'chunk2_train': ('2007-01-01', '2014-12-31'), 'chunk2_val': ('2015-01-01', '2015-12-31'), 'chunk2_test': ('2016-01-01', '2016-12-31'), 'chunk3_train': ('2017-01-01', '2022-12-31'), 'chunk3_val': ('2023-01-01', '2023-12-31'), 'chunk3_test': ('2024-01-01', '2024-12-31')}
for name, src in [('var_cvar', 'outputs/results/risk/var_cvar.csv'), ('liquidity', 'outputs/results/risk/liquidity.csv')]:
    print(f'\nSplitting {name}...')
    df = pd.read_csv(src, parse_dates=['date'])
    for split, (start, end) in splits.items():
        sub = df[(df['date'] >= start) & (df['date'] <= end)]
        path = out / f'{name}_{split}.csv'
        sub.to_csv(path, index=False)
        print(f'  {path}: {len(sub):,} rows')
PY
```

---

## 19. Validation Commands

### 19.1 Check global outputs

```bash
cd ~/fin-glassbox && ls -lh outputs/results/risk/var_cvar.csv outputs/results/risk/liquidity.csv outputs/results/risk/xai/var_cvar_explanations.json outputs/results/risk/xai/liquidity_explanations.json
```

### 19.2 Check columns and row counts

```bash
cd ~/fin-glassbox && python - <<'PY'
import pandas as pd
from pathlib import Path
for p in [Path('outputs/results/risk/var_cvar.csv'), Path('outputs/results/risk/liquidity.csv')]:
    df = pd.read_csv(p, nrows=5)
    rows = sum(1 for _ in open(p)) - 1
    print('\n', p)
    print('rows=', rows)
    print('columns=', list(df.columns))
    print(df.head().to_string(index=False))
PY
```

### 19.3 Validate chunk outputs

```bash
cd ~/fin-glassbox && python - <<'PY'
import pandas as pd
from pathlib import Path
base = Path('outputs/results/risk/chunks')
for name in ['var_cvar', 'liquidity']:
    print(f'\n========== {name} ==========')
    for chunk in [1, 2, 3]:
        for split in ['train', 'val', 'test']:
            p = base / f'{name}_chunk{chunk}_{split}.csv'
            if not p.exists():
                print('MISSING', p)
                continue
            df = pd.read_csv(p, nrows=5)
            total = sum(1 for _ in open(p)) - 1
            print(f'OK {p} rows={total:,} cols={list(df.columns)}')
PY
```

### 19.4 Check finite numeric values

```bash
cd ~/fin-glassbox && python - <<'PY'
import numpy as np
import pandas as pd
for p in ['outputs/results/risk/var_cvar.csv', 'outputs/results/risk/liquidity.csv']:
    df = pd.read_csv(p, nrows=100000)
    num = df.select_dtypes(include='number')
    print(p, 'finite_sample=', float(np.isfinite(num.values).mean()))
PY
```

---

## 20. Expected Runtime

VaR/CVaR is slower than liquidity because it computes rolling distributions per ticker.

Observed runtime during production was roughly:

```text
VaR/CVaR computation: about 1.1 to 1.2 hours for 2,500 tickers
VaR/CVaR XAI: about 24 to 25 minutes
Liquidity scoring: comparatively faster, but large CSV apply operation is still heavy
Liquidity XAI: about 5 minutes
```

Runtime varies with worker count, CPU speed, disk I/O, and whether the data is stored on HDD, SSD, or NVMe.

---

## 21. Limitations

### 21.1 Historical VaR limitations

Historical VaR assumes the recent historical distribution is informative for current risk. It may underestimate risk if a new market crisis begins and the past window was calm.

### 21.2 CVaR limitations

CVaR is more informative than VaR, but it still depends on the historical sample. Rare crises may be underrepresented if they are not present in the rolling window.

### 21.3 Liquidity limitations

Liquidity is estimated from proxies. The module does not directly observe the full order book, bid-ask spread, market depth, or actual execution costs.

The slippage estimate is therefore a practical proxy, not a broker-grade execution simulator.

---

## 22. Why These Modules Are Important 

These modules strengthen the project because they bring classical financial risk logic into a deep learning framework.

The system is not only using neural predictions; it also checks:

- historical loss thresholds,
- tail-loss severity,
- liquidity/tradability,
- execution feasibility,
- and rule-trace explanations.

This supports the project’s central idea:

```text
specialisation + multimodality + explainability + modular integration + risk-aware decision-making
```

---

## 23. Final Status

At the current final project state:

```text
Historical VaR: complete
CVaR / Expected Shortfall: complete
Liquidity Risk: complete
XAI outputs: complete
Chunk-aligned output files: complete
Position Sizing integration: complete
```

The outputs are ready for:

- Position Sizing,
- Quantitative Analyst,
- Fusion Engine,
- and final system-level risk explanations.

---

## 24. Summary

Historical VaR, CVaR, and Liquidity Risk are the classical interpretable risk modules inside the project’s Risk Engine.

They provide:

- empirical loss thresholds,
- tail-risk severity,
- liquidity and execution feasibility,
- per-ticker daily risk outputs,
- XAI rule traces,
- and chunk-aligned files for downstream integration.

They require no neural training and remain transparent by design, making them a critical stabilising component in the full explainable financial risk management framework.
