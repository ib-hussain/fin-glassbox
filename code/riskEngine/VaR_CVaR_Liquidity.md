Let me explain each module clearly first, then we'll build them.

---

## Historical VaR (Value at Risk)

**What it answers:** "What's the worst daily loss I should expect 95% of the time?"

VaR is the threshold loss that won't be exceeded with a given confidence level. If a stock's 95% VaR is -2.5%, that means: on 95% of trading days, your loss will be less than 2.5%. On 5% of days, it could be worse.

**How it works:** Take 2 years (504 trading days) of daily returns, sort them from worst to best, and pick the 5th percentile (for 95% VaR) or 1st percentile (for 99% VaR). No model training — it's purely statistical.

---

## CVaR (Conditional Value at Risk / Expected Shortfall)

**What it answers:** "When things go really bad (beyond VaR), how bad do they get on average?"

VaR tells you the threshold. CVaR tells you the average loss in the tail beyond that threshold. If VaR 95% = -2.5%, but the average of all returns worse than -2.5% is -4.1%, that -4.1% is CVaR 95%.

CVaR is always worse than VaR and provides a more complete picture of tail risk. Regulators prefer CVaR because it's a "coherent" risk measure.

---

## Liquidity Risk

**What it answers:** "Can I actually trade this stock without moving the price against me?"

A stock might have great returns but if nobody trades it, you can't enter or exit without massive slippage. Liquidity risk measures trading feasibility:

- **Dollar volume** (close × volume): How much money trades daily. $10B/day is liquid; $50K/day is not.
- **Volume z-score**: Is today's volume unusually high or low relative to the past month? Spikes might signal news/events.
- **Volume ratio**: Today's volume divided by 21-day average. Values above 2.0 mean unusual activity.
- **Turnover proxy**: Volume relative to 252-day average. Captures long-term changes in trading activity.

### Code File
```bash
python code/riskEngine/var_cvar_liquidity.py --workers 4
```

**What it produces:**

| File | Contents |
|------|----------|
| `outputs/results/risk/var_cvar.csv` | Per ticker per day: var_95, cvar_95, var_99, cvar_99, tail_risk_ratio, window_size |
| `outputs/results/risk/liquidity.csv` | Per ticker per day: liquidity_score (0-1), slippage_estimate_pct, days_to_liquidate_1M, tradable (bool), component scores |

These feed directly into the **Position Sizing Engine** later. No training — they're ready immediately.

THE OUTPUT:
```bash
(venv3.12.7) fin-glassbox$ python code/riskEngine/var_cvar_liquidity.py --workers 4 --chunk 1 && python code/riskEngine/var_cvar_liquidity.py --workers thon code/riskEngine/var_cvar_liquidity.py --workers 4 --chunk 3
============================================================
RISK ENGINE: VaR, CVaR & Liquidity
============================================================
Non-parametric / rule-based — no training required


=== HISTORICAL VaR & CVaR ===
Loading returns from data/yFinance/processed/returns_long.csv...
  Tickers: 2,500
  Computing VaR/CVaR:  62%|████████████████▎                                              | 1551/2500 [46:09<24:39,  
  Computing VaR/CVaR:  76%|████████████▉                              | 1888/2500 [55:03<14:31,
  Computing VaR/CVaR:  95%|███████████████▏     | 2380/2500 [1:08:07<02:19,  1.44s/it]  
  Computing VaR/CVaR: 100%|█| 2500/2500 [1:11:13<00:00,  1.71s/it]  
  Saved: outputs/results/risk/var_cvar.csv (15,712,457 rows)
  Columns: ['var_95', 'var_99', 'cvar_95', 'cvar_99', 'tail_ratio_95', 'tail_ratio_99', 'date', 'ticker', 'window_size']
  VaR 95% mean: -0.0432
  CVaR 95% mean: -0.0649
  VaR 99% mean: -0.0759
  CVaR 99% mean: -0.1015

=== XAI: VaR/CVaR Explanations ===
  Generating VaR/CVaR XAI:  18%|█                                                                                                 | 457/2500 [04:18<18:56,  1.80it/s]  
  Generating VaR/CVaR XAI: 100%|██████████████████| 2500/2500 [25:11<00:00,  1.65it/s]  
  Saved: outputs/results/risk/xai/var_cvar_explanations.json

=== LIQUIDITY RISK ===
Loading liquidity features from data/yFinance/processed/liquidity_features.csv...
  Rows: 15,715,000
  Computing liquidity scores...
  Saved: outputs/results/risk/liquidity.csv (15,715,000 rows)
  Columns: ['date', 'ticker', 'liquidity_score', 'slippage_estimate_pct', 'days_to_liquidate_1M', 'tradable', 'dv_score', 'vr_score', 'to_score']

  Liquidity score distribution:
    ≥ 0.9: 2.5%
    ≥ 0.7: 17.6%
    ≥ 0.5: 66.9%
    ≥ 0.3: 95.2%
    ≥ 0.1: 99.9%
  Tradable: 95.2%
  Median slippage estimate: 0.005889

=== XAI: Liquidity Risk Explanations ===
  Generating Liquidity XAI: 100%|███████████████████| 500/500 [04:52<00:00,  1.71it/s]  
  Saved: outputs/results/risk/xai/liquidity_explanations.json

============================================================
COMPLETE
============================================================
  outputs/results/risk/var_cvar.csv
  outputs/results/risk/liquidity.csv
============================================================
RISK ENGINE: VaR, CVaR & Liquidity
============================================================
Non-parametric / rule-based — no training required


=== HISTORICAL VaR & CVaR ===
Loading returns from data/yFinance/processed/returns_long.csv...
  Tickers: 2,500
  Computing VaR/CVaR: 100%|███████████████████| 2500/2500 [1:07:21<00:00,  1.62s/it]  
  Saved: outputs/results/risk/var_cvar.csv (15,712,457 rows)
  Columns: ['var_95', 'var_99', 'cvar_95', 'cvar_99', 'tail_ratio_95', 'tail_ratio_99', 'date', 'ticker', 'window_size']
  VaR 95% mean: -0.0432
  CVaR 95% mean: -0.0649
  VaR 99% mean: -0.0759
  CVaR 99% mean: -0.1015

=== XAI: VaR/CVaR Explanations ===
  Generating VaR/CVaR XAI:  52%|█████████████████████████████████████████████████████████████▉                                                        | 1312/2500 [12:39<11:35,  1.71it/s]
  Generating VaR/CVaR XAI: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2500/2500 [25:13<00:00,  1.65it/s]  Saved: outputs/results/risk/xai/var_cvar_explanations.json

=== LIQUIDITY RISK ===
Loading liquidity features from data/yFinance/processed/liquidity_features.csv...
  Rows: 15,715,000
  Computing liquidity scores...
  Saved: outputs/results/risk/liquidity.csv (15,715,000 rows)
  Columns: ['date', 'ticker', 'liquidity_score', 'slippage_estimate_pct', 'days_to_liquidate_1M', 'tradable', 'dv_score', 'vr_score', 'to_score']

  Liquidity score distribution:
    ≥ 0.9: 2.5%
    ≥ 0.7: 17.6%
    ≥ 0.5: 66.9%
    ≥ 0.3: 95.2%
    ≥ 0.1: 99.9%
  Tradable: 95.2%
  Median slippage estimate: 0.005889

=== XAI: Liquidity Risk Explanations ===
  Generating Liquidity XAI: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [05:28<00:00,  1.52it/s]  Saved: outputs/results/risk/xai/liquidity_explanations.json

============================================================
COMPLETE
============================================================
  outputs/results/risk/var_cvar.csv
  outputs/results/risk/liquidity.csv
============================================================
RISK ENGINE: VaR, CVaR & Liquidity
============================================================
Non-parametric / rule-based — no training required


=== HISTORICAL VaR & CVaR ===
Loading returns from data/yFinance/processed/returns_long.csv...
  Tickers: 2,500
  Computing VaR/CVaR: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2500/2500 [1:07:17<00:00,  1.62s/it]  Saved: outputs/results/risk/var_cvar.csv (15,712,457 rows)
  Columns: ['var_95', 'var_99', 'cvar_95', 'cvar_99', 'tail_ratio_95', 'tail_ratio_99', 'date', 'ticker', 'window_size']
  VaR 95% mean: -0.0432
  CVaR 95% mean: -0.0649
  VaR 99% mean: -0.0759
  CVaR 99% mean: -0.1015

=== XAI: VaR/CVaR Explanations ===
  Generating VaR/CVaR XAI: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2500/2500 [23:54<00:00,  1.74it/s]  Saved: outputs/results/risk/xai/var_cvar_explanations.json

=== LIQUIDITY RISK ===
Loading liquidity features from data/yFinance/processed/liquidity_features.csv...
  Rows: 15,715,000
  Computing liquidity scores...


```