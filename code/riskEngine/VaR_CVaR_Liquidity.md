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