You already have my previous master prompt in context, and I am also giving you:
- the `.env` (below)
- the folder structure for the repository(below)
- `DATA.md`
```bash
#ignore the ibrahim_hussain@IbLaptop:/mnt/c/Users/ibrahim/Downloads/ path at start of the repo as this is the absolute path of another group member and for me it is different LBRepoPath = "/mnt/d/Deeplearning/fin-glassbox"

ibrahim_hussain@IbLaptop:/mnt/c/Users/ibrahim/Downloads/fin-glassbox$ ls data/sec_edgar/
logs  processed  raw
ibrahim_hussain@IbLaptop:/mnt/c/Users/ibrahim/Downloads/fin-glassbox$ ls data/sec_edgar/raw/
api  bulk  filings_txt  indexes
ibrahim_hussain@IbLaptop:/mnt/c/Users/ibrahim/Downloads/fin-glassbox$ ls data/sec_edgar/processed/
companyfacts  manifests
ibrahim_hussain@IbLaptop:/mnt/c/Users/ibrahim/Downloads/fin-glassbox$ ls data/sec_edgar/processed/companyfacts/
_shards  companyfacts_errors.csv  companyfacts_inventory.csv
_tmp     companyfacts_flat        companyfacts_summary.json
ibrahim_hussain@IbLaptop:/mnt/c/Users/ibrahim/Downloads/fin-glassbox$ ls data/sec_edgar/processed/companyfacts/ -R | wc -l
209
ibrahim_hussain@IbLaptop:/mnt/c/Users/ibrahim/Downloads/fin-glassbox$ cd data/sec_edgar/processed/companyfacts/
ibrahim_hussain@IbLaptop:/mnt/c/Users/ibrahim/Downloads/fin-glassbox$
ibrahim_hussain@IbLaptop:/mnt/c/Users/ibrahim/Downloads/fin-glassbox$ cat .env 
API_KEY_IB = "0d3acacabf93691c3cb4cea1b0fb4a6d"
API_KEY_SB = "a2688903e279a9178d98899ea097fd08"
API_KEY_LB = "39cc89e8e96ba809e81198414fc22186"

SBRepoPath = ""
LBRepoPath = "/mnt/d/Deeplearning/fin-glassbox"
IBRepoPath = "/mnt/d/Downloads/University/DeepLearning/fin-glassbox"

DEBUG_MODE = 0
# 0 = no debug, 1 = debug mode on (prints additional info for debugging purposes)
ENDING_WEEK ="21"
PROCESSOR = "cpu"
# "cuda" or "cpu"

datasets_in_tickerCollector_path = "assignment2work/datasetsIn"
datasets_out_tickerCollector_path = "assignment2work/datasetsOut"

datasets_FourierGNN_path = "assignment2work/FourierGNN/data"
datasets_FourierGNN_output_path = "assignment2work/FourierGNN/output"

datasets_MTGNN_path = "assignment2work/MTGNN/data"
save_MTGNN_path = "assignment2work/MTGNN/save"
MTGNN_baseline_path = "assignment2work/MTGNN"

datasets_StemGNN_path = "assignment2work/StemGNN/datasets"
model_StemGNN_path = "assignment2work/StemGNN/model"
result_file_StemGNN_path = "assignment2work/StemGNN/output"
base_StemGNN_path = "assignment2work/StemGNN"

outputsPathGlobal = "outputs"
embeddingsPathGlobal = "outputs/embeddings"
figuresPathGlobal = "outputs/figs"
modelsPathGlobal = "outputs/models"
resultsPathGlobal = "outputs/results"
codeOutputsPathGlobal = "outputs/codeResults"

dataPathGlobal = "data"
secDataPathGlobal = "data/sec_edgar"ibrahim_hussain@IbLaptop:/mnt/c/Users/ibrahim/Downloads/fin-glassbox$ 

```
Treat all of that as already available context.

Your role in this chat is to act as my **Senior Data Engineering and Macro-Regime Pipeline Partner** for the **macro / regime data module** of my project:

# Project
**An Explainable Distributed Deep Learning Framework for Financial Risk Management**

This chat is specifically responsible for the **Macro / Regime Data** family only.

Do **not** drift into SEC filings/companyfacts or Yahoo market-data engineering in this chat unless needed for alignment discussion. Those are being handled separately. Your job is to independently design and implement the macro/regime side as strongly as possible.

---

# 1. FULL PROJECT CONTEXT YOU MUST REMEMBER

The full architecture has 5 data families:

1. Time-Series Market Data
2. Financial Text Data
3. Fundamental Company Data
4. Macro / Regime Data
5. Cross-Asset Relation Data

This chat is responsible for **Data Family #4: Macro / Regime Data**.

The broader finalized architecture includes:
- Shared Temporal Attention Encoder for technical market sequences
- FinBERT for text
- Fundamental model for structured company fundamentals
- Risk engine:
  - volatility model
  - drawdown model
  - Historical VaR
  - Historical CVaR
  - GNN contagion/correlation risk
  - liquidity model
  - regime detection model
  - position sizing engine
- Fusion layer
- XAI layer

The macro/regime data from this chat will primarily support:
- regime detection
- macro context modeling
- risk interpretation
- market environment classification
- later fusion and XAI context

---

# 2. DATA TARGET FOR THIS CHAT

You are responsible for acquiring and engineering **large-scale free U.S. macro / regime data**.

## Historical range
Target:
- **2000-01-01 to 2024-12-31**

## Required source
Primary source:
- **FRED (Federal Reserve Economic Data)**

This is the main source for this task.

Use FRED as the first-choice free source.
You may mention ALFRED / vintage-aware considerations if relevant, but the immediate implementation should focus on a strong FRED-based historical macro/regime pipeline unless a clearly better free-first design is justified.

---

# 3. EXACT TYPES OF MACRO / REGIME DATA NEEDED

You should propose and collect a strong set of regime-relevant U.S. macro series.

At minimum, the pipeline should strongly consider categories like these:

## Interest rates / policy
- Federal Funds Rate
- Treasury yields at multiple maturities
- short-term bill rates
- yield curve spreads

## Inflation
- CPI
- Core CPI
- PCE if available/justified
- Core PCE if available/justified

## Labor market
- Unemployment rate
- payroll / employment indicators if practical

## Growth / production / activity
- industrial production
- retail sales
- recession indicators
- economic activity proxies

## Credit / stress / risk proxies
- corporate yield series or spreads if available
- financial condition / stress proxies if available in FRED
- volatility/fear proxies if available through FRED-linked series

## Regime labels / state helpers
- recession indicators
- yield curve inversion indicators
- inflation regime markers
- rate regime markers
- macro slowdown / acceleration proxies

You should propose a rigorous macro feature universe, not just a few random series.

---

# 4. DATA ENGINEERING RESPONSIBILITIES FOR THIS CHAT

This chat must do full macro/regime engineering, not just download a few CSVs.

## Phase A — Raw acquisition
Download and preserve:
- raw FRED series
- metadata per series
- update timestamps
- units/frequency information

## Phase B — Standardization
Create clean macro tables with:
- standardized series IDs
- clear names/descriptions
- dates parsed correctly
- frequency preserved
- missing-value handling defined
- consistent storage format

## Phase C — Regime-oriented engineering
Engineer useful macro features such as:

### Level and change features
- raw levels
- first differences
- percentage changes
- year-over-year changes
- month-over-month changes where appropriate

### Spread features
- 10Y–2Y
- 10Y–3M
- other yield spreads if justified

### Inflation regime helpers
- rising/falling inflation flags
- inflation momentum
- inflation trend windows

### Rate regime helpers
- rising/falling rate environments
- inversion states
- tightening/easing proxies

### Recession / macro state helpers
- recession indicator alignment
- risk-on / risk-off style macro flags if justified
- rolling z-scores and macro shock indicators

## Phase D — Frequency alignment
This is important.

Because macro series come in mixed frequencies:
- daily
- weekly
- monthly
- quarterly

You must design a clean strategy for:
- preserving raw frequency
- also building a **daily-aligned macro/regime feature table** suitable for later joining with daily market data

This must be done carefully to avoid leakage.

## Phase E — Quality controls
Include:
- missingness reports
- date coverage per series
- frequency reports
- series metadata summary
- transformation audit

---

# 5. IMPORTANT DESIGN RULES

## Rule 1 — No leakage
Macro features must not use future values.
When aligning to daily market dates, use only information that would have been available by that date.

## Rule 2 — Keep raw and processed separate
Preserve a raw archive and processed outputs separately.

## Rule 3 — Daily-aligned processed output is required
Even though raw macro series have mixed frequencies, we eventually need a daily-aligned macro/regime feature table for the broader financial framework.

## Rule 4 — Resume/incremental support
The pipeline should:
- resume safely
- avoid destructive reruns
- skip already completed downloads where possible
- preserve logs/progress

---

# 6. STORAGE / OUTPUT PHILOSOPHY

Use the environment / folder structure I provide, but follow this design principle:

## Raw layer
Should preserve:
- downloaded series values
- series metadata
- raw API responses if useful

## Processed layer
Should contain:
- macro series master table
- transformed series tables
- spread tables
- regime-oriented feature tables
- daily-aligned macro feature table
- quality audit outputs

Preferred formats:
- CSV is acceptable
- Parquet may be proposed if justified
- explain all format decisions clearly

---

# 7. WHAT THIS CHAT MUST DELIVER

I want this chat to help build **Linux-ready optimized Python code** that does as much useful work as possible in each long run.

Every script should:
- be optimized
- show progress
- be robust
- support resume/incrementality where useful
- separate raw and processed outputs
- handle mixed-frequency data carefully
- be suitable for large, reproducible runs

The code should be:
- Python 3.12+
- Linux-oriented
- production-style
- strongly commented
- typed where useful
- memory-conscious
- designed for correctness first, then speed

---

# 8. WHAT OUTPUTS I ULTIMATELY WANT FROM THIS CHAT

At minimum, I want this chat to help create:

1. **macro series master / metadata table**
2. **raw downloaded macro archive**
3. **clean transformed macro tables**
4. **spread / change / regime feature tables**
5. **daily-aligned macro-regime table**
6. **quality audit tables**
7. any additional summary outputs useful for later regime modeling

---

# 9. HOW THIS MODULE CONNECTS TO THE FULL PROJECT

This macro/regime data later feeds:
- regime detection model
- risk interpretation
- fusion context
- XAI explanations around macro environment
- possibly cross-asset relation interpretation

So the outputs must be designed for later joining with daily market data and later model use.

---

# 10. WHAT NOT TO DO

- Do not work on SEC filings/companyfacts in this chat.
- Do not work on Yahoo market-data engineering in this chat.
- Do not stop at raw download only.
- Do not ignore progress reporting.
- Do not ignore mixed-frequency alignment issues.
- Do not silently introduce future-data leakage.

---

# 11. IMMEDIATE TASK

Start by doing this:

1. propose the **full macro/regime acquisition and engineering plan**
2. define the **FRED series universe** for 2000–2024
3. define the **raw + processed folder layout**
4. define the **transformation and daily-alignment strategy**
5. then generate the first **optimized Linux-ready Python pipeline file** for this module

That first code file should aim to do as much of the macro/regime acquisition + standardization + transformation + daily alignment as reasonably possible in one run.

Continue from there as my dedicated macro/regime engineering partner.