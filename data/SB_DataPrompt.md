You already have my previous master prompt in context, and I am also giving you:
- the `.env` (below)
- the folder structure for the repository(below)
- `DATA.md`
```bash
#ignore the ibrahim_hussain@IbLaptop:/mnt/c/Users/ibrahim/Downloads/ path at start of the repo as this is the absolute path of another group member and for me it is different SBRepoPath 

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

Your role in this chat is to act as my **Senior Data Engineering and Market Data Pipeline Partner** for the **time-series market data module** of my project:

# Project
**An Explainable Distributed Deep Learning Framework for Financial Risk Management**

This chat is specifically responsible for the **Time-Series Market Data** family only.

Do **not** drift into SEC filings/companyfacts engineering in this chat unless absolutely required for alignment discussion. That work is being handled separately. Your primary job is to independently design and implement the market-data side as strongly as possible.

---

# 1. FULL PROJECT CONTEXT YOU MUST REMEMBER

The full architecture has 5 data families:

1. Time-Series Market Data
2. Financial Text Data
3. Fundamental Company Data
4. Macro / Regime Data
5. Cross-Asset Relation Data

This chat is responsible for **Data Family #1: Time-Series Market Data**.

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
  - regime model
  - position sizing
- Fusion layer
- XAI layer

The market data from this chat will feed:
- technical encoder
- technical analyst
- volatility model
- drawdown model
- historical VaR/CVaR
- liquidity model
- parts of regime modeling
- later cross-asset relation engineering

So your work is central.

---

# 2. DATA TARGET FOR THIS CHAT

You are responsible for acquiring and engineering **large-scale free U.S. stock market time-series data**.

## Historical range
Target:
- **2000-01-01 to 2024-12-31**

## Universe target
Target:
- **at least 300–500 U.S. stocks**
- preferably liquid U.S. equities
- must also include:
  - major benchmark ETFs
  - major sector ETFs
  - major indices if possible through source availability

## Required source
Primary source:
- **yfinance / Yahoo Finance**

This is the main source for this task.

You may suggest optional fallbacks or validation backups, but the implemented pipeline should be built around **yfinance first**, unless a better free-first design is strongly justified.

---

# 3. EXACT TYPES OF MARKET DATA NEEDED

You must collect and preserve the following as much as possible from Yahoo Finance:

## Core price/volume data
- Open
- High
- Low
- Close
- Adjusted Close (if available through retrieval path)
- Volume

## Corporate actions
- Dividends
- Stock splits

## Metadata if available
- ticker
- company name
- exchange
- sector / industry if available
- currency
- market cap if available
- shares outstanding if available from yfinance metadata
- listing-related metadata where accessible

## Benchmark / context series
At minimum also collect:
- SPY
- QQQ
- DIA
- IWM

And preferably sector ETFs such as:
- XLF
- XLK
- XLE
- XLV
- XLI
- XLP
- XLY
- XLU
- XLB
- XLRE
- XLC

You should propose a robust benchmark/universe set and implement it.

---

# 4. DATA ENGINEERING RESPONSIBILITIES FOR THIS CHAT

This chat must not stop at raw download.

You must design and implement the **full market-data engineering pipeline**.

## Phase A — Raw acquisition
Download and preserve:
- raw OHLCV
- corporate actions
- ticker metadata
- benchmark/ETF data

Store raw and processed separately.

## Phase B — Standardization
Create a clean standardized daily market panel with:
- consistent ticker identifiers
- normalized column names
- parsed date index
- sorted dates
- deduped rows
- clear missing-value handling
- consistent adjusted/unadjusted price choices

## Phase C — Feature engineering
Generate technical and risk-relevant features such as:

### Return features
- daily returns
- log returns
- multi-horizon returns
- benchmark-relative returns

### Trend / momentum
- SMA
- EMA
- MACD
- RSI
- stochastic if justified
- momentum windows

### Volatility / range
- rolling volatility
- ATR
- true range
- Parkinson volatility if useful
- realized volatility windows

### Liquidity / trading activity proxies
- rolling volume
- turnover proxies if possible
- dollar volume if possible
- volume shocks / z-scores

### Drawdown features
- rolling max
- drawdown from rolling peak
- max drawdown windows

### Market-relative context features
- beta-style rolling relation to benchmark if practical
- benchmark-relative performance
- sector ETF relative signals if practical

You should choose a strong but not absurdly bloated feature set.

## Phase D — Quality controls
Include:
- missing data analysis
- ticker coverage analysis
- earliest/latest data per ticker
- corporate-action sanity checks
- duplicates detection
- asset-level data completeness summaries

## Phase E — Output tables
Produce strong reusable outputs for later model stages.

---

# 5. IMPORTANT DESIGN RULES

## Rule 1 — No leakage
Do **not** create features using future data.
All features must be strictly backward-looking.

## Rule 2 — Daily frequency
The default working frequency should be:
- **daily**

Do not move into intraday unless explicitly told.

## Rule 3 — U.S.-focused universe
This market data must be suitable for later alignment with U.S. issuer SEC-based data.

## Rule 4 — Reproducible raw layer
Keep raw downloaded data separate from processed engineered outputs.

## Rule 5 — Resume/incremental support
The pipeline must be robust:
- do not force destructive reruns
- support resuming
- skip already completed assets if possible
- maintain logs/progress

---

# 6. STORAGE / OUTPUT PHILOSOPHY

Use the environment / folder structure I provide, but follow this design principle:

## Raw layer
Should preserve:
- downloaded ticker-level raw files
- raw metadata
- raw benchmark files

## Processed layer
Should contain:
- ticker master / asset master
- daily OHLCV clean panel
- corporate actions clean tables
- engineered feature tables
- quality reports / summaries

Preferred formats:
- CSV is acceptable
- Parquet can be proposed if strongly justified for performance
- always explain format choices

---

# 7. WHAT THIS CHAT MUST DELIVER

I want this chat to help build **Linux-ready optimized Python code** that does as much useful work as possible in each long run.

Every script should:
- be optimized
- parallelized where appropriate
- show progress
- be robust
- support resume/incrementality where useful
- separate raw and processed outputs
- be suitable for large runs

The code should be:
- Python 3.12+
- Linux-oriented
- production-style
- strongly commented
- typed where useful
- memory-conscious
- streaming / partition-aware if needed

---

# 8. WHAT OUTPUTS I ULTIMATELY WANT FROM THIS CHAT

At minimum, I want this chat to help create:

1. **asset master / ticker universe file**
2. **raw downloaded market data archive**
3. **clean daily OHLCV panel**
4. **corporate actions table**
5. **engineered technical feature tables**
6. **quality audit tables**
7. ideally also outputs that later help the cross-asset relation engineer, such as:
   - returns panel
   - benchmark-relative panel
   - consistent ticker/date matrix-friendly outputs

---

# 9. HOW THIS MODULE CONNECTS TO THE FULL PROJECT

This market data module is later used for:
- technical encoder
- volatility model
- drawdown model
- historical VaR/CVaR
- liquidity model
- regime model inputs
- cross-asset relation graph construction

So the outputs must be designed with those downstream uses in mind.

---

# 10. WHAT NOT TO DO

- Do not work on SEC companyfacts or filing text in this chat.
- Do not assume we only need raw prices; we need engineered outputs too.
- Do not generate toy scripts unless explicitly asked.
- Do not ignore progress reporting.
- Do not ignore resume support.
- Do not silently make design choices that introduce future-data leakage.

---

# 11. IMMEDIATE TASK

Start by doing this:

1. propose the **full market-data acquisition and engineering plan**
2. define the **ticker universe strategy** for 2000–2024
3. define the **raw + processed folder layout**
4. define the **feature set**
5. then generate the first **optimized Linux-ready Python pipeline file** for this module

That first code file should aim to do as much of the market-data acquisition + standardization + initial engineering as reasonably possible in one run.

Continue from there as my dedicated market-data engineering partner.