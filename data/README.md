# Data Collection, Processing, Cleaning, and Engineering Methodology

The project began with five distinct data families, each corresponding to a specific modality or modelling requirement within the Explainable Distributed Deep Learning Framework for Financial Risk Management. These data families were not collected randomly; each one was selected because it served a defined role in the final system architecture. Together, they allowed the framework to combine market behaviour, corporate disclosures, macroeconomic conditions, and cross-asset dependencies into a unified risk-aware modelling pipeline.

The five data families were:

1. **Tabular fundamentals data**
2. **SEC textual filings data**
3. **Macro/regime data**
4. **Time-series market data**
5. **Cross-asset relation/graph data**

Each family required a different acquisition strategy, cleaning process, engineering pipeline, and quality-control method. The complete data preparation process was therefore not a single download step, but a multi-stage engineering workflow involving large raw archives, API-based retrieval, local storage constraints, multi-machine transfer, repeated reruns, targeted top-ups, and careful alignment to avoid lookahead bias.

---

## 1. Overview of the Five Data Families

### 1.1 Tabular Fundamentals Data

The first structured data family consisted of company-level fundamentals such as revenue, net income, assets, liabilities, equity, margins, leverage indicators, cash-flow values, and derived profitability ratios. This data was intended to represent the internal financial condition of firms.

The main source was **SEC EDGAR**, specifically the bulk `companyfacts.zip` and `submissions.zip` archives. The `companyfacts.zip` archive contained XBRL-tagged financial facts for SEC filers, while `submissions.zip` contained company identifiers, CIK records, ticker information, entity metadata, and filing histories. The final cleaned fundamentals pipeline produced point-in-time quarterly fundamentals and derived features, including a cleaned CIK-to-ticker mapping and a cleaned issuer master table. 

This stage was one of the earliest and most difficult parts of the project. The raw SEC archives were very large: `companyfacts.zip` contained roughly 17–20k JSON files and occupied about 17 GB, while `submissions.zip` contained around 900k–970k JSON files and occupied about 5 GB. Extracting and processing these files on a 12 GB RAM machine was extremely slow. The extraction alone took almost a full day, and the full fundamentals pipeline required approximately 2.5 days.

---

### 1.2 SEC Textual Filings Data

The second data family consisted of textual SEC filings. This data was required for the text stream of the framework, especially for financial language understanding, sentiment-style analysis, news/event interpretation, and later integration with the FinBERT-based encoder.

A correction is important here: **SEC EDGAR does not provide “news sentiment” directly.** Instead, it provides official company filings and disclosure documents. In this project, SEC filings served as the primary financial text source because they contain management discussion, risk factors, business descriptions, governance disclosures, legal events, liquidity warnings, and other company-level narrative information.

The SEC text pipeline focused on raw filing documents such as:

* `10-K`
* `10-Q`
* `8-K`
* `DEF 14A`

The text pipeline was designed around a partial but carefully engineered corpus because downloading the complete SEC filing universe was not feasible under available storage and time constraints. The text engineering script contained stages for inventory, cleaning, section extraction, quality reporting, dataset balancing, and FinBERT-ready chunk generation. 

This dataset took around **4–5 days** to collect and process. The initial download of about 98,000 files consumed most of an 80 GB HDD, leaving only around 17 GB free, while the files were still concentrated mostly around the year 2000. The data was then transferred by SSH to a spare laptop with a 512 GB HDD. Multiple year-range downloads were launched to improve coverage across the full 2000–2024 period. Eventually, around 288k–300k+ raw filings were collected, occupying approximately 304.8 GB. During this stage, the system repeatedly encountered storage exhaustion errors.

A major issue occurred when the pipeline reported only **23 usable years** under the balancing rule. This did not mean only 23 years existed; rather, it meant two years did not meet the minimum document threshold for balanced modelling. A targeted top-up script was therefore created to detect weak years and generate additional download manifests without fabricating or duplicating data. 

---

### 1.3 Macro/Regime Data

The third data family consisted of U.S. macroeconomic and regime indicators. This data was required to capture the wider economic environment in which market movements occur. It included interest rates, inflation indicators, labour-market variables, credit spreads, volatility proxies, recession indicators, exchange rates, and other stress/regime markers.

The primary source was **FRED**. The initial specification for this module was given to a group member through a detailed prompt, which required raw acquisition, metadata preservation, transformation, daily alignment, missingness reporting, and leakage-aware processing. 

The pipeline evolved in stages. The first version used a manually selected set of 34 macro series. However, this produced sparse results because many macroeconomic series are monthly or quarterly. The workflow then expanded into a larger scan of available FRED files, filtering down to interpretable series with sufficient 2000–2024 coverage. The final cleaned macro/regime dataset contained **6,288 NYSE trading days**, covering **2000-01-03 to 2024-12-31**, with **49 final features** and no missing values after cleaning. 

The most important processing challenge in this dataset was frequency alignment. FRED series come in daily, weekly, monthly, and quarterly frequencies. The final processed table needed to align these features to daily market dates without using future information. Therefore, forward filling and release-aware alignment were used carefully so that macro values would only become available after their assumed release dates.

---

### 1.4 Time-Series Market Data

The time-series market dataset became the most important and central data family in the project. It was the main pillar for the technical stream, risk engine, liquidity module, VaR/CVaR calculations, drawdown modelling, volatility estimation, regime modelling, and cross-asset graph construction.

The required data consisted of:

* Open, high, low, close, and volume values
* Adjusted prices where available
* Daily returns and log returns
* Trading-volume and liquidity features
* Technical indicators such as RSI, MACD, Bollinger position, volatility, and price-range features
* Benchmark-relative features and SPY correlation

The market-data module was initially assigned to a group member using a detailed prompt. The group member produced the first yfinance-based pipeline, which downloaded raw Yahoo Finance OHLCV data, built a ticker universe from Wikipedia and GitHub sources, included benchmark/sector ETFs, and stored raw files as Parquet with metadata. 

However, the initial yfinance output did not fully match the SEC-linked ticker universe, so the pipeline had to be expanded and completed using additional sources. The final market dataset combined:

* yfinance/Yahoo Finance
* Stooq historical data
* Huge Market Dataset
* Kaggle NYSE/NASDAQ/NYSE-A/OTC 1962–2024 dataset
* NYSE trading calendar

The final history document records that the market pipeline evolved from a basic yfinance download into a multi-source engineering pipeline with source filtering, standardisation, adjusted/unadjusted price reconciliation, complete panel construction, filling, and feature engineering. 

A major technical issue was that different sources used different price conventions. Stooq prices were unadjusted, while Huge Market and yfinance prices were adjusted. To solve this, overlapping dates were used to compute a per-ticker median scaling ratio, allowing Stooq OHLC values to be scaled onto the adjusted-price scale before merging. 

The final market output contained **2,500 tickers** over **6,286 trading dates**, producing **15,715,000 OHLCV rows** with no missing values after the filling process. The engineered outputs included `returns_panel_wide.csv`, `returns_long.csv`, `liquidity_features.csv`, and `features_temporal.csv`. 

---

### 1.5 Cross-Asset Relation / Graph Data

The final data family was the cross-asset relation dataset. Unlike the other four data families, this was not directly downloaded from an external source. Instead, it was derived from the final market-data panel.

This data was required because financial assets do not move independently. A single-stock model can only analyse one company at a time, whereas a graph-based model can represent how shocks, correlations, sector movements, ETF-like flows, and market-wide stress propagate across assets.

The cross-asset relation specification defined several graph types, including correlation networks, sector graphs, ETF-style relationships, index membership graphs, and relationship vectors. 

The implemented graph dataset used the final 2,500-ticker market universe rather than the originally planned 4,428-ticker SEC universe, because the 2,500-ticker set had complete market coverage. The key input was `returns_panel_wide.csv`, containing 2,500 stocks and 6,285 daily log-return observations. 

Because live ETF holdings, yfinance sector calls, and market-cap API calls were impractical, the graph pipeline used derived alternatives:

* SIC-code-based sector mapping instead of yfinance sector metadata
* ETF return correlation instead of live ETF holdings
* Median dollar volume as a market-cap proxy
* Beta versus SPY from returns
* Rolling return correlations for dynamic graph snapshots

The final graph pipeline produced 313 rolling correlation snapshots, 8,578 static edges, 2,515 graph nodes, and a sample NetworkX graph with 105,545 edges. It also achieved 100% coverage for sector mapping, market-cap proxy values, and beta estimates. 

---

## 2. Why These Five Data Families Were Necessary

The project is multimodal by design. It does not rely on one monolithic dataset or one black-box model. Each data family contributes a different view of financial risk:

| Data Family             | Main Role in the System                                                        |
| ----------------------- | ------------------------------------------------------------------------------ |
| Tabular fundamentals    | Company financial condition and structured firm health indicators              |
| SEC textual filings     | Financial language, risk disclosures, management discussion, event context     |
| Macro/regime data       | Economic environment, policy conditions, stress regimes, recession context     |
| Market time-series data | Price behaviour, volatility, liquidity, drawdown, VaR/CVaR, technical features |
| Cross-asset graph data  | Asset dependencies, contagion pathways, sector/ETF similarity, systemic links  |

The market dataset became the strongest and most central pillar because it was the only family with complete daily alignment across 2,500 assets and 25 years. However, the other data families were still essential for building a defensible multimodal framework and for giving the system broader interpretability.

---

## 3. Start of the Pipeline: Fundamentals Data

The first major dataset engineered was the SEC fundamentals dataset. This stage began with the raw SEC EDGAR bulk archives. The goal was to transform deeply nested raw SEC JSON data into clean, tabular, point-in-time company fundamentals.

The process began with two raw archives:

* `submissions.zip`
* `companyfacts.zip`

The `submissions.zip` archive contained nearly one million JSON files, each corresponding to an SEC filer. These files contained CIK identifiers, company names, tickers, filing histories, former names, addresses, SIC codes, and other issuer-level metadata. The `companyfacts.zip` archive contained company-level XBRL-tagged financial observations. These observations were deeply nested by taxonomy, concept, unit, filing date, fiscal year, and fiscal period.

The first task was simply extraction and flattening. This was computationally difficult because the archive contained hundreds of thousands of small files. Extracting the submissions archive on a system with 12 GB RAM was extremely slow and took almost a full day. The next stage involved processing and flattening the extracted JSON files into structured CSV tables.

The SEC fundamentals pipeline then followed these major steps:

1. **Raw acquisition and archive extraction**
2. **Submissions flattening**
3. **Companyfacts flattening**
4. **Issuer master construction**
5. **U.S. issuer and NYSE/Nasdaq filtering**
6. **CIK-to-ticker mapping**
7. **Point-in-time fundamentals extraction**
8. **Derived ratio and growth feature engineering**
9. **Cleaning, deduplication, null-column removal, and outlier capping**

A key methodological decision was the use of **point-in-time extraction**. For each company and fiscal period, the pipeline selected the earliest filing date rather than later amendments or restatements. This prevented lookahead bias, because a historical model should only use the financial information that would have been available at that time. 

The final cleaned fundamentals dataset contained company-period rows with raw financial metrics and derived ratios. However, later analysis showed that the overlap between the final market universe and the fundamentals dataset was limited. This reduced the usefulness of the fundamentals stream for final modelling, but the data engineering process remained an important part of the project because it established SEC identifiers, ticker mappings, issuer filtering, and reusable company metadata.

---

## 4. Methodological Emphasis

The data processing phase should not be described as a simple “data download” process. It involved repeated engineering decisions under real constraints:

* limited local storage,
* slow HDD performance,
* RAM limitations,
* multi-day downloads,
* SSH transfers between machines,
* large raw-file counts,
* no-space-left errors,
* source mismatch problems,
* adjusted versus unadjusted price inconsistencies,
* sparse macroeconomic series,
* incomplete SEC text coverage,
* weak-year top-ups,
* and downstream alignment requirements.

The final methodology therefore emphasises both the **technical pipeline** and the **practical engineering constraints**. This is important because the quality of the final datasets came not only from selecting good sources, but also from making defensible decisions when the ideal full-scale dataset was not feasible.
