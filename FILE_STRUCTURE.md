# FILE STRUCTURE:
prd@smartforum:~/fin-glassbox$ tree -L 2
.
├── 1ignore.txt
├── activating_linux_venv.txt
├── code
│   ├── analysts
│   ├── config
│   ├── daily_inference.py
│   ├── encoders
│   ├── fusion
│   ├── gnn
│   ├── hyperparameter_search.py
│   ├── ModelCodeRequirements.md
│   ├── riskEngine
│   ├── train_all.py
│   ├── train_analyst.py
│   ├── train_encoder.py
│   ├── train_risk.py
│   └── yfinance_ib
├── data
│   ├── CLUSTER.md
│   ├── convert_parquet_to_csv.py
│   ├── DATA.md
│   ├── FRED_data
│   ├── fred_regime.ipynb
│   ├── graphs
│   ├── LB_DataPrompt.md
│   ├── market_dates_ONLY_NYSE.csv
│   ├── SB_DataPrompt.md
│   ├── sec_data_cleaning_pipeline.py
│   ├── sec_edgar
│   ├── sec_filings_core_pipeline.py
│   ├── sec_filings_text_pipeline.py
│   ├── sec_filings_text_pipeline_v2.py
│   ├── sec_filings_topup_manifest.py
│   ├── sec_fundamentals.ipynb
│   ├── tickerList_total.csv
│   ├── yFinance
│   ├── yfinance1.py
│   ├── yfin_build_complete_panel.py
│   ├── yfin_build_master_panel.py
│   ├── yfin_dataFilling_pipeline.py
│   ├── yfin_engineer_features.py
│   ├── yfin_extracter.py
│   ├── yfin_fill_from_kaggle.py
│   ├── yfin_merge_sources.py
│   └── yfin_standardize_sources.py
├── final
│   ├── filings_finbert_chunks_balanced_25y_cap40000.csv
│   ├── filings_finbert_chunks_balanced_25y_quick.csv
│   └── finbert_25y_cap40000_metadata.json
├── gnn_ignore.txt
├── ignore_no_performance.sh
├── ignore_perf_max.sh
├── ignore_performance.sh
├── ignore.py
├── ignore_un_perf_max.sh
├── LICENSE
├── outputs
│   ├── codeResults
│   ├── embeddings
│   ├── models
│   └── results
├── README.md
├── requirements_linux_venv.txt
├── researchPapers
│   ├── 10. Explainable Artificial Intelligence Credit Risk Assessment using Machine Learning.pdf
│   ├── 11. An Explainable AI for Stock Market Prediction A Machine Learning Approach with XAI and Deep Neural Networks.pdf
│   ├── 12. A Survey of Explainable Artificial Intelligence (XAI) in Financial Time Series Forecasting.pdf
│   ├── 13. Financial asset price prediction with graph neural network-based temporal deep learning models.pdf
│   ├── 1. TradingAgents Multi-Agents LLM Financial Trading Framework.pdf
│   ├── 2. A Multi-Agent Decision Support System for Stock Trading.pdf
│   ├── 3. Model-agnostic explainable artificial intelligence methods in finance a systematic review, recent developments, limitations, challenges and future directions.pdf
│   ├── 4. Explainable artificial intelligence (XAI) in finance a systematic literature review.pdf
│   ├── 5. GNNExplainer Generating Explanations for Graph Neural Networks.pdf│   ├── 6. A Comprehensive Review on Financial Explainable AI.pdf
│   ├── 7. Big Data and Machine Learning Methods in Financial Risk Prediction.pdf
│   ├── 8. Fraud Detection using Machine Learning and Deep Learning.pdf
│   ├── 9. Explainable AI (XAI) in Deep Learning-Based Financial Forecasting Improving Model Transparency and Investor Trust.pdf
│   ├── final-workflow1.png
│   ├── final-workflow2.drawio.png
│   ├── Hyperparameter_Config.md
│   ├── i232649_i232591_i232626_DL_Assignment2.pdf
│   ├── inDepth-workflow.png
│   ├── literature_review_finrisk_sabeel.pdf
│   ├── LiteratureReview.xlsx
│   ├── lubabah idea paper.docx
│   ├── MASTER_PROMPT.md
│   ├── ProjectProposal.pdf
│   ├── proj_info.txt
│   ├── ProposalPresentation.pptx
│   ├── sabeels generated pipeline.pdf
│   ├── StemGNN.pdf
│   ├── UpdatedWorkflow.md
│   ├── workflow.drawio
│   ├── WORKFLOW.md
│   └── XAI_Specifications.md
└── venv3.12.7
    ├── bin
    ├── etc
    ├── include
    ├── lib
    ├── lib64 -> lib
    ├── pyvenv.cfg
    └── share

27 directories, 76 files
prd@smartforum:~/fin-glassbox/code$ tree -L 3
.
├── analysts
│   ├── news_analyst.py
│   ├── __pycache__
│   │   └── sentiment_analyst.cpython-312.pyc
│   ├── sentiment_analyst.py
│   ├── technical_analyst.py
│   ├── text_embedding_loader.py
│   ├── text_market_label_builder.py
│   └── TextualAnalysts.md
├── config
│   ├── hyperparameters.yaml
│   ├── regularization.yaml
│   └── training_chunks.yaml
├── daily_inference.py
├── encoders
│   ├── finbert_encoder.py
│   ├── __pycache__
│   │   ├── finbert_encoder.cpython-312.pyc
│   │   ├── run_finbert_full_pipeline.cpython-312.pyc
│   │   ├── run_finbert_resume_after_hpo.cpython-312.pyc
│   │   └── temporal_encoder.cpython-312.pyc
│   ├── run_finbert_full_pipeline.py
│   ├── run_finbert_resume_after_hpo.py
│   ├── TemporalEncoder_PreImplementation.md
│   ├── temporal_encoder.py
│   └── TextEncoder.md
├── fusion
│   └── fusion_engine.py
├── gnn
│   ├── build_cross_asset_graph.py
│   ├── config_gnn.py
│   ├── GNN_Pre_Specifications.md
│   ├── graph_utils.py
│   ├── __init__.py
│   ├── mtgnn_regime.py
│   ├── __pycache__
│   │   ├── stemgnn_base_model.cpython-312.pyc
│   │   ├── stemgnn_contagion.cpython-312.pyc
│   │   ├── stemgnn_forecast_dataloader.cpython-312.pyc
│   │   ├── stemgnn_handler.cpython-312.pyc
│   │   └── stemgnn_utils.cpython-312.pyc
│   ├── README_RUN_STEMGNN.md
│   ├── run_regime_detection.py
│   ├── stemgnn_base_model.py
│   ├── stemgnn_contagion.py
│   ├── stemgnn_forecast_dataloader.py
│   ├── stemgnn_handler.py
│   ├── StemGNN.md
│   ├── stemgnn_utils.py
│   └── train_contagion_gnn.py
├── hyperparameter_search.py
├── ModelCodeRequirements.md
├── riskEngine
│   ├── contagion_gnn.py
│   ├── drawdown.py
│   ├── position_sizing.py
│   ├── regime_gnn.py
│   ├── VaR_CVaR_Liquidity.md
│   ├── var_cvar_liquidity.py
│   └── volatility.py
├── train_all.py
├── train_analyst.py
├── train_encoder.py
├── train_risk.py
└── yfinance_ib
    ├── base.py
    ├── cache.py
    ├── const.py
    ├── data.py
    ├── exceptions.py
    ├── __init__.py
    ├── multi.py
    ├── README.md
    ├── scrapers
    │   ├── analysis.py
    │   ├── fundamentals.py
    │   ├── history_old.py
    │   ├── history.py
    │   ├── holders.py
    │   ├── __init__.py
    │   └── quote.py
    ├── shared.py
    ├── ticker.py
    ├── tickers.py
    ├── utils.py
    └── version.py

11 directories, 75 files
prd@smartforum:~/fin-glassbox/data$ tree -L 3
.
├── CLUSTER.md
├── convert_parquet_to_csv.py
├── DATA.md
├── FRED_data
│   ├── fred_daily_series_list.csv
│   ├── fred_daily_series_list_unfiltered.csv
│   ├── outputs
│   │   ├── cleaning_metadata.json
│   │   ├── column_summary.csv
│   │   ├── grand_table_raw.csv
│   │   ├── macro_all_features.csv
│   │   ├── macro_features_trading_days_clean.csv
│   │   └── pipeline_metadata.json
│   ├── qualifying_fred_series.json
│   └── raw
│       ├── 4BIGEURORECD.csv
│       ├── 4BIGEURORECDM.csv
│       ├── 4BIGEURORECDP.csv
│       ├── AAA10Y.csv
│       ├── AAAFF.csv
│       ├── AB1020AAAMT.csv
│       ...........
│       └── NASDAQNQCADIVT.csv
├── fred_regime.ipynb
├── graphs
│   ├── combined
│   │   └── sample_graph.pkl
│   ├── CrossAssetRelationData.md
│   ├── metadata
│   │   ├── betas.csv
│   │   ├── market_cap_proxy.csv
│   │   ├── sector_map.csv
│   │   ├── sector_similarity.csv
│   │   └── ticker_universe.csv
│   ├── returns
│   │   └── returns_matrix.csv
│   ├── snapshots
│   │   ├── edges_2000-01-04.csv
│   │   ├── edges_2000-02-02.csv
│   │   ├── edges_2000-03-02.csv
│   │   ├── edges_2000-03-30.csv
│   │   ├── edges_2000-04-28.csv
│   │   ....................
│   │   └── edges_2024-10-23.csv
│   └── static
│       ├── edges.csv
│       └── nodes.csv
├── LB_DataPrompt.md
├── market_dates_ONLY_NYSE.csv
├── SB_DataPrompt.md
├── sec_data_cleaning_pipeline.py
├── sec_edgar
│   ├── existing_23y
│   │   ├── filings_balance_summary.json
│   │   ├── filings_finbert_chunks_balanced_23y.csv
│   │   ├── filings_finbert_summary.json
│   │   └── filings_year_coverage.csv
│   ├── logs
│   │   ├── config
│   │   ├── github_fin_glassbox
│   │   └── github_fin_glassbox.pub
│   ├── processed
│   │   ├── cleaned
│   │   ├── DataProcessing.md
│   │   ├── filings_text
│   │   ├── fundamentals
│   │   └── issuer_master
│   └── raw
│       └── filings_txt
├── sec_filings_core_pipeline.py
├── sec_filings_text_pipeline.py
├── sec_filings_text_pipeline_v2.py
├── sec_filings_topup_manifest.py
├── sec_fundamentals.ipynb
├── tickerList_total.csv
├── yFinance
│   ├── processed
│   │   ├── common_tickers.csv
│   │   ├── features_temporal.csv
│   │   ├── liquidity_features.csv
│   │   ├── ohlcv_final.csv
│   │   ├── original_coverage.csv
│   │   ├── returns_long.csv
│   │   └── returns_panel_wide.csv
│   └── yFinance.md
├── yfinance1.py
├── yfin_build_complete_panel.py
├── yfin_build_master_panel.py
├── yfin_dataFilling_pipeline.py
├── yfin_engineer_features.py
├── yfin_extracter.py
├── yfin_fill_from_kaggle.py
├── yfin_merge_sources.py
└── yfin_standardize_sources.py

22 directories, 840 files
prd@smartforum:~/fin-glassbox/outputs$ tree -L 2
.
├── codeResults
│   ├── analysts
│   ├── FinBERT
│   ├── StemGNN
│   └── TemporalEncoder
├── embeddings
│   ├── analysts
│   ├── FinBERT
│   └── TemporalEncoder
├── models
│   ├── analysts
│   ├── FinBERT
│   ├── StemGNN
│   └── TemporalEncoder
└── results
    ├── analysts
    ├── FinBERT
    └── TemporalEncoder

18 directories, 0 files
prd@smartforum:~/fin-glassbox/outputs$ tree -L 3
.
├── codeResults
│   ├── analysts
│   │   ├── labels
│   │   ├── news
│   │   └── sentiment
│   ├── FinBERT
│   │   ├── finbert_config_resolved.json
│   │   ├── finbert_full_pipeline_20260425_024330.txt
│   │   ├── finbert_full_training_config.json
│   │   ├── finbert_resume_20260425_065122.log
│   │   ├── finbert_resume_tmux_20260425_161859.log
│   │   ├── finbert_resume_training_config.json
│   │   └── hpo
│   ├── StemGNN
│   │   └── hpo.db
│   └── TemporalEncoder
│       └── hpo
├── embeddings
│   ├── analysts
│   │   ├── news
│   │   └── sentiment
│   ├── FinBERT
│   │   ├── chunk1_test_embeddings.npy
│   │   ├── chunk1_test_manifest.json
│   │   ├── chunk1_test_metadata.csv
│   │   ├── chunk1_train_embeddings.npy
│   │   ├── chunk1_train_manifest.json
│   │   ├── chunk1_train_metadata.csv
│   │   ├── chunk1_val_embeddings.npy
│   │   ├── chunk1_val_manifest.json
│   │   ├── chunk1_val_metadata.csv
│   │   ├── chunk2_test_embeddings.npy
│   │   ├── chunk2_test_manifest.json
│   │   ├── chunk2_test_metadata.csv
│   │   ├── chunk2_train_embeddings.npy
│   │   ├── chunk2_train_manifest.json
│   │   ├── chunk2_train_metadata.csv
│   │   ├── chunk2_val_embeddings.npy
│   │   ├── chunk2_val_manifest.json
│   │   ├── chunk2_val_metadata.csv
│   │   ├── chunk3_pca_768_to_256.pkl
│   │   ├── chunk3_pca_manifest.json
│   │   ├── chunk3_test_embeddings768.npy
│   │   ├── chunk3_test_embeddings.npy
│   │   ├── chunk3_test_manifest768.json
│   │   ├── chunk3_test_manifest.json
│   │   ├── chunk3_test_metadata.csv
│   │   ├── chunk3_train_embeddings768.npy
│   │   ├── chunk3_train_embeddings.npy
│   │   ├── chunk3_train_manifest768.json
│   │   ├── chunk3_train_manifest.json
│   │   ├── chunk3_train_metadata.csv
│   │   ├── chunk3_val_embeddings768.npy
│   │   ├── chunk3_val_embeddings.npy
│   │   ├── chunk3_val_manifest768.json
│   │   ├── chunk3_val_manifest.json
│   │   └── chunk3_val_metadata.csv
│   └── TemporalEncoder
│       ├── chunk1_test_embeddings.npy
│       ├── chunk1_train_embeddings.npy
│       └── chunk1_val_embeddings.npy
├── models
│   ├── analysts
│   │   ├── news
│   │   └── sentiment
│   ├── FinBERT
│   │   ├── chunk1
│   │   ├── chunk2
│   │   ├── chunk3
│   │   └── hpo
│   ├── StemGNN
│   │   ├── chunk1
│   │   └── smoke
│   └── TemporalEncoder
│       ├── chunk1
│       └── chunk2
└── results
    ├── analysts
    │   ├── labels
    │   ├── news
    │   └── sentiment
    ├── FinBERT
    │   ├── chunk1_mlm_history.csv
    │   ├── chunk2_mlm_history.csv
    │   ├── chunk3_mlm_history.csv
    │   └── hpo
    └── TemporalEncoder
        └── xai

40 directories, 48 files
prd@smartforum:~/fin-glassbox/outputs$ 