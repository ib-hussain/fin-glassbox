#!/usr/bin/env python3
"""
code/riskEngine/regime_gnn.py

Thin wrapper around code/gnn/mtgnn_regime.py.

Purpose:
    Keep the actual graph-learning implementation inside code/gnn/
    while exposing a risk-engine-facing entrypoint under code/riskEngine/.

Usage:
    python code/riskEngine/regime_gnn.py inspect --repo-root .
    python code/riskEngine/regime_gnn.py hpo --repo-root . --chunk 1 --trials 30 --device cuda --fresh
    python code/riskEngine/regime_gnn.py train-best --repo-root . --chunk 1 --device cuda --fresh
    python code/riskEngine/regime_gnn.py predict --repo-root . --chunk 1 --split test --device cuda
"""

from __future__ import annotations

import sys
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parents[2]
GNN_DIR = REPO_ROOT / "code" / "gnn"

if str(GNN_DIR) not in sys.path:
    sys.path.insert(0, str(GNN_DIR))

from mtgnn_regime import (  # noqa: E402,F401
    MTGNNRegimeConfig,
    MTGNNRegimeModel,
    RegimeSnapshotDataset,
    apply_hpo_params_if_available,
    build_graph_summary,
    cmd_inspect,
    cmd_smoke,
    predict_with_xai,
    run_hpo,
    train_regime_model,
    main,
)


if __name__ == "__main__":
    main()

'''
    python code/riskEngine/regime_gnn.py hpo --repo-root . --chunk 1 --trials 30 --device cuda --fresh --node-limit 2500
    python code/riskEngine/regime_gnn.py train-best --repo-root . --chunk 1 --device cuda --fresh --node-limit 2500
    python code/riskEngine/regime_gnn.py predict --repo-root . --chunk 1 --split val --device cuda --node-limit 2500
    python code/riskEngine/regime_gnn.py predict --repo-root . --chunk 1 --split test --device cuda --node-limit 2500

    python code/riskEngine/regime_gnn.py train-best --repo-root . --chunk 1 --device cuda --fresh  --node-limit 2500 && python code/riskEngine/regime_gnn.py predict --repo-root . --chunk 1 --split val --device cuda --node-limit 2500 && python code/riskEngine/regime_gnn.py predict --repo-root . --chunk 1 --split test --device cuda --node-limit 2500 

    python code/riskEngine/regime_gnn.py train-best --repo-root . --chunk 2 --device cuda --fresh  --node-limit 2500 && python code/riskEngine/regime_gnn.py predict --repo-root . --chunk 2 --split val --device cuda --node-limit 2500 && python code/riskEngine/regime_gnn.py predict --repo-root . --chunk 2 --split test --device cuda --node-limit 2500 
    
    python code/riskEngine/regime_gnn.py train-best --repo-root . --chunk 3 --device cuda --fresh  --node-limit 2500 && python code/riskEngine/regime_gnn.py predict --repo-root . --chunk 3 --split val --device cuda --node-limit 2500 && python code/riskEngine/regime_gnn.py predict --repo-root . --chunk 3 --split test --device cuda --node-limit 2500 
'''