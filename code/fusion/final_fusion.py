#!/usr/bin/env python3
"""
code/fusion/fnal_fusion.py

Hybrid Final Fusion CLI
=======================

This file is intentionally named `fnal_fusion.py` because the user requested
that exact filename.

It wraps:
    code/fusion/fusion_layer.py

Architecture:
    Learned Fusion Layer
        +
    User Rule Barrier

CLI:
    inspect
    smoke
    hpo
    train-best
    predict
    predict-all
    validate
    run
    run-all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import torch

# _THIS_DIR = Path(__file__).resolve().parent
# if str(_THIS_DIR) not in sys.path:
#     sys.path.insert(0, str(_THIS_DIR))
# from stemgnn_base_model import Model as StemGNNBase
# Allow direct execution from repo root without requiring package installation.
# THIS_FILE = Path(__file__).resolve()
REPO_ROOT_GUESS = Path(__file__).resolve().parent
if str(REPO_ROOT_GUESS) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_GUESS))

from fusion_layer import (  # noqa: E402
    FusionConfig,
    UserRuleBarrierConfig,
    apply_best_params,
    inspect_inputs,
    load_best_params,
    predict_fusion,
    run_hpo,
    smoke_test,
    train_fusion_model,
    validate_predictions,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hybrid Learned + Rule-Based Final Fusion Engine"
    )

    sub = parser.add_subparsers(dest="command")

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--repo-root", type=str, default="")
        p.add_argument("--device", type=str, default="cuda")

        p.add_argument("--batch-size", type=int, default=None)
        p.add_argument("--epochs", type=int, default=None)
        p.add_argument("--lr", type=float, default=None)
        p.add_argument("--num-workers", type=int, default=None)
        p.add_argument("--max-train-rows", type=int, default=None)

        p.add_argument("--exposure-mode", type=str, default="moderate", choices=["conservative", "moderate", "aggressive"])
        p.add_argument("--horizon-mode", type=str, default="short", choices=["short", "long"])

        # User rule barrier overrides.
        p.add_argument("--max-single-stock-default", type=float, default=None)
        p.add_argument("--crisis-short-cap", type=float, default=None)
        p.add_argument("--crisis-long-cap", type=float, default=None)
        p.add_argument("--min-liquidity-score", type=float, default=None)
        p.add_argument("--contagion-veto", type=float, default=None)
        p.add_argument("--drawdown-cap-threshold", type=float, default=None)
        p.add_argument("--quant-risk-buy-veto", type=float, default=None)
        p.add_argument("--severe-risk-sell", type=float, default=None)
        p.add_argument("--buy-signal-threshold", type=float, default=None)
        p.add_argument("--sell-signal-threshold", type=float, default=None)
        p.add_argument("--min-confidence-for-buy", type=float, default=None)
        p.add_argument("--allow-missing-qualitative", action="store_true")
        p.add_argument("--no-strict-quant-schema", action="store_true")

    p = sub.add_parser("inspect")
    add_common(p)

    p = sub.add_parser("smoke")
    add_common(p)

    p = sub.add_parser("hpo")
    p.add_argument("--chunk", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--fresh", action="store_true")
    add_common(p)

    p = sub.add_parser("train-best")
    p.add_argument("--chunk", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--fresh", action="store_true")
    add_common(p)

    p = sub.add_parser("predict")
    p.add_argument("--chunk", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--split", type=str, required=True, choices=["train", "val", "test"])
    add_common(p)

    p = sub.add_parser("predict-all")
    p.add_argument("--chunks", type=int, nargs="+", default=[1, 2, 3], choices=[1, 2, 3])
    p.add_argument("--splits", type=str, nargs="+", default=["val", "test"], choices=["train", "val", "test"])
    add_common(p)

    p = sub.add_parser("validate")
    p.add_argument("--chunk", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--split", type=str, required=True, choices=["train", "val", "test"])
    add_common(p)

    p = sub.add_parser("run")
    p.add_argument("--chunk", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--predict-splits", type=str, nargs="+", default=["val", "test"], choices=["train", "val", "test"])
    add_common(p)

    p = sub.add_parser("run-all")
    p.add_argument("--chunks", type=int, nargs="+", default=[1, 2, 3], choices=[1, 2, 3])
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--predict-splits", type=str, nargs="+", default=["val", "test"], choices=["train", "val", "test"])
    add_common(p)

    return parser


def config_from_args(args: argparse.Namespace) -> FusionConfig:
    rule = UserRuleBarrierConfig()

    if getattr(args, "max_single_stock_default", None) is not None:
        rule.max_single_stock_default = float(args.max_single_stock_default)

    if getattr(args, "crisis_short_cap", None) is not None:
        rule.crisis_short_horizon_cap = float(args.crisis_short_cap)

    if getattr(args, "crisis_long_cap", None) is not None:
        rule.crisis_long_horizon_cap = float(args.crisis_long_cap)

    if getattr(args, "min_liquidity_score", None) is not None:
        rule.min_liquidity_score = float(args.min_liquidity_score)

    if getattr(args, "contagion_veto", None) is not None:
        rule.contagion_veto_threshold = float(args.contagion_veto)

    if getattr(args, "drawdown_cap_threshold", None) is not None:
        rule.drawdown_cap_threshold = float(args.drawdown_cap_threshold)

    if getattr(args, "quant_risk_buy_veto", None) is not None:
        rule.quant_risk_buy_veto_threshold = float(args.quant_risk_buy_veto)

    if getattr(args, "severe_risk_sell", None) is not None:
        rule.severe_risk_sell_threshold = float(args.severe_risk_sell)

    if getattr(args, "buy_signal_threshold", None) is not None:
        rule.buy_signal_threshold = float(args.buy_signal_threshold)

    if getattr(args, "sell_signal_threshold", None) is not None:
        rule.sell_signal_threshold = float(args.sell_signal_threshold)

    if getattr(args, "min_confidence_for_buy", None) is not None:
        rule.min_confidence_for_buy = float(args.min_confidence_for_buy)

    config = FusionConfig(rule_config=rule)

    if getattr(args, "repo_root", ""):
        config.repo_root = args.repo_root

    config.device = getattr(args, "device", "cuda")
    if config.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        config.device = "cpu"

    if getattr(args, "batch_size", None) is not None:
        config.batch_size = int(args.batch_size)

    if getattr(args, "epochs", None) is not None:
        config.epochs = int(args.epochs)

    if getattr(args, "lr", None) is not None:
        config.lr = float(args.lr)

    if getattr(args, "num_workers", None) is not None:
        config.num_workers = int(args.num_workers)

    if getattr(args, "max_train_rows", None) is not None:
        config.max_train_rows = int(args.max_train_rows)

    config.exposure_mode = getattr(args, "exposure_mode", "moderate")
    config.horizon_mode = getattr(args, "horizon_mode", "short")

    # Default should be permissive for qualitative because qualitative data is sparse.
    config.allow_missing_qualitative = True
    if getattr(args, "allow_missing_qualitative", False):
        config.allow_missing_qualitative = True

    config.strict_quantitative_attention_schema = not bool(getattr(args, "no_strict_quant_schema", False))

    return config.resolve_paths()


def cmd_hpo(config: FusionConfig, chunk: int, trials: int, fresh: bool) -> None:
    print("=" * 90)
    print(f"FUSION ENGINE HPO — chunk{chunk} ({trials} trials)")
    print("=" * 90)
    run_hpo(config, chunk, trials, fresh=fresh)


def cmd_train_best(config: FusionConfig, chunk: int, fresh: bool) -> None:
    config.resolve_paths()

    best = load_best_params(config, chunk)
    if best is not None:
        print(f"Loaded best Fusion params for chunk{chunk}: {best}")
        config = apply_best_params(config, best)
    else:
        print(f"No Fusion HPO params found for chunk{chunk}; using default config.")

    print("=" * 90)
    print(f"FUSION ENGINE TRAINING — chunk{chunk}")
    print("=" * 90)

    _, best_val, _ = train_fusion_model(config, chunk, fresh=fresh, hpo_mode=False)
    print(f"  Complete. Best val loss: {best_val:.6f}")


def cmd_predict_all(config: FusionConfig, chunks: List[int], splits: List[str]) -> None:
    for c in chunks:
        for s in splits:
            print("=" * 90)
            print(f"FUSION ENGINE PREDICT — chunk{c}_{s}")
            print("=" * 90)
            predict_fusion(config, c, s)


def cmd_run(config: FusionConfig, chunk: int, trials: int, fresh: bool, predict_splits: List[str]) -> None:
    cmd_hpo(config, chunk, trials, fresh)
    cmd_train_best(config, chunk, fresh)
    cmd_predict_all(config, [chunk], predict_splits)
    for s in predict_splits:
        validate_predictions(config, chunk, s)


def cmd_run_all(config: FusionConfig, chunks: List[int], trials: int, fresh: bool, predict_splits: List[str]) -> None:
    for chunk in chunks:
        cmd_run(config, chunk, trials, fresh, predict_splits)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    config = config_from_args(args)

    if args.command == "inspect":
        inspect_inputs(config)

    elif args.command == "smoke":
        smoke_test(config)

    elif args.command == "hpo":
        cmd_hpo(config, args.chunk, args.trials, args.fresh)

    elif args.command == "train-best":
        cmd_train_best(config, args.chunk, args.fresh)

    elif args.command == "predict":
        print("=" * 90)
        print(f"FUSION ENGINE PREDICT — chunk{args.chunk}_{args.split}")
        print("=" * 90)
        predict_fusion(config, args.chunk, args.split)

    elif args.command == "predict-all":
        cmd_predict_all(config, args.chunks, args.splits)

    elif args.command == "validate":
        validate_predictions(config, args.chunk, args.split)

    elif args.command == "run":
        cmd_run(config, args.chunk, args.trials, args.fresh, args.predict_splits)

    elif args.command == "run-all":
        cmd_run_all(config, args.chunks, args.trials, args.fresh, args.predict_splits)


if __name__ == "__main__":
    main()


# ── Run instructions ─────────────────────────────────────────────────────────
# Compile:
# python -m py_compile code/fusion/fusion_layer.py code/fusion/final_fusion.py
#
# Inspect:
# python code/fusion/final_fusion.py inspect --repo-root .
#
# Smoke:
# python code/fusion/final_fusion.py smoke --repo-root . --device cuda
#
# Chunk 1 sanity:
# python code/fusion/final_fusion.py hpo --repo-root . --chunk 1 --trials 3 --device cuda --fresh
# python code/fusion/final_fusion.py train-best --repo-root . --chunk 1 --device cuda --fresh
# python code/fusion/final_fusion.py predict --repo-root . --chunk 1 --split test --device cuda
# python code/fusion/final_fusion.py validate --repo-root . --chunk 1 --split test
#
# Chunk 1 full:
# python code/fusion/final_fusion.py run --repo-root . --chunk 1 --trials 30 --device cuda --fresh --predict-splits val test
#
# Full run after all Quantitative chunks have trained attention schema and train outputs:
# python code/fusion/final_fusion.py run-all --repo-root . --chunks 1 2 3 --trials 30 --device cuda --fresh --predict-splits val test