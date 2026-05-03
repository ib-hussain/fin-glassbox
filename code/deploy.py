#!/usr/bin/env python3
"""
code/deploy.py

Streamlit deployment interface for fin-glassbox.

This file is intentionally thin: it imports and uses code/inference.py rather than
duplicating model-loading or inference logic.

Supported UI modes:
1. Historical replay:
   Reads existing fused FusionEngine outputs.

2. Frozen-cached inference:
   Loads cached prepared PositionSizing + Qualitative rows, then runs frozen
   QuantitativeAnalyst and FusionEngine checkpoints in memory.

3. Manual what-if inference:
   Lets the user edit key risk/technical/qualitative inputs and sends those
   values into the frozen inference path.

Run:
    streamlit run code/deploy.py -- --repo-root . --device cuda

Minimal local demo files:
    code/inference.py
    code/deploy.py
    code/fusion/fusion_layer.py
    code/analysts/quantitative_analyst.py
    code/analysts/qualitative_analyst.py
    outputs/models/QuantitativeAnalyst/chunk3/{best_model.pt,final_model.pt,scaler.npz}
    outputs/models/FusionEngine/chunk3/{best_model.pt,final_model.pt,scaler.npz}
    outputs/results/PositionSizing/position_sizing_chunk3_{val,test}.csv
    outputs/results/QualitativeAnalyst/qualitative_daily_chunk3_{val,test}.csv
    outputs/results/FusionEngine/fused_decisions_chunk3_{val,test}.csv  # needed for historical mode only

This app is for academic/research demonstration only. It is not financial advice.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# CLI arguments passed after: streamlit run code/deploy.py -- --repo-root .
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--default-chunk", type=int, default=3)
    parser.add_argument("--default-split", default="test")
    return parser.parse_known_args()[0]


ARGS = parse_args()
REPO_ROOT = Path(ARGS.repo_root).resolve()
CODE_DIR = REPO_ROOT / "code"

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

try:
    import streamlit as st
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Streamlit is required for code/deploy.py. Install it with: pip install streamlit"
    ) from exc

from inference import (  # noqa: E402
    ExplanationNarrator,
    FinGlassboxInferenceEngine,
    InferenceRuntimeConfig,
    fusion_csv_path,
    position_sizing_csv_path,
    json_safe,
)


# ---------------------------------------------------------------------------
# Cached resources / data
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_engine(repo_root: str, device: str, chunk: int, split: str, exposure_mode: str, horizon_mode: str, prefer_best: bool) -> FinGlassboxInferenceEngine:
    cfg = InferenceRuntimeConfig(
        repo_root=repo_root,
        device=device,
        chunk=int(chunk),
        split=str(split),
        exposure_mode=exposure_mode,
        horizon_mode=horizon_mode,
        prefer_final_model=not bool(prefer_best),
    )
    return FinGlassboxInferenceEngine(cfg)


@st.cache_resource(show_spinner=False)
def get_narrator(model_name: str, device: str, allow_download: bool) -> ExplanationNarrator:
    narrator = ExplanationNarrator(
        model_name=model_name,
        device=device,
        local_files_only=not bool(allow_download),
        max_new_tokens=240,
    )
    narrator.load()
    return narrator


@st.cache_data(show_spinner=False)
def load_picker_frame(repo_root: str, mode: str, chunk: int, split: str) -> pd.DataFrame:
    root = Path(repo_root)
    if mode == "Historical replay":
        path = fusion_csv_path(root, int(chunk), str(split))
    else:
        path = position_sizing_csv_path(root, int(chunk), str(split))

    if not path.exists():
        return pd.DataFrame(columns=["ticker", "date"])

    df = pd.read_csv(path, usecols=["ticker", "date"], dtype={"ticker": str}, low_memory=False)
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["ticker", "date"]).drop_duplicates(["ticker", "date"])
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def format_pct(value: Any) -> str:
    try:
        return f"{float(value):.3f}%"
    except Exception:
        return "n/a"


def format_num(value: Any, nd: int = 4) -> str:
    try:
        return f"{float(value):.{nd}f}"
    except Exception:
        return "n/a"


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def render_header() -> None:
    st.set_page_config(
        page_title="fin-glassbox",
        page_icon="🥀",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("fin-glassbox")
    st.caption("Explainable multimodal financial risk inference — research demonstration, not financial advice.")


def render_decision_cards(result: Dict[str, Any]) -> None:
    decision = result.get("decision", {})
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Recommendation", str(decision.get("final_recommendation", "UNKNOWN")))
    c2.metric("Confidence", format_num(decision.get("final_fusion_confidence"), 3))
    c3.metric("Risk score", format_num(decision.get("final_fusion_risk_score"), 3))
    c4.metric("Position", format_pct(decision.get("final_position_pct")))
    c5.metric("Top risk", str(decision.get("top_attention_risk_driver", "unknown")))

    st.subheader("User-facing explanation")
    st.info(result.get("human_explanation", "No explanation available."))


def render_risk_breakdown(result: Dict[str, Any]) -> None:
    row = result.get("full_row", {})
    decision = result.get("decision", {})
    risk_cols = {
        "Volatility": row.get("volatility_risk_score", decision.get("risk_attention_volatility")),
        "Drawdown": row.get("drawdown_risk_score", decision.get("risk_attention_drawdown")),
        "VaR/CVaR": row.get("var_cvar_risk_score"),
        "Contagion": row.get("contagion_risk_score"),
        "Liquidity": row.get("liquidity_risk_score"),
        "Regime": row.get("regime_risk_score"),
    }
    risk_df = pd.DataFrame(
        [{"risk_component": k, "score": safe_float(v, 0.0)} for k, v in risk_cols.items() if v is not None]
    )
    att_cols = {
        "Volatility": decision.get("risk_attention_volatility"),
        "Drawdown": decision.get("risk_attention_drawdown"),
        "VaR/CVaR": decision.get("risk_attention_var_cvar"),
        "Contagion": decision.get("risk_attention_contagion"),
        "Liquidity": decision.get("risk_attention_liquidity"),
        "Regime": decision.get("risk_attention_regime"),
    }
    att_df = pd.DataFrame(
        [{"risk_component": k, "attention_weight": safe_float(v, 0.0)} for k, v in att_cols.items() if v is not None]
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Risk scores")
        if not risk_df.empty:
            st.bar_chart(risk_df.set_index("risk_component"))
        else:
            st.write("No risk-score fields available.")
    with right:
        st.subheader("Quantitative risk attention")
        if not att_df.empty:
            st.bar_chart(att_df.set_index("risk_component"))
        else:
            st.write("No attention fields available.")


def render_transparency(result: Dict[str, Any], show_full: bool, show_modules: bool, show_raw: bool) -> None:
    if show_full:
        with st.expander("Full system decision JSON", expanded=False):
            st.json(json_safe(result.get("decision", {})))

    if show_modules:
        with st.expander("Model/module outputs", expanded=False):
            inter = result.get("intermediate_outputs", {})
            if inter:
                st.write("Intermediate outputs generated during frozen/manual inference:")
                st.json(json_safe(inter))
            else:
                row = result.get("full_row", {})
                module_keys = [
                    "quantitative_xai_summary",
                    "qualitative_xai_summary",
                    "fusion_xai_summary",
                    "technical_direction_score",
                    "trend_score",
                    "momentum_score",
                    "timing_confidence",
                    "volatility_risk_score",
                    "drawdown_risk_score",
                    "var_cvar_risk_score",
                    "contagion_risk_score",
                    "liquidity_risk_score",
                    "regime_risk_score",
                    "regime_label",
                    "size_reduction_reasons",
                    "rule_barrier_reasons",
                ]
                st.json({k: row.get(k) for k in module_keys if k in row})

    if show_raw:
        with st.expander("Raw result object", expanded=False):
            st.json(json_safe(result))


def render_source_details(result: Dict[str, Any]) -> None:
    sources = result.get("runtime_sources") or {}
    source_file = result.get("source_file")
    if sources or source_file:
        with st.expander("Runtime/source files", expanded=False):
            if source_file:
                st.write({"source_file": source_file})
            if sources:
                st.json(json_safe(sources))


def apply_llm_narrator(result: Dict[str, Any], model_name: str, device: str, allow_download: bool) -> Dict[str, Any]:
    if not model_name.strip():
        return result
    narrator = get_narrator(model_name.strip(), device, allow_download)
    result = dict(result)
    result["human_explanation"] = narrator.narrate(result.get("decision", {}))
    result["llm_narrator"] = {
        "model": model_name.strip(),
        "loaded": bool(narrator.available),
        "fallback_used": not bool(narrator.available),
        "load_error": getattr(narrator, "load_error", ""),
    }
    return result


def choose_ticker_date(repo_root: str, mode: str, chunk: int, split: str) -> Tuple[str, Optional[str], pd.DataFrame]:
    picker = load_picker_frame(repo_root, mode, int(chunk), str(split))
    if picker.empty:
        ticker = st.text_input("Ticker", value="A").strip().upper()
        date = st.text_input("Date YYYY-MM-DD; leave blank for latest", value="").strip()
        return ticker, date or None, picker

    tickers = sorted(picker["ticker"].dropna().unique().tolist())
    default_idx = tickers.index("A") if "A" in tickers else 0
    ticker = st.selectbox("Ticker", tickers, index=default_idx)

    dates = picker.loc[picker["ticker"] == ticker, "date"].dropna().unique().tolist()
    dates = sorted(dates)
    default_date = dates[-1] if dates else ""
    date = st.selectbox("Date", dates, index=max(0, len(dates) - 1)) if dates else st.text_input("Date", value=default_date)
    return str(ticker), str(date) if date else None, picker


def run_selected_mode(
    engine: FinGlassboxInferenceEngine,
    mode: str,
    ticker: str,
    date: Optional[str],
    chunk: int,
    split: str,
    manual_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if mode == "Historical replay":
        return engine.historical(ticker=ticker, date=date, chunk=chunk, split=split)
    if mode == "Frozen-cached inference":
        return engine.frozen_cached(ticker=ticker, date=date, chunk=chunk, split=split)
    if mode == "Manual what-if":
        if manual_payload is None:
            raise ValueError("manual_payload is required for Manual what-if mode")
        return engine.manual(manual_payload, chunk=chunk, split=split)
    raise ValueError(f"Unsupported mode: {mode}")


def manual_payload_form(ticker: str, date: Optional[str], chunk: int, split: str) -> Dict[str, Any]:
    st.subheader("Manual what-if inputs")
    st.caption("These fields feed the frozen QuantitativeAnalyst and FusionEngine path. Missing advanced fields are filled by the inference engine's module defaults.")

    left, right = st.columns(2)
    with left:
        trend_score = st.slider("trend_score", 0.0, 1.0, 0.50, 0.01)
        momentum_score = st.slider("momentum_score", 0.0, 1.0, 0.50, 0.01)
        timing_confidence = st.slider("timing_confidence", 0.0, 1.0, 0.50, 0.01)
        technical_confidence = st.slider("technical_confidence", 0.0, 1.0, 0.50, 0.01)
        recommended_capital_pct = st.slider("recommended_capital_pct", 0.0, 15.0, 5.0, 0.25)
    with right:
        volatility_risk = st.slider("volatility_risk_score", 0.0, 1.0, 0.35, 0.01)
        drawdown_risk = st.slider("drawdown_risk_score", 0.0, 1.0, 0.43, 0.01)
        var_cvar_risk = st.slider("var_cvar_risk_score", 0.0, 1.0, 0.43, 0.01)
        contagion_risk = st.slider("contagion_risk_score", 0.0, 1.0, 0.42, 0.01)
        liquidity_risk = st.slider("liquidity_risk_score", 0.0, 1.0, 0.21, 0.01)
        regime_risk = st.slider("regime_risk_score", 0.0, 1.0, 0.78, 0.01)

    qual_left, qual_right = st.columns(2)
    with qual_left:
        qualitative_score = st.slider("qualitative_score", -1.0, 1.0, 0.0, 0.01)
        qualitative_risk = st.slider("qualitative_risk_score", 0.0, 1.0, 0.5, 0.01)
    with qual_right:
        qualitative_conf = st.slider("qualitative_confidence", 0.0, 1.0, 0.0, 0.01)
        text_available = st.checkbox("text_available", value=False)

    recommended_capital_fraction = recommended_capital_pct / 100.0
    position_fraction_of_max = min(max(recommended_capital_fraction / 0.10, 0.0), 1.0)

    position_sizing = {
        "ticker": ticker,
        "date": date,
        "chunk": chunk,
        "split": split,
        "exposure_mode": "moderate",
        "horizon_mode": "short",
        "trend_score": trend_score,
        "momentum_score": momentum_score,
        "timing_confidence": timing_confidence,
        "technical_confidence": technical_confidence,
        "volatility_risk_score": volatility_risk,
        "drawdown_risk_score": drawdown_risk,
        "var_cvar_risk_score": var_cvar_risk,
        "contagion_risk_score": contagion_risk,
        "liquidity_risk_score": liquidity_risk,
        "regime_risk_score": regime_risk,
        "combined_risk_score": (
            0.20 * volatility_risk
            + 0.15 * drawdown_risk
            + 0.15 * var_cvar_risk
            + 0.25 * contagion_risk
            + 0.15 * liquidity_risk
            + 0.10 * regime_risk
        ),
        "position_fraction_of_max": position_fraction_of_max,
        "recommended_capital_fraction": recommended_capital_fraction,
        "recommended_capital_pct": recommended_capital_pct,
        "max_single_stock_exposure": 0.10,
        "regime_confidence": 0.50,
        "hard_cap_applied": 0.0,
        "pre_cap_position_fraction_of_max": position_fraction_of_max,
        "pre_cap_capital_fraction": recommended_capital_fraction,
        "risk_bucket_fraction": position_fraction_of_max,
        "size_bucket": "manual",
        "binding_cap_source": "manual",
        "regime_label": "manual",
        "tradable": 1.0,
    }

    qualitative = {
        "ticker": ticker,
        "date": date,
        "event_count": 1.0 if text_available else 0.0,
        "sentiment_event_count": 1.0 if text_available else 0.0,
        "news_event_count": 1.0 if text_available else 0.0,
        "qualitative_score": qualitative_score,
        "qualitative_risk_score": qualitative_risk,
        "qualitative_confidence": qualitative_conf,
        "qualitative_recommendation": "BUY" if qualitative_score > 0.2 else "SELL" if qualitative_score < -0.2 else "HOLD",
        "max_event_risk_score": qualitative_risk,
        "mean_event_risk_score": qualitative_risk,
        "mean_sentiment_score": qualitative_score,
        "mean_news_impact_score": qualitative_score,
        "mean_news_importance": qualitative_conf,
        "dominant_qualitative_driver": "manual_text_input" if text_available else "no_text_event",
        "qualitative_xai_summary": "Manual qualitative what-if input.",
    }

    return {
        "ticker": ticker,
        "date": date,
        "chunk": chunk,
        "split": split,
        "position_sizing": position_sizing,
        "qualitative": qualitative,
    }


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    render_header()

    with st.sidebar:
        st.header("Runtime")
        repo_root = st.text_input("Repository root", value=str(REPO_ROOT))
        device = st.selectbox("Device", ["cuda", "cpu"], index=0 if ARGS.device.startswith("cuda") else 1)
        chunk = st.selectbox("Chunk", [1, 2, 3], index=[1, 2, 3].index(int(ARGS.default_chunk)) if int(ARGS.default_chunk) in [1, 2, 3] else 2)
        split = st.selectbox("Split", ["val", "test"], index=1 if ARGS.default_split == "test" else 0)
        mode = st.radio("Inference mode", ["Historical replay", "Frozen-cached inference", "Manual what-if"], index=1)

        st.header("Model options")
        exposure_mode = st.selectbox("Exposure mode", ["conservative", "moderate", "aggressive"], index=1)
        horizon_mode = st.selectbox("Horizon mode", ["short", "long"], index=0)
        prefer_best = st.checkbox("Prefer best_model.pt over final_model.pt", value=False)

        st.header("Narrator")
        llm_model = st.text_input("Optional local/HF narrator model", value="", placeholder="Qwen/Qwen3-0.6B or local path")
        allow_download = st.checkbox("Allow narrator model download", value=False)

        st.header("Transparency")
        show_full = st.checkbox("Show full system output", value=True)
        show_modules = st.checkbox("Show per-model/module outputs", value=True)
        show_raw = st.checkbox("Show raw JSON", value=False)

    engine = get_engine(repo_root, device, int(chunk), str(split), exposure_mode, horizon_mode, prefer_best)

    st.subheader("Input selection")
    picker_mode = "Historical replay" if mode == "Historical replay" else "Frozen-cached inference"
    ticker, date, picker = choose_ticker_date(repo_root, picker_mode, int(chunk), str(split))

    if not picker.empty:
        st.caption(f"Loaded {len(picker):,} ticker-date choices for chunk{chunk}_{split}.")

    manual_payload = None
    if mode == "Manual what-if":
        manual_payload = manual_payload_form(ticker=ticker, date=date, chunk=int(chunk), split=str(split))

    run = st.button("Run inference", type="primary", use_container_width=True)

    if not run:
        st.info("Select a ticker/date, choose an inference mode, then run inference.")
        return

    with st.spinner("Running inference..."):
        try:
            result = run_selected_mode(
                engine=engine,
                mode=mode,
                ticker=ticker,
                date=date,
                chunk=int(chunk),
                split=str(split),
                manual_payload=manual_payload,
            )
            result = apply_llm_narrator(result, llm_model, device, allow_download)
        except Exception as exc:
            st.error("Inference failed.")
            st.exception(exc)
            return

    render_decision_cards(result)
    render_risk_breakdown(result)
    render_source_details(result)
    render_transparency(result, show_full=show_full, show_modules=show_modules, show_raw=show_raw)

    st.download_button(
        "Download result JSON",
        data=json.dumps(json_safe(result), indent=2, default=str),
        file_name=f"fin_glassbox_{mode.lower().replace(' ', '_')}_{ticker}_{date or 'latest'}.json",
        mime="application/json",
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
