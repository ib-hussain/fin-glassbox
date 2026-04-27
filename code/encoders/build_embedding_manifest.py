#!/usr/bin/env python3
"""
ONE-TIME SCRIPT: Generate embedding manifests
=============================================
Run once to create manifest files mapping each embedding row to (ticker, date).
All downstream modules load these manifests instead of rebuilding sequence ordering.

Usage:
  python code/encoders/build_embedding_manifest.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

FEATURES_PATH = "data/yFinance/processed/features_temporal.csv"
EMB_DIR = Path("outputs/embeddings/TemporalEncoder")
FEATURE_NAMES = [
    "log_return", "vol_5d", "vol_21d", "rsi_14", "macd_hist",
    "bb_pos", "volume_ratio", "hl_ratio", "price_pos", "spy_corr_63d",
]

CHUNK_CONFIG = {
    1: {"train": (2000, 2004), "val": (2005, 2005), "test": (2006, 2006), "label": "chunk1"},
    2: {"train": (2007, 2014), "val": (2015, 2015), "test": (2016, 2016), "label": "chunk2"},
    3: {"train": (2017, 2022), "val": (2023, 2023), "test": (2024, 2024), "label": "chunk3"},
}


def build_manifest_for_split(features_df, chunk_id, split, seq_len=30):
    """Reconstruct exact sequence ordering and save manifest."""
    chunk_cfg = CHUNK_CONFIG[chunk_id]
    years = chunk_cfg[split]
    label = chunk_cfg["label"]

    emb_path = EMB_DIR / f"{label}_{split}_embeddings.npy"
    if not emb_path.exists():
        print(f"  ⚠ No embeddings for {label}_{split} — skipping")
        return

    embeddings = np.load(emb_path, mmap_mode='r')

    # Filter to split years
    df = features_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    mask = (df["year"] >= years[0]) & (df["year"] <= years[1])
    df = df[mask]

    # Build sequences in EXACT same order as MarketSequenceDataset
    records = []
    ticker_groups = df.groupby("ticker")

    for ticker, group in tqdm(ticker_groups, desc=f"  {label}_{split}"):
        vals = group[FEATURE_NAMES].values.astype(np.float32)
        vals = np.nan_to_num(vals, nan=0.0)
        dates = group["date"].values

        if len(vals) < seq_len:
            continue

        for i in range(seq_len - 1, len(vals)):
            records.append({
                "ticker": ticker,
                "date": dates[i],
                "seq_start": dates[i - seq_len + 1],
                "seq_end": dates[i],
            })

    # Verify alignment
    if len(records) != len(embeddings):
        print(f"  ⚠ MISMATCH: {len(records)} sequences vs {len(embeddings)} embeddings!")
        print(f"    Truncating to min({len(records)}, {len(embeddings)})")
        n = min(len(records), len(embeddings))
        records = records[:n]
    else:
        print(f"  ✓ Exact match: {len(records):,} sequences")

    # Save manifest
    manifest_df = pd.DataFrame(records)
    manifest_path = EMB_DIR / f"{label}_{split}_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    print(f"  ✓ Saved: {manifest_path} ({len(manifest_df):,} rows)")


def main():
    print("=" * 60)
    print("BUILDING EMBEDDING MANIFESTS")
    print("=" * 60)

    features_df = pd.read_csv(FEATURES_PATH, dtype={"ticker": str}, parse_dates=["date"])
    print(f"Features: {len(features_df):,} rows, {features_df['ticker'].nunique()} tickers")
    print(f"Date range: {features_df['date'].min().date()} → {features_df['date'].max().date()}")

    for chunk_id in [1, 2, 3]:
        for split in ["train", "val", "test"]:
            build_manifest_for_split(features_df, chunk_id, split)

    print("\n✓ All manifests generated. Downstream modules can now use:")
    print("  manifest = pd.read_csv(f'outputs/embeddings/TemporalEncoder/chunk1_train_manifest.csv')")
    print("  embeddings = np.load(f'outputs/embeddings/TemporalEncoder/chunk1_train_embeddings.npy', mmap_mode='r')")
    print("  # manifest['ticker'][i] gives ticker for embeddings[i]")


if __name__ == "__main__":
    main()