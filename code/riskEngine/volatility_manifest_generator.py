import numpy as np, pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch

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

# Get seq_len from checkpoint
ckpt = torch.load('outputs/models/TemporalEncoder/chunk1/best_model.pt', map_location='cpu', weights_only=False)
SEQ_LEN = ckpt['config']['seq_len']
print(f"Seq length from checkpoint: {SEQ_LEN}")

features_df = pd.read_csv(FEATURES_PATH, dtype={"ticker": str}, parse_dates=["date"])
print(f"Features: {len(features_df):,} rows, {features_df['ticker'].nunique()} tickers")

for chunk_id in [1, 2, 3]:
    for split in ["train", "val", "test"]:
        chunk_cfg = CHUNK_CONFIG[chunk_id]
        years = chunk_cfg[split]
        label = chunk_cfg["label"]
        
        emb_path = EMB_DIR / f"{label}_{split}_embeddings.npy"
        if not emb_path.exists():
            print(f"  ⚠ {label}_{split}: no embeddings yet — skipping")
            continue
        
        embeddings = np.load(emb_path, mmap_mode='r')
        print(f"\n  {label}_{split}: {embeddings.shape[0]:,} embeddings")
        
        # Filter to split years
        df = features_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year
        mask = (df["year"] >= years[0]) & (df["year"] <= years[1])
        df = df[mask]
        
        # Build sequences in EXACT same order as MarketSequenceDataset
        records = []
        for ticker, group in tqdm(df.groupby("ticker"), desc=f"    Building", leave=False):
            vals = group[FEATURE_NAMES].values.astype(np.float32)
            dates = group["date"].values
            if len(vals) < SEQ_LEN:
                continue
            for i in range(SEQ_LEN - 1, len(vals)):
                records.append({"ticker": ticker, "date": str(dates[i])[:10]})
        
        # Verify and save
        count = len(records)
        if count != len(embeddings):
            print(f"    ⚠ MISMATCH: {count:,} sequences vs {len(embeddings):,} embeddings")
            print(f"    Using min({count}, {len(embeddings)})")
            count = min(count, len(embeddings))
            records = records[:count]
        else:
            print(f"    ✓ EXACT MATCH: {count:,}")
        
        manifest_df = pd.DataFrame(records[:count])
        manifest_path = EMB_DIR / f"{label}_{split}_manifest.csv"
        manifest_df.to_csv(manifest_path, index=False)
        print(f"    ✓ Saved: {manifest_path.name}")

print("\n✓ All manifests generated.")