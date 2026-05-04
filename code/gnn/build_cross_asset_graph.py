#!/usr/bin/env python3
"""
code/gnn/build_cross_asset_graph.py

Cross-Asset Graph Builder for Financial Risk Management Framework.
Builds all cross-asset relationship graphs from market data.

Produces:
  data/graphs/metadata/     - ticker universe, sector map, market cap proxy, betas
  data/graphs/returns/      - returns_matrix.csv
  data/graphs/static/       - static graph edges (sector hierarchy + ETF similarity)
  data/graphs/snapshots/    - correlation edge lists per 20-day window
  data/graphs/combined/     - PyTorch Geometric Data objects per window

Usage:
  python code/gnn/build_cross_asset_graph.py --workers 4
  python code/gnn/build_cross_asset_graph.py --workers 4 --skip-snapshots  # metadata only
"""

import os, sys, argparse, pickle, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent.parent
RETURNS_WIDE= ROOT / "data/yFinance/processed/returns_panel_wide.csv"
OHLCV_FILE  = ROOT / "data/yFinance/processed/ohlcv_final.csv"
ISSUER_FILE = ROOT / "data/sec_edgar/processed/cleaned/issuer_master_cleaned.csv"
GRAPHS_DIR  = ROOT / "data/graphs"

# ── Parameters ─────────────────────────────────────────────
K_EDGES     = 50           # √2500 ≈ 50 edges per node
CORR_WINDOW = 30           # trading days for correlation
SNAP_STRIDE = 20           # trading days between snapshots
BETA_WINDOW = 252          # days for beta
MCAP_WINDOW = 252          # days for dollar volume median

# ── ETF universe (only those in our data) ──────────────────
ETF_TICKERS = ["SPY", "QQQ", "DIA"]

# ── SIC → GICS Sector Mapping ─────────────────────────────
# Maps SIC descriptions to 11 GICS sectors + "Other"
SIC_TO_GICS = {
    # Technology
    "Services-Prepackaged Software": "Information Technology",
    "Services-Computer Integrated Systems Design": "Information Technology",
    "Services-Computer Processing & Data Preparation": "Information Technology",
    "Computer Communications Equipment": "Information Technology",
    "Computer Peripheral Equipment, NEC": "Information Technology",
    "Electronic Computers": "Information Technology",
    "Computer Storage Devices": "Information Technology",
    "Semiconductors & Related Devices": "Information Technology",
    "Electronic Components, NEC": "Information Technology",
    "Printed Circuit Boards": "Information Technology",
    "Services-Computer Programming Services": "Information Technology",
    
    # Financials
    "State Commercial Banks": "Financials",
    "National Commercial Banks": "Financials",
    "Savings Institution, Federally Chartered": "Financials",
    "Federal Savings Institutions": "Financials",
    "Commercial Banks, NEC": "Financials",
    "Security Brokers, Dealers & Flotation Companies": "Financials",
    "Investment Advice": "Financials",
    "Finance Services": "Financials",
    "Fire, Marine & Casualty Insurance": "Financials",
    "Life Insurance": "Financials",
    "Accident & Health Insurance": "Financials",
    "Surety Insurance": "Financials",
    "Title Insurance": "Financials",
    "Insurance Agents, Brokers & Service": "Financials",
    "Personal Credit Institutions": "Financials",
    "Short-Term Business Credit Institutions": "Financials",
    "Mortgage Bankers & Loan Correspondents": "Financials",
    "Security & Commodity Brokers, Dealers, Exchanges & Services": "Financials",
    "Commodity Contracts Brokers & Dealers": "Financials",
    "Real Estate Investment Trusts": "Financials",  # REITs classified under Financials in GICS

    # Health Care
    "Pharmaceutical Preparations": "Health Care",
    "Surgical & Medical Instruments & Apparatus": "Health Care",
    "Biological Products, (No Diagnostic Substances)": "Health Care",
    "In Vitro & In Vivo Diagnostic Substances": "Health Care",
    "Hospital & Medical Service Plans": "Health Care",
    "Services-Medical Laboratories": "Health Care",
    "Dental Equipment & Supplies": "Health Care",
    "Orthopedic, Prosthetic & Surgical Appliances & Supplies": "Health Care",
    "Medicinal Chemicals & Botanical Products": "Health Care",
    "Electromedical & Electrotherapeutic Apparatus": "Health Care",
    "Services-Home Health Care Services": "Health Care",
    "Services-Specialty Outpatient Facilities, NEC": "Health Care",

    # Energy
    "Crude Petroleum & Natural Gas": "Energy",
    "Drilling Oil & Gas Wells": "Energy",
    "Oil & Gas Field Services, NEC": "Energy",
    "Petroleum Refining": "Energy",
    "Natural Gas Transmission": "Energy",
    "Oil & Gas Field Exploration Services": "Energy",
    "Oil Royalty Traders": "Energy",
    "Natural Gas Distribution": "Energy",

    # Industrials
    "Motor Vehicle Parts & Accessories": "Industrials",
    "Industrial Instruments For Measurement, Display, and Control": "Industrials",
    "Services-Engineering Services": "Industrials",
    "Aircraft": "Industrials",
    "Construction Machinery & Equipment": "Industrials",
    "General Industrial Machinery & Equipment": "Industrials",
    "Pumps & Pumping Equipment": "Industrials",
    "Air-Conditioning & Warm Air Heating Equipment": "Industrials",
    "Industrial Trucks, Tractors & Trailers": "Industrials",
    "Railroad Equipment": "Industrials",
    "Trucking (No Local)": "Industrials",
    "Arrangement of Transportation of Freight & Cargo": "Industrials",
    "Transportation Services": "Industrials",
    "Air Courier Services": "Industrials",
    "Water Transportation": "Industrials",
    "Deep Sea Foreign Transportation": "Industrials",
    "Waste Management": "Industrials",
    "Pollution & Treatment Controls": "Industrials",

    # Consumer Discretionary
    "Retail-Eating  Places": "Consumer Discretionary",
    "Retail-Auto Dealers & Gasoline Stations": "Consumer Discretionary",
    "Retail-Home Furniture, Furnishings & Equipment Stores": "Consumer Discretionary",
    "Retail-Misc Retail Stores, NEC": "Consumer Discretionary",
    "Retail-Department Stores": "Consumer Discretionary",
    "Retail-Shoe Stores": "Consumer Discretionary",
    "Retail-Variety Stores": "Consumer Discretionary",
    "Services-Amusement & Recreation Services": "Consumer Discretionary",
    "Services-Motion Picture & Video Tape Production": "Consumer Discretionary",
    "Hotels & Motels": "Consumer Discretionary",
    "Radio Broadcasting Stations": "Consumer Discretionary",
    "Television Broadcasting Stations": "Consumer Discretionary",
    "Cable & Other Pay Television Services": "Consumer Discretionary",
    "Restaurants": "Consumer Discretionary",
    "Motor Vehicles & Passenger Car Bodies": "Consumer Discretionary",
    "Games, Toys & Children's Vehicles": "Consumer Discretionary",
    "Dolls & Stuffed Toys": "Consumer Discretionary",
    "Amusement Parks": "Consumer Discretionary",
    "Book Publishing": "Consumer Discretionary",
    "Newspapers: Publishing or Publishing & Printing": "Consumer Discretionary",
    "Periodicals: Publishing or Publishing & Printing": "Consumer Discretionary",

    # Consumer Staples
    "Food & Kindred Products": "Consumer Staples",
    "Beverages": "Consumer Staples",
    "Grain Mill Products": "Consumer Staples",
    "Meat Packing Plants": "Consumer Staples",
    "Dairy Products": "Consumer Staples",
    "Canned, Frozen & Preserved Fruit, Veg & Food Specialties": "Consumer Staples",
    "Sugar & Confectionery Products": "Consumer Staples",
    "Fats & Oils": "Consumer Staples",
    "Retail-Grocery Stores": "Consumer Staples",
    "Retail-Convenience Stores": "Consumer Staples",
    "Services-Personal Services": "Consumer Staples",
    "Perfumes, Cosmetics & Other Toilet Preparations": "Consumer Staples",

    # Utilities
    "Electric Services": "Utilities",
    "Electric & Other Services Combined": "Utilities",
    "Natural Gas Transmission & Distribution": "Utilities",
    "Water Supply": "Utilities",
    "Gas & Other Services Combined": "Utilities",
    
    # Communication Services
    "Telephone Communications (No Radiotelephone)": "Communication Services",
    "Radiotelephone Communications": "Communication Services",
    "Communications Services, NEC": "Communication Services",
    "Services-Miscellaneous Business Services": "Communication Services",

    # Materials
    "Metal Mining": "Materials",
    "Gold and Silver Ores": "Materials",
    "Industrial Organic Chemicals": "Materials",
    "Plastic Materials, Synth Resins & Nonvulcan Elastomers": "Materials",
    "Paints, Varnishes, Lacquers, Enamels & Allied Products": "Materials",
    "Steel Works, Blast Furnaces & Rolling Mills": "Materials",
    "Steel Pipe & Tubes": "Materials",
    "Aluminum": "Materials",
    "Primary Smelting & Refining of Nonferrous Metals": "Materials",
    "Mining & Quarrying of Nonmetallic Minerals": "Materials",
    "Forest Products": "Materials",
    "Paper Mills": "Materials",
    "Lumber & Wood Products": "Materials",
    "Concrete Products, Except Block & Brick": "Materials",
    "Abrasive, Asbestos & Misc Nonmetallic Mineral Products": "Materials",
    "Construction Special Trade Contractors": "Materials",

    # Real Estate
    "Real Estate": "Real Estate",
    "Real Estate Operators (No Developers) & Lessors": "Real Estate",
    "Real Estate Agents & Managers": "Real Estate",
    "Land Subdividers & Developers": "Real Estate",
    "Operators of Nonresidential Buildings": "Real Estate",
    "Operators of Apartment Buildings": "Real Estate",
    "Services-Real Estate Agents & Managers": "Real Estate",
}

GICS_SECTORS = [
    "Information Technology", "Financials", "Health Care", "Energy",
    "Industrials", "Consumer Discretionary", "Consumer Staples",
    "Utilities", "Communication Services", "Materials", "Real Estate"
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--skip-snapshots", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("CROSS-ASSET GRAPH BUILDER")
    print("=" * 60)
    print(f"Nodes: 2,500 tickers + 3 ETFs + {len(GICS_SECTORS)} sectors")
    print(f"Edges per node: {K_EDGES}")
    print(f"Correlation window: {CORR_WINDOW} days")
    print(f"Snapshot stride: {SNAP_STRIDE} days")
    print()
    
    # Create directories
    for subdir in ["metadata", "returns", "static", "snapshots", "combined"]:
        (GRAPHS_DIR / subdir).mkdir(parents=True, exist_ok=True)
    
    # ══════════════════════════════════════════════════════
    # STEP 1: Load market data and returns
    # ══════════════════════════════════════════════════════
    print("STEP 1: Loading returns matrix...")
    returns_wide = pd.read_csv(RETURNS_WIDE, index_col=0, parse_dates=True)
    tickers_all = list(returns_wide.columns)
    dates_all = returns_wide.index
    print(f"  Returns: {returns_wide.shape[0]} dates × {returns_wide.shape[1]} tickers")
    
    # ══════════════════════════════════════════════════════
    # STEP 2: Build metadata
    # ══════════════════════════════════════════════════════
    print("\nSTEP 2: Building metadata...")
    
    # ── 2A. Sector mapping from SIC ──
    issuer = pd.read_csv(ISSUER_FILE, usecols=["primary_ticker", "sic_description"])
    issuer["primary_ticker"] = issuer["primary_ticker"].str.upper()
    issuer = issuer.dropna(subset=["primary_ticker"])
    issuer = issuer[issuer["primary_ticker"].isin(tickers_all)]
    
    sector_map = {}
    for _, row in issuer.iterrows():
        t = row["primary_ticker"]
        sic = str(row["sic_description"]) if pd.notna(row["sic_description"]) else ""
        gics = SIC_TO_GICS.get(sic, "Other")
        sector_map[t] = gics
    
    # Fill any missing tickers with "Other"
    for t in tickers_all:
        if t not in sector_map:
            sector_map[t] = "Other"
    
    # Save
    sector_df = pd.DataFrame({"ticker": list(sector_map.keys()), "gics_sector": list(sector_map.values())})
    sector_df.to_csv(GRAPHS_DIR / "metadata/sector_map.csv", index=False)
    counts = sector_df["gics_sector"].value_counts()
    print(f"  Sectors: {dict(counts)}")
    
    # ── 2B. Market cap proxy (median dollar volume 252d) ──
    print("  Computing market cap proxy...")
    ohlcv = pd.read_csv(OHLCV_FILE, dtype={"ticker": str, "date": str})
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])
    ohlcv["dollar_vol"] = ohlcv["close"] * ohlcv["volume"]
    
    mcap_proxy = {}
    for ticker in tqdm(tickers_all, desc="  Market cap proxy"):
        sub = ohlcv[ohlcv["ticker"] == ticker]
        if len(sub) > 0:
            mcap_proxy[ticker] = sub["dollar_vol"].rolling(MCAP_WINDOW).median().iloc[-1]
        else:
            mcap_proxy[ticker] = 0
    
    pd.DataFrame({"ticker": list(mcap_proxy.keys()), "market_cap_proxy": list(mcap_proxy.values())}
                ).to_csv(GRAPHS_DIR / "metadata/market_cap_proxy.csv", index=False)
    print(f"  Market cap proxy: {len(mcap_proxy)} tickers")
    
    # ── 2C. Beta vs SPY ──
    print("  Computing betas vs SPY...")
    betas = {}
    if "SPY" in returns_wide.columns:
        spy_ret = returns_wide["SPY"]
        for ticker in tqdm(tickers_all, desc="  Beta"):
            if ticker == "SPY":
                betas[ticker] = 1.0
                continue
            ret = returns_wide[ticker]
            common = ret.notna() & spy_ret.notna()
            if common.sum() > BETA_WINDOW:
                cov = ret[common].rolling(BETA_WINDOW).cov(spy_ret[common])
                var = spy_ret[common].rolling(BETA_WINDOW).var()
                b = (cov / var).iloc[-1]
                betas[ticker] = b if pd.notna(b) else 1.0
            else:
                betas[ticker] = 1.0
    else:
        for t in tickers_all:
            betas[t] = 1.0
    
    pd.DataFrame({"ticker": list(betas.keys()), "beta": list(betas.values())}
                ).to_csv(GRAPHS_DIR / "metadata/betas.csv", index=False)
    print(f"  Betas: {len(betas)} tickers")
    
    # ── 2D. Save ticker universe ──
    pd.DataFrame({"ticker": tickers_all}).to_csv(GRAPHS_DIR / "metadata/ticker_universe.csv", index=False)
    
    # ══════════════════════════════════════════════════════
    # STEP 3: Save returns matrix copy
    # ══════════════════════════════════════════════════════
    print("\nSTEP 3: Saving returns matrix...")
    returns_wide.to_csv(GRAPHS_DIR / "returns/returns_matrix.csv")
    print(f"  Saved: {GRAPHS_DIR / 'returns/returns_matrix.csv'}")
    
    # ══════════════════════════════════════════════════════
    # STEP 4: Sector similarity matrix
    # ══════════════════════════════════════════════════════
    print("\nSTEP 4: Building sector similarity matrix...")
    sector_returns = {}
    for sector in GICS_SECTORS:
        sector_tickers = [t for t, s in sector_map.items() if s == sector]
        if sector_tickers:
            sector_returns[sector] = returns_wide[sector_tickers].mean(axis=1)
    
    sector_corr = pd.DataFrame(index=GICS_SECTORS, columns=GICS_SECTORS, dtype=float)
    for s1 in GICS_SECTORS:
        for s2 in GICS_SECTORS:
            if s1 in sector_returns and s2 in sector_returns:
                r1 = sector_returns[s1].dropna()
                r2 = sector_returns[s2].dropna()
                common = r1.index.intersection(r2.index)
                if len(common) > 100:
                    sector_corr.loc[s1, s2] = r1[common].corr(r2[common])
                else:
                    sector_corr.loc[s1, s2] = 0.0
            else:
                sector_corr.loc[s1, s2] = 0.0
    
    sector_similarity = (sector_corr.astype(float) + 1) / 2  # normalize to 0-1
    sector_similarity.to_csv(GRAPHS_DIR / "metadata/sector_similarity.csv")
    print(f"  Saved sector similarity: {sector_similarity.shape}")
    
    # ══════════════════════════════════════════════════════
    # STEP 5: Static graph edges
    # ══════════════════════════════════════════════════════
    print("\nSTEP 5: Building static graph edges...")
    static_edges = []
    
    # 5A. Ticker → Sector membership (weight = 1.0)
    for ticker, sector in sector_map.items():
        static_edges.append({
            "source": ticker, "target": f"SECTOR_{sector}",
            "edge_type": "sector_membership", "weight": 1.0
        })
    
    # 5B. Ticker → ETF similarity (returns correlation with ETF)
    for etf in ETF_TICKERS:
        if etf not in returns_wide.columns:
            continue
        etf_ret = returns_wide[etf]
        for ticker in tqdm(tickers_all, desc=f"  ETF {etf}"):
            if ticker == etf:
                continue
            ret = returns_wide[ticker]
            common = ret.notna() & etf_ret.notna()
            if common.sum() > 100:
                corr = ret[common].corr(etf_ret[common])
                weight = max(0, corr)  # only positive correlations
                if weight > 0.1:  # threshold to keep graph sparse
                    static_edges.append({
                        "source": ticker, "target": f"ETF_{etf}",
                        "edge_type": "etf_similarity", "weight": round(weight, 4)
                    })
    
    # 5C. Sector → Sector similarity
    for s1 in GICS_SECTORS:
        for s2 in GICS_SECTORS:
            if s1 < s2:
                w = sector_similarity.loc[s1, s2]
                if pd.notna(w) and w > 0.3:
                    static_edges.append({
                        "source": f"SECTOR_{s1}", "target": f"SECTOR_{s2}",
                        "edge_type": "sector_similarity", "weight": round(float(w), 4)
                    })
    
    static_df = pd.DataFrame(static_edges)
    static_df.to_csv(GRAPHS_DIR / "static/edges.csv", index=False)
    print(f"  Static edges: {len(static_df)}")
    
    # Save all nodes
    nodes = set()
    for _, row in static_df.iterrows():
        nodes.add(row["source"])
        nodes.add(row["target"])
    pd.DataFrame({"node_id": sorted(nodes)}).to_csv(GRAPHS_DIR / "static/nodes.csv", index=False)
    print(f"  Nodes: {len(nodes)}")
    
    if args.skip_snapshots:
        print("\nSkipping correlation snapshots (--skip-snapshots). Done.")
        return
    
    # ══════════════════════════════════════════════════════
    # STEP 6: Correlation snapshots
    # ══════════════════════════════════════════════════════
    print("\nSTEP 6: Building correlation snapshots...")
    
    n_dates = len(dates_all)
    window_starts = list(range(0, n_dates - CORR_WINDOW, SNAP_STRIDE))
    print(f"  {len(window_starts)} snapshots to build")
    
    for ws_idx in tqdm(range(len(window_starts)), desc="  Snapshots"):
        ws = window_starts[ws_idx]
        we = ws + CORR_WINDOW
        window_dates = dates_all[ws:we]
        
        # Get returns for this window
        window_ret = returns_wide.loc[window_dates]
        
        # Compute correlation matrix (only for tickers with enough data)
        valid_tickers = [t for t in tickers_all if window_ret[t].notna().sum() > 10]
        if len(valid_tickers) < 2:
            continue
        
        corr_matrix = window_ret[valid_tickers].corr()
        
        # For each ticker, find top-K neighbors
        edge_list = []
        for ticker in valid_tickers:
            if ticker not in corr_matrix.index:
                continue
            row = corr_matrix.loc[ticker].drop(ticker, errors="ignore")
            row = row.dropna()
            top_k = row.nlargest(K_EDGES)
            
            for neighbor, corr_val in top_k.items():
                if corr_val > 0.1:  # threshold
                    edge_list.append({
                        "window_start": str(window_dates[0].date()),
                        "source": ticker,
                        "target": neighbor,
                        "correlation": round(corr_val, 6)
                    })
        
        if edge_list:
            snap_date = str(window_dates[0].date())
            pd.DataFrame(edge_list).to_csv(
                GRAPHS_DIR / f"snapshots/edges_{snap_date}.csv", index=False
            )
    
    print(f"  Snapshots saved to: {GRAPHS_DIR / 'snapshots/'}")
    
    # ══════════════════════════════════════════════════════
    # STEP 7: Save as pickled NetworkX graphs (lightweight, no torch-geometric dependency)
    # ══════════════════════════════════════════════════════
    print("\nSTEP 7: Building graph objects...")
    
    try:
        import networkx as nx
        
        # Build one sample combined graph from the latest snapshot
        snap_files = sorted((GRAPHS_DIR / "snapshots").glob("edges_*.csv"))
        if snap_files:
            latest = snap_files[-1]
            snap_edges = pd.read_csv(latest)
            
            G = nx.Graph()
            
            # Add nodes
            for t in tickers_all:
                G.add_node(t, type="ticker", sector=sector_map.get(t, "Other"),
                          mcap=mcap_proxy.get(t, 0), beta=betas.get(t, 1.0))
            for etf in ETF_TICKERS:
                if etf in returns_wide.columns:
                    G.add_node(f"ETF_{etf}", type="etf")
            for sector in GICS_SECTORS:
                G.add_node(f"SECTOR_{sector}", type="sector")
            
            # Add correlation edges
            for _, row in snap_edges.iterrows():
                G.add_edge(row["source"], row["target"], 
                          edge_type="correlation", weight=row["correlation"])
            
            # Add static edges
            for _, row in static_df.iterrows():
                G.add_edge(row["source"], row["target"],
                          edge_type=row["edge_type"], weight=row["weight"])
            
            # Save
            with open(GRAPHS_DIR / "combined/sample_graph.pkl", "wb") as f:
                pickle.dump(G, f)
            print(f"  Sample graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            print(f"  Saved: {GRAPHS_DIR / 'combined/sample_graph.pkl'}")
    except ImportError:
        print("  networkx not installed — skipping graph object. pip install networkx")
    
    # ══════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("GRAPH BUILD COMPLETE")
    print("=" * 60)
    print(f"  Tickers: {len(tickers_all)}")
    print(f"  Sectors: {len(GICS_SECTORS)}")
    print(f"  ETFs: {len(ETF_TICKERS)}")
    print(f"  Static edges: {len(static_df)}")
    print(f"  Snapshots: {len(window_starts)}")
    print(f"  Output: {GRAPHS_DIR}")
    

if __name__ == "__main__":
    main()

# Install networkx if needed
# pip install networkx -q

# # Run (without snapshots first to test metadata):
# python code/gnn/build_cross_asset_graph.py --workers 4 --skip-snapshots

# # Full run:
# python code/gnn/build_cross_asset_graph.py --workers 4
