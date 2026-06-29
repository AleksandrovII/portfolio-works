"""
correlation_matrix.py
---------------------
Reads data/portfolio.csv, fetches daily price history from T-Invest API for
shares and ETFs, computes Pearson (or Spearman) correlation on log-returns,
and saves a heatmap PNG plus CSV exports to Results/.

Usage:
    export INVEST_TOKEN='your_token'
    /opt/miniconda3/bin/python correlation_matrix.py              # 2-year window
    /opt/miniconda3/bin/python correlation_matrix.py 365          # 1-year window
    /opt/miniconda3/bin/python correlation_matrix.py --top 8      # top-8 by value
    /opt/miniconda3/bin/python correlation_matrix.py --spearman
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys
from datetime import date, timedelta
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import pearsonr, spearmanr

_ROOT    = pathlib.Path(__file__).parent
sys.path.insert(0, str(_ROOT))

from portfolio_works_library import (
    CURRENCY_FIGI,
    CorrMethod,
    build_figi_map_from_csv,
    build_returns_matrix,
    fetch_price_history,
    portfolio_weights_from_csv,
)

SNAPSHOT  = _ROOT / "data" / "portfolio.csv"
RESULTS   = _ROOT / "Results"
OUT_PNG   = RESULTS / "correlation_matrix.png"
OUT_CSV   = RESULTS / "correlation_matrix.csv"
OUT_PAIRS = RESULTS / "correlation_pairs.csv"

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

INCLUDE_TYPES = {"share", "etf"}

# External market indicators fetched via Yahoo Finance (pip install yfinance)
# Brent is the closest free proxy for Urals (r ≈ 0.99, Urals = Brent − discount)
# Nutrien (NTR) is the world's largest fertilizer producer — proxy for fertilizer prices
EXTERNAL_TICKERS: Dict[str, str] = {
    "BZ=F": "BRENT",    # Brent crude futures — closest free proxy for Urals
    "NTR":  "NUTRIEN",  # Nutrien Ltd (NYSE) — fertilizer proxy
}

# MOEX indices fetched directly from MOEX ISS (more reliable than Yahoo Finance for RU indices)
MOEX_INDICES: Dict[str, tuple] = {
    "IMOEX": ("stock", "index", "SNDX"),   # Moscow Exchange Index (RUB)
    "RTSI":  ("stock", "index", "RTSI"),   # RTS Index (USD-denominated)
}

# ── Dark theme ──────────────────────────────────────────────────────────────────

_DARK_BG  = "#0D1117"
_PANEL_BG = "#161B22"
_TEXT     = "#E6EDF3"
_DIM      = "#8B949E"
_GRID     = "#30363D"
_ACCENT1  = "#58A6FF"


def _dark_theme() -> None:
    plt.rcParams.update({
        "figure.facecolor": _DARK_BG, "axes.facecolor":  _PANEL_BG,
        "axes.edgecolor":   _GRID,    "axes.labelcolor":  _DIM,
        "text.color":       _TEXT,    "xtick.color":      _DIM,
        "ytick.color":      _DIM,     "grid.color":       _GRID,
        "font.family":      "sans-serif",
    })


# ── MOEX index history ─────────────────────────────────────────────────────────

def fetch_moex_indices(days: int) -> Dict[str, pd.Series]:
    """Fetch IMOEX and RTSI history from MOEX ISS (free, no auth)."""
    import time
    import requests as _req

    start = (date.today() - timedelta(days=days)).isoformat()
    end   = date.today().isoformat()
    result: Dict[str, pd.Series] = {}

    for label, (engine, market, board) in MOEX_INDICES.items():
        url = (
            f"https://iss.moex.com/iss/history/engines/{engine}/markets/{market}"
            f"/boards/{board}/securities/{label}.json"
        )
        rows = []
        start_row = 0
        while True:
            try:
                resp = _req.get(url, params={"from": start, "till": end,
                                             "start": start_row, "limit": 100}, timeout=15)
                if resp.status_code != 200:
                    break
                data = resp.json()
            except Exception as e:
                logger.warning("MOEX ISS error for %s: %s", label, e)
                break

            hist    = data.get("history", {})
            cols    = hist.get("columns", [])
            records = hist.get("data", [])
            if not records:
                break

            try:
                date_idx  = cols.index("TRADEDATE")
                value_idx = cols.index("CLOSE")
            except ValueError:
                break

            for row in records:
                d, v = row[date_idx], row[value_idx]
                if d and v is not None:
                    rows.append((pd.Timestamp(d), float(v)))

            cursor = data.get("history.cursor", {}).get("data", [])
            if cursor:
                idx, total, pagesize = cursor[0]
                if idx + pagesize >= total:
                    break
                start_row = idx + pagesize
                time.sleep(0.2)
            else:
                break

        if len(rows) > 10:
            s = pd.Series({ts: px for ts, px in rows}, dtype=float).sort_index()
            result[label] = s[~s.index.duplicated(keep="last")]
            logger.info("  %-14s %d days loaded (MOEX ISS)", label, len(result[label]))
        else:
            logger.warning("  %-14s no index data from MOEX ISS", label)

    return result


# ── External market data (Yahoo Finance) ───────────────────────────────────────

def fetch_external_prices(days: int) -> Dict[str, pd.Series]:
    """
    Fetch daily close prices from Yahoo Finance for external indicators.
    Returns empty dict (with warning) if yfinance is not installed.
    """
    try:
        import yfinance as yf
    except ImportError:
        print("⚠  yfinance not installed — run: pip install yfinance")
        return {}

    start = (date.today() - timedelta(days=days)).isoformat()
    end   = date.today().isoformat()
    result: Dict[str, pd.Series] = {}

    for yf_ticker, label in EXTERNAL_TICKERS.items():
        try:
            hist = yf.Ticker(yf_ticker).history(start=start, end=end)
            if hist.empty:
                logger.warning("  %-14s no data from Yahoo Finance", label)
                continue
            s = hist["Close"].dropna()
            idx = pd.to_datetime(s.index)
            if idx.tz is not None:
                idx = idx.tz_convert("UTC").tz_localize(None)
            s.index = idx.normalize()
            if len(s) > 10:
                result[label] = s
                logger.info("  %-14s %d days loaded (Yahoo Finance)", label, len(s))
            else:
                logger.warning("  %-14s too few data points", label)
        except Exception as e:
            logger.warning("  %-14s Yahoo Finance error: %s", label, e)

    return result


# ── Portfolio loading ───────────────────────────────────────────────────────────

def load_snapshot(top_n: Optional[int] = None) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Load portfolio.csv filtered to shares and ETFs (no futures).

    Returns (df, figi_map) where figi_map is {figi: ticker}.
    """
    if not SNAPSHOT.exists():
        sys.exit(f"Portfolio snapshot not found: {SNAPSHOT}\nRun call_api.py first.")

    df = pd.read_csv(SNAPSHOT)
    df["rub_value"]  = pd.to_numeric(df["rub_value"],  errors="coerce").fillna(0)
    df["weight_pct"] = pd.to_numeric(df["weight_pct"], errors="coerce").fillna(0)

    df = df[~df["figi"].str.startswith("FUT", na=False)]
    df = df[df["instrument_type"].isin(INCLUDE_TYPES)]

    figi_map = build_figi_map_from_csv(df, top_n=top_n)
    return df, figi_map


# ── Correlation ─────────────────────────────────────────────────────────────────

def compute_correlation(
    returns: pd.DataFrame,
    method: CorrMethod = "pearson",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    corr = returns.corr(method=method)
    n    = len(corr)
    pval = pd.DataFrame(np.ones((n, n)), index=corr.index, columns=corr.columns)
    fn   = spearmanr if method == "spearman" else pearsonr
    for i in corr.index:
        for j in corr.columns:
            if i == j:
                pval.loc[i, j] = 0.0
                continue
            a, b = returns[i].dropna(), returns[j].dropna()
            idx  = a.index.intersection(b.index)
            if len(idx) >= 3:
                _, p = fn(a[idx], b[idx])
                pval.loc[i, j] = float(p)
    return corr, pval


# ── Console report ──────────────────────────────────────────────────────────────

def print_report(
    corr: pd.DataFrame,
    pval: pd.DataFrame,
    weights: pd.Series,
    days: int,
    method: str,
) -> None:
    n = len(corr)
    print()
    print("═" * 72)
    print(f"  ASSET CORRELATION MATRIX  ·  {days}d  ·  {n} assets  ·  {method}")
    print("═" * 72)

    col_w = max(8, max(len(t) for t in corr.columns) + 1)
    print(f"{'':>{col_w}}" + "".join(f"{t:>{col_w}}" for t in corr.columns))
    for i in corr.index:
        row = f"{i:>{col_w}}"
        for j in corr.columns:
            cell = "1.00" if i == j else f"{corr.loc[i,j]:+.2f}"
            row += f"{cell:>{col_w}}"
        print(row)

    off = sorted(
        [
            (corr.iloc[i, j], pval.iloc[i, j], corr.index[i], corr.columns[j])
            for i in range(n)
            for j in range(i)
        ],
        key=lambda x: x[0],
        reverse=True,
    )
    if not off:
        print("═" * 72)
        return

    avg = float(np.mean([x[0] for x in off]))
    print(f"\n  Average pairwise correlation: {avg:.3f}")

    if not weights.empty:
        print("\n  Portfolio weights (%):")
        for t, w in weights.sort_values(ascending=False).items():
            print(f"    {t:<14} {w:5.1f}%")

    print("\n  Highest correlation pairs (most similar):")
    for r, p, a, b in off[:5]:
        sig = " *" if p < 0.05 else ""
        print(f"    {a} ↔ {b}:  r = {r:+.3f}{sig}")

    print("\n  Best diversifiers (lowest correlation):")
    for r, p, a, b in off[-5:]:
        print(f"    {a} ↔ {b}:  r = {r:+.3f}")

    print("═" * 72)


# ── CSV export ──────────────────────────────────────────────────────────────────

def export_csv(corr: pd.DataFrame, pval: pd.DataFrame, weights: pd.Series) -> None:
    RESULTS.mkdir(exist_ok=True)
    corr.to_csv(OUT_CSV, float_format="%.4f")
    print(f"✓ Matrix CSV → {OUT_CSV}")

    tickers = list(corr.columns)
    rows = []
    for i, a in enumerate(tickers):
        for j, b in enumerate(tickers):
            if j >= i:
                continue
            rows.append({
                "asset_a":      a,
                "asset_b":      b,
                "correlation":  corr.loc[a, b],
                "p_value":      pval.loc[a, b],
                "weight_a_pct": weights.get(a, 0.0),
                "weight_b_pct": weights.get(b, 0.0),
            })
    (
        pd.DataFrame(rows)
        .sort_values("correlation", ascending=False)
        .to_csv(OUT_PAIRS, index=False, float_format="%.6f")
    )
    print(f"✓ Pairs CSV  → {OUT_PAIRS}")


# ── Heatmap ─────────────────────────────────────────────────────────────────────

def plot_heatmap(corr: pd.DataFrame, pval: pd.DataFrame, days: int, method: str) -> None:
    n = len(corr)
    _dark_theme()
    sz  = max(10, n * 0.9 + 2)
    fig, ax = plt.subplots(figsize=(sz, sz * 0.88), facecolor=_DARK_BG)
    ax.set_facecolor(_PANEL_BG)

    cmap = LinearSegmentedColormap.from_list(
        "fin_corr",
        ["#C0392B", "#922B21", _PANEL_BG, "#1A5276", "#2980B9"],
        N=256,
    )
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

    tickers = list(corr.columns)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tickers, rotation=45, ha="right", fontsize=9, color=_TEXT)
    ax.set_yticklabels(tickers, fontsize=9, color=_TEXT)

    for i in range(n):
        for j in range(n):
            if j > i:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color=_DARK_BG, zorder=2))
                continue
            val, p = corr.iloc[i, j], pval.iloc[i, j]
            star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            if i == j:
                ax.text(j, i, tickers[i], ha="center", va="center",
                        fontsize=8, fontweight="bold", color=_ACCENT1)
            else:
                tc = _TEXT if abs(val) < 0.6 else "white"
                ax.text(j, i, f"{val:.2f}\n{star}", ha="center", va="center",
                        fontsize=8, color=tc,
                        fontweight="bold" if abs(val) > 0.5 else "normal")

    for k in range(n + 1):
        ax.axhline(k - 0.5, color=_DARK_BG, linewidth=0.7)
        ax.axvline(k - 0.5, color=_DARK_BG, linewidth=0.7)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color=_DIM)
    cbar.outline.set_edgecolor(_GRID)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=_DIM, fontsize=8)
    cbar.set_label("Spearman ρ" if method == "spearman" else "Pearson r",
                   color=_DIM, fontsize=9)

    ax.set_title(
        f"ASSET CORRELATION MATRIX   ·   {days}d   ·   {n} assets   ·   {method}",
        color=_TEXT, fontsize=13, fontweight="bold", pad=16,
    )
    fig.text(0.01, 0.01, "* p<0.05  ** p<0.01  *** p<0.001", color=_DIM, fontsize=8)
    fig.text(0.99, 0.01, "T-Invest · Portfolio Analytics",
             ha="right", color=_DIM, fontsize=8, style="italic")

    plt.tight_layout()
    RESULTS.mkdir(exist_ok=True)
    fig.savefig(OUT_PNG, dpi=180, bbox_inches="tight", facecolor=_DARK_BG)
    plt.close(fig)
    print(f"✓ Heatmap    → {OUT_PNG}")


# ── CLI ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Portfolio correlation matrix via T-Invest API"
    )
    parser.add_argument(
        "days", nargs="?", type=int, default=730,
        help="Lookback window in calendar days (default: 730 = 2 years)",
    )
    parser.add_argument("--top",      type=int, default=None, help="Use only top-N assets by portfolio value")
    parser.add_argument("--spearman", action="store_true",    help="Use Spearman instead of Pearson correlation")
    args = parser.parse_args()

    method = "spearman" if args.spearman else "pearson"

    df, figi_map = load_snapshot(top_n=args.top)
    if len(figi_map) < 2:
        print("⚠  Need at least 2 shares/ETFs in portfolio.")
        return

    figi_map[CURRENCY_FIGI["usd"]] = "USDRUB"

    print(f"Found {len(figi_map)} assets: {', '.join(figi_map.values())}")
    print(f"Fetching {args.days}-day price history from T-Invest…")

    history = fetch_price_history(figi_map, days=args.days)
    if len(history) < 2:
        print("⚠  Not enough price history loaded.")
        return

    print("Fetching Brent, Nutrien from Yahoo Finance…")
    history.update(fetch_external_prices(args.days))

    print("Fetching IMOEX, RTSI from MOEX ISS…")
    history.update(fetch_moex_indices(args.days))

    # Strip timezone from all series — T-Invest returns tz-aware UTC, yfinance varies
    history = {
        t: pd.Series(s.values, index=(
            s.index.tz_convert("UTC").tz_localize(None) if s.index.tz is not None else s.index
        ).normalize())
        for t, s in history.items()
    }

    try:
        returns = build_returns_matrix(history)
    except ValueError as e:
        print(f"⚠  {e}")
        return

    corr, pval = compute_correlation(returns, method=method)
    weights    = portfolio_weights_from_csv(df, list(corr.columns))

    print_report(corr, pval, weights, args.days, method)
    export_csv(corr, pval, weights)
    plot_heatmap(corr, pval, args.days, method)


if __name__ == "__main__":
    main()
