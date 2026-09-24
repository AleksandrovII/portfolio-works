"""
cc.py  —  Completion Portfolio + Income Efficient Frontier
----------------------------------------------------------
Reads data/portfolio.csv, resolves missing FIGIs (Finam assets), fetches
price history and dividend data from T-Invest, then:

  1. Finds satellite weights that complete the portfolio around a fixed core
     (minimise variance OR maximise Sharpe).

  2. With --frontier: computes and plots the Income Efficient Frontier —
     the tradeoff between dividend income and portfolio risk.

Usage:
    export INVEST_TOKEN='your_token'

    # Core = oil 25% + gold 25%, minimise variance for the rest:
    /opt/miniconda3/bin/python cc.py --core "SIBN:0.15,ROSN:0.10,GLDRUB_TOM:0.25"

    # Add income efficient frontier:
    /opt/miniconda3/bin/python cc.py --core "SIBN:0.15,ROSN:0.10,GLDRUB_TOM:0.25" --frontier

    # Maximise Sharpe, cap satellite assets at 20%:
    /opt/miniconda3/bin/python cc.py --core "SIBN:0.25,GLDRUB_TOM:0.25" \\
        --objective max_sharpe --max-weight 0.20
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys
from datetime import timedelta
from typing import Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

_ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(_ROOT))

from portfolio_works_library import (
    TOKEN,
    build_figi_map_from_csv,
    build_returns_matrix,
    fetch_price_history,
    portfolio_weights_from_csv,
)

SNAPSHOT  = _ROOT / "data" / "portfolio.csv"
IMAGES    = _ROOT / "data" / "images"
OUT_CC    = IMAGES / "completion_portfolio.png"
OUT_IEF   = IMAGES / "income_frontier.png"

RISK_FREE_RATE = 0.16
INCLUDE_TYPES  = {"share", "etf", "precious_metal"}

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Dark theme ──────────────────────────────────────────────────────────────────

_DARK_BG  = "#0D1117"
_PANEL_BG = "#161B22"
_TEXT     = "#E6EDF3"
_DIM      = "#8B949E"
_GRID     = "#21262D"
_ACCENT1  = "#58A6FF"
_WARN     = "#F78166"
_GOLD     = "#FFD700"
_CORE_CLR = "#D29922"


def _dark_theme() -> None:
    plt.rcParams.update({
        "figure.facecolor": _DARK_BG, "axes.facecolor":  _PANEL_BG,
        "axes.edgecolor":   _GRID,    "axes.labelcolor":  _DIM,
        "axes.titlecolor":  _TEXT,    "xtick.color":      _DIM,
        "ytick.color":      _DIM,     "grid.color":       _GRID,
        "grid.linewidth":   0.5,      "text.color":       _TEXT,
        "legend.facecolor": _PANEL_BG,"legend.edgecolor": _GRID,
        "legend.labelcolor":_TEXT,
    })


# ── Finam FIGI resolution ───────────────────────────────────────────────────────

def resolve_finam_figis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Finam assets in portfolio.csv have empty FIGI fields.
    Try to resolve them via T-Invest find_instrument using the ISIN/ticker
    (stripping the @MISX suffix that Finam appends).
    """
    from t_tech.invest import Client

    missing = df["figi"].isna() | (df["figi"] == "")
    if not missing.any():
        return df

    print(f"Resolving FIGIs for {missing.sum()} Finam asset(s) via T-Invest…")
    df = df.copy()

    with Client(TOKEN) as client:
        for idx in df[missing].index:
            raw = str(df.at[idx, "ticker"])
            query = raw.split("@")[0].strip()
            try:
                resp = client.instruments.find_instrument(query=query)
                for inst in resp.instruments:
                    figi  = getattr(inst, "figi", None) or getattr(inst, "id", None)
                    itype = getattr(inst, "instrument_type", "")
                    if figi and itype != "unknown":
                        df.at[idx, "figi"]   = figi
                        df.at[idx, "ticker"] = getattr(inst, "ticker", query)
                        name = getattr(inst, "name", "")
                        if name:
                            df.at[idx, "name"] = name
                        logger.info("Resolved %-20s → %-10s (FIGI: %s)",
                                    query, df.at[idx, "ticker"], figi)
                        break
                else:
                    logger.warning("Could not resolve FIGI for %s", query)
            except Exception as e:
                logger.warning("FIGI lookup failed for %s: %s", query, e)

    return df


# ── Portfolio loading ───────────────────────────────────────────────────────────

def load_snapshot(
    top_n: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, str]]:
    """
    Load portfolio.csv. Resolves Finam FIGIs, excludes futures and derivatives.
    Returns (df, figi_map {figi→ticker}, name_map {ticker→name}).
    """
    if not SNAPSHOT.exists():
        sys.exit(f"Portfolio snapshot not found: {SNAPSHOT}\nRun call_api.py first.")

    df = pd.read_csv(SNAPSHOT)
    df["rub_value"]  = pd.to_numeric(df["rub_value"],  errors="coerce").fillna(0)
    df["weight_pct"] = pd.to_numeric(df["weight_pct"], errors="coerce").fillna(0)

    df = resolve_finam_figis(df)

    df = df[~df["figi"].str.startswith("FUT", na=False)]
    df = df[df["instrument_type"].isin(INCLUDE_TYPES)]

    figi_map = build_figi_map_from_csv(df, top_n=top_n)
    name_map = df.drop_duplicates("ticker").set_index("ticker")["name"].to_dict()
    return df, figi_map, name_map


# ── Core spec parser ────────────────────────────────────────────────────────────

def parse_core(spec: str) -> Dict[str, float]:
    """Parse 'SIBN:0.15,ROSN:0.10,GLDRUB_TOM:0.25' → {ticker: weight (0–1)}."""
    core: Dict[str, float] = {}
    for part in spec.split(","):
        part = part.strip()
        if ":" not in part:
            sys.exit(f"Bad core spec '{part}'. Use TICKER:weight, e.g. SIBN:0.15")
        ticker, raw = part.split(":", 1)
        ticker = ticker.strip().upper()
        try:
            w = float(raw.strip())
        except ValueError:
            sys.exit(f"Invalid weight '{raw}' for '{ticker}'")
        if not 0 < w < 1:
            sys.exit(f"Weight for '{ticker}' must be between 0 and 1 (got {w})")
        core[ticker] = w

    if sum(core.values()) >= 1.0:
        sys.exit(
            f"Core weights sum to {sum(core.values()):.1%} — must be < 100% "
            "to leave room for satellite."
        )
    return core


# ── Statistics helper ───────────────────────────────────────────────────────────

def _stats(
    w: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
) -> Tuple[float, float, float]:
    r = float(w @ mu)
    v = float(np.sqrt(np.maximum(w @ cov @ w, 0.0)))
    s = (r - RISK_FREE_RATE) / v if v > 1e-9 else 0.0
    return r, v, s


# ── MOEX ISS fallback for assets without FIGI ──────────────────────────────────

def fetch_empty_figi_history(
    df: pd.DataFrame,
    days: int,
    skip_tickers: Optional[Set[str]] = None,
) -> Dict[str, pd.Series]:
    """
    Fetch MOEX ISS price history for assets that are missing from T-Invest history.

    skip_tickers: set of ticker labels already successfully loaded — these are skipped.
    If skip_tickers is None, only processes rows with empty/NaN FIGI (legacy mode).

    Returns {original_ticker_label: pd.Series} so weight lookups against df stay intact.
    """
    import time
    import requests as _req
    from datetime import date, timedelta

    if skip_tickers is not None:
        # Fetch MOEX for any asset not already in the T-Invest history
        missing = df[~df["ticker"].isin(skip_tickers)].drop_duplicates("ticker")
    else:
        # Legacy: only assets with missing FIGI
        missing = df[df["figi"].isna() | (df["figi"] == "")].drop_duplicates("ticker")

    if missing.empty:
        return {}

    result: Dict[str, pd.Series] = {}
    end_date   = date.today().isoformat()
    start_date = (date.today() - timedelta(days=days)).isoformat()

    for _, row in missing.iterrows():
        original_ticker = str(row["ticker"])       # e.g. "RU000A10CFM8@MISX"
        isin            = original_ticker.split("@")[0].strip()

        # Step 1: find primary board on MOEX
        try:
            resp = _req.get(
                f"https://iss.moex.com/iss/securities/{isin}.json", timeout=10
            )
            if resp.status_code != 200:
                logger.warning("MOEX: security not found — %s", isin)
                continue
            meta = resp.json()
        except Exception as e:
            logger.warning("MOEX: search error for %s: %s", isin, e)
            continue

        boards_data = meta.get("boards", {})
        cols        = boards_data.get("columns", [])
        records     = boards_data.get("data", [])

        try:
            secid_i    = cols.index("SECID")
            boardid_i  = cols.index("BOARDID")
            engine_i   = cols.index("engine")
            market_i   = cols.index("market")
            primary_i  = cols.index("is_primary")
        except ValueError:
            logger.warning("MOEX: unexpected board schema for %s", isin)
            continue

        engine = market = board = secid = None
        for b in records:
            if b[primary_i] == 1:
                engine, market, board, secid = (
                    b[engine_i], b[market_i], b[boardid_i], b[secid_i]
                )
                break

        if not engine:
            logger.warning("MOEX: no primary board for %s", isin)
            continue

        logger.info("  %-24s → engine=%-8s market=%-8s board=%s",
                    isin, engine, market, board)

        # Step 2: fetch historical prices with pagination
        url  = (
            f"https://iss.moex.com/iss/history/engines/{engine}"
            f"/markets/{market}/boards/{board}/securities/{secid}.json"
        )
        rows: List[Tuple[pd.Timestamp, float]] = []
        start_row = 0

        while True:
            try:
                r = _req.get(url, params={
                    "from": start_date, "till": end_date,
                    "start": start_row, "limit": 100,
                }, timeout=15)
                if r.status_code != 200:
                    break
                payload = r.json()
            except Exception as e:
                logger.warning("MOEX: history error for %s: %s", isin, e)
                break

            hist    = payload.get("history", {})
            h_cols  = hist.get("columns", [])
            records = hist.get("data", [])
            if not records:
                break

            # Prefer CLOSE, fall back to VALUE (indices)
            price_col = "CLOSE" if "CLOSE" in h_cols else "VALUE"
            try:
                d_idx = h_cols.index("TRADEDATE")
                p_idx = h_cols.index(price_col)
            except ValueError:
                break

            for rec in records:
                d, p = rec[d_idx], rec[p_idx]
                if d and p is not None:
                    rows.append((pd.Timestamp(d), float(p)))

            cursor = payload.get("history.cursor", {}).get("data", [])
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
            s = s[~s.index.duplicated(keep="last")]
            result[original_ticker] = s
            logger.info("  %-24s %d days loaded (MOEX ISS fallback)", original_ticker, len(s))
        else:
            logger.warning("  %-24s insufficient history on MOEX ISS", isin)

    return result


# ── Completion Portfolio optimisation ───────────────────────────────────────────

def optimize_completion(
    returns: pd.DataFrame,
    core_weights: Dict[str, float],
    objective: str = "min_var",
    max_sat_weight: float = 1.0,
) -> Dict:
    """
    Fix core allocations, optimise satellite weights.
    objective: 'min_var' | 'max_sharpe'
    """
    tickers    = list(returns.columns)
    mu         = returns.mean().values * 252
    cov        = returns.cov().values  * 252
    ticker_idx = {t: i for i, t in enumerate(tickers)}

    core_in  = {t: w for t, w in core_weights.items() if t in ticker_idx}
    core_out = [t for t in core_weights if t not in ticker_idx]
    if core_out:
        print(f"⚠  Core tickers missing from price history: {', '.join(core_out)}")

    sat_tickers = [t for t in tickers if t not in core_in]
    if not sat_tickers:
        sys.exit("No satellite assets remain after fixing the core.")

    w_fixed  = np.zeros(len(tickers))
    sat_idx  = []
    for t, w in core_in.items():
        w_fixed[ticker_idx[t]] = w
    for t in sat_tickers:
        sat_idx.append(ticker_idx[t])

    sat_budget = 1.0 - sum(core_in.values())
    cap        = min(max_sat_weight, sat_budget)
    n_sat      = len(sat_idx)

    def make_w(w_sat: np.ndarray) -> np.ndarray:
        w = w_fixed.copy()
        for i, idx in enumerate(sat_idx):
            w[idx] = w_sat[i]
        return w

    bounds      = [(0.0, cap)] * n_sat
    constraints = {"type": "eq", "fun": lambda w: w.sum() - sat_budget}
    init        = np.full(n_sat, sat_budget / n_sat)

    if objective == "min_var":
        obj = lambda ws: float(make_w(ws) @ cov @ make_w(ws))
    else:
        obj = lambda ws: -_stats(make_w(ws), mu, cov)[2]

    res   = minimize(obj, init, method="SLSQP", bounds=bounds,
                     constraints=constraints, options={"ftol": 1e-10, "maxiter": 1000})
    w_opt = make_w(res.x)
    r, v, s = _stats(w_opt, mu, cov)

    return {
        "weights":      pd.Series(w_opt, index=tickers),
        "core_tickers": list(core_in.keys()),
        "sat_tickers":  sat_tickers,
        "sat_budget":   sat_budget,
        "return":       r,
        "volatility":   v,
        "sharpe":       s,
        "objective":    objective,
        "converged":    res.success,
        # keep for frontier
        "_mu":          mu,
        "_cov":         cov,
        "_w_fixed":     w_fixed,
        "_sat_idx":     sat_idx,
    }


# ── Dividend yields ─────────────────────────────────────────────────────────────

def fetch_dividend_yields(figi_map: Dict[str, str], days: int = 365) -> Dict[str, float]:
    """
    Fetch last-year dividends from T-Invest and compute annual dividend yield
    (annual_dividends / current_price) for each asset.
    Returns {ticker: yield_fraction}.  Missing or zero → 0.0.
    """
    from t_tech.invest import Client
    from t_tech.invest.utils import now

    yields: Dict[str, float] = {ticker: 0.0 for ticker in figi_map.values()}
    end   = now()
    start = end - timedelta(days=days)

    with Client(TOKEN) as client:
        for figi, ticker in figi_map.items():
            try:
                # Current price
                pr = client.market_data.get_last_prices(figi=[figi])
                if not pr.last_prices:
                    continue
                q     = pr.last_prices[0].price
                price = float(q.units) + float(q.nano) / 1e9
                if price <= 0:
                    continue

                # Dividends over the past year
                divs = client.instruments.get_dividends(figi=figi, from_=start, to=end)
                annual = sum(
                    float(d.dividend_net.units) + float(d.dividend_net.nano) / 1e9
                    for d in divs.dividends
                    if d.dividend_net
                )
                if annual > 0:
                    yields[ticker] = annual / price
                    logger.info("  %-14s div yield = %.2f%%", ticker, yields[ticker] * 100)
                else:
                    logger.info("  %-14s no dividends found", ticker)

            except Exception as e:
                logger.warning("  %-14s dividend fetch error: %s", ticker, e)

    return yields


# ── Income Efficient Frontier ───────────────────────────────────────────────────

def compute_income_frontier(
    returns: pd.DataFrame,
    div_yields: Dict[str, float],
    core_weights: Dict[str, float],
    max_sat_weight: float = 1.0,
    n_points: int = 35,
) -> List[Dict]:
    """
    Compute the Income Efficient Frontier: for income targets ranging from
    min to max, find the minimum-variance portfolio with core fixed.

    Returns list of {vol, income, monthly_yield, return, sharpe, weights}.
    """
    tickers    = list(returns.columns)
    mu         = returns.mean().values * 252
    cov        = returns.cov().values  * 252
    ticker_idx = {t: i for i, t in enumerate(tickers)}
    dy         = np.array([div_yields.get(t, 0.0) for t in tickers])

    core_in    = {t: w for t, w in core_weights.items() if t in ticker_idx}
    sat_idx    = [ticker_idx[t] for t in tickers if t not in core_in]

    w_fixed    = np.zeros(len(tickers))
    for t, w in core_in.items():
        w_fixed[ticker_idx[t]] = w

    sat_budget = 1.0 - sum(core_in.values())
    cap        = min(max_sat_weight, sat_budget)
    n_sat      = len(sat_idx)
    init       = np.full(n_sat, sat_budget / n_sat)
    bounds     = [(0.0, cap)] * n_sat
    eq_con     = {"type": "eq", "fun": lambda ws: ws.sum() - sat_budget}

    def make_w(ws: np.ndarray) -> np.ndarray:
        w = w_fixed.copy()
        for i, idx in enumerate(sat_idx):
            w[idx] = ws[i]
        return w

    def port_var(ws):
        w = make_w(ws)
        return float(w @ cov @ w)

    # Max income portfolio
    res_max = minimize(
        lambda ws: -(dy @ make_w(ws)),
        init, method="SLSQP", bounds=bounds, constraints=eq_con,
    )
    # Min variance portfolio (may have low income)
    res_mv = minimize(
        port_var, init, method="SLSQP", bounds=bounds, constraints=eq_con,
    )

    if not res_max.success or not res_mv.success:
        logger.warning("Frontier pre-solve did not converge.")
        return []

    max_income = float(dy @ make_w(res_max.x))
    min_income = float(dy @ make_w(res_mv.x))

    if max_income - min_income < 1e-5:
        print("⚠  Income range too narrow for a meaningful frontier "
              "(most assets have zero dividend yield).")
        return []

    frontier: List[Dict] = []
    for target in np.linspace(min_income, max_income, n_points):
        cons = [
            eq_con,
            {"type": "ineq", "fun": lambda ws, t=target: float(dy @ make_w(ws)) - t},
        ]
        res = minimize(port_var, init, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"ftol": 1e-9, "maxiter": 500})
        if not res.success:
            continue
        w       = make_w(res.x)
        r, v, s = _stats(w, mu, cov)
        income  = float(dy @ w)
        frontier.append({
            "vol":     v,
            "income":  income,   # annual yield fraction
            "return":  r,
            "sharpe":  s,
            "weights": pd.Series(w, index=tickers),
        })

    return frontier


# ── Console report ──────────────────────────────────────────────────────────────

def print_report(
    opt: Dict,
    actual_weights: pd.Series,
    name_map: Dict[str, str],
    div_yields: Optional[Dict[str, float]] = None,
    portfolio_value: float = 0.0,
) -> None:
    w      = opt["weights"]
    core_t: Set[str] = set(opt["core_tickers"])

    dy = div_yields or {}
    opt_income  = float(sum(w[t] * dy.get(t, 0.0) for t in w.index))
    act_income  = float(sum((actual_weights.get(t, 0.0) / 100) * dy.get(t, 0.0)
                            for t in actual_weights.index))

    print()
    print("═" * 76)
    print("  COMPLETION PORTFOLIO")
    obj_label = "Minimize Variance" if opt["objective"] == "min_var" else "Maximize Sharpe"
    print(f"  Objective  : {obj_label}")
    core_str = "  +  ".join(f"{t} {w[t]*100:.1f}%" for t in opt["core_tickers"])
    print(f"  Core       : {core_str}  →  total {sum(w[t] for t in opt['core_tickers'])*100:.1f}%")
    print(f"  Satellite  : {opt['sat_budget']*100:.1f}%  ({len(opt['sat_tickers'])} assets)")
    print("═" * 76)
    print(f"  Annualised return     : {opt['return']*100:+.2f}%")
    print(f"  Annualised volatility : {opt['volatility']*100:.2f}%")
    print(f"  Sharpe ratio          : {opt['sharpe']:.3f}  (Rf = {RISK_FREE_RATE*100:.1f}%)")
    if dy:
        monthly_opt = opt_income * portfolio_value / 12
        monthly_act = act_income * portfolio_value / 12
        print(f"  Annual div yield      : {opt_income*100:.2f}%  "
              f"→  {monthly_opt:,.0f} RUB/month")
        print(f"  Actual div yield      : {act_income*100:.2f}%  "
              f"→  {monthly_act:,.0f} RUB/month")
    print()
    print(f"  {'Ticker':<14} {'Name':<26} {'Role':<7} "
          f"{'Div%':>5}  {'Optimal':>8}  {'Actual':>8}  {'Delta':>8}")
    print("  " + "─" * 76)

    for t in sorted(w.index, key=lambda x: -w[x]):
        if w[t] < 0.0005 and actual_weights.get(t, 0.0) < 0.05:
            continue
        role   = "CORE" if t in core_t else "sat"
        name   = name_map.get(t, "")[:25]
        actual = actual_weights.get(t, 0.0) / 100
        delta  = w[t] - actual
        arrow  = "↑" if delta > 0.005 else ("↓" if delta < -0.005 else "~")
        dy_pct = f"{dy.get(t, 0.0)*100:.1f}" if dy else "—"
        print(
            f"  {t:<14} {name:<26} {role:<7} "
            f"{dy_pct:>5}  {w[t]*100:>7.1f}%  {actual*100:>7.1f}%  "
            f"{arrow}{abs(delta)*100:.1f}pp"
        )

    if not opt["converged"]:
        print("\n  ⚠  Optimiser did not fully converge — results may be suboptimal.")
    print("═" * 76)


# ── Completion portfolio chart ──────────────────────────────────────────────────

def plot_results(
    opt: Dict,
    actual_weights: pd.Series,
    name_map: Dict[str, str],
) -> None:
    _dark_theme()
    w      = opt["weights"]
    core_t: Set[str] = set(opt["core_tickers"])

    tickers = [
        t for t in sorted(w.index, key=lambda x: -w[x])
        if w[t] >= 0.005 or actual_weights.get(t, 0.0) / 100 >= 0.005
    ]

    n   = len(tickers)
    x   = np.arange(n)
    fig, ax = plt.subplots(figsize=(max(12, n * 1.2), 6), facecolor=_DARK_BG)
    ax.set_facecolor(_PANEL_BG)

    w_opt    = [w[t] * 100 for t in tickers]
    w_act    = [actual_weights.get(t, 0.0) for t in tickers]
    bar_clrs = [_CORE_CLR if t in core_t else _ACCENT1 for t in tickers]

    bars = ax.bar(x - 0.2, w_opt, 0.38, color=bar_clrs, alpha=0.9,
                  label="Completion Portfolio")
    ax.bar(x + 0.2, w_act, 0.38, color=_WARN, alpha=0.75, label="Actual")

    for bar in bars:
        h = bar.get_height()
        if h >= 1.0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                    f"{h:.1f}%", ha="center", va="bottom", fontsize=7, color=_TEXT)

    labels = [
        f"{t}\n{name_map.get(t,'')[:14]}" + ("\n[CORE]" if t in core_t else "")
        for t in tickers
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Weight (%)", fontsize=9)

    obj_label = "Min Variance" if opt["objective"] == "min_var" else "Max Sharpe"
    ax.set_title(
        f"COMPLETION PORTFOLIO  ·  {obj_label}  ·  "
        f"Return {opt['return']*100:+.1f}%  "
        f"Vol {opt['volatility']*100:.1f}%  "
        f"Sharpe {opt['sharpe']:.2f}",
        color=_TEXT, fontsize=11, fontweight="bold",
    )

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=_CORE_CLR, label="Core (fixed)"),
        Patch(facecolor=_ACCENT1,  label="Satellite (optimised)"),
        Patch(facecolor=_WARN,     label="Actual"),
    ], fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.15)

    budget_pct = opt["sat_budget"] * 100
    ax.axhline(budget_pct, color=_DIM, linewidth=0.8, linestyle="--")
    ax.text(n - 0.3, budget_pct + 0.4, f"satellite cap {budget_pct:.0f}%",
            ha="right", fontsize=7, color=_DIM)

    plt.tight_layout()
    IMAGES.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_CC, dpi=180, bbox_inches="tight", facecolor=_DARK_BG)
    plt.close(fig)
    print(f"✓ Completion chart → {OUT_CC}")


# ── Income Efficient Frontier chart ────────────────────────────────────────────

def plot_income_frontier(
    frontier: List[Dict],
    returns: pd.DataFrame,
    div_yields: Dict[str, float],
    actual_weights_frac: pd.Series,
    opt: Dict,
    portfolio_value: float,
) -> None:
    if not frontier:
        return

    _dark_theme()
    mu  = returns.mean().values * 252
    cov = returns.cov().values  * 252

    def portfolio_point(w_series: pd.Series) -> Tuple[float, float, float]:
        w      = w_series.reindex(returns.columns).fillna(0).values
        dy     = np.array([div_yields.get(t, 0.0) for t in returns.columns])
        _, v, _ = _stats(w, mu, cov)
        income = float(w @ dy)
        monthly = income * portfolio_value / 12
        return v * 100, income * 100, monthly / 1000  # vol%, yield%, kRUB/mo

    act_v, act_y, act_m = portfolio_point(actual_weights_frac)
    opt_v, opt_y, opt_m = portfolio_point(opt["weights"])

    vols    = [p["vol"] * 100  for p in frontier]
    yields  = [p["income"] * 100 for p in frontier]
    monthly = [p["income"] * portfolio_value / 12 / 1000 for p in frontier]
    sharpes = [p["sharpe"] for p in frontier]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), facecolor=_DARK_BG)
    fig.suptitle("INCOME EFFICIENT FRONTIER", color=_TEXT, fontsize=14, fontweight="bold")

    for ax in (ax1, ax2):
        ax.set_facecolor(_PANEL_BG)
        ax.grid(True, alpha=0.15)

    # ── Left: annual yield % vs volatility % ──────────────────────────────────
    sc = ax1.scatter(vols, yields, c=sharpes, cmap="viridis", s=18, alpha=0.85, zorder=3)
    fig.colorbar(sc, ax=ax1, label="Sharpe", fraction=0.03, pad=0.02)

    ax1.scatter([act_v], [act_y], marker="o", s=200, color=_WARN, zorder=6,
                edgecolors="white", linewidths=0.5, label=f"Actual  ({act_y:.1f}%)")
    ax1.scatter([opt_v], [opt_y], marker="*", s=300, color=_GOLD, zorder=6,
                edgecolors="white", linewidths=0.5,
                label=f"Completion  ({opt_y:.1f}%)")

    ax1.set_xlabel("Annual Volatility (%)", fontsize=9)
    ax1.set_ylabel("Annual Dividend Yield (%)", fontsize=9)
    ax1.set_title("Yield vs Risk", color=_TEXT, fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8)

    # ── Right: monthly income kRUB vs volatility % ────────────────────────────
    ax2.scatter(vols, monthly, c=sharpes, cmap="viridis", s=18, alpha=0.85, zorder=3)

    ax2.scatter([act_v], [act_m], marker="o", s=200, color=_WARN, zorder=6,
                edgecolors="white", linewidths=0.5,
                label=f"Actual  {act_m:.0f}k RUB/mo")
    ax2.scatter([opt_v], [opt_m], marker="*", s=300, color=_GOLD, zorder=6,
                edgecolors="white", linewidths=0.5,
                label=f"Completion  {opt_m:.0f}k RUB/mo")

    ax2.set_xlabel("Annual Volatility (%)", fontsize=9)
    ax2.set_ylabel("Monthly Dividend Income (kRUB)", fontsize=9)
    ax2.set_title("Monthly Income vs Risk", color=_TEXT, fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8)

    fig.text(0.99, 0.01, "T-Invest · Completion Portfolio Analytics",
             ha="right", fontsize=8, color=_DIM, style="italic")

    plt.tight_layout()
    IMAGES.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_IEF, dpi=180, bbox_inches="tight", facecolor=_DARK_BG)
    plt.close(fig)
    print(f"✓ Income frontier → {OUT_IEF}")


# ── CLI ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Completion Portfolio + Income Efficient Frontier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cc.py --core "SIBN:0.15,ROSN:0.10,GLDRUB_TOM:0.25"
  python cc.py --core "SIBN:0.15,ROSN:0.10,GLDRUB_TOM:0.25" --frontier
  python cc.py --core "SIBN:0.25,GLDRUB_TOM:0.25" --objective max_sharpe --max-weight 0.20
""",
    )
    parser.add_argument(
        "--core", required=True,
        help='Fixed core: TICKER:weight pairs, e.g. "SIBN:0.15,ROSN:0.10,GLDRUB_TOM:0.25"',
    )
    parser.add_argument(
        "--objective", choices=["min_var", "max_sharpe"], default="min_var",
    )
    parser.add_argument("--days",       type=int,   default=365, help="History window in days")
    parser.add_argument("--max-weight", type=float, default=1.0, dest="max_weight",
                        help="Max weight for any single satellite asset (0–1)")
    parser.add_argument("--top",        type=int,   default=None,
                        help="Consider only top-N assets by portfolio value")
    parser.add_argument("--frontier",   action="store_true",
                        help="Compute and plot Income Efficient Frontier")
    args = parser.parse_args()

    core_weights = parse_core(args.core)

    df, figi_map, name_map = load_snapshot(top_n=args.top)
    if len(figi_map) < 2:
        print("⚠  Not enough tradable assets.")
        return

    print(f"\nAssets        : {', '.join(figi_map.values())}")
    print(f"Core (fixed)  : {', '.join(f'{t}={w*100:.0f}%' for t, w in core_weights.items())}")
    print(f"Satellite     : {(1 - sum(core_weights.values()))*100:.0f}%")
    print(f"\nFetching {args.days}-day price history from T-Invest…")

    history = fetch_price_history(figi_map, days=args.days)

    # MOEX ISS fallback: for any asset in portfolio not loaded via T-Invest
    # (covers empty-FIGI Finam assets AND assets with FIGI but no candle history)
    loaded_tickers = set(history.keys())
    moex_fallback  = fetch_empty_figi_history(df, days=args.days, skip_tickers=loaded_tickers)
    if moex_fallback:
        print(f"MOEX ISS fallback loaded: {', '.join(moex_fallback.keys())}")
        history.update(moex_fallback)

    if len(history) < 2:
        print("⚠  Not enough price history.")
        return

    try:
        returns = build_returns_matrix(history)
    except ValueError as e:
        print(f"⚠  {e}")
        return

    opt = optimize_completion(
        returns,
        core_weights=core_weights,
        objective=args.objective,
        max_sat_weight=args.max_weight,
    )

    actual_weights = portfolio_weights_from_csv(df, list(returns.columns))
    portfolio_value = float(df["rub_value"].sum())

    # ── Dividend data (always fetched, used in report + optionally frontier) ───
    print("\nFetching dividend data from T-Invest…")
    div_yields = fetch_dividend_yields(figi_map, days=args.days)

    print_report(opt, actual_weights, name_map, div_yields, portfolio_value)
    plot_results(opt, actual_weights, name_map)

    # ── Income Efficient Frontier ─────────────────────────────────────────────
    if args.frontier:
        print("\nComputing Income Efficient Frontier…")
        actual_frac = actual_weights / 100
        frontier = compute_income_frontier(
            returns, div_yields, core_weights,
            max_sat_weight=args.max_weight,
        )
        if frontier:
            print(f"Frontier: {len(frontier)} points computed.")
            plot_income_frontier(
                frontier, returns, div_yields,
                actual_frac, opt, portfolio_value,
            )
        else:
            print("⚠  Could not compute frontier.")


if __name__ == "__main__":
    main()
