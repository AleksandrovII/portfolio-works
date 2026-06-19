"""
portfolio_works_library.py
--------------------------
Core library: T-Invest API connection and portfolio data extraction.

All scripts import from here — never instantiate API clients directly.

    export INVEST_TOKEN='your_token'
"""

import asyncio
import logging
import os
from datetime import timedelta
from decimal import Decimal
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from t_tech.invest import AsyncClient, Client, CandleInterval
from t_tech.invest.utils import now

logger = logging.getLogger(__name__)

TOKEN: str = os.getenv("INVEST_TOKEN", "")
if not TOKEN:
    raise SystemExit("INVEST_TOKEN not set. Run:  export INVEST_TOKEN='your_token'")

# ── Constants ──────────────────────────────────────────────────────────────────

CURRENCY_FIGI: Dict[str, str] = {
    "usd": "BBG0013HGFT4",
    "eur": "BBG0013HJJ31",
    "cny": "BBG0013HRTL0",
    "hkd": "BBG0013HSW87",
    "chf": "BBG0013HQ5K4",
}

# Instrument types used in price-history / Markowitz analysis
TRADABLE_TYPES = {"share", "bond", "etf", "precious_metal"}

ReturnMethod = Literal["log", "simple"]
CorrMethod   = Literal["pearson", "spearman"]


# ── Low-level helpers ──────────────────────────────────────────────────────────

def _q(q) -> Decimal:
    """Convert T-Invest Quotation proto to Decimal."""
    return Decimal(q.units) + Decimal(q.nano) / Decimal(1_000_000_000)


# ── Currency rates ─────────────────────────────────────────────────────────────

def get_rates(client: Client) -> Dict[str, Decimal]:
    """Fetch current exchange rates to RUB for all major currencies."""
    rates: Dict[str, Decimal] = {"rub": Decimal("1")}
    for curr, figi in CURRENCY_FIGI.items():
        try:
            resp = client.market_data.get_last_prices(figi=[figi])
            if resp.last_prices:
                rates[curr] = _q(resp.last_prices[0].price)
        except Exception as e:
            logger.warning("Could not fetch rate for %s: %s", curr.upper(), e)
            rates[curr] = Decimal("1")
    return rates


# ── Instrument metadata ────────────────────────────────────────────────────────

def get_instrument_info(client: Client, figi: str) -> Optional[Dict]:
    """
    Fetch instrument metadata by FIGI.

    Returns None for derivatives (instrument_type == 'unknown') so callers
    can skip them.  Gold/metals are reclassified to 'precious_metal'.
    """
    if not figi or not figi.strip():
        return None
    try:
        resp = client.instruments.find_instrument(query=figi)
        for inst in resp.instruments:
            inst_figi = getattr(inst, "figi", getattr(inst, "id", None))
            if inst_figi != figi:
                continue
            itype = getattr(inst, "instrument_type", "N/A")
            if itype == "unknown":
                return None
            name = getattr(inst, "name", "Unknown")
            if any(x in name.lower() for x in ("золот", "gold", "metal")):
                itype = "precious_metal"
            return {
                "name":            name,
                "ticker":          getattr(inst, "ticker", figi[:10]),
                "currency":        getattr(inst, "currency", "rub"),
                "instrument_type": itype,
            }
    except Exception as e:
        logger.error("Error fetching instrument %s: %s", figi, e)
    return None


# ── Portfolio positions ────────────────────────────────────────────────────────

def fetch_all_positions(
    client: Client,
    rates: Dict[str, Decimal],
) -> List[Dict]:
    """
    Return a flat list of all position dicts across every account.

    Each dict contains:
        account, figi, name, ticker, currency, instrument_type,
        quantity, avg_price, cur_price, rub_value

    Skips:
        - derivative/unknown instruments (get_instrument_info returns None)
        - options heuristic: unnamed positions with negative quantity
    """
    try:
        accounts = client.users.get_accounts().accounts
    except Exception as e:
        logger.error("Cannot fetch accounts: %s", e)
        return []

    rows: List[Dict] = []
    for account in accounts:
        try:
            portfolio = client.operations.get_portfolio(account_id=account.id)
        except Exception as e:
            logger.warning("Skipping account %s: %s", account.name, e)
            continue

        for pos in portfolio.positions:
            info = get_instrument_info(client, pos.figi)
            if info is None:
                logger.debug("Skipping derivative: %s", pos.figi)
                continue

            qty = _q(pos.quantity) if pos.quantity else Decimal("0")

            name_unknown = info["name"] in ("", "Unknown", "Неизвестно", "N/A")
            if name_unknown and qty < 0:
                logger.info("Skipping option (unknown name, qty<0): %s", pos.figi)
                continue

            avg_price: Optional[Decimal] = (
                _q(pos.average_position_price) if pos.average_position_price else None
            )
            cur_price: Optional[Decimal] = None
            currency = "rub"
            if pos.current_price:
                cur_price = _q(pos.current_price)
                currency  = pos.current_price.currency.lower()

            rub_value = Decimal("0")
            if cur_price and qty > 0:
                val  = cur_price * qty
                rate = rates.get(currency, Decimal("1")) if currency != "rub" else Decimal("1")
                rub_value = val * rate

            rows.append({
                "account":         account.name,
                "figi":            pos.figi,
                "name":            info["name"],
                "ticker":          info["ticker"],
                "currency":        currency,
                "instrument_type": info["instrument_type"],
                "quantity":        qty,
                "avg_price":       avg_price,
                "cur_price":       cur_price,
                "rub_value":       rub_value,
            })

    return rows


# ── Historical prices (async) ─────────────────────────────────────────────────

async def _fetch_candles(
    client: AsyncClient,
    figi: str,
    days: int,
) -> Optional[pd.Series]:
    """Fetch daily close prices as a UTC-date-indexed Series."""
    try:
        end   = now()
        start = end - timedelta(days=days)
        resp  = await client.market_data.get_candles(
            instrument_id=figi,
            from_=start,
            to=end,
            interval=CandleInterval.CANDLE_INTERVAL_DAY,
        )
        rows = []
        for c in resp.candles:
            if not c.close or not c.time:
                continue
            price = float(c.close.units) + float(c.close.nano) / 1e9
            rows.append((pd.Timestamp(c.time).normalize(), price))
        if len(rows) <= 10:
            return None
        series = pd.Series({ts: px for ts, px in rows}, dtype=float).sort_index()
        return series[~series.index.duplicated(keep="last")]
    except Exception as e:
        logger.warning("No candle history for %s: %s", figi, e)
        return None


async def _fetch_all_candles(
    figi_map: Dict[str, str],
    days: int,
) -> Dict[str, pd.Series]:
    async with AsyncClient(TOKEN) as client:
        figis   = list(figi_map.keys())
        results = await asyncio.gather(
            *[_fetch_candles(client, f, days) for f in figis],
            return_exceptions=True,
        )
    return {
        figi_map[figi]: res
        for figi, res in zip(figis, results)
        if isinstance(res, pd.Series)
    }


def fetch_price_history(
    figi_map: Dict[str, str],
    days: int = 365,
) -> Dict[str, pd.Series]:
    """
    Fetch daily close price history for a set of instruments in parallel.

    Args:
        figi_map: dict mapping FIGI → ticker label
        days:     lookback window in calendar days

    Returns:
        dict mapping ticker → pd.Series of daily close prices (date index)
    """
    return asyncio.run(_fetch_all_candles(figi_map, days))


# ── Portfolio construction helpers ─────────────────────────────────────────────

def build_figi_map(
    positions: List[Dict],
    min_weight_pct: float = 0.0,
    top_n: Optional[int] = None,
) -> Dict[str, str]:
    """
    Build a FIGI → ticker mapping for tradable positions only.

    Args:
        positions:      output of fetch_all_positions()
        min_weight_pct: exclude positions below this % of total NAV
        top_n:          keep only the N largest positions by RUB value
    """
    total_rub = sum(float(p.get("rub_value") or 0) for p in positions)
    candidates: List[Tuple[float, str, str]] = []

    for p in positions:
        if p.get("instrument_type") not in TRADABLE_TYPES:
            continue
        figi   = p.get("figi", "")
        ticker = p.get("ticker", "")
        if not figi or not ticker:
            continue
        rub    = float(p.get("rub_value") or 0)
        weight = (rub / total_rub * 100) if total_rub > 0 else 0.0
        if weight < min_weight_pct:
            continue
        candidates.append((rub, figi, ticker))

    if top_n is not None and top_n > 0:
        candidates.sort(key=lambda x: x[0], reverse=True)
        candidates = candidates[:top_n]

    figi_map: Dict[str, str] = {}
    seen: set = set()
    for _, figi, ticker in sorted(candidates, key=lambda x: x[0], reverse=True):
        if ticker in seen:
            continue
        figi_map[figi] = ticker
        seen.add(ticker)
    return figi_map


def build_figi_map_from_csv(
    df: pd.DataFrame,
    min_weight_pct: float = 0.0,
    top_n: Optional[int] = None,
) -> Dict[str, str]:
    """
    Build a FIGI → ticker mapping from a portfolio CSV snapshot.

    Args:
        df:             DataFrame loaded from data/portfolio.csv
        min_weight_pct: exclude positions below this % of total NAV
        top_n:          keep only the N largest positions by RUB value

    Returns:
        dict mapping FIGI → ticker (only tradable types)
    """
    tradable = df[df["instrument_type"].isin(TRADABLE_TYPES)].copy()
    if min_weight_pct > 0:
        tradable = tradable[tradable["weight_pct"] >= min_weight_pct]

    # Aggregate duplicate (figi, ticker) pairs across accounts
    agg = (
        tradable.groupby(["figi", "ticker"])["rub_value"]
        .sum()
        .reset_index()
        .sort_values("rub_value", ascending=False)
    )

    if top_n is not None and top_n > 0:
        agg = agg.head(top_n)

    figi_map: Dict[str, str] = {}
    seen: set = set()
    for _, row in agg.iterrows():
        if row["ticker"] in seen:
            continue
        figi_map[row["figi"]] = row["ticker"]
        seen.add(row["ticker"])
    return figi_map


def build_returns_matrix(
    history: Dict[str, pd.Series],
    method: ReturnMethod = "log",
) -> pd.DataFrame:
    """
    Align price series on common trading dates and compute returns.

    Args:
        history: ticker → price Series (output of fetch_price_history)
        method:  'log' for log-returns (default), 'simple' for pct_change
    """
    if len(history) < 2:
        raise ValueError("Need at least 2 assets with price history")
    prices = pd.DataFrame(history).dropna(how="any")
    if len(prices) < 11:
        raise ValueError(f"Only {len(prices)} overlapping trading days — need ≥ 11")
    if method == "log":
        return np.log(prices / prices.shift(1)).dropna()
    return prices.pct_change().dropna()


def portfolio_weights_from_csv(
    df: pd.DataFrame,
    tickers: List[str],
) -> pd.Series:
    """
    Compute RUB weight (%) per ticker from a portfolio CSV snapshot.

    Sums across all accounts for each ticker.
    """
    agg = (
        df[df["ticker"].isin(tickers)]
        .groupby("ticker")["rub_value"]
        .sum()
    )
    total = agg.sum()
    if total <= 0:
        return pd.Series(0.0, index=tickers)
    return pd.Series(
        {t: agg.get(t, 0.0) / total * 100 for t in tickers},
        name="weight_pct",
    )
