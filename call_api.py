"""
call_api.py
-----------
Fetches a current portfolio snapshot from T-Invest API and saves it to
data/portfolio.csv.

Run this script once before using any Instruments scripts.  Re-run any time
you want to refresh prices and positions.

Usage:
    export INVEST_TOKEN='your_token'
    /opt/miniconda3/bin/python call_api.py
"""

import logging
import os
import pathlib
from datetime import datetime, timezone

import pandas as pd
from t_tech.invest import Client

from portfolio_works_library import TOKEN, get_rates, fetch_all_positions

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

_HERE        = pathlib.Path(__file__).parent
DATA_DIR     = _HERE / "data"
SNAPSHOT_PATH = DATA_DIR / "portfolio.csv"


def _build_snapshot(positions: list) -> pd.DataFrame:
    total_rub  = sum(float(p.get("rub_value") or 0) for p in positions)
    updated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    rows = []
    for p in positions:
        rub = float(p.get("rub_value") or 0)
        rows.append({
            "updated_at":      updated_at,
            "figi":            p["figi"],
            "ticker":          p["ticker"],
            "name":            p["name"],
            "account":         p["account"],
            "instrument_type": p["instrument_type"],
            "currency":        p["currency"],
            "quantity":        float(p["quantity"]),
            "avg_price":       float(p["avg_price"]) if p.get("avg_price") is not None else "",
            "cur_price":       float(p["cur_price"]) if p.get("cur_price") is not None else "",
            "rub_value":       round(rub, 2),
            "weight_pct":      round(rub / total_rub * 100, 4) if total_rub > 0 else 0.0,
        })

    return pd.DataFrame(rows)


def _print_summary(df: pd.DataFrame) -> None:
    total = df["rub_value"].sum()
    print(f"\n{'='*60}")
    print(f"  Portfolio snapshot  ·  {len(df)} positions  ·  {total:,.0f} ₽")
    print(f"{'='*60}")

    by_type = (
        df.groupby("instrument_type")["rub_value"]
        .sum()
        .sort_values(ascending=False)
    )
    for itype, val in by_type.items():
        pct = val / total * 100
        print(f"  {itype:<20} {val:>14,.0f} ₽   {pct:5.1f}%")

    print(f"{'='*60}")
    print(f"\n  Saved → {SNAPSHOT_PATH}\n")


def main() -> None:
    logger.info("Connecting to T-Invest API…")
    with Client(TOKEN) as client:
        rates     = get_rates(client)
        positions = fetch_all_positions(client, rates)
    logger.info("Fetched %d positions", len(positions))

    if not positions:
        logger.error("No positions found — check your token and accounts.")
        return

    df = _build_snapshot(positions)

    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(SNAPSHOT_PATH, index=False)

    _print_summary(df)


if __name__ == "__main__":
    main()
