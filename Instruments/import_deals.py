"""
import_deals.py
---------------
Imports trade history and cash flows from T-Invest API for all accounts.

Output: data/deals.csv  (appends; duplicates skipped by operation_id)

Column reference
----------------
  operation_id        unique ID from T-Invest
  date                ISO-8601 UTC timestamp
  account             brokerage account name
  type                trade | deposit | withdrawal | dividend | coupon | tax | other
  direction           buy | sell  (only for type == trade, else empty)
  figi                instrument FIGI (empty for deposits / withdrawals)
  ticker              instrument ticker
  name                instrument full name
  quantity            lots traded (only for trades)
  price               unit price in instrument's native currency
  price_currency      currency of price
  payment             total settlement amount (negative = paid, positive = received)
  payment_currency    currency of payment (usually RUB)
  commission          broker fee (negative value)
  commission_currency currency of commission
  raw_type            original operation type string from the API

Usage
-----
    export INVEST_TOKEN='your_token'
    /opt/miniconda3/bin/python import_deals.py        # last 3 months (default)
    /opt/miniconda3/bin/python import_deals.py 6      # last 6 months
    /opt/miniconda3/bin/python import_deals.py 12     # last 12 months
"""

import logging
import pathlib
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Set

import pandas as pd
from t_tech.invest import Client, OperationState

_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from portfolio_works_library import TOKEN, get_instrument_info

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

DATA_DIR   = _ROOT / "data"
DEALS_PATH = DATA_DIR / "deals.csv"

CSV_COLUMNS = [
    "operation_id", "date", "account", "type", "direction",
    "figi", "ticker", "name",
    "quantity", "price", "price_currency",
    "payment", "payment_currency",
    "commission", "commission_currency",
    "raw_type",
]

# ── Operation type classification ──────────────────────────────────────────────

_TRADE_KW      = ("покупка", "продажа", "buy", "sell", "обмен валют")
_DEPOSIT_KW    = ("пополнение", "зачисление", "ввод")
_WITHDRAWAL_KW = ("вывод",)
_DIVIDEND_KW   = ("дивиденд", "dividend")
_COUPON_KW     = ("купон", "coupon", "нкд")
_TAX_KW        = ("налог", "tax")


def _classify(raw_type: str) -> str:
    t = raw_type.lower()
    if any(k in t for k in _TRADE_KW):
        return "trade"
    if any(k in t for k in _DEPOSIT_KW):
        return "deposit"
    if any(k in t for k in _WITHDRAWAL_KW):
        return "withdrawal"
    if any(k in t for k in _DIVIDEND_KW):
        return "dividend"
    if any(k in t for k in _COUPON_KW):
        return "coupon"
    if any(k in t for k in _TAX_KW):
        return "tax"
    return "other"


def _direction(raw_type: str) -> str:
    t = raw_type.lower()
    if "продажа" in t or "sell" in t:
        return "sell"
    if "покупка" in t or "buy" in t:
        return "buy"
    return ""


# ── Protobuf value helpers ─────────────────────────────────────────────────────

def _q(proto) -> Optional[Decimal]:
    """Quotation proto → Decimal."""
    if proto is None:
        return None
    return Decimal(proto.units) + Decimal(proto.nano) / Decimal(1_000_000_000)


def _mv(proto) -> Optional[Decimal]:
    """MoneyValue proto → Decimal."""
    if proto is None:
        return None
    try:
        return Decimal(proto.units) + Decimal(proto.nano) / Decimal(1_000_000_000)
    except Exception:
        return None


def _mv_cur(proto) -> Optional[str]:
    """MoneyValue proto → currency string."""
    return getattr(proto, "currency", None) if proto is not None else None


# ── FIGI → instrument cache ────────────────────────────────────────────────────

def _build_figi_cache() -> Dict[str, Dict]:
    """
    Pre-load ticker / name / currency from portfolio.csv so we don't
    need extra API calls for instruments already in the snapshot.
    """
    snapshot = DATA_DIR / "portfolio.csv"
    if not snapshot.exists():
        return {}
    try:
        df = pd.read_csv(snapshot, usecols=["figi", "ticker", "name", "currency"])
        return {
            row["figi"]: {
                "ticker":   str(row["ticker"]),
                "name":     str(row["name"]),
                "currency": str(row["currency"]),
            }
            for _, row in df.drop_duplicates("figi").iterrows()
        }
    except Exception as e:
        logger.warning("Could not read portfolio.csv for cache: %s", e)
        return {}


# ── Deduplication ──────────────────────────────────────────────────────────────

def _existing_ids() -> Set[str]:
    """Return set of operation_ids already saved in deals.csv."""
    if not DEALS_PATH.exists():
        return set()
    try:
        df = pd.read_csv(DEALS_PATH, usecols=["operation_id"], dtype=str)
        return set(df["operation_id"].dropna())
    except Exception as e:
        logger.warning("Could not read existing deals.csv: %s", e)
        return set()


# ── Core fetch ─────────────────────────────────────────────────────────────────

def _resolve_instrument(
    client: Client,
    figi: str,
    cache: Dict[str, Dict],
) -> Dict[str, str]:
    """Return {ticker, name, currency} for a FIGI, using cache first."""
    if not figi:
        return {"ticker": "", "name": "", "currency": ""}
    if figi in cache:
        return cache[figi]
    info = get_instrument_info(client, figi)
    if info:
        result = {
            "ticker":   info.get("ticker", ""),
            "name":     info.get("name", ""),
            "currency": info.get("currency", ""),
        }
    else:
        result = {"ticker": "", "name": "", "currency": ""}
    cache[figi] = result
    return result


def fetch_operations(client: Client, months: int) -> List[Dict]:
    """
    Fetch all executed operations across all accounts for the last `months` months.
    Returns a list of row dicts ready for a DataFrame.
    """
    to_   = datetime.now(timezone.utc)
    from_ = to_ - timedelta(days=months * 30)

    try:
        accounts = client.users.get_accounts().accounts
    except Exception as e:
        logger.error("Cannot fetch accounts: %s", e)
        return []

    figi_cache = _build_figi_cache()
    rows: List[Dict] = []

    for account in accounts:
        logger.info("Account: %s — fetching %d months of operations…", account.name, months)
        try:
            resp = client.operations.get_operations(
                account_id=account.id,
                from_=from_,
                to=to_,
            )
        except Exception as e:
            logger.warning("Skipping account %s: %s", account.name, e)
            continue

        skipped = 0
        for op in resp.operations:
            if op.state != OperationState.OPERATION_STATE_EXECUTED:
                skipped += 1
                continue

            raw_type = getattr(op, "type", "") or ""
            op_type  = _classify(raw_type)

            if op_type == "other":
                logger.debug("Unclassified operation type: %r", raw_type)

            figi     = getattr(op, "figi", "") or ""
            instr    = _resolve_instrument(client, figi, figi_cache)

            # ── Quantity and price (trades only) ───────────────────────────
            qty   = None
            price = None
            if op_type == "trade":
                raw_qty = getattr(op, "quantity", None)
                if not raw_qty:
                    raw_qty = getattr(op, "quantity_executed", None)
                qty = int(raw_qty) if raw_qty else None

                price_proto = getattr(op, "price", None)
                if price_proto:
                    price = float(_q(price_proto))

            # ── Payment (total settlement) ─────────────────────────────────
            payment      = None
            payment_curr = None
            pay_proto = getattr(op, "payment", None)
            if pay_proto:
                val = _mv(pay_proto)
                if val is not None:
                    payment      = float(val)
                    payment_curr = _mv_cur(pay_proto)

            # ── Commission ─────────────────────────────────────────────────
            commission      = None
            commission_curr = None
            comm_proto = getattr(op, "commission", None)
            if comm_proto:
                val = _mv(comm_proto)
                if val is not None and val != 0:
                    commission      = float(val)
                    commission_curr = _mv_cur(comm_proto)

            op_date = getattr(op, "date", None)
            op_id   = str(getattr(op, "id", "") or "")

            rows.append({
                "operation_id":        op_id,
                "date":                op_date.isoformat() if op_date else "",
                "account":             account.name,
                "type":                op_type,
                "direction":           _direction(raw_type) if op_type == "trade" else "",
                "figi":                figi,
                "ticker":              instr["ticker"],
                "name":                instr["name"],
                "quantity":            qty,
                "price":               price,
                "price_currency":      instr["currency"] if op_type == "trade" else "",
                "payment":             payment,
                "payment_currency":    payment_curr,
                "commission":          commission,
                "commission_currency": commission_curr,
                "raw_type":            raw_type,
            })

        logger.info(
            "  → %d executed, %d skipped (non-executed)",
            len([r for r in rows if r["account"] == account.name]), skipped,
        )

    return rows


# ── Save (append + dedup) ──────────────────────────────────────────────────────

def save_deals(rows: List[Dict]) -> None:
    if not rows:
        logger.info("No operations fetched.")
        return

    df_new = pd.DataFrame(rows, columns=CSV_COLUMNS)
    df_new = df_new.sort_values("date")

    existing = _existing_ids()
    df_new   = df_new[~df_new["operation_id"].astype(str).isin(existing)]

    if df_new.empty:
        print("\nNo new operations to save — deals.csv is already up to date.")
        return

    DATA_DIR.mkdir(exist_ok=True)
    write_header = not DEALS_PATH.exists()
    df_new.to_csv(DEALS_PATH, mode="a", header=write_header, index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    total_new = len(df_new)
    print(f"\n{'='*60}")
    print(f"  New operations saved: {total_new}")
    print(f"{'='*60}")

    by_type = df_new["type"].value_counts()
    for op_type, count in by_type.items():
        subset = df_new[df_new["type"] == op_type]
        if op_type in ("trade",):
            buys  = (subset["direction"] == "buy").sum()
            sells = (subset["direction"] == "sell").sum()
            print(f"  {op_type:<16} {count:>4}  (buy: {buys}, sell: {sells})")
        elif op_type in ("deposit", "withdrawal", "dividend", "coupon", "tax"):
            total_pay = subset["payment"].sum()
            curr = subset["payment_currency"].iloc[0] if not subset.empty else ""
            print(f"  {op_type:<16} {count:>4}  total: {total_pay:+,.2f} {curr}")
        else:
            print(f"  {op_type:<16} {count:>4}")

    date_min = df_new["date"].min()[:10]
    date_max = df_new["date"].max()[:10]
    print(f"{'='*60}")
    print(f"  Period: {date_min} → {date_max}")
    print(f"  Saved → {DEALS_PATH}\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    months = 3
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if not arg.isdigit():
            print(f"Usage: python import_deals.py [months]\n  months must be a positive integer, got: {arg!r}")
            sys.exit(1)
        months = int(arg)
        if months < 1:
            print("months must be ≥ 1")
            sys.exit(1)

    logger.info("Importing last %d months of operations…", months)
    with Client(TOKEN) as client:
        rows = fetch_operations(client, months)
    logger.info("Total executed operations fetched: %d", len(rows))
    save_deals(rows)


if __name__ == "__main__":
    main()
