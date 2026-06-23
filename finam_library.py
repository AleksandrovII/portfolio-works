"""
finam_library.py
----------------
Finam Trade API client using the official finam-trade-api SDK.

Required environment variable
------------------------------
    FINAM_TOKEN — API secret key from Finam personal account
                  (Profile → Security → Trade API → Create token)
                  Format: tapi_sk_...

Notes
-----
- No FINAM_CLIENT_ID needed: account IDs are discovered from the token itself.
- figi is left empty: Finam uses its own security codes, not FIGI.
  Finam positions therefore appear in portfolio.csv / portfolio_visual.py
  but are skipped by Markowitz / correlation (which require FIGI).
- FORTS accounts (futures) are skipped automatically.
- rub_value = current_price × quantity; assumes prices are already in RUB.
  USD-denominated positions on SPB Exchange will be slightly off until
  FX conversion is added.

Install SDK (once):
    pip install finam-trade-api
"""

import asyncio
import logging
import os
from decimal import Decimal
from typing import Any, Dict, List

from finam_trade_api import TokenManager
from finam_trade_api.access import TokenClient
from finam_trade_api.account import AccountClient

logger = logging.getLogger(__name__)

FINAM_TOKEN = os.getenv("FINAM_TOKEN", "")

_SKIP_ACCOUNT_TYPES = {"FORTS"}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _dec(val: Any) -> Decimal:
    """Convert FinamDecimal / dict / str / number to Decimal."""
    if val is None:
        return Decimal("0")
    if hasattr(val, "value"):           # FinamDecimal pydantic model
        return Decimal(val.value or "0")
    if isinstance(val, dict):
        return Decimal(str(val.get("value", 0) or 0))
    return Decimal(str(val))


# ── Core async fetch ───────────────────────────────────────────────────────────

async def _fetch_all_async() -> List[Dict]:
    tm = TokenManager(token=FINAM_TOKEN)
    tc = TokenClient(token_manager=tm)

    # Step 1: exchange secret key for short-lived JWT
    await tc.set_jwt_token()

    # Step 2: discover account IDs from the JWT
    details = await tc.get_jwt_token_details()
    account_ids: List[str] = details.account_ids
    logger.info("Finam: %d accounts found", len(account_ids))

    ac = AccountClient(token_manager=tm)
    rows: List[Dict] = []

    for acc_id in account_ids:
        # Use raw request to avoid SDK pydantic validation gaps
        resp, ok = await ac._exec_request("GET", f"/accounts/{acc_id}")
        if not ok:
            logger.warning("Finam: account %s error: %s", acc_id, resp)
            continue

        account_type = resp.get("type", "")
        if account_type in _SKIP_ACCOUNT_TYPES:
            logger.debug("Finam: skipping %s account %s", account_type, acc_id)
            continue

        account_name = f"Finam {acc_id}"

        # ── Equity positions ───────────────────────────────────────────────────
        for pos in resp.get("positions", []):
            symbol = str(pos.get("symbol") or "").strip()
            if not symbol:
                continue

            qty       = _dec(pos.get("quantity"))
            avg_price = _dec(pos.get("average_price"))
            cur_price = _dec(pos.get("current_price"))

            if qty == 0:
                continue

            rub_value = cur_price * qty

            rows.append({
                "account":         account_name,
                "figi":            "",
                "name":            symbol,
                "ticker":          symbol,
                "currency":        "rub",
                "instrument_type": "share",
                "quantity":        qty,
                "avg_price":       avg_price,
                "cur_price":       cur_price,
                "rub_value":       rub_value,
            })

        # ── Cash balances ──────────────────────────────────────────────────────
        for cash in resp.get("cash", []):
            currency = str(cash.get("currency_code") or "").lower()
            units    = str(cash.get("units") or "0")
            nanos    = int(cash.get("nanos") or 0)
            balance  = Decimal(units) + Decimal(nanos) / Decimal(1_000_000_000)

            if balance <= 0:
                continue

            rows.append({
                "account":         account_name,
                "figi":            "",
                "name":            f"Деньги ({currency.upper()})",
                "ticker":          currency.upper(),
                "currency":        currency,
                "instrument_type": "currency",
                "quantity":        Decimal("1"),
                "avg_price":       balance,
                "cur_price":       balance,
                "rub_value":       balance,
            })

    logger.info("Finam: %d positions total across all accounts", len(rows))
    return rows


# ── Public API ─────────────────────────────────────────────────────────────────

def fetch_finam_positions() -> List[Dict]:
    """
    Fetch all positions from Finam Trade API.

    Returns list of position dicts matching the schema used by
    portfolio_works_library.fetch_all_positions().  Returns [] if
    FINAM_TOKEN is not set or on any API error.
    """
    if not FINAM_TOKEN:
        logger.info("FINAM_TOKEN not set — Finam skipped.")
        return []

    try:
        return asyncio.run(_fetch_all_async())
    except Exception as exc:
        logger.error("Finam API error: %s", exc)
        return []
