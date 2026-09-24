"""Block 5 — backtest against the known FY2025 actual (45.41 RUB/share).

Per spec, the fallback-mode tolerance is +-15%. If the model misses it, the
assertion message carries the full block-by-block decomposition (revenue is
exact by construction — it's the actual reported figure — so any gap is in
the EBITDA-margin regression, the waterfall assumptions, or payout) instead of
silently loosening the tolerance to make the test pass.
"""

from __future__ import annotations

import pytest

from src.config import load_config
from src.data.findata_client import FindataClient
from src.model.backtest import run_backtest


@pytest.fixture(scope="module")
def client():
    return FindataClient()


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_backtest_2025_dividend_mapping_matches_actual(client):
    """Sanity check independent of the model: the S1(year=Y)+Y(year=Y+1)
    dividend-bucketing logic must reproduce the known FY2025 actual exactly."""
    dps = client.dividends_by_fiscal_year()
    assert dps.loc[2025] == pytest.approx(45.41, abs=0.01)


def test_backtest_2025_within_tolerance(client, cfg):
    result = run_backtest(client, cfg, 2025)
    assert result.within_tolerance, (
        f"FY2025 backtest outside +-{cfg.get('fiscal.backtest_tolerance'):.0%} tolerance.\n"
        + result.decomposition()
    )
