"""Block 5 — sanity backtest against the known FY2024 actual (79.17 RUB/share,
S1=51.96 paid 2024-10-14 + Y=27.21 paid 2025-07-08)."""

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


def test_backtest_2024_dividend_mapping_matches_actual(client):
    dps = client.dividends_by_fiscal_year()
    assert dps.loc[2024] == pytest.approx(79.17, abs=0.01)


def test_backtest_2024_within_tolerance(client, cfg):
    result = run_backtest(client, cfg, 2024)
    assert result.within_tolerance, (
        f"FY2024 backtest outside +-{cfg.get('fiscal.backtest_tolerance'):.0%} tolerance.\n"
        + result.decomposition()
    )
