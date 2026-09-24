"""Block 3 — P&L margins as a function of netback (regression), R² and residuals.

Design note: the "textbook" driver would be the full netback formula
(Brent - discount - freight) x USDRUB. But a historical Urals discount series
isn't available (the Minfin CSV is a user-filled stub, see urals_minfin.py),
and SIBN's own disclosed production volumes have a metric-definition break in
2021-2023 (see netback.py). So the regression driver used here is the
always-available, gap-free `brent_usd_avg x usdrub_avg` for the calendar year
— "Brent in rubles" — which is the dominant term of the netback formula and
moves with the same price regime. This is disclosed as a modelling
simplification: forecast years apply the same driver definition (computed from
Brent/discount/FX scenario draws via `netback.crude_netback_rub_per_bbl`
gross of discount/freight, i.e. brent_usd x usdrub) so historical fit and
forecast application stay consistent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from src.config import Config
from src.data.brent import fetch_brent
from src.data.cbr_fx import fetch_usdrub
from src.data.findata_client import FindataClient


def historical_price_driver(cfg: Config) -> pd.Series:
    """Calendar-year average Brent-in-rubles (₽/bbl), `fiscal.history_start_year`
    through `history_end_year`. Available for every year — no disclosure gap.
    """
    start_year = cfg.get("fiscal.history_start_year")
    end_year = cfg.get("fiscal.history_end_year")
    start, end = f"{start_year}-01-01", f"{end_year}-12-31"

    fx = fetch_usdrub(start, end)
    brent = fetch_brent(start, end)
    fx["year"] = fx["date"].dt.year
    brent["year"] = brent["date"].dt.year

    fx_avg = fx.groupby("year")["usdrub"].mean()
    brent_avg = brent.groupby("year")["brent_usd"].mean()
    driver = (brent_avg * fx_avg).rename("brent_rub_per_bbl")
    return driver.loc[start_year:end_year]


def historical_margins(client: FindataClient, cfg: Config) -> pd.DataFrame:
    """Actual revenue/EBITDA/NI margins from IFRS annual reports, history window."""
    start_year = cfg.get("fiscal.history_start_year")
    end_year = cfg.get("fiscal.history_end_year")
    reports = client.annual_ifrs_reports()
    reports = reports[(reports["year"] >= start_year) & (reports["year"] <= end_year)]
    df = reports.set_index("year")[["revenue", "ebitda", "earnings_stock_holders", "fcf"]].copy()
    df.columns = ["revenue_mrub", "ebitda_mrub", "ni_mrub", "fcf_mrub"]
    df["ebitda_margin"] = df["ebitda_mrub"] / df["revenue_mrub"]
    df["ni_margin"] = df["ni_mrub"] / df["revenue_mrub"]
    df["fcf_margin"] = df["fcf_mrub"] / df["revenue_mrub"]
    return df


def historical_pnl_detail(client: FindataClient, cfg: Config) -> pd.DataFrame:
    """Full EBITDA -> NI waterfall components, history window.

    Why this exists: a single-variable regression of NI margin directly on the
    price driver was tried first and badly over-predicted 2024/2025 NI margins
    (leave-one-out backtest error +70%/+121%) — it can't see the rising
    leverage/interest burden (net_debt/EBITDA: 0.29x in 2021 -> 1.43x in 2025).
    Going through the actual waterfall (EBITDA -[D&A]-> EBIT -[interest]-> EBT
    -[tax]-> NI -[minority]-> NI attributable) fixes that because interest
    scales with the model's own debt roll-forward instead of being smuggled
    into a price-only regression.
    """
    start_year = cfg.get("fiscal.history_start_year")
    end_year = cfg.get("fiscal.history_end_year")
    reports = client.annual_ifrs_reports()
    reports = reports[(reports["year"] >= start_year) & (reports["year"] <= end_year)]
    cols = [
        "revenue", "ebitda", "depr_depl_amort", "interest_expense",
        "net_debt", "earnings_wo_tax", "earnings", "earnings_stock_holders",
    ]
    df = reports.set_index("year")[cols].copy()
    df.columns = [
        "revenue_mrub", "ebitda_mrub", "da_mrub", "interest_expense_mrub",
        "net_debt_mrub", "ebt_mrub", "ni_consolidated_mrub", "ni_mrub",
    ]
    df["ebitda_margin"] = df["ebitda_mrub"] / df["revenue_mrub"]
    df["da_margin"] = df["da_mrub"] / df["revenue_mrub"]
    df["interest_rate_on_net_debt"] = df["interest_expense_mrub"] / df["net_debt_mrub"]
    df["income_tax_rate"] = (df["ebt_mrub"] - df["ni_consolidated_mrub"]) / df["ebt_mrub"]
    df["minority_share"] = (df["ni_consolidated_mrub"] - df["ni_mrub"]) / df["ni_consolidated_mrub"]
    return df


def trailing_average(series: pd.Series, n: int, exclude_year: int | None = None) -> float:
    """Mean of the last `n` non-null observations, optionally excluding one
    year (for leave-one-out backtests)."""
    s = series.drop(index=[exclude_year], errors="ignore") if exclude_year is not None else series
    s = s.dropna()
    return float(s.tail(n).mean())


@dataclass
class RegressionResult:
    slope: float
    intercept: float
    r_squared: float
    p_value: float
    std_err: float
    fitted: pd.Series
    residuals: pd.Series

    def predict(self, x: float) -> float:
        return self.intercept + self.slope * x


def fit_margin_regression(x: pd.Series, y: pd.Series) -> RegressionResult:
    """Simple OLS: margin = intercept + slope * price_driver."""
    aligned = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    if len(aligned) < 3:
        raise ValueError(
            f"Only {len(aligned)} overlapping data points for regression — need >=3."
        )
    reg = stats.linregress(aligned["x"], aligned["y"])
    fitted = pd.Series(reg.intercept + reg.slope * aligned["x"], index=aligned.index)
    residuals = aligned["y"] - fitted
    return RegressionResult(
        slope=float(reg.slope),
        intercept=float(reg.intercept),
        r_squared=float(reg.rvalue**2),
        p_value=float(reg.pvalue),
        std_err=float(reg.stderr),
        fitted=fitted,
        residuals=residuals,
    )


def project_margin(reg: RegressionResult, x_forecast: float, clip: tuple[float, float]) -> float:
    raw = reg.predict(x_forecast)
    return float(np.clip(raw, clip[0], clip[1]))
