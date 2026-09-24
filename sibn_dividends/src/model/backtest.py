"""Block 5 — backtest: leave-one-out validation of the revenue -> margin ->
waterfall -> payout -> DPS chain against actual reported DPS.

Leave-one-out (not in-sample) on purpose: with n=5 calibration points, fitting
the EBITDA-margin regression on all 5 years and then "predicting" one of those
same years would be circular. Excluding the test year from the fit (and from
the trailing-average waterfall ratios) is the honest version of "does the
model reproduce the historical fact" with this little data.

Revenue itself is taken as the actual reported figure for the test year (this
model does not attempt to forecast revenue — see margins.py) so the backtest
isolates the price-driver -> EBITDA margin -> waterfall -> payout -> DPS chain,
which is what Blocks 3/4 actually claim to model.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.config import Config
from src.data.findata_client import FindataClient
from src.model import dividend, margins


@dataclass
class BacktestResult:
    year: int
    actual_dps: float
    predicted_dps: float
    pct_error: float
    within_tolerance: bool
    price_driver_rub_per_bbl: float
    actual_ebitda_margin: float
    predicted_ebitda_margin: float
    actual_ni_margin: float
    predicted_ni_margin: float
    payout_ratio_used: float
    ebitda_reg_r2: float
    da_margin_used: float
    interest_rate_used: float
    income_tax_rate_used: float
    minority_share_used: float

    def decomposition(self) -> str:
        """Human-readable block-by-block breakdown of the error, per the spec's
        "не подгоняй молча — покажи разложение расхождения по блокам" requirement.
        """
        margin_gap_ebitda = self.predicted_ebitda_margin - self.actual_ebitda_margin
        margin_gap_ni = self.predicted_ni_margin - self.actual_ni_margin
        lines = [
            f"Backtest FY{self.year}: actual={self.actual_dps:.2f} RUB, "
            f"predicted={self.predicted_dps:.2f} RUB, error={self.pct_error:+.1%}",
            f"  price driver (Brent x USDRUB): {self.price_driver_rub_per_bbl:,.0f} RUB/bbl",
            f"  EBITDA margin: actual={self.actual_ebitda_margin:.1%} "
            f"predicted={self.predicted_ebitda_margin:.1%} (gap {margin_gap_ebitda:+.1%}, "
            f"regression R2={self.ebitda_reg_r2:.2f})",
            f"  NI margin (implied): actual={self.actual_ni_margin:.1%} "
            f"predicted={self.predicted_ni_margin:.1%} (gap {margin_gap_ni:+.1%})",
            f"  waterfall assumptions: D&A%={self.da_margin_used:.1%}, "
            f"interest/net_debt={self.interest_rate_used:.1%}, "
            f"income tax={self.income_tax_rate_used:.1%}, "
            f"minority={self.minority_share_used:.1%}",
            f"  payout ratio used: {self.payout_ratio_used:.0%}",
        ]
        return "\n".join(lines)


def run_backtest(client: FindataClient, cfg: Config, year: int) -> BacktestResult:
    price_driver = margins.historical_price_driver(cfg)
    hist_margins = margins.historical_margins(client, cfg)
    pnl_detail = margins.historical_pnl_detail(client, cfg)

    if year not in price_driver.index or year not in hist_margins.index:
        raise ValueError(f"No historical data for {year} in the calibration window.")

    # Leave-one-out: exclude the test year from the EBITDA-margin regression fit.
    x_train = price_driver.drop(index=year)
    ebitda_train = hist_margins["ebitda_margin"].drop(index=year)
    reg_ebitda = margins.fit_margin_regression(x_train, ebitda_train)

    x_test = float(price_driver.loc[year])
    predicted_ebitda_margin = margins.project_margin(
        reg_ebitda, x_test, tuple(cfg.get("margins.ebitda_margin_clip"))
    )

    # Waterfall ratios: trailing average excluding the test year (leave-one-out).
    n = cfg.get("waterfall.trailing_years")
    da_margin = margins.trailing_average(pnl_detail["da_margin"], n, exclude_year=year)
    interest_rate = margins.trailing_average(
        pnl_detail["interest_rate_on_net_debt"], n, exclude_year=year
    )
    income_tax_rate = margins.trailing_average(
        pnl_detail["income_tax_rate"], n, exclude_year=year
    )
    minority_share = margins.trailing_average(
        pnl_detail["minority_share"], n, exclude_year=year
    )

    fcf_train = hist_margins["fcf_margin"].drop(index=year)
    reg_fcf = margins.fit_margin_regression(x_train, fcf_train)
    predicted_fcf_margin = margins.project_margin(
        reg_fcf, x_test, tuple(cfg.get("margins.fcf_margin_clip"))
    )

    actual_revenue_mrub = float(hist_margins.loc[year, "revenue_mrub"])
    reports = client.annual_ifrs_reports().set_index("year")
    prior_net_debt_mrub = float(reports.loc[year - 1, "net_debt"])
    n_shares = client.shares_outstanding()

    result = dividend.compute_year(
        year=year,
        revenue_mrub=actual_revenue_mrub,
        ebitda_margin=predicted_ebitda_margin,
        da_margin=da_margin,
        interest_rate_on_net_debt=interest_rate,
        income_tax_rate=income_tax_rate,
        minority_share=minority_share,
        fcf_margin=predicted_fcf_margin,
        prior_net_debt_mrub=prior_net_debt_mrub,
        n_shares=n_shares,
        cfg=cfg,
    )

    actual_dps = float(client.dividends_by_fiscal_year().loc[year])
    pct_error = (result.dps_rub - actual_dps) / actual_dps
    tolerance = cfg.get("fiscal.backtest_tolerance")

    return BacktestResult(
        year=year,
        actual_dps=actual_dps,
        predicted_dps=result.dps_rub,
        pct_error=pct_error,
        within_tolerance=abs(pct_error) <= tolerance,
        price_driver_rub_per_bbl=x_test,
        actual_ebitda_margin=float(hist_margins.loc[year, "ebitda_margin"]),
        predicted_ebitda_margin=predicted_ebitda_margin,
        actual_ni_margin=float(hist_margins.loc[year, "ni_margin"]),
        predicted_ni_margin=result.ni_mrub / actual_revenue_mrub,
        payout_ratio_used=result.payout_ratio,
        ebitda_reg_r2=reg_ebitda.r_squared,
        da_margin_used=da_margin,
        interest_rate_used=interest_rate,
        income_tax_rate_used=income_tax_rate,
        minority_share_used=minority_share,
    )
