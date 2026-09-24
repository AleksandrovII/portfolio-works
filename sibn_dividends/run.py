#!/usr/bin/env python
"""CLI entry point — MVP stage.

Prints:
  1. Backtest results for FY2024/FY2025 against actual DPS (leave-one-out).
  2. A deterministic single-scenario DPS-2026 point estimate, using the
     full-sample (non-LOO) regression and a trailing-90-day spot price driver
     as a placeholder for the proper base/escalation/hormuz scenarios (stage 2).

Config version and key assumptions are printed for reproducibility.
"""

from __future__ import annotations

import datetime as dt

from src.config import load_config
from src.data.brent import fetch_brent
from src.data.cbr_fx import fetch_usdrub
from src.data.findata_client import FindataClient
from src.model import dividend, margins
from src.model.backtest import run_backtest
from src.scenarios.calibration import calibrate
from src.scenarios.definitions import list_scenarios, load_scenario
from src.scenarios.montecarlo import run_scenario


def print_header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main() -> None:
    cfg = load_config()
    client = FindataClient()

    print_header(f"SIBN DPS MODEL — MVP  (config version {cfg.version})")

    print_header("Backtest: leave-one-out vs. actual DPS")
    for year in (2024, 2025):
        result = run_backtest(client, cfg, year)
        print(result.decomposition())
        status = "PASS" if result.within_tolerance else "FAIL"
        print(f"  -> {status} (tolerance +-{cfg.get('fiscal.backtest_tolerance'):.0%})\n")

    print_header("Historical margin regression (full sample, 2021-2025)")
    x = margins.historical_price_driver(cfg)
    hist = margins.historical_margins(client, cfg)
    reg_ebitda = margins.fit_margin_regression(x, hist["ebitda_margin"])
    reg_fcf = margins.fit_margin_regression(x, hist["fcf_margin"])
    print(f"EBITDA margin ~ price driver: R2={reg_ebitda.r_squared:.2f}, "
          f"p={reg_ebitda.p_value:.2f}, n={len(x)}")
    print(f"  residuals: {reg_ebitda.residuals.round(4).to_dict()}")
    print(f"FCF margin ~ price driver:    R2={reg_fcf.r_squared:.2f}, "
          f"p={reg_fcf.p_value:.2f}, n={len(x)}")
    print(f"  residuals: {reg_fcf.residuals.round(4).to_dict()}")

    print_header("Deterministic DPS-2026 point estimate (placeholder spot conditions)")
    end = dt.date.today().isoformat()
    start = (dt.date.today() - dt.timedelta(days=90)).isoformat()
    brent = fetch_brent(start, end)
    fx = fetch_usdrub(start, end)
    spot_price_driver = float(brent["brent_usd"].mean() * fx["usdrub"].mean())
    print(f"Trailing 90d Brent avg: {brent['brent_usd'].mean():.1f} USD/bbl")
    print(f"Trailing 90d USDRUB avg: {fx['usdrub'].mean():.1f}")
    print(f"Price driver (Brent x USDRUB): {spot_price_driver:,.0f} RUB/bbl")

    ebitda_margin = margins.project_margin(
        reg_ebitda, spot_price_driver, tuple(cfg.get("margins.ebitda_margin_clip"))
    )
    fcf_margin = margins.project_margin(
        reg_fcf, spot_price_driver, tuple(cfg.get("margins.fcf_margin_clip"))
    )

    pnl_detail = margins.historical_pnl_detail(client, cfg)
    n = cfg.get("waterfall.trailing_years")
    da_margin = margins.trailing_average(pnl_detail["da_margin"], n)
    interest_rate = margins.trailing_average(pnl_detail["interest_rate_on_net_debt"], n)
    income_tax_rate = margins.trailing_average(pnl_detail["income_tax_rate"], n)
    minority_share = margins.trailing_average(pnl_detail["minority_share"], n)

    reports = client.annual_ifrs_reports().set_index("year")
    latest_year = int(reports.index.max())
    prior_net_debt_mrub = float(reports.loc[latest_year, "net_debt"])
    revenue_mrub = float(reports.loc[latest_year, "revenue"])  # placeholder: flat revenue
    n_shares = client.shares_outstanding()

    forecast = dividend.compute_year(
        year=latest_year + 1,
        revenue_mrub=revenue_mrub,
        ebitda_margin=ebitda_margin,
        da_margin=da_margin,
        interest_rate_on_net_debt=interest_rate,
        income_tax_rate=income_tax_rate,
        minority_share=minority_share,
        fcf_margin=fcf_margin,
        prior_net_debt_mrub=prior_net_debt_mrub,
        n_shares=n_shares,
        cfg=cfg,
    )
    print(f"\nProjected EBITDA margin: {ebitda_margin:.1%}, FCF margin: {fcf_margin:.1%}")
    print(f"Projected DPS-{latest_year + 1}: {forecast.dps_rub:.2f} RUB/share "
          f"(payout {forecast.payout_ratio:.0%}, Net Debt/EBITDA "
          f"{forecast.net_debt_ebitda:.2f}x)")
    print(
        "\nNB: this is an MVP placeholder point estimate (flat revenue, trailing-90d "
        "spot price, full-sample regression) — see the scenario Monte Carlo below for "
        "the actual stage-2 forecast, and the backtest above for the model's known accuracy."
    )

    print_header("Historical vol/correlation calibration (Brent, USDRUB), 2022-present")
    calib = calibrate(cfg)
    print(calib.as_table().to_string(index=False))
    print(
        "\nNB: corr(Brent, USDRUB) is weak and slightly POSITIVE in this window — the "
        "'ruble as built-in shock absorber' thesis (oil down -> ruble weakens, cushioning "
        "ruble netback) is NOT clearly supported by daily log-returns 2022-2026. Scenario "
        "fx_shift_multiplier assumptions (config/scenarios/*.yaml) encode the thesis "
        "structurally/illustratively, not from this empirical correlation — flagged, not hidden."
    )

    print_header(f"Scenario Monte Carlo (n={cfg.get('monte_carlo.n_paths'):,} paths each)")
    for name in list_scenarios():
        scenario = load_scenario(name)
        draws = run_scenario(client, cfg, calib, scenario)
        q = draws.quantiles()
        print(f"\n{name} — {scenario.description}")
        print("  DPS quantiles (RUB/share): " + ", ".join(f"{k}={v:.1f}" for k, v in q.items()))
        print(f"  mean Brent={draws.brent_avg.mean():.1f} USD/bbl, "
              f"mean USDRUB={draws.usdrub_avg.mean():.1f}, "
              f"mean EBITDA margin={draws.ebitda_margin.mean():.1%}")


if __name__ == "__main__":
    main()
