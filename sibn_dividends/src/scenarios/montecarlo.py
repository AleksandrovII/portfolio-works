"""Block 4 — Monte Carlo over (Brent, Urals discount, USDRUB) -> DPS-2026.

Brent and USDRUB are simulated as correlated daily log-return random walks
(driftless — no view on direction beyond what a scenario's mean-shift adds;
see scenarios/definitions.py) over one trading year, using volatility/
correlation from `calibration.py`. The discount is sampled independently
(see calibration.py docstring for why) from a triangular distribution using
the scenario's min/base/max. Each path's *annual average* Brent/USDRUB feeds
the same price-driver -> EBITDA/FCF-margin regression used in the backtest,
then the Block 3/4 waterfall, to produce one DPS draw.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import Config
from src.data.findata_client import FindataClient
from src.model import dividend, margins, netback
from src.scenarios.calibration import CalibrationResult
from src.scenarios.copula import sample_correlated


def simulate_annual_price_paths(
    n_paths: int,
    n_days: int,
    calib: CalibrationResult,
    dist: str,
    df: float | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (brent_avg[n_paths], usdrub_avg[n_paths]) — the annual-average
    price/FX realized along each simulated driftless daily path, starting
    from the current spot levels in `calib`.
    """
    vol_brent_daily = calib.vol_brent_annual / np.sqrt(252)
    vol_fx_daily = calib.vol_fx_annual / np.sqrt(252)

    total = n_paths * n_days
    z1, z2 = sample_correlated(total, calib.corr_brent_fx, dist, df, rng)
    r_brent = (z1 * vol_brent_daily).reshape(n_paths, n_days)
    r_fx = (z2 * vol_fx_daily).reshape(n_paths, n_days)

    # Driftless random walk: no view on direction beyond the scenario mean-shift.
    log_brent_path = np.cumsum(r_brent, axis=1)
    log_fx_path = np.cumsum(r_fx, axis=1)

    brent_path = calib.brent_spot * np.exp(log_brent_path)
    fx_path = calib.usdrub_spot * np.exp(log_fx_path)

    return brent_path.mean(axis=1), fx_path.mean(axis=1)


@dataclass
class ScenarioDraws:
    dps: np.ndarray
    brent_avg: np.ndarray
    usdrub_avg: np.ndarray
    discount: np.ndarray
    ebitda_margin: np.ndarray
    payout_ratio: np.ndarray

    def quantiles(self, qs=(0.10, 0.25, 0.50, 0.75, 0.90)) -> pd.Series:
        return pd.Series(np.quantile(self.dps, qs), index=[f"P{int(q*100)}" for q in qs])


def run_monte_carlo(
    client: FindataClient,
    cfg: Config,
    calib: CalibrationResult,
    discount_min: float,
    discount_base: float,
    discount_max: float,
    refining_volume_shock: float = 0.0,
    n_paths: int | None = None,
    dist: str | None = None,
    seed: int | None = None,
) -> ScenarioDraws:
    n_paths = n_paths or cfg.get("monte_carlo.n_paths")
    dist = dist or cfg.get("monte_carlo.copula")
    df = cfg.get("monte_carlo.student_t_df") if dist == "student_t" else None
    rng = np.random.default_rng(seed if seed is not None else cfg.get("seed"))

    brent_avg, usdrub_avg = simulate_annual_price_paths(
        n_paths, 252, calib, dist, df, rng
    )
    discount = rng.triangular(discount_min, discount_base, discount_max, size=n_paths)
    # NB: `discount` is sampled and returned (for Block 1 netback/upstream-downstream
    # reporting and Block 6 sensitivity tables) but deliberately does NOT enter
    # `price_driver` below — the margin regression was calibrated gross-of-discount
    # (see margins.py docstring: no historical Urals series to net it out), so the
    # forecast driver must stay consistent with that calibration. Wiring discount
    # into the DPS chain directly is a stage-4 item (exact НДПИ uses Urals price,
    # i.e. Brent - discount, as its base — a natural integration point).
    price_driver = brent_avg * usdrub_avg  # consistent with margins.py's driver definition

    x_hist = margins.historical_price_driver(cfg)
    hist_margins = margins.historical_margins(client, cfg)
    reg_ebitda = margins.fit_margin_regression(x_hist, hist_margins["ebitda_margin"])
    reg_fcf = margins.fit_margin_regression(x_hist, hist_margins["fcf_margin"])

    ebitda_clip = tuple(cfg.get("margins.ebitda_margin_clip"))
    fcf_clip = tuple(cfg.get("margins.fcf_margin_clip"))
    ebitda_margin = np.clip(
        reg_ebitda.intercept + reg_ebitda.slope * price_driver, *ebitda_clip
    )
    fcf_margin = np.clip(reg_fcf.intercept + reg_fcf.slope * price_driver, *fcf_clip)

    pnl_detail = margins.historical_pnl_detail(client, cfg)
    n = cfg.get("waterfall.trailing_years")
    da_margin = margins.trailing_average(pnl_detail["da_margin"], n)
    interest_rate = margins.trailing_average(pnl_detail["interest_rate_on_net_debt"], n)
    income_tax_rate = margins.trailing_average(pnl_detail["income_tax_rate"], n)
    minority_share = margins.trailing_average(pnl_detail["minority_share"], n)

    reports = client.annual_ifrs_reports().set_index("year")
    latest_year = int(reports.index.max())
    prior_net_debt_mrub = float(reports.loc[latest_year, "net_debt"])
    revenue_mrub_base = float(reports.loc[latest_year, "revenue"])
    n_shares = client.shares_outstanding()

    # Wire the refining-volume shock (escalation scenario's main lever, e.g.
    # NPZ attacks) into revenue via the downstream revenue share implied by
    # Block 1's physical netback model at the latest actual volumes/spot —
    # a first-order approximation (revenue moves proportionally to the
    # downstream share x shock), not a full re-run of the physical model
    # per path.
    revenue_mrub = revenue_mrub_base
    if refining_volume_shock != 0.0:
        volumes = netback.production_and_refining_volumes(client, cfg)
        latest_vol_year = int(volumes.dropna().index.max())
        vol_row = volumes.loc[latest_vol_year]
        modeled = netback.modeled_revenue_mrub(
            hydrocarbon_tonnes=vol_row["hydrocarbon_tonnes"],
            refining_tonnes=vol_row["refining_tonnes"],
            domestic_refining_share=vol_row["domestic_refining_share"],
            brent_usd=calib.brent_spot,
            discount_usd=discount_base,
            usdrub=calib.usdrub_spot,
            cfg=cfg,
        )
        downstream_share = (
            modeled["downstream_revenue_mrub"] / modeled["total_revenue_mrub"]
            if modeled["total_revenue_mrub"]
            else 0.0
        )
        revenue_mrub = revenue_mrub_base * (1 + downstream_share * refining_volume_shock)

    dps = np.empty(n_paths)
    payout = np.empty(n_paths)
    for i in range(n_paths):
        result = dividend.compute_year(
            year=latest_year + 1,
            revenue_mrub=revenue_mrub,
            ebitda_margin=float(ebitda_margin[i]),
            da_margin=da_margin,
            interest_rate_on_net_debt=interest_rate,
            income_tax_rate=income_tax_rate,
            minority_share=minority_share,
            fcf_margin=float(fcf_margin[i]),
            prior_net_debt_mrub=prior_net_debt_mrub,
            n_shares=n_shares,
            cfg=cfg,
        )
        dps[i] = result.dps_rub
        payout[i] = result.payout_ratio

    return ScenarioDraws(
        dps=dps,
        brent_avg=brent_avg,
        usdrub_avg=usdrub_avg,
        discount=discount,
        ebitda_margin=ebitda_margin,
        payout_ratio=payout,
    )


def run_scenario(
    client: FindataClient,
    cfg: Config,
    calib: CalibrationResult,
    scenario,  # scenarios.definitions.Scenario — typed loosely to avoid a cyclic import
    n_paths: int | None = None,
    dist: str | None = None,
    seed: int | None = None,
) -> ScenarioDraws:
    from src.scenarios.definitions import apply_scenario_to_calibration

    scenario_calib = apply_scenario_to_calibration(calib, scenario)
    return run_monte_carlo(
        client=client,
        cfg=cfg,
        calib=scenario_calib,
        discount_min=scenario.discount_min,
        discount_base=scenario.discount_base,
        discount_max=scenario.discount_max,
        refining_volume_shock=scenario.refining_volume_shock,
        n_paths=n_paths,
        dist=dist,
        seed=seed,
    )
