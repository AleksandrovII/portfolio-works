"""Historical volatility/correlation calibration for the Monte Carlo engine
(Block 4 spec: "корреляции и волатильности не задавай экспертно — оцени на
дневных данных 2022-2026").

Only Brent and USDRUB are calibrated from data — the third Monte Carlo
variable (Urals-Brent discount) has no reliable historical daily series
available (see urals_minfin.py: the Minfin CSV is a user-filled stub), so it
is sampled independently from a scenario-defined range rather than jointly
correlated with Brent/FX. This is a disclosed simplification, not a silent one.

The central thesis under test — "ruble as a built-in shock absorber" (oil down
-> ruble weakens, cushioning ruble netback) — shows up as a *negative*
correlation between Brent log-returns and USDRUB log-returns (Brent down,
USDRUB up = ruble weaker). That correlation, plus its tail behaviour under a
Gaussian vs. Student-t copula, is exactly what `calibrate()` reports.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import Config
from src.data.brent import fetch_brent
from src.data.cbr_fx import fetch_usdrub
from src.scenarios.copula import sample_correlated


@dataclass
class CalibrationResult:
    start: str
    end: str
    n_obs: int
    vol_brent_annual: float
    vol_fx_annual: float
    corr_brent_fx: float
    student_t_df: float
    brent_spot: float
    usdrub_spot: float
    tail_prob_empirical: float
    tail_prob_gaussian: float
    tail_prob_student_t: float

    def as_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "metric": [
                    "n observations",
                    "Brent annualised vol",
                    "USDRUB annualised vol",
                    "corr(Brent, USDRUB) daily log-returns",
                    "fitted Student-t df",
                    f"P(Brent<q10 & USDRUB>q90) empirical [{self.start}:{self.end}]",
                    "same tail prob, Gaussian copula (simulated)",
                    "same tail prob, Student-t copula (simulated)",
                ],
                "value": [
                    self.n_obs,
                    f"{self.vol_brent_annual:.1%}",
                    f"{self.vol_fx_annual:.1%}",
                    f"{self.corr_brent_fx:+.3f}",
                    f"{self.student_t_df:.1f}",
                    f"{self.tail_prob_empirical:.2%}",
                    f"{self.tail_prob_gaussian:.2%}",
                    f"{self.tail_prob_student_t:.2%}",
                ],
            }
        )


def _fit_student_t_df(standardized_residuals: np.ndarray) -> float:
    """Method-of-moments df from excess kurtosis: kurtosis_excess = 6/(df-4)
    for a Student-t (df>4). Clipped to a sane [4.5, 30] range — kurtosis-based
    df estimates are noisy and this is a diagnostic, not a precision fit."""
    kurt = float(pd.Series(standardized_residuals).kurt())  # excess kurtosis
    if kurt <= 0.1:
        return 30.0
    df = 6.0 / kurt + 4.0
    return float(np.clip(df, 4.5, 30.0))


def calibrate(cfg: Config, seed: int | None = None) -> CalibrationResult:
    start = cfg.get("monte_carlo.calibration_window_start")
    end = dt.date.today().isoformat()

    fx = fetch_usdrub(start, end)
    brent = fetch_brent(start, end)
    merged = pd.merge(brent, fx, on="date", how="inner").sort_values("date")
    merged["r_brent"] = np.log(merged["brent_usd"]).diff()
    merged["r_fx"] = np.log(merged["usdrub"]).diff()
    merged = merged.dropna(subset=["r_brent", "r_fx"])

    vol_brent_daily = merged["r_brent"].std()
    vol_fx_daily = merged["r_fx"].std()
    corr = merged["r_brent"].corr(merged["r_fx"])

    z_brent = (merged["r_brent"] - merged["r_brent"].mean()) / vol_brent_daily
    z_fx = (merged["r_fx"] - merged["r_fx"].mean()) / vol_fx_daily
    df_fit = _fit_student_t_df(np.concatenate([z_brent.values, z_fx.values]))

    # Empirical tail co-movement: Brent in its bottom decile while USDRUB is
    # in its top decile (oil slumps, ruble weakens) — the amortizer signature.
    q10_brent = merged["r_brent"].quantile(0.10)
    q90_fx = merged["r_fx"].quantile(0.90)
    tail_empirical = float(
        ((merged["r_brent"] <= q10_brent) & (merged["r_fx"] >= q90_fx)).mean()
    )

    rng = np.random.default_rng(seed if seed is not None else cfg.get("seed"))
    n_sim = 200_000
    z1, z2 = sample_correlated(n_sim, corr, "gauss", df=None, rng=rng)
    tail_gauss = float(((z1 <= np.quantile(z1, 0.10)) & (z2 >= np.quantile(z2, 0.90))).mean())
    t1, t2 = sample_correlated(n_sim, corr, "student_t", df=df_fit, rng=rng)
    tail_t = float(((t1 <= np.quantile(t1, 0.10)) & (t2 >= np.quantile(t2, 0.90))).mean())

    return CalibrationResult(
        start=start,
        end=end,
        n_obs=len(merged),
        vol_brent_annual=float(vol_brent_daily * np.sqrt(252)),
        vol_fx_annual=float(vol_fx_daily * np.sqrt(252)),
        corr_brent_fx=float(corr),
        student_t_df=df_fit,
        brent_spot=float(brent["brent_usd"].iloc[-1]),
        usdrub_spot=float(fx["usdrub"].iloc[-1]),
        tail_prob_empirical=tail_empirical,
        tail_prob_gaussian=tail_gauss,
        tail_prob_student_t=tail_t,
    )
