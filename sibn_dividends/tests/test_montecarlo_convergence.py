"""Block 4 property test: Monte Carlo mean DPS converges as n_paths grows,
consistent with a finite-variance estimator (standard error ~ 1/sqrt(n))."""

from __future__ import annotations

import numpy as np
import pytest

from src.config import load_config
from src.data.findata_client import FindataClient
from src.scenarios.calibration import calibrate
from src.scenarios.montecarlo import run_monte_carlo


@pytest.fixture(scope="module")
def client():
    return FindataClient()


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def calib(cfg):
    return calibrate(cfg)


def _base_run(client, cfg, calib, n_paths, seed):
    return run_monte_carlo(
        client, cfg, calib,
        discount_min=17, discount_base=20, discount_max=23,
        n_paths=n_paths, seed=seed,
    )


def test_two_independent_large_runs_agree_within_standard_error(client, cfg, calib):
    """Two independent large-n runs (different seeds) should land within a
    handful of standard errors of each other — a basic sanity check that the
    estimator isn't biased/broken (e.g. seed leakage, non-stationary loop state)."""
    n = 5000
    draws_a = _base_run(client, cfg, calib, n, seed=1)
    draws_b = _base_run(client, cfg, calib, n, seed=2)

    se_a = draws_a.dps.std(ddof=1) / np.sqrt(n)
    se_b = draws_b.dps.std(ddof=1) / np.sqrt(n)
    combined_se = np.sqrt(se_a**2 + se_b**2)

    assert abs(draws_a.dps.mean() - draws_b.dps.mean()) < 5 * combined_se


def test_standard_error_shrinks_with_more_paths(client, cfg, calib):
    """SE(mean) should shrink roughly as 1/sqrt(n) — quadrupling n should
    roughly halve the standard error."""
    small = _base_run(client, cfg, calib, n_paths=1000, seed=42)
    large = _base_run(client, cfg, calib, n_paths=4000, seed=42)

    se_small = small.dps.std(ddof=1) / np.sqrt(1000)
    se_large = large.dps.std(ddof=1) / np.sqrt(4000)

    # Allow generous slack (sampling noise in the SE estimate itself) around
    # the theoretical ~0.5x ratio.
    ratio = se_large / se_small
    assert 0.3 < ratio < 0.8, f"SE ratio {ratio:.2f} not consistent with 1/sqrt(n) scaling"


def test_dps_draws_are_finite_and_nonnegative(client, cfg, calib):
    draws = _base_run(client, cfg, calib, n_paths=2000, seed=7)
    assert np.all(np.isfinite(draws.dps))
    assert np.all(draws.dps >= 0)
