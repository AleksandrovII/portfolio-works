"""Shared correlated-sampling primitive (Gaussian vs Student-t copula),
used by both calibration.py (tail-probability diagnostics) and
montecarlo.py (path simulation) — split out to avoid a circular import
between the two.
"""

from __future__ import annotations

import numpy as np


def sample_correlated(
    n: int, corr: float, dist: str, df: float | None, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """n correlated draws, each marginally standard-normal (dist='gauss') or
    standard Student-t with `df` degrees of freedom (dist='student_t'),
    correlation `corr`."""
    cov = np.array([[1.0, corr], [corr, 1.0]])
    z = rng.multivariate_normal(mean=[0.0, 0.0], cov=cov, size=n)
    if dist == "gauss":
        return z[:, 0], z[:, 1]
    if dist == "student_t":
        if df is None:
            raise ValueError("df required for student_t")
        w = rng.chisquare(df, size=n)
        scale = np.sqrt(df / w)
        return z[:, 0] * scale, z[:, 1] * scale
    raise ValueError(f"Unknown dist {dist!r}")
