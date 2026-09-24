"""Block 2 property test: the tax wedge must fall faster than revenue when
price drops (progressivity of extraction-tax withdrawal), per the spec.
"""

from __future__ import annotations

import pytest

from src.config import load_config
from src.model.tax import effective_tax_rate, tax_take_mrub


@pytest.fixture
def cfg():
    return load_config()


def test_tax_rate_rises_with_price(cfg):
    usdrub = 90.0
    rates = [effective_tax_rate(p, usdrub, cfg) for p in (20, 40, 60, 80, 100, 150)]
    assert rates == sorted(rates), "effective tax rate should be non-decreasing in price"


def test_tax_vanishes_below_threshold(cfg):
    threshold = cfg.get("tax.effective_rate_fallback.threshold_usd_per_bbl")
    assert effective_tax_rate(threshold - 5, usdrub=90.0, cfg=cfg) == cfg.get(
        "tax.effective_rate_fallback.clip_min"
    )


def test_tax_falls_faster_than_revenue_when_price_drops(cfg):
    """Core property: for any price cut above the threshold, the *percentage*
    drop in absolute tax take exceeds the percentage drop in revenue — i.e.
    the tax has elasticity > 1 w.r.t. price (progressive withdrawal).

    Revenue must scale with price here (revenue = price x volume, volume
    held fixed) — holding revenue artificially constant across the two price
    points (as an earlier version of this test did) tests something else
    entirely: the rate's own sensitivity to price, not tax vs. revenue.
    """
    usdrub = 90.0
    revenue_at_price_high_mrub = 1_000_000.0

    price_high, price_low = 90.0, 60.0
    revenue_at_price_low_mrub = revenue_at_price_high_mrub * (price_low / price_high)

    tax_high = tax_take_mrub(revenue_at_price_high_mrub, price_high, usdrub, cfg)
    tax_low = tax_take_mrub(revenue_at_price_low_mrub, price_low, usdrub, cfg)

    revenue_drop_pct = (
        revenue_at_price_high_mrub - revenue_at_price_low_mrub
    ) / revenue_at_price_high_mrub
    tax_drop_pct = (tax_high - tax_low) / tax_high

    assert tax_drop_pct > revenue_drop_pct, (
        f"tax dropped {tax_drop_pct:.1%} vs revenue proxy {revenue_drop_pct:.1%} "
        "— tax should fall strictly faster than revenue as price declines"
    )


def test_tax_elasticity_exceeds_one_near_threshold(cfg):
    """Elasticity is highest just above the threshold (tax -> 0 there) and
    converges toward 1 at high price — but should always stay >= 1 above the
    threshold, and the near-threshold elasticity should be materially larger
    than the high-price elasticity (the progressivity signature)."""
    usdrub = 90.0
    threshold = cfg.get("tax.effective_rate_fallback.threshold_usd_per_bbl")

    def elasticity(price: float, usdrub: float, eps: float = 0.5) -> float:
        r1 = effective_tax_rate(price, usdrub, cfg) * price
        r2 = effective_tax_rate(price + eps, usdrub, cfg) * (price + eps)
        return ((r2 - r1) / r1) / (eps / price)

    near_threshold_elasticity = elasticity(threshold + 2.0, usdrub)
    high_price_elasticity = elasticity(200.0, usdrub)

    assert near_threshold_elasticity > high_price_elasticity > 1.0 - 1e-6
