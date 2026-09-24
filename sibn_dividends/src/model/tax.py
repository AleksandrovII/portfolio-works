"""Block 2 — tax wedge on oil extraction: fallback effective_tax_rate(price) and
a stub for the exact НДПИ + demper (damper) calculation (stage 4).

Data limitation (disclosed, not papered over): the fallback mode was specified
to be "calibrated by regression on historical reporting 2021-2025", but
Financemarker's `reports` has no dedicated НДПИ/export-duty line — those are
embedded inside `cost_of_sales`, which is only populated for 2025 (null for
2021-2024), and `cash_paid_for_tax` is null throughout. There is nothing to
regress. Instead, `effective_tax_rate` uses an illustrative threshold +
marginal-rate structure matching the real НДПИ formula's shape (a subtracted
per-barrel threshold below which the tax vanishes, then a marginal rate above
it) with parameters from public estimates of the Urals tax wedge — not fit to
SIBN's own P&L. This still gives the required progressivity property
(tax is more price-elastic than revenue) and is designed to be swapped for
the exact chapter-26 НДПИ/damper formula in stage 4 without changing the
call signature.
"""

from __future__ import annotations

from src.config import Config


def effective_tax_rate(price_usd_per_bbl: float, usdrub: float, cfg: Config) -> float:
    """Effective extraction-tax rate (fraction of revenue) as a function of price.

    tax_rub_per_bbl = marginal_rate * max(price - threshold, 0) * usdrub
    effective_tax_rate = tax_rub_per_bbl / (price_usd_per_bbl * usdrub)

    Below the threshold, the rate is 0 (progressivity floor). Above it, the
    rate rises monotonically toward `marginal_rate` as price -> infinity —
    i.e. tax always falls *faster* than revenue when price drops (elasticity
    of tax w.r.t. price is always > 1 above the threshold; see
    `tests/test_tax_progressivity.py`).
    """
    threshold = cfg.get("tax.effective_rate_fallback.threshold_usd_per_bbl")
    marginal_rate = cfg.get("tax.effective_rate_fallback.marginal_rate")
    clip_min = cfg.get("tax.effective_rate_fallback.clip_min")
    clip_max = cfg.get("tax.effective_rate_fallback.clip_max")

    if price_usd_per_bbl <= threshold:
        return clip_min

    taxable = price_usd_per_bbl - threshold
    raw_rate = marginal_rate * taxable / price_usd_per_bbl
    return min(max(raw_rate, clip_min), clip_max)


def tax_take_mrub(
    revenue_mrub: float, price_usd_per_bbl: float, usdrub: float, cfg: Config
) -> float:
    """Absolute tax take in million RUB, given total revenue and the price
    driver for that year/scenario draw."""
    rate = effective_tax_rate(price_usd_per_bbl, usdrub, cfg)
    return revenue_mrub * rate
