"""Block 1 — netback (ruble economics of a barrel) with upstream/downstream split.

Design note (agreed with the user given data constraints): Financemarker's
`operations` endpoint has no segment revenue (₽) — only physical volumes, and
those have a real disclosure gap (detailed oil_production/gas_production stop
after 2020; 2021-2022 has nothing; 2023-2025 only has aggregated
`hydrocarbon_production` + `oil_refining`). So this module builds a *physical*
netback model:

  upstream_revenue  = exported (non-refined) hydrocarbon volume  x  crude netback/bbl
  downstream_revenue = refined volume  x  downstream netback/tonne

For historical years we don't reconstruct revenue bottom-up from this chain —
we use the actual reported IFRS revenue instead (see `margins.py`) and only use
this module's *modeled* revenue (and its implied "revenue per boe") as the
explanatory driver for forecast years, and — via `historical_netback_proxy` —
as an *observed* per-boe realization (actual revenue / actual boe produced) for
calibrating the margin regression in Block 3. This sidesteps needing a
historical Urals-Brent discount series, which isn't available without the
(currently empty, user-filled) Minfin Urals CSV.

All monetary figures are in million RUB (`_mrub`), matching the units the
Financemarker `reports` endpoint uses — see findata_client.py docstring.
"""

from __future__ import annotations

import pandas as pd

from src.config import Config
from src.data.findata_client import FindataClient

# operation_metric_id values that plausibly represent total hydrocarbon output,
# in the order we prefer them (detailed oil_production predates the 2021-2022
# disclosure gap; hydrocarbon_production is the only thing reported 2023+).
_PRODUCTION_METRIC_PRIORITY = ("oil_production", "hydrocarbon_production")
_REFINING_METRIC = "oil_refining"


def _true_tonnes(df: pd.DataFrame) -> pd.Series:
    """Financemarker's `value`/`amount` display-scaling is inconsistent across
    metric vintages (verified: old oil_production divides by amount=1000, new
    hydrocarbon_production/oil_refining use amount=1e6 as a no-op multiplier).
    `original_value * original_amount` is unambiguous — it's always in the
    literal `original_unit` (tonnes, for the metrics we use here).
    """
    if df.empty:
        return pd.Series(dtype=float)
    return (df["original_value"] * df["original_amount"]).astype(float)


def production_and_refining_volumes(client: FindataClient, cfg: Config) -> pd.DataFrame:
    """Annual hydrocarbon production and refining volumes, in tonnes.

    Returns a DataFrame indexed by year with columns:
      hydrocarbon_tonnes, refining_tonnes, domestic_refining_share
    Gaps (no operational disclosure that year) are forward/back-filled; if a
    year has no coverage at all even after filling, `domestic_refining_share`
    falls back to `physical.domestic_refining_share_fallback` from config.
    """
    production = pd.Series(dtype=float)
    for metric_id in _PRODUCTION_METRIC_PRIORITY:
        series = _true_tonnes(client.operations(metric_id))
        if series.empty:
            continue
        years = client.operations(metric_id)["year"]
        series.index = years
        production = series.combine_first(production)

    refining_df = client.operations(_REFINING_METRIC)
    refining = _true_tonnes(refining_df)
    refining.index = refining_df["year"]

    df = pd.DataFrame({"hydrocarbon_tonnes": production, "refining_tonnes": refining})
    df = df.sort_index()
    df["domestic_refining_share"] = df["refining_tonnes"] / df["hydrocarbon_tonnes"]
    df["domestic_refining_share"] = df["domestic_refining_share"].clip(0.0, 1.0)
    df["domestic_refining_share"] = df["domestic_refining_share"].ffill().bfill()
    fallback = cfg.get("physical.domestic_refining_share_fallback")
    df["domestic_refining_share"] = df["domestic_refining_share"].fillna(fallback)
    return df


def crude_netback_rub_per_bbl(
    brent_usd: float, discount_usd: float, freight_usd: float, usdrub: float
) -> float:
    """netback_rub = (Brent - Urals discount - freight/logistics) * USDRUB."""
    return (brent_usd - discount_usd - freight_usd) * usdrub


def downstream_netback_rub_per_tonne(
    downstream_margin_usd_per_tonne: float, usdrub: float
) -> float:
    return downstream_margin_usd_per_tonne * usdrub


def modeled_revenue_mrub(
    hydrocarbon_tonnes: float,
    refining_tonnes: float,
    domestic_refining_share: float,
    brent_usd: float,
    discount_usd: float,
    usdrub: float,
    cfg: Config,
    refining_volume_shock: float = 0.0,
) -> dict[str, float]:
    """Physical-volume netback model for one year/scenario draw.

    Returns upstream/downstream/total revenue in million RUB, plus total boe
    and the blended "netback proxy" (revenue per boe) used as the Block 3
    regression driver for forecast years.
    """
    bbl_per_tonne = cfg.get("physical.bbl_per_tonne_oil")
    freight_usd = cfg.get("netback.freight_logistics_usd_per_bbl")
    downstream_margin_usd = cfg.get("netback.downstream_margin_usd_per_tonne")

    exported_tonnes = hydrocarbon_tonnes * (1 - domestic_refining_share)
    refining_tonnes_shocked = refining_tonnes * (1 + refining_volume_shock)

    crude_rub_bbl = crude_netback_rub_per_bbl(brent_usd, discount_usd, freight_usd, usdrub)
    downstream_rub_t = downstream_netback_rub_per_tonne(downstream_margin_usd, usdrub)

    upstream_revenue_rub = exported_tonnes * bbl_per_tonne * crude_rub_bbl
    downstream_revenue_rub = refining_tonnes_shocked * downstream_rub_t
    total_revenue_rub = upstream_revenue_rub + downstream_revenue_rub

    total_boe = hydrocarbon_tonnes * bbl_per_tonne
    netback_proxy_rub_per_boe = total_revenue_rub / total_boe if total_boe else float("nan")

    return {
        "upstream_revenue_mrub": upstream_revenue_rub / 1e6,
        "downstream_revenue_mrub": downstream_revenue_rub / 1e6,
        "total_revenue_mrub": total_revenue_rub / 1e6,
        "total_boe": total_boe,
        "netback_proxy_rub_per_boe": netback_proxy_rub_per_boe,
    }


def historical_netback_proxy(
    client: FindataClient, cfg: Config, revenue_by_year_mrub: pd.Series
) -> pd.Series:
    """Observed "revenue per boe" for historical years: actual IFRS revenue
    divided by actual hydrocarbon output (boe). This is the X-regressor for
    Block 3's margin regression — no discount/FX assumption needed since it
    uses realized revenue directly.
    """
    volumes = production_and_refining_volumes(client, cfg)
    bbl_per_tonne = cfg.get("physical.bbl_per_tonne_oil")
    boe = volumes["hydrocarbon_tonnes"] * bbl_per_tonne
    revenue_rub = revenue_by_year_mrub * 1e6
    proxy = (revenue_rub / boe).reindex(revenue_by_year_mrub.index)
    proxy.name = "netback_proxy_rub_per_boe"
    return proxy
