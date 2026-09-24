"""Block 3 (P&L waterfall) and Block 4 (deterministic single-scenario DPS).

Revenue -> EBITDA (netback-regression margin) -> EBIT (- D&A) -> EBT (- interest
on modeled net debt) -> consolidated NI (- income tax) -> NI attributable to
shareholders (- minority interest) -> DPS = payout * NI / n_shares.

D&A%, interest rate, income tax rate and minority share are trailing historical
averages (see margins.historical_pnl_detail) rather than netback-regressed —
they're structural/accounting ratios, not price-driven. Interest is charged on
the model's own rolled-forward net debt, which is what fixes the leverage blind
spot a direct NI-margin-on-price regression had (see backtest.py docstring).

Debt: grows by the shortfall if FCF after dividends is negative; otherwise
unchanged (no discretionary paydown modeled) — "долг моделируй просто" per spec.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.config import Config


@dataclass
class YearResult:
    year: int
    revenue_mrub: float
    ebitda_mrub: float
    da_mrub: float
    ebit_mrub: float
    interest_expense_mrub: float
    ebt_mrub: float
    ni_consolidated_mrub: float
    ni_mrub: float
    fcf_mrub: float
    prior_net_debt_mrub: float
    net_debt_ebitda: float
    payout_ratio: float
    dividends_mrub: float
    dps_rub: float
    fcf_after_dividends_mrub: float
    debt_change_mrub: float
    new_net_debt_mrub: float


def select_payout_ratio(net_debt_ebitda: float, cfg: Config) -> float:
    threshold = cfg.get("payout.net_debt_ebitda_stress_threshold")
    if net_debt_ebitda > threshold:
        return cfg.get("payout.stress_ratio")
    return cfg.get("payout.base_ratio")


def compute_year(
    year: int,
    revenue_mrub: float,
    ebitda_margin: float,
    da_margin: float,
    interest_rate_on_net_debt: float,
    income_tax_rate: float,
    minority_share: float,
    fcf_margin: float,
    prior_net_debt_mrub: float,
    n_shares: int,
    cfg: Config,
) -> YearResult:
    ebitda_mrub = revenue_mrub * ebitda_margin
    da_mrub = revenue_mrub * da_margin
    ebit_mrub = ebitda_mrub - da_mrub

    interest_expense_mrub = prior_net_debt_mrub * interest_rate_on_net_debt
    ebt_mrub = ebit_mrub - interest_expense_mrub

    ni_consolidated_mrub = ebt_mrub * (1 - income_tax_rate)
    ni_mrub = ni_consolidated_mrub * (1 - minority_share)

    fcf_mrub = revenue_mrub * fcf_margin

    net_debt_ebitda = prior_net_debt_mrub / ebitda_mrub if ebitda_mrub > 0 else float("inf")
    payout = select_payout_ratio(net_debt_ebitda, cfg)

    dividends_mrub = payout * ni_mrub
    dps_rub = dividends_mrub * 1e6 / n_shares

    fcf_after_dividends_mrub = fcf_mrub - dividends_mrub
    debt_change_mrub = max(0.0, -fcf_after_dividends_mrub) if cfg.get("debt.roll_forward") else 0.0
    new_net_debt_mrub = prior_net_debt_mrub + debt_change_mrub

    return YearResult(
        year=year,
        revenue_mrub=revenue_mrub,
        ebitda_mrub=ebitda_mrub,
        da_mrub=da_mrub,
        ebit_mrub=ebit_mrub,
        interest_expense_mrub=interest_expense_mrub,
        ebt_mrub=ebt_mrub,
        ni_consolidated_mrub=ni_consolidated_mrub,
        ni_mrub=ni_mrub,
        fcf_mrub=fcf_mrub,
        prior_net_debt_mrub=prior_net_debt_mrub,
        net_debt_ebitda=net_debt_ebitda,
        payout_ratio=payout,
        dividends_mrub=dividends_mrub,
        dps_rub=dps_rub,
        fcf_after_dividends_mrub=fcf_after_dividends_mrub,
        debt_change_mrub=debt_change_mrub,
        new_net_debt_mrub=new_net_debt_mrub,
    )
