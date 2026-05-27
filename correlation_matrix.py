"""
Portfolio asset correlation matrix.

Fetches daily candles from T-Invest, aligns prices by date, computes Pearson
(or Spearman) correlation on log-returns, and produces heatmap + CSV exports.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import pearsonr, spearmanr

from t_tech.invest import AsyncClient, CandleInterval
from t_tech.invest.utils import now

logger = logging.getLogger(__name__)

TRADABLE_TYPES = {'share', 'bond', 'etf', 'precious_metal'}
ReturnMethod = Literal['log', 'simple']
CorrMethod = Literal['pearson', 'spearman']

# Bloomberg-style palette (shared with tinvest_to_finance charts)
_DARK_BG = '#0D1117'
_PANEL_BG = '#161B22'
_TEXT = '#E6EDF3'
_DIM = '#8B949E'
_GRID = '#30363D'
_ACCENT1 = '#58A6FF'


@dataclass
class CorrelationResult:
    """Correlation analysis output."""

    corr: pd.DataFrame
    pvalues: pd.DataFrame
    returns: pd.DataFrame
    weights: pd.Series
    days: int
    tickers: List[str]
    corr_method: CorrMethod = 'pearson'
    return_method: ReturnMethod = 'log'


def build_figi_map(
    positions: List[Dict],
    *,
    min_weight_pct: float = 0.0,
    top_n: Optional[int] = None,
) -> Dict[str, str]:
    """
    Map FIGI → ticker for tradable portfolio positions.

    Optionally filter to positions above min_weight_pct of total RUB value,
    or keep only the top_n largest by weight.
    """
    total_rub = sum(float(p.get('rub_value') or 0) for p in positions)
    candidates: List[Tuple[float, str, str]] = []

    for p in positions:
        if p.get('instrument_type') not in TRADABLE_TYPES:
            continue
        figi = p.get('figi', '')
        ticker = p.get('ticker', figi[:6] if figi else '')
        if not figi or not ticker:
            continue
        rub = float(p.get('rub_value') or 0)
        weight = (rub / total_rub * 100) if total_rub > 0 else 0.0
        if weight < min_weight_pct:
            continue
        candidates.append((rub, figi, ticker))

    if top_n is not None and top_n > 0:
        candidates.sort(key=lambda x: x[0], reverse=True)
        candidates = candidates[:top_n]

    figi_map: Dict[str, str] = {}
    seen: set = set()
    for _, figi, ticker in sorted(candidates, key=lambda x: x[0], reverse=True):
        if ticker in seen:
            continue
        figi_map[figi] = ticker
        seen.add(ticker)
    return figi_map


async def _get_candles(
    client: AsyncClient,
    figi: str,
    days: int,
) -> Optional[pd.Series]:
    """Daily close prices indexed by UTC date."""
    try:
        end = now()
        start = end - timedelta(days=days)
        resp = await client.market_data.get_candles(
            instrument_id=figi,
            from_=start,
            to=end,
            interval=CandleInterval.CANDLE_INTERVAL_DAY,
        )
        rows = []
        for c in resp.candles:
            if not c.close or not c.time:
                continue
            price = float(c.close.units) + float(c.close.nano) / 1e9
            rows.append((pd.Timestamp(c.time).normalize(), price))
        if len(rows) <= 10:
            return None
        series = pd.Series(
            {ts: px for ts, px in rows},
            dtype=float,
        ).sort_index()
        return series[~series.index.duplicated(keep='last')]
    except Exception as e:
        logger.warning('No history for %s: %s', figi, e)
        return None


async def fetch_price_history(
    figi_map: Dict[str, str],
    days: int,
    token: str,
) -> Dict[str, pd.Series]:
    """Fetch aligned daily close series for each FIGI."""
    async with AsyncClient(token) as client:
        figis = list(figi_map.keys())
        results = await asyncio.gather(
            *[_get_candles(client, f, days) for f in figis],
            return_exceptions=True,
        )
    history: Dict[str, pd.Series] = {}
    for figi, res in zip(figis, results):
        if isinstance(res, pd.Series):
            history[figi_map[figi]] = res
    return history


def build_returns_matrix(
    history: Dict[str, pd.Series],
    method: ReturnMethod = 'log',
) -> pd.DataFrame:
    """Align prices on common trading dates and compute returns."""
    if len(history) < 2:
        raise ValueError('Need at least two assets with price history')

    prices = pd.DataFrame(history).dropna(how='all')
    prices = prices.dropna(how='any')
    if len(prices) < 11:
        raise ValueError(
            f'Insufficient overlapping history ({len(prices)} days); need ≥11'
        )

    if method == 'log':
        return np.log(prices / prices.shift(1)).dropna()
    return prices.pct_change().dropna()


def _pair_pvalue(
    x: pd.Series,
    y: pd.Series,
    method: CorrMethod,
) -> float:
    mask = x.notna() & y.notna()
    a, b = x[mask], y[mask]
    if len(a) < 3:
        return 1.0
    if method == 'spearman':
        _, p = spearmanr(a, b)
    else:
        _, p = pearsonr(a, b)
    return float(p)


def compute_correlation(
    returns: pd.DataFrame,
    method: CorrMethod = 'pearson',
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Correlation matrix and pairwise p-values."""
    corr = returns.corr(method=method)
    n = len(corr)
    pval = pd.DataFrame(
        np.ones((n, n)),
        index=corr.index,
        columns=corr.columns,
    )
    for i in corr.index:
        for j in corr.columns:
            if i != j:
                pval.loc[i, j] = _pair_pvalue(returns[i], returns[j], method)
            else:
                pval.loc[i, j] = 0.0
    return corr, pval


def portfolio_weights(
    positions: List[Dict],
    tickers: List[str],
) -> pd.Series:
    """RUB weight (%) per ticker for assets in the correlation set."""
    rub_by_ticker: Dict[str, float] = {}
    for p in positions:
        t = p.get('ticker', '')
        if t not in tickers:
            continue
        rub_by_ticker[t] = rub_by_ticker.get(t, 0.0) + float(p.get('rub_value') or 0)
    total = sum(rub_by_ticker.values())
    if total <= 0:
        return pd.Series(0.0, index=tickers)
    return pd.Series(
        {t: rub_by_ticker.get(t, 0) / total * 100 for t in tickers},
        name='weight_pct',
    )


def print_summary(result: CorrelationResult) -> None:
    """Print correlation matrix and highlight pairs to console."""
    corr, pval = result.corr, result.pvalues
    n = len(corr)

    print('\n' + '═' * 72)
    print(f'  CORRELATION MATRIX  ·  {result.days}d  ·  {n} assets  ·  '
          f'{len(result.returns)} return observations')
    print('═' * 72)

    col_w = max(8, max(len(t) for t in corr.columns) + 1)
    header = f"{'':>{col_w}}" + ''.join(f'{t:>{col_w}}' for t in corr.columns)
    print(header)
    for i in corr.index:
        row = f'{i:>{col_w}}'
        for j in corr.columns:
            v = corr.loc[i, j]
            cell = '1.00' if i == j else f'{v:+.2f}'
            row += f'{cell:>{col_w}}'
        print(row)

    off_diag = []
    for i, a in enumerate(corr.index):
        for j, b in enumerate(corr.columns):
            if j >= i:
                continue
            off_diag.append((corr.loc[a, b], pval.loc[a, b], a, b))
    if not off_diag:
        print('═' * 72 + '\n')
        return

    avg = float(np.mean([x[0] for x in off_diag]))
    print(f'\n  Average pairwise correlation: {avg:.3f}')

    if not result.weights.empty:
        print('\n  Portfolio weights (%):')
        for t, w in result.weights.sort_values(ascending=False).items():
            print(f'    {t:<12} {w:5.1f}%')

    off_diag.sort(key=lambda x: x[0], reverse=True)
    print('\n  Highest correlation pairs:')
    for r, p, a, b in off_diag[:5]:
        sig = ' (significant)' if p < 0.05 else ''
        print(f'    {a} ↔ {b}:  r = {r:+.3f}{sig}')

    off_diag.sort(key=lambda x: x[0])
    print('\n  Lowest correlation pairs (diversification):')
    for r, p, a, b in off_diag[:5]:
        print(f'    {a} ↔ {b}:  r = {r:+.3f}')

    print('═' * 72 + '\n')


def export_csv(
    result: CorrelationResult,
    out_dir: str = '.',
) -> Tuple[str, str]:
    """Write correlation matrix and pairwise list to CSV."""
    import os

    matrix_path = os.path.join(out_dir, 'correlation_matrix.csv')
    pairs_path = os.path.join(out_dir, 'correlation_pairs.csv')

    result.corr.to_csv(matrix_path, float_format='%.4f')

    rows = []
    tickers = list(result.corr.columns)
    for i, a in enumerate(tickers):
        for j, b in enumerate(tickers):
            if j >= i:
                continue
            rows.append({
                'asset_a': a,
                'asset_b': b,
                'correlation': result.corr.loc[a, b],
                'p_value': result.pvalues.loc[a, b],
                'weight_a_pct': result.weights.get(a, 0),
                'weight_b_pct': result.weights.get(b, 0),
            })
    pairs = pd.DataFrame(rows).sort_values('correlation', ascending=False)
    pairs.to_csv(pairs_path, index=False, float_format='%.6f')

    return matrix_path, pairs_path


def _dark_theme() -> None:
    plt.rcParams.update({
        'figure.facecolor': _DARK_BG,
        'axes.facecolor': _PANEL_BG,
        'axes.edgecolor': _GRID,
        'axes.labelcolor': _DIM,
        'text.color': _TEXT,
        'xtick.color': _DIM,
        'ytick.color': _DIM,
        'grid.color': _GRID,
        'font.family': 'sans-serif',
    })


def plot_heatmap(
    result: CorrelationResult,
    out_path: str = 'correlation_matrix.png',
) -> None:
    """Lower-triangle correlation heatmap with significance stars."""
    corr, pval = result.corr, result.pvalues
    n = len(corr)

    _dark_theme()
    sz = max(10, n * 0.9 + 2)
    fig, ax = plt.subplots(figsize=(sz, sz * 0.88), facecolor=_DARK_BG)
    ax.set_facecolor(_PANEL_BG)

    cmap = LinearSegmentedColormap.from_list(
        'fin_corr',
        ['#C0392B', '#922B21', _PANEL_BG, '#1A5276', '#2980B9'],
        N=256,
    )

    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
    tickers = list(corr.columns)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tickers, rotation=45, ha='right', fontsize=9, color=_TEXT)
    ax.set_yticklabels(tickers, fontsize=9, color=_TEXT)

    for i in range(n):
        for j in range(n):
            if j > i:
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1, color=_DARK_BG, zorder=2,
                ))
                continue
            val = corr.iloc[i, j]
            p = pval.iloc[i, j]
            star = (
                '***' if p < 0.001 else
                '**' if p < 0.01 else
                '*' if p < 0.05 else ''
            )
            if i == j:
                ax.text(
                    j, i, tickers[i], ha='center', va='center',
                    fontsize=8, fontweight='bold', color=_ACCENT1,
                )
            else:
                tc = _TEXT if abs(val) < 0.6 else 'white'
                ax.text(
                    j, i, f'{val:.2f}\n{star}',
                    ha='center', va='center', fontsize=8, color=tc,
                    fontweight='bold' if abs(val) > 0.5 else 'normal',
                )

    for k in range(n + 1):
        ax.axhline(k - 0.5, color=_DARK_BG, linewidth=0.7)
        ax.axvline(k - 0.5, color=_DARK_BG, linewidth=0.7)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color=_DIM)
    cbar.outline.set_edgecolor(_GRID)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=_DIM, fontsize=8)
    cbar_label = 'Spearman ρ' if result.corr_method == 'spearman' else 'Pearson r'
    cbar.set_label(cbar_label, color=_DIM, fontsize=9)

    ret_label = 'log-return' if result.return_method == 'log' else 'simple return'
    ax.set_title(
        f'ASSET CORRELATION MATRIX   ·   {result.days}d {ret_label}  ·  '
        f'{n} assets',
        color=_TEXT, fontsize=13, fontweight='bold', pad=16,
    )
    fig.text(0.01, 0.01, '* p<0.05  ** p<0.01  *** p<0.001', color=_DIM, fontsize=8)
    fig.text(
        0.99, 0.01, 'T-Invest Portfolio Analytics',
        ha='right', color=_DIM, fontsize=8, style='italic',
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches='tight', facecolor=_DARK_BG)
    plt.close(fig)


def run_correlation_analysis(
    positions: List[Dict],
    *,
    token: str,
    days: int = 180,
    min_weight_pct: float = 0.0,
    top_n: Optional[int] = None,
    corr_method: CorrMethod = 'pearson',
    return_method: ReturnMethod = 'log',
    plot: bool = True,
    export: bool = True,
    print_report: bool = True,
    out_dir: str = '.',
    heatmap_path: str = 'correlation_matrix.png',
) -> Optional[CorrelationResult]:
    """
    End-to-end correlation analysis for portfolio positions.

    Returns CorrelationResult or None if insufficient data.
    """
    figi_map = build_figi_map(
        positions,
        min_weight_pct=min_weight_pct,
        top_n=top_n,
    )
    if len(figi_map) < 2:
        print('⚠  Not enough instruments for correlation analysis.')
        return None

    print(f'Loading {days}-day history for {len(figi_map)} assets…')
    history = asyncio.run(fetch_price_history(figi_map, days, token))
    if len(history) < 2:
        print('⚠  Could not load enough price history.')
        return None

    try:
        returns = build_returns_matrix(history, method=return_method)
    except ValueError as e:
        print(f'⚠  {e}')
        return None

    corr, pval = compute_correlation(returns, method=corr_method)
    weights = portfolio_weights(positions, list(corr.columns))
    result = CorrelationResult(
        corr=corr,
        pvalues=pval,
        returns=returns,
        weights=weights,
        days=days,
        tickers=list(corr.columns),
        corr_method=corr_method,
        return_method=return_method,
    )

    if print_report:
        print_summary(result)
    if export:
        matrix_csv, pairs_csv = export_csv(result, out_dir)
        print(f'✓ Correlation matrix CSV → {matrix_csv}')
        print(f'✓ Pairwise correlations  → {pairs_csv}')
    if plot:
        plot_heatmap(result, heatmap_path)
        print(f'✓ Correlation heatmap    → {heatmap_path}')

    return result
