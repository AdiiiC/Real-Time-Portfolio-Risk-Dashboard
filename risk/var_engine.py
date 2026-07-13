"""
Risk Engine — Monte Carlo Gaussian VaR, Historical Simulation VaR, Student-t VaR,
Component VaR, Incremental VaR, and Portfolio Performance Metrics.

Three VaR methods compared:
  1. Monte Carlo (Gaussian) — fast, assumes normal returns
  2. Historical Simulation  — uses actual past data, no distributional assumption
  3. Student-t              — fat-tail adjusted, captures extreme events better
"""
import numpy as np
import pandas as pd
from scipy.stats import t as t_dist
from config.settings import MC_SIMULATIONS, PORTFOLIO, RISK_FREE_RATE


def compute_performance_metrics(port_ret_series: np.ndarray) -> dict:
    """
    Compute drawdown, Sortino, Calmar, and annualized return from daily return series.
    """
    cumulative  = np.cumprod(1 + port_ret_series)
    running_max = np.maximum.accumulate(cumulative)
    drawdown_series = (cumulative - running_max) / (running_max + 1e-12)
    max_drawdown    = float(drawdown_series.min())

    daily_rf     = RISK_FREE_RATE / 252
    excess       = port_ret_series - daily_rf
    downside     = excess[excess < 0]
    downside_std = float(np.std(downside)) if len(downside) > 1 else 1e-12
    sortino      = round(float(excess.mean() / (downside_std + 1e-12) * np.sqrt(252)), 3)

    ann_return = float((1 + port_ret_series.mean()) ** 252 - 1)
    calmar     = round(ann_return / (abs(max_drawdown) + 1e-12), 3)

    return {
        "max_drawdown":      round(max_drawdown * 100, 2),   # as %
        "sortino":           sortino,
        "calmar":            calmar,
        "ann_return":        round(ann_return * 100, 2),     # as %
        "drawdown_series":   drawdown_series,
        "cumulative_series": cumulative,
    }


def compute_var_monte_carlo(returns_df: pd.DataFrame, prices: dict,
                             confidence: float = 0.95,
                             portfolio: dict = None) -> dict:
    """
    Compute portfolio risk metrics using three VaR methods + performance analytics.

    Args:
        returns_df:  Daily returns DataFrame (dates x tickers)
        prices:      Current market prices {ticker: price}
        confidence:  VaR confidence level (default 0.95 = 95%)
        portfolio:   {ticker: shares} — defaults to settings.PORTFOLIO

    Returns dict with keys:
        var_95, cvar_95         — Monte Carlo VaR and Conditional VaR ($)
        var_hist_sim            — Historical Simulation VaR ($)
        var_t                   — Student-t fat-tail VaR ($)
        total_nav               — Portfolio market value ($)
        component_var           — Per-ticker risk contribution ($)
        incremental_var         — VaR change if each position removed ($)
        attribution             — Expected daily P&L per ticker ($)
        beta, sharpe, sortino   — Risk-adjusted performance ratios
        max_drawdown, calmar    — Drawdown analytics
        port_ret_series         — Historical daily portfolio returns (array)
        drawdown_series         — Historical drawdown at each day (array)
        cumulative_series       — Cumulative wealth index (array)
    """
    if portfolio is None:
        portfolio = PORTFOLIO

    tickers     = [t for t in returns_df.columns if t in portfolio]
    weights_nav = np.array([prices.get(t, 0) * portfolio[t] for t in tickers], dtype=float)
    total_nav   = weights_nav.sum()

    _empty = dict(
        var_95=0, cvar_95=0, var_hist_sim=0, var_t=0, total_nav=0,
        corr_matrix=None, tickers=tickers,
        attribution={}, component_var={}, incremental_var={},
        beta=None, sharpe=None, max_drawdown=None, sortino=None,
        calmar=None, ann_return=None, port_ret_series=None,
        drawdown_series=None, cumulative_series=None,
        ticker_mean_returns={},
    )
    if total_nav == 0 or not tickers:
        return _empty

    weights      = weights_nav / total_nav
    cov_matrix   = returns_df[tickers].cov().values
    mean_returns = returns_df[tickers].mean().values
    corr_matrix  = returns_df[tickers].corr()

    # Historical portfolio daily returns
    hist_port_returns = returns_df[tickers].values @ weights

    # ── Method 1: Monte Carlo (Gaussian) ──────────────────────────────────────
    rng = np.random.default_rng()
    L   = np.linalg.cholesky(cov_matrix + np.eye(len(weights)) * 1e-8)
    z   = rng.standard_normal((MC_SIMULATIONS, len(weights)))
    sim_returns       = mean_returns + (z @ L.T)
    portfolio_returns = sim_returns @ weights
    var_pct  = float(np.percentile(portfolio_returns, (1 - confidence) * 100))
    cvar_pct = float(portfolio_returns[portfolio_returns <= var_pct].mean())

    # ── Method 2: Historical Simulation ───────────────────────────────────────
    var_hist_pct = float(np.percentile(hist_port_returns, (1 - confidence) * 100))

    # ── Method 3: Student-t (fat-tail correction) ─────────────────────────────
    try:
        df_t, loc_t, scale_t = t_dist.fit(hist_port_returns, floc=hist_port_returns.mean())
        var_t_pct = float(t_dist.ppf(1 - confidence, df=df_t, loc=loc_t, scale=scale_t))
    except Exception:
        var_t_pct = var_pct

    # ── Component VaR (per-ticker risk contribution) ──────────────────────────
    port_std  = float(np.sqrt(weights @ cov_matrix @ weights))
    z_score   = float(abs(
        np.percentile(np.random.default_rng(0).standard_normal(100_000),
                      (1 - confidence) * 100)
    ))
    marginal_var  = cov_matrix @ weights / (port_std + 1e-12) * z_score
    component_var = {
        t: round(float(w * mv) * total_nav, 2)
        for t, w, mv in zip(tickers, weights, marginal_var)
    }

    # ── Incremental VaR (impact of removing each position) ───────────────────
    base_var        = round(-var_pct * total_nav, 2)
    incremental_var = {}
    for i, ticker in enumerate(tickers):
        mask    = [j for j in range(len(tickers)) if j != i]
        if not mask:
            incremental_var[ticker] = base_var
            continue
        wn_excl  = weights_nav[mask]
        nav_excl = wn_excl.sum()
        if nav_excl == 0:
            incremental_var[ticker] = base_var
            continue
        w_excl   = wn_excl / nav_excl
        cov_excl = cov_matrix[np.ix_(mask, mask)]
        mn_excl  = mean_returns[mask]
        try:
            L_excl  = np.linalg.cholesky(cov_excl + np.eye(len(w_excl)) * 1e-8)
            z_excl  = rng.standard_normal((5_000, len(w_excl)))
            sim_ex  = mn_excl + (z_excl @ L_excl.T)
            pr_excl = sim_ex @ w_excl
            var_excl = round(
                -float(np.percentile(pr_excl, (1 - confidence) * 100)) * nav_excl, 2
            )
        except Exception:
            var_excl = base_var
        incremental_var[ticker] = round(base_var - var_excl, 2)

    # ── P&L Attribution ───────────────────────────────────────────────────────
    attribution = {
        t: round(float(w * m) * total_nav, 2)
        for t, w, m in zip(tickers, weights, mean_returns)
    }

    # ── Beta vs SPY ───────────────────────────────────────────────────────────
    beta = None
    if "SPY" in returns_df.columns:
        spy_ret = returns_df["SPY"].values
        cov_ps  = np.cov(hist_port_returns, spy_ret)
        beta    = round(float(cov_ps[0, 1] / (cov_ps[1, 1] + 1e-12)), 3)

    # ── Sharpe Ratio (annualised) ──────────────────────────────────────────────
    daily_rf = RISK_FREE_RATE / 252
    excess   = hist_port_returns - daily_rf
    sharpe   = round(float(excess.mean() / (excess.std() + 1e-12) * np.sqrt(252)), 3)

    # ── Drawdown + Sortino + Calmar ───────────────────────────────────────────
    perf = compute_performance_metrics(hist_port_returns)

    return {
        "var_95":            round(-var_pct      * total_nav, 2),
        "cvar_95":           round(-cvar_pct     * total_nav, 2),
        "var_hist_sim":      round(-var_hist_pct * total_nav, 2),
        "var_t":             round(-var_t_pct    * total_nav, 2),
        "total_nav":         round(total_nav, 2),
        "corr_matrix":       corr_matrix,
        "tickers":           tickers,
        "attribution":       attribution,
        "component_var":     component_var,
        "incremental_var":   incremental_var,
        "beta":              beta,
        "sharpe":            sharpe,
        "max_drawdown":      perf["max_drawdown"],
        "sortino":           perf["sortino"],
        "calmar":            perf["calmar"],
        "ann_return":        perf["ann_return"],
        "drawdown_series":   perf["drawdown_series"],
        "cumulative_series": perf["cumulative_series"],
        "port_ret_series":   hist_port_returns,
        "ticker_mean_returns": {t: float(m) for t, m in zip(tickers, mean_returns)},
    }
