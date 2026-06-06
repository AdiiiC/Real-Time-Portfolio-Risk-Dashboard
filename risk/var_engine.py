import numpy as np
import pandas as pd
from config.settings import MC_SIMULATIONS, PORTFOLIO, RISK_FREE_RATE

def compute_var_monte_carlo(returns_df, prices: dict, confidence: float = 0.95) -> dict:
    tickers = [t for t in returns_df.columns if t in PORTFOLIO]
    weights_nav = np.array([
        prices.get(t, 0) * PORTFOLIO[t] for t in tickers
    ], dtype=float)
    total_nav = weights_nav.sum()
    if total_nav == 0:
        return {"var_95": 0, "cvar_95": 0, "total_nav": 0,
                "corr_matrix": None, "tickers": tickers,
                "attribution": {}, "component_var": {},
                "beta": None, "sharpe": None}

    weights = weights_nav / total_nav

    cov_matrix   = returns_df[tickers].cov().values
    mean_returns = returns_df[tickers].mean().values
    corr_matrix  = returns_df[tickers].corr()

    # ── Monte Carlo ───────────────────────────────────────────────────────────
    rng = np.random.default_rng(42)
    L   = np.linalg.cholesky(cov_matrix + np.eye(len(weights)) * 1e-8)
    z   = rng.standard_normal((MC_SIMULATIONS, len(weights)))
    sim_returns = mean_returns + (z @ L.T)
    portfolio_returns = sim_returns @ weights

    var_pct  = float(np.percentile(portfolio_returns, (1 - confidence) * 100))
    cvar_pct = float(portfolio_returns[portfolio_returns <= var_pct].mean())

    # ── Component VaR (marginal contribution) ─────────────────────────────────
    port_std    = float(np.sqrt(weights @ cov_matrix @ weights))
    z_score     = float(np.abs(np.percentile(np.random.default_rng(0).standard_normal(100_000),
                                              (1 - confidence) * 100)))
    marginal_var = cov_matrix @ weights / (port_std + 1e-12) * z_score
    component_var = {
        t: round(float(w * mv) * total_nav, 2)
        for t, w, mv in zip(tickers, weights, marginal_var)
    }

    # ── P&L attribution ───────────────────────────────────────────────────────
    attribution = {
        t: round(float(w * m) * total_nav, 2)
        for t, w, m in zip(tickers, weights, mean_returns)
    }

    # ── Portfolio beta vs SPY ─────────────────────────────────────────────────
    beta = None
    if "SPY" in returns_df.columns:
        spy_ret   = returns_df["SPY"].values
        port_ret  = returns_df[tickers].values @ weights
        cov_ps    = np.cov(port_ret, spy_ret)
        beta      = round(float(cov_ps[0, 1] / (cov_ps[1, 1] + 1e-12)), 3)

    # ── Sharpe ratio (annualised) ─────────────────────────────────────────────
    port_ret_series = returns_df[tickers].values @ weights
    daily_rf = RISK_FREE_RATE / 252
    excess   = port_ret_series - daily_rf
    sharpe   = round(float(excess.mean() / (excess.std() + 1e-12) * np.sqrt(252)), 3)

    # ── Intraday equity curve support ─────────────────────────────────────────
    # Returns per-ticker daily mean return for dashboard use
    ticker_mean_returns = {t: float(m) for t, m in zip(tickers, mean_returns)}

    return {
        "var_95":               round(-var_pct  * total_nav, 2),
        "cvar_95":              round(-cvar_pct * total_nav, 2),
        "total_nav":            round(total_nav, 2),
        "corr_matrix":          corr_matrix,
        "tickers":              tickers,
        "attribution":          attribution,
        "component_var":        component_var,
        "beta":                 beta,
        "sharpe":               sharpe,
        "ticker_mean_returns":  ticker_mean_returns,
    }

    tickers = [t for t in returns_df.columns if t in PORTFOLIO]
    weights = np.array([
        prices.get(t, 0) * PORTFOLIO[t] for t in tickers
    ], dtype=float)
    total_nav = weights.sum()
    if total_nav == 0:
        return {"var_95": 0, "cvar_95": 0, "total_nav": 0, "corr_matrix": None, "tickers": tickers}

    weights /= total_nav

    cov_matrix   = returns_df[tickers].cov().values
    mean_returns = returns_df[tickers].mean().values
    corr_matrix  = returns_df[tickers].corr()

    rng = np.random.default_rng(42)
    L   = np.linalg.cholesky(cov_matrix + np.eye(len(weights)) * 1e-8)
    z   = rng.standard_normal((MC_SIMULATIONS, len(weights)))
    sim_returns = mean_returns + (z @ L.T)

    portfolio_returns_np = sim_returns @ weights

    var  = float(np.percentile(portfolio_returns_np, (1 - confidence) * 100))
    cvar = float(portfolio_returns_np[portfolio_returns_np <= var].mean())

    # Per-ticker P&L attribution: weight * mean_return * total_nav
    attribution = {
        t: round(float(w * m) * total_nav, 2)
        for t, w, m in zip(tickers, weights, mean_returns)
    }

    return {
        "var_95":      round(-var  * total_nav, 2),
        "cvar_95":     round(-cvar * total_nav, 2),
        "total_nav":   round(total_nav, 2),
        "corr_matrix": corr_matrix,
        "tickers":     tickers,
        "attribution": attribution,
    }