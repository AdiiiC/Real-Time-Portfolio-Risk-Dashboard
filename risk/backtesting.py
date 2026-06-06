"""
VaR Backtesting — Kupiec Proportion of Failures (POF) test.
Compares predicted VaR against realized next-day returns.
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2


def kupiec_test(var_series: pd.Series, realized_series: pd.Series,
                confidence: float = 0.95) -> dict:
    """
    var_series      : Series of predicted VaR values (positive = loss) aligned by date
    realized_series : Series of realized portfolio P&L (negative = loss)
    confidence      : VaR confidence level (e.g. 0.95)
    Returns dict with test statistic, p-value, and breach details.
    """
    aligned = pd.concat(
        [var_series.rename("var"), realized_series.rename("pnl")], axis=1
    ).dropna()

    # Breach = realized loss exceeds VaR
    aligned["breach"] = aligned["pnl"] < -aligned["var"]
    n_total   = len(aligned)
    n_breach  = int(aligned["breach"].sum())
    p_actual  = n_breach / n_total if n_total > 0 else 0
    p_expect  = 1 - confidence

    # Kupiec LR statistic
    if n_breach in (0, n_total):
        lr_stat = 0.0
    else:
        lr_stat = -2 * (
            n_breach * np.log(p_expect / p_actual)
            + (n_total - n_breach) * np.log((1 - p_expect) / (1 - p_actual))
        )

    p_value  = float(1 - chi2.cdf(lr_stat, df=1))
    passed   = p_value > 0.05  # fail to reject H0 ⇒ model is accurate

    return {
        "n_total":      n_total,
        "n_breach":     n_breach,
        "breach_rate":  round(p_actual * 100, 2),
        "expected_rate": round(p_expect * 100, 2),
        "lr_statistic": round(lr_stat, 4),
        "p_value":      round(p_value, 4),
        "passed":       passed,
        "breaches":     aligned[aligned["breach"]].index.tolist(),
        "aligned_df":   aligned,
    }


def run_historical_backtest(returns_df, prices: dict,
                            confidence: float = 0.95,
                            lookback: int = 60) -> dict:
    """
    Rolls a VaR estimate over the last `lookback` trading days and checks
    if actual next-day returns exceeded it.
    """
    from config.settings import PORTFOLIO, MC_SIMULATIONS, RISK_FREE_RATE

    tickers = [t for t in returns_df.columns if t in PORTFOLIO]
    weights_nav = np.array([prices.get(t, 0) * PORTFOLIO[t] for t in tickers], dtype=float)
    total_nav   = weights_nav.sum()
    if total_nav == 0 or len(returns_df) < lookback + 2:
        return {}

    weights = weights_nav / total_nav
    port_returns = returns_df[tickers].values @ weights  # daily portfolio returns

    var_estimates, realized_pnl, dates = [], [], []

    for i in range(lookback, len(port_returns) - 1):
        window  = port_returns[i - lookback: i]
        var_est = float(-np.percentile(window, (1 - confidence) * 100)) * total_nav
        actual  = float(port_returns[i + 1]) * total_nav   # next-day realized
        var_estimates.append(var_est)
        realized_pnl.append(actual)
        dates.append(returns_df.index[i + 1])

    var_s = pd.Series(var_estimates, index=dates)
    pnl_s = pd.Series(realized_pnl,  index=dates)

    result = kupiec_test(var_s, pnl_s, confidence)
    result["var_series"] = var_s
    result["pnl_series"] = pnl_s
    return result
