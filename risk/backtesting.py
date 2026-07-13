"""
VaR Backtesting Suite:
  - Kupiec Proportion of Failures (POF) test
  - Christoffersen Independence test (are breaches clustered?)
  - Basel II/III Traffic Light classification
  - Multi-confidence level comparison (90%, 95%, 99%)

Lookback extended to 250 trading days (~1 calendar year) for statistical validity.
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2


def kupiec_test(var_series: pd.Series, realized_series: pd.Series,
                confidence: float = 0.95) -> dict:
    """
    Kupiec Proportion of Failures (POF) test.
    Tests whether the observed breach rate matches the model's expected breach rate.

    H0: Model is correctly calibrated (observed rate = expected rate).
    Pass (p > 0.05): No statistically significant evidence of model failure.
    """
    aligned = pd.concat(
        [var_series.rename("var"), realized_series.rename("pnl")], axis=1
    ).dropna()

    aligned["breach"] = aligned["pnl"] < -aligned["var"]
    n_total  = len(aligned)
    n_breach = int(aligned["breach"].sum())
    p_actual = n_breach / n_total if n_total > 0 else 0
    p_expect = 1 - confidence

    if n_breach in (0, n_total):
        lr_stat = 0.0
    else:
        lr_stat = -2 * (
            n_breach       * np.log(p_expect / p_actual)
            + (n_total - n_breach) * np.log((1 - p_expect) / (1 - p_actual))
        )

    p_value = float(1 - chi2.cdf(lr_stat, df=1))
    passed  = p_value > 0.05

    return {
        "n_total":       n_total,
        "n_breach":      n_breach,
        "breach_rate":   round(p_actual * 100, 2),
        "expected_rate": round(p_expect * 100, 2),
        "lr_statistic":  round(lr_stat, 4),
        "p_value":       round(p_value, 4),
        "passed":        passed,
        "breaches":      aligned[aligned["breach"]].index.tolist(),
        "aligned_df":    aligned,
    }


def christoffersen_independence_test(breach_series: pd.Series) -> dict:
    """
    Christoffersen Independence Test.
    Tests whether VaR breaches occur randomly or cluster together.
    Clustered breaches (e.g. all during one crash week) indicate the model
    fails to capture volatility regimes.

    H0: Breaches are independent (good).
    Pass (p > 0.05): No evidence of clustering.
    """
    b   = breach_series.astype(int).values
    n00 = sum(1 for i in range(len(b) - 1) if b[i] == 0 and b[i + 1] == 0)
    n01 = sum(1 for i in range(len(b) - 1) if b[i] == 0 and b[i + 1] == 1)
    n10 = sum(1 for i in range(len(b) - 1) if b[i] == 1 and b[i + 1] == 0)
    n11 = sum(1 for i in range(len(b) - 1) if b[i] == 1 and b[i + 1] == 1)

    pi01 = n01 / (n00 + n01 + 1e-12)
    pi11 = n11 / (n10 + n11 + 1e-12)
    pi   = (n01 + n11) / (n00 + n01 + n10 + n11 + 1e-12)

    if pi in (0.0, 1.0) or (pi01 == pi11):
        return {"lr_ind": 0.0, "p_ind": 1.0, "independent": True}

    try:
        lr_ind = -2 * (
            (n00 + n10) * np.log(max(1 - pi,   1e-12))
            + (n01 + n11) * np.log(max(pi,      1e-12))
            - n00 * np.log(max(1 - pi01, 1e-12))
            - n01 * np.log(max(pi01,     1e-12))
            - n10 * np.log(max(1 - pi11, 1e-12))
            - n11 * np.log(max(pi11,     1e-12))
        )
    except Exception:
        lr_ind = 0.0

    p_ind = float(1 - chi2.cdf(max(lr_ind, 0), df=1))
    return {
        "lr_ind":      round(lr_ind, 4),
        "p_ind":       round(p_ind, 4),
        "independent": p_ind > 0.05,
    }


def basel_traffic_light(n_breach: int, n_total: int, confidence: float = 0.95) -> str:
    """
    Basel II/III Traffic Light for model validation.
    Compares actual breach count to expected count at the given confidence level.

    Green  : breaches ≤ 1.5× expected  — model performing well
    Yellow : breaches 1.5–3× expected  — model under scrutiny
    Red    : breaches > 3× expected    — model requires recalibration
    """
    expected = (1 - confidence) * n_total
    ratio    = n_breach / (expected + 1e-12)
    if ratio <= 1.5:
        return "green"
    elif ratio <= 3.0:
        return "yellow"
    else:
        return "red"


def run_historical_backtest(returns_df: pd.DataFrame, prices: dict,
                             confidence: float = 0.95,
                             lookback: int = 250) -> dict:
    """
    Rolling VaR backtest over the last `lookback` trading days.
    250 days (~1 year) is the Basel standard for adequate statistical power.

    Returns Kupiec POF results, Christoffersen independence test,
    Basel traffic light, and full aligned P&L/VaR series.
    """
    from config.settings import PORTFOLIO

    tickers     = [t for t in returns_df.columns if t in PORTFOLIO]
    weights_nav = np.array([prices.get(t, 0) * PORTFOLIO[t] for t in tickers], dtype=float)
    total_nav   = weights_nav.sum()
    if total_nav == 0 or len(returns_df) < lookback + 2:
        return {}

    weights      = weights_nav / total_nav
    port_returns = returns_df[tickers].values @ weights

    var_estimates, realized_pnl, dates = [], [], []
    for i in range(lookback, len(port_returns) - 1):
        window  = port_returns[i - lookback: i]
        var_est = float(-np.percentile(window, (1 - confidence) * 100)) * total_nav
        actual  = float(port_returns[i + 1]) * total_nav
        var_estimates.append(var_est)
        realized_pnl.append(actual)
        dates.append(returns_df.index[i + 1])

    var_s  = pd.Series(var_estimates, index=dates)
    pnl_s  = pd.Series(realized_pnl,  index=dates)
    result = kupiec_test(var_s, pnl_s, confidence)

    # Christoffersen independence test
    result["christoffersen"] = christoffersen_independence_test(
        result["aligned_df"]["breach"]
    )

    # Basel traffic light
    result["traffic_light"] = basel_traffic_light(
        result["n_breach"], result["n_total"], confidence
    )

    result["var_series"] = var_s
    result["pnl_series"] = pnl_s
    return result


def run_multi_confidence_backtest(returns_df: pd.DataFrame, prices: dict,
                                   confidences: tuple = (0.90, 0.95, 0.99),
                                   lookback: int = 250) -> dict:
    """
    Run backtest at multiple confidence levels simultaneously.
    Useful for comparing model performance across risk tolerance bands.
    Returns {confidence: backtest_result_dict}.
    """
    return {
        conf: run_historical_backtest(returns_df, prices, conf, lookback)
        for conf in confidences
    }
