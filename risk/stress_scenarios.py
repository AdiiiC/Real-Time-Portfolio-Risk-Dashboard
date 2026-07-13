"""
Stress Testing — Pre-defined macro shock scenarios and custom scenario builder.

Each scenario applies sector-specific shocks to portfolio positions and computes
the resulting dollar P&L impact. Scenarios are based on historical market events
and analyst-defined shock assumptions.
"""
from config.settings import PORTFOLIO

TECH       = ["AAPL", "MSFT", "GOOGL", "NVDA", "META", "AMZN", "TSLA", "AMD"]
FINANCIALS = ["JPM", "GS", "BAC", "BLK", "V", "MA"]
HEALTHCARE = ["JNJ", "UNH", "PFE", "ABBV"]
ENERGY     = ["XOM", "CVX", "NEE"]
CONSUMER   = ["COST", "WMT", "NKE"]
INDUSTRIAL = ["CAT", "LMT", "BA"]
ETFS       = ["SPY", "QQQ", "IWM", "GLD", "TLT"]

# Built-in scenarios: {name: {ticker: shock_fraction}}
SCENARIOS = {
    "Market Crash -20%": {t: -0.20 for t in PORTFOLIO},

    "Tech Selloff -15%": {
        **{t: -0.15 for t in TECH},
        **{t: -0.05 for t in FINANCIALS},
        **{t: -0.02 for t in HEALTHCARE},
        **{t: -0.03 for t in ENERGY + CONSUMER + INDUSTRIAL},
        **{t: -0.10 for t in ["SPY", "QQQ"]},
        **{t: -0.03 for t in ["IWM", "GLD", "TLT"]},
    },

    "Rate Spike +100bps": {
        **{t: -0.08 for t in TECH},
        **{t: +0.05 for t in FINANCIALS},
        **{t: -0.05 for t in HEALTHCARE},
        **{t: +0.03 for t in ENERGY},
        **{t: -0.04 for t in CONSUMER + INDUSTRIAL},
        **{t: -0.12 for t in ["TLT"]},
        **{t: -0.06 for t in ["SPY", "QQQ", "IWM"]},
        **{t: -0.02 for t in ["GLD"]},
    },

    "Financial Crisis -30%": {
        **{t: -0.30 for t in FINANCIALS},
        **{t: -0.20 for t in TECH + CONSUMER + INDUSTRIAL},
        **{t: -0.10 for t in HEALTHCARE},
        **{t: -0.15 for t in ENERGY},
        **{t: -0.25 for t in ["SPY", "QQQ", "IWM"]},
        **{t: +0.15 for t in ["GLD", "TLT"]},
    },

    "Energy Shock +50%": {
        **{t: +0.30 for t in ENERGY},
        **{t: -0.08 for t in TECH + CONSUMER},
        **{t: -0.05 for t in FINANCIALS + HEALTHCARE + INDUSTRIAL},
        **{t: -0.06 for t in ["SPY", "QQQ", "IWM"]},
        **{t: +0.10 for t in ["GLD"]},
        **{t: -0.03 for t in ["TLT"]},
    },

    "Flash Crash -10%":     {t: -0.10 for t in PORTFOLIO},
    "Bull Run +10%":        {t: +0.10 for t in PORTFOLIO},

    "AI Bubble Burst -25%": {
        **{t: -0.25 for t in ["NVDA", "AMD", "MSFT", "GOOGL", "META"]},
        **{t: -0.15 for t in ["AAPL", "AMZN", "TSLA"]},
        **{t: -0.05 for t in FINANCIALS + HEALTHCARE + ENERGY},
        **{t: -0.03 for t in CONSUMER + INDUSTRIAL + ["IWM"]},
        **{t: -0.18 for t in ["QQQ"]},
        **{t: -0.08 for t in ["SPY"]},
        **{t: +0.05 for t in ["GLD", "TLT"]},
    },
}


def run_stress_tests(prices: dict,
                     portfolio: dict = None,
                     custom_scenarios: dict = None) -> dict:
    """
    Apply all built-in scenarios (plus any custom ones) to the portfolio.

    Args:
        prices:           Current market prices {ticker: price}
        portfolio:        {ticker: shares} — defaults to settings.PORTFOLIO
        custom_scenarios: Optional {scenario_name: {ticker: shock_fraction}} to add

    Returns:
        {scenario_name: dollar_pnl}
    """
    if portfolio is None:
        portfolio = PORTFOLIO

    scenarios = dict(SCENARIOS)
    if custom_scenarios:
        scenarios.update(custom_scenarios)

    results = {}
    for scenario, shocks in scenarios.items():
        pnl = sum(
            prices.get(t, 0) * portfolio.get(t, 0) * shocks.get(t, 0)
            for t in portfolio
        )
        results[scenario] = round(pnl, 2)
    return results


def run_sensitivity_analysis(prices: dict, ticker: str,
                              portfolio: dict = None,
                              shock_range: tuple = (-0.30, 0.30),
                              steps: int = 13) -> dict:
    """
    Compute portfolio P&L as a single ticker's shock varies from min to max.
    Useful for identifying breakeven points and non-linear risk exposures.

    Returns {shock_pct: dollar_pnl}
    """
    import numpy as np
    if portfolio is None:
        portfolio = PORTFOLIO

    shocks  = np.linspace(shock_range[0], shock_range[1], steps)
    results = {}
    for s in shocks:
        pnl = prices.get(ticker, 0) * portfolio.get(ticker, 0) * s
        results[round(float(s) * 100, 1)] = round(pnl, 2)
    return results
