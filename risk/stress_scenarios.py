from config.settings import PORTFOLIO

TECH       = ["AAPL", "MSFT", "GOOGL", "NVDA", "META", "AMZN", "TSLA", "AMD"]
FINANCIALS = ["JPM", "GS", "BAC", "BLK", "V", "MA"]
HEALTHCARE = ["JNJ", "UNH", "PFE", "ABBV"]
ENERGY     = ["XOM", "CVX", "NEE"]
CONSUMER   = ["COST", "WMT", "NKE"]
INDUSTRIAL = ["CAT", "LMT", "BA"]
ETFS       = ["SPY", "QQQ", "IWM", "GLD", "TLT"]

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

    "Flash Crash -10%":  {t: -0.10 for t in PORTFOLIO},

    "Bull Run +10%":     {t: +0.10 for t in PORTFOLIO},

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

def run_stress_tests(prices: dict) -> dict:
    results = {}
    for scenario, shocks in SCENARIOS.items():
        pnl = sum(
            prices.get(t, 0) * PORTFOLIO[t] * shocks.get(t, 0)
            for t in PORTFOLIO
        )
        results[scenario] = round(pnl, 2)
    return results