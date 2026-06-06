import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import date
from config.settings import RISK_FREE_RATE


def black_scholes_greeks(S, K, T, r, sigma, option_type="call"):
    """S=spot, K=strike, T=years to expiry, r=risk-free rate, sigma=implied vol"""
    if T <= 0:
        return {}
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        delta = norm.cdf(d1)
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        delta = norm.cdf(d1) - 1
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365

    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega  = S * norm.pdf(d1) * np.sqrt(T) / 100

    return {
        "delta": round(delta, 4),
        "gamma": round(gamma, 6),
        "theta": round(theta, 4),
        "vega":  round(vega,  4),
    }


def fetch_option_chain_greeks(ticker: str, spot_price: float,
                               n_strikes: int = 5) -> pd.DataFrame:
    """
    Fetch the nearest expiry option chain for `ticker` via yfinance,
    compute Black-Scholes Greeks for each contract, return as DataFrame.
    Falls back to empty DataFrame on any error.
    """
    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)
        expiries = tk.options
        if not expiries:
            return pd.DataFrame()
        expiry = expiries[0]   # nearest expiry
        chain  = tk.option_chain(expiry)
        today  = date.today()
        exp_dt = date.fromisoformat(expiry)
        T      = max((exp_dt - today).days / 365, 1/365)
        r      = RISK_FREE_RATE

        rows = []
        for opt_type, df in [("call", chain.calls), ("put", chain.puts)]:
            # Keep n_strikes closest to ATM
            df = df.copy()
            df["dist"] = (df["strike"] - spot_price).abs()
            df = df.nsmallest(n_strikes, "dist")
            for _, row in df.iterrows():
                iv = float(row.get("impliedVolatility", 0.3) or 0.3)
                g  = black_scholes_greeks(
                    S=spot_price, K=float(row["strike"]),
                    T=T, r=r, sigma=iv, option_type=opt_type
                )
                rows.append({
                    "Type":   opt_type.upper(),
                    "Strike": row["strike"],
                    "Expiry": expiry,
                    "IV":     round(iv * 100, 1),
                    "Bid":    row.get("bid", 0),
                    "Ask":    row.get("ask", 0),
                    "OI":     row.get("openInterest", 0),
                    **g,
                })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()
