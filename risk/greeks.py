"""
Options Greeks — Dividend-adjusted Black-Scholes-Merton model.

Computes Delta, Gamma, Theta, Vega, and Rho for option chains.
Supports expiry selection, Greeks ladder (time decay progression),
and dividend yield adjustment for US equities.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import date
from config.settings import RISK_FREE_RATE


def _get_dividend_yield(ticker: str) -> float:
    """Fetch the trailing annual dividend yield from yfinance (as decimal, e.g. 0.012)."""
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        return float(info.get("dividendYield") or 0.0)
    except Exception:
        return 0.0


def black_scholes_greeks(S: float, K: float, T: float, r: float, sigma: float,
                          option_type: str = "call", q: float = 0.0) -> dict:
    """
    Dividend-adjusted Black-Scholes-Merton Greeks.

    Args:
        S:           Spot price
        K:           Strike price
        T:           Time to expiry in years
        r:           Risk-free rate (annualised)
        sigma:       Implied volatility (annualised)
        option_type: 'call' or 'put'
        q:           Continuous dividend yield (annualised)

    Returns:
        delta:  Sensitivity of option price to $1 move in underlying
        gamma:  Rate of delta change per $1 move in underlying
        theta:  Daily time decay ($ per day, typically negative for long options)
        vega:   P&L per 1% rise in implied volatility
        rho:    P&L per 1% rise in risk-free rate (divided by 100)
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return {}

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    gamma = float(np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T)))
    vega  = float(S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100)

    if option_type == "call":
        delta = float(np.exp(-q * T) * norm.cdf(d1))
        theta = float((
            -(S * np.exp(-q * T) * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
            - r * K * np.exp(-r * T) * norm.cdf(d2)
            + q * S * np.exp(-q * T) * norm.cdf(d1)
        ) / 365)
        rho   = float(K * T * np.exp(-r * T) * norm.cdf(d2)  / 100)
    else:
        delta = float(np.exp(-q * T) * (norm.cdf(d1) - 1))
        theta = float((
            -(S * np.exp(-q * T) * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
            + r * K * np.exp(-r * T) * norm.cdf(-d2)
            - q * S * np.exp(-q * T) * norm.cdf(-d1)
        ) / 365)
        rho   = float(-K * T * np.exp(-r * T) * norm.cdf(-d2) / 100)

    return {
        "delta": round(delta, 4),
        "gamma": round(gamma, 6),
        "theta": round(theta, 4),
        "vega":  round(vega,  4),
        "rho":   round(rho,   4),
    }


def fetch_available_expiries(ticker: str) -> list:
    """Return list of available option expiry date strings for a ticker (up to 10)."""
    try:
        import yfinance as yf
        return list(yf.Ticker(ticker).options)[:10]
    except Exception:
        return []


def fetch_option_chain_greeks(ticker: str, spot_price: float,
                               n_strikes: int = 7,
                               expiry: str = None) -> pd.DataFrame:
    """
    Fetch option chain for a ticker and compute dividend-adjusted Greeks.

    Args:
        ticker:      Stock symbol
        spot_price:  Current market price
        n_strikes:   Number of strikes around ATM to include
        expiry:      Specific expiry date string (uses nearest if None)

    Returns DataFrame with columns:
        Type, Strike, Expiry, IV, Bid, Ask, OI, delta, gamma, theta, vega, rho
    """
    try:
        import yfinance as yf
        tk       = yf.Ticker(ticker)
        expiries = tk.options
        if not expiries:
            return pd.DataFrame()

        exp_use = expiry if (expiry and expiry in expiries) else expiries[0]
        chain   = tk.option_chain(exp_use)
        today   = date.today()
        exp_dt  = date.fromisoformat(exp_use)
        T       = max((exp_dt - today).days / 365, 1 / 365)
        r       = RISK_FREE_RATE
        q       = _get_dividend_yield(ticker)

        rows = []
        for opt_type, df in [("call", chain.calls), ("put", chain.puts)]:
            df = df.copy()
            df["dist"] = (df["strike"] - spot_price).abs()
            df = df.nsmallest(n_strikes, "dist")
            for _, row in df.iterrows():
                iv = float(row.get("impliedVolatility", 0.3) or 0.3)
                g  = black_scholes_greeks(
                    S=spot_price, K=float(row["strike"]),
                    T=T, r=r, sigma=iv, option_type=opt_type, q=q
                )
                rows.append({
                    "Type":   opt_type.upper(),
                    "Strike": float(row["strike"]),
                    "Expiry": exp_use,
                    "IV":     round(iv * 100, 1),
                    "Bid":    float(row.get("bid", 0) or 0),
                    "Ask":    float(row.get("ask", 0) or 0),
                    "OI":     int(row.get("openInterest", 0) or 0),
                    **g,
                })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


def fetch_greeks_ladder(ticker: str, spot_price: float,
                         n_expiries: int = 5) -> pd.DataFrame:
    """
    Greeks Ladder — ATM option Greeks across multiple expiry dates.

    Shows how Delta, Gamma, Theta, Vega, and Rho evolve as time to expiry changes.
    This is essential for understanding time decay (Theta) progression and
    how sensitivity (Delta, Gamma) shifts as options approach expiration.

    Returns DataFrame with one row per (expiry, option type).
    """
    try:
        import yfinance as yf
        tk       = yf.Ticker(ticker)
        expiries = list(tk.options)[:n_expiries]
        r        = RISK_FREE_RATE
        q        = _get_dividend_yield(ticker)
        rows     = []
        today    = date.today()

        for exp in expiries:
            exp_dt = date.fromisoformat(exp)
            T      = max((exp_dt - today).days / 365, 1 / 365)
            chain  = tk.option_chain(exp)
            for opt_label, df in [("CALL", chain.calls), ("PUT", chain.puts)]:
                df = df.copy()
                df["dist"] = (df["strike"] - spot_price).abs()
                atm = df.nsmallest(1, "dist")
                if atm.empty:
                    continue
                atm_row = atm.iloc[0]
                iv  = float(atm_row.get("impliedVolatility", 0.3) or 0.3)
                g   = black_scholes_greeks(
                    S=spot_price, K=float(atm_row["strike"]),
                    T=T, r=r, sigma=iv,
                    option_type=opt_label.lower(), q=q
                )
                rows.append({
                    "Expiry":  exp,
                    "DTE":     (exp_dt - today).days,
                    "Type":    opt_label,
                    "Strike":  float(atm_row["strike"]),
                    "IV (%)":  round(iv * 100, 1),
                    **g,
                })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()
