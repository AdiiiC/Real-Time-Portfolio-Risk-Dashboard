import yfinance as yf
from config.settings import (
    PORTFOLIO, LOOKBACK_DAYS, USE_LIVE_PORTFOLIO,
    ALPACA_API_KEY, ALPACA_SECRET_KEY
)

def _get_trading_client():
    from alpaca.trading.client import TradingClient
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        raise EnvironmentError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env")
    return TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)

def get_account_info() -> dict:
    account = _get_trading_client().get_account()
    balance_change = float(account.equity) - float(account.last_equity)
    if account.trading_blocked:
        print("WARNING: Account is currently restricted from trading.")
    # unrealized_pl removed in newer alpaca-py; approximate from equity - cash - accrued_fees
    unrealized_pnl = float(account.equity) - float(account.cash) - float(account.accrued_fees)
    return {
        "equity":             float(account.equity),
        "cash":               float(account.cash),
        "buying_power":       float(account.buying_power),
        "last_equity":        float(account.last_equity),
        "daily_pnl":          round(balance_change, 2),
        "unrealized_pnl":     round(unrealized_pnl, 2),
        "trading_blocked":    account.trading_blocked,
        "account_blocked":    account.account_blocked,
        "pattern_day_trader": account.pattern_day_trader,
    }

def get_live_portfolio() -> dict:
    """Pull open positions from Alpaca — returns {symbol: qty}."""
    positions = _get_trading_client().get_all_positions()
    return {p.symbol: int(float(p.qty)) for p in positions}

def get_live_positions_detail() -> list:
    """Full position details: entry price, unrealized P&L, market value, cost basis."""
    positions = _get_trading_client().get_all_positions()
    result = []
    for p in positions:
        result.append({
            "symbol":          p.symbol,
            "qty":             float(p.qty),
            "side":            str(p.side),
            "entry_price":     float(p.avg_entry_price),
            "current_price":   float(p.current_price),
            "market_value":    float(p.market_value),
            "cost_basis":      float(p.cost_basis),
            "unrealized_pl":   float(p.unrealized_pl),
            "unrealized_plpc": float(p.unrealized_plpc) * 100,  # as %
            "intraday_pl":     float(p.unrealized_intraday_pl),
            "intraday_plpc":   float(p.unrealized_intraday_plpc) * 100,
        })
    return result

def get_single_position(symbol: str) -> dict:
    p = _get_trading_client().get_open_position(symbol)
    return {
        "symbol":        p.symbol,
        "qty":           float(p.qty),
        "entry_price":   float(p.avg_entry_price),
        "current_price": float(p.current_price),
        "market_value":  float(p.market_value),
        "unrealized_pl": float(p.unrealized_pl),
    }

def get_historical_returns(portfolio: dict = None):
    if portfolio is None:
        portfolio = get_live_portfolio() if USE_LIVE_PORTFOLIO else PORTFOLIO
    tickers = list(portfolio.keys())
    data = yf.download(tickers, period=f"{LOOKBACK_DAYS}d", progress=False)["Close"]
    return data.pct_change().dropna()

def mark_to_market(prices: dict, portfolio: dict = None) -> dict:
    if portfolio is None:
        portfolio = get_live_portfolio() if USE_LIVE_PORTFOLIO else PORTFOLIO
    positions = {}
    total = 0.0
    for ticker, shares in portfolio.items():
        price = prices.get(ticker, 0)
        value = price * shares
        positions[ticker] = {"price": price, "shares": shares, "value": value}
        total += value
    positions["total_nav"] = total
    return positions