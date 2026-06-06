from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass
from config.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY, PORTFOLIO

def validate_portfolio_assets() -> dict:
    """Check every ticker in PORTFOLIO is tradable on Alpaca."""
    client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
    results = {}
    for symbol in PORTFOLIO:
        try:
            asset = client.get_asset(symbol)
            results[symbol] = {
                "tradable":   asset.tradable,
                "shortable":  asset.shortable,
                "marginable": asset.marginable,
                "status":     str(asset.status),
            }
            if not asset.tradable:
                print(f"WARNING: {symbol} is NOT tradable on Alpaca")
        except Exception as e:
            results[symbol] = {"tradable": False, "error": str(e)}
            print(f"ERROR: Could not fetch asset info for {symbol}: {e}")
    tradable_count = sum(1 for v in results.values() if v.get("tradable"))
    print(f"{tradable_count}/{len(PORTFOLIO)} portfolio assets are tradable")
    return results

def get_all_us_equities():
    """Fetch full list of tradable US equities from Alpaca."""
    client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
    search_params = GetAssetsRequest(asset_class=AssetClass.US_EQUITY)
    return client.get_all_assets(search_params)