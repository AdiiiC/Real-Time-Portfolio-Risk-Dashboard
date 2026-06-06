from alpaca.trading.client import TradingClient
from alpaca.trading.stream import TradingStream
from alpaca.trading.requests import (
    MarketOrderRequest, LimitOrderRequest,
    TakeProfitRequest, StopLossRequest,
    TrailingStopOrderRequest, GetOrdersRequest
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, QueryOrderStatus
from config.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY

def _client():
    return TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)

# --- Place Orders ---

def place_market_order(symbol: str, qty: float, side: str = "buy") -> dict:
    order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
    req = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=order_side,
        time_in_force=TimeInForce.DAY,
    )
    order = _client().submit_order(order_data=req)
    print(f"Market {side.upper()} {qty} {symbol} submitted: #{order.id}")
    return {"id": str(order.id), "symbol": symbol, "qty": qty, "side": side}

def place_limit_order(symbol: str, qty: float, limit_price: float,
                      side: str = "buy") -> dict:
    order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
    req = LimitOrderRequest(
        symbol=symbol,
        qty=qty,
        limit_price=limit_price,
        side=order_side,
        time_in_force=TimeInForce.DAY,
    )
    order = _client().submit_order(order_data=req)
    print(f"Limit {side.upper()} {qty} {symbol} @ ${limit_price}: #{order.id}")
    return {"id": str(order.id), "symbol": symbol, "qty": qty,
            "limit_price": limit_price, "side": side}

def place_bracket_order(symbol: str, qty: float, take_profit: float,
                        stop_loss: float) -> dict:
    """Buy with automatic take-profit and stop-loss."""
    req = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
        order_class=OrderClass.BRACKET,
        take_profit=TakeProfitRequest(limit_price=take_profit),
        stop_loss=StopLossRequest(stop_price=stop_loss),
    )
    order = _client().submit_order(order_data=req)
    print(f"Bracket BUY {qty} {symbol} | TP: ${take_profit} SL: ${stop_loss}")
    return {"id": str(order.id), "symbol": symbol, "type": "bracket"}

def place_trailing_stop(symbol: str, qty: float,
                        trail_percent: float = 1.0) -> dict:
    """Sell with trailing stop (hwm * trail_percent%)."""
    req = TrailingStopOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.GTC,
        trail_percent=trail_percent,
    )
    order = _client().submit_order(order_data=req)
    print(f"Trailing stop SELL {qty} {symbol} @ {trail_percent}%: #{order.id}")
    return {"id": str(order.id), "symbol": symbol, "trail_percent": trail_percent}

# --- Query Orders ---

def get_recent_orders(limit: int = 20, status: str = "all") -> list:
    status_map = {
        "open":   QueryOrderStatus.OPEN,
        "closed": QueryOrderStatus.CLOSED,
        "all":    QueryOrderStatus.ALL,
    }
    req = GetOrdersRequest(
        status=status_map.get(status, QueryOrderStatus.ALL),
        limit=limit,
        nested=True,
    )
    orders = _client().get_orders(filter=req)
    return [
        {
            "id":          str(o.id),
            "symbol":      o.symbol,
            "qty":         float(o.qty or 0),
            "side":        str(o.side),
            "type":        str(o.order_type),
            "status":      str(o.status),
            "filled_qty":  float(o.filled_qty or 0),
            "filled_price":float(o.filled_avg_price or 0),
            "submitted_at":str(o.submitted_at),
        }
        for o in orders
    ]

def cancel_all_orders():
    _client().cancel_orders()
    print("All open orders cancelled.")

# --- Real-time Order Stream ---

def stream_order_updates(on_update_callback):
    """Stream live order status updates via WebSocket."""
    stream = TradingStream(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)

    async def on_msg(data):
        print(f"Order update: {data.event} — {data.order.symbol}")
        on_update_callback(data)

    stream.subscribe_trade_updates(on_msg)
    print("Listening for live order updates...")
    stream.run()