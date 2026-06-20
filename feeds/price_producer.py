import json, time, random
from confluent_kafka import Producer
from alpaca.data.live import StockDataStream
from config.settings import (
    KAFKA_BROKER, TOPIC_PRICES, PORTFOLIO,
    ALPACA_API_KEY, ALPACA_SECRET_KEY
)

producer = Producer({"bootstrap.servers": KAFKA_BROKER})
latest_prices = {}

def _publish():
    payload = json.dumps({"timestamp": time.time(), "prices": dict(latest_prices)})
    producer.produce(TOPIC_PRICES, value=payload)
    producer.flush()

async def on_trade(trade):
    latest_prices[trade.symbol] = float(trade.price)
    if set(latest_prices.keys()) >= set(PORTFOLIO.keys()):
        print(f"[LIVE] {latest_prices}")
        _publish()

async def on_bar(bar):
    latest_prices[bar.symbol] = float(bar.close)
    if set(latest_prices.keys()) >= set(PORTFOLIO.keys()):
        _publish()

def simulate_prices():
    """Fetch last close prices via yfinance and simulate ticks (market closed fallback)."""
    import yfinance as yf
    tickers = list(PORTFOLIO.keys())
    print(f"Market closed — fetching last close prices for {len(tickers)} tickers via yfinance...")
    data = yf.download(tickers, period="5d", auto_adjust=True, progress=False)["Close"]
    base = data.dropna().iloc[-1].to_dict()
    print(f"[SIM BASE] { {k: round(v,2) for k,v in base.items()} }")
    prices = dict(base)
    while True:
        for t in tickers:
            prices[t] = round(prices[t] * (1 + random.gauss(0, 0.0003)), 4)
        payload = json.dumps({"timestamp": time.time(), "prices": prices})
        producer.produce(TOPIC_PRICES, value=payload)
        producer.flush()
        print(f"[SIM] { {k: round(v,2) for k,v in prices.items()} }")
        time.sleep(3)

def stream_prices():
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        raise EnvironmentError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env")
    wss = StockDataStream(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    tickers = list(PORTFOLIO.keys())
    wss.subscribe_trades(on_trade, *tickers)
    print(f"Streaming live trades for {len(tickers)} tickers")
    wss.run()

if __name__ == "__main__":
    import datetime, pytz
    now_et = datetime.datetime.now(pytz.timezone("US/Eastern"))
    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    is_market_hours = (
        now_et.weekday() < 5 and
        market_open <= now_et <= market_close
    )
    if is_market_hours:
        stream_prices()
    else:
        print(f"Market closed ({now_et.strftime('%H:%M ET')}). Running simulation mode.")
        simulate_prices()