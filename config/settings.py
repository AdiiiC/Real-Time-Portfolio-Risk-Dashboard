import os
from dotenv import load_dotenv

load_dotenv()

# Kafka
KAFKA_BROKER  = os.getenv("KAFKA_BROKER",  "localhost:9092")
TOPIC_PRICES  = os.getenv("TOPIC_PRICES",  "live_prices")
TOPIC_RISK    = os.getenv("TOPIC_RISK",    "risk_metrics")

# Alpaca
ALPACA_API_KEY    = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

USE_LIVE_PORTFOLIO = False

PORTFOLIO = {
    "AAPL": 100, "MSFT": 80, "GOOGL": 50, "NVDA": 60,
    "META": 40,  "AMZN": 30, "TSLA": 60,  "JPM":  70,
    "SPY":  200, "QQQ":  100,
}

RISK_FREE_RATE   = 0.05
MC_SIMULATIONS   = 10_000
HOLDING_PERIOD   = 1
CONFIDENCE_LEVEL = 0.95
LOOKBACK_DAYS    = 252