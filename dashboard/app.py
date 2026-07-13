"""
Real-Time Portfolio Risk Dashboard
===================================
Designed for financial analysts — every metric is explained in plain English.
Runs on live Kafka price feed with Alpaca WebSocket; falls back to simulation
when markets are closed.

Tabs:
  1. Overview           — Portfolio snapshot, live P&L, stress tests
  2. Risk Deep Dive     — VaR comparison (3 methods), component & incremental VaR
  3. Performance        — Returns, drawdown, rolling Sharpe/Sortino, Calmar
  4. Backtesting        — Kupiec + Christoffersen + Basel Traffic Light
  5. Options Greeks     — Full option chain with Rho, expiry picker, Greeks ladder
  6. Multi-Portfolio    — Side-by-side risk comparison with CSV export
  7. Scenario Builder   — Build and test custom shock scenarios in real time
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import json
import time
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from confluent_kafka import Consumer
import streamlit_authenticator as stauth

from portfolio.portfolio import (
    get_historical_returns, mark_to_market,
    get_account_info, get_live_positions_detail,
)
from risk.var_engine import compute_var_monte_carlo
from risk.stress_scenarios import run_stress_tests, run_sensitivity_analysis
from risk.greeks import (
    fetch_option_chain_greeks, fetch_available_expiries, fetch_greeks_ladder,
)
from risk.backtesting import run_historical_backtest, run_multi_confidence_backtest
from trading.orders import get_recent_orders
from db.persistence import save_prices, save_var_snapshot, load_var_history
from config.settings import (
    KAFKA_BROKER, TOPIC_PRICES, USE_LIVE_PORTFOLIO, PORTFOLIO,
)

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Portfolio Risk Dashboard",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — professional card styling ────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: #1e2130;
    border-radius: 10px;
    padding: 14px 18px;
    border-left: 4px solid #636EFA;
    margin-bottom: 8px;
}
.risk-green  { border-left-color: #00CC96; }
.risk-yellow { border-left-color: #FFA15A; }
.risk-red    { border-left-color: #EF553B; }
.section-header {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #888;
    margin: 16px 0 6px 0;
}
.help-box {
    background: #16213e;
    border-radius: 6px;
    padding: 10px 14px;
    font-size: 0.82rem;
    color: #aab;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# ── Authentication ─────────────────────────────────────────────────────────────
AUTH_PATH = Path(__file__).parent.parent / "config" / "auth.yaml"
with open(AUTH_PATH) as f:
    auth_cfg = yaml.safe_load(f)

authenticator = stauth.Authenticate(
    auth_cfg["credentials"],
    auth_cfg["cookie"]["name"],
    auth_cfg["cookie"]["key"],
    auth_cfg["cookie"]["expiry_days"],
)
authenticator.login()

if not st.session_state.get("authentication_status"):
    if st.session_state.get("authentication_status") is False:
        st.error("Incorrect username or password.")
    else:
        st.info("Please log in — default: **admin / admin123**")
    st.stop()

# ── Constants ──────────────────────────────────────────────────────────────────
SECTOR_MAP = {
    "AAPL": "Tech",     "MSFT": "Tech",    "GOOGL": "Tech",  "NVDA": "Tech",
    "META": "Tech",     "AMZN": "Tech",    "TSLA": "Tech",   "AMD": "Tech",
    "JPM":  "Finance",  "GS":   "Finance", "BAC":  "Finance","BLK": "Finance",
    "V":    "Finance",  "MA":   "Finance",
    "JNJ":  "Health",   "UNH":  "Health",  "PFE":  "Health", "ABBV": "Health",
    "XOM":  "Energy",   "CVX":  "Energy",  "NEE":  "Energy",
    "COST": "Consumer", "WMT":  "Consumer","NKE":  "Consumer",
    "CAT":  "Indust.",  "LMT":  "Indust.", "BA":   "Indust.",
    "SPY":  "ETF",      "QQQ":  "ETF",     "IWM":  "ETF",
    "GLD":  "ETF",      "TLT":  "ETF",
}
SECTOR_COLORS = {
    "Tech":     "#636EFA", "Finance": "#EF553B", "Health":   "#00CC96",
    "Energy":   "#AB63FA", "Consumer":"#FFA15A", "Indust.":  "#19D3F3",
    "ETF":      "#FF6692", "Other":   "#B6E880",
}
PORTFOLIOS = {
    "Main":       PORTFOLIO,
    "Tech Heavy": {t: v for t, v in PORTFOLIO.items() if SECTOR_MAP.get(t) in ("Tech", "ETF")},
    "Defensive":  {t: v for t, v in PORTFOLIO.items() if SECTOR_MAP.get(t) in ("Health", "Consumer", "ETF")},
}


def fmt_money(val: float) -> str:
    """Format dollar value in analyst-friendly abbreviated form."""
    if abs(val) >= 1_000_000:
        return f"${val / 1_000_000:.2f}M"
    if abs(val) >= 1_000:
        return f"${val / 1_000:.1f}K"
    return f"${val:,.0f}"


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Dashboard Controls")
    authenticator.logout("Logout", "sidebar")
    st.divider()

    st.markdown('<p class="section-header">Portfolio</p>', unsafe_allow_html=True)
    active_portfolio_name = st.selectbox(
        "Active Portfolio",
        list(PORTFOLIOS.keys()),
        help="Switch between pre-defined portfolio configurations. 'Main' is the full portfolio.",
    )
    active_portfolio = PORTFOLIOS[active_portfolio_name]

    st.markdown('<p class="section-header">Risk Parameters</p>', unsafe_allow_html=True)
    confidence = st.select_slider(
        "Confidence Level",
        [0.90, 0.95, 0.99],
        value=0.95,
        format_func=lambda x: f"{int(x * 100)}%",
        help=(
            "The probability that actual losses will NOT exceed the VaR figure. "
            "95% means: in 95 out of 100 days, losses should stay below the VaR threshold."
        ),
    )
    holding_period = st.slider(
        "Holding Period (days)",
        1, 10, 1,
        help=(
            "How many trading days you assume it takes to exit the position. "
            "VaR is scaled by √days — a 5-day VaR is √5 × 1-day VaR."
        ),
    )
    refresh_secs = st.slider(
        "Auto-Refresh (seconds)",
        3, 30, 5,
        help="How often the dashboard pulls new prices and recalculates risk metrics.",
    )
    alert_var = st.number_input(
        "VaR Alert Threshold ($)",
        min_value=0, value=5_000, step=500,
        help="A red banner will appear at the top of the dashboard when VaR exceeds this amount.",
    )

    st.divider()
    st.markdown('<p class="section-header">Display Options</p>', unsafe_allow_html=True)
    show_corr      = st.checkbox("Correlation Heatmap",    True)
    show_waterfall = st.checkbox("P&L Attribution",        True)
    show_backtest  = st.checkbox("VaR Backtesting",        True)
    show_greeks    = st.checkbox("Options Greeks",         True)
    rolling_window = st.slider(
        "Rolling Metrics Window (days)", 20, 90, 30,
        help="Window size used for rolling Sharpe, Sortino, and correlation charts.",
    )

    st.divider()
    st.markdown('<p class="section-header">Options</p>', unsafe_allow_html=True)
    greeks_ticker = st.selectbox(
        "Greeks Ticker",
        list(active_portfolio.keys()),
        help="Select the ticker to analyse options Greeks for.",
    )

    st.caption("Market closed → simulation mode  |  DB: risk_dashboard.db")

# ── Session State Init ─────────────────────────────────────────────────────────
for key, default in [
    ("last_prices", None), ("base_prices", None),
    ("equity_curve", []),  ("eq_timestamps", []),
    ("alert_shown", False), ("returns_df", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Data Helpers ───────────────────────────────────────────────────────────────
def get_latest_prices():
    c = Consumer({
        "bootstrap.servers": KAFKA_BROKER,
        "group.id": f"dash-{time.time()}",
        "auto.offset.reset": "earliest",
    })
    c.subscribe([TOPIC_PRICES])
    latest  = None
    deadline = time.time() + 5.0
    while time.time() < deadline:
        msg = c.poll(timeout=1.0)
        if msg and not msg.error():
            latest = json.loads(msg.value())["prices"]
    c.close()
    return latest


def ticker_html(prices: dict, base_prices: dict) -> str:
    items = []
    for t, p in sorted(prices.items()):
        base = base_prices.get(t, p)
        chg  = ((p - base) / base) * 100 if base else 0
        col  = "#00CC96" if chg >= 0 else "#EF553B"
        arr  = "▲" if chg >= 0 else "▼"
        items.append(
            f'<span style="margin:0 14px;font-size:13px;font-weight:600">'
            f'{t} <span style="color:{col}">${p:,.2f} {arr}{abs(chg):.2f}%</span></span>'
        )
    return (
        '<div style="background:#0e1117;padding:8px 12px;border-radius:6px;'
        'border:1px solid #262730;overflow-x:auto;white-space:nowrap;margin-bottom:10px">'
        + "".join(items) + "</div>"
    )


# ── Chart Builders ─────────────────────────────────────────────────────────────
def build_gauge(var: float, nav: float, conf: float) -> go.Figure:
    """Three-band VaR gauge — green <2%, yellow 2-5%, red 5-10% of NAV."""
    pct = min((var / nav) * 100, 10) if nav > 0 else 0
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pct,
        number={"suffix": "%", "font": {"size": 30}},
        delta={"reference": 2.0, "suffix": "%", "increasing": {"color": "#EF553B"},
               "decreasing": {"color": "#00CC96"}},
        title={"text": f"<b>VaR {int(conf*100)}%</b> as % of Portfolio Value",
               "font": {"size": 13}},
        gauge={
            "axis":  {"range": [0, 10], "ticksuffix": "%"},
            "bar":   {"color": "#EF553B", "thickness": 0.28},
            "steps": [
                {"range": [0, 2],   "color": "#d4edda"},  # green — low risk
                {"range": [2, 5],   "color": "#fff3cd"},  # yellow — moderate
                {"range": [5, 10],  "color": "#f8d7da"},  # red — elevated
            ],
            "threshold": {"line": {"color": "black", "width": 3}, "value": pct},
        },
    ))
    fig.update_layout(height=265, margin=dict(t=60, b=10, l=20, r=20))
    return fig


def build_treemap(positions: dict) -> go.Figure:
    rows = [
        {"Ticker": t, "Sector": SECTOR_MAP.get(t, "Other"), "Value": v["value"]}
        for t, v in positions.items()
        if t != "total_nav" and v["value"] > 0
    ]
    if not rows:
        return go.Figure()
    df  = pd.DataFrame(rows)
    fig = px.treemap(
        df, path=["Sector", "Ticker"], values="Value",
        color="Sector", color_discrete_map=SECTOR_COLORS,
        title="Portfolio Holdings by Sector & Position Size",
    )
    fig.update_traces(textinfo="label+value",
                      texttemplate="%{label}<br>$%{value:,.0f}")
    fig.update_layout(height=390, margin=dict(t=50, b=5, l=5, r=5))
    return fig


def build_waterfall(attribution: dict) -> go.Figure:
    items  = sorted(attribution.items(), key=lambda x: x[1])
    labels = [k for k, _ in items] + ["Total"]
    values = [v for _, v in items]
    total  = sum(values)
    fig    = go.Figure(go.Waterfall(
        measure=["relative"] * len(values) + ["total"],
        x=labels, y=values + [total],
        connector={"line": {"color": "rgba(100,100,100,0.3)"}},
        increasing={"marker": {"color": "#00CC96"}},
        decreasing={"marker": {"color": "#EF553B"}},
        totals={"marker": {"color": "#636EFA"}},
        text=[f"${v:+,.0f}" for v in values] + [f"${total:+,.0f}"],
        textposition="outside",
    ))
    fig.update_layout(
        title="Expected Daily P&L by Position (based on 1-year average returns)",
        yaxis_title="Expected P&L ($)",
        height=330, margin=dict(t=50, b=5, l=5, r=5),
        xaxis_tickangle=-30,
    )
    return fig


def build_comp_var(component_var: dict, title: str = "Component VaR by Position") -> go.Figure:
    """Shows how much risk each position contributes to the total portfolio VaR."""
    items  = sorted(component_var.items(), key=lambda x: x[1], reverse=True)
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    fig    = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=["#EF553B" if v > 0 else "#00CC96" for v in values],
        text=[f"${v:,.0f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title=title, yaxis_title="VaR Contribution ($)",
        height=310, margin=dict(t=50, b=5, l=5, r=5),
    )
    return fig


def build_incremental_var(incremental_var: dict) -> go.Figure:
    """Shows how much the portfolio VaR would decrease if each position were removed."""
    items  = sorted(incremental_var.items(), key=lambda x: x[1], reverse=True)
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    fig    = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=["#EF553B" if v > 0 else "#00CC96" for v in values],
        text=[f"${v:+,.0f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title="Incremental VaR — VaR Reduction if Position Removed",
        yaxis_title="VaR Reduction ($)",
        height=310, margin=dict(t=50, b=5, l=5, r=5),
    )
    return fig


def build_corr(corr_matrix: pd.DataFrame, tickers: list,
               title: str = "Return Correlation Matrix") -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=corr_matrix.values, x=tickers, y=tickers,
        colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
        text=corr_matrix.round(2).values,
        texttemplate="%{text}",
        colorbar={"title": "Correlation"},
    ))
    fig.update_layout(title=title, height=400, margin=dict(t=50, b=5, l=5, r=5))
    return fig


def build_equity_curve(ts: list, pnl: list) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=ts, y=pnl, mode="lines",
        line=dict(color="#636EFA", width=2),
        fill="tozeroy", fillcolor="rgba(99,110,250,0.10)",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(
        title="Intraday Portfolio P&L",
        yaxis_title="P&L ($)", height=280,
        margin=dict(t=45, b=5, l=5, r=5),
    )
    return fig


def build_var_hist(db_df: pd.DataFrame, conf: float) -> go.Figure:
    if db_df.empty:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=db_df["ts"], y=db_df["var_95"],
        mode="lines", name="VaR",
        line=dict(color="#EF553B", width=2),
        fill="tozeroy", fillcolor="rgba(239,85,59,0.08)",
    ))
    fig.update_layout(
        title=f"VaR {int(conf*100)}% History — Persistent (SQLite)",
        yaxis_title="VaR ($)", height=260,
        margin=dict(t=45, b=5, l=5, r=5),
    )
    return fig


def build_backtest(bt_result: dict) -> go.Figure:
    if not bt_result or "aligned_df" not in bt_result:
        return go.Figure()
    df  = bt_result["aligned_df"]
    fig = go.Figure()

    # Basel traffic light background band
    tl = bt_result.get("traffic_light", "green")
    tl_color = {"green": "rgba(0,204,150,0.05)", "yellow": "rgba(255,161,90,0.07)",
                "red":   "rgba(239,85,59,0.07)"}.get(tl, "rgba(0,0,0,0)")
    fig.add_hrect(
        y0=df["pnl"].min(), y1=0,
        fillcolor=tl_color, line_width=0,
        annotation_text=f"Basel: {tl.upper()}", annotation_position="top left",
    )
    fig.add_trace(go.Scatter(
        x=df.index, y=-df["var"], mode="lines",
        line=dict(color="#EF553B", width=1.5, dash="dot"),
        name="−VaR Limit",
    ))
    fig.add_trace(go.Bar(
        x=df.index, y=df["pnl"],
        marker_color=["#EF553B" if b else "#00CC96" for b in df["breach"]],
        name="Realized P&L",
    ))
    fig.update_layout(
        title="VaR Backtest: Realized P&L vs VaR Limit (red bars = breaches)",
        yaxis_title="P&L ($)", height=360,
        margin=dict(t=50, b=5, l=5, r=5),
    )
    return fig


def build_drawdown_chart(drawdown_series: np.ndarray,
                          returns_index: pd.Index) -> go.Figure:
    """Underwater curve showing % drawdown from peak over time."""
    if drawdown_series is None:
        return go.Figure()
    idx = returns_index[-len(drawdown_series):]
    fig = go.Figure(go.Scatter(
        x=idx, y=drawdown_series * 100,
        mode="lines", fill="tozeroy",
        line=dict(color="#EF553B", width=1.5),
        fillcolor="rgba(239,85,59,0.10)",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.4)
    fig.update_layout(
        title="Drawdown Chart — % Below Peak (Underwater Curve)",
        yaxis_title="Drawdown (%)", height=280,
        margin=dict(t=50, b=5, l=5, r=5),
    )
    return fig


def build_cumulative_return(cumulative_series: np.ndarray,
                             returns_index: pd.Index) -> go.Figure:
    if cumulative_series is None:
        return go.Figure()
    idx = returns_index[-len(cumulative_series):]
    pct = (cumulative_series - 1) * 100
    fig = go.Figure(go.Scatter(
        x=idx, y=pct, mode="lines",
        line=dict(color="#636EFA", width=2),
        fill="tozeroy", fillcolor="rgba(99,110,250,0.08)",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.4)
    fig.update_layout(
        title="Cumulative Portfolio Return (1-Year Lookback)",
        yaxis_title="Return (%)", height=280,
        margin=dict(t=50, b=5, l=5, r=5),
    )
    return fig


def build_rolling_metric(port_ret_series: np.ndarray,
                          returns_index: pd.Index,
                          window: int = 30) -> go.Figure:
    """Rolling Sharpe and Sortino ratio over a moving window."""
    from config.settings import RISK_FREE_RATE
    if port_ret_series is None or len(port_ret_series) < window + 5:
        return go.Figure()

    daily_rf = RISK_FREE_RATE / 252
    sharpes, sortinos, dates = [], [], []

    for i in range(window, len(port_ret_series)):
        w  = port_ret_series[i - window: i]
        ex = w - daily_rf
        dn = ex[ex < 0]
        sharpes.append(float(ex.mean() / (ex.std() + 1e-12) * np.sqrt(252)))
        ds = float(np.std(dn)) if len(dn) > 1 else 1e-12
        sortinos.append(float(ex.mean() / (ds + 1e-12) * np.sqrt(252)))
        dates.append(returns_index[i])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=sharpes, mode="lines", name=f"Rolling Sharpe ({window}d)",
        line=dict(color="#636EFA", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=sortinos, mode="lines", name=f"Rolling Sortino ({window}d)",
        line=dict(color="#00CC96", width=2),
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.4)
    fig.add_hline(y=1, line_dash="dot", line_color="#FFA15A", opacity=0.5,
                  annotation_text="Sharpe = 1 (target)", annotation_position="right")
    fig.update_layout(
        title=f"Rolling {window}-Day Sharpe & Sortino Ratios",
        yaxis_title="Ratio", height=300,
        margin=dict(t=50, b=5, l=5, r=5),
        legend=dict(orientation="h", y=-0.15),
    )
    return fig


def build_var_method_comparison(risk: dict, holding_period: int) -> go.Figure:
    """Side-by-side bar chart comparing three VaR methods."""
    sq = holding_period ** 0.5
    methods = ["Monte Carlo\n(Gaussian)", "Historical\nSimulation", "Student-t\n(Fat-tail)"]
    values  = [
        risk["var_95"]       * sq,
        risk["var_hist_sim"] * sq,
        risk["var_t"]        * sq,
    ]
    colors = ["#636EFA", "#00CC96", "#EF553B"]
    fig = go.Figure(go.Bar(
        x=methods, y=values, marker_color=colors,
        text=[f"${v:,.0f}" for v in values], textposition="outside",
    ))
    fig.update_layout(
        title=f"VaR Comparison — Three Methods ({holding_period}-Day Horizon)",
        yaxis_title="VaR ($)", height=320,
        margin=dict(t=50, b=5, l=5, r=5),
    )
    return fig


# ── Load Data ──────────────────────────────────────────────────────────────────
# Reload returns if portfolio switched
cache_portfolio_key = tuple(sorted(active_portfolio.items()))
if (st.session_state.returns_df is None
        or st.session_state.get("_active_portfolio_key") != cache_portfolio_key):
    with st.spinner("Loading 1-year price history…"):
        st.session_state.returns_df = get_historical_returns(portfolio=active_portfolio)
    st.session_state["_active_portfolio_key"] = cache_portfolio_key

returns_df = st.session_state.returns_df

prices = get_latest_prices()
if prices is not None:
    if st.session_state.base_prices is None:
        st.session_state.base_prices = dict(prices)
    st.session_state.last_prices = prices
    save_prices(prices)
elif st.session_state.last_prices is not None:
    prices = st.session_state.last_prices
else:
    st.warning("⏳ Waiting for price feed — is the producer running?")
    time.sleep(3)
    st.rerun()

positions   = mark_to_market(prices, portfolio=active_portfolio)
risk        = compute_var_monte_carlo(returns_df, prices,
                                       confidence=confidence,
                                       portfolio=active_portfolio)
stress      = run_stress_tests(prices, portfolio=active_portfolio)
var_scaled  = round(risk["var_95"]   * (holding_period ** 0.5), 2)
cvar_scaled = round(risk["cvar_95"]  * (holding_period ** 0.5), 2)
var_hist_sc = round(risk["var_hist_sim"] * (holding_period ** 0.5), 2)
var_t_sc    = round(risk["var_t"]    * (holding_period ** 0.5), 2)

save_var_snapshot(risk, confidence)
db_var_hist = load_var_history(500)

base_nav = sum(
    st.session_state.base_prices.get(t, p) * active_portfolio.get(t, 0)
    for t, p in (st.session_state.base_prices or prices).items()
)
intraday_pnl = risk["total_nav"] - base_nav
st.session_state.equity_curve.append(intraday_pnl)
st.session_state.eq_timestamps.append(pd.Timestamp.now())
if len(st.session_state.equity_curve) > 500:
    st.session_state.equity_curve  = st.session_state.equity_curve[-500:]
    st.session_state.eq_timestamps = st.session_state.eq_timestamps[-500:]

var_pct_nav = (var_scaled / risk["total_nav"] * 100) if risk["total_nav"] else 0
alert_active = var_scaled > alert_var

# ── Page Header ────────────────────────────────────────────────────────────────
col_title, col_status = st.columns([3, 1])
with col_title:
    st.title(f"📊 Portfolio Risk Dashboard — {active_portfolio_name}")
with col_status:
    market_now = pd.Timestamp.now(tz="US/Eastern")
    is_market  = (
        market_now.weekday() < 5
        and 9 * 60 + 30 <= market_now.hour * 60 + market_now.minute <= 16 * 60
    )
    st.metric(
        "Market Status",
        "🟢 LIVE" if is_market else "🟡 Simulation",
        help="Green = US market hours (9:30–16:00 ET). Off-hours uses simulated prices.",
    )

# ── Alert Banner ───────────────────────────────────────────────────────────────
if alert_active:
    st.error(
        f"⚠️  **VaR ALERT** — Current {holding_period}-day VaR of "
        f"**{fmt_money(var_scaled)}** exceeds your alert threshold of "
        f"**{fmt_money(alert_var)}**. "
        f"Portfolio is taking elevated risk ({var_pct_nav:.1f}% of NAV).",
        icon="🚨",
    )

# ── Live Ticker Strip ──────────────────────────────────────────────────────────
st.markdown(
    ticker_html(prices, st.session_state.base_prices or prices),
    unsafe_allow_html=True,
)

# ── Top KPI Metrics ────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
k1.metric(
    "Portfolio Value (NAV)",
    fmt_money(risk["total_nav"]),
    help="Total market value of all holdings at current prices. NAV = Net Asset Value.",
)
k2.metric(
    f"VaR {int(confidence*100)}% ({holding_period}d)",
    fmt_money(var_scaled),
    delta_color="inverse",
    help=(
        f"Maximum expected loss over {holding_period} trading day(s) at {int(confidence*100)}% confidence. "
        f"On {int((1-confidence)*100)}% of days, losses may exceed this amount."
    ),
)
k3.metric(
    "Expected Shortfall (CVaR)",
    fmt_money(cvar_scaled),
    delta_color="inverse",
    help=(
        "Average loss in the worst-case scenarios beyond the VaR threshold. "
        "Also called 'Expected Shortfall'. This is what you expect to lose on a bad day."
    ),
)
k4.metric(
    "VaR as % of Portfolio",
    f"{var_pct_nav:.2f}%",
    help=(
        "Risk as a share of portfolio value. "
        "Industry convention: <2% = low, 2–5% = moderate, >5% = elevated risk."
    ),
)
k5.metric(
    "Beta (vs S&P 500)",
    f"{risk['beta']:.3f}" if risk["beta"] is not None else "—",
    help=(
        "Sensitivity to broad market moves. Beta=1.0 means the portfolio moves "
        "in lockstep with the S&P 500. Beta>1 = more volatile than the market."
    ),
)
k6.metric(
    "Sharpe Ratio (Ann.)",
    f"{risk['sharpe']:.2f}" if risk["sharpe"] is not None else "—",
    help=(
        "Return earned per unit of risk taken (annualised). "
        ">1.0 = good, >2.0 = excellent, <0 = risk-free rate beats the portfolio."
    ),
)
k7.metric(
    "Intraday P&L",
    f"${intraday_pnl:+,.0f}",
    delta=f"${intraday_pnl:+,.0f}",
    delta_color="normal",
    help="Change in portfolio value since today's first price observation.",
)

# ── Live Account (if enabled) ──────────────────────────────────────────────────
if USE_LIVE_PORTFOLIO:
    try:
        acct = get_account_info()
        st.divider()
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Account Equity",  f"${acct['equity']:,.2f}",
                  help="Total account equity including cash and open positions.")
        a2.metric("Buying Power",    f"${acct['buying_power']:,.2f}",
                  help="Cash available to deploy into new positions.")
        a3.metric("Daily P&L",       f"${acct['daily_pnl']:+,.2f}",
                  delta=f"${acct['daily_pnl']:+,.2f}", delta_color="normal",
                  help="Realized and unrealized gain/loss since market open.")
        a4.metric("Unrealized P&L",  f"${acct['unrealized_pnl']:+,.2f}",
                  delta_color="normal",
                  help="Paper gains/losses on open positions (not yet locked in).")
    except Exception:
        pass

# ── Tabs ───────────────────────────────────────────────────────────────────────
(tab_main, tab_risk, tab_perf,
 tab_bt, tab_greeks, tab_multi, tab_builder) = st.tabs([
    "📈 Overview",
    "🔬 Risk Deep Dive",
    "🏆 Performance",
    "📋 Backtesting",
    "⚗️ Options Greeks",
    "📊 Multi-Portfolio",
    "🎯 Scenario Builder",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab_main:
    c1, c2 = st.columns([1, 2])
    with c1:
        st.plotly_chart(
            build_gauge(var_scaled, risk["total_nav"], confidence),
            width="stretch", key="gauge",
        )
        risk_label = (
            "🟢 LOW RISK" if var_pct_nav < 2 else
            "🟡 MODERATE RISK" if var_pct_nav < 5 else
            "🔴 ELEVATED RISK"
        )
        st.markdown(
            f'<div style="text-align:center;font-size:1.1rem;font-weight:700;'
            f'margin-top:-10px">{risk_label}</div>',
            unsafe_allow_html=True,
        )
    with c2:
        if len(st.session_state.equity_curve) > 1:
            st.plotly_chart(
                build_equity_curve(st.session_state.eq_timestamps,
                                   st.session_state.equity_curve),
                width="stretch", key="equity_curve",
            )
        else:
            st.info("Intraday P&L curve builds after a few data ticks.")

    st.divider()
    c3, c4 = st.columns([3, 2])
    with c3:
        st.plotly_chart(build_treemap(positions), key="treemap")
    with c4:
        st.subheader("Current Positions")
        if USE_LIVE_PORTFOLIO:
            try:
                pd_data = get_live_positions_detail()
                pos_df  = pd.DataFrame([{
                    "Ticker": p["symbol"],
                    "Sector": SECTOR_MAP.get(p["symbol"], "Other"),
                    "Qty":    int(p["qty"]),
                    "Price":  f"${p['current_price']:,.2f}",
                    "Value":  f"${p['market_value']:,.0f}",
                    "Unreal. P&L": f"${p['unrealized_pl']:+,.0f}",
                    "P&L %": f"{p['unrealized_plpc']:+.2f}%",
                } for p in pd_data])
                st.dataframe(pos_df, width="stretch", hide_index=True, height=360)
            except Exception as e:
                st.warning(str(e))
        else:
            pdf = pd.DataFrame([{
                "Ticker": t,
                "Sector": SECTOR_MAP.get(t, "Other"),
                "Shares": active_portfolio.get(t, 0),
                "Price":  f"${p:,.2f}",
                "Value":  f"${positions[t]['value']:,.0f}",
                "Weight": f"{positions[t]['value'] / risk['total_nav'] * 100:.1f}%"
                          if risk["total_nav"] else "—",
            } for t, p in prices.items() if t in positions])
            st.dataframe(pdf, width="stretch", hide_index=True, height=360)

    st.divider()
    st.subheader("Stress Test Results — Portfolio P&L Under Macro Shocks")
    st.markdown(
        '<div class="help-box">These scenarios apply historical-style shocks to each position. '
        'Red bars = portfolio losses. Green bars = portfolio gains (e.g. short duration / safe-haven positions).</div>',
        unsafe_allow_html=True,
    )
    sdf     = pd.DataFrame([{"Scenario": k, "P&L ($)": v} for k, v in stress.items()
                             if not k.startswith("__custom__")])
    fig_stress = go.Figure(go.Bar(
        x=sdf["Scenario"], y=sdf["P&L ($)"],
        marker_color=["#EF553B" if v < 0 else "#00CC96" for v in sdf["P&L ($)"]],
        text=sdf["P&L ($)"].apply(lambda x: f"${x:+,.0f}"),
        textposition="outside",
    ))
    fig_stress.update_layout(
        yaxis_title="Portfolio P&L ($)",
        xaxis_tickangle=-30, height=350,
        margin=dict(t=10, b=5, l=5, r=5),
    )
    st.plotly_chart(fig_stress, key="stress")

    st.divider()
    st.subheader("Recent Orders")
    try:
        orders = get_recent_orders(limit=15, status="all")
        if orders:
            odf = pd.DataFrame(orders)
            odf["filled_price"] = odf["filled_price"].apply(
                lambda x: f"${x:,.2f}" if x > 0 else "—"
            )
            st.dataframe(odf, width="stretch", hide_index=True, height=280)
        else:
            st.info("No recent orders found.")
    except Exception as e:
        st.warning(str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — RISK DEEP DIVE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_risk:
    st.markdown(
        '<div class="help-box">'
        '<b>Three VaR Methods Explained:</b><br>'
        '• <b>Monte Carlo (Gaussian)</b>: Simulates 10,000 random market scenarios assuming normally distributed returns. Fast but underestimates fat-tail events.<br>'
        '• <b>Historical Simulation</b>: Uses actual historical daily returns — no distributional assumption. What actually happened in the past 252 days.<br>'
        '• <b>Student-t (Fat-tail)</b>: Fits a heavy-tailed distribution to capture extreme market events better than the Gaussian model.'
        '</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        build_var_method_comparison(risk, holding_period),
        width="stretch", key="var_comparison",
    )

    st.divider()
    r1, r2 = st.columns(2)
    with r1:
        if show_waterfall and risk.get("attribution"):
            st.plotly_chart(
                build_waterfall(risk["attribution"]),
                width="stretch", key="waterfall",
            )
    with r2:
        if risk.get("component_var"):
            st.plotly_chart(
                build_comp_var(risk["component_var"]),
                width="stretch", key="comp_var",
            )

    st.divider()
    st.subheader("Incremental VaR — Impact of Removing Each Position")
    st.markdown(
        '<div class="help-box">'
        'Shows how much the portfolio VaR would decrease if each position were fully removed. '
        'Large positive bars = positions adding the most risk. '
        'Negative bars = positions providing diversification (hedging portfolio risk).'
        '</div>',
        unsafe_allow_html=True,
    )
    if risk.get("incremental_var"):
        st.plotly_chart(
            build_incremental_var(risk["incremental_var"]),
            width="stretch", key="incr_var",
        )

    st.divider()
    if show_corr and risk.get("corr_matrix") is not None:
        corr_opt = st.radio(
            "Correlation Window",
            ["Full Period (252 days)", f"Rolling {rolling_window}-Day"],
            horizontal=True,
            help="Full period shows long-run correlations. Rolling shows recent co-movement.",
        )
        if corr_opt.startswith("Rolling") and returns_df is not None:
            rolling_corr = returns_df[risk["tickers"]].tail(rolling_window).corr()
            st.plotly_chart(
                build_corr(rolling_corr, risk["tickers"],
                           title=f"Return Correlation — Last {rolling_window} Days"),
                width="stretch", key="corr_roll",
            )
        else:
            st.plotly_chart(
                build_corr(risk["corr_matrix"], risk["tickers"]),
                width="stretch", key="corr_full",
            )

    st.divider()
    st.subheader("VaR History (Persistent — SQLite)")
    if not db_var_hist.empty:
        st.plotly_chart(
            build_var_hist(db_var_hist, confidence),
            width="stretch", key="var_hist_db",
        )
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Average VaR",   fmt_money(db_var_hist["var_95"].mean()),
                  help="Average VaR across all recorded snapshots in the database.")
        d2.metric("Peak VaR",      fmt_money(db_var_hist["var_95"].max()),
                  help="Highest VaR level ever recorded — your worst risk reading.")
        d3.metric("Average Beta",  f"{db_var_hist['beta'].mean():.3f}"
                  if db_var_hist["beta"].notna().any() else "—")
        d4.metric("Average Sharpe",f"{db_var_hist['sharpe'].mean():.2f}"
                  if db_var_hist["sharpe"].notna().any() else "—")
    else:
        st.info("VaR history will appear after the first data cycle.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_perf:
    st.markdown(
        '<div class="help-box">'
        'Historical performance analytics based on 1-year (252 trading day) lookback. '
        'All ratios are annualised. Performance is based on simulated prices when markets are closed.'
        '</div>',
        unsafe_allow_html=True,
    )

    p1, p2, p3, p4, p5 = st.columns(5)
    p1.metric(
        "Ann. Return",
        f"{risk['ann_return']:.1f}%" if risk["ann_return"] is not None else "—",
        help="Annualized historical return: what 1 year of this portfolio's performance would yield.",
    )
    p2.metric(
        "Max Drawdown",
        f"{risk['max_drawdown']:.1f}%" if risk["max_drawdown"] is not None else "—",
        help=(
            "Worst peak-to-trough decline over the lookback period. "
            "E.g. −15% means the portfolio fell 15% from its high before recovering."
        ),
    )
    p3.metric(
        "Sharpe Ratio",
        f"{risk['sharpe']:.2f}" if risk["sharpe"] is not None else "—",
        help="Return per unit of total risk (volatility). >1 = good, >2 = excellent.",
    )
    p4.metric(
        "Sortino Ratio",
        f"{risk['sortino']:.2f}" if risk["sortino"] is not None else "—",
        help=(
            "Like Sharpe, but only penalises downside volatility. "
            "More relevant for analysts — upside volatility is not a risk."
        ),
    )
    p5.metric(
        "Calmar Ratio",
        f"{risk['calmar']:.2f}" if risk["calmar"] is not None else "—",
        help=(
            "Annualized return divided by maximum drawdown. "
            "Shows how much return you earn per unit of drawdown pain. >1 = good."
        ),
    )

    st.divider()
    if risk.get("cumulative_series") is not None and returns_df is not None:
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(
                build_cumulative_return(risk["cumulative_series"], returns_df.index),
                width="stretch", key="cum_return",
            )
        with col_b:
            st.plotly_chart(
                build_drawdown_chart(risk["drawdown_series"], returns_df.index),
                width="stretch", key="drawdown",
            )

    st.divider()
    if risk.get("port_ret_series") is not None and returns_df is not None:
        st.plotly_chart(
            build_rolling_metric(risk["port_ret_series"], returns_df.index, rolling_window),
            width="stretch", key="rolling_ratios",
        )

    # Daily return distribution
    if risk.get("port_ret_series") is not None:
        st.divider()
        st.subheader("Daily Return Distribution")
        ret_pct = risk["port_ret_series"] * 100
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=ret_pct, nbinsx=50,
            marker_color="#636EFA", opacity=0.75,
            name="Daily Returns",
        ))
        fig_dist.add_vline(
            x=float(np.percentile(ret_pct, (1 - confidence) * 100)),
            line_dash="dash", line_color="#EF553B",
            annotation_text=f"VaR {int(confidence*100)}%",
            annotation_position="top left",
        )
        fig_dist.update_layout(
            title="Distribution of Historical Daily Returns",
            xaxis_title="Daily Return (%)",
            yaxis_title="Frequency",
            height=300, margin=dict(t=50, b=5, l=5, r=5),
        )
        st.plotly_chart(fig_dist, key="ret_dist")
        st.caption(
            f"Skewness: {float(pd.Series(ret_pct).skew()):.2f}  |  "
            f"Kurtosis: {float(pd.Series(ret_pct).kurtosis()):.2f}  |  "
            f"(Normal distribution has skew=0, excess kurtosis=0. "
            f"Fat tails show positive excess kurtosis.)"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — BACKTESTING
# ═══════════════════════════════════════════════════════════════════════════════
with tab_bt:
    if show_backtest:
        st.markdown(
            '<div class="help-box">'
            '<b>How to read this:</b> The backtest rolls a 250-day window over historical data '
            'and counts how often actual losses exceeded the VaR forecast.<br>'
            '• <b>Kupiec Test</b>: Checks if the breach <i>rate</i> is statistically correct.<br>'
            '• <b>Christoffersen Test</b>: Checks if breaches are <i>independent</i> (random) vs clustered during market stress.<br>'
            '• <b>Basel Traffic Light</b>: Regulatory classification — Green = model valid, Yellow = under scrutiny, Red = recalibrate now.'
            '</div>',
            unsafe_allow_html=True,
        )

        # Single confidence backtest (main view)
        with st.spinner("Running backtest (250-day rolling window)…"):
            bt = run_historical_backtest(returns_df, prices, confidence)

        if bt:
            tl     = bt.get("traffic_light", "green")
            tl_col = {"green": "🟢", "yellow": "🟡", "red": "🔴"}.get(tl, "⚪")
            ind    = bt.get("christoffersen", {})
            ind_ok = ind.get("independent", True)

            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Observations",    bt["n_total"],
                      help="Total trading days in the backtest window.")
            c2.metric("Breaches",        bt["n_breach"],
                      help="Days where actual loss exceeded the VaR forecast.")
            c3.metric("Actual Rate",     f"{bt['breach_rate']}%",
                      help="Observed breach rate = Breaches / Observations × 100")
            c4.metric("Expected Rate",   f"{bt['expected_rate']}%",
                      help=f"Theoretical breach rate at {int(confidence*100)}% confidence = {100-int(confidence*100)}%")
            c5.metric("Kupiec p-value",
                      "✅ PASS" if bt["passed"] else "❌ FAIL",
                      delta=f"p={bt['p_value']}", delta_color="off",
                      help="p > 0.05 means the breach rate is statistically consistent with the model.")
            c6.metric("Basel Rating",
                      f"{tl_col} {tl.upper()}",
                      help="Basel II/III Traffic Light: Green (≤1.5× expected) / Yellow (≤3×) / Red (>3×).")

            # Christoffersen test
            st.markdown(
                f'**Independence Test (Christoffersen):** '
                f'{"✅ Breaches are random (no clustering) — model captures volatility regimes" if ind_ok else "⚠️ Breaches are CLUSTERED — model underperforms during market stress"}'
                f'  *(p = {ind.get("p_ind", "—")})*'
            )

            st.plotly_chart(build_backtest(bt), key="backtest")

            if bt.get("breaches"):
                st.caption(
                    "Breach dates: " + ", ".join(str(d)[:10] for d in bt["breaches"][:12])
                )

            # Multi-confidence comparison
            st.divider()
            st.subheader("Multi-Confidence Level Comparison")
            with st.spinner("Running 90% / 95% / 99% backtests…"):
                multi_bt = run_multi_confidence_backtest(returns_df, prices)

            mc_rows = []
            for cf, res in multi_bt.items():
                if not res:
                    continue
                tl_m = res.get("traffic_light", "green")
                tl_m_icon = {"green": "🟢", "yellow": "🟡", "red": "🔴"}.get(tl_m, "⚪")
                mc_rows.append({
                    "Confidence":     f"{int(cf*100)}%",
                    "Observations":   res["n_total"],
                    "Breaches":       res["n_breach"],
                    "Actual Rate":    f"{res['breach_rate']}%",
                    "Expected Rate":  f"{res['expected_rate']}%",
                    "Kupiec p":       res["p_value"],
                    "Result":         "✅ PASS" if res["passed"] else "❌ FAIL",
                    "Basel":          f"{tl_m_icon} {tl_m.upper()}",
                })
            if mc_rows:
                st.dataframe(pd.DataFrame(mc_rows), width="stretch", hide_index=True)
        else:
            st.info("Not enough historical data for backtesting (need >252 days).")
    else:
        st.info("Enable 'VaR Backtesting' in the sidebar to view this tab.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — OPTIONS GREEKS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_greeks:
    if show_greeks:
        spot = prices.get(greeks_ticker, 0)
        st.subheader(f"Option Chain & Greeks — {greeks_ticker}  @  ${spot:,.2f}")
        st.markdown(
            '<div class="help-box">'
            '<b>Greeks Quick Reference:</b><br>'
            '• <b>Delta</b>: P&L per $1 move in the stock. Call delta: 0→1. Put delta: -1→0.<br>'
            '• <b>Gamma</b>: Rate at which delta changes — highest at-the-money, near expiry.<br>'
            '• <b>Theta</b>: Daily time decay in $. Negative = option loses value each day (long options).<br>'
            '• <b>Vega</b>:  P&L per 1% rise in implied volatility.<br>'
            '• <b>Rho</b>:   P&L per 1% rise in the risk-free rate (most relevant for long-dated options).<br>'
            '• <b>IV</b>:    Implied Volatility — the market\'s consensus forecast of future price swings.'
            '</div>',
            unsafe_allow_html=True,
        )

        # Expiry selector
        col_exp1, col_exp2 = st.columns([2, 3])
        with col_exp1:
            with st.spinner(f"Fetching available expiries for {greeks_ticker}…"):
                available_expiries = fetch_available_expiries(greeks_ticker)
            selected_expiry = st.selectbox(
                "Option Expiry Date",
                available_expiries if available_expiries else ["—"],
                help="Select the expiry date for the option chain. Nearest expiry is default.",
            )

        with st.spinner(f"Computing Greeks for {greeks_ticker} ({selected_expiry})…"):
            greeks_df = fetch_option_chain_greeks(
                greeks_ticker, spot,
                expiry=selected_expiry if available_expiries else None,
            )

        if not greeks_df.empty:
            calls_df = greeks_df[greeks_df["Type"] == "CALL"].copy()
            puts_df  = greeks_df[greeks_df["Type"] == "PUT"].copy()

            gc, gp = st.columns(2)
            with gc:
                st.markdown("**📗 Calls**")
                fmt_cols = {
                    "delta": "{:.4f}", "gamma": "{:.6f}",
                    "theta": "{:.4f}", "vega": "{:.4f}", "rho": "{:.4f}",
                }
                st.dataframe(
                    calls_df.set_index("Strike").style.format(fmt_cols),
                    width="stretch", height=300,
                )
            with gp:
                st.markdown("**📕 Puts**")
                st.dataframe(
                    puts_df.set_index("Strike").style.format(fmt_cols),
                    width="stretch", height=300,
                )

            # IV Smile
            fig_iv = go.Figure()
            fig_iv.add_trace(go.Scatter(
                x=calls_df["Strike"], y=calls_df["IV"],
                mode="lines+markers", name="Call IV (%)",
                line=dict(color="#636EFA"),
            ))
            fig_iv.add_trace(go.Scatter(
                x=puts_df["Strike"], y=puts_df["IV"],
                mode="lines+markers", name="Put IV (%)",
                line=dict(color="#EF553B"),
            ))
            fig_iv.add_vline(
                x=spot, line_dash="dash", line_color="gray",
                annotation_text=f"Spot ${spot:,.2f}", annotation_position="top right",
            )
            fig_iv.update_layout(
                title=f"Implied Volatility Smile — {greeks_ticker} ({selected_expiry})",
                xaxis_title="Strike Price ($)", yaxis_title="Implied Volatility (%)",
                height=310, margin=dict(t=50, b=5, l=5, r=5),
            )
            st.plotly_chart(fig_iv, key="iv_smile")

            # Greeks Ladder
            st.divider()
            st.subheader("Greeks Ladder — Time Decay Progression Across Expiries")
            st.markdown(
                '<div class="help-box">'
                'Shows how ATM option Greeks evolve as you move further out in time. '
                'Key insight: Theta (time decay) is highest for near-term options. '
                'Vega (volatility sensitivity) grows with time to expiry.'
                '</div>',
                unsafe_allow_html=True,
            )
            with st.spinner("Building Greeks ladder…"):
                ladder_df = fetch_greeks_ladder(greeks_ticker, spot)
            if not ladder_df.empty:
                st.dataframe(
                    ladder_df.set_index(["Expiry", "Type"]),
                    width="stretch",
                )
                # Theta ladder chart
                call_ladder = ladder_df[ladder_df["Type"] == "CALL"]
                put_ladder  = ladder_df[ladder_df["Type"] == "PUT"]
                fig_ladder  = go.Figure()
                fig_ladder.add_trace(go.Bar(
                    x=call_ladder["Expiry"], y=call_ladder["theta"],
                    name="Call Theta ($/day)", marker_color="#636EFA",
                ))
                fig_ladder.add_trace(go.Bar(
                    x=put_ladder["Expiry"], y=put_ladder["theta"],
                    name="Put Theta ($/day)", marker_color="#EF553B",
                ))
                fig_ladder.update_layout(
                    barmode="group",
                    title=f"ATM Daily Theta (Time Decay) — {greeks_ticker}",
                    yaxis_title="Theta ($/day)", height=280,
                    margin=dict(t=50, b=5, l=5, r=5),
                )
                st.plotly_chart(fig_ladder, key="theta_ladder")
            else:
                st.info("Greeks ladder requires multiple expiries to be available.")
        else:
            st.warning(f"No option chain data available for {greeks_ticker}.")
    else:
        st.info("Enable 'Options Greeks' in the sidebar to view this tab.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — MULTI-PORTFOLIO
# ═══════════════════════════════════════════════════════════════════════════════
with tab_multi:
    st.subheader("Portfolio Risk Comparison")
    st.markdown(
        '<div class="help-box">'
        'Compare risk and return metrics across three portfolio configurations side by side. '
        'Use this to evaluate whether rebalancing toward a more defensive or tech-concentrated '
        'allocation changes your overall risk profile meaningfully.'
        '</div>',
        unsafe_allow_html=True,
    )

    rows = []
    for pname, pfolio in PORTFOLIOS.items():
        ret = get_historical_returns(portfolio=pfolio)
        r   = compute_var_monte_carlo(ret, prices, confidence=confidence, portfolio=pfolio)
        nav = r["total_nav"]
        rows.append({
            "Portfolio":        pname,
            "NAV":              fmt_money(nav),
            f"VaR {int(confidence*100)}%": fmt_money(r["var_95"]),
            "CVaR":             fmt_money(r["cvar_95"]),
            "VaR % NAV":        f"{(r['var_95']/nav*100):.2f}%" if nav else "—",
            "Beta":             f"{r['beta']:.3f}" if r["beta"] else "—",
            "Sharpe":           f"{r['sharpe']:.2f}" if r["sharpe"] else "—",
            "Sortino":          f"{r['sortino']:.2f}" if r["sortino"] else "—",
            "Max Drawdown":     f"{r['max_drawdown']:.1f}%" if r["max_drawdown"] else "—",
            "Ann. Return":      f"{r['ann_return']:.1f}%" if r["ann_return"] else "—",
        })

    comp_df = pd.DataFrame(rows)
    st.dataframe(comp_df, width="stretch", hide_index=True)

    # CSV Export
    csv_data = comp_df.to_csv(index=False)
    st.download_button(
        "⬇️ Export Comparison to CSV",
        data=csv_data,
        file_name="portfolio_risk_comparison.csv",
        mime="text/csv",
        help="Download the comparison table as a CSV file for use in Excel or reporting.",
    )

    st.divider()
    st.subheader("Stress P&L Comparison Across Portfolios")
    stress_rows = {
        pname: run_stress_tests(prices, portfolio=pfolio)
        for pname, pfolio in PORTFOLIOS.items()
    }
    stress_comp = pd.DataFrame(stress_rows)
    fig_multi   = go.Figure()
    colors      = ["#636EFA", "#EF553B", "#00CC96"]
    for i, pname in enumerate(PORTFOLIOS.keys()):
        fig_multi.add_trace(go.Bar(
            name=pname, x=stress_comp.index, y=stress_comp[pname],
            marker_color=colors[i % 3],
        ))
    fig_multi.update_layout(
        barmode="group", height=400,
        yaxis_title="P&L ($)", xaxis_tickangle=-30,
        title="Stress Test P&L — All Portfolios",
        margin=dict(t=50, b=5, l=5, r=5),
    )
    st.plotly_chart(fig_multi, key="multi_stress")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7 — SCENARIO BUILDER
# ═══════════════════════════════════════════════════════════════════════════════
with tab_builder:
    st.subheader("Custom Scenario Builder")
    st.markdown(
        '<div class="help-box">'
        'Build and test your own "what if" scenarios in real time. '
        'Set individual price shocks for each ticker using the sliders below, '
        'or use the single-ticker sensitivity sweep to find your breakeven point.'
        '</div>',
        unsafe_allow_html=True,
    )

    tickers_list = sorted(active_portfolio.keys())

    with st.expander("📐 Custom Scenario: Set Shocks Per Ticker", expanded=True):
        st.caption("Set the % price change for each position in the scenario. 0% = no impact.")
        n_cols     = 4
        shock_vals = {}
        ticker_chunks = [tickers_list[i:i+n_cols] for i in range(0, len(tickers_list), n_cols)]
        for chunk in ticker_chunks:
            cols = st.columns(n_cols)
            for j, t in enumerate(chunk):
                shock_vals[t] = cols[j].slider(
                    f"{t}", -50, 50, 0, 1,
                    key=f"shock_{t}",
                    help=f"Price shock for {t} as % of current price (${prices.get(t,0):,.2f})",
                )

        # Compute custom scenario P&L
        custom_pnl = sum(
            prices.get(t, 0) * active_portfolio.get(t, 0) * (shock_vals[t] / 100)
            for t in active_portfolio
        )
        custom_nav_impact = (custom_pnl / risk["total_nav"] * 100) if risk["total_nav"] else 0
        pnl_color = "#00CC96" if custom_pnl >= 0 else "#EF553B"

        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("Scenario P&L",       f"${custom_pnl:+,.0f}",
                      help="Total dollar gain/loss from this scenario.")
        col_r2.metric("Impact on NAV",       f"{custom_nav_impact:+.2f}%",
                      help="P&L as a % of current portfolio value.")
        col_r3.metric("NAV After Scenario",  fmt_money(risk["total_nav"] + custom_pnl))

        # Per-ticker breakdown bar
        ticker_pnl = {
            t: round(prices.get(t, 0) * active_portfolio.get(t, 0) * (shock_vals[t] / 100), 2)
            for t in active_portfolio if shock_vals.get(t, 0) != 0
        }
        if ticker_pnl:
            fig_custom = go.Figure(go.Bar(
                x=list(ticker_pnl.keys()),
                y=list(ticker_pnl.values()),
                marker_color=["#00CC96" if v >= 0 else "#EF553B" for v in ticker_pnl.values()],
                text=[f"${v:+,.0f}" for v in ticker_pnl.values()],
                textposition="outside",
            ))
            fig_custom.update_layout(
                title="Custom Scenario — P&L Breakdown by Position",
                yaxis_title="P&L ($)", height=300,
                margin=dict(t=50, b=5, l=5, r=5),
            )
            st.plotly_chart(fig_custom, key="custom_scenario")

    st.divider()
    with st.expander("📊 Single-Ticker Sensitivity — Breakeven Analysis", expanded=False):
        sens_ticker = st.selectbox(
            "Select Ticker for Sensitivity Sweep",
            tickers_list, key="sens_ticker",
            help="Vary this ticker's price across a range and see the portfolio impact.",
        )
        sens_results = run_sensitivity_analysis(
            prices, sens_ticker, portfolio=active_portfolio,
            shock_range=(-0.30, 0.30), steps=13,
        )
        fig_sens = go.Figure(go.Bar(
            x=[f"{k:+.0f}%" for k in sens_results.keys()],
            y=list(sens_results.values()),
            marker_color=["#00CC96" if v >= 0 else "#EF553B" for v in sens_results.values()],
            text=[f"${v:+,.0f}" for v in sens_results.values()],
            textposition="outside",
        ))
        fig_sens.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig_sens.update_layout(
            title=f"Portfolio P&L vs {sens_ticker} Price Change (−30% to +30%)",
            xaxis_title=f"{sens_ticker} Price Shock",
            yaxis_title="Portfolio P&L ($)",
            height=310, margin=dict(t=50, b=5, l=5, r=5),
        )
        st.plotly_chart(fig_sens, key="sensitivity")

    st.divider()
    with st.expander("⚡ Shock Magnitude Adjuster — Scale Built-in Scenarios", expanded=False):
        st.caption("Scale all built-in scenario shocks up or down by a multiplier.")
        multiplier = st.slider(
            "Shock Multiplier", 0.25, 3.0, 1.0, 0.25,
            help="1.0 = original shocks. 2.0 = twice as severe. 0.5 = half severity.",
        )
        from risk.stress_scenarios import SCENARIOS
        scaled_scenarios = {
            name: {t: s * multiplier for t, s in shocks.items()}
            for name, shocks in SCENARIOS.items()
        }
        scaled_pnl = run_stress_tests(prices, portfolio=active_portfolio,
                                      custom_scenarios=scaled_scenarios)
        # Remove non-scaled entries (SCENARIOS keys overlap)
        final_scaled = {
            k: v for k, v in scaled_pnl.items()
            if k in scaled_scenarios
        }
        sdf_sc = pd.DataFrame([{"Scenario": k, "P&L ($)": v}
                                for k, v in final_scaled.items()])
        fig_sc = go.Figure(go.Bar(
            x=sdf_sc["Scenario"], y=sdf_sc["P&L ($)"],
            marker_color=["#EF553B" if v < 0 else "#00CC96" for v in sdf_sc["P&L ($)"]],
            text=sdf_sc["P&L ($)"].apply(lambda x: f"${x:+,.0f}"),
            textposition="outside",
        ))
        fig_sc.update_layout(
            title=f"Scaled Stress Tests (Multiplier: {multiplier}×)",
            yaxis_title="P&L ($)", xaxis_tickangle=-30,
            height=340, margin=dict(t=50, b=5, l=5, r=5),
        )
        st.plotly_chart(fig_sc, key="scaled_stress")


# ── Footer + Refresh Countdown ─────────────────────────────────────────────────
st.divider()
footer_col, refresh_col = st.columns([3, 1])
with footer_col:
    st.caption(
        f"Last updated: {pd.Timestamp.now().strftime('%H:%M:%S')}  |  "
        f"Confidence: {int(confidence*100)}%  |  "
        f"Holding: {holding_period}d  |  "
        f"Portfolio: {active_portfolio_name}  |  "
        f"User: {st.session_state.get('name', '')}"
    )
with refresh_col:
    countdown = st.empty()
    for remaining in range(refresh_secs, 0, -1):
        countdown.caption(f"🔄 Refreshing in {remaining}s…")
        time.sleep(1)
    countdown.caption("🔄 Refreshing now…")

st.rerun()
