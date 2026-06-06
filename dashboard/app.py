import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import json, time, yaml
import pandas as pd
import numpy as np
from pathlib import Path
from confluent_kafka import Consumer
import streamlit_authenticator as stauth

from portfolio.portfolio import (
    get_historical_returns, mark_to_market,
    get_account_info, get_live_positions_detail
)
from risk.var_engine import compute_var_monte_carlo
from risk.stress_scenarios import run_stress_tests
from risk.greeks import fetch_option_chain_greeks
from risk.backtesting import run_historical_backtest
from trading.orders import get_recent_orders
from db.persistence import save_prices, save_var_snapshot, load_var_history
from config.settings import (
    KAFKA_BROKER, TOPIC_PRICES, USE_LIVE_PORTFOLIO, PORTFOLIO
)

st.set_page_config(page_title="Portfolio Risk Dashboard", layout="wide", page_icon="📊")

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
        st.error("Incorrect username or password")
    else:
        st.info("Please log in — default: **admin / admin123**")
    st.stop()

SECTOR_MAP = {
    "AAPL":"Tech","MSFT":"Tech","GOOGL":"Tech","NVDA":"Tech",
    "META":"Tech","AMZN":"Tech","TSLA":"Tech","AMD":"Tech",
    "JPM":"Finance","GS":"Finance","BAC":"Finance","BLK":"Finance",
    "V":"Finance","MA":"Finance",
    "JNJ":"Health","UNH":"Health","PFE":"Health","ABBV":"Health",
    "XOM":"Energy","CVX":"Energy","NEE":"Energy",
    "COST":"Consumer","WMT":"Consumer","NKE":"Consumer",
    "CAT":"Indust.","LMT":"Indust.","BA":"Indust.",
    "SPY":"ETF","QQQ":"ETF","IWM":"ETF","GLD":"ETF","TLT":"ETF",
}
SECTOR_COLORS = {
    "Tech":"#636EFA","Finance":"#EF553B","Health":"#00CC96",
    "Energy":"#AB63FA","Consumer":"#FFA15A","Indust.":"#19D3F3",
    "ETF":"#FF6692","Other":"#B6E880",
}
PORTFOLIOS = {
    "Main":       PORTFOLIO,
    "Tech Heavy": {t:v for t,v in PORTFOLIO.items() if SECTOR_MAP.get(t) in ("Tech","ETF")},
    "Defensive":  {t:v for t,v in PORTFOLIO.items() if SECTOR_MAP.get(t) in ("Health","Consumer","ETF")},
}

with st.sidebar:
    st.title("⚙️ Controls")
    authenticator.logout("Logout", "sidebar")
    st.divider()
    active_portfolio_name = st.selectbox("Portfolio", list(PORTFOLIOS.keys()))
    active_portfolio      = PORTFOLIOS[active_portfolio_name]
    confidence     = st.select_slider("Confidence", [0.90,0.95,0.99], 0.95,
                                      format_func=lambda x: f"{int(x*100)}%")
    holding_period = st.slider("Holding Period (days)", 1, 10, 1)
    refresh_secs   = st.slider("Refresh (s)", 3, 30, 5)
    alert_var      = st.number_input("VaR Alert ($)", min_value=0, value=5000, step=500)
    st.divider()
    show_corr      = st.checkbox("Correlation Heatmap", True)
    show_waterfall = st.checkbox("P&L Attribution", True)
    show_backtest  = st.checkbox("VaR Backtesting", True)
    show_greeks    = st.checkbox("Options Greeks", True)
    st.divider()
    greeks_ticker  = st.selectbox("Greeks Ticker", list(active_portfolio.keys()))
    st.caption("Off-hours: simulation mode\nDB: risk_dashboard.db")

for key, default in [
    ("last_prices",None),("base_prices",None),
    ("equity_curve",[]),("eq_timestamps",[]),("alert_shown",False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

def get_latest_prices():
    c = Consumer({"bootstrap.servers":KAFKA_BROKER,
                  "group.id":f"dash-{time.time()}",
                  "auto.offset.reset":"earliest"})
    c.subscribe([TOPIC_PRICES])
    latest = None
    deadline = time.time() + 5.0
    while time.time() < deadline:
        msg = c.poll(timeout=1.0)
        if msg and not msg.error():
            latest = json.loads(msg.value())["prices"]
    c.close()
    return latest

def build_gauge(var, nav, conf):
    pct = min((var/nav)*100, 10) if nav > 0 else 0
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta", value=pct,
        number={"suffix":"%","font":{"size":28}},
        delta={"reference":2.0,"suffix":"%"},
        title={"text":f"VaR {int(conf*100)}% / NAV"},
        gauge={"axis":{"range":[0,10],"ticksuffix":"%"},
               "bar":{"color":"#EF553B"},
               "steps":[{"range":[0,2],"color":"#d4edda"},
                        {"range":[2,5],"color":"#fff3cd"},
                        {"range":[5,10],"color":"#f8d7da"}],
               "threshold":{"line":{"color":"black","width":3},"value":pct}}))
    fig.update_layout(height=260, margin=dict(t=50,b=10,l=20,r=20))
    return fig

def build_treemap(positions):
    rows = [{"Ticker":t,"Sector":SECTOR_MAP.get(t,"Other"),"Value":v["value"]}
            for t,v in positions.items() if t!="total_nav" and v["value"]>0]
    if not rows: return go.Figure()
    df = pd.DataFrame(rows)
    fig = px.treemap(df, path=["Sector","Ticker"], values="Value",
                     color="Sector", color_discrete_map=SECTOR_COLORS,
                     title="Portfolio Treemap")
    fig.update_traces(textinfo="label+value",
                      texttemplate="%{label}<br>$%{value:,.0f}")
    fig.update_layout(height=380, margin=dict(t=40,b=5,l=5,r=5))
    return fig

def build_waterfall(attribution):
    items  = sorted(attribution.items(), key=lambda x: x[1])
    labels = [k for k,_ in items]+["Total"]
    values = [v for _,v in items]
    total  = sum(values)
    fig = go.Figure(go.Waterfall(
        measure=["relative"]*len(values)+["total"],
        x=labels, y=values+[total],
        connector={"line":{"color":"rgba(100,100,100,0.3)"}},
        increasing={"marker":{"color":"#00CC96"}},
        decreasing={"marker":{"color":"#EF553B"}},
        totals={"marker":{"color":"#636EFA"}},
        text=[f"${v:+,.0f}" for v in values]+[f"${total:+,.0f}"],
        textposition="outside"))
    fig.update_layout(title="P&L Attribution (Expected Daily)",
                      yaxis_title="P&L ($)",height=320,
                      margin=dict(t=40,b=5,l=5,r=5),xaxis_tickangle=-30)
    return fig

def build_comp_var(component_var):
    items  = sorted(component_var.items(), key=lambda x: x[1], reverse=True)
    labels = [k for k,_ in items]
    values = [v for _,v in items]
    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=["#EF553B" if v>0 else "#00CC96" for v in values],
        text=[f"${v:,.0f}" for v in values], textposition="outside"))
    fig.update_layout(title="Component VaR by Ticker",
                      yaxis_title="VaR ($)", height=300,
                      margin=dict(t=40,b=5,l=5,r=5))
    return fig

def build_corr(corr_matrix, tickers):
    fig = go.Figure(go.Heatmap(
        z=corr_matrix.values, x=tickers, y=tickers,
        colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
        text=corr_matrix.round(2).values, texttemplate="%{text}"))
    fig.update_layout(title="Return Correlation Matrix",
                      height=380, margin=dict(t=40,b=5,l=5,r=5))
    return fig

def build_equity_curve(ts, pnl):
    fig = go.Figure(go.Scatter(
        x=ts, y=pnl, mode="lines",
        line=dict(color="#636EFA",width=2),
        fill="tozeroy", fillcolor="rgba(99,110,250,0.12)"))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(title="Intraday Portfolio P&L",
                      yaxis_title="P&L ($)", height=280,
                      margin=dict(t=40,b=5,l=5,r=5))
    return fig

def build_var_hist(db_df, conf):
    if db_df.empty: return go.Figure()
    fig = go.Figure(go.Scatter(
        x=db_df["ts"], y=db_df["var_95"], mode="lines",
        line=dict(color="#EF553B",width=2),
        fill="tozeroy", fillcolor="rgba(239,85,59,0.1)"))
    fig.update_layout(title=f"VaR {int(conf*100)}% History (SQLite)",
                      yaxis_title="VaR ($)", height=260,
                      margin=dict(t=40,b=5,l=5,r=5))
    return fig

def build_backtest(bt_result):
    if not bt_result or "aligned_df" not in bt_result: return go.Figure()
    df = bt_result["aligned_df"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=-df["var"], mode="lines",
                             line=dict(color="#EF553B",width=1,dash="dot"),
                             name="-VaR Limit"))
    fig.add_trace(go.Bar(x=df.index, y=df["pnl"],
                         marker_color=["#EF553B" if b else "#00CC96" for b in df["breach"]],
                         name="Realized P&L"))
    fig.update_layout(title="Kupiec VaR Backtest",
                      yaxis_title="P&L ($)", height=340,
                      margin=dict(t=40,b=5,l=5,r=5))
    return fig

def ticker_html(prices, base_prices):
    items = []
    for t, p in sorted(prices.items()):
        base = base_prices.get(t, p)
        chg  = ((p-base)/base)*100 if base else 0
        col  = "#00CC96" if chg>=0 else "#EF553B"
        arr  = "▲" if chg>=0 else "▼"
        items.append(
            f'<span style="margin:0 16px;font-size:14px;font-weight:600">'
            f'{t} <span style="color:{col}">${p:,.2f} {arr}{abs(chg):.2f}%</span></span>')
    return ('<div style="background:#0e1117;padding:8px 12px;border-radius:6px;'
            'border:1px solid #262730;overflow-x:auto;white-space:nowrap;margin-bottom:12px">'
            + "".join(items) + "</div>")

if "returns_df" not in st.session_state:
    st.session_state.returns_df = get_historical_returns(portfolio=active_portfolio)
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
    st.warning("Waiting for price feed...")
    time.sleep(3)
    st.rerun()

positions   = mark_to_market(prices, portfolio=active_portfolio)
risk        = compute_var_monte_carlo(returns_df, prices, confidence=confidence)
stress      = run_stress_tests(prices)
var_scaled  = round(risk["var_95"]  * (holding_period**0.5), 2)
cvar_scaled = round(risk["cvar_95"] * (holding_period**0.5), 2)

save_var_snapshot(risk, confidence)
db_var_hist = load_var_history(500)

base_nav = sum(
    st.session_state.base_prices.get(t,p) * active_portfolio.get(t,0)
    for t,p in (st.session_state.base_prices or prices).items()
)
intraday_pnl = risk["total_nav"] - base_nav
st.session_state.equity_curve.append(intraday_pnl)
st.session_state.eq_timestamps.append(pd.Timestamp.now())
if len(st.session_state.equity_curve) > 500:
    st.session_state.equity_curve  = st.session_state.equity_curve[-500:]
    st.session_state.eq_timestamps = st.session_state.eq_timestamps[-500:]

if var_scaled > alert_var:
    st.session_state.alert_shown = True
else:
    st.session_state.alert_shown = False

st.title(f"📊 Portfolio Risk — {active_portfolio_name}")

if st.session_state.alert_shown:
    st.error(f"⚠️ VaR ALERT: ${var_scaled:,.0f} exceeds threshold ${alert_var:,.0f}")

st.markdown(ticker_html(prices, st.session_state.base_prices or prices),
                    unsafe_allow_html=True)

k1,k2,k3,k4,k5,k6,k7 = st.columns(7)
k1.metric("Total NAV",     f"${risk['total_nav']:,.0f}")
k2.metric(f"VaR {int(confidence*100)}%", f"${var_scaled:,.0f}", delta_color="inverse")
k3.metric("CVaR / ES",     f"${cvar_scaled:,.0f}", delta_color="inverse")
k4.metric("VaR % NAV",
          f"{(var_scaled/risk['total_nav']*100):.2f}%" if risk['total_nav'] else "—")
k5.metric("Beta vs SPY",   f"{risk['beta']:.3f}"  if risk['beta']   is not None else "—")
k6.metric("Sharpe (Ann.)", f"{risk['sharpe']:.2f}" if risk['sharpe'] is not None else "—")
k7.metric("Intraday P&L",  f"${intraday_pnl:+,.0f}",
          delta=f"{intraday_pnl:+,.0f}", delta_color="normal")

if USE_LIVE_PORTFOLIO:
            try:
                acct = get_account_info()
                st.divider()
                a1,a2,a3,a4 = st.columns(4)
                a1.metric("Equity",       f"${acct['equity']:,.2f}")
                a2.metric("Buying Power", f"${acct['buying_power']:,.2f}")
                a3.metric("Daily P&L",    f"${acct['daily_pnl']:+,.2f}",
                          delta=f"{acct['daily_pnl']:+,.2f}", delta_color="normal")
                a4.metric("Unreal P&L",   f"${acct['unrealized_pnl']:+,.2f}",
                          delta_color="normal")
            except Exception:
                pass

tab_main, tab_risk, tab_bt, tab_greeks, tab_multi = st.tabs([
    "📈 Overview", "🔬 Risk Analysis",
    "📋 Backtesting", "⚗️ Options Greeks", "📊 Multi-Portfolio"
])

with tab_main:
            c1,c2 = st.columns([1,2])
            with c1:
                st.plotly_chart(build_gauge(var_scaled,risk["total_nav"],confidence),
                                use_container_width=True, key="gauge")
            with c2:
                if len(st.session_state.equity_curve) > 1:
                    st.plotly_chart(build_equity_curve(
                        st.session_state.eq_timestamps,
                        st.session_state.equity_curve),
                        use_container_width=True, key="equity_curve")
                else:
                    st.info("Equity curve builds after a few ticks.")
            st.divider()
            c3,c4 = st.columns([3,2])
            with c3:
                st.plotly_chart(build_treemap(positions),
                                use_container_width=True, key="treemap")
            with c4:
                st.subheader("Positions")
                if USE_LIVE_PORTFOLIO:
                    try:
                        pd_data = get_live_positions_detail()
                        pos_df = pd.DataFrame([{
                            "Ticker":p["symbol"],"Sector":SECTOR_MAP.get(p["symbol"],"Other"),
                            "Qty":p["qty"],"Price":f"${p['current_price']:,.2f}",
                            "Value":f"${p['market_value']:,.0f}",
                            "P&L":f"${p['unrealized_pl']:+,.0f}",
                            "P&L%":f"{p['unrealized_plpc']:+.2f}%",
                        } for p in pd_data])
                        st.dataframe(pos_df, use_container_width=True,
                                     hide_index=True, height=360)
                    except Exception as e:
                        st.warning(str(e))
                else:
                    pdf = pd.DataFrame([{
                        "Ticker":t,"Sector":SECTOR_MAP.get(t,"Other"),
                        "Price":f"${p:,.2f}",
                        "Shares":active_portfolio.get(t,0),
                        "Value":f"${positions[t]['value']:,.0f}",
                    } for t,p in prices.items() if t in positions])
                    st.dataframe(pdf, use_container_width=True,
                                 hide_index=True, height=360)
            st.divider()
            st.subheader("Stress Scenario P&L")
            sdf = pd.DataFrame([{"Scenario":k,"P&L ($)":v} for k,v in stress.items()])
            fig_s = go.Figure(go.Bar(
                x=sdf["Scenario"], y=sdf["P&L ($)"],
                marker_color=["#EF553B" if v<0 else "#00CC96" for v in sdf["P&L ($)"]],
                text=sdf["P&L ($)"].apply(lambda x: f"${x:,.0f}"),
                textposition="outside"))
            fig_s.update_layout(yaxis_title="P&L ($)", xaxis_tickangle=-30,
                                height=340, margin=dict(t=10,b=5,l=5,r=5))
            st.plotly_chart(fig_s, use_container_width=True, key="stress")
            st.divider()
            st.subheader("Recent Orders")
            try:
                orders = get_recent_orders(limit=15, status="all")
                if orders:
                    odf = pd.DataFrame(orders)
                    odf["filled_price"] = odf["filled_price"].apply(
                        lambda x: f"${x:,.2f}" if x>0 else "—")
                    st.dataframe(odf, use_container_width=True,
                                 hide_index=True, height=280)
                else:
                    st.info("No recent orders.")
            except Exception as e:
                st.warning(str(e))

with tab_risk:
            r1,r2 = st.columns(2)
            with r1:
                if show_waterfall and risk.get("attribution"):
                    st.plotly_chart(build_waterfall(risk["attribution"]),
                                    use_container_width=True, key="waterfall")
            with r2:
                if risk.get("component_var"):
                    st.plotly_chart(build_comp_var(risk["component_var"]),
                                    use_container_width=True, key="comp_var")
            if show_corr and risk.get("corr_matrix") is not None:
                st.plotly_chart(build_corr(risk["corr_matrix"], risk["tickers"]),
                                use_container_width=True, key="corr")
            st.divider()
            st.subheader("Persistent VaR History (SQLite)")
            if not db_var_hist.empty:
                st.plotly_chart(build_var_hist(db_var_hist, confidence),
                                use_container_width=True, key="var_hist_db")
                d1,d2,d3,d4 = st.columns(4)
                d1.metric("Avg VaR",   f"${db_var_hist['var_95'].mean():,.0f}")
                d2.metric("Max VaR",   f"${db_var_hist['var_95'].max():,.0f}")
                d3.metric("Avg Beta",  f"{db_var_hist['beta'].mean():.3f}"
                          if db_var_hist['beta'].notna().any() else "—")
                d4.metric("Avg Sharpe",f"{db_var_hist['sharpe'].mean():.2f}"
                          if db_var_hist['sharpe'].notna().any() else "—")
            else:
                st.info("VaR history appears after first data cycle.")

with tab_bt:
            if show_backtest:
                with st.spinner("Running Kupiec backtest…"):
                    bt = run_historical_backtest(returns_df, prices, confidence)
                if bt:
                    c1,c2,c3,c4,c5 = st.columns(5)
                    c1.metric("Observations", bt["n_total"])
                    c2.metric("Breaches",     bt["n_breach"])
                    c3.metric("Breach Rate",  f"{bt['breach_rate']}%")
                    c4.metric("Expected",     f"{bt['expected_rate']}%")
                    c5.metric("Kupiec Test",
                              "✅ PASS" if bt["passed"] else "❌ FAIL",
                              delta=f"p={bt['p_value']}", delta_color="off")
                    st.plotly_chart(build_backtest(bt),
                                    use_container_width=True, key="backtest")
                    if bt["breaches"]:
                        st.caption("Breach dates: " +
                                   ", ".join(str(d)[:10] for d in bt["breaches"][:10]))
                else:
                    st.info("Not enough historical data for backtesting.")
            else:
                st.info("Enable 'VaR Backtesting' in the sidebar.")

with tab_greeks:
            if show_greeks:
                spot = prices.get(greeks_ticker, 0)
                st.subheader(f"Option Chain Greeks — {greeks_ticker} @ ${spot:,.2f}")
                with st.spinner(f"Fetching option chain for {greeks_ticker}…"):
                    greeks_df = fetch_option_chain_greeks(greeks_ticker, spot)
                if not greeks_df.empty:
                    calls_df = greeks_df[greeks_df["Type"]=="CALL"]
                    puts_df  = greeks_df[greeks_df["Type"]=="PUT"]
                    gc,gp = st.columns(2)
                    with gc:
                        st.markdown("**Calls**")
                        st.dataframe(calls_df.set_index("Strike"),
                                     use_container_width=True, height=300)
                    with gp:
                        st.markdown("**Puts**")
                        st.dataframe(puts_df.set_index("Strike"),
                                     use_container_width=True, height=300)
                    fig_iv = go.Figure()
                    fig_iv.add_trace(go.Scatter(x=calls_df["Strike"],y=calls_df["IV"],
                                                mode="lines+markers",name="Call IV",
                                                line=dict(color="#636EFA")))
                    fig_iv.add_trace(go.Scatter(x=puts_df["Strike"],y=puts_df["IV"],
                                                mode="lines+markers",name="Put IV",
                                                line=dict(color="#EF553B")))
                    fig_iv.add_vline(x=spot, line_dash="dash", line_color="gray",
                                     annotation_text="Spot")
                    fig_iv.update_layout(title="Implied Volatility Smile",
                                         xaxis_title="Strike",yaxis_title="IV (%)",
                                         height=300, margin=dict(t=40,b=5,l=5,r=5))
                    st.plotly_chart(fig_iv, use_container_width=True, key="iv_smile")
                else:
                    st.warning(f"No option chain data for {greeks_ticker}.")
            else:
                st.info("Enable 'Options Greeks' in the sidebar.")

with tab_multi:
            st.subheader("Portfolio Comparison")
            rows = []
            for pname, pfolio in PORTFOLIOS.items():
                ret = get_historical_returns(portfolio=pfolio)
                r   = compute_var_monte_carlo(ret, prices, confidence=confidence)
                rows.append({
                    "Portfolio":pname,
                    "NAV ($)":f"${r['total_nav']:,.0f}",
                    "VaR ($)":f"${r['var_95']:,.0f}",
                    "CVaR ($)":f"${r['cvar_95']:,.0f}",
                    "VaR%NAV":f"{(r['var_95']/r['total_nav']*100):.2f}%" if r['total_nav'] else "—",
                    "Beta":f"{r['beta']:.3f}" if r['beta'] else "—",
                    "Sharpe":f"{r['sharpe']:.2f}" if r['sharpe'] else "—",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.divider()
            st.subheader("Stress P&L Comparison")
            stress_rows = {pname: run_stress_tests(prices) for pname in PORTFOLIOS}
            stress_comp = pd.DataFrame(stress_rows)
            fig_multi = go.Figure()
            for i, pname in enumerate(PORTFOLIOS.keys()):
                fig_multi.add_trace(go.Bar(
                    name=pname, x=stress_comp.index, y=stress_comp[pname],
                    marker_color=["#636EFA","#EF553B","#00CC96"][i%3]))
            fig_multi.update_layout(barmode="group", height=380,
                                    yaxis_title="P&L ($)", xaxis_tickangle=-30,
                                    margin=dict(t=10,b=5,l=5,r=5))
            st.plotly_chart(fig_multi, use_container_width=True, key="multi_stress")

st.caption(
    f"Last updated: {pd.Timestamp.now().strftime('%H:%M:%S')}  |  "
    f"Conf: {int(confidence*100)}%  |  Holding: {holding_period}d  |  "
    f"Portfolio: {active_portfolio_name}  |  "
    f"User: {st.session_state.get('name','')}"
)

time.sleep(refresh_secs)
st.rerun()
