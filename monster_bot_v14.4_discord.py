"""
╔══════════════════════════════════════════════════════════════════════════╗
║  MONSTER MATRIX UI v17.0 - CYBERPUNK COMMAND CENTER                      ║
║  🎯 Theme: Cyberpunk Dark Mode | TradingView Integrated                 ║
║  ✅ Features: Neon Colors | Soft Eyes | Professional Charts             ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

# 🔧 CRITICAL: st.set_page_config MUST BE FIRST
import streamlit as st
st.set_page_config(page_title="MONSTER MATRIX v17.0", layout="wide", page_icon="⚡")

import pandas as pd
import json
import os
import sys
import time
import numpy as np
from datetime import datetime, timedelta
import streamlit.components.v1 as components
import subprocess
import psutil
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION - SYNCED WITH ENGINE
# ════════════════════════════════════════════════════════════════════════════

STATE_FILE = os.path.abspath("bot_state_v14_4.json")
BACKUP_DIR = "backups"
ROLLING_WINDOW = 200

# ════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def is_bot_running():
    """Check if monster_engine.py is running"""
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            if proc.info['cmdline']:
                cmdline = ' '.join(proc.info['cmdline'])
                if 'monster_engine.py' in cmdline and 'python' in cmdline.lower():
                    return True, proc.info['pid']
    except Exception as e:
        st.sidebar.warning(f"Process check error: {e}")
    return False, None

def kill_bot(pid):
    """Stop the bot process"""
    try:
        process = psutil.Process(pid)
        process.terminate()
        process.wait(timeout=5)
        return True, "Bot terminated successfully"
    except psutil.TimeoutExpired:
        try:
            process.kill()
            return True, "Bot force killed"
        except Exception as e:
            return False, f"Kill failed: {e}"
    except Exception as e:
        return False, f"Error: {e}"

def load_data():
    """Load bot state with proper exception handling"""
    if not os.path.exists(STATE_FILE):
        return None
    
    try:
        with open(STATE_FILE, "r", encoding='utf-8') as f:
            content = f.read().strip()
            
            if not content:
                return None
            
            data = json.loads(content)
            
            if data.get('current_price', 0) == 0 and data.get('trade_history'):
                try:
                    last_trade = data['trade_history'][0]
                    exit_price_str = last_trade.get('exit_price', '$0.00')
                    data['current_price'] = float(exit_price_str.replace('$', '').replace(',', ''))
                except:
                    data['current_price'] = 0
            
            return data
            
    except json.JSONDecodeError:
        st.sidebar.error("⚠️ JSON file corrupted (Engine is writing...)")
        return None
    except Exception as e:
        st.sidebar.error(f"⚠️ Load error: {e}")
        return None

def send_kill_signal():
    """Send kill signal via JSON state file"""
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
            
            state['bot_status'] = 'Kill Signal Received'
            state['should_stop'] = True
            
            with open(STATE_FILE, 'w') as f:
                json.dump(state, f, indent=2)
            
            return True
    except:
        return False

def backup_state():
    """Create timestamped backup of current state"""
    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)
    
    if os.path.exists(STATE_FILE):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = os.path.join(BACKUP_DIR, f"bot_state_{timestamp}.json")
        try:
            import shutil
            shutil.copy(STATE_FILE, backup_file)
            return True, backup_file
        except Exception as e:
            return False, str(e)
    return False, "State file not found"

def calculate_total_pnl(trade_history):
    """Calculate cumulative PnL from trade history"""
    total_pnl = 0
    for trade in trade_history:
        try:
            pnl_str = trade.get('dollar_pnl', '$0.00')
            pnl_value = float(pnl_str.replace('$', '').replace(',', ''))
            total_pnl += pnl_value
        except:
            pass
    return total_pnl

def get_system_stats():
    """Get CPU and RAM usage"""
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        return cpu_percent, memory.percent
    except:
        return 0, 0

def parse_ai_confidence(data):
    """Extract AI confidence scores from latest prediction"""
    try:
        latest_probs = data.get('latest_ai_probs', {})
        if latest_probs:
            return (
                latest_probs.get('neutral', 0.33),
                latest_probs.get('buy', 0.33),
                latest_probs.get('sell', 0.33)
            )
        return (0.33, 0.33, 0.33)
    except:
        return (0.33, 0.33, 0.33)

def create_price_chart_with_signals(data, history):
    """Create Plotly chart with Cyberpunk theme and clear markers"""
    try:
        recent_trades = history[:min(20, len(history))]
        
        if not recent_trades:
            return None
        
        fig = go.Figure()
        
        timestamps = []
        entry_prices = []
        exit_prices = []
        sides = []
        pnls = []
        
        for trade in reversed(recent_trades):
            try:
                entry_time_str = trade.get('entry_time', '')
                if entry_time_str:
                    timestamps.append(entry_time_str)
                
                entry_price_str = trade.get('entry_price', '$0')
                entry_price = float(entry_price_str.replace('$', '').replace(',', ''))
                entry_prices.append(entry_price)
                
                exit_price_str = trade.get('exit_price', '$0')
                exit_price = float(exit_price_str.replace('$', '').replace(',', ''))
                exit_prices.append(exit_price)
                
                sides.append(trade.get('side', 'N/A'))
                
                pnl_str = trade.get('net_pnl', '0%')
                pnl = float(pnl_str.replace('%', ''))
                pnls.append(pnl)
            except:
                continue
        
        if not entry_prices:
            return None
        
        # ✅ CYBERPUNK: Đường giá màu vàng hổ phách
        fig.add_trace(go.Scatter(
            x=list(range(len(entry_prices))),
            y=entry_prices,
            mode='lines',
            name='Price',
            line=dict(color='#ffbf00', width=3),  # Amber/Gold
            hovertemplate='Price: $%{y:,.2f}<extra></extra>'
        ))
        
        # ✅ CYBERPUNK: BUY markers - Cyan neon
        buy_indices = [i for i, side in enumerate(sides) if side == 'LONG']
        buy_prices = [entry_prices[i] for i in buy_indices]
        
        if buy_indices:
            fig.add_trace(go.Scatter(
                x=buy_indices,
                y=buy_prices,
                mode='markers',
                name='BUY Signal',
                marker=dict(
                    symbol='triangle-up',
                    size=18,
                    color='#00f2ff',  # Cyan neon
                    line=dict(color='#ffffff', width=2)
                ),
                hovertemplate='BUY @ $%{y:,.2f}<extra></extra>'
            ))
        
        # ✅ CYBERPUNK: SELL markers - Magenta neon
        sell_indices = [i for i, side in enumerate(sides) if side == 'SHORT']
        sell_prices = [entry_prices[i] for i in sell_indices]
        
        if sell_indices:
            fig.add_trace(go.Scatter(
                x=sell_indices,
                y=sell_prices,
                mode='markers',
                name='SELL Signal',
                marker=dict(
                    symbol='triangle-down',
                    size=18,
                    color='#ff00aa',  # Magenta neon
                    line=dict(color='#ffffff', width=2)
                ),
                hovertemplate='SELL @ $%{y:,.2f}<extra></extra>'
            ))
        
        # ✅ CYBERPUNK: Dark theme layout
        fig.update_layout(
            title=dict(
                text=f"Last {len(entry_prices)} Trades - Signal Analysis",
                font=dict(color='#00f2ff', size=18)
            ),
            paper_bgcolor='rgba(10,10,15,0.95)',
            plot_bgcolor='rgba(30,30,30,0.95)',
            font=dict(color='#e0e0e0', family='Consolas, monospace'),
            xaxis=dict(
                title="Trade Index",
                gridcolor='rgba(255,255,255,0.08)',
                showgrid=True,
                color='#e0e0e0'
            ),
            yaxis=dict(
                title="Price (USD)",
                gridcolor='rgba(255,255,255,0.08)',
                showgrid=True,
                color='#e0e0e0'
            ),
            hovermode='x unified',
            height=500,
            legend=dict(
                bgcolor='rgba(30,30,30,0.9)',
                bordercolor='#00f2ff',
                borderwidth=1,
                font=dict(color='#e0e0e0')
            )
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Chart error: {e}")
        return None

# ════════════════════════════════════════════════════════════════════════════
# CYBERPUNK DARK MODE CSS
# ════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    /* ═══════════════════════════════════════════════════════════════ */
    /* CYBERPUNK DARK MODE THEME                                        */
    /* Primary: Cyan Neon #00f2ff | Secondary: Magenta #bd00ff         */
    /* ═══════════════════════════════════════════════════════════════ */
    
    /* Base App Background */
    .stApp { 
        background: linear-gradient(180deg, #0a0a0f 0%, #1a1a25 100%);
        color: #e0e0e0;
    }
    
    /* Metrics - Cyberpunk Neon */
    [data-testid="stMetricValue"] {
        color: #00f2ff !important;
        text-shadow: 0 0 10px rgba(0, 242, 255, 0.5);
        font-family: 'Consolas', 'Monaco', monospace;
        font-size: 2rem !important;
        font-weight: 600;
    }
    
    [data-testid="stMetricLabel"] {
        color: #e0e0e0 !important;
        font-family: 'Consolas', monospace;
        font-size: 0.9rem;
    }
    
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(30,30,45,0.8) 0%, rgba(20,20,35,0.8)100%);
        border: 1px solid rgba(0, 242, 255, 0.2);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0, 242, 255, 0.1);
        transition: all 0.3s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        border-color: rgba(0, 242, 255, 0.5);
        box-shadow: 0 6px 30px rgba(0, 242, 255, 0.2);
        transform: translateY(-2px);
    }
    
    /* Headers - Cyan Neon */
    h1, h2, h3 {
        color: #00f2ff !important;
        text-shadow: 0 0 15px rgba(0, 242, 255, 0.4);
        font-family: 'Consolas', monospace;
        font-weight: 700;
    }
    
    h1 {
        font-size: 2.5rem !important;
    }
    
    /* Buttons - Cyberpunk Style */
    .stButton>button {
        background: linear-gradient(135deg, #00f2ff 0%, #0099ff 100%);
        color: #0a0a0f;
        border: none;
        font-weight: bold;
        padding: 12px 32px;
        border-radius: 8px;
        box-shadow: 0 4px 15px rgba(0, 242, 255, 0.3);
        transition: all 0.3s ease;
        font-family: 'Consolas', monospace;
        font-size: 0.95rem;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(0, 242, 255, 0.5);
        background: linear-gradient(135deg, #00ddee 0%, #0088dd 100%);
    }
    
    /* Kill Switch Button */
    .kill-switch>button {
        background: linear-gradient(135deg, #ff0055 0%, #cc0044 100%) !important;
        color: white !important;
        font-weight: bold;
        animation: pulse-red 2s infinite;
    }
    
    @keyframes pulse-red {
        0%, 100% { box-shadow: 0 0 15px rgba(255, 0, 85, 0.4); }
        50% { box-shadow: 0 0 30px rgba(255, 0, 85, 0.8); }
    }
    
    /* Sidebar - Dark Gradient */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a0f 0%, #1e1e2e 100%);
        border-right: 1px solid rgba(0, 242, 255, 0.15);
    }
    
    /* DataFrames - Soft Dark */
    .dataframe {
        background-color: #1e1e1e !important;
        color: #e0e0e0 !important;
        border: 1px solid rgba(0, 242, 255, 0.2) !important;
    }
    
    /* Terminal Box - Charcoal Background */
    .terminal-box {
        background: #1e1e1e;
        border: 1px solid rgba(0, 242, 255, 0.3);
        border-radius: 8px;
        padding: 20px;
        font-family: 'Consolas', 'Courier New', monospace;
        color: #e0e0e0;
        max-height: 400px;
        overflow-y: auto;
        box-shadow: 0 4px 20px rgba(0, 242, 255, 0.1);
        font-size: 0.9rem;
        line-height: 1.6;
    }
    
    /* Status Badges */
    .status-badge {
        display: inline-block;
        padding: 8px 20px;
        border-radius: 20px;
        font-weight: bold;
        font-family: 'Consolas', monospace;
        margin: 5px;
        font-size: 0.9rem;
    }
    
    .status-online {
        background: linear-gradient(135deg, #00ff88 0%, #00cc66 100%);
        color: #0a0a0f;
        box-shadow: 0 0 15px rgba(0, 255, 136, 0.5);
        animation: pulse-green 2s infinite;
    }
    
    .status-offline {
        background: linear-gradient(135deg, #ff4466 0%, #cc3355 100%);
        color: #fff;
        box-shadow: 0 0 10px rgba(255, 68, 102, 0.3);
    }
    
    @keyframes pulse-green {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    /* Info Cards - Soft Shadow */
    .info-card {
        background: linear-gradient(135deg, rgba(30,30,45,0.9) 0%, rgba(20,20,35,0.9) 100%);
        border: 1px solid rgba(0, 242, 255, 0.2);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    
    .info-card h3 {
        margin-top: 0;
        color: #bd00ff !important;
        text-shadow: 0 0 10px rgba(189, 0, 255, 0.4);
    }
    
    /* Code Blocks */
    code {
        background: #1e1e1e !important;
        color: #00f2ff !important;
        border: 1px solid rgba(0, 242, 255, 0.2) !important;
        font-family: 'Consolas', 'Courier New', monospace;
        padding: 2px 6px;
        border-radius: 4px;
    }
    
    /* Progress Bars */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #00f2ff 0%, #bd00ff 100%);
        box-shadow: 0 0 10px rgba(0, 242, 255, 0.5);
    }
    
    /* Links */
    a {
        color: #00f2ff !important;
        text-decoration: none;
        transition: color 0.3s ease;
    }
    
    a:hover {
        color: #bd00ff !important;
        text-shadow: 0 0 8px rgba(189, 0, 255, 0.6);
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1a1a;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #00f2ff 0%, #bd00ff 100%);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #00ddee 0%, #aa00ee 100%);
    }
    
    /* Caption Text */
    .caption {
        color: #999999;
        font-size: 0.85rem;
        font-family: 'Consolas', monospace;
    }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR - CONTROL PANEL
# ════════════════════════════════════════════════════════════════════════════

st.sidebar.markdown("# ⚡ CONTROL PANEL")
st.sidebar.markdown("---")

bot_running, bot_pid = is_bot_running()

if bot_running:
    st.sidebar.markdown(
        '<div class="status-badge status-online">🟢 ENGINE ONLINE</div>',
        unsafe_allow_html=True
    )
    st.sidebar.caption(f"PID: {bot_pid}")
else:
    st.sidebar.markdown(
        '<div class="status-badge status-offline">🔴 ENGINE OFFLINE</div>',
        unsafe_allow_html=True
    )

st.sidebar.markdown("---")

st.sidebar.markdown("### ⚠️ EMERGENCY CONTROLS")

col_kill1, col_kill2 = st.sidebar.columns(2)

with col_kill1:
    if st.button("🛑 KILL SWITCH", key="kill_btn", help="Emergency stop"):
        if bot_running:
            signal_sent = send_kill_signal()
            success, msg = kill_bot(bot_pid)
            
            if success:
                st.sidebar.success(f"✅ {msg}")
            else:
                st.sidebar.error(f"❌ {msg}")
            
            time.sleep(1)
            st.rerun()
        else:
            st.sidebar.warning("Bot is not running")

with col_kill2:
    if st.button("🔄 REFRESH", key="refresh_btn"):
        st.rerun()

st.sidebar.markdown("---")

st.sidebar.markdown("### 💾 BACKUP")
if st.sidebar.button("Create Backup"):
    success, result = backup_state()
    if success:
        st.sidebar.success(f"✅ Backup created:\n{os.path.basename(result)}")
    else:
        st.sidebar.error(f"❌ Backup failed: {result}")

st.sidebar.markdown("---")

st.sidebar.markdown("### ⚙️ SETTINGS")
auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
if auto_refresh:
    refresh_interval = st.sidebar.slider("Refresh Interval (s)", 3, 30, 5)
else:
    refresh_interval = 999999

st.sidebar.markdown("---")

st.sidebar.markdown("### 📁 FILE INFO")
st.sidebar.caption(f"State File: `{os.path.basename(STATE_FILE)}`")
if os.path.exists(STATE_FILE):
    file_size = os.path.getsize(STATE_FILE)
    file_modified = datetime.fromtimestamp(os.path.getmtime(STATE_FILE))
    st.sidebar.caption(f"Size: {file_size} bytes")
    st.sidebar.caption(f"Modified: {file_modified.strftime('%H:%M:%S')}")
else:
    st.sidebar.caption("❌ File not found")

# ════════════════════════════════════════════════════════════════════════════
# MAIN DASHBOARD
# ════════════════════════════════════════════════════════════════════════════

st.markdown("# ⚡ MONSTER MATRIX v17.0")
st.markdown("### Cyberpunk Trading Command Center")

data = load_data()

if data:
    cpu_usage, ram_usage = get_system_stats()
    
    current_price = data.get('current_price', 0)
    history = data.get('trade_history', [])
    open_trades = data.get('open_trades', [])
    pending_orders = data.get('pending_orders', [])
    regime = data.get('latest_regime', 'UNKNOWN')
    
    total_pnl = calculate_total_pnl(history)
    
    if history:
        wins = len([t for t in history if float(str(t.get('net_pnl', '0%')).replace('%', '')) > 0])
        wr = (wins / len(history)) * 100 if len(history) > 0 else 0
    else:
        wins = 0
        wr = 0
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 4 METRIC BLOCKS
    # ═══════════════════════════════════════════════════════════════════════════
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="💰 BTC Price",
            value=f"${current_price:,.2f}" if current_price > 0 else "Loading...",
            delta=None
        )
    
    with col2:
        pnl_color = "normal" if total_pnl >= 0 else "inverse"
        st.metric(
            label="📊 Total PnL",
            value=f"${total_pnl:,.2f}",
            delta=f"{wr:.1f}% Win Rate",
            delta_color=pnl_color
        )
    
    with col3:
        regime_emoji = "📈" if regime == "TRENDING" else "↔️" if regime == "SIDEWAY" else "❓"
        st.metric(
            label="🎯 Market Regime",
            value=f"{regime_emoji} {regime}",
            delta=f"{len(open_trades)} Open"
        )
    
    with col4:
        st.metric(
            label="📦 Order Status",
            value=f"{len(open_trades)} Open",
            delta=f"{len(pending_orders)} Pending"
        )
    
    st.markdown("---")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ✅ TRADINGVIEW CHART (Restored)
    # ═══════════════════════════════════════════════════════════════════════════
    
    st.markdown("## 📈 LIVE MARKET - TRADINGVIEW")
    
    tv_html = """
    <div style="height:500px; border: 1px solid rgba(0, 242, 255, 0.3); border-radius: 10px; overflow: hidden; box-shadow: 0 4px 20px rgba(0, 242, 255, 0.15);">
        <div id="tv_chart" style="height:100%;"></div>
        <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
        <script type="text/javascript">
        new TradingView.widget({
            "autosize": true,
            "symbol": "KRAKEN:BTCUSDT",
            "interval": "15",
            "timezone": "Etc/UTC",
            "theme": "dark",
            "style": "1",
            "locale": "en",
            "toolbar_bg": "#1e1e1e",
            "enable_publishing": false,
            "allow_symbol_change": true,
            "container_id": "tv_chart",
            "studies": [
                "RSI@tv-basicstudies",
                "MACD@tv-basicstudies",
                "BB@tv-basicstudies"
            ],
            "backgroundColor": "#1a1a25",
            "gridColor": "rgba(255, 255, 255, 0.06)",
            "hide_side_toolbar": false
        });
        </script>
    </div>
    """
    components.html(tv_html, height=520)
    
    st.markdown("---")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # AI CONFIDENCE ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════
    
    st.markdown("## 🤖 AI CONFIDENCE ANALYSIS")
    
    prob_neutral, prob_buy, prob_sell = parse_ai_confidence(data)
    
    fig_ai = go.Figure()
    
    fig_ai.add_trace(go.Bar(
        y=['NEUTRAL', 'BUY', 'SELL'],
        x=[prob_neutral * 100, prob_buy * 100, prob_sell * 100],
        orientation='h',
        marker=dict(
            color=['#ffbf00', '#00f2ff', '#ff00aa'],
            line=dict(color='rgba(255,255,255,0.3)', width=1)
        ),
        text=[f"{prob_neutral*100:.1f}%", f"{prob_buy*100:.1f}%", f"{prob_sell*100:.1f}%"],
        textposition='auto',
        textfont=dict(color='#0a0a0f', size=12, family='Consolas'),
        hovertemplate='%{y}: %{x:.2f}%<extra></extra>'
    ))
    
    fig_ai.update_layout(
        title=dict(
            text="AI Model Confidence (Latest Prediction)",
            font=dict(color='#00f2ff', size=16)
        ),
        paper_bgcolor='rgba(10,10,15,0.95)',
        plot_bgcolor='rgba(30,30,30,0.95)',
        font=dict(color='#e0e0e0', family='Consolas'),
        xaxis=dict(
            title="Probability (%)",
            gridcolor='rgba(255,255,255,0.08)',
            range=[0, 100],
            color='#e0e0e0'
        ),
        yaxis=dict(
            title="",
            gridcolor='rgba(255,255,255,0.08)',
            color='#e0e0e0'
        ),
        height=250,
        showlegend=False
    )
    
    st.plotly_chart(fig_ai, use_container_width=True)
    
    st.markdown("---")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PRICE CHART WITH SIGNALS (Cyberpunk Style)
    # ═══════════════════════════════════════════════════════════════════════════
    
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.markdown("## 📊 TRADING SIGNALS")
        
        if history and len(history) > 0:
            chart_fig = create_price_chart_with_signals(data, history)
            if chart_fig:
                st.plotly_chart(chart_fig, use_container_width=True)
            else:
                st.info("Chart generation in progress...")
        else:
            st.info("⏳ Waiting for trade history...")
        
        st.markdown(
            f'<p class="caption">📊 Rolling Window: {ROLLING_WINDOW} candles | Showing last {min(20, len(history))} trades</p>',
            unsafe_allow_html=True
        )
    
    with col_right:
        st.markdown("## 📈 ANALYTICS OVERVIEW")
        
        if history:
            losses = len(history) - wins
            
            st.markdown(f"""
            <div class="info-card">
                <h3>📈 PERFORMANCE METRICS</h3>
                <p><strong>Total Trades:</strong> {len(history)}</p>
                <p><strong>Wins:</strong> {wins} 🟢 | <strong>Losses:</strong> {losses} 🔴</p>
                <p><strong>Win Rate:</strong> {wr:.1f}%</p>
                <p><strong>Total PnL:</strong> ${total_pnl:,.2f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            pnl_values = []
            for trade in history[:10]:
                try:
                    pnl = float(str(trade.get('net_pnl', '0%')).replace('%', ''))
                    pnl_values.append(pnl)
                except:
                    pass
            
            if pnl_values:
                fig_pnl = go.Figure()
                colors = ['#00ff88' if x > 0 else '#ff4466' for x in pnl_values]
                
                fig_pnl.add_trace(go.Bar(
                    y=pnl_values,
                    marker_color=colors,
                    name='PnL %',
                    text=[f"{x:.1f}%" for x in pnl_values],
                    textposition='auto',
                    textfont=dict(color='#ffffff', size=11)
                ))
                
                fig_pnl.update_layout(
                    title=dict(
                        text="Last 10 Trades - PnL Distribution",
                        font=dict(color='#00f2ff', size=14)
                    ),
                    paper_bgcolor='rgba(10,10,15,0.95)',
                    plot_bgcolor='rgba(30,30,30,0.95)',
                    font=dict(color='#e0e0e0', family='Consolas'),
                    height=300,
                    showlegend=False,
                    xaxis=dict(
                        title="Trade Index",
                        gridcolor='rgba(255,255,255,0.08)',
                        color='#e0e0e0'
                    ),
                    yaxis=dict(
                        title="PnL (%)",
                        gridcolor='rgba(255,255,255,0.08)',
                        color='#e0e0e0'
                    )
                )
                
                st.plotly_chart(fig_pnl, use_container_width=True)
        else:
            st.info("No trade data available yet")
    
    st.markdown("---")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TERMINAL OUTPUT
    # ═══════════════════════════════════════════════════════════════════════════
    
    st.markdown("## 💻 SYSTEM TERMINAL")
    
    last_update = data.get('last_update_time', 'N/A')
    try:
        if last_update != 'N/A':
            dt = datetime.fromisoformat(last_update.replace('Z', '+00:00'))
            dt_gmt7 = dt + timedelta(hours=7)
            last_update_display = dt_gmt7.strftime('%Y-%m-%d %H:%M:%S GMT+7')
        else:
            last_update_display = 'N/A'
    except:
        last_update_display = last_update
    
    terminal_lines = [
        "╔════════════════════════════════════════════════════════════════╗",
        "║  MONSTER ENGINE v17.0 - CYBERPUNK EDITION                      ║",
        "╚════════════════════════════════════════════════════════════════╝",
        "",
        f"[ENGINE]  Status: {data.get('bot_status', 'Unknown')}",
        f"[PRICE]   BTC/USDT: ${current_price:,.2f}",
        f"[REGIME]  Market Mode: {regime}",
        f"[TRADES]  Open Positions: {len(open_trades)}",
        f"[ORDERS]  Pending Limit: {len(pending_orders)}",
        f"[STATS]   Total Trades: {len(history)}",
        f"[STATS]   Win Rate: {wr:.1f}%",
        f"[STATS]   Total PnL: ${total_pnl:,.2f}",
        f"[SYSTEM]  CPU: {cpu_usage:.1f}% | RAM: {ram_usage:.1f}%",
        f"[TIME]    Last Update: {last_update_display}",
        f"[CONFIG]  Rolling Window: {ROLLING_WINDOW} candles",
        f"[FILE]    State: {os.path.basename(STATE_FILE)}",
        "",
        "════════════════════════════════════════════════════════════════",
        "✅ All systems operational. Monitoring active.",
        "════════════════════════════════════════════════════════════════"
    ]
    
    terminal_output = "\n".join(terminal_lines)
    
    st.markdown(f"""
    <div class="terminal-box">
        <pre>{terminal_output}</pre>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TRADE HISTORY TABLE
    # ═══════════════════════════════════════════════════════════════════════════
    
    st.markdown("## 📜 TRADE HISTORY")
    
    if history:
        df_history = pd.DataFrame(history[:15])
        
        st.dataframe(
            df_history,
            use_container_width=True,
            height=400
        )
        
        csv = df_history.to_csv(index=False)
        st.download_button(
            label="📥 Download Full History (CSV)",
            data=csv,
            file_name=f"monster_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("⏳ No trade history available yet. Waiting for first trade...")
    
    st.markdown("---")
    
    st.markdown(f"""
    <div style="text-align: center; color: #00f2ff; font-family: 'Consolas'; padding: 20px;">
        <p style="font-size: 1.2rem;">⚡ MONSTER MATRIX v17.0 - CYBERPUNK EDITION</p>
        <p style="font-size: 0.9rem; color: #bd00ff;">Engine PID: {bot_pid if bot_running else 'N/A'} | Refresh: {refresh_interval}s</p>
        <p style="font-size: 0.8rem; color: #999999;">State: {os.path.basename(STATE_FILE)}</p>
    </div>
    """, unsafe_allow_html=True)

else:
    # ═══════════════════════════════════════════════════════════════════════════
    # WAITING FOR DATA
    # ═══════════════════════════════════════════════════════════════════════════
    
    st.warning("📡 Waiting for data from Monster Engine...")
    
    st.markdown(f"""
    <div class="info-card">
        <h3>🔍 TROUBLESHOOTING</h3>
        <p>If you're seeing this message:</p>
        <ol>
            <li>✅ Verify <code>monster_engine.py</code> is running</li>
            <li>✅ Check that <code>bot_state_v14_4.json</code> exists</li>
            <li>✅ Wait 10-15 seconds for first data collection</li>
            <li>✅ Check Engine terminal for errors</li>
        </ol>
        <p><strong>State File Path:</strong> <code>{STATE_FILE}</code></p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.spinner("Initializing system..."):
        time.sleep(2)

# ════════════════════════════════════════════════════════════════════════════
# ✅ AUTO-REFRESH AT END (Critical: Must be last to avoid interrupting widgets)
# ════════════════════════════════════════════════════════════════════════════

if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()
