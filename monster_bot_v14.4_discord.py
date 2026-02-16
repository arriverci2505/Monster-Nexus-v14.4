import streamlit as st
import pandas as pd
import json
import os
import time
import ccxt
import numpy as np
from datetime import datetime
import streamlit.components.v1 as components
import subprocess

# --- ĐOẠN CODE KÍCH HOẠT BOT NGẦM ---
def start_engine():
    # Kiểm tra xem Bot đã chạy chưa (dựa vào file bot_state.json)
    # Hoặc dùng một biến session_state để tránh chạy đè nhiều con Bot
    if 'engine_started' not in st.session_state:
        st.write("⏳ Đang khởi động Monster Engine ngầm...")
        # Lệnh này sẽ chạy file monster_engine.py như một tiến trình riêng
        subprocess.Popen(["python", "monster_engine.py"])
        st.session_state['engine_started'] = True
        time.sleep(2) # Đợi 2 giây để Bot kịp tạo file json

start_engine()

# Cấu hình trang
st.set_page_config(page_title="MONSTER MATRIX UI v14.4", layout="wide")

# Matrix Style CSS (Giữ nguyên của bạn)
st.markdown("""
<style>
    .stApp { background-color: #0d1117; }
    [data-testid="stMetricValue"] {
        color: #00ff41 !important;
        text-shadow: 0 0 10px #00ff41;
        font-family: 'Courier New', monospace;
    }
    div[data-testid="metric-container"] {
        background-color: #0a0e17;
        border: 1px solid #00ff4133;
        padding: 20px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

STATE_FILE = "bot_state.json"

def load_data():
    if os.path.exists(STATE_FILE):
        try:
            # Kiểm tra kích thước file để tránh đọc file đang ghi dở
            if os.path.getsize(STATE_FILE) > 0:
                with open(STATE_FILE, "r") as f:
                    return json.load(f)
        except:
            return None
    return None

# Tiêu đề Header
st.markdown("<h1 style='text-align: center; color: #00ff41; text-shadow: 0 0 20px #00ff41;'>👾 MONSTER NEXUS COMMAND CENTER</h1>", unsafe_allow_html=True)

data = load_data()

if data:
    col_left, col_right = st.columns([1, 2.5])

    with col_left:
        st.markdown("<h3 style='color:#00ff41;'>SYSTEM ANALYTICS</h3>", unsafe_allow_html=True)
        
        # Hiển thị các chỉ số
        st.metric("CURRENT PRICE", f"${data.get('current_price', 0):,.2f}")
        
        # Tính toán Winrate nếu bot_state chưa có sẵn key win_rate
        wr = data.get('win_rate')
        if wr is None:
            history = data.get('trade_history', [])
            wins = len([t for t in history if float(str(t.get('net_pnl', 0)).replace('%','')) > 0])
            wr = (wins / len(history) * 100) if history else 0
            
        st.metric("WIN RATE", f"{wr:.1f}%")
        st.metric("ACCOUNT BALANCE", f"${data.get('balance', 0):,.2f}")
        
        st.markdown("---")
        status = "ONLINE" if data.get('bot_status') != "Stopped" else "OFFLINE"
        st.markdown(f"**STATUS:** <span style='color:#00ff41'>{status}</span>", unsafe_allow_html=True)
        st.markdown(f"**LAST SYNC:** {data.get('last_update_time', 'N/A')}")
        
        st.markdown("### 📜 RECENT TRADES")
        if 'trade_history' in data and data['trade_history']:
            df_history = pd.DataFrame(data['trade_history'])
            st.dataframe(df_history.head(10), use_container_width=True)
        else:
            st.info("Chưa có lịch sử giao dịch.")

    with col_right:
        # TradingView Chart (BTC/USDT)
        tv_html = """
        <div style="height:600px; border: 1px solid #00ff4144;">
            <div id="tv_chart" style="height:100%;"></div>
            <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
            <script type="text/javascript">
            new TradingView.widget({
                "autosize": true, "symbol": "BINANCE:BTCUSDT",
                "interval": "15", "theme": "dark", "style": "1",
                "locale": "en", "container_id": "tv_chart"
            });
            </script>
        </div>
        """
        components.html(tv_html, height=620)
else:
    st.warning("📡 Đang đợi file `bot_state.json` từ Engine... Hãy chắc chắn Bot chính đang chạy.")

# Cơ chế Auto-refresh thông minh
time.sleep(5)
st.rerun()
