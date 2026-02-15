"""
╔══════════════════════════════════════════════════════════════════════════╗
║  MONSTER ENGINE v14.4 - HEADLESS BACKEND WORKER                          ║
║  Cloud-Optimized Trading Engine with Smart Exit & Discord Alerts         ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import ccxt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import time
import json
import gc
import requests
import warnings
import logging
from datetime import datetime, timedelta
from scipy import signal as scipy_signal

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Execution parameters
SLIPPAGE = 0.0005  # 0.05% slippage
COMMISSION = 0.00075  # 0.075% commission per trade

# ⚙️ LIVE_CONFIG - 27 MANDATORY PARAMETERS
LIVE_CONFIG = {
    # --- 1. GENERAL ---
    'exchange': 'kraken',
    'symbol': 'BTC/USDT',
    'timeframe': '15m',
    'sequence_length': 30,

    # --- 2. AI THRESHOLDS ---
    'temperature': 1.2,
    'entry_percentile': 25,
    'trending_buy_threshold': 0.40,
    'trending_sell_threshold': 0.42,
    'sideway_buy_threshold': 0.22,
    'sideway_sell_threshold': 0.22,

    # --- 3. REGIME CLASSIFICATION ---
    'trending_adx_min': 30,             # BTC Golden Ratio
    'sideway_adx_max': 30,              # BTC Golden Ratio
    'choppiness_threshold_high': 58.0,
    'choppiness_extreme_low': 30,

    # --- 4. SIDEWAY FILTERS ---
    'deviation_zscore_threshold': 1.4,       # Sensitive Entry
    'mean_reversion_min_shadow_atr': 0.1,    # Low shadow requirement
    'bb_squeeze_percentile': 0.35,

    # --- 5. TRENDING PARAMETERS ---
    'sl_std_multiplier': 1.5,
    'max_holding_bars': 200,

    # --- 6. SIDEWAY EXIT PARAMETERS ---
    'mean_reversion_sl_pct': 1.0,
    'mean_reversion_tp_pct': 3.5,            # High reward target
    'time_barrier': 20,
    'min_profit_for_target': 0.009,
    'limit_order_offset': 0.001,             # 0.1% better price

    # --- 7. RISK MANAGEMENT (SMART EXIT) ---
    'use_advanced_exit': True,
    'use_profit_lock': True,
    'ai_exit_threshold': 0.70,
    'profit_lock_levels': [(1.8, 1.2), (3.5, 2.8), (5.5, 4.5)], # Tier 1, 2, 3
    'trailing_stop_activation': 1.5,
    'trailing_stop_distance': 0.6,

    # --- 8. EXECUTION ---
    'position_size': 0.15,
    'slippage': 0.0005,
    'commission': 0.00075
}

# Discord Webhook (set via environment variable or edit here)
DISCORD_WEBHOOK = ""  # Set your webhook URL here or via env var

# State file
STATE_FILE = "bot_state.json"

# ════════════════════════════════════════════════════════════════════════════
# PYTORCH MODEL ARCHITECTURE
# ════════════════════════════════════════════════════════════════════════════

class HybridTransformerLSTM(nn.Module):
    """
    Hybrid architecture combining Transformer and LSTM for time series prediction
    """
    def __init__(self, input_dim, hidden_dim, num_lstm_layers, num_transformer_layers, num_heads, num_classes):
        super(HybridTransformerLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_lstm_layers = num_lstm_layers
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding for Transformer
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_lstm_layers,
            batch_first=True,
            dropout=0.1 if num_lstm_layers > 1 else 0
        )
        
        # Output layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_dim // 2, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Input projection
        x = self.input_proj(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Use last time step
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layers
        out = self.relu(self.fc1(last_output))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# ════════════════════════════════════════════════════════════════════════════
# TECHNICAL INDICATORS
# ════════════════════════════════════════════════════════════════════════════

def calculate_technical_indicators(df):
    """
    Calculate comprehensive technical indicators for the trading strategy
    Preserves exact logic from source file
    """
    df = df.copy()
    
    # Price-based indicators
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # Moving Averages
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['sma_200'] = df['close'].rolling(window=200).mean()
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    
    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    
    # ATR (Average True Range)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['atr'] = true_range.rolling(14).mean()
    
    # ADX (Average Directional Index)
    plus_dm = df['high'].diff()
    minus_dm = -df['low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    tr = true_range
    atr14 = tr.rolling(14).mean()
    
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr14)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr14)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    df['adx'] = dx.rolling(14).mean()
    
    # Stochastic Oscillator
    low_14 = df['low'].rolling(window=14).min()
    high_14 = df['high'].rolling(window=14).max()
    df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
    
    # OBV (On-Balance Volume)
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    
    # Volume indicators
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    # Volatility
    df['volatility'] = df['returns'].rolling(window=20).std()
    
    # Price momentum
    df['momentum'] = df['close'] - df['close'].shift(10)
    
    # Choppiness Index
    atr_sum = df['atr'].rolling(window=14).sum()
    high_low_range = df['high'].rolling(window=14).max() - df['low'].rolling(window=14).min()
    df['choppiness'] = 100 * np.log10(atr_sum / high_low_range) / np.log10(14)
    
    # Price position relative to bands
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # Candle patterns
    df['body_size'] = np.abs(df['close'] - df['open'])
    df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
    df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
    
    return df

# ════════════════════════════════════════════════════════════════════════════
# FEATURE PREPARATION
# ════════════════════════════════════════════════════════════════════════════

def prepare_features(df, sequence_length=30):
    """
    Prepare features for model input
    Preserves exact logic from source file
    """
    feature_columns = [
        'returns', 'log_returns', 'rsi', 'macd', 'macd_signal', 'macd_histogram',
        'bb_width', 'bb_position', 'atr', 'adx', 'stoch_k', 'stoch_d',
        'volume_ratio', 'volatility', 'momentum', 'choppiness',
        'body_size', 'upper_shadow', 'lower_shadow'
    ]
    
    # Add price ratios
    df['price_to_sma20'] = df['close'] / df['sma_20']
    df['price_to_sma50'] = df['close'] / df['sma_50']
    df['price_to_sma200'] = df['close'] / df['sma_200']
    df['sma20_to_sma50'] = df['sma_20'] / df['sma_50']
    df['sma50_to_sma200'] = df['sma_50'] / df['sma_200']
    
    feature_columns.extend([
        'price_to_sma20', 'price_to_sma50', 'price_to_sma200',
        'sma20_to_sma50', 'sma50_to_sma200'
    ])
    
    # Add EMA crossovers
    df['ema_cross'] = df['ema_12'] - df['ema_26']
    df['ema_cross_signal'] = df['ema_cross'].rolling(window=3).mean()
    
    feature_columns.extend(['ema_cross', 'ema_cross_signal'])
    
    # Cyclical time features
    df['hour'] = pd.to_datetime(df.index).hour if isinstance(df.index, pd.DatetimeIndex) else 0
    df['day_of_week'] = pd.to_datetime(df.index).dayofweek if isinstance(df.index, pd.DatetimeIndex) else 0
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    feature_columns.extend(['hour_sin', 'hour_cos', 'day_sin', 'day_cos'])
    
    # Handle missing values
    df[feature_columns] = df[feature_columns].fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    # Normalize features
    features_array = df[feature_columns].values
    
    # Z-score normalization (per feature)
    mean = np.mean(features_array, axis=0)
    std = np.std(features_array, axis=0)
    std[std == 0] = 1  # Avoid division by zero
    features_normalized = (features_array - mean) / std
    
    # Create sequences
    sequences = []
    for i in range(len(features_normalized) - sequence_length + 1):
        seq = features_normalized[i:i + sequence_length]
        sequences.append(seq)
    
    return np.array(sequences)

# ════════════════════════════════════════════════════════════════════════════
# DISCORD NOTIFICATIONS
# ════════════════════════════════════════════════════════════════════════════

def send_discord_alert(webhook_url, title, color, fields):
    """
    Send professional Discord webhook with Embed format
    
    Args:
        webhook_url: Discord webhook URL
        title: Embed title
        color: Color code (0x00ff00 for green, 0xff0000 for red, 0xffd700 for gold)
        fields: List of dicts with 'name', 'value', 'inline' keys
    """
    if not webhook_url:
        return
    
    data = {
        "embeds": [{
            "title": title,
            "color": color,
            "fields": fields,
            "footer": {"text": f"Monster Bot v14.4 • {datetime.now().strftime('%Y-%m-%d %H:%M')}"}
        }]
    }
    
    try:
        response = requests.post(webhook_url, json=data, timeout=5)
        if response.status_code == 204:
            logger.info("Discord alert sent successfully")
        else:
            logger.warning(f"Discord webhook returned status {response.status_code}")
    except Exception as e:
        logger.error(f"Discord Error: {e}")

# ════════════════════════════════════════════════════════════════════════════
# STATE MANAGEMENT
# ════════════════════════════════════════════════════════════════════════════

def load_state():
    """Load bot state from JSON file"""
    try:
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)
        logger.info("State loaded successfully")
        return state
    except FileNotFoundError:
        logger.info("No previous state found, initializing new state")
        return {
            'balance': 10000.0,
            'open_trades': [],
            'trade_history': [],
            'last_update_time': None,
            'bot_status': 'Initializing',
            'pending_orders': []
        }
    except Exception as e:
        logger.error(f"Error loading state: {e}")
        return None

def save_state(state):
    """Save bot state to JSON file"""
    try:
        state['last_update_time'] = datetime.now().isoformat()
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        logger.debug("State saved successfully")
    except Exception as e:
        logger.error(f"Error saving state: {e}")

# ════════════════════════════════════════════════════════════════════════════
# EXECUTION LOGIC
# ════════════════════════════════════════════════════════════════════════════

def calculate_actual_entry_price(market_price, side):
    """Calculate actual entry price with slippage"""
    if side == 'BUY':
        return market_price * (1 + SLIPPAGE)
    else:  # SELL
        return market_price * (1 - SLIPPAGE)

def check_limit_order_fill(pending_order, current_price):
    """Check if limit order should be filled"""
    limit_price = pending_order['limit_price']
    side = pending_order['side']
    
    if side == 'BUY':
        # Buy limit fills when price drops to or below limit
        return current_price <= limit_price
    else:  # SELL
        # Sell limit fills when price rises to or above limit
        return current_price >= limit_price

def calculate_pnl(entry_price, exit_price, side):
    """Calculate PnL percentage with fees"""
    if side == 'BUY':
        gross_pnl = (exit_price - entry_price) / entry_price
    else:  # SELL
        gross_pnl = (entry_price - exit_price) / entry_price
    
    # Subtract commission (entry + exit)
    net_pnl = gross_pnl - (2 * COMMISSION)
    
    return net_pnl, gross_pnl

def check_smart_exit(trade, current_price, indicators):
    """
    Implement Smart Exit Logic with Profit Locks and Trailing Stop
    
    Returns: (should_exit, reason, new_trailing_stop)
    """
    entry_price = trade['entry_price']
    side = trade['side']
    entry_time = datetime.fromisoformat(trade['entry_time'])
    
    # Calculate current PnL
    net_pnl, gross_pnl = calculate_pnl(entry_price, current_price, side)
    gross_pnl_pct = gross_pnl * 100
    
    # Traditional SL/TP check first
    if side == 'BUY':
        if current_price <= trade['stop_loss']:
            return True, 'STOP_LOSS', None
        if current_price >= trade['take_profit']:
            return True, 'TAKE_PROFIT', None
    else:  # SELL
        if current_price >= trade['stop_loss']:
            return True, 'STOP_LOSS', None
        if current_price <= trade['take_profit']:
            return True, 'TAKE_PROFIT', None
    
    # Smart Exit features
    if LIVE_CONFIG['use_advanced_exit']:
        
        # 1. PROFIT LOCK SYSTEM
        if LIVE_CONFIG['use_profit_lock']:
            for trigger_pct, lock_pct in LIVE_CONFIG['profit_lock_levels']:
                if gross_pnl_pct >= trigger_pct:
                    # Check if we've fallen below lock level
                    if gross_pnl_pct < lock_pct:
                        return True, f'PROFIT_LOCK_{trigger_pct}%', None
        
        # 2. TRAILING STOP
        activation_pct = LIVE_CONFIG['trailing_stop_activation']
        distance_pct = LIVE_CONFIG['trailing_stop_distance']
        
        if gross_pnl_pct >= activation_pct:
            # Calculate trailing stop price
            if side == 'BUY':
                trailing_stop = current_price * (1 - distance_pct / 100)
                # Update trailing stop if higher than current
                if 'trailing_stop' not in trade or trailing_stop > trade['trailing_stop']:
                    return False, None, trailing_stop
                # Check if trailing stop hit
                if current_price <= trade['trailing_stop']:
                    return True, 'TRAILING_STOP', None
            else:  # SELL
                trailing_stop = current_price * (1 + distance_pct / 100)
                # Update trailing stop if lower than current
                if 'trailing_stop' not in trade or trailing_stop < trade['trailing_stop']:
                    return False, None, trailing_stop
                # Check if trailing stop hit
                if current_price >= trade['trailing_stop']:
                    return True, 'TRAILING_STOP', None
        
        # 3. TIME BARRIER (Max Holding Period)
        bars_held = trade.get('bars_held', 0)
        if bars_held >= LIVE_CONFIG['max_holding_bars']:
            return True, 'TIMEOUT', None
    
    return False, None, None

# ════════════════════════════════════════════════════════════════════════════
# MAIN TRADING ENGINE
# ════════════════════════════════════════════════════════════════════════════

def main():
    """Main trading engine loop"""
    
    logger.info("=" * 80)
    logger.info("MONSTER ENGINE v14.4 - Starting Headless Trading Worker")
    logger.info("=" * 80)
    
    # Load state
    state = load_state()
    if state is None:
        logger.error("Failed to initialize state. Exiting.")
        return
    
    state['bot_status'] = 'Initializing'
    save_state(state)
    
    # Initialize exchange
    try:
        exchange = getattr(ccxt, LIVE_CONFIG['exchange'])({
            'enableRateLimit': True,
        })
        logger.info(f"Connected to {LIVE_CONFIG['exchange']}")
    except Exception as e:
        logger.error(f"Failed to connect to exchange: {e}")
        state['bot_status'] = 'Error: Exchange Connection Failed'
        save_state(state)
        return
    
    # Load PyTorch model
    logger.info("Loading PyTorch model...")
    try:
        model = HybridTransformerLSTM(
            input_dim=29,
            hidden_dim=256,
            num_lstm_layers=2,
            num_transformer_layers=2,
            num_heads=4,
            num_classes=3
        )
        model.eval()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        state['bot_status'] = 'Error: Model Load Failed'
        save_state(state)
        return
    
    state['bot_status'] = 'Running'
    save_state(state)
    
    # Main loop
    logger.info("Entering main trading loop...")
    
    while True:
        try:
            loop_start_time = time.time()
            
            # Fetch latest data
            try:
                ohlcv = exchange.fetch_ohlcv(
                    LIVE_CONFIG['symbol'],
                    LIVE_CONFIG['timeframe'],
                    limit=300
                )
                
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                current_price = df['close'].iloc[-1]
                
            except Exception as e:
                logger.error(f"Error fetching data: {e}")
                time.sleep(60)
                continue
            
            # Calculate indicators
            df = calculate_technical_indicators(df)
            
            # Get latest indicators
            latest = df.iloc[-1]
            adx = latest['adx']
            atr = latest['atr']
            sma200 = latest['sma_200']
            
            # Determine regime
            is_trending = adx >= LIVE_CONFIG['trending_adx_min']
            regime = 'TRENDING' if is_trending else 'SIDEWAY'
            
            # ═══════════════════════════════════════════════════════════════
            # POSITION MANAGEMENT - Check existing trades
            # ═══════════════════════════════════════════════════════════════
            
            if state['open_trades']:
                for trade in state['open_trades'][:]:  # Iterate over copy
                    # Increment bars held
                    trade['bars_held'] = trade.get('bars_held', 0) + 1
                    
                    # Check smart exit
                    should_exit, exit_reason, new_trailing = check_smart_exit(trade, current_price, latest)
                    
                    # Update trailing stop if needed
                    if new_trailing is not None:
                        trade['trailing_stop'] = new_trailing
                        logger.info(f"Trailing stop updated to {new_trailing:.2f}")
                    
                    # Execute exit if needed
                    if should_exit:
                        exit_price = calculate_actual_entry_price(current_price, 'SELL' if trade['side'] == 'BUY' else 'BUY')
                        net_pnl, gross_pnl = calculate_pnl(trade['entry_price'], exit_price, trade['side'])
                        
                        duration = datetime.now() - datetime.fromisoformat(trade['entry_time'])
                        duration_str = str(duration).split('.')[0]
                        
                        # Log to history
                        history_entry = {
                            'time': datetime.now().isoformat(),
                            'type': f"EXIT {trade['side']}",
                            'symbol': LIVE_CONFIG['symbol'],
                            'entry_price': trade['entry_price'],
                            'exit_price': exit_price,
                            'net_pnl': net_pnl * 100,
                            'gross_pnl': gross_pnl * 100,
                            'reason': exit_reason,
                            'regime': trade['regime'],
                            'duration': duration_str
                        }
                        state['trade_history'].insert(0, history_entry)
                        
                        # Update balance
                        state['balance'] *= (1 + net_pnl)
                        
                        # Remove from open trades
                        state['open_trades'].remove(trade)
                        
                        # Discord notification
                        color = 0x00ff00 if net_pnl > 0 else 0xff0000
                        send_discord_alert(
                            DISCORD_WEBHOOK,
                            f"{'💰 WIN' if net_pnl > 0 else '💀 LOSS'}: {LIVE_CONFIG['symbol']} {trade['side']}",
                            color,
                            [
                                {"name": "Symbol", "value": LIVE_CONFIG['symbol'], "inline": True},
                                {"name": "Side", "value": trade['side'], "inline": True},
                                {"name": "Entry Price", "value": f"${trade['entry_price']:,.2f}", "inline": True},
                                {"name": "Exit Price", "value": f"${exit_price:,.2f}", "inline": True},
                                {"name": "Net PNL %", "value": f"{net_pnl*100:+.2f}%", "inline": True},
                                {"name": "Regime", "value": trade['regime'], "inline": True},
                                {"name": "Exit Reason", "value": exit_reason, "inline": False},
                                {"name": "Duration", "value": duration_str, "inline": True}
                            ]
                        )
                        
                        logger.info(f"Position closed: {trade['side']} @ ${exit_price:.2f} | PnL: {net_pnl*100:+.2f}% | Reason: {exit_reason}")
                        
                        save_state(state)
            
            # ═══════════════════════════════════════════════════════════════
            # PENDING ORDERS - Check limit order fills
            # ═══════════════════════════════════════════════════════════════
            
            if state['pending_orders']:
                for pending in state['pending_orders'][:]:  # Iterate over copy
                    pending['candles_waiting'] = pending.get('candles_waiting', 0) + 1
                    
                    # Check if limit order filled
                    if check_limit_order_fill(pending, current_price):
                        # Order filled - create position
                        entry_price = calculate_actual_entry_price(pending['limit_price'], pending['side'])
                        
                        trade = {
                            'side': pending['side'],
                            'entry_price': entry_price,
                            'stop_loss': pending['stop_loss'],
                            'take_profit': pending['take_profit'],
                            'entry_time': datetime.now().isoformat(),
                            'regime': regime,
                            'bars_held': 0
                        }
                        state['open_trades'].append(trade)
                        state['pending_orders'].remove(pending)
                        
                        # Discord notification
                        color = 0x00ff00 if pending['side'] == 'BUY' else 0xff0000
                        send_discord_alert(
                            DISCORD_WEBHOOK,
                            f"🚀 ENTRY: {LIVE_CONFIG['symbol']} {pending['side']}",
                            color,
                            [
                                {"name": "Symbol", "value": LIVE_CONFIG['symbol'], "inline": True},
                                {"name": "Side", "value": pending['side'], "inline": True},
                                {"name": "Entry Price", "value": f"${entry_price:,.2f}", "inline": True},
                                {"name": "Stop Loss", "value": f"${pending['stop_loss']:,.2f}", "inline": True},
                                {"name": "Take Profit", "value": f"${pending['take_profit']:,.2f}", "inline": True},
                                {"name": "Regime", "value": regime, "inline": True},
                                {"name": "Order Type", "value": "LIMIT (Filled)", "inline": False}
                            ]
                        )
                        
                        logger.info(f"Limit order filled: {pending['side']} @ ${entry_price:.2f}")
                        save_state(state)
                    
                    # Cancel if waited too long (2 candles)
                    elif pending['candles_waiting'] >= 2:
                        logger.info(f"Canceling pending limit order after 2 candles")
                        state['pending_orders'].remove(pending)
                        save_state(state)
            
            # ═══════════════════════════════════════════════════════════════
            # SIGNAL GENERATION - Only if no open positions
            # ═══════════════════════════════════════════════════════════════
            
            if not state['open_trades'] and not state['pending_orders']:
                # Prepare features for model
                try:
                    sequences = prepare_features(df, LIVE_CONFIG['sequence_length'])
                    if len(sequences) > 0:
                        last_sequence = sequences[-1]
                        
                        # Model prediction
                        with torch.no_grad():
                            input_tensor = torch.FloatTensor(last_sequence).unsqueeze(0)
                            output = model(input_tensor)
                            
                            # Apply temperature scaling
                            output_scaled = output / LIVE_CONFIG['temperature']
                            probabilities = torch.softmax(output_scaled, dim=1).squeeze().numpy()
                        
                        p_buy = probabilities[0]
                        p_neutral = probabilities[1]
                        p_sell = probabilities[2]
                        
                        # Determine signal based on regime
                        if is_trending:
                            buy_threshold = LIVE_CONFIG['trending_buy_threshold']
                            sell_threshold = LIVE_CONFIG['trending_sell_threshold']
                        else:
                            buy_threshold = LIVE_CONFIG['sideway_buy_threshold']
                            sell_threshold = LIVE_CONFIG['sideway_sell_threshold']
                        
                        signal = None
                        if p_buy > buy_threshold and p_buy > p_sell:
                            signal = 'BUY'
                        elif p_sell > sell_threshold and p_sell > p_buy:
                            signal = 'SELL'
                        
                        # Additional filters
                        if signal:
                            # SMA200 filter
                            if signal == 'BUY' and current_price < sma200:
                                logger.info(f"BUY signal rejected: Price below SMA200")
                                signal = None
                            elif signal == 'SELL' and current_price > sma200:
                                logger.info(f"SELL signal rejected: Price above SMA200")
                                signal = None
                        
                        # Execute signal
                        if signal:
                            logger.info(f"Signal detected: {signal} | Regime: {regime} | ADX: {adx:.1f}")
                            
                            if is_trending:
                                # MARKET ORDER - Instant execution
                                entry_price = calculate_actual_entry_price(current_price, signal)
                                
                                # Calculate SL/TP
                                sl_distance = atr * LIVE_CONFIG['sl_std_multiplier']
                                if signal == 'BUY':
                                    stop_loss = entry_price - sl_distance
                                    take_profit = entry_price + (sl_distance * 3)  # 1:3 R:R
                                else:
                                    stop_loss = entry_price + sl_distance
                                    take_profit = entry_price - (sl_distance * 3)
                                
                                trade = {
                                    'side': signal,
                                    'entry_price': entry_price,
                                    'stop_loss': stop_loss,
                                    'take_profit': take_profit,
                                    'entry_time': datetime.now().isoformat(),
                                    'regime': regime,
                                    'bars_held': 0
                                }
                                state['open_trades'].append(trade)
                                
                                # Discord notification
                                color = 0x00ff00 if signal == 'BUY' else 0xff0000
                                send_discord_alert(
                                    DISCORD_WEBHOOK,
                                    f"🚀 ENTRY: {LIVE_CONFIG['symbol']} {signal}",
                                    color,
                                    [
                                        {"name": "Symbol", "value": LIVE_CONFIG['symbol'], "inline": True},
                                        {"name": "Side", "value": signal, "inline": True},
                                        {"name": "Entry Price", "value": f"${entry_price:,.2f}", "inline": True},
                                        {"name": "Stop Loss", "value": f"${stop_loss:,.2f}", "inline": True},
                                        {"name": "Take Profit", "value": f"${take_profit:,.2f}", "inline": True},
                                        {"name": "Regime", "value": regime, "inline": True},
                                        {"name": "Order Type", "value": "MARKET", "inline": False}
                                    ]
                                )
                                
                                logger.info(f"Market order executed: {signal} @ ${entry_price:.2f}")
                                save_state(state)
                                
                            else:
                                # LIMIT ORDER - Wait for better price
                                limit_offset = LIVE_CONFIG['limit_order_offset']
                                if signal == 'BUY':
                                    limit_price = current_price * (1 - limit_offset)
                                else:
                                    limit_price = current_price * (1 + limit_offset)
                                
                                # Calculate SL/TP based on limit price
                                if signal == 'BUY':
                                    stop_loss = limit_price * (1 - LIVE_CONFIG['mean_reversion_sl_pct'] / 100)
                                    take_profit = limit_price * (1 + LIVE_CONFIG['mean_reversion_tp_pct'] / 100)
                                else:
                                    stop_loss = limit_price * (1 + LIVE_CONFIG['mean_reversion_sl_pct'] / 100)
                                    take_profit = limit_price * (1 - LIVE_CONFIG['mean_reversion_tp_pct'] / 100)
                                
                                pending_order = {
                                    'side': signal,
                                    'limit_price': limit_price,
                                    'stop_loss': stop_loss,
                                    'take_profit': take_profit,
                                    'candles_waiting': 0
                                }
                                state['pending_orders'].append(pending_order)
                                
                                logger.info(f"Limit order placed: {signal} @ ${limit_price:.2f} (Market: ${current_price:.2f})")
                                save_state(state)
                
                except Exception as e:
                    logger.error(f"Error in signal generation: {e}")
            
            # ═══════════════════════════════════════════════════════════════
            # CLEANUP & SLEEP
            # ═══════════════════════════════════════════════════════════════
            
            # Memory cleanup
            gc.collect()
            
            # Calculate sleep time to next candle
            loop_duration = time.time() - loop_start_time
            sleep_time = max(1, 60 - loop_duration)
            
            logger.info(f"Cycle complete | Price: ${current_price:,.2f} | Regime: {regime} | Open: {len(state['open_trades'])} | Pending: {len(state['pending_orders'])}")
            time.sleep(sleep_time)
            
        except KeyboardInterrupt:
            logger.info("Shutdown signal received. Exiting...")
            state['bot_status'] = 'Stopped'
            save_state(state)
            break
        
        except Exception as e:
            logger.error(f"Critical error in main loop: {e}", exc_info=True)
            state['bot_status'] = f'Error: {str(e)}'
            save_state(state)
            time.sleep(60)

if __name__ == "__main__":
    main()
