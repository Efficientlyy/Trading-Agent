import os
import json
import time
import threading
import datetime
from flask import Flask, render_template, jsonify, request, send_from_directory
from mexc_api import MexcAPI
import pandas as pd
import numpy as np
from prometheus_client import Counter, Gauge, Histogram, start_http_server

# Initialize Metrics
MARKET_DATA_THROUGHPUT = Counter(
    'market_data_updates_total', 
    'Number of market data updates processed',
    ['trading_pair', 'update_type']
)

ORDER_EXECUTION_LATENCY = Histogram(
    'order_execution_latency_seconds',
    'Time taken from strategy signal to order submission',
    ['trading_pair', 'order_type', 'side'],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

TRADING_BALANCE = Gauge(
    'trading_balance',
    'Current balance in paper trading account',
    ['currency']
)

CPU_USAGE = Gauge(
    'cpu_usage_percent',
    'CPU usage percentage',
    ['process']
)

MEMORY_USAGE = Gauge(
    'memory_usage_bytes',
    'Memory usage in bytes',
    ['process']
)

# Trading pairs to monitor
TRADING_PAIRS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT", "DOGEUSDT"]

# Initialize Flask app
app = Flask(__name__, static_folder='static')

# Initialize MEXC API
mexc_api = MexcAPI()

# Store for market data
market_data = {
    'tickers': {},
    'orderbooks': {},
    'klines': {},
    'trades': {}
}

# Technical indicators
indicators = {}

# Order and position tracking
paper_orders = []
paper_positions = {}
order_history = []

# Initialize system metrics
system_metrics = {
    'cpu': 0,
    'memory': 0,
    'start_time': time.time(),
    'uptime': 0
}

# Calculate technical indicators
def calculate_indicators(symbol, timeframe='1m'):
    global indicators, market_data
    
    if symbol not in market_data['klines'] or timeframe not in market_data['klines'][symbol]:
        return
    
    # Convert klines to pandas DataFrame
    klines = market_data['klines'][symbol][timeframe]
    if not klines:
        return
    
    df = pd.DataFrame(klines)
    df['close'] = pd.to_numeric(df['close'])
    df['high'] = pd.to_numeric(df['high'])
    df['low'] = pd.to_numeric(df['low'])
    df['open'] = pd.to_numeric(df['open'])
    df['volume'] = pd.to_numeric(df['volume'])
    
    # Calculate Moving Averages
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['sma_200'] = df['close'].rolling(window=200).mean()
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    
    # Calculate MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Calculate RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # Calculate Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
    
    # Store calculated indicators
    if symbol not in indicators:
        indicators[symbol] = {}
    indicators[symbol][timeframe] = df.iloc[-1].to_dict()

# Ticker update callback
def ticker_callback(symbol, ticker):
    global market_data
    market_data['tickers'][symbol] = ticker
    MARKET_DATA_THROUGHPUT.labels(trading_pair=symbol, update_type='ticker').inc()
    
    # Update Prometheus metrics for paper trading balances
    for currency, balance in mexc_api.get_paper_balances().items():
        TRADING_BALANCE.labels(currency=currency).set(balance)

# Orderbook update callback
def orderbook_callback(symbol, orderbook):
    global market_data
    market_data['orderbooks'][symbol] = orderbook
    MARKET_DATA_THROUGHPUT.labels(trading_pair=symbol, update_type='orderbook').inc()

# Kline update callback
def kline_callback(symbol, interval, kline):
    global market_data
    
    # Convert interval format
    interval_map = {
        'Min1': '1m',
        'Min5': '5m',
        'Min15': '15m',
        'Hour1': '1h',
        'Hour4': '4h',
        'Day1': '1d'
    }
    
    if interval in interval_map:
        interval = interval_map[interval]
    
    if symbol not in market_data['klines']:
        market_data['klines'][symbol] = {}
    
    if interval not in market_data['klines'][symbol]:
        market_data['klines'][symbol][interval] = []
    
    # Update or append kline
    updated = False
    for i, existing_k in enumerate(market_data['klines'][symbol][interval]):
        if existing_k['open_time'] == kline['open_time']:
            market_data['klines'][symbol][interval][i] = kline
            updated = True
            break
    
    if not updated:
        market_data['klines'][symbol][interval].append(kline)
        # Keep only the most recent 500 klines
        if len(market_data['klines'][symbol][interval]) > 500:
            market_data['klines'][symbol][interval] = market_data['klines'][symbol][interval][-500:]
    
    MARKET_DATA_THROUGHPUT.labels(trading_pair=symbol, update_type='kline').inc()
    
    # Calculate technical indicators
    calculate_indicators(symbol, interval)

# Trade update callback
def trade_callback(symbol, trade):
    global market_data
    
    if symbol not in market_data['trades']:
        market_data['trades'][symbol] = []
    
    market_data['trades'][symbol].append(trade)
    # Keep only the most recent 100 trades
    if len(market_data['trades'][symbol]) > 100:
        market_data['trades'][symbol] = market_data['trades'][symbol][-100:]
    
    MARKET_DATA_THROUGHPUT.labels(trading_pair=symbol, update_type='trade').inc()

# Paper trading functions
def paper_place_order(symbol, side, order_type, quantity, price=None):
    start_time = time.time()
    
    # Place paper order
    order = mexc_api.paper_place_order(symbol, side, order_type, quantity, price)
    
    # Record execution latency
    latency = time.time() - start_time
    ORDER_EXECUTION_LATENCY.labels(
        trading_pair=symbol, 
        order_type=order_type.lower(), 
        side=side.lower()
    ).observe(latency)
    
    # Update paper orders and positions
    if 'error' not in order:
        paper_orders.append(order)
        order_history.append(order)
        
        # Update positions
        base_currency = symbol[:-4]  # e.g., "BTC" from "BTCUSDT"
        if base_currency not in paper_positions:
            paper_positions[base_currency] = {'quantity': 0, 'avg_price': 0, 'total_cost': 0}
        
        position = paper_positions[base_currency]
        
        if side == 'BUY':
            # Update average price and quantity
            new_quantity = position['quantity'] + quantity
            new_cost = position['total_cost'] + (price * quantity)
            position['avg_price'] = new_cost / new_quantity if new_quantity > 0 else 0
            position['quantity'] = new_quantity
            position['total_cost'] = new_cost
        else:  # SELL
            # Reduce position
            position['quantity'] -= quantity
            position['total_cost'] -= position['avg_price'] * quantity
            if position['quantity'] <= 0:
                position['avg_price'] = 0
                position['total_cost'] = 0
                if position['quantity'] < 0:
                    position['quantity'] = 0
    
    return order

def update_system_metrics():
    """Update system metrics periodically"""
    global system_metrics
    
    while True:
        # Update CPU and memory usage (simple simulation)
        system_metrics['cpu'] = 20 + 15 * np.sin(time.time() / 10)  # 5-35% CPU usage
        system_metrics['memory'] = (200 + 50 * np.sin(time.time() / 20)) * 1024 * 1024  # 150-250 MB
        system_metrics['uptime'] = time.time() - system_metrics['start_time']
        
        # Update Prometheus metrics
        CPU_USAGE.labels(process='market-data-processor').set(system_metrics['cpu'])
        MEMORY_USAGE.labels(process='market-data-processor').set(system_metrics['memory'])
        
        time.sleep(1)

# Generate simulated ticker data when no API key is provided
def simulate_market_data():
    """Generate simulated market data when no real data is available"""
    print("Starting market data simulation...")
    
    # Initialize prices
    prices = {
        "BTCUSDT": 35000,
        "ETHUSDT": 1800,
        "BNBUSDT": 240,
        "ADAUSDT": 0.30,
        "SOLUSDT": 85,
        "DOGEUSDT": 0.08
    }
    
    # Volatilities
    volatilities = {
        "BTCUSDT": 0.01,
        "ETHUSDT": 0.012,
        "BNBUSDT": 0.015,
        "ADAUSDT": 0.02,
        "SOLUSDT": 0.025,
        "DOGEUSDT": 0.03
    }
    
    while True:
        for symbol in TRADING_PAIRS:
            # Generate random price change
            change_pct = np.random.normal(0, volatilities[symbol])
            price_change = prices[symbol] * change_pct
            prices[symbol] += price_change
            
            # Create ticker data
            ticker = {
                'price': prices[symbol],
                'volume': np.random.uniform(100, 1000) * prices[symbol],
                'high': prices[symbol] * (1 + np.random.uniform(0, 0.005)),
                'low': prices[symbol] * (1 - np.random.uniform(0, 0.005)),
                'open': prices[symbol] - price_change,
                'close': prices[symbol],
                'timestamp': int(time.time() * 1000)
            }
            
            # Update ticker
            ticker_callback(symbol, ticker)
            
            # Create kline data
            current_time = int(time.time() * 1000)
            for interval in ['1m', '5m', '15m', '1h']:
                if interval == '1m':
                    timeframe = 60 * 1000
                    interval_name = 'Min1'
                elif interval == '5m':
                    timeframe = 5 * 60 * 1000
                    interval_name = 'Min5'
                elif interval == '15m':
                    timeframe = 15 * 60 * 1000
                    interval_name = 'Min15'
                else:  # 1h
                    timeframe = 60 * 60 * 1000
                    interval_name = 'Hour1'
                
                # Calculate open time
                open_time = current_time - (current_time % timeframe)
                close_time = open_time + timeframe - 1
                
                kline = {
                    'open_time': open_time,
                    'close_time': close_time,
                    'open': ticker['open'],
                    'high': ticker['high'],
                    'low': ticker['low'],
                    'close': ticker['close'],
                    'volume': ticker['volume']
                }
                
                # Update kline
                kline_callback(symbol, interval_name, kline)
            
            # Create orderbook
            orderbook = {
                'asks': [],
                'bids': [],
                'timestamp': int(time.time() * 1000)
            }
            
            # Generate asks (sell orders)
            current_price = prices[symbol]
            for i in range(20):
                price = current_price * (1 + 0.0001 * (i + 1))
                size = np.random.uniform(0.1, 2) * 10 / price
                orderbook['asks'].append([price, size])
            
            # Generate bids (buy orders)
            for i in range(20):
                price = current_price * (1 - 0.0001 * (i + 1))
                size = np.random.uniform(0.1, 2) * 10 / price
                orderbook['bids'].append([price, size])
            
            # Update orderbook
            orderbook_callback(symbol, orderbook)
            
            # Create trade
            trade = {
                'id': str(int(time.time() * 1000)),
                'price': prices[symbol],
                'quantity': np.random.uniform(0.001, 0.1),
                'time': int(time.time() * 1000),
                'buyer_maker': np.random.choice([True, False]),
                'best_match': True
            }
            
            # Update trade
            trade_callback(symbol, trade)
        
        time.sleep(1)

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/api/tickers')
def get_tickers():
    return jsonify(market_data['tickers'])

@app.route('/api/ticker/<symbol>')
def get_ticker(symbol):
    if symbol in market_data['tickers']:
        return jsonify(market_data['tickers'][symbol])
    return jsonify({'error': 'Symbol not found'}), 404

@app.route('/api/orderbook/<symbol>')
def get_orderbook(symbol):
    if symbol in market_data['orderbooks']:
        return jsonify(market_data['orderbooks'][symbol])
    return jsonify({'error': 'Symbol not found'}), 404

@app.route('/api/klines/<symbol>/<interval>')
def get_klines(symbol, interval):
    if symbol in market_data['klines'] and interval in market_data['klines'][symbol]:
        return jsonify(market_data['klines'][symbol][interval])
    return jsonify({'error': 'Data not found'}), 404

@app.route('/api/trades/<symbol>')
def get_trades(symbol):
    if symbol in market_data['trades']:
        return jsonify(market_data['trades'][symbol])
    return jsonify({'error': 'Symbol not found'}), 404

@app.route('/api/indicators/<symbol>/<interval>')
def get_indicators(symbol, interval):
    if symbol in indicators and interval in indicators[symbol]:
        return jsonify(indicators[symbol][interval])
    return jsonify({'error': 'Indicators not found'}), 404

@app.route('/api/balances')
def get_balances():
    return jsonify(mexc_api.get_paper_balances())

@app.route('/api/positions')
def get_positions():
    return jsonify(paper_positions)

@app.route('/api/orders')
def get_orders():
    return jsonify(paper_orders)

@app.route('/api/order_history')
def get_order_history():
    return jsonify(order_history)

@app.route('/api/place_order', methods=['POST'])
def api_place_order():
    data = request.json
    symbol = data.get('symbol')
    side = data.get('side')
    order_type = data.get('type')
    quantity = float(data.get('quantity'))
    price = float(data.get('price')) if data.get('price') else None
    
    result = paper_place_order(symbol, side, order_type, quantity, price)
    return jsonify(result)

@app.route('/api/system_metrics')
def get_system_metrics():
    return jsonify(system_metrics)

@app.route('/health')
def health():
    return 'healthy'

# Initialize the app
def initialize():
    # Create static directory if it doesn't exist
    os.makedirs('static', exist_ok=True)
    
    # Register callbacks
    mexc_api.add_ticker_callback(ticker_callback)
    mexc_api.add_orderbook_callback(orderbook_callback)
    mexc_api.add_kline_callback(kline_callback)
    mexc_api.add_trade_callback(trade_callback)
    
    # Connect to WebSocket
    try:
        mexc_api.connect()
        print("Connected to MEXC WebSocket API")
    except Exception as e:
        print(f"Failed to connect to MEXC WebSocket API: {e}")
        print("Starting simulation mode...")
        # Start simulation thread if API connection fails
        sim_thread = threading.Thread(target=simulate_market_data, daemon=True)
        sim_thread.start()
    
    # Start system metrics update thread
    metrics_thread = threading.Thread(target=update_system_metrics, daemon=True)
    metrics_thread.daemon = True
    metrics_thread.start()
    
    # Start Prometheus metrics server
    start_http_server(8081)
    print("Prometheus metrics server started on port 8081")

if __name__ == '__main__':
    initialize()
    print("Starting Trading Dashboard on http://localhost:8080")
    app.run(host='0.0.0.0', port=8080)
