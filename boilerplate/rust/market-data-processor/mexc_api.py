import requests
import hmac
import hashlib
import time
import json
import websocket
import threading
from urllib.parse import urlencode

class MexcAPI:
    """
    MEXC API connector for real-time market data and trading
    """
    
    def __init__(self, api_key="", api_secret="", use_testnet=True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.use_testnet = use_testnet
        
        # Base URLs
        if use_testnet:
            self.base_url = "https://api.mexc.com/api/v3"
            self.ws_url = "wss://wbs.mexc.com/ws"
        else:
            self.base_url = "https://api.mexc.com/api/v3"
            self.ws_url = "wss://wbs.mexc.com/ws"
        
        # Initialize data containers
        self.tickers = {}
        self.orderbooks = {}
        self.klines = {}
        self.positions = {}
        self.open_orders = {}
        self.trade_history = []
        self.balances = {"BTC": 1.0, "USDT": 10000.0}  # Default paper trading balances
        
        # WebSocket connections
        self.ws = None
        self.ws_thread = None
        self.keep_running = False
        
        # Callbacks
        self.ticker_callbacks = []
        self.orderbook_callbacks = []
        self.kline_callbacks = []
        self.trade_callbacks = []
        
    def _generate_signature(self, params):
        """Generate HMAC SHA256 signature for API authentication"""
        query_string = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _get_headers(self):
        """Get default headers for API requests"""
        return {
            'X-MEXC-APIKEY': self.api_key,
            'Content-Type': 'application/json'
        }
    
    def _request(self, method, endpoint, params=None, signed=False):
        """Make an HTTP request to the MEXC API"""
        url = f"{self.base_url}{endpoint}"
        headers = self._get_headers()
        
        if signed and self.api_key:
            if params is None:
                params = {}
            params['timestamp'] = int(time.time() * 1000)
            params['signature'] = self._generate_signature(params)
        
        if method == 'GET':
            if params:
                response = requests.get(url, headers=headers, params=params)
            else:
                response = requests.get(url, headers=headers)
        elif method == 'POST':
            response = requests.post(url, headers=headers, json=params)
        elif method == 'DELETE':
            response = requests.delete(url, headers=headers, params=params)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"API Error: {response.status_code} - {response.text}")
            return None
    
    # Public API methods
    def get_exchange_info(self):
        """Get exchange trading rules and symbol information"""
        return self._request('GET', '/exchangeInfo')
    
    def get_ticker(self, symbol):
        """Get latest price for a symbol"""
        params = {'symbol': symbol}
        return self._request('GET', '/ticker/24hr', params)
    
    def get_all_tickers(self):
        """Get latest price for all symbols"""
        return self._request('GET', '/ticker/24hr')
    
    def get_orderbook(self, symbol, limit=100):
        """Get order book for a symbol"""
        params = {'symbol': symbol, 'limit': limit}
        return self._request('GET', '/depth', params)
    
    def get_recent_trades(self, symbol, limit=500):
        """Get recent trades for a symbol"""
        params = {'symbol': symbol, 'limit': limit}
        return self._request('GET', '/trades', params)
    
    def get_klines(self, symbol, interval='1m', limit=500, startTime=None, endTime=None):
        """Get candlestick data for a symbol"""
        params = {'symbol': symbol, 'interval': interval, 'limit': limit}
        if startTime:
            params['startTime'] = startTime
        if endTime:
            params['endTime'] = endTime
        return self._request('GET', '/klines', params)
    
    # Private API methods (require authentication)
    def get_account(self):
        """Get account information"""
        return self._request('GET', '/account', signed=True)
    
    def get_balances(self):
        """Get account balances"""
        account = self._request('GET', '/account', signed=True)
        if account and 'balances' in account:
            return account['balances']
        return []
    
    def place_order(self, symbol, side, order_type, quantity, price=None, time_in_force='GTC'):
        """Place a new order"""
        params = {
            'symbol': symbol,
            'side': side,
            'type': order_type,
            'quantity': quantity,
            'timestamp': int(time.time() * 1000)
        }
        
        if price:
            params['price'] = price
            
        if order_type == 'LIMIT':
            params['timeInForce'] = time_in_force
        
        return self._request('POST', '/order', params, signed=True)
    
    def cancel_order(self, symbol, order_id=None, client_order_id=None):
        """Cancel an existing order"""
        params = {'symbol': symbol}
        if order_id:
            params['orderId'] = order_id
        elif client_order_id:
            params['origClientOrderId'] = client_order_id
        else:
            raise ValueError("Either order_id or client_order_id must be provided")
        
        return self._request('DELETE', '/order', params, signed=True)
    
    def get_open_orders(self, symbol=None):
        """Get all open orders"""
        params = {}
        if symbol:
            params['symbol'] = symbol
        return self._request('GET', '/openOrders', params, signed=True)
    
    def get_order_status(self, symbol, order_id=None, client_order_id=None):
        """Get order status"""
        params = {'symbol': symbol}
        if order_id:
            params['orderId'] = order_id
        elif client_order_id:
            params['origClientOrderId'] = client_order_id
        else:
            raise ValueError("Either order_id or client_order_id must be provided")
        
        return self._request('GET', '/order', params, signed=True)
    
    # WebSocket methods for real-time data
    def _on_ws_message(self, ws, message):
        """Handle WebSocket messages"""
        try:
            data = json.loads(message)
            
            if 'c' in data and 's' in data:  # Ticker data
                symbol = data['s']
                self.tickers[symbol] = {
                    'price': float(data['c']),
                    'volume': float(data['v']),
                    'high': float(data['h']),
                    'low': float(data['l']),
                    'open': float(data['o']),
                    'close': float(data['c']),
                    'timestamp': int(time.time() * 1000)
                }
                for callback in self.ticker_callbacks:
                    callback(symbol, self.tickers[symbol])
            
            elif 'asks' in data and 'bids' in data and 's' in data:  # Orderbook data
                symbol = data['s']
                self.orderbooks[symbol] = {
                    'asks': data['asks'],
                    'bids': data['bids'],
                    'timestamp': int(time.time() * 1000)
                }
                for callback in self.orderbook_callbacks:
                    callback(symbol, self.orderbooks[symbol])
            
            elif 'k' in data and 's' in data:  # Kline/candlestick data
                symbol = data['s']
                kline = data['k']
                interval = kline['i']
                if symbol not in self.klines:
                    self.klines[symbol] = {}
                if interval not in self.klines[symbol]:
                    self.klines[symbol][interval] = []
                
                k = {
                    'open_time': kline['t'],
                    'close_time': kline['T'],
                    'open': float(kline['o']),
                    'high': float(kline['h']),
                    'low': float(kline['l']),
                    'close': float(kline['c']),
                    'volume': float(kline['v'])
                }
                
                # Update or append kline
                updated = False
                for i, existing_k in enumerate(self.klines[symbol][interval]):
                    if existing_k['open_time'] == k['open_time']:
                        self.klines[symbol][interval][i] = k
                        updated = True
                        break
                
                if not updated:
                    self.klines[symbol][interval].append(k)
                    # Keep only the most recent 500 klines
                    if len(self.klines[symbol][interval]) > 500:
                        self.klines[symbol][interval] = self.klines[symbol][interval][-500:]
                
                for callback in self.kline_callbacks:
                    callback(symbol, interval, k)
            
            elif 'e' in data and data['e'] == 'trade':  # Trade data
                symbol = data['s']
                trade = {
                    'id': data['t'],
                    'price': float(data['p']),
                    'quantity': float(data['q']),
                    'time': data['T'],
                    'buyer_maker': data['m'],
                    'best_match': data['M']
                }
                for callback in self.trade_callbacks:
                    callback(symbol, trade)
        
        except Exception as e:
            print(f"WebSocket message handling error: {e}")
    
    def _on_ws_error(self, ws, error):
        """Handle WebSocket errors"""
        print(f"WebSocket error: {error}")
    
    def _on_ws_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket connection close"""
        print(f"WebSocket closed: {close_status_code} - {close_msg}")
        # Try to reconnect if we're supposed to be running
        if self.keep_running:
            print("Attempting to reconnect WebSocket...")
            self._connect_websocket()
    
    def _on_ws_open(self, ws):
        """Handle WebSocket connection open"""
        print("WebSocket connection established")
        # Subscribe to symbols
        for symbol in ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT"]:
            # Subscribe to ticker
            self._ws_subscribe(f"spot@public.ticker.v3.api@{symbol}")
            # Subscribe to orderbook
            self._ws_subscribe(f"spot@public.depth.v3.api@{symbol}")
            # Subscribe to klines (1m, 5m, 15m, 1h)
            for interval in ["Min1", "Min5", "Min15", "Hour1"]:
                self._ws_subscribe(f"spot@public.kline.v3.api@{symbol}@{interval}")
    
    def _ws_subscribe(self, channel):
        """Subscribe to a WebSocket channel"""
        if self.ws and self.ws.sock and self.ws.sock.connected:
            self.ws.send(json.dumps({
                "method": "SUBSCRIPTION",
                "params": [channel]
            }))
    
    def _connect_websocket(self):
        """Connect to the WebSocket API"""
        websocket.enableTrace(False)
        self.ws = websocket.WebSocketApp(
            self.ws_url,
            on_open=self._on_ws_open,
            on_message=self._on_ws_message,
            on_error=self._on_ws_error,
            on_close=self._on_ws_close
        )
        self.ws_thread = threading.Thread(target=self.ws.run_forever)
        self.ws_thread.daemon = True
        self.ws_thread.start()
    
    def connect(self):
        """Connect to the WebSocket API and start receiving real-time data"""
        self.keep_running = True
        self._connect_websocket()
    
    def disconnect(self):
        """Disconnect from the WebSocket API"""
        self.keep_running = False
        if self.ws:
            self.ws.close()
    
    def add_ticker_callback(self, callback):
        """Add a callback for ticker updates"""
        self.ticker_callbacks.append(callback)
    
    def add_orderbook_callback(self, callback):
        """Add a callback for orderbook updates"""
        self.orderbook_callbacks.append(callback)
    
    def add_kline_callback(self, callback):
        """Add a callback for kline updates"""
        self.kline_callbacks.append(callback)
    
    def add_trade_callback(self, callback):
        """Add a callback for trade updates"""
        self.trade_callbacks.append(callback)
    
    # Paper trading methods
    def paper_update_balance(self, currency, amount):
        """Update paper trading balance"""
        if currency in self.balances:
            self.balances[currency] += amount
        else:
            self.balances[currency] = amount
    
    def paper_place_order(self, symbol, side, order_type, quantity, price=None):
        """Place a paper trading order"""
        order_id = str(int(time.time() * 1000))
        base_currency = symbol[:-4]  # e.g., "BTC" from "BTCUSDT"
        quote_currency = symbol[-4:]  # e.g., "USDT" from "BTCUSDT"
        
        # Get current price if not provided
        if price is None:
            if symbol in self.tickers:
                price = self.tickers[symbol]['price']
            else:
                return {"error": "Symbol not found in current tickers"}
        
        # Check if we have enough balance
        if side == "BUY":
            required_amount = price * quantity
            if quote_currency not in self.balances or self.balances[quote_currency] < required_amount:
                return {"error": "Insufficient balance for buy order"}
            
            # Update balances
            self.paper_update_balance(quote_currency, -required_amount)
            self.paper_update_balance(base_currency, quantity)
        
        elif side == "SELL":
            if base_currency not in self.balances or self.balances[base_currency] < quantity:
                return {"error": "Insufficient balance for sell order"}
            
            # Update balances
            self.paper_update_balance(base_currency, -quantity)
            self.paper_update_balance(quote_currency, price * quantity)
        
        # Create order record
        order = {
            "orderId": order_id,
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "quantity": quantity,
            "price": price,
            "status": "FILLED",
            "time": int(time.time() * 1000)
        }
        
        # Add to trade history
        self.trade_history.append(order)
        
        return order
    
    def get_paper_balances(self):
        """Get paper trading balances"""
        return self.balances
    
    def get_paper_trades(self):
        """Get paper trading history"""
        return self.trade_history
