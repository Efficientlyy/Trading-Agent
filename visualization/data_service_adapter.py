#!/usr/bin/env python
"""
Data Service Adapter for Trading-Agent System

This module provides a DataService adapter that wraps the MultiAssetDataService
to provide a consistent interface for dashboard visualization components.
"""

import os
import sys
import json
import time
import logging
import threading
from typing import Dict, List, Any, Optional, Union

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required modules
from visualization.data_service import MultiAssetDataService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_service_adapter.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("data_service_adapter")

class DataService:
    """Adapter for MultiAssetDataService to provide a consistent interface for dashboard visualization"""
    
    def __init__(self, symbols: List[str] = None, timeframes: List[str] = None):
        """Initialize data service adapter
        
        Args:
            symbols: List of symbols to track (default: ["BTCUSDC", "ETHUSDC", "SOLUSDC"])
            timeframes: List of timeframes to support (default: ["1m", "5m", "15m", "1h", "4h", "1d"])
        """
        # Convert symbols format if needed
        self.symbols = symbols or ["BTCUSDC", "ETHUSDC", "SOLUSDC"]
        self.internal_symbols = [s.replace('USDC', '/USDC') for s in self.symbols]
        
        # Initialize underlying data service
        self.data_service = MultiAssetDataService(
            symbols=self.internal_symbols,
            timeframes=timeframes
        )
        
        # Initialize data stores
        self.market_data = {}
        self.trading_data = {}
        self.signal_data = {}
        self.decision_data = {}
        
        # Initialize
        self.initialize()
        
        logger.info(f"Initialized DataService adapter with symbols={self.symbols}")
    
    def initialize(self):
        """Initialize data stores"""
        # Initialize market data
        for symbol in self.symbols:
            self.market_data[symbol] = {
                'symbol': symbol,
                'last_price': 0.0,
                'bid_price': 0.0,
                'ask_price': 0.0,
                'volume_24h': 0.0,
                'price_change_24h': 0.0,
                'price_change_pct_24h': 0.0,
                'high_24h': 0.0,
                'low_24h': 0.0,
                'timestamp': int(time.time() * 1000),
                'price_history': [],
                'volume_history': []
            }
        
        # Initialize trading data
        self.trading_data = {
            'balance': {},
            'positions': {},
            'orders': {},
            'trades': [],
            'pnl_history': []
        }
        
        # Initialize signal data
        self.signal_data = {
            'recent_signals': [],
            'signal_history': {}
        }
        
        # Initialize decision data
        self.decision_data = {
            'recent_decisions': [],
            'decision_history': {}
        }
    
    def update_market_data(self, market_data: Dict):
        """Update market data
        
        Args:
            market_data: Market data dictionary
        """
        try:
            # Update market data store
            self.market_data.update(market_data)
            
            logger.debug("Market data updated")
        except Exception as e:
            logger.error(f"Error updating market data: {str(e)}")
    
    def update_trading_data(self, trading_data: Dict):
        """Update trading data
        
        Args:
            trading_data: Trading data dictionary
        """
        try:
            # Update trading data store
            self.trading_data.update(trading_data)
            
            logger.debug("Trading data updated")
        except Exception as e:
            logger.error(f"Error updating trading data: {str(e)}")
    
    def update_signal_data(self, signal_data: Dict):
        """Update signal data
        
        Args:
            signal_data: Signal data dictionary
        """
        try:
            # Update signal data store
            self.signal_data.update(signal_data)
            
            logger.debug("Signal data updated")
        except Exception as e:
            logger.error(f"Error updating signal data: {str(e)}")
    
    def update_decision_data(self, decision_data: Dict):
        """Update decision data
        
        Args:
            decision_data: Decision data dictionary
        """
        try:
            # Update decision data store
            self.decision_data.update(decision_data)
            
            logger.debug("Decision data updated")
        except Exception as e:
            logger.error(f"Error updating decision data: {str(e)}")
    
    def get_market_data(self, symbol: str = None):
        """Get market data
        
        Args:
            symbol: Symbol to get data for (optional)
            
        Returns:
            Market data dictionary or specific symbol data
        """
        if symbol:
            return self.market_data.get(symbol, {})
        else:
            return self.market_data
    
    def get_trading_data(self):
        """Get trading data
        
        Returns:
            Trading data dictionary
        """
        return self.trading_data
    
    def get_signal_data(self, symbol: str = None):
        """Get signal data
        
        Args:
            symbol: Symbol to get data for (optional)
            
        Returns:
            Signal data dictionary or specific symbol data
        """
        if symbol:
            return {
                'recent_signals': [s for s in self.signal_data.get('recent_signals', []) if s.get('symbol') == symbol],
                'signal_history': self.signal_data.get('signal_history', {}).get(symbol, [])
            }
        else:
            return self.signal_data
    
    def get_decision_data(self, symbol: str = None):
        """Get decision data
        
        Args:
            symbol: Symbol to get data for (optional)
            
        Returns:
            Decision data dictionary or specific symbol data
        """
        if symbol:
            return {
                'recent_decisions': [d for d in self.decision_data.get('recent_decisions', []) if d.get('symbol') == symbol],
                'decision_history': self.decision_data.get('decision_history', {}).get(symbol, [])
            }
        else:
            return self.decision_data
    
    def get_klines(self, symbol: str, timeframe: str = "1h", limit: int = 100):
        """Get historical klines (candlestick) data
        
        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDC")
            timeframe: Timeframe (e.g., "1m", "5m", "15m", "1h", "4h", "1d")
            limit: Number of candles to retrieve
            
        Returns:
            DataFrame with OHLCV data
        """
        # Convert symbol format
        internal_symbol = symbol.replace('USDC', '/USDC')
        
        # Get klines from underlying data service
        return self.data_service.get_klines(internal_symbol, timeframe, limit)
    
    def get_ticker(self, symbol: str):
        """Get current ticker data
        
        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDC")
            
        Returns:
            Dictionary with ticker data
        """
        # Convert symbol format
        internal_symbol = symbol.replace('USDC', '/USDC')
        
        # Get ticker from underlying data service
        return self.data_service.get_ticker(internal_symbol)
    
    def get_orderbook(self, symbol: str, limit: int = 20):
        """Get current orderbook
        
        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDC")
            limit: Number of levels to retrieve
            
        Returns:
            Dictionary with orderbook data
        """
        # Convert symbol format
        internal_symbol = symbol.replace('USDC', '/USDC')
        
        # Get orderbook from underlying data service
        return self.data_service.get_orderbook(internal_symbol, limit)
    
    def get_recent_trades(self, symbol: str, limit: int = 50):
        """Get recent trades
        
        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDC")
            limit: Number of trades to retrieve
            
        Returns:
            List of trade dictionaries
        """
        # Convert symbol format
        internal_symbol = symbol.replace('USDC', '/USDC')
        
        # Get trades from underlying data service
        return self.data_service.get_recent_trades(internal_symbol, limit)
    
    def subscribe_ticker(self, symbol: str, callback):
        """Subscribe to ticker updates
        
        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDC")
            callback: Callback function to handle ticker updates
            
        Returns:
            True if subscription successful, False otherwise
        """
        # Convert symbol format
        internal_symbol = symbol.replace('USDC', '/USDC')
        
        # Subscribe to ticker updates
        return self.data_service.subscribe_ticker(internal_symbol, callback)
    
    def subscribe_klines(self, symbol: str, timeframe: str, callback):
        """Subscribe to klines updates
        
        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDC")
            timeframe: Timeframe (e.g., "1m", "5m", "15m", "1h", "4h", "1d")
            callback: Callback function to handle klines updates
            
        Returns:
            True if subscription successful, False otherwise
        """
        # Convert symbol format
        internal_symbol = symbol.replace('USDC', '/USDC')
        
        # Subscribe to klines updates
        return self.data_service.subscribe_klines(internal_symbol, timeframe, callback)
    
    def subscribe_trades(self, symbol: str, callback):
        """Subscribe to trades updates
        
        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDC")
            callback: Callback function to handle trades updates
            
        Returns:
            True if subscription successful, False otherwise
        """
        # Convert symbol format
        internal_symbol = symbol.replace('USDC', '/USDC')
        
        # Subscribe to trades updates
        return self.data_service.subscribe_trades(internal_symbol, callback)


# Example usage
if __name__ == "__main__":
    # Create data service
    data_service = DataService()
    
    # Get ticker
    ticker = data_service.get_ticker("BTCUSDC")
    print(f"BTCUSDC ticker: {ticker}")
    
    # Get klines
    klines = data_service.get_klines("BTCUSDC", "1h", 10)
    print(f"BTCUSDC klines: {klines.head()}")
    
    # Update market data
    data_service.update_market_data({
        "BTCUSDC": {
            "last_price": 105000.0,
            "bid_price": 104900.0,
            "ask_price": 105100.0
        }
    })
    
    # Get market data
    market_data = data_service.get_market_data("BTCUSDC")
    print(f"BTCUSDC market data: {market_data}")
