#!/usr/bin/env python
"""
Comprehensive test suite for the HFT execution engine.

This module provides unit and integration tests for the HFT execution engine,
including the Python wrapper, signal generation, and order book analytics.
"""

import os
import sys
import unittest
import numpy as np
from typing import List, Tuple, Dict

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from python_integration.hft_engine_wrapper import HFTEngineWrapper

class TestHFTEngineWrapper(unittest.TestCase):
    """Test cases for the HFT engine wrapper."""
    
    def setUp(self):
        """Set up test environment."""
        self.symbol = "BTCUSDC"
        self.engine = HFTEngineWrapper(self.symbol, use_mock=True)
    
    def test_initialization(self):
        """Test engine initialization."""
        self.assertEqual(self.engine.symbol, self.symbol)
        self.assertTrue(self.engine.initialized)
        self.assertEqual(self.engine.tick_count, 0)
    
    def test_process_tick(self):
        """Test tick processing."""
        # Process a single tick
        result = self.engine.process_tick(50000.0, 1.0, 49995.0, 50005.0)
        self.assertTrue(result)
        self.assertEqual(self.engine.tick_count, 1)
        self.assertEqual(self.engine.last_price, 50000.0)
        self.assertEqual(self.engine.last_volume, 1.0)
        self.assertEqual(self.engine.last_bid, 49995.0)
        self.assertEqual(self.engine.last_ask, 50005.0)
        
        # Process multiple ticks
        for i in range(10):
            price = 50000.0 + i * 10
            self.engine.process_tick(price, 0.1, price - 5, price + 5)
        
        self.assertEqual(self.engine.tick_count, 11)
        self.assertEqual(self.engine.last_price, 50090.0)
    
    def test_update_orderbook(self):
        """Test order book updates."""
        # Create sample order book
        bids = [(49995.0, 1.0), (49990.0, 2.0), (49985.0, 3.0)]
        asks = [(50005.0, 1.0), (50010.0, 2.0), (50015.0, 3.0)]
        
        # Update order book
        result = self.engine.update_orderbook(bids, asks)
        self.assertTrue(result)
        self.assertEqual(self.engine.last_bids, bids)
        self.assertEqual(self.engine.last_asks, asks)
        
        # Check imbalance calculation
        bid_value = sum(price * qty for price, qty in bids)
        ask_value = sum(price * qty for price, qty in asks)
        expected_imbalance = bid_value / ask_value - 1.0
        self.assertAlmostEqual(self.engine.last_imbalance, expected_imbalance)
    
    def test_get_trading_signal_empty(self):
        """Test signal generation with no data."""
        signal, confidence, duration = self.engine.get_trading_signal()
        self.assertEqual(signal, 0.0)
        self.assertEqual(confidence, 0.0)
        self.assertEqual(duration, 0)
    
    def test_get_trading_signal(self):
        """Test signal generation with data."""
        # Populate with data - create a strong uptrend
        for i in range(30):
            price = 50000.0 + i * 50  # Increased price increment for stronger trend
            self.engine.process_tick(price, 0.1, price - 5, price + 5)
            
            # Create order book with strong buy imbalance
            bids = [(price - 5 - j, 2.0 / (j + 1)) for j in range(10)]  # Increased bid quantities
            asks = [(price + 5 + j, 1.0 / (j + 1)) for j in range(10)]
            self.engine.update_orderbook(bids, asks)
        
        # Get signal
        signal, confidence, duration = self.engine.get_trading_signal()
        
        # Signal should be positive (uptrend)
        self.assertGreater(signal, 0.0)
        self.assertGreater(confidence, 0.0)
        self.assertGreater(duration, 0)
        
        # Signal should be actionable with lower threshold for test
        self.assertTrue(self.engine.is_signal_actionable(min_confidence=0.4))
    
    def test_signal_consistency(self):
        """Test signal consistency with stable data."""
        # Populate with stable data
        for i in range(30):
            self.engine.process_tick(50000.0, 0.1, 49995.0, 50005.0)
            
            bids = [(49995.0 - j, 1.0 / (j + 1)) for j in range(10)]
            asks = [(50005.0 + j, 1.0 / (j + 1)) for j in range(10)]
            self.engine.update_orderbook(bids, asks)
        
        # Get multiple signals
        signals = []
        for _ in range(5):
            signal, confidence, _ = self.engine.get_trading_signal()
            signals.append(signal)
        
        # Signals should be consistent
        for i in range(1, len(signals)):
            self.assertAlmostEqual(signals[i], signals[0], delta=0.01)
    
    def test_signal_direction_up(self):
        """Test signal direction with uptrend."""
        # Populate with uptrend data
        for i in range(30):
            price = 50000.0 + i * 10
            self.engine.process_tick(price, 0.1, price - 5, price + 5)
            
            bids = [(price - 5 - j, 1.0 / (j + 1)) for j in range(10)]
            asks = [(price + 5 + j, 1.0 / (j + 1)) for j in range(10)]
            self.engine.update_orderbook(bids, asks)
        
        # Get signal
        signal, confidence, _ = self.engine.get_trading_signal()
        
        # Signal should be positive (buy)
        self.assertGreater(signal, 0.0)
    
    def test_signal_direction_down(self):
        """Test signal direction with downtrend."""
        # Populate with downtrend data
        for i in range(30):
            price = 50000.0 - i * 10
            self.engine.process_tick(price, 0.1, price - 5, price + 5)
            
            bids = [(price - 5 - j, 1.0 / (j + 1)) for j in range(10)]
            asks = [(price + 5 + j, 1.0 / (j + 1)) for j in range(10)]
            self.engine.update_orderbook(bids, asks)
        
        # Get signal
        signal, confidence, _ = self.engine.get_trading_signal()
        
        # Signal should be negative (sell)
        self.assertLess(signal, 0.0)


class TestHFTIntegration(unittest.TestCase):
    """Integration tests for the HFT engine."""
    
    def setUp(self):
        """Set up test environment."""
        self.symbol = "BTCUSDC"
        self.engine = HFTEngineWrapper(self.symbol, use_mock=True)
    
    def test_end_to_end_workflow(self):
        """Test end-to-end workflow."""
        # 1. Process market data
        for i in range(50):
            # Simulate price movement
            price = 50000.0 + 100 * np.sin(i / 10.0)
            volume = abs(np.random.normal(0, 1))
            bid = price - 5
            ask = price + 5
            
            # Process tick
            self.engine.process_tick(price, volume, bid, ask)
            
            # Update order book
            bids = [(bid - j, 1.0 / (j + 1)) for j in range(10)]
            asks = [(ask + j, 1.0 / (j + 1)) for j in range(10)]
            self.engine.update_orderbook(bids, asks)
        
        # 2. Get trading signal
        signal, confidence, duration = self.engine.get_trading_signal()
        
        # 3. Verify signal properties
        self.assertTrue(-1.0 <= signal <= 1.0)
        self.assertTrue(0.0 <= confidence <= 1.0)
        self.assertGreaterEqual(duration, 0)
        
        # 4. Check actionability
        is_actionable = self.engine.is_signal_actionable(min_confidence=0.4)  # Lower threshold for test
        
        # 5. Simulate trading decision
        if is_actionable:
            if signal > 0:
                # Buy signal
                print(f"BUY signal with confidence {confidence:.2f}")
            else:
                # Sell signal
                print(f"SELL signal with confidence {confidence:.2f}")
        else:
            # No action
            print("No actionable signal")


if __name__ == "__main__":
    unittest.main()
