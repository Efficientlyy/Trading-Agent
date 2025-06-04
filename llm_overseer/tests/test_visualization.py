#!/usr/bin/env python
"""
Test module for chart visualization components.

This module provides tests for the chart visualization components,
including real-time data feeds, chart rendering, and event handling.
"""

import os
import sys
import json
import logging
import asyncio
import unittest
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules to test
from llm_overseer.core.event_bus import EventBus
from llm_overseer.config.config import Config
from llm_overseer.visualization.chart_visualization import ChartVisualization
from llm_overseer.data.market_data_service import MarketDataService
from llm_overseer.analysis.pattern_recognition import PatternRecognition

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_visualization.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("test_visualization")

class TestChartVisualization(unittest.TestCase):
    """Test case for chart visualization components."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Create configuration
        cls.config = Config()
        
        # Create event bus
        cls.event_bus = EventBus()
        
        # Create chart visualization
        cls.chart_visualization = ChartVisualization(cls.config, cls.event_bus)
        
        # Create market data service
        cls.market_data_service = MarketDataService(cls.config, cls.event_bus)
        
        # Create pattern recognition
        cls.pattern_recognition = PatternRecognition(cls.config, cls.event_bus)
        
        # Start market data service
        asyncio.run(cls.market_data_service.start())
        
        # Wait for initial data
        time.sleep(5)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        # Stop market data service
        asyncio.run(cls.market_data_service.stop())
    
    def test_chart_creation(self):
        """Test chart creation for all symbols."""
        symbols = self.config.get("trading.symbols", ["BTC/USDC", "ETH/USDC", "SOL/USDC"])
        
        for symbol in symbols:
            # Generate chart
            chart_path = asyncio.run(self.chart_visualization.update_chart(symbol))
            
            # Check if chart was created
            self.assertTrue(os.path.exists(chart_path), f"Chart not created for {symbol}")
            
            # Check file size (should be non-zero)
            self.assertGreater(os.path.getsize(chart_path), 0, f"Chart file is empty for {symbol}")
            
            logger.info(f"Chart created successfully for {symbol}: {chart_path}")
    
    def test_indicator_calculation(self):
        """Test indicator calculation for all symbols."""
        symbols = self.config.get("trading.symbols", ["BTC/USDC", "ETH/USDC", "SOL/USDC"])
        
        for symbol in symbols:
            # Calculate indicators
            self.chart_visualization.calculate_indicators(symbol)
            
            # Check if indicators were calculated
            self.assertIn(symbol, self.chart_visualization.indicators, f"Indicators not calculated for {symbol}")
            
            # Check if SMA was calculated
            self.assertIn("sma", self.chart_visualization.indicators[symbol], f"SMA not calculated for {symbol}")
            
            # Check if RSI was calculated
            self.assertIn("rsi", self.chart_visualization.indicators[symbol], f"RSI not calculated for {symbol}")
            
            logger.info(f"Indicators calculated successfully for {symbol}")
    
    def test_pattern_detection(self):
        """Test pattern detection for all symbols."""
        symbols = self.config.get("trading.symbols", ["BTC/USDC", "ETH/USDC", "SOL/USDC"])
        
        for symbol in symbols:
            # Run pattern detection
            asyncio.run(self.pattern_recognition.detect_patterns(symbol))
            
            # We can't guarantee patterns will be detected, but we can check if the function runs without errors
            logger.info(f"Pattern detection ran successfully for {symbol}")
    
    def test_event_propagation(self):
        """Test event propagation from market data to chart visualization."""
        # Create a test event handler
        events_received = []
        
        async def test_event_handler(topic, data):
            events_received.append((topic, data))
        
        # Subscribe to visualization events
        subscription_id = self.event_bus.subscribe("visualization.chart_updated", test_event_handler)
        
        # Generate market data event
        symbol = "BTC/USDC"
        market_data = {
            "success": True,
            "symbol": symbol,
            "price": 50000.0,
            "volume_24h": 1000.0,
            "timestamp": datetime.now().isoformat()
        }
        
        # Publish market data event
        asyncio.run(self.event_bus.publish("trading.market_data", market_data))
        
        # Wait for event propagation
        time.sleep(2)
        
        # Check if visualization event was received
        self.assertGreater(len(events_received), 0, "No visualization events received")
        
        # Unsubscribe from events
        self.event_bus.unsubscribe("visualization.chart_updated", subscription_id)
        
        logger.info(f"Event propagation test successful, received {len(events_received)} events")
    
    def test_real_time_updates(self):
        """Test real-time updates for all symbols."""
        symbols = self.config.get("trading.symbols", ["BTC/USDC", "ETH/USDC", "SOL/USDC"])
        
        # Create a counter for chart updates
        chart_updates = {symbol: 0 for symbol in symbols}
        
        async def count_chart_updates(topic, data):
            if data.get("symbol") in chart_updates:
                chart_updates[data.get("symbol")] += 1
        
        # Subscribe to chart updated events
        subscription_id = self.event_bus.subscribe("visualization.chart_updated", count_chart_updates)
        
        # Wait for real-time updates
        time.sleep(10)
        
        # Check if charts were updated
        for symbol in symbols:
            self.assertGreater(chart_updates[symbol], 0, f"No chart updates for {symbol}")
            logger.info(f"Real-time updates test successful for {symbol}: {chart_updates[symbol]} updates")
        
        # Unsubscribe from events
        self.event_bus.unsubscribe("visualization.chart_updated", subscription_id)
    
    def test_performance(self):
        """Test visualization performance."""
        symbol = "BTC/USDC"
        
        # Measure chart generation time
        start_time = time.time()
        
        # Generate chart
        chart_path = asyncio.run(self.chart_visualization.update_chart(symbol))
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Check if chart was created in reasonable time
        self.assertLess(elapsed_time, 2.0, f"Chart generation too slow: {elapsed_time:.2f} seconds")
        
        logger.info(f"Performance test successful: chart generated in {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    unittest.main()
