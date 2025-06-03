#!/usr/bin/env python
"""
Test module for visualization components integration.

This module tests the integration between chart visualization, pattern recognition,
and the dashboard components to ensure proper event flow and data display.
"""

import os
import sys
import json
import logging
import asyncio
import unittest
from unittest import IsolatedAsyncioTestCase
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for more detailed logging
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_visualization.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("test_visualization")

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import components
from core.event_bus import EventBus
from data.unified_pipeline import UnifiedDataPipeline
from visualization.chart_visualization import ChartVisualization
from analysis.pattern_recognition import PatternRecognition
from visualization.bridge import LLMVisualizationBridge
from integration.trading_system_connector import TradingSystemConnector


class TestVisualization(IsolatedAsyncioTestCase):
    """Test case for visualization components integration."""
    
    async def asyncSetUp(self):
        """Set up test environment asynchronously."""
        logger.debug("Setting up test environment")
        
        # Create components
        self.event_bus = EventBus()
        self.data_pipeline = UnifiedDataPipeline()
        self.chart_visualization = ChartVisualization()
        self.pattern_recognition = PatternRecognition()
        self.llm_visualization_bridge = LLMVisualizationBridge()
        
        # Connect components
        self.data_pipeline.set_event_bus(self.event_bus)
        self.chart_visualization.set_event_bus(self.event_bus)
        self.chart_visualization.set_data_pipeline(self.data_pipeline)
        self.pattern_recognition.set_event_bus(self.event_bus)
        self.pattern_recognition.set_data_pipeline(self.data_pipeline)
        self.llm_visualization_bridge.set_event_bus(self.event_bus)
        
        # Start event processing
        self.event_bus.start_processing()
        
        # Activate chart for testing
        self.chart_visualization.activate_chart("BTC/USDC", "1h")
        
        # Test data
        self.test_symbol = "BTC/USDC"
        self.test_timeframe = "1h"
        
        # Create sample kline data
        self.sample_klines = self._create_sample_klines()
        
        # Debug: Add direct event listeners for debugging
        self.indicator_signals_received = []
        self.patterns_detected = []
        
        # Register direct event listeners
        self.event_bus.subscribe("indicator.signal", self._debug_indicator_signal_handler)
        self.event_bus.subscribe("visualization.pattern_detected", self._debug_pattern_detected_handler)
        
        logger.debug("Test environment setup complete")
    
    async def asyncTearDown(self):
        """Tear down test environment asynchronously."""
        logger.debug("Tearing down test environment")
        # Stop event processing
        self.event_bus.stop_processing()
        logger.debug("Test environment teardown complete")
    
    async def _debug_indicator_signal_handler(self, topic, data):
        """Debug handler for indicator signals."""
        logger.debug(f"DEBUG: Received indicator signal: {data}")
        self.indicator_signals_received.append(data)
    
    async def _debug_pattern_detected_handler(self, topic, data):
        """Debug handler for pattern detected events."""
        logger.debug(f"DEBUG: Received pattern detected: {data}")
        self.patterns_detected.append(data)
    
    def _create_sample_klines(self):
        """Create sample kline data for testing."""
        logger.debug("Creating sample kline data")
        
        # Create timestamps for the last 100 hours
        timestamps = []
        for i in range(100):
            timestamps.append(datetime.now() - timedelta(hours=99-i))
        
        # Create price data with a trend and some volatility
        base_price = 50000
        prices = []
        for i in range(100):
            # Add trend
            trend = i * 10
            # Add some volatility
            volatility = (i % 10) * 50
            # Add some randomness
            import random
            randomness = random.randint(-100, 100)
            
            price = base_price + trend + volatility + randomness
            prices.append(price)
        
        # Create klines with specific patterns to trigger detection
        klines = []
        for i in range(100):
            # Create a double bottom pattern in the data
            if i >= 70 and i <= 90:
                # Create a W shape for double bottom
                if i == 75 or i == 85:
                    low_price = prices[i] - 500  # Create two similar lows
                elif i == 80:
                    low_price = prices[i] - 200  # Higher low in the middle
                else:
                    low_price = prices[i] - 300
            else:
                low_price = prices[i] - 100
            
            # Create RSI oversold condition near the end
            if i >= 90:
                # Make recent prices drop to trigger RSI oversold
                prices[i] = prices[i] - 800
                low_price = prices[i] - 200
            
            kline = {
                "timestamp": timestamps[i].isoformat(),
                "open": prices[i] - 50,
                "high": prices[i] + 100,
                "low": low_price,
                "close": prices[i],
                "volume": 10 + (i % 10)
            }
            klines.append(kline)
        
        logger.debug(f"Created {len(klines)} sample klines")
        return klines
    
    async def _publish_klines_update(self):
        """Publish klines update event."""
        logger.debug(f"Publishing klines update for {self.test_symbol} {self.test_timeframe}")
        await self.event_bus.publish("pipeline.klines_updated", {
            "symbol": self.test_symbol,
            "timeframe": self.test_timeframe,
            "klines": self.sample_klines
        })
        logger.debug("Klines update published")
    
    async def _publish_strategic_decision(self):
        """Publish strategic decision event."""
        logger.debug("Publishing strategic decision")
        await self.event_bus.publish("llm.strategic_decision", {
            "decision_type": "entry",
            "symbol": self.test_symbol,
            "direction": "bullish",
            "confidence": 0.85,
            "price": 50500,
            "summary": "Enter long position based on bullish pattern",
            "timestamp": datetime.now().isoformat()
        })
        logger.debug("Strategic decision published")
    
    async def _publish_pattern_detected(self):
        """Publish pattern detected event."""
        logger.debug("Publishing pattern detected")
        await self.event_bus.publish("visualization.pattern_detected", {
            "pattern_type": "double_bottom",
            "symbol": self.test_symbol,
            "timeframe": self.test_timeframe,
            "direction": "bullish",
            "confidence": 0.8,
            "price": 50200,
            "timestamp": datetime.now().isoformat()
        })
        logger.debug("Pattern detected published")
    
    async def test_klines_update_flow(self):
        """Test klines update event flow."""
        logger.debug("Starting test_klines_update_flow")
        
        # Subscribe to chart updates
        chart_updates = []
        
        def chart_update_callback(update_data):
            logger.debug(f"Chart update callback received: {update_data}")
            chart_updates.append(update_data)
        
        subscription_id = self.chart_visualization.subscribe_to_chart_updates(
            self.test_symbol, self.test_timeframe, chart_update_callback
        )
        
        # Publish klines update
        await self._publish_klines_update()
        
        # Wait for event processing
        logger.debug("Waiting for event processing")
        await asyncio.sleep(0.5)
        
        # Check if chart was updated
        self.assertTrue(len(chart_updates) > 0, "Chart update callback not called")
        
        # Check chart data
        chart_data = self.chart_visualization.get_chart_data(self.test_symbol, self.test_timeframe)
        self.assertIsNotNone(chart_data, "Chart data not available")
        self.assertEqual(chart_data["symbol"], self.test_symbol, "Chart data symbol mismatch")
        self.assertEqual(chart_data["timeframe"], self.test_timeframe, "Chart data timeframe mismatch")
        
        # Check if klines were processed
        self.assertTrue(len(chart_data["klines"]) > 0, "Klines not processed")
        
        # Check if indicators were calculated
        self.assertTrue(len(chart_data["indicators"]) > 0, "Indicators not calculated")
        
        # Unsubscribe from chart updates
        self.chart_visualization.unsubscribe_from_chart_updates(
            self.test_symbol, self.test_timeframe, subscription_id
        )
        
        logger.debug("test_klines_update_flow completed")
    
    async def test_pattern_recognition_flow(self):
        """Test pattern recognition event flow."""
        logger.debug("Starting test_pattern_recognition_flow")
        
        # Track indicator signals
        indicator_signals = []
        
        async def indicator_signal_handler(topic, data):
            logger.debug(f"Indicator signal handler received: {data}")
            indicator_signals.append(data)
        
        # Subscribe to indicator signals
        self.event_bus.subscribe("indicator.signal", indicator_signal_handler)
        
        # Publish klines update to trigger pattern recognition
        await self._publish_klines_update()
        
        # Wait for event processing
        logger.debug("Waiting for event processing")
        await asyncio.sleep(1.0)  # Increased wait time
        
        # Debug: Check direct event listeners
        logger.debug(f"DEBUG: Indicator signals received directly: {len(self.indicator_signals_received)}")
        for signal in self.indicator_signals_received:
            logger.debug(f"DEBUG: Signal: {signal.get('indicator')} - {signal.get('signal')}")
        
        # Use either the handler's collection or the direct listener's collection
        all_signals = indicator_signals + self.indicator_signals_received
        
        # Check if indicator signals were generated
        self.assertTrue(len(all_signals) > 0, "No indicator signals generated")
        
        # Check signal properties
        if len(all_signals) > 0:
            for signal in all_signals:
                self.assertIn("indicator", signal, "Signal missing indicator field")
                self.assertIn("symbol", signal, "Signal missing symbol field")
                self.assertIn("timeframe", signal, "Signal missing timeframe field")
                self.assertIn("signal", signal, "Signal missing signal field")
                self.assertIn("direction", signal, "Signal missing direction field")
                self.assertIn("confidence", signal, "Signal missing confidence field")
        
        logger.debug("test_pattern_recognition_flow completed")
    
    async def test_strategic_decision_visualization(self):
        """Test strategic decision visualization flow."""
        logger.debug("Starting test_strategic_decision_visualization")
        
        # Publish klines update first to have chart data
        await self._publish_klines_update()
        
        # Wait for event processing
        await asyncio.sleep(0.5)
        
        # Get initial markers count
        initial_chart_data = self.chart_visualization.get_chart_data(self.test_symbol, self.test_timeframe)
        initial_markers_count = len(initial_chart_data["markers"])
        
        # Publish strategic decision
        await self._publish_strategic_decision()
        
        # Wait for event processing
        await asyncio.sleep(0.5)
        
        # Check if marker was added to chart
        updated_chart_data = self.chart_visualization.get_chart_data(self.test_symbol, self.test_timeframe)
        self.assertTrue(
            len(updated_chart_data["markers"]) > initial_markers_count,
            "Strategic decision marker not added to chart"
        )
        
        # Check marker properties
        latest_marker = updated_chart_data["markers"][-1]
        self.assertEqual(latest_marker["type"], "decision", "Marker type mismatch")
        self.assertEqual(latest_marker["decision_type"], "entry", "Decision type mismatch")
        self.assertEqual(latest_marker["direction"], "bullish", "Direction mismatch")
        
        logger.debug("test_strategic_decision_visualization completed")
    
    async def test_pattern_detected_visualization(self):
        """Test pattern detected visualization flow."""
        logger.debug("Starting test_pattern_detected_visualization")
        
        # Publish klines update first to have chart data
        await self._publish_klines_update()
        
        # Wait for event processing
        await asyncio.sleep(0.5)
        
        # Get initial markers count
        initial_chart_data = self.chart_visualization.get_chart_data(self.test_symbol, self.test_timeframe)
        initial_markers_count = len(initial_chart_data["markers"])
        
        # Publish pattern detected
        await self._publish_pattern_detected()
        
        # Wait for event processing
        await asyncio.sleep(0.5)
        
        # Check if marker was added to chart
        updated_chart_data = self.chart_visualization.get_chart_data(self.test_symbol, self.test_timeframe)
        self.assertTrue(
            len(updated_chart_data["markers"]) > initial_markers_count,
            "Pattern detected marker not added to chart"
        )
        
        # Check marker properties
        latest_marker = updated_chart_data["markers"][-1]
        self.assertEqual(latest_marker["type"], "pattern", "Marker type mismatch")
        self.assertEqual(latest_marker["pattern_type"], "double_bottom", "Pattern type mismatch")
        self.assertEqual(latest_marker["direction"], "bullish", "Direction mismatch")
        
        logger.debug("test_pattern_detected_visualization completed")
    
    async def test_end_to_end_visualization_flow(self):
        """Test end-to-end visualization flow."""
        logger.debug("Starting test_end_to_end_visualization_flow")
        
        # Create a mock LLM Overseer with debug logging
        class MockLLMOverseer:
            def __init__(self):
                self.strategic_decisions = []
                self.pattern_detections = []
                logger.debug("MockLLMOverseer initialized")
            
            def update_market_data(self, data):
                logger.debug(f"MockLLMOverseer.update_market_data called with: {data}")
                if "pattern_detected" in data:
                    self.pattern_detections.append(data)
                    logger.debug(f"Pattern detection added, total: {len(self.pattern_detections)}")
            
            def publish_strategic_decision(self, decision):
                logger.debug(f"MockLLMOverseer.publish_strategic_decision called with: {decision}")
                self.strategic_decisions.append(decision)
            
            def notify_pattern_detected(self, pattern):
                logger.debug(f"MockLLMOverseer.notify_pattern_detected called with: {pattern}")
                self.pattern_detections.append(pattern)
                logger.debug(f"Pattern detection added, total: {len(self.pattern_detections)}")
        
        mock_llm_overseer = MockLLMOverseer()
        self.llm_visualization_bridge.set_llm_overseer(mock_llm_overseer)
        logger.debug("MockLLMOverseer set to LLMVisualizationBridge")
        
        # Publish klines update to trigger pattern recognition
        await self._publish_klines_update()
        
        # Wait for event processing
        logger.debug("Waiting for event processing after klines update")
        await asyncio.sleep(1.0)  # Increased wait time
        
        # Debug: Check patterns detected directly
        logger.debug(f"DEBUG: Patterns detected directly: {len(self.patterns_detected)}")
        for pattern in self.patterns_detected:
            logger.debug(f"DEBUG: Pattern: {pattern.get('pattern_type')}")
        
        # Manually publish pattern detected to ensure bridge receives it
        logger.debug("Publishing pattern detected manually")
        await self._publish_pattern_detected()
        
        # Wait for event processing
        logger.debug("Waiting for event processing after pattern detected")
        await asyncio.sleep(1.0)  # Increased wait time
        
        # Debug: Check LLM Overseer state
        logger.debug(f"DEBUG: MockLLMOverseer pattern detections: {len(mock_llm_overseer.pattern_detections)}")
        
        # Check if pattern was forwarded to LLM Overseer
        self.assertTrue(len(mock_llm_overseer.pattern_detections) > 0, "Pattern not forwarded to LLM Overseer")
        
        # Publish strategic decision
        await self._publish_strategic_decision()
        
        # Wait for event processing
        await asyncio.sleep(0.5)
        
        # Check if chart was updated with both markers
        chart_data = self.chart_visualization.get_chart_data(self.test_symbol, self.test_timeframe)
        
        # Count markers by type
        decision_markers = [m for m in chart_data["markers"] if m["type"] == "decision"]
        pattern_markers = [m for m in chart_data["markers"] if m["type"] == "pattern"]
        
        logger.debug(f"Decision markers: {len(decision_markers)}")
        logger.debug(f"Pattern markers: {len(pattern_markers)}")
        
        self.assertTrue(len(decision_markers) > 0, "Decision markers not added to chart")
        self.assertTrue(len(pattern_markers) > 0, "Pattern markers not added to chart")
        
        logger.debug("test_end_to_end_visualization_flow completed")


if __name__ == "__main__":
    # Run tests
    print("Running visualization integration tests...")
    unittest.main()
