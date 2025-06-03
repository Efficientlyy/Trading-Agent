#!/usr/bin/env python
"""
Integration Test for LLM Strategic Overseer

This module provides tests for the integration between LLM Strategic Overseer
and the trading system core, visualization components, and data pipeline.
"""

import os
import sys
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("integration_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("integration_test")

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import components
from core.event_bus import EventBus
from data.unified_pipeline import UnifiedDataPipeline
from visualization.bridge import LLMVisualizationBridge
from integration.trading_system_connector import TradingSystemConnector
from core.llm_manager import TieredLLMManager
from core.context_manager import ContextManager
from core.token_tracker import TokenTracker
from config.config import Config

# Mock classes for testing
class MockFlashTrading:
    """Mock Flash Trading module for testing."""
    
    def __init__(self):
        """Initialize Mock Flash Trading."""
        self.active = False
        self.orders = []
        logger.info("Mock Flash Trading initialized")
    
    def start(self):
        """Start Flash Trading."""
        self.active = True
        logger.info("Mock Flash Trading started")
        return {"status": "started"}
    
    def stop(self):
        """Stop Flash Trading."""
        self.active = False
        logger.info("Mock Flash Trading stopped")
        return {"status": "stopped"}
    
    def place_order(self, symbol, side, quantity, price=None, order_type="limit"):
        """Place order."""
        order = {
            "order_id": f"order_{datetime.now().timestamp()}",
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "order_type": order_type,
            "status": "placed",
            "timestamp": datetime.now().isoformat()
        }
        self.orders.append(order)
        logger.info(f"Mock Flash Trading placed order: {order}")
        return order
    
    def cancel_order(self, order_id=None, symbol=None):
        """Cancel order."""
        cancelled = []
        for order in self.orders:
            if (order_id and order["order_id"] == order_id) or \
               (symbol and order["symbol"] == symbol):
                order["status"] = "cancelled"
                cancelled.append(order)
        
        logger.info(f"Mock Flash Trading cancelled orders: {cancelled}")
        return {"orders_cancelled": len(cancelled)}


class MockPaperTrading:
    """Mock Paper Trading module for testing."""
    
    def __init__(self):
        """Initialize Mock Paper Trading."""
        self.active = False
        self.orders = []
        self.balance = {
            "BTC": 1.0,
            "ETH": 10.0,
            "SOL": 100.0,
            "USDC": 50000.0
        }
        logger.info("Mock Paper Trading initialized")
    
    def start(self):
        """Start Paper Trading."""
        self.active = True
        logger.info("Mock Paper Trading started")
        return {"status": "started"}
    
    def stop(self):
        """Stop Paper Trading."""
        self.active = False
        logger.info("Mock Paper Trading stopped")
        return {"status": "stopped"}
    
    def place_order(self, symbol, side, quantity, price=None, order_type="limit"):
        """Place order."""
        order = {
            "order_id": f"order_{datetime.now().timestamp()}",
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "order_type": order_type,
            "status": "placed",
            "timestamp": datetime.now().isoformat()
        }
        self.orders.append(order)
        logger.info(f"Mock Paper Trading placed order: {order}")
        return order
    
    def cancel_order(self, order_id=None, symbol=None):
        """Cancel order."""
        cancelled = []
        for order in self.orders:
            if (order_id and order["order_id"] == order_id) or \
               (symbol and order["symbol"] == symbol):
                order["status"] = "cancelled"
                cancelled.append(order)
        
        logger.info(f"Mock Paper Trading cancelled orders: {cancelled}")
        return {"orders_cancelled": len(cancelled)}
    
    def get_balance(self):
        """Get balance."""
        return self.balance


class MockOrderBookAnalytics:
    """Mock Order Book Analytics module for testing."""
    
    def __init__(self):
        """Initialize Mock Order Book Analytics."""
        logger.info("Mock Order Book Analytics initialized")
    
    def analyze(self, symbol):
        """Analyze order book."""
        result = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "bid_ask_imbalance": 0.75,
            "depth_imbalance": 0.65,
            "pressure_direction": "buy",
            "liquidity_score": 0.85
        }
        logger.info(f"Mock Order Book Analytics analyzed {symbol}: {result}")
        return result


class MockTickDataProcessor:
    """Mock Tick Data Processor module for testing."""
    
    def __init__(self):
        """Initialize Mock Tick Data Processor."""
        logger.info("Mock Tick Data Processor initialized")
    
    def process(self, symbol):
        """Process tick data."""
        result = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "tick_direction": "up",
            "momentum_score": 0.68,
            "volatility": 0.12,
            "trend_strength": 0.75
        }
        logger.info(f"Mock Tick Data Processor processed {symbol}: {result}")
        return result


class MockLLMOverseer:
    """Mock LLM Overseer for testing."""
    
    def __init__(self):
        """Initialize Mock LLM Overseer."""
        self.context = {
            "system_status": {},
            "market_data": [],
            "trading_history": [],
            "performance_metrics": {},
            "risk_parameters": {}
        }
        logger.info("Mock LLM Overseer initialized")
    
    def update_system_status(self, status):
        """Update system status."""
        self.context["system_status"].update(status)
        logger.info(f"Mock LLM Overseer updated system status: {status}")
    
    def update_market_data(self, data):
        """Update market data."""
        self.context["market_data"].append(data)
        if len(self.context["market_data"]) > 100:
            self.context["market_data"] = self.context["market_data"][-100:]
        logger.info(f"Mock LLM Overseer updated market data: {data}")
    
    def update_trading_history(self, trade):
        """Update trading history."""
        self.context["trading_history"].append(trade)
        if len(self.context["trading_history"]) > 50:
            self.context["trading_history"] = self.context["trading_history"][-50:]
        logger.info(f"Mock LLM Overseer updated trading history: {trade}")
    
    def update_performance_metrics(self, metrics):
        """Update performance metrics."""
        self.context["performance_metrics"].update(metrics)
        logger.info(f"Mock LLM Overseer updated performance metrics: {metrics}")
    
    def update_risk_parameters(self, parameters):
        """Update risk parameters."""
        self.context["risk_parameters"].update(parameters)
        logger.info(f"Mock LLM Overseer updated risk parameters: {parameters}")


class MockChartComponent:
    """Mock Chart Component for testing."""
    
    def __init__(self):
        """Initialize Mock Chart Component."""
        self.charts = {}
        logger.info("Mock Chart Component initialized")
    
    def update_chart(self, symbol, data):
        """Update chart."""
        if symbol not in self.charts:
            self.charts[symbol] = []
        
        self.charts[symbol].append(data)
        logger.info(f"Mock Chart Component updated chart for {symbol}")
        return True
    
    def add_marker(self, symbol, marker):
        """Add marker to chart."""
        if symbol not in self.charts:
            self.charts[symbol] = []
        
        self.charts[symbol].append({"marker": marker})
        logger.info(f"Mock Chart Component added marker to chart for {symbol}: {marker}")
        return True


async def test_integration():
    """Test integration between components."""
    logger.info("Starting integration test")
    
    # Initialize components
    event_bus = EventBus()
    data_pipeline = UnifiedDataPipeline()
    llm_visualization_bridge = LLMVisualizationBridge()
    trading_system_connector = TradingSystemConnector()
    
    # Initialize mock components
    mock_llm_overseer = MockLLMOverseer()
    mock_flash_trading = MockFlashTrading()
    mock_paper_trading = MockPaperTrading()
    mock_order_book_analytics = MockOrderBookAnalytics()
    mock_tick_data_processor = MockTickDataProcessor()
    mock_chart_component = MockChartComponent()
    
    # Connect components
    data_pipeline.set_event_bus(event_bus)
    llm_visualization_bridge.set_event_bus(event_bus)
    llm_visualization_bridge.set_llm_overseer(mock_llm_overseer)
    trading_system_connector.set_event_bus(event_bus)
    trading_system_connector.set_llm_overseer(mock_llm_overseer)
    trading_system_connector.set_data_pipeline(data_pipeline)
    
    # Connect mock trading components
    trading_system_connector.connect_flash_trading(mock_flash_trading)
    trading_system_connector.connect_paper_trading(mock_paper_trading)
    trading_system_connector.connect_order_book_analytics(mock_order_book_analytics)
    trading_system_connector.connect_tick_data_processor(mock_tick_data_processor)
    
    # Start event processing
    event_bus.start_processing()
    
    # Start trading system connector
    await trading_system_connector.start()
    
    # Test 1: Market data flow
    logger.info("Test 1: Market data flow")
    
    # Publish market data event
    await event_bus.publish("trading.market_update", {
        "symbol": "BTC/USDC",
        "price": 50000.0,
        "volume": 100.0,
        "bid": 49990.0,
        "ask": 50010.0,
        "timestamp": datetime.now().isoformat()
    })
    
    # Wait for event processing
    await asyncio.sleep(0.1)
    
    # Verify data in pipeline
    market_data = data_pipeline.get_market_data("BTC/USDC")
    assert market_data is not None, "Market data not found in pipeline"
    assert market_data["symbol"] == "BTC/USDC", "Incorrect symbol in market data"
    
    # Verify data in LLM Overseer context
    assert len(mock_llm_overseer.context["market_data"]) > 0, "Market data not found in LLM Overseer context"
    assert mock_llm_overseer.context["market_data"][-1]["symbol"] == "BTC/USDC", "Incorrect symbol in LLM Overseer context"
    
    logger.info("Test 1 passed: Market data flow verified")
    
    # Test 2: Trading command execution
    logger.info("Test 2: Trading command execution")
    
    # Execute trade command
    result = await trading_system_connector.execute_command("execute_trade", {
        "symbol": "BTC/USDC",
        "side": "buy",
        "quantity": 0.1,
        "price": 50000.0,
        "order_type": "limit"
    })
    
    # Verify command result
    assert result["success"] is True, "Trade command failed"
    assert result["order"]["symbol"] == "BTC/USDC", "Incorrect symbol in order"
    assert result["order"]["side"] == "buy", "Incorrect side in order"
    
    # Verify order in LLM Overseer context
    assert len(mock_llm_overseer.context["trading_history"]) > 0, "Order not found in LLM Overseer context"
    assert mock_llm_overseer.context["trading_history"][-1]["symbol"] == "BTC/USDC", "Incorrect symbol in trading history"
    
    logger.info("Test 2 passed: Trading command execution verified")
    
    # Test 3: Strategic decision visualization
    logger.info("Test 3: Strategic decision visualization")
    
    # Register chart component with visualization bridge
    def update_chart_with_decision(decision):
        symbol = decision.get("symbol", "BTC/USDC")
        mock_chart_component.add_marker(symbol, {
            "type": "decision",
            "decision_type": decision.get("decision_type", "unknown"),
            "direction": decision.get("direction", "neutral"),
            "timestamp": decision.get("timestamp", datetime.now().isoformat())
        })
    
    subscription_id = llm_visualization_bridge.subscribe_to_strategic_decisions(update_chart_with_decision)
    
    # Publish strategic decision
    decision = {
        "decision_type": "entry",
        "symbol": "BTC/USDC",
        "direction": "bullish",
        "confidence": 0.85,
        "summary": "Enter long position based on bullish pattern",
        "timestamp": datetime.now().isoformat()
    }
    
    await llm_visualization_bridge.publish_strategic_decision(decision)
    
    # Wait for event processing
    await asyncio.sleep(0.1)
    
    # Verify decision marker in chart
    assert "BTC/USDC" in mock_chart_component.charts, "Chart for BTC/USDC not found"
    assert len(mock_chart_component.charts["BTC/USDC"]) > 0, "No markers in chart"
    assert "marker" in mock_chart_component.charts["BTC/USDC"][-1], "Marker not found in chart"
    assert mock_chart_component.charts["BTC/USDC"][-1]["marker"]["type"] == "decision", "Incorrect marker type"
    
    logger.info("Test 3 passed: Strategic decision visualization verified")
    
    # Test 4: Pattern recognition feedback loop
    logger.info("Test 4: Pattern recognition feedback loop")
    
    # Publish pattern recognition result
    pattern = {
        "pattern_type": "double_bottom",
        "symbol": "BTC/USDC",
        "timeframe": "1h",
        "confidence": 0.8,
        "direction": "bullish",
        "price_target": 52000.0,
        "stop_loss": 49000.0,
        "timestamp": datetime.now().isoformat()
    }
    
    await llm_visualization_bridge.publish_pattern_recognition(pattern)
    
    # Wait for event processing
    await asyncio.sleep(0.1)
    
    # Verify pattern in LLM Overseer context
    found_pattern = False
    for data in mock_llm_overseer.context["market_data"]:
        if "pattern_detected" in data and data["pattern_detected"] == "double_bottom":
            found_pattern = True
            break
    
    assert found_pattern, "Pattern not found in LLM Overseer context"
    
    logger.info("Test 4 passed: Pattern recognition feedback loop verified")
    
    # Test 5: Market analysis command
    logger.info("Test 5: Market analysis command")
    
    # Execute market analysis command
    result = await trading_system_connector.execute_command("market_analysis", {
        "symbol": "BTC/USDC",
        "analysis_type": "orderbook"
    })
    
    # Verify command result
    assert result["success"] is True, "Market analysis command failed"
    assert result["analysis_type"] == "orderbook", "Incorrect analysis type in result"
    assert "bid_ask_imbalance" in result["result"], "Bid-ask imbalance not found in result"
    
    logger.info("Test 5 passed: Market analysis command verified")
    
    # Stop components
    await trading_system_connector.stop()
    event_bus.stop_processing()
    
    logger.info("All integration tests passed")


if __name__ == "__main__":
    asyncio.run(test_integration())
