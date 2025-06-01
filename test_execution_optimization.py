#!/usr/bin/env python
"""
Test suite for Execution Optimization components.

This module provides comprehensive testing for the execution optimization
components, including order routing, latency profiling, and smart order types.
"""

import os
import time
import unittest
import logging
import json
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional
from unittest.mock import MagicMock, patch

# Import execution optimization components
from execution_optimization import (
    Order, OrderType, OrderSide, OrderStatus,
    LatencyProfiler, OrderRouter, SmartOrderRouter,
    ExecutionOptimizer, AsyncOrderRouter, AsyncExecutionOptimizer
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_execution_optimization')

class MockExchangeClient:
    """Mock exchange client for testing."""
    
    def __init__(self, latency: float = 0.001, fail_rate: float = 0.0):
        """Initialize the mock exchange client.
        
        Args:
            latency: Simulated network latency in seconds
            fail_rate: Rate of simulated failures (0.0 to 1.0)
        """
        self.latency = latency
        self.fail_rate = fail_rate
        self.orders = {}
    
    def submit_order(self, order: Order) -> Dict:
        """Submit an order to the mock exchange.
        
        Args:
            order: Order to submit
            
        Returns:
            Response dictionary
        """
        # Simulate network latency
        time.sleep(self.latency)
        
        # Simulate random failures
        if np.random.random() < self.fail_rate:
            raise Exception("Simulated exchange error")
        
        # Store order
        self.orders[order.id] = {
            'order': order,
            'status': OrderStatus.OPEN,
            'filled_quantity': 0.0,
            'average_fill_price': None
        }
        
        return {
            'status': OrderStatus.OPEN,
            'id': order.id
        }
    
    def cancel_order(self, order: Order) -> Dict:
        """Cancel an order on the mock exchange.
        
        Args:
            order: Order to cancel
            
        Returns:
            Response dictionary
        """
        # Simulate network latency
        time.sleep(self.latency)
        
        # Simulate random failures
        if np.random.random() < self.fail_rate:
            raise Exception("Simulated exchange error")
        
        # Check if order exists
        if order.id not in self.orders:
            raise Exception(f"Order {order.id} not found")
        
        # Update order status
        self.orders[order.id]['status'] = OrderStatus.CANCELED
        
        return {
            'status': OrderStatus.CANCELED
        }
    
    def get_order_status(self, order: Order) -> Dict:
        """Get the status of an order on the mock exchange.
        
        Args:
            order: Order to check
            
        Returns:
            Response dictionary
        """
        # Simulate network latency
        time.sleep(self.latency)
        
        # Simulate random failures
        if np.random.random() < self.fail_rate:
            raise Exception("Simulated exchange error")
        
        # Check if order exists
        if order.id not in self.orders:
            raise Exception(f"Order {order.id} not found")
        
        # Simulate order execution (50% chance of being filled)
        if self.orders[order.id]['status'] == OrderStatus.OPEN and np.random.random() < 0.5:
            self.orders[order.id]['status'] = OrderStatus.FILLED
            self.orders[order.id]['filled_quantity'] = order.quantity
            self.orders[order.id]['average_fill_price'] = 100.0  # Simulated price
        
        return {
            'status': self.orders[order.id]['status'],
            'filled_quantity': self.orders[order.id]['filled_quantity'],
            'average_fill_price': self.orders[order.id]['average_fill_price']
        }
    
    async def submit_order_async(self, order: Order) -> Dict:
        """Submit an order to the mock exchange asynchronously.
        
        Args:
            order: Order to submit
            
        Returns:
            Response dictionary
        """
        # Simulate network latency
        await asyncio.sleep(self.latency)
        
        # Simulate random failures
        if np.random.random() < self.fail_rate:
            raise Exception("Simulated exchange error")
        
        # Store order
        self.orders[order.id] = {
            'order': order,
            'status': OrderStatus.OPEN,
            'filled_quantity': 0.0,
            'average_fill_price': None
        }
        
        return {
            'status': OrderStatus.OPEN,
            'id': order.id
        }
    
    async def cancel_order_async(self, order: Order) -> Dict:
        """Cancel an order on the mock exchange asynchronously.
        
        Args:
            order: Order to cancel
            
        Returns:
            Response dictionary
        """
        # Simulate network latency
        await asyncio.sleep(self.latency)
        
        # Simulate random failures
        if np.random.random() < self.fail_rate:
            raise Exception("Simulated exchange error")
        
        # Check if order exists
        if order.id not in self.orders:
            raise Exception(f"Order {order.id} not found")
        
        # Update order status
        self.orders[order.id]['status'] = OrderStatus.CANCELED
        
        return {
            'status': OrderStatus.CANCELED
        }
    
    async def get_order_status_async(self, order: Order) -> Dict:
        """Get the status of an order on the mock exchange asynchronously.
        
        Args:
            order: Order to check
            
        Returns:
            Response dictionary
        """
        # Simulate network latency
        await asyncio.sleep(self.latency)
        
        # Simulate random failures
        if np.random.random() < self.fail_rate:
            raise Exception("Simulated exchange error")
        
        # Check if order exists
        if order.id not in self.orders:
            raise Exception(f"Order {order.id} not found")
        
        # Simulate order execution (50% chance of being filled)
        if self.orders[order.id]['status'] == OrderStatus.OPEN and np.random.random() < 0.5:
            self.orders[order.id]['status'] = OrderStatus.FILLED
            self.orders[order.id]['filled_quantity'] = order.quantity
            self.orders[order.id]['average_fill_price'] = 100.0  # Simulated price
        
        return {
            'status': self.orders[order.id]['status'],
            'filled_quantity': self.orders[order.id]['filled_quantity'],
            'average_fill_price': self.orders[order.id]['average_fill_price']
        }


class TestOrder(unittest.TestCase):
    """Test cases for the Order class."""
    
    def test_order_initialization(self):
        """Test order initialization."""
        order = Order(
            id="test_order",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            quantity=1.0
        )
        
        self.assertEqual(order.id, "test_order")
        self.assertEqual(order.symbol, "BTC/USD")
        self.assertEqual(order.side, OrderSide.BUY)
        self.assertEqual(order.type, OrderType.MARKET)
        self.assertEqual(order.quantity, 1.0)
        self.assertEqual(order.status, OrderStatus.PENDING)
        self.assertIsNotNone(order.created_at)
        self.assertIsNotNone(order.updated_at)
    
    def test_order_to_dict(self):
        """Test order to dictionary conversion."""
        order = Order(
            id="test_order",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            quantity=1.0,
            price=100.0
        )
        
        order_dict = order.to_dict()
        
        self.assertEqual(order_dict['id'], "test_order")
        self.assertEqual(order_dict['symbol'], "BTC/USD")
        self.assertEqual(order_dict['side'], "buy")
        self.assertEqual(order_dict['type'], "market")
        self.assertEqual(order_dict['quantity'], 1.0)
        self.assertEqual(order_dict['price'], 100.0)
        self.assertEqual(order_dict['status'], "pending")
    
    def test_order_from_dict(self):
        """Test order from dictionary conversion."""
        order_dict = {
            'id': "test_order",
            'symbol': "BTC/USD",
            'side': "buy",
            'type': "limit",
            'quantity': 1.0,
            'price': 100.0,
            'status': "open"
        }
        
        order = Order.from_dict(order_dict)
        
        self.assertEqual(order.id, "test_order")
        self.assertEqual(order.symbol, "BTC/USD")
        self.assertEqual(order.side, OrderSide.BUY)
        self.assertEqual(order.type, OrderType.LIMIT)
        self.assertEqual(order.quantity, 1.0)
        self.assertEqual(order.price, 100.0)
        self.assertEqual(order.status, OrderStatus.OPEN)


class TestLatencyProfiler(unittest.TestCase):
    """Test cases for the LatencyProfiler class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.metrics_file = "test_latency_metrics.json"
        self.profiler = LatencyProfiler(metrics_file=self.metrics_file)
    
    def tearDown(self):
        """Tear down test fixtures."""
        if os.path.exists(self.metrics_file):
            os.remove(self.metrics_file)
    
    def test_timer_functionality(self):
        """Test timer functionality."""
        # Start timer
        self.profiler.start_timer("test_operation")
        
        # Sleep for a short time
        time.sleep(0.01)
        
        # Stop timer and record latency
        latency = self.profiler.stop_timer("test_operation", "test_category")
        
        # Check that latency is positive
        self.assertGreater(latency, 0)
        
        # Check that latency is recorded
        metrics = self.profiler.get_metrics()
        self.assertIn("test_category", metrics)
        self.assertEqual(len(metrics["test_category"]["count"]), 1)
    
    def test_multiple_timers(self):
        """Test multiple timers."""
        # Start and stop multiple timers
        for i in range(10):
            self.profiler.start_timer(f"test_operation_{i}")
            time.sleep(0.001)
            self.profiler.stop_timer(f"test_operation_{i}", "test_category")
        
        # Check that all latencies are recorded
        metrics = self.profiler.get_metrics()
        self.assertIn("test_category", metrics)
        self.assertEqual(metrics["test_category"]["count"], 10)
    
    def test_save_metrics(self):
        """Test saving metrics to file."""
        # Record some latencies
        for i in range(5):
            self.profiler.start_timer(f"test_operation_{i}")
            time.sleep(0.001)
            self.profiler.stop_timer(f"test_operation_{i}", "test_category")
        
        # Save metrics
        self.profiler.save_metrics()
        
        # Check that metrics file exists
        self.assertTrue(os.path.exists(self.metrics_file))
        
        # Load metrics from file
        with open(self.metrics_file, 'r') as f:
            loaded_metrics = json.load(f)
        
        # Check that metrics are correct
        self.assertIn("test_category", loaded_metrics)
        self.assertEqual(loaded_metrics["test_category"]["count"], 5)


class TestOrderRouter(unittest.TestCase):
    """Test cases for the OrderRouter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.profiler = LatencyProfiler(metrics_file="test_latency_metrics.json")
        self.router = OrderRouter(latency_profiler=self.profiler)
        self.exchange = MockExchangeClient()
        self.router.register_exchange("mock_exchange", self.exchange)
    
    def tearDown(self):
        """Tear down test fixtures."""
        if os.path.exists("test_latency_metrics.json"):
            os.remove("test_latency_metrics.json")
    
    def test_register_exchange(self):
        """Test registering an exchange."""
        # Register a new exchange
        new_exchange = MockExchangeClient()
        self.router.register_exchange("new_exchange", new_exchange)
        
        # Check that exchange is registered
        self.assertIn("new_exchange", self.router.exchanges)
        self.assertEqual(self.router.exchanges["new_exchange"], new_exchange)
    
    def test_submit_order(self):
        """Test submitting an order."""
        # Create an order
        order = Order(
            id="test_order",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            quantity=1.0
        )
        
        # Submit order
        result = self.router.submit_order(order, exchange_id="mock_exchange")
        
        # Check that order is submitted
        self.assertEqual(result.status, OrderStatus.OPEN)
        self.assertEqual(result.exchange_id, "mock_exchange")
    
    def test_submit_order_unknown_exchange(self):
        """Test submitting an order to an unknown exchange."""
        # Create an order
        order = Order(
            id="test_order",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            quantity=1.0
        )
        
        # Submit order to unknown exchange
        result = self.router.submit_order(order, exchange_id="unknown_exchange")
        
        # Check that order is rejected
        self.assertEqual(result.status, OrderStatus.REJECTED)
    
    def test_submit_order_with_retries(self):
        """Test submitting an order with retries."""
        # Create an order
        order = Order(
            id="test_order",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            quantity=1.0
        )
        
        # Create exchange with high failure rate
        failing_exchange = MockExchangeClient(fail_rate=0.9)
        self.router.register_exchange("failing_exchange", failing_exchange)
        
        # Set max retries
        self.router.max_retries = 10
        
        # Submit order
        result = self.router.submit_order(order, exchange_id="failing_exchange")
        
        # Check that order is submitted (eventually)
        self.assertIn(result.status, [OrderStatus.OPEN, OrderStatus.REJECTED])
    
    def test_cancel_order(self):
        """Test canceling an order."""
        # Create and submit an order
        order = Order(
            id="test_order",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            quantity=1.0
        )
        
        order = self.router.submit_order(order, exchange_id="mock_exchange")
        
        # Cancel order
        result = self.router.cancel_order(order)
        
        # Check that order is canceled
        self.assertEqual(result.status, OrderStatus.CANCELED)
    
    def test_get_order_status(self):
        """Test getting order status."""
        # Create and submit an order
        order = Order(
            id="test_order",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            quantity=1.0
        )
        
        order = self.router.submit_order(order, exchange_id="mock_exchange")
        
        # Get order status
        result = self.router.get_order_status(order)
        
        # Check that status is returned
        self.assertIn(result.status, [OrderStatus.OPEN, OrderStatus.FILLED])


class TestSmartOrderRouter(unittest.TestCase):
    """Test cases for the SmartOrderRouter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.profiler = LatencyProfiler(metrics_file="test_latency_metrics.json")
        self.order_router = OrderRouter(latency_profiler=self.profiler)
        self.exchange = MockExchangeClient()
        self.order_router.register_exchange("mock_exchange", self.exchange)
        self.smart_router = SmartOrderRouter(order_router=self.order_router, latency_profiler=self.profiler)
    
    def tearDown(self):
        """Tear down test fixtures."""
        if os.path.exists("test_latency_metrics.json"):
            os.remove("test_latency_metrics.json")
    
    def test_route_market_order(self):
        """Test routing a market order."""
        # Create a market order
        order = Order(
            id="test_market_order",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            quantity=1.0
        )
        
        # Route order
        results = self.smart_router.route_order(order)
        
        # Check that order is routed
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].status, OrderStatus.OPEN)
    
    def test_route_limit_order(self):
        """Test routing a limit order."""
        # Create a limit order
        order = Order(
            id="test_limit_order",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            quantity=1.0,
            price=100.0
        )
        
        # Route order
        results = self.smart_router.route_order(order)
        
        # Check that order is routed
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].status, OrderStatus.OPEN)
    
    def test_route_iceberg_order(self):
        """Test routing an iceberg order."""
        # Create an iceberg order
        order = Order(
            id="test_iceberg_order",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            type=OrderType.ICEBERG,
            quantity=10.0,
            price=100.0,
            metadata={'display_size': 2.0}
        )
        
        # Route order
        results = self.smart_router.route_order(order)
        
        # Check that order is split into child orders
        self.assertEqual(len(results), 5)  # 10.0 / 2.0 = 5 child orders
        for result in results:
            self.assertEqual(result.status, OrderStatus.OPEN)
    
    def test_route_twap_order(self):
        """Test routing a TWAP order."""
        # Create a TWAP order with short duration
        order = Order(
            id="test_twap_order",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            type=OrderType.TWAP,
            quantity=3.0,
            price=100.0,
            metadata={'duration': 0.3, 'num_slices': 3}
        )
        
        # Route order
        results = self.smart_router.route_order(order)
        
        # Check that order is split into child orders
        self.assertEqual(len(results), 3)  # 3 slices
        for result in results:
            self.assertEqual(result.status, OrderStatus.OPEN)
    
    def test_route_vwap_order(self):
        """Test routing a VWAP order."""
        # Create a VWAP order with short duration
        order = Order(
            id="test_vwap_order",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            type=OrderType.VWAP,
            quantity=3.0,
            price=100.0,
            metadata={'duration': 0.3, 'num_slices': 3}
        )
        
        # Route order
        results = self.smart_router.route_order(order)
        
        # Check that order is split into child orders
        self.assertEqual(len(results), 3)  # 3 slices
        for result in results:
            self.assertEqual(result.status, OrderStatus.OPEN)
    
    def test_route_smart_order(self):
        """Test routing a smart order."""
        # Create a smart order
        order = Order(
            id="test_smart_order",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            type=OrderType.SMART,
            quantity=1.0,
            price=100.0,
            metadata={'strategy': 'adaptive'}
        )
        
        # Route order
        results = self.smart_router.route_order(order)
        
        # Check that order is routed
        self.assertGreater(len(results), 0)
        for result in results:
            self.assertEqual(result.status, OrderStatus.OPEN)


class TestExecutionOptimizer(unittest.TestCase):
    """Test cases for the ExecutionOptimizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.profiler = LatencyProfiler(metrics_file="test_latency_metrics.json")
        self.order_router = OrderRouter(latency_profiler=self.profiler)
        self.exchange = MockExchangeClient()
        self.order_router.register_exchange("mock_exchange", self.exchange)
        self.smart_router = SmartOrderRouter(order_router=self.order_router, latency_profiler=self.profiler)
        self.optimizer = ExecutionOptimizer(smart_order_router=self.smart_router, latency_profiler=self.profiler)
    
    def tearDown(self):
        """Tear down test fixtures."""
        if os.path.exists("test_latency_metrics.json"):
            os.remove("test_latency_metrics.json")
        
        # Stop optimizer if running
        if self.optimizer.running:
            self.optimizer.stop()
    
    def test_start_stop(self):
        """Test starting and stopping the optimizer."""
        # Start optimizer
        self.optimizer.start()
        self.assertTrue(self.optimizer.running)
        
        # Stop optimizer
        self.optimizer.stop()
        self.assertFalse(self.optimizer.running)
    
    def test_submit_order(self):
        """Test submitting an order for execution."""
        # Start optimizer
        self.optimizer.start()
        
        # Create an order
        order = Order(
            id="test_order",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            quantity=1.0
        )
        
        # Submit order
        order_id = self.optimizer.submit_order(order)
        
        # Check that order is queued
        self.assertEqual(order_id, "test_order")
        self.assertIn(order_id, self.optimizer.order_history)
        self.assertEqual(self.optimizer.order_history[order_id]['status'], 'queued')
        
        # Wait for order to be processed
        time.sleep(0.5)
        
        # Check that order is executed
        self.assertEqual(self.optimizer.order_history[order_id]['status'], 'executed')
    
    def test_get_order_status(self):
        """Test getting order status."""
        # Start optimizer
        self.optimizer.start()
        
        # Create and submit an order
        order = Order(
            id="test_order",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            quantity=1.0
        )
        
        order_id = self.optimizer.submit_order(order)
        
        # Get order status
        status = self.optimizer.get_order_status(order_id)
        
        # Check that status is returned
        self.assertIn(status['status'], ['queued', 'processing', 'executed'])
        
        # Wait for order to be processed
        time.sleep(0.5)
        
        # Get updated status
        status = self.optimizer.get_order_status(order_id)
        
        # Check that status is updated
        self.assertEqual(status['status'], 'executed')
    
    def test_unknown_order_status(self):
        """Test getting status of unknown order."""
        # Get status of unknown order
        status = self.optimizer.get_order_status("unknown_order")
        
        # Check that status is unknown
        self.assertEqual(status['status'], 'unknown')


class TestAsyncOrderRouter(unittest.TestCase):
    """Test cases for the AsyncOrderRouter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.profiler = LatencyProfiler(metrics_file="test_latency_metrics.json")
        self.router = AsyncOrderRouter(latency_profiler=self.profiler)
        self.exchange = MockExchangeClient()
        self.router.register_exchange("mock_exchange", self.exchange)
    
    def tearDown(self):
        """Tear down test fixtures."""
        if os.path.exists("test_latency_metrics.json"):
            os.remove("test_latency_metrics.json")
    
    def test_register_exchange(self):
        """Test registering an exchange."""
        # Register a new exchange
        new_exchange = MockExchangeClient()
        self.router.register_exchange("new_exchange", new_exchange)
        
        # Check that exchange is registered
        self.assertIn("new_exchange", self.router.exchanges)
        self.assertEqual(self.router.exchanges["new_exchange"], new_exchange)
    
    async def async_test_submit_order(self):
        """Test submitting an order asynchronously."""
        # Create an order
        order = Order(
            id="test_order",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            quantity=1.0
        )
        
        # Submit order
        result = await self.router.submit_order(order, exchange_id="mock_exchange")
        
        # Check that order is submitted
        self.assertEqual(result.status, OrderStatus.OPEN)
        self.assertEqual(result.exchange_id, "mock_exchange")
        
        return result
    
    async def async_test_cancel_order(self):
        """Test canceling an order asynchronously."""
        # Create and submit an order
        order = await self.async_test_submit_order()
        
        # Cancel order
        result = await self.router.cancel_order(order)
        
        # Check that order is canceled
        self.assertEqual(result.status, OrderStatus.CANCELED)
    
    async def async_test_get_order_status(self):
        """Test getting order status asynchronously."""
        # Create and submit an order
        order = await self.async_test_submit_order()
        
        # Get order status
        result = await self.router.get_order_status(order)
        
        # Check that status is returned
        self.assertIn(result.status, [OrderStatus.OPEN, OrderStatus.FILLED])
    
    def test_submit_order(self):
        """Test submitting an order."""
        asyncio.run(self.async_test_submit_order())
    
    def test_cancel_order(self):
        """Test canceling an order."""
        asyncio.run(self.async_test_cancel_order())
    
    def test_get_order_status(self):
        """Test getting order status."""
        asyncio.run(self.async_test_get_order_status())


class TestAsyncExecutionOptimizer(unittest.TestCase):
    """Test cases for the AsyncExecutionOptimizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.profiler = LatencyProfiler(metrics_file="test_latency_metrics.json")
        self.router = AsyncOrderRouter(latency_profiler=self.profiler)
        self.exchange = MockExchangeClient()
        self.router.register_exchange("mock_exchange", self.exchange)
        self.optimizer = AsyncExecutionOptimizer(async_order_router=self.router, latency_profiler=self.profiler)
    
    def tearDown(self):
        """Tear down test fixtures."""
        if os.path.exists("test_latency_metrics.json"):
            os.remove("test_latency_metrics.json")
    
    async def async_test_start_stop(self):
        """Test starting and stopping the optimizer asynchronously."""
        # Start optimizer
        await self.optimizer.start()
        self.assertTrue(self.optimizer.running)
        
        # Stop optimizer
        await self.optimizer.stop()
        self.assertFalse(self.optimizer.running)
    
    async def async_test_submit_order(self):
        """Test submitting an order for execution asynchronously."""
        # Start optimizer
        await self.optimizer.start()
        
        # Create an order
        order = Order(
            id="test_order",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            quantity=1.0
        )
        
        # Submit order
        order_id = await self.optimizer.submit_order(order)
        
        # Check that order is queued
        self.assertEqual(order_id, "test_order")
        self.assertIn(order_id, self.optimizer.order_history)
        self.assertEqual(self.optimizer.order_history[order_id]['status'], 'queued')
        
        # Wait for order to be processed
        await asyncio.sleep(0.5)
        
        # Check that order is executed
        self.assertEqual(self.optimizer.order_history[order_id]['status'], 'executed')
        
        # Stop optimizer
        await self.optimizer.stop()
    
    async def async_test_get_order_status(self):
        """Test getting order status asynchronously."""
        # Start optimizer
        await self.optimizer.start()
        
        # Create and submit an order
        order = Order(
            id="test_order",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            quantity=1.0
        )
        
        order_id = await self.optimizer.submit_order(order)
        
        # Get order status
        status = self.optimizer.get_order_status(order_id)
        
        # Check that status is returned
        self.assertIn(status['status'], ['queued', 'processing', 'executed'])
        
        # Wait for order to be processed
        await asyncio.sleep(0.5)
        
        # Get updated status
        status = self.optimizer.get_order_status(order_id)
        
        # Check that status is updated
        self.assertEqual(status['status'], 'executed')
        
        # Stop optimizer
        await self.optimizer.stop()
    
    def test_start_stop(self):
        """Test starting and stopping the optimizer."""
        asyncio.run(self.async_test_start_stop())
    
    def test_submit_order(self):
        """Test submitting an order for execution."""
        asyncio.run(self.async_test_submit_order())
    
    def test_get_order_status(self):
        """Test getting order status."""
        asyncio.run(self.async_test_get_order_status())


class TestPerformance(unittest.TestCase):
    """Test cases for performance benchmarking."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.profiler = LatencyProfiler(metrics_file="test_latency_metrics.json")
        
        # Synchronous components
        self.order_router = OrderRouter(latency_profiler=self.profiler)
        self.exchange = MockExchangeClient(latency=0.0001)  # Low latency for benchmarking
        self.order_router.register_exchange("mock_exchange", self.exchange)
        self.smart_router = SmartOrderRouter(order_router=self.order_router, latency_profiler=self.profiler)
        self.optimizer = ExecutionOptimizer(smart_order_router=self.smart_router, latency_profiler=self.profiler)
        
        # Asynchronous components
        self.async_router = AsyncOrderRouter(latency_profiler=self.profiler)
        self.async_router.register_exchange("mock_exchange", self.exchange)
        self.async_optimizer = AsyncExecutionOptimizer(async_order_router=self.async_router, latency_profiler=self.profiler)
    
    def tearDown(self):
        """Tear down test fixtures."""
        if os.path.exists("test_latency_metrics.json"):
            os.remove("test_latency_metrics.json")
        
        # Stop optimizers if running
        if self.optimizer.running:
            self.optimizer.stop()
    
    def test_order_router_performance(self):
        """Test order router performance."""
        # Create an order
        order = Order(
            id="test_order",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            quantity=1.0
        )
        
        # Benchmark order submission
        num_orders = 100
        start_time = time.time()
        
        for i in range(num_orders):
            order.id = f"test_order_{i}"
            self.order_router.submit_order(order, exchange_id="mock_exchange")
        
        end_time = time.time()
        
        # Calculate throughput
        elapsed_time = end_time - start_time
        throughput = num_orders / elapsed_time
        
        # Log results
        logger.info(f"Order router throughput: {throughput:.2f} orders/second")
        
        # Check that throughput is reasonable
        self.assertGreater(throughput, 10)  # At least 10 orders per second
    
    def test_smart_order_router_performance(self):
        """Test smart order router performance."""
        # Create an order
        order = Order(
            id="test_order",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            quantity=1.0
        )
        
        # Benchmark order routing
        num_orders = 10
        start_time = time.time()
        
        for i in range(num_orders):
            order.id = f"test_order_{i}"
            self.smart_router.route_order(order)
        
        end_time = time.time()
        
        # Calculate throughput
        elapsed_time = end_time - start_time
        throughput = num_orders / elapsed_time
        
        # Log results
        logger.info(f"Smart order router throughput: {throughput:.2f} orders/second")
        
        # Check that throughput is reasonable
        self.assertGreater(throughput, 1)  # At least 1 order per second
    
    async def async_test_async_order_router_performance(self):
        """Test async order router performance."""
        # Create an order
        order = Order(
            id="test_order",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            quantity=1.0
        )
        
        # Benchmark order submission
        num_orders = 100
        start_time = time.time()
        
        # Create tasks
        tasks = []
        for i in range(num_orders):
            order_copy = Order(
                id=f"test_order_{i}",
                symbol=order.symbol,
                side=order.side,
                type=order.type,
                quantity=order.quantity
            )
            task = self.async_router.submit_order(order_copy, exchange_id="mock_exchange")
            tasks.append(task)
        
        # Wait for all tasks to complete
        await asyncio.gather(*tasks)
        
        end_time = time.time()
        
        # Calculate throughput
        elapsed_time = end_time - start_time
        throughput = num_orders / elapsed_time
        
        # Log results
        logger.info(f"Async order router throughput: {throughput:.2f} orders/second")
        
        # Check that throughput is reasonable
        self.assertGreater(throughput, 10)  # At least 10 orders per second
    
    def test_async_order_router_performance(self):
        """Test async order router performance."""
        asyncio.run(self.async_test_async_order_router_performance())
    
    async def async_test_async_execution_optimizer_performance(self):
        """Test async execution optimizer performance."""
        # Start optimizer
        await self.async_optimizer.start()
        
        # Create an order
        order = Order(
            id="test_order",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            quantity=1.0
        )
        
        # Benchmark order submission
        num_orders = 10
        start_time = time.time()
        
        # Create tasks
        tasks = []
        for i in range(num_orders):
            order_copy = Order(
                id=f"test_order_{i}",
                symbol=order.symbol,
                side=order.side,
                type=order.type,
                quantity=order.quantity
            )
            task = self.async_optimizer.submit_order(order_copy)
            tasks.append(task)
        
        # Wait for all tasks to complete
        await asyncio.gather(*tasks)
        
        # Wait for orders to be processed
        await asyncio.sleep(1.0)
        
        end_time = time.time()
        
        # Calculate throughput
        elapsed_time = end_time - start_time
        throughput = num_orders / elapsed_time
        
        # Log results
        logger.info(f"Async execution optimizer throughput: {throughput:.2f} orders/second")
        
        # Check that throughput is reasonable
        self.assertGreater(throughput, 1)  # At least 1 order per second
        
        # Stop optimizer
        await self.async_optimizer.stop()
    
    def test_async_execution_optimizer_performance(self):
        """Test async execution optimizer performance."""
        asyncio.run(self.async_test_async_execution_optimizer_performance())


class TestEdgeCases(unittest.TestCase):
    """Test cases for edge cases."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.profiler = LatencyProfiler(metrics_file="test_latency_metrics.json")
        self.order_router = OrderRouter(latency_profiler=self.profiler)
        self.exchange = MockExchangeClient()
        self.order_router.register_exchange("mock_exchange", self.exchange)
        self.smart_router = SmartOrderRouter(order_router=self.order_router, latency_profiler=self.profiler)
        self.optimizer = ExecutionOptimizer(smart_order_router=self.smart_router, latency_profiler=self.profiler)
    
    def tearDown(self):
        """Tear down test fixtures."""
        if os.path.exists("test_latency_metrics.json"):
            os.remove("test_latency_metrics.json")
        
        # Stop optimizer if running
        if self.optimizer.running:
            self.optimizer.stop()
    
    def test_zero_quantity_order(self):
        """Test order with zero quantity."""
        # Create an order with zero quantity
        order = Order(
            id="test_order",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            quantity=0.0
        )
        
        # Submit order
        result = self.order_router.submit_order(order, exchange_id="mock_exchange")
        
        # Check that order is submitted (exchange should handle validation)
        self.assertEqual(result.status, OrderStatus.OPEN)
    
    def test_negative_quantity_order(self):
        """Test order with negative quantity."""
        # Create an order with negative quantity
        order = Order(
            id="test_order",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            quantity=-1.0
        )
        
        # Submit order
        result = self.order_router.submit_order(order, exchange_id="mock_exchange")
        
        # Check that order is submitted (exchange should handle validation)
        self.assertEqual(result.status, OrderStatus.OPEN)
    
    def test_negative_price_order(self):
        """Test order with negative price."""
        # Create an order with negative price
        order = Order(
            id="test_order",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            quantity=1.0,
            price=-100.0
        )
        
        # Submit order
        result = self.order_router.submit_order(order, exchange_id="mock_exchange")
        
        # Check that order is submitted (exchange should handle validation)
        self.assertEqual(result.status, OrderStatus.OPEN)
    
    def test_empty_symbol_order(self):
        """Test order with empty symbol."""
        # Create an order with empty symbol
        order = Order(
            id="test_order",
            symbol="",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            quantity=1.0
        )
        
        # Submit order
        result = self.order_router.submit_order(order, exchange_id="mock_exchange")
        
        # Check that order is submitted (exchange should handle validation)
        self.assertEqual(result.status, OrderStatus.OPEN)
    
    def test_high_failure_rate(self):
        """Test with high exchange failure rate."""
        # Create exchange with high failure rate
        failing_exchange = MockExchangeClient(fail_rate=1.0)  # 100% failure rate
        self.order_router.register_exchange("failing_exchange", failing_exchange)
        
        # Set max retries
        self.order_router.max_retries = 3
        
        # Create an order
        order = Order(
            id="test_order",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            quantity=1.0
        )
        
        # Submit order
        result = self.order_router.submit_order(order, exchange_id="failing_exchange")
        
        # Check that order is rejected after max retries
        self.assertEqual(result.status, OrderStatus.REJECTED)
    
    def test_high_latency(self):
        """Test with high exchange latency."""
        # Create exchange with high latency
        slow_exchange = MockExchangeClient(latency=0.1)  # 100ms latency
        self.order_router.register_exchange("slow_exchange", slow_exchange)
        
        # Set timeout
        self.order_router.timeout = 0.05  # 50ms timeout
        
        # Create an order
        order = Order(
            id="test_order",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            quantity=1.0
        )
        
        # Submit order
        result = self.order_router.submit_order(order, exchange_id="slow_exchange")
        
        # Check that order is rejected due to timeout
        self.assertEqual(result.status, OrderStatus.REJECTED)


class TestIntegration(unittest.TestCase):
    """Test cases for integration testing."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.profiler = LatencyProfiler(metrics_file="test_latency_metrics.json")
        self.order_router = OrderRouter(latency_profiler=self.profiler)
        self.exchange = MockExchangeClient()
        self.order_router.register_exchange("mock_exchange", self.exchange)
        self.smart_router = SmartOrderRouter(order_router=self.order_router, latency_profiler=self.profiler)
        self.optimizer = ExecutionOptimizer(smart_order_router=self.smart_router, latency_profiler=self.profiler)
    
    def tearDown(self):
        """Tear down test fixtures."""
        if os.path.exists("test_latency_metrics.json"):
            os.remove("test_latency_metrics.json")
        
        # Stop optimizer if running
        if self.optimizer.running:
            self.optimizer.stop()
    
    def test_end_to_end_market_order(self):
        """Test end-to-end market order execution."""
        # Start optimizer
        self.optimizer.start()
        
        # Create a market order
        order = Order(
            id="test_market_order",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            quantity=1.0
        )
        
        # Submit order
        order_id = self.optimizer.submit_order(order)
        
        # Wait for order to be processed
        time.sleep(0.5)
        
        # Check order status
        status = self.optimizer.get_order_status(order_id)
        
        # Check that order is executed
        self.assertEqual(status['status'], 'executed')
        self.assertEqual(len(status['executed_orders']), 1)
        self.assertIn(status['executed_orders'][0].status, [OrderStatus.OPEN, OrderStatus.FILLED])
    
    def test_end_to_end_limit_order(self):
        """Test end-to-end limit order execution."""
        # Start optimizer
        self.optimizer.start()
        
        # Create a limit order
        order = Order(
            id="test_limit_order",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            quantity=1.0,
            price=100.0
        )
        
        # Submit order
        order_id = self.optimizer.submit_order(order)
        
        # Wait for order to be processed
        time.sleep(0.5)
        
        # Check order status
        status = self.optimizer.get_order_status(order_id)
        
        # Check that order is executed
        self.assertEqual(status['status'], 'executed')
        self.assertEqual(len(status['executed_orders']), 1)
        self.assertIn(status['executed_orders'][0].status, [OrderStatus.OPEN, OrderStatus.FILLED])
    
    def test_end_to_end_iceberg_order(self):
        """Test end-to-end iceberg order execution."""
        # Start optimizer
        self.optimizer.start()
        
        # Create an iceberg order
        order = Order(
            id="test_iceberg_order",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            type=OrderType.ICEBERG,
            quantity=10.0,
            price=100.0,
            metadata={'display_size': 2.0}
        )
        
        # Submit order
        order_id = self.optimizer.submit_order(order)
        
        # Wait for order to be processed
        time.sleep(1.0)
        
        # Check order status
        status = self.optimizer.get_order_status(order_id)
        
        # Check that order is executed
        self.assertEqual(status['status'], 'executed')
        self.assertEqual(len(status['executed_orders']), 5)  # 10.0 / 2.0 = 5 child orders
        for executed_order in status['executed_orders']:
            self.assertIn(executed_order.status, [OrderStatus.OPEN, OrderStatus.FILLED])
    
    def test_end_to_end_twap_order(self):
        """Test end-to-end TWAP order execution."""
        # Start optimizer
        self.optimizer.start()
        
        # Create a TWAP order with short duration
        order = Order(
            id="test_twap_order",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            type=OrderType.TWAP,
            quantity=3.0,
            price=100.0,
            metadata={'duration': 0.3, 'num_slices': 3}
        )
        
        # Submit order
        order_id = self.optimizer.submit_order(order)
        
        # Wait for order to be processed
        time.sleep(1.0)
        
        # Check order status
        status = self.optimizer.get_order_status(order_id)
        
        # Check that order is executed
        self.assertEqual(status['status'], 'executed')
        self.assertEqual(len(status['executed_orders']), 3)  # 3 slices
        for executed_order in status['executed_orders']:
            self.assertIn(executed_order.status, [OrderStatus.OPEN, OrderStatus.FILLED])
    
    def test_end_to_end_vwap_order(self):
        """Test end-to-end VWAP order execution."""
        # Start optimizer
        self.optimizer.start()
        
        # Create a VWAP order with short duration
        order = Order(
            id="test_vwap_order",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            type=OrderType.VWAP,
            quantity=3.0,
            price=100.0,
            metadata={'duration': 0.3, 'num_slices': 3}
        )
        
        # Submit order
        order_id = self.optimizer.submit_order(order)
        
        # Wait for order to be processed
        time.sleep(1.0)
        
        # Check order status
        status = self.optimizer.get_order_status(order_id)
        
        # Check that order is executed
        self.assertEqual(status['status'], 'executed')
        self.assertEqual(len(status['executed_orders']), 3)  # 3 slices
        for executed_order in status['executed_orders']:
            self.assertIn(executed_order.status, [OrderStatus.OPEN, OrderStatus.FILLED])
    
    def test_end_to_end_smart_order(self):
        """Test end-to-end smart order execution."""
        # Start optimizer
        self.optimizer.start()
        
        # Create a smart order
        order = Order(
            id="test_smart_order",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            type=OrderType.SMART,
            quantity=1.0,
            price=100.0,
            metadata={'strategy': 'adaptive'}
        )
        
        # Submit order
        order_id = self.optimizer.submit_order(order)
        
        # Wait for order to be processed
        time.sleep(1.0)
        
        # Check order status
        status = self.optimizer.get_order_status(order_id)
        
        # Check that order is executed
        self.assertEqual(status['status'], 'executed')
        self.assertGreater(len(status['executed_orders']), 0)
        for executed_order in status['executed_orders']:
            self.assertIn(executed_order.status, [OrderStatus.OPEN, OrderStatus.FILLED])


if __name__ == "__main__":
    # Create output directory for test results
    os.makedirs("test_results", exist_ok=True)
    
    # Run tests
    unittest.main(verbosity=2)
