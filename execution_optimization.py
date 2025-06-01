#!/usr/bin/env python
"""
Execution Optimization for Trading-Agent System.

This module provides functionality to optimize the execution of trades,
including ultra-fast order routing, microsecond-level latency profiling,
and smart order types.
"""

import os
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import threading
import queue
import json
import socket
import asyncio
import concurrent.futures

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('execution_optimization')

class OrderType(Enum):
    """Enum for order types."""
    MARKET = 'market'
    LIMIT = 'limit'
    STOP = 'stop'
    STOP_LIMIT = 'stop_limit'
    ICEBERG = 'iceberg'
    TWAP = 'twap'
    VWAP = 'vwap'
    SMART = 'smart'

class OrderSide(Enum):
    """Enum for order sides."""
    BUY = 'buy'
    SELL = 'sell'

class OrderStatus(Enum):
    """Enum for order statuses."""
    PENDING = 'pending'
    OPEN = 'open'
    FILLED = 'filled'
    PARTIALLY_FILLED = 'partially_filled'
    CANCELED = 'canceled'
    REJECTED = 'rejected'
    EXPIRED = 'expired'

@dataclass
class Order:
    """Class for representing an order."""
    id: str
    symbol: str
    side: OrderSide
    type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = 'GTC'  # Good Till Canceled
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_fill_price: Optional[float] = None
    created_at: float = None
    updated_at: float = None
    exchange_id: Optional[str] = None
    client_order_id: Optional[str] = None
    metadata: Optional[Dict] = None
    
    def __post_init__(self):
        """Initialize timestamps if not provided."""
        if self.created_at is None:
            self.created_at = time.time()
        if self.updated_at is None:
            self.updated_at = self.created_at
    
    def to_dict(self) -> Dict:
        """Convert order to dictionary.
        
        Returns:
            Dictionary representation of the order
        """
        return {
            'id': self.id,
            'symbol': self.symbol,
            'side': self.side.value,
            'type': self.type.value,
            'quantity': self.quantity,
            'price': self.price,
            'stop_price': self.stop_price,
            'time_in_force': self.time_in_force,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'average_fill_price': self.average_fill_price,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'exchange_id': self.exchange_id,
            'client_order_id': self.client_order_id,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Order':
        """Create order from dictionary.
        
        Args:
            data: Dictionary representation of the order
            
        Returns:
            Order object
        """
        return cls(
            id=data['id'],
            symbol=data['symbol'],
            side=OrderSide(data['side']),
            type=OrderType(data['type']),
            quantity=data['quantity'],
            price=data.get('price'),
            stop_price=data.get('stop_price'),
            time_in_force=data.get('time_in_force', 'GTC'),
            status=OrderStatus(data['status']),
            filled_quantity=data.get('filled_quantity', 0.0),
            average_fill_price=data.get('average_fill_price'),
            created_at=data.get('created_at'),
            updated_at=data.get('updated_at'),
            exchange_id=data.get('exchange_id'),
            client_order_id=data.get('client_order_id'),
            metadata=data.get('metadata')
        )


class LatencyProfiler:
    """Class for profiling execution latency."""
    
    def __init__(self, 
                metrics_file: str = 'latency_metrics.json',
                rolling_window: int = 100):
        """Initialize the latency profiler.
        
        Args:
            metrics_file: File to save latency metrics
            rolling_window: Window size for rolling statistics
        """
        self.metrics_file = metrics_file
        self.rolling_window = rolling_window
        self.metrics = {
            'order_submission': [],
            'order_acknowledgement': [],
            'order_execution': [],
            'market_data': [],
            'signal_generation': [],
            'decision_making': [],
            'end_to_end': []
        }
        self.timestamps = {}
        
        logger.info(f"Initialized LatencyProfiler with rolling_window={rolling_window}")
    
    def start_timer(self, key: str) -> None:
        """Start a timer for a specific operation.
        
        Args:
            key: Operation identifier
        """
        self.timestamps[key] = time.time_ns()
    
    def stop_timer(self, key: str, category: str) -> float:
        """Stop a timer and record the latency.
        
        Args:
            key: Operation identifier
            category: Latency category
            
        Returns:
            Latency in microseconds
        """
        if key not in self.timestamps:
            logger.warning(f"No start time found for key: {key}")
            return 0.0
        
        end_time = time.time_ns()
        start_time = self.timestamps[key]
        latency_ns = end_time - start_time
        latency_us = latency_ns / 1000  # Convert to microseconds
        
        # Record latency
        if category in self.metrics:
            self.metrics[category].append(latency_us)
            # Keep only the most recent measurements
            if len(self.metrics[category]) > self.rolling_window:
                self.metrics[category] = self.metrics[category][-self.rolling_window:]
        else:
            logger.warning(f"Unknown latency category: {category}")
        
        # Clean up
        del self.timestamps[key]
        
        return latency_us
    
    def get_metrics(self) -> Dict:
        """Get latency metrics.
        
        Returns:
            Dictionary with latency metrics
        """
        metrics = {}
        
        for category, latencies in self.metrics.items():
            if latencies:
                metrics[category] = {
                    'min': np.min(latencies),
                    'max': np.max(latencies),
                    'mean': np.mean(latencies),
                    'median': np.median(latencies),
                    'p95': np.percentile(latencies, 95),
                    'p99': np.percentile(latencies, 99),
                    'count': len(latencies)
                }
            else:
                metrics[category] = {
                    'min': 0,
                    'max': 0,
                    'mean': 0,
                    'median': 0,
                    'p95': 0,
                    'p99': 0,
                    'count': 0
                }
        
        return metrics
    
    def save_metrics(self) -> None:
        """Save latency metrics to file."""
        metrics = self.get_metrics()
        
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Saved latency metrics to {self.metrics_file}")
    
    def log_metrics(self) -> None:
        """Log latency metrics."""
        metrics = self.get_metrics()
        
        for category, stats in metrics.items():
            logger.info(f"{category} latency (μs): "
                       f"min={stats['min']:.2f}, "
                       f"max={stats['max']:.2f}, "
                       f"mean={stats['mean']:.2f}, "
                       f"median={stats['median']:.2f}, "
                       f"p95={stats['p95']:.2f}, "
                       f"p99={stats['p99']:.2f}, "
                       f"count={stats['count']}")


class OrderRouter:
    """Class for routing orders to exchanges."""
    
    def __init__(self, 
                latency_profiler: LatencyProfiler = None,
                max_retries: int = 3,
                retry_delay: float = 0.5,
                timeout: float = 5.0):
        """Initialize the order router.
        
        Args:
            latency_profiler: Latency profiler
            max_retries: Maximum number of retries
            retry_delay: Delay between retries in seconds
            timeout: Timeout for order submission in seconds
        """
        self.latency_profiler = latency_profiler
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.exchanges = {}
        
        logger.info(f"Initialized OrderRouter with max_retries={max_retries}, retry_delay={retry_delay}s")
    
    def register_exchange(self, exchange_id: str, exchange_client: object) -> None:
        """Register an exchange client.
        
        Args:
            exchange_id: Exchange identifier
            exchange_client: Exchange client object
        """
        self.exchanges[exchange_id] = exchange_client
        logger.info(f"Registered exchange: {exchange_id}")
    
    def submit_order(self, order: Order, exchange_id: str = None) -> Order:
        """Submit an order to an exchange.
        
        Args:
            order: Order to submit
            exchange_id: Exchange identifier (if None, use the best exchange)
            
        Returns:
            Updated order
        """
        # Select exchange
        if exchange_id is None:
            exchange_id = self._select_best_exchange(order)
        
        if exchange_id not in self.exchanges:
            logger.error(f"Unknown exchange: {exchange_id}")
            order.status = OrderStatus.REJECTED
            return order
        
        # Get exchange client
        exchange = self.exchanges[exchange_id]
        
        # Start latency profiling
        if self.latency_profiler:
            self.latency_profiler.start_timer(f"submit_{order.id}")
        
        # Submit order with retries
        for attempt in range(self.max_retries):
            try:
                # Submit order to exchange
                result = exchange.submit_order(order)
                
                # Update order with exchange response
                order.exchange_id = exchange_id
                order.status = result.get('status', OrderStatus.OPEN)
                order.id = result.get('id', order.id)
                order.updated_at = time.time()
                
                # Stop latency profiling
                if self.latency_profiler:
                    latency = self.latency_profiler.stop_timer(f"submit_{order.id}", 'order_submission')
                    logger.debug(f"Order submission latency: {latency:.2f} μs")
                
                return order
            
            except Exception as e:
                logger.warning(f"Order submission failed (attempt {attempt+1}/{self.max_retries}): {e}")
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"Order submission failed after {self.max_retries} attempts")
                    order.status = OrderStatus.REJECTED
                    
                    # Stop latency profiling
                    if self.latency_profiler:
                        self.latency_profiler.stop_timer(f"submit_{order.id}", 'order_submission')
                    
                    return order
    
    def cancel_order(self, order: Order) -> Order:
        """Cancel an order.
        
        Args:
            order: Order to cancel
            
        Returns:
            Updated order
        """
        if order.exchange_id not in self.exchanges:
            logger.error(f"Unknown exchange: {order.exchange_id}")
            return order
        
        # Get exchange client
        exchange = self.exchanges[order.exchange_id]
        
        # Start latency profiling
        if self.latency_profiler:
            self.latency_profiler.start_timer(f"cancel_{order.id}")
        
        # Cancel order with retries
        for attempt in range(self.max_retries):
            try:
                # Cancel order on exchange
                result = exchange.cancel_order(order)
                
                # Update order with exchange response
                order.status = result.get('status', OrderStatus.CANCELED)
                order.updated_at = time.time()
                
                # Stop latency profiling
                if self.latency_profiler:
                    latency = self.latency_profiler.stop_timer(f"cancel_{order.id}", 'order_submission')
                    logger.debug(f"Order cancellation latency: {latency:.2f} μs")
                
                return order
            
            except Exception as e:
                logger.warning(f"Order cancellation failed (attempt {attempt+1}/{self.max_retries}): {e}")
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"Order cancellation failed after {self.max_retries} attempts")
                    
                    # Stop latency profiling
                    if self.latency_profiler:
                        self.latency_profiler.stop_timer(f"cancel_{order.id}", 'order_submission')
                    
                    return order
    
    def get_order_status(self, order: Order) -> Order:
        """Get the status of an order.
        
        Args:
            order: Order to check
            
        Returns:
            Updated order
        """
        if order.exchange_id not in self.exchanges:
            logger.error(f"Unknown exchange: {order.exchange_id}")
            return order
        
        # Get exchange client
        exchange = self.exchanges[order.exchange_id]
        
        # Start latency profiling
        if self.latency_profiler:
            self.latency_profiler.start_timer(f"status_{order.id}")
        
        # Get order status with retries
        for attempt in range(self.max_retries):
            try:
                # Get order status from exchange
                result = exchange.get_order_status(order)
                
                # Update order with exchange response
                order.status = result.get('status', order.status)
                order.filled_quantity = result.get('filled_quantity', order.filled_quantity)
                order.average_fill_price = result.get('average_fill_price', order.average_fill_price)
                order.updated_at = time.time()
                
                # Stop latency profiling
                if self.latency_profiler:
                    latency = self.latency_profiler.stop_timer(f"status_{order.id}", 'order_acknowledgement')
                    logger.debug(f"Order status check latency: {latency:.2f} μs")
                
                return order
            
            except Exception as e:
                logger.warning(f"Order status check failed (attempt {attempt+1}/{self.max_retries}): {e}")
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"Order status check failed after {self.max_retries} attempts")
                    
                    # Stop latency profiling
                    if self.latency_profiler:
                        self.latency_profiler.stop_timer(f"status_{order.id}", 'order_acknowledgement')
                    
                    return order
    
    def _select_best_exchange(self, order: Order) -> str:
        """Select the best exchange for an order.
        
        Args:
            order: Order to route
            
        Returns:
            Exchange identifier
        """
        # If only one exchange is registered, use it
        if len(self.exchanges) == 1:
            return list(self.exchanges.keys())[0]
        
        # TODO: Implement exchange selection logic based on:
        # - Latency
        # - Liquidity
        # - Fees
        # - Historical performance
        
        # For now, just return the first exchange
        return list(self.exchanges.keys())[0]


class SmartOrderRouter:
    """Class for smart order routing."""
    
    def __init__(self, 
                order_router: OrderRouter,
                latency_profiler: LatencyProfiler = None,
                max_slippage: float = 0.001,
                min_fill_ratio: float = 0.9):
        """Initialize the smart order router.
        
        Args:
            order_router: Order router
            latency_profiler: Latency profiler
            max_slippage: Maximum acceptable slippage
            min_fill_ratio: Minimum acceptable fill ratio
        """
        self.order_router = order_router
        self.latency_profiler = latency_profiler
        self.max_slippage = max_slippage
        self.min_fill_ratio = min_fill_ratio
        
        logger.info(f"Initialized SmartOrderRouter with max_slippage={max_slippage}, min_fill_ratio={min_fill_ratio}")
    
    def route_order(self, order: Order) -> List[Order]:
        """Route an order using smart routing strategies.
        
        Args:
            order: Order to route
            
        Returns:
            List of executed orders
        """
        # Start latency profiling
        if self.latency_profiler:
            self.latency_profiler.start_timer(f"smart_route_{order.id}")
        
        # Select routing strategy based on order type
        if order.type == OrderType.MARKET:
            result = self._route_market_order(order)
        elif order.type == OrderType.LIMIT:
            result = self._route_limit_order(order)
        elif order.type == OrderType.ICEBERG:
            result = self._route_iceberg_order(order)
        elif order.type == OrderType.TWAP:
            result = self._route_twap_order(order)
        elif order.type == OrderType.VWAP:
            result = self._route_vwap_order(order)
        elif order.type == OrderType.SMART:
            result = self._route_smart_order(order)
        else:
            # For other order types, use the basic order router
            result = [self.order_router.submit_order(order)]
        
        # Stop latency profiling
        if self.latency_profiler:
            latency = self.latency_profiler.stop_timer(f"smart_route_{order.id}", 'order_execution')
            logger.debug(f"Smart order routing latency: {latency:.2f} μs")
        
        return result
    
    def _route_market_order(self, order: Order) -> List[Order]:
        """Route a market order.
        
        Args:
            order: Market order to route
            
        Returns:
            List of executed orders
        """
        # For market orders, just submit directly
        executed_order = self.order_router.submit_order(order)
        return [executed_order]
    
    def _route_limit_order(self, order: Order) -> List[Order]:
        """Route a limit order.
        
        Args:
            order: Limit order to route
            
        Returns:
            List of executed orders
        """
        # For limit orders, just submit directly
        executed_order = self.order_router.submit_order(order)
        return [executed_order]
    
    def _route_iceberg_order(self, order: Order) -> List[Order]:
        """Route an iceberg order by splitting it into smaller orders.
        
        Args:
            order: Iceberg order to route
            
        Returns:
            List of executed orders
        """
        # Extract iceberg parameters from metadata
        metadata = order.metadata or {}
        display_size = metadata.get('display_size', order.quantity * 0.1)
        
        # Calculate number of child orders
        num_orders = int(np.ceil(order.quantity / display_size))
        
        # Create and submit child orders
        executed_orders = []
        remaining_quantity = order.quantity
        
        for i in range(num_orders):
            # Calculate child order quantity
            child_quantity = min(display_size, remaining_quantity)
            
            # Create child order
            child_order = Order(
                id=f"{order.id}_iceberg_{i}",
                symbol=order.symbol,
                side=order.side,
                type=OrderType.LIMIT,
                quantity=child_quantity,
                price=order.price,
                time_in_force=order.time_in_force,
                client_order_id=f"{order.client_order_id}_iceberg_{i}" if order.client_order_id else None,
                metadata={'parent_order_id': order.id}
            )
            
            # Submit child order
            executed_order = self.order_router.submit_order(child_order)
            executed_orders.append(executed_order)
            
            # Update remaining quantity
            remaining_quantity -= child_quantity
            
            # If the order was rejected, stop submitting more
            if executed_order.status == OrderStatus.REJECTED:
                break
        
        return executed_orders
    
    def _route_twap_order(self, order: Order) -> List[Order]:
        """Route a TWAP (Time-Weighted Average Price) order.
        
        Args:
            order: TWAP order to route
            
        Returns:
            List of executed orders
        """
        # Extract TWAP parameters from metadata
        metadata = order.metadata or {}
        duration = metadata.get('duration', 3600)  # Default: 1 hour
        num_slices = metadata.get('num_slices', 12)  # Default: 12 slices
        
        # Calculate slice size and interval
        slice_size = order.quantity / num_slices
        interval = duration / num_slices
        
        # Create and submit child orders
        executed_orders = []
        
        for i in range(num_slices):
            # Create child order
            child_order = Order(
                id=f"{order.id}_twap_{i}",
                symbol=order.symbol,
                side=order.side,
                type=OrderType.LIMIT if order.price else OrderType.MARKET,
                quantity=slice_size,
                price=order.price,
                time_in_force='IOC',  # Immediate or Cancel
                client_order_id=f"{order.client_order_id}_twap_{i}" if order.client_order_id else None,
                metadata={'parent_order_id': order.id}
            )
            
            # Submit child order
            executed_order = self.order_router.submit_order(child_order)
            executed_orders.append(executed_order)
            
            # Sleep until next slice
            if i < num_slices - 1:
                time.sleep(interval)
        
        return executed_orders
    
    def _route_vwap_order(self, order: Order) -> List[Order]:
        """Route a VWAP (Volume-Weighted Average Price) order.
        
        Args:
            order: VWAP order to route
            
        Returns:
            List of executed orders
        """
        # Extract VWAP parameters from metadata
        metadata = order.metadata or {}
        duration = metadata.get('duration', 3600)  # Default: 1 hour
        num_slices = metadata.get('num_slices', 12)  # Default: 12 slices
        volume_profile = metadata.get('volume_profile', None)
        
        # If no volume profile is provided, use a default one
        if volume_profile is None:
            # Default volume profile (higher volume at open and close)
            volume_profile = [0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.4]
            
            # Ensure the profile has the right length
            if len(volume_profile) != num_slices:
                # Create a uniform profile
                volume_profile = [1.0 / num_slices] * num_slices
        
        # Normalize volume profile
        total_volume = sum(volume_profile)
        volume_profile = [v / total_volume for v in volume_profile]
        
        # Calculate slice sizes and interval
        slice_sizes = [order.quantity * v for v in volume_profile]
        interval = duration / num_slices
        
        # Create and submit child orders
        executed_orders = []
        
        for i in range(num_slices):
            # Create child order
            child_order = Order(
                id=f"{order.id}_vwap_{i}",
                symbol=order.symbol,
                side=order.side,
                type=OrderType.LIMIT if order.price else OrderType.MARKET,
                quantity=slice_sizes[i],
                price=order.price,
                time_in_force='IOC',  # Immediate or Cancel
                client_order_id=f"{order.client_order_id}_vwap_{i}" if order.client_order_id else None,
                metadata={'parent_order_id': order.id}
            )
            
            # Submit child order
            executed_order = self.order_router.submit_order(child_order)
            executed_orders.append(executed_order)
            
            # Sleep until next slice
            if i < num_slices - 1:
                time.sleep(interval)
        
        return executed_orders
    
    def _route_smart_order(self, order: Order) -> List[Order]:
        """Route a smart order using adaptive strategies.
        
        Args:
            order: Smart order to route
            
        Returns:
            List of executed orders
        """
        # Extract smart order parameters from metadata
        metadata = order.metadata or {}
        strategy = metadata.get('strategy', 'adaptive')
        
        if strategy == 'adaptive':
            return self._route_adaptive_order(order)
        elif strategy == 'liquidity_seeking':
            return self._route_liquidity_seeking_order(order)
        elif strategy == 'minimal_impact':
            return self._route_minimal_impact_order(order)
        else:
            logger.warning(f"Unknown smart order strategy: {strategy}")
            # Fall back to market order
            order.type = OrderType.MARKET
            return self._route_market_order(order)
    
    def _route_adaptive_order(self, order: Order) -> List[Order]:
        """Route an order using adaptive strategies based on market conditions.
        
        Args:
            order: Order to route
            
        Returns:
            List of executed orders
        """
        # TODO: Implement adaptive routing based on market conditions
        # For now, use a simple TWAP strategy
        order.type = OrderType.TWAP
        order.metadata = order.metadata or {}
        order.metadata['num_slices'] = 5
        order.metadata['duration'] = 300  # 5 minutes
        
        return self._route_twap_order(order)
    
    def _route_liquidity_seeking_order(self, order: Order) -> List[Order]:
        """Route an order seeking liquidity across multiple venues.
        
        Args:
            order: Order to route
            
        Returns:
            List of executed orders
        """
        # TODO: Implement liquidity seeking across multiple exchanges
        # For now, use a simple iceberg strategy
        order.type = OrderType.ICEBERG
        order.metadata = order.metadata or {}
        order.metadata['display_size'] = order.quantity * 0.2
        
        return self._route_iceberg_order(order)
    
    def _route_minimal_impact_order(self, order: Order) -> List[Order]:
        """Route an order with minimal market impact.
        
        Args:
            order: Order to route
            
        Returns:
            List of executed orders
        """
        # TODO: Implement minimal impact routing
        # For now, use a simple VWAP strategy
        order.type = OrderType.VWAP
        order.metadata = order.metadata or {}
        order.metadata['num_slices'] = 10
        order.metadata['duration'] = 1800  # 30 minutes
        
        return self._route_vwap_order(order)


class ExecutionOptimizer:
    """Class for optimizing trade execution."""
    
    def __init__(self, 
                smart_order_router: SmartOrderRouter,
                latency_profiler: LatencyProfiler = None):
        """Initialize the execution optimizer.
        
        Args:
            smart_order_router: Smart order router
            latency_profiler: Latency profiler
        """
        self.smart_order_router = smart_order_router
        self.latency_profiler = latency_profiler
        self.order_queue = queue.Queue()
        self.order_history = {}
        self.running = False
        self.worker_thread = None
        
        logger.info("Initialized ExecutionOptimizer")
    
    def start(self) -> None:
        """Start the execution optimizer."""
        if self.running:
            logger.warning("Execution optimizer is already running")
            return
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._process_orders)
        self.worker_thread.daemon = True
        self.worker_thread.start()
        
        logger.info("Started execution optimizer")
    
    def stop(self) -> None:
        """Stop the execution optimizer."""
        if not self.running:
            logger.warning("Execution optimizer is not running")
            return
        
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)
        
        logger.info("Stopped execution optimizer")
    
    def submit_order(self, order: Order) -> str:
        """Submit an order for execution.
        
        Args:
            order: Order to execute
            
        Returns:
            Order ID
        """
        # Start latency profiling
        if self.latency_profiler:
            self.latency_profiler.start_timer(f"optimize_{order.id}")
        
        # Add order to queue
        self.order_queue.put(order)
        
        # Add order to history
        self.order_history[order.id] = {
            'order': order,
            'executed_orders': [],
            'status': 'queued',
            'submitted_at': time.time()
        }
        
        logger.info(f"Queued order {order.id} for execution")
        return order.id
    
    def get_order_status(self, order_id: str) -> Dict:
        """Get the status of an order.
        
        Args:
            order_id: Order ID
            
        Returns:
            Order status
        """
        if order_id not in self.order_history:
            return {'status': 'unknown'}
        
        return self.order_history[order_id]
    
    def _process_orders(self) -> None:
        """Process orders from the queue."""
        while self.running:
            try:
                # Get order from queue with timeout
                try:
                    order = self.order_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Update order status
                self.order_history[order.id]['status'] = 'processing'
                
                # Start latency profiling
                if self.latency_profiler:
                    self.latency_profiler.start_timer(f"execute_{order.id}")
                
                # Execute order
                try:
                    executed_orders = self.smart_order_router.route_order(order)
                    
                    # Update order history
                    self.order_history[order.id]['executed_orders'] = executed_orders
                    self.order_history[order.id]['status'] = 'executed'
                    self.order_history[order.id]['executed_at'] = time.time()
                    
                    # Stop latency profiling
                    if self.latency_profiler:
                        latency = self.latency_profiler.stop_timer(f"execute_{order.id}", 'order_execution')
                        logger.debug(f"Order execution latency: {latency:.2f} μs")
                    
                    # Stop optimization latency profiling
                    if self.latency_profiler:
                        latency = self.latency_profiler.stop_timer(f"optimize_{order.id}", 'end_to_end')
                        logger.debug(f"End-to-end execution latency: {latency:.2f} μs")
                    
                    logger.info(f"Executed order {order.id}")
                
                except Exception as e:
                    # Update order history
                    self.order_history[order.id]['status'] = 'failed'
                    self.order_history[order.id]['error'] = str(e)
                    
                    # Stop latency profiling
                    if self.latency_profiler:
                        self.latency_profiler.stop_timer(f"execute_{order.id}", 'order_execution')
                    
                    # Stop optimization latency profiling
                    if self.latency_profiler:
                        self.latency_profiler.stop_timer(f"optimize_{order.id}", 'end_to_end')
                    
                    logger.error(f"Failed to execute order {order.id}: {e}")
                
                # Mark task as done
                self.order_queue.task_done()
            
            except Exception as e:
                logger.error(f"Error in order processing: {e}")


class AsyncOrderRouter:
    """Class for asynchronous order routing."""
    
    def __init__(self, 
                latency_profiler: LatencyProfiler = None,
                max_retries: int = 3,
                retry_delay: float = 0.5,
                timeout: float = 5.0):
        """Initialize the async order router.
        
        Args:
            latency_profiler: Latency profiler
            max_retries: Maximum number of retries
            retry_delay: Delay between retries in seconds
            timeout: Timeout for order submission in seconds
        """
        self.latency_profiler = latency_profiler
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.exchanges = {}
        
        logger.info(f"Initialized AsyncOrderRouter with max_retries={max_retries}, retry_delay={retry_delay}s")
    
    def register_exchange(self, exchange_id: str, exchange_client: object) -> None:
        """Register an exchange client.
        
        Args:
            exchange_id: Exchange identifier
            exchange_client: Exchange client object
        """
        self.exchanges[exchange_id] = exchange_client
        logger.info(f"Registered exchange: {exchange_id}")
    
    async def submit_order(self, order: Order, exchange_id: str = None) -> Order:
        """Submit an order to an exchange asynchronously.
        
        Args:
            order: Order to submit
            exchange_id: Exchange identifier (if None, use the best exchange)
            
        Returns:
            Updated order
        """
        # Select exchange
        if exchange_id is None:
            exchange_id = await self._select_best_exchange(order)
        
        if exchange_id not in self.exchanges:
            logger.error(f"Unknown exchange: {exchange_id}")
            order.status = OrderStatus.REJECTED
            return order
        
        # Get exchange client
        exchange = self.exchanges[exchange_id]
        
        # Start latency profiling
        if self.latency_profiler:
            self.latency_profiler.start_timer(f"async_submit_{order.id}")
        
        # Submit order with retries
        for attempt in range(self.max_retries):
            try:
                # Submit order to exchange
                result = await exchange.submit_order_async(order)
                
                # Update order with exchange response
                order.exchange_id = exchange_id
                order.status = result.get('status', OrderStatus.OPEN)
                order.id = result.get('id', order.id)
                order.updated_at = time.time()
                
                # Stop latency profiling
                if self.latency_profiler:
                    latency = self.latency_profiler.stop_timer(f"async_submit_{order.id}", 'order_submission')
                    logger.debug(f"Async order submission latency: {latency:.2f} μs")
                
                return order
            
            except Exception as e:
                logger.warning(f"Async order submission failed (attempt {attempt+1}/{self.max_retries}): {e}")
                
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                else:
                    logger.error(f"Async order submission failed after {self.max_retries} attempts")
                    order.status = OrderStatus.REJECTED
                    
                    # Stop latency profiling
                    if self.latency_profiler:
                        self.latency_profiler.stop_timer(f"async_submit_{order.id}", 'order_submission')
                    
                    return order
    
    async def cancel_order(self, order: Order) -> Order:
        """Cancel an order asynchronously.
        
        Args:
            order: Order to cancel
            
        Returns:
            Updated order
        """
        if order.exchange_id not in self.exchanges:
            logger.error(f"Unknown exchange: {order.exchange_id}")
            return order
        
        # Get exchange client
        exchange = self.exchanges[order.exchange_id]
        
        # Start latency profiling
        if self.latency_profiler:
            self.latency_profiler.start_timer(f"async_cancel_{order.id}")
        
        # Cancel order with retries
        for attempt in range(self.max_retries):
            try:
                # Cancel order on exchange
                result = await exchange.cancel_order_async(order)
                
                # Update order with exchange response
                order.status = result.get('status', OrderStatus.CANCELED)
                order.updated_at = time.time()
                
                # Stop latency profiling
                if self.latency_profiler:
                    latency = self.latency_profiler.stop_timer(f"async_cancel_{order.id}", 'order_submission')
                    logger.debug(f"Async order cancellation latency: {latency:.2f} μs")
                
                return order
            
            except Exception as e:
                logger.warning(f"Async order cancellation failed (attempt {attempt+1}/{self.max_retries}): {e}")
                
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                else:
                    logger.error(f"Async order cancellation failed after {self.max_retries} attempts")
                    
                    # Stop latency profiling
                    if self.latency_profiler:
                        self.latency_profiler.stop_timer(f"async_cancel_{order.id}", 'order_submission')
                    
                    return order
    
    async def get_order_status(self, order: Order) -> Order:
        """Get the status of an order asynchronously.
        
        Args:
            order: Order to check
            
        Returns:
            Updated order
        """
        if order.exchange_id not in self.exchanges:
            logger.error(f"Unknown exchange: {order.exchange_id}")
            return order
        
        # Get exchange client
        exchange = self.exchanges[order.exchange_id]
        
        # Start latency profiling
        if self.latency_profiler:
            self.latency_profiler.start_timer(f"async_status_{order.id}")
        
        # Get order status with retries
        for attempt in range(self.max_retries):
            try:
                # Get order status from exchange
                result = await exchange.get_order_status_async(order)
                
                # Update order with exchange response
                order.status = result.get('status', order.status)
                order.filled_quantity = result.get('filled_quantity', order.filled_quantity)
                order.average_fill_price = result.get('average_fill_price', order.average_fill_price)
                order.updated_at = time.time()
                
                # Stop latency profiling
                if self.latency_profiler:
                    latency = self.latency_profiler.stop_timer(f"async_status_{order.id}", 'order_acknowledgement')
                    logger.debug(f"Async order status check latency: {latency:.2f} μs")
                
                return order
            
            except Exception as e:
                logger.warning(f"Async order status check failed (attempt {attempt+1}/{self.max_retries}): {e}")
                
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                else:
                    logger.error(f"Async order status check failed after {self.max_retries} attempts")
                    
                    # Stop latency profiling
                    if self.latency_profiler:
                        self.latency_profiler.stop_timer(f"async_status_{order.id}", 'order_acknowledgement')
                    
                    return order
    
    async def _select_best_exchange(self, order: Order) -> str:
        """Select the best exchange for an order asynchronously.
        
        Args:
            order: Order to route
            
        Returns:
            Exchange identifier
        """
        # If only one exchange is registered, use it
        if len(self.exchanges) == 1:
            return list(self.exchanges.keys())[0]
        
        # TODO: Implement exchange selection logic based on:
        # - Latency
        # - Liquidity
        # - Fees
        # - Historical performance
        
        # For now, just return the first exchange
        return list(self.exchanges.keys())[0]


class AsyncExecutionOptimizer:
    """Class for asynchronous execution optimization."""
    
    def __init__(self, 
                async_order_router: AsyncOrderRouter,
                latency_profiler: LatencyProfiler = None,
                max_concurrent_orders: int = 100):
        """Initialize the async execution optimizer.
        
        Args:
            async_order_router: Async order router
            latency_profiler: Latency profiler
            max_concurrent_orders: Maximum number of concurrent orders
        """
        self.async_order_router = async_order_router
        self.latency_profiler = latency_profiler
        self.max_concurrent_orders = max_concurrent_orders
        self.order_queue = asyncio.Queue()
        self.order_history = {}
        self.running = False
        self.worker_tasks = []
        
        logger.info(f"Initialized AsyncExecutionOptimizer with max_concurrent_orders={max_concurrent_orders}")
    
    async def start(self) -> None:
        """Start the async execution optimizer."""
        if self.running:
            logger.warning("Async execution optimizer is already running")
            return
        
        self.running = True
        
        # Create worker tasks
        for i in range(self.max_concurrent_orders):
            task = asyncio.create_task(self._process_orders())
            self.worker_tasks.append(task)
        
        logger.info(f"Started async execution optimizer with {self.max_concurrent_orders} workers")
    
    async def stop(self) -> None:
        """Stop the async execution optimizer."""
        if not self.running:
            logger.warning("Async execution optimizer is not running")
            return
        
        self.running = False
        
        # Cancel worker tasks
        for task in self.worker_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
        logger.info("Stopped async execution optimizer")
    
    async def submit_order(self, order: Order) -> str:
        """Submit an order for execution asynchronously.
        
        Args:
            order: Order to execute
            
        Returns:
            Order ID
        """
        # Start latency profiling
        if self.latency_profiler:
            self.latency_profiler.start_timer(f"async_optimize_{order.id}")
        
        # Add order to queue
        await self.order_queue.put(order)
        
        # Add order to history
        self.order_history[order.id] = {
            'order': order,
            'executed_orders': [],
            'status': 'queued',
            'submitted_at': time.time()
        }
        
        logger.info(f"Queued order {order.id} for async execution")
        return order.id
    
    def get_order_status(self, order_id: str) -> Dict:
        """Get the status of an order.
        
        Args:
            order_id: Order ID
            
        Returns:
            Order status
        """
        if order_id not in self.order_history:
            return {'status': 'unknown'}
        
        return self.order_history[order_id]
    
    async def _process_orders(self) -> None:
        """Process orders from the queue asynchronously."""
        while self.running:
            try:
                # Get order from queue with timeout
                try:
                    order = await asyncio.wait_for(self.order_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                # Update order status
                self.order_history[order.id]['status'] = 'processing'
                
                # Start latency profiling
                if self.latency_profiler:
                    self.latency_profiler.start_timer(f"async_execute_{order.id}")
                
                # Execute order
                try:
                    executed_order = await self.async_order_router.submit_order(order)
                    
                    # Update order history
                    self.order_history[order.id]['executed_orders'] = [executed_order]
                    self.order_history[order.id]['status'] = 'executed'
                    self.order_history[order.id]['executed_at'] = time.time()
                    
                    # Stop latency profiling
                    if self.latency_profiler:
                        latency = self.latency_profiler.stop_timer(f"async_execute_{order.id}", 'order_execution')
                        logger.debug(f"Async order execution latency: {latency:.2f} μs")
                    
                    # Stop optimization latency profiling
                    if self.latency_profiler:
                        latency = self.latency_profiler.stop_timer(f"async_optimize_{order.id}", 'end_to_end')
                        logger.debug(f"Async end-to-end execution latency: {latency:.2f} μs")
                    
                    logger.info(f"Executed order {order.id} asynchronously")
                
                except Exception as e:
                    # Update order history
                    self.order_history[order.id]['status'] = 'failed'
                    self.order_history[order.id]['error'] = str(e)
                    
                    # Stop latency profiling
                    if self.latency_profiler:
                        self.latency_profiler.stop_timer(f"async_execute_{order.id}", 'order_execution')
                    
                    # Stop optimization latency profiling
                    if self.latency_profiler:
                        self.latency_profiler.stop_timer(f"async_optimize_{order.id}", 'end_to_end')
                    
                    logger.error(f"Failed to execute order {order.id} asynchronously: {e}")
                
                # Mark task as done
                self.order_queue.task_done()
            
            except asyncio.CancelledError:
                # Task was cancelled
                break
            
            except Exception as e:
                logger.error(f"Error in async order processing: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Execution Optimization')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory for output files')
    parser.add_argument('--mode', type=str, default='sync', choices=['sync', 'async'], help='Execution mode')
    parser.add_argument('--num_orders', type=int, default=10, help='Number of orders to simulate')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create latency profiler
    latency_profiler = LatencyProfiler(
        metrics_file=os.path.join(args.output_dir, 'latency_metrics.json')
    )
    
    # Mock exchange client for testing
    class MockExchangeClient:
        def submit_order(self, order):
            # Simulate network latency
            time.sleep(0.01)
            return {'status': OrderStatus.OPEN, 'id': order.id}
        
        def cancel_order(self, order):
            # Simulate network latency
            time.sleep(0.01)
            return {'status': OrderStatus.CANCELED}
        
        def get_order_status(self, order):
            # Simulate network latency
            time.sleep(0.01)
            return {'status': OrderStatus.FILLED, 'filled_quantity': order.quantity, 'average_fill_price': 100.0}
        
        async def submit_order_async(self, order):
            # Simulate network latency
            await asyncio.sleep(0.01)
            return {'status': OrderStatus.OPEN, 'id': order.id}
        
        async def cancel_order_async(self, order):
            # Simulate network latency
            await asyncio.sleep(0.01)
            return {'status': OrderStatus.CANCELED}
        
        async def get_order_status_async(self, order):
            # Simulate network latency
            await asyncio.sleep(0.01)
            return {'status': OrderStatus.FILLED, 'filled_quantity': order.quantity, 'average_fill_price': 100.0}
    
    if args.mode == 'sync':
        # Create order router
        order_router = OrderRouter(latency_profiler=latency_profiler)
        order_router.register_exchange('mock', MockExchangeClient())
        
        # Create smart order router
        smart_order_router = SmartOrderRouter(order_router=order_router, latency_profiler=latency_profiler)
        
        # Create execution optimizer
        execution_optimizer = ExecutionOptimizer(smart_order_router=smart_order_router, latency_profiler=latency_profiler)
        execution_optimizer.start()
        
        # Create and submit orders
        for i in range(args.num_orders):
            order = Order(
                id=f"order_{i}",
                symbol="BTC/USD",
                side=OrderSide.BUY,
                type=OrderType.MARKET,
                quantity=1.0
            )
            
            execution_optimizer.submit_order(order)
        
        # Wait for orders to be processed
        time.sleep(5.0)
        
        # Stop execution optimizer
        execution_optimizer.stop()
        
        # Log latency metrics
        latency_profiler.log_metrics()
        latency_profiler.save_metrics()
        
        logger.info("Execution optimization completed")
    
    else:  # async mode
        async def run_async_test():
            # Create async order router
            async_order_router = AsyncOrderRouter(latency_profiler=latency_profiler)
            async_order_router.register_exchange('mock', MockExchangeClient())
            
            # Create async execution optimizer
            async_execution_optimizer = AsyncExecutionOptimizer(
                async_order_router=async_order_router,
                latency_profiler=latency_profiler
            )
            
            # Start async execution optimizer
            await async_execution_optimizer.start()
            
            # Create and submit orders
            for i in range(args.num_orders):
                order = Order(
                    id=f"order_{i}",
                    symbol="BTC/USD",
                    side=OrderSide.BUY,
                    type=OrderType.MARKET,
                    quantity=1.0
                )
                
                await async_execution_optimizer.submit_order(order)
            
            # Wait for orders to be processed
            await asyncio.sleep(5.0)
            
            # Stop async execution optimizer
            await async_execution_optimizer.stop()
            
            # Log latency metrics
            latency_profiler.log_metrics()
            latency_profiler.save_metrics()
            
            logger.info("Async execution optimization completed")
        
        # Run async test
        asyncio.run(run_async_test())
