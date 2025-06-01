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
        
        # Record latency - dynamically add new categories if needed
        if category not in self.metrics:
            self.metrics[category] = []
            logger.info(f"Created new latency category: {category}")
        
        self.metrics[category].append(latency_us)
        # Keep only the most recent measurements
        if len(self.metrics[category]) > self.rolling_window:
            self.metrics[category] = self.metrics[category][-self.rolling_window:]
        
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
                # Check for high latency conditions
                if hasattr(exchange, 'latency') and exchange.latency > 1.0:  # 1 second threshold
                    logger.warning(f"High latency detected ({exchange.latency}s), rejecting order")
                    order.status = OrderStatus.REJECTED
                    
                    # Stop latency profiling
                    if self.latency_profiler:
                        self.latency_profiler.stop_timer(f"submit_{order.id}", 'order_submission')
                    
                    return order
                
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
        # For now, just return the first exchange
        if self.exchanges:
            return list(self.exchanges.keys())[0]
        
        return None


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
        num_slices = metadata.get('num_slices', 10)
        
        # Calculate time interval and slice size
        interval = duration / num_slices
        slice_size = order.quantity / num_slices
        
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
                time_in_force=order.time_in_force,
                client_order_id=f"{order.client_order_id}_twap_{i}" if order.client_order_id else None,
                metadata={'parent_order_id': order.id, 'slice_index': i}
            )
            
            # Submit child order
            executed_order = self.order_router.submit_order(child_order)
            executed_orders.append(executed_order)
            
            # Sleep until next interval (in a real system, this would be scheduled)
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
        num_slices = metadata.get('num_slices', 10)
        volume_profile = metadata.get('volume_profile', [0.1] * num_slices)
        
        # Normalize volume profile
        total_volume = sum(volume_profile)
        volume_profile = [v / total_volume for v in volume_profile]
        
        # Calculate time interval and slice sizes
        interval = duration / num_slices
        slice_sizes = [order.quantity * v for v in volume_profile]
        
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
                time_in_force=order.time_in_force,
                client_order_id=f"{order.client_order_id}_vwap_{i}" if order.client_order_id else None,
                metadata={'parent_order_id': order.id, 'slice_index': i}
            )
            
            # Submit child order
            executed_order = self.order_router.submit_order(child_order)
            executed_orders.append(executed_order)
            
            # Sleep until next interval (in a real system, this would be scheduled)
            if i < num_slices - 1:
                time.sleep(interval)
        
        return executed_orders
    
    def _route_smart_order(self, order: Order) -> List[Order]:
        """Route a smart order using adaptive execution.
        
        Args:
            order: Smart order to route
            
        Returns:
            List of executed orders
        """
        # Extract smart order parameters from metadata
        metadata = order.metadata or {}
        urgency = metadata.get('urgency', 0.5)  # 0.0 to 1.0
        
        # Determine execution strategy based on urgency
        if urgency > 0.8:
            # High urgency: Market order
            child_order = Order(
                id=f"{order.id}_smart_market",
                symbol=order.symbol,
                side=order.side,
                type=OrderType.MARKET,
                quantity=order.quantity,
                client_order_id=f"{order.client_order_id}_smart" if order.client_order_id else None,
                metadata={'parent_order_id': order.id}
            )
            
            executed_order = self.order_router.submit_order(child_order)
            return [executed_order]
        
        elif urgency > 0.5:
            # Medium urgency: Iceberg order
            return self._route_iceberg_order(Order(
                id=f"{order.id}_smart_iceberg",
                symbol=order.symbol,
                side=order.side,
                type=OrderType.ICEBERG,
                quantity=order.quantity,
                price=order.price,
                client_order_id=f"{order.client_order_id}_smart" if order.client_order_id else None,
                metadata={
                    'parent_order_id': order.id,
                    'display_size': order.quantity * 0.2
                }
            ))
        
        else:
            # Low urgency: TWAP order
            return self._route_twap_order(Order(
                id=f"{order.id}_smart_twap",
                symbol=order.symbol,
                side=order.side,
                type=OrderType.TWAP,
                quantity=order.quantity,
                price=order.price,
                client_order_id=f"{order.client_order_id}_smart" if order.client_order_id else None,
                metadata={
                    'parent_order_id': order.id,
                    'duration': 1800,  # 30 minutes
                    'num_slices': 5
                }
            ))


class ExecutionOptimizer:
    """Class for optimizing order execution."""
    
    def __init__(self, 
                order_router: SmartOrderRouter = None,
                smart_order_router: SmartOrderRouter = None,  # For backward compatibility
                latency_profiler: LatencyProfiler = None):
        """Initialize the execution optimizer.
        
        Args:
            order_router: Smart order router
            smart_order_router: Smart order router (alias for order_router, for backward compatibility)
            latency_profiler: Latency profiler
        """
        # Use smart_order_router if order_router is not provided (for backward compatibility)
        self.order_router = order_router if order_router is not None else smart_order_router
        self.latency_profiler = latency_profiler
        self.order_history = {}
        self.running = False
        
        logger.info("Initialized ExecutionOptimizer")
    
    def submit_order(self, order: Order) -> str:
        """Submit an order for execution.
        
        Args:
            order: Order to execute
            
        Returns:
            Order identifier
        """
        # Generate order ID if not provided
        if not order.id:
            order.id = f"order_{int(time.time() * 1000)}"
        
        # Store order in history
        self.order_history[order.id] = {
            'order': order,
            'status': 'queued',
            'child_orders': [],
            'created_at': time.time(),
            'updated_at': time.time()
        }
        
        # Start execution in a separate thread
        threading.Thread(target=self._execute_order, args=(order,)).start()
        
        return order.id
    
    def get_order_status(self, order_id: str) -> Dict:
        """Get the status of an order.
        
        Args:
            order_id: Order identifier
            
        Returns:
            Order status
        """
        if order_id not in self.order_history:
            return {'status': 'unknown'}
        
        return {
            'status': self.order_history[order_id]['status'],
            'updated_at': self.order_history[order_id]['updated_at']
        }
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order.
        
        Args:
            order_id: Order identifier
            
        Returns:
            True if order was canceled, False otherwise
        """
        if order_id not in self.order_history:
            return False
        
        # Update order status
        self.order_history[order_id]['status'] = 'canceling'
        self.order_history[order_id]['updated_at'] = time.time()
        
        # Cancel child orders
        for child_order in self.order_history[order_id]['child_orders']:
            if child_order.status not in [OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED]:
                self.order_router.cancel_order(child_order)
        
        # Update order status
        self.order_history[order_id]['status'] = 'canceled'
        self.order_history[order_id]['updated_at'] = time.time()
        
        return True
    
    def stop(self):
        """Stop the execution optimizer."""
        self.running = False
    
    def _execute_order(self, order: Order) -> None:
        """Execute an order.
        
        Args:
            order: Order to execute
        """
        # Start latency profiling
        if self.latency_profiler:
            self.latency_profiler.start_timer(f"execute_{order.id}")
        
        # Update order status
        self.order_history[order.id]['status'] = 'processing'
        self.order_history[order.id]['updated_at'] = time.time()
        
        try:
            # Route order
            child_orders = self.order_router.route_order(order)
            
            # Store child orders
            self.order_history[order.id]['child_orders'] = child_orders
            
            # Check if all child orders were filled
            all_filled = all(child.status == OrderStatus.FILLED for child in child_orders)
            any_rejected = any(child.status == OrderStatus.REJECTED for child in child_orders)
            
            # Update order status
            if all_filled:
                self.order_history[order.id]['status'] = 'executed'
            elif any_rejected:
                self.order_history[order.id]['status'] = 'partially_rejected'
            else:
                self.order_history[order.id]['status'] = 'partially_executed'
            
        except Exception as e:
            logger.error(f"Order execution failed: {e}")
            self.order_history[order.id]['status'] = 'failed'
            self.order_history[order.id]['error'] = str(e)
        
        # Update timestamp
        self.order_history[order.id]['updated_at'] = time.time()
        
        # Stop latency profiling
        if self.latency_profiler:
            latency = self.latency_profiler.stop_timer(f"execute_{order.id}", 'end_to_end')
            logger.debug(f"Order execution latency: {latency:.2f} μs")


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
    
    # Synchronous wrapper for backward compatibility
    def submit_order(self, order: Order, exchange_id: str = None) -> asyncio.Future:
        """Submit an order to an exchange (synchronous wrapper for backward compatibility).
        
        Args:
            order: Order to submit
            exchange_id: Exchange identifier (if None, use the best exchange)
            
        Returns:
            Future that will resolve to the updated order
        """
        # Check if we're in an event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an event loop, create a task
                return asyncio.create_task(self.submit_order_async(order, exchange_id))
            else:
                # No running event loop, use run_until_complete
                return loop.run_until_complete(self.submit_order_async(order, exchange_id))
        except RuntimeError:
            # No event loop, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self.submit_order_async(order, exchange_id))
    
    async def submit_order_async(self, order: Order, exchange_id: str = None) -> Order:
        """Submit an order to an exchange asynchronously.
        
        Args:
            order: Order to submit
            exchange_id: Exchange identifier (if None, use the best exchange)
            
        Returns:
            Updated order
        """
        # Select exchange
        if exchange_id is None:
            exchange_id = await self._select_best_exchange_async(order)
        
        if exchange_id not in self.exchanges:
            logger.error(f"Unknown exchange: {exchange_id}")
            order.status = OrderStatus.REJECTED
            return order
        
        # Get exchange client
        exchange = self.exchanges[exchange_id]
        
        # Start latency profiling
        if self.latency_profiler:
            self.latency_profiler.start_timer(f"submit_async_{order.id}")
        
        # Submit order with retries
        for attempt in range(self.max_retries):
            try:
                # Check for high latency conditions
                if hasattr(exchange, 'latency') and exchange.latency > 1.0:  # 1 second threshold
                    logger.warning(f"High latency detected ({exchange.latency}s), rejecting order")
                    order.status = OrderStatus.REJECTED
                    
                    # Stop latency profiling
                    if self.latency_profiler:
                        self.latency_profiler.stop_timer(f"submit_async_{order.id}", 'order_submission')
                    
                    return order
                
                # Submit order to exchange
                result = await exchange.submit_order_async(order)
                
                # Update order with exchange response
                order.exchange_id = exchange_id
                order.status = result.get('status', OrderStatus.OPEN)
                order.id = result.get('id', order.id)
                order.updated_at = time.time()
                
                # Stop latency profiling
                if self.latency_profiler:
                    latency = self.latency_profiler.stop_timer(f"submit_async_{order.id}", 'order_submission')
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
                        self.latency_profiler.stop_timer(f"submit_async_{order.id}", 'order_submission')
                    
                    return order
    
    # Synchronous wrapper for backward compatibility
    def cancel_order(self, order: Order) -> asyncio.Future:
        """Cancel an order (synchronous wrapper for backward compatibility).
        
        Args:
            order: Order to cancel
            
        Returns:
            Future that will resolve to the updated order
        """
        # Check if we're in an event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an event loop, create a task
                return asyncio.create_task(self.cancel_order_async(order))
            else:
                # No running event loop, use run_until_complete
                return loop.run_until_complete(self.cancel_order_async(order))
        except RuntimeError:
            # No event loop, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self.cancel_order_async(order))
    
    async def cancel_order_async(self, order: Order) -> Order:
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
            self.latency_profiler.start_timer(f"cancel_async_{order.id}")
        
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
                    latency = self.latency_profiler.stop_timer(f"cancel_async_{order.id}", 'order_submission')
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
                        self.latency_profiler.stop_timer(f"cancel_async_{order.id}", 'order_submission')
                    
                    return order
    
    # Synchronous wrapper for backward compatibility
    def get_order_status(self, order: Order) -> asyncio.Future:
        """Get the status of an order (synchronous wrapper for backward compatibility).
        
        Args:
            order: Order to check
            
        Returns:
            Future that will resolve to the updated order
        """
        # Check if we're in an event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an event loop, create a task
                return asyncio.create_task(self.get_order_status_async(order))
            else:
                # No running event loop, use run_until_complete
                return loop.run_until_complete(self.get_order_status_async(order))
        except RuntimeError:
            # No event loop, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self.get_order_status_async(order))
    
    async def get_order_status_async(self, order: Order) -> Order:
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
            self.latency_profiler.start_timer(f"status_async_{order.id}")
        
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
                    latency = self.latency_profiler.stop_timer(f"status_async_{order.id}", 'order_acknowledgement')
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
                        self.latency_profiler.stop_timer(f"status_async_{order.id}", 'order_acknowledgement')
                    
                    return order
    
    async def _select_best_exchange_async(self, order: Order) -> str:
        """Select the best exchange for an order asynchronously.
        
        Args:
            order: Order to route
            
        Returns:
            Exchange identifier
        """
        # For now, just return the first exchange
        if self.exchanges:
            return list(self.exchanges.keys())[0]
        
        return None


class AsyncExecutionOptimizer:
    """Class for asynchronous order execution optimization."""
    
    def __init__(self, 
                order_router: AsyncOrderRouter = None,
                async_order_router: AsyncOrderRouter = None,  # For backward compatibility
                latency_profiler: LatencyProfiler = None,
                max_concurrent_orders: int = 100):
        """Initialize the async execution optimizer.
        
        Args:
            order_router: Async order router
            async_order_router: Async order router (alias for order_router, for backward compatibility)
            latency_profiler: Latency profiler
            max_concurrent_orders: Maximum number of concurrent orders
        """
        # Use async_order_router if order_router is not provided (for backward compatibility)
        self.order_router = order_router if order_router is not None else async_order_router
        self.latency_profiler = latency_profiler
        self.max_concurrent_orders = max_concurrent_orders
        self.order_history = {}
        self.order_queue = None  # Will be initialized in start()
        self.workers = []
        
        logger.info(f"Initialized AsyncExecutionOptimizer with max_concurrent_orders={max_concurrent_orders}")
    
    # Synchronous wrapper for backward compatibility
    def submit_order(self, order: Order) -> asyncio.Future:
        """Submit an order for execution (synchronous wrapper for backward compatibility).
        
        Args:
            order: Order to execute
            
        Returns:
            Future that will resolve to the order identifier
        """
        # Check if we're in an event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an event loop, create a task
                return asyncio.create_task(self.submit_order_async(order))
            else:
                # No running event loop, use run_until_complete
                return loop.run_until_complete(self.submit_order_async(order))
        except RuntimeError:
            # No event loop, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self.submit_order_async(order))
    
    async def start(self, num_workers: int = 10) -> None:
        """Start the execution optimizer.
        
        Args:
            num_workers: Number of worker tasks
        """
        # Initialize queue if not already done
        if self.order_queue is None:
            self.order_queue = asyncio.Queue()
        
        # Create worker tasks
        self.workers = [
            asyncio.create_task(self._worker())
            for _ in range(num_workers)
        ]
        
        logger.info(f"Started AsyncExecutionOptimizer with {num_workers} workers")
    
    async def stop(self) -> None:
        """Stop the execution optimizer."""
        # Cancel worker tasks
        for worker in self.workers:
            worker.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self.workers, return_exceptions=True)
        
        logger.info("Stopped AsyncExecutionOptimizer")
    
    async def submit_order_async(self, order: Order) -> str:
        """Submit an order for execution.
        
        Args:
            order: Order to execute
            
        Returns:
            Order identifier
        """
        # Generate order ID if not provided
        if not order.id:
            order.id = f"order_{int(time.time() * 1000)}"
        
        # Store order in history
        self.order_history[order.id] = {
            'order': order,
            'status': 'queued',
            'child_orders': [],
            'created_at': time.time(),
            'updated_at': time.time()
        }
        
        # Initialize queue if not already done
        if self.order_queue is None:
            self.order_queue = asyncio.Queue()
        
        # Add order to queue
        await self.order_queue.put(order)
        
        return order.id
    
    # Synchronous wrapper for backward compatibility
    def get_order_status(self, order_id: str) -> asyncio.Future:
        """Get the status of an order (synchronous wrapper for backward compatibility).
        
        Args:
            order_id: Order identifier
            
        Returns:
            Future that will resolve to the order status
        """
        # Check if we're in an event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an event loop, create a task
                return asyncio.create_task(self.get_order_status_async(order_id))
            else:
                # No running event loop, use run_until_complete
                return loop.run_until_complete(self.get_order_status_async(order_id))
        except RuntimeError:
            # No event loop, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self.get_order_status_async(order_id))
    
    async def get_order_status_async(self, order_id: str) -> Dict:
        """Get the status of an order.
        
        Args:
            order_id: Order identifier
            
        Returns:
            Order status
        """
        if order_id not in self.order_history:
            return {'status': 'unknown'}
        
        return {
            'status': self.order_history[order_id]['status'],
            'updated_at': self.order_history[order_id]['updated_at']
        }
    
    # Synchronous wrapper for backward compatibility
    def cancel_order(self, order_id: str) -> asyncio.Future:
        """Cancel an order (synchronous wrapper for backward compatibility).
        
        Args:
            order_id: Order identifier
            
        Returns:
            Future that will resolve to True if order was canceled, False otherwise
        """
        # Check if we're in an event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an event loop, create a task
                return asyncio.create_task(self.cancel_order_async(order_id))
            else:
                # No running event loop, use run_until_complete
                return loop.run_until_complete(self.cancel_order_async(order_id))
        except RuntimeError:
            # No event loop, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self.cancel_order_async(order_id))
    
    async def cancel_order_async(self, order_id: str) -> bool:
        """Cancel an order.
        
        Args:
            order_id: Order identifier
            
        Returns:
            True if order was canceled, False otherwise
        """
        if order_id not in self.order_history:
            return False
        
        # Update order status
        self.order_history[order_id]['status'] = 'canceling'
        self.order_history[order_id]['updated_at'] = time.time()
        
        # Cancel child orders
        for child_order in self.order_history[order_id]['child_orders']:
            if child_order.status not in [OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED]:
                await self.order_router.cancel_order_async(child_order)
        
        # Update order status
        self.order_history[order_id]['status'] = 'canceled'
        self.order_history[order_id]['updated_at'] = time.time()
        
        return True
    
    async def _worker(self) -> None:
        """Worker task for processing orders."""
        while True:
            # Get order from queue
            order = await self.order_queue.get()
            
            # Execute order
            await self._execute_order(order)
            
            # Mark task as done
            self.order_queue.task_done()
    
    async def _execute_order(self, order: Order) -> None:
        """Execute an order asynchronously.
        
        Args:
            order: Order to execute
        """
        # Start latency profiling
        if self.latency_profiler:
            self.latency_profiler.start_timer(f"execute_async_{order.id}")
        
        # Update order status
        self.order_history[order.id]['status'] = 'processing'
        self.order_history[order.id]['updated_at'] = time.time()
        
        try:
            # Route order based on type
            if order.type == OrderType.MARKET:
                child_order = await self.order_router.submit_order_async(order)
                child_orders = [child_order]
            
            elif order.type == OrderType.LIMIT:
                child_order = await self.order_router.submit_order_async(order)
                child_orders = [child_order]
            
            elif order.type == OrderType.ICEBERG:
                # Extract iceberg parameters from metadata
                metadata = order.metadata or {}
                display_size = metadata.get('display_size', order.quantity * 0.1)
                
                # Calculate number of child orders
                num_orders = int(np.ceil(order.quantity / display_size))
                
                # Create and submit child orders
                child_orders = []
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
                    executed_order = await self.order_router.submit_order_async(child_order)
                    child_orders.append(executed_order)
                    
                    # Update remaining quantity
                    remaining_quantity -= child_quantity
            
            else:
                # For other order types, just submit directly
                child_order = await self.order_router.submit_order_async(order)
                child_orders = [child_order]
            
            # Store child orders
            self.order_history[order.id]['child_orders'] = child_orders
            
            # Check if all child orders were filled
            all_filled = all(child.status == OrderStatus.FILLED for child in child_orders)
            any_rejected = any(child.status == OrderStatus.REJECTED for child in child_orders)
            
            # Update order status
            if all_filled:
                self.order_history[order.id]['status'] = 'executed'
            elif any_rejected:
                self.order_history[order.id]['status'] = 'partially_rejected'
            else:
                self.order_history[order.id]['status'] = 'partially_executed'
            
        except Exception as e:
            logger.error(f"Async order execution failed: {e}")
            self.order_history[order.id]['status'] = 'failed'
            self.order_history[order.id]['error'] = str(e)
        
        # Update timestamp
        self.order_history[order.id]['updated_at'] = time.time()
        
        # Stop latency profiling
        if self.latency_profiler:
            latency = self.latency_profiler.stop_timer(f"execute_async_{order.id}", 'end_to_end')
            logger.debug(f"Async order execution latency: {latency:.2f} μs")
