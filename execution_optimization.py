#!/usr/bin/env python
"""
Execution Optimization Module for Trading-Agent System

This module provides execution optimization capabilities for the Trading-Agent system,
including order routing, latency profiling, and smart order execution.
"""

import os
import sys
import time
import uuid
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from enum import Enum
from datetime import datetime, timedelta
from collections import deque, OrderedDict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("execution_optimization.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("execution_optimization")

class OrderType(Enum):
    """Order types supported by the system"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    ICEBERG = "iceberg"
    TWAP = "twap"
    VWAP = "vwap"
    SMART = "smart"

class OrderSide(Enum):
    """Order sides supported by the system"""
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    """Order statuses supported by the system"""
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class Order:
    """Order class for the Trading-Agent system"""
    
    def __init__(self, 
                 symbol: str, 
                 side: OrderSide, 
                 order_type: OrderType, 
                 quantity: float, 
                 price: Optional[float] = None,
                 stop_price: Optional[float] = None,
                 client_order_id: Optional[str] = None,
                 time_in_force: str = "GTC",
                 iceberg_qty: Optional[float] = None,
                 execution_window: Optional[int] = None,
                 execution_style: Optional[str] = None):
        """Initialize order
        
        Args:
            symbol: Trading symbol
            side: Order side (buy/sell)
            order_type: Order type
            quantity: Order quantity
            price: Order price (optional for market orders)
            stop_price: Stop price (for stop orders)
            client_order_id: Client order ID
            time_in_force: Time in force
            iceberg_qty: Iceberg quantity
            execution_window: Execution window in seconds
            execution_style: Execution style
        """
        self.symbol = symbol
        self.side = side
        self.order_type = order_type
        self.quantity = quantity
        self.price = price
        self.stop_price = stop_price
        self.client_order_id = client_order_id or str(uuid.uuid4())
        self.time_in_force = time_in_force
        self.iceberg_qty = iceberg_qty
        self.execution_window = execution_window
        self.execution_style = execution_style
        
        # Order status
        self.status = OrderStatus.PENDING
        self.filled_quantity = 0.0
        self.average_price = 0.0
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.exchange_order_id = None
        
        # Execution metrics
        self.submission_latency = None
        self.execution_latency = None
        self.slippage = None
        self.market_impact = None
        
        # Child orders (for smart orders)
        self.child_orders = []
    
    def update_status(self, status: OrderStatus, filled_quantity: float = None, average_price: float = None):
        """Update order status
        
        Args:
            status: New order status
            filled_quantity: Filled quantity
            average_price: Average fill price
        """
        self.status = status
        if filled_quantity is not None:
            self.filled_quantity = filled_quantity
        if average_price is not None:
            self.average_price = average_price
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict:
        """Convert order to dictionary
        
        Returns:
            dict: Order as dictionary
        """
        return {
            "symbol": self.symbol,
            "side": self.side.value if isinstance(self.side, OrderSide) else self.side,
            "order_type": self.order_type.value if isinstance(self.order_type, OrderType) else self.order_type,
            "quantity": self.quantity,
            "price": self.price,
            "stop_price": self.stop_price,
            "client_order_id": self.client_order_id,
            "time_in_force": self.time_in_force,
            "iceberg_qty": self.iceberg_qty,
            "execution_window": self.execution_window,
            "execution_style": self.execution_style,
            "status": self.status.value if isinstance(self.status, OrderStatus) else self.status,
            "filled_quantity": self.filled_quantity,
            "average_price": self.average_price,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "exchange_order_id": self.exchange_order_id,
            "submission_latency": self.submission_latency,
            "execution_latency": self.execution_latency,
            "slippage": self.slippage,
            "market_impact": self.market_impact
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Order':
        """Create order from dictionary
        
        Args:
            data: Order data
            
        Returns:
            Order: Order object
        """
        order = cls(
            symbol=data["symbol"],
            side=OrderSide(data["side"]) if isinstance(data["side"], str) else data["side"],
            order_type=OrderType(data["order_type"]) if isinstance(data["order_type"], str) else data["order_type"],
            quantity=data["quantity"],
            price=data.get("price"),
            stop_price=data.get("stop_price"),
            client_order_id=data.get("client_order_id"),
            time_in_force=data.get("time_in_force", "GTC"),
            iceberg_qty=data.get("iceberg_qty"),
            execution_window=data.get("execution_window"),
            execution_style=data.get("execution_style")
        )
        
        if "status" in data:
            order.status = OrderStatus(data["status"]) if isinstance(data["status"], str) else data["status"]
        if "filled_quantity" in data:
            order.filled_quantity = data["filled_quantity"]
        if "average_price" in data:
            order.average_price = data["average_price"]
        if "created_at" in data:
            order.created_at = datetime.fromisoformat(data["created_at"]) if isinstance(data["created_at"], str) else data["created_at"]
        if "updated_at" in data:
            order.updated_at = datetime.fromisoformat(data["updated_at"]) if isinstance(data["updated_at"], str) else data["updated_at"]
        if "exchange_order_id" in data:
            order.exchange_order_id = data["exchange_order_id"]
        if "submission_latency" in data:
            order.submission_latency = data["submission_latency"]
        if "execution_latency" in data:
            order.execution_latency = data["execution_latency"]
        if "slippage" in data:
            order.slippage = data["slippage"]
        if "market_impact" in data:
            order.market_impact = data["market_impact"]
        
        return order

class LatencyProfiler:
    """Latency profiler for execution optimization"""
    
    def __init__(self, rolling_window: int = 100):
        """Initialize latency profiler
        
        Args:
            rolling_window: Rolling window size for latency metrics
        """
        self.rolling_window = rolling_window
        self.latency_metrics = {}
        self.timers = {}
        self.thresholds = {}
        self.simulated_latencies = {}
        
        logger.info(f"Initialized LatencyProfiler with rolling_window={rolling_window}")
    
    def start_timer(self, operation: str, id: str = None):
        """Start timer for operation
        
        Args:
            operation: Operation name
            id: Operation ID
        """
        timer_id = f"{operation}_{id}" if id else operation
        self.timers[timer_id] = time.time()
    
    def stop_timer(self, operation: str, id: str = None) -> float:
        """Stop timer for operation and record latency
        
        Args:
            operation: Operation name
            id: Operation ID
            
        Returns:
            float: Latency in milliseconds
        """
        timer_id = f"{operation}_{id}" if id else operation
        
        if timer_id not in self.timers:
            logger.warning(f"Timer {timer_id} not started")
            return 0.0
        
        # Calculate latency
        latency = (time.time() - self.timers[timer_id]) * 1000  # Convert to milliseconds
        
        # Add simulated latency if configured
        if operation in self.simulated_latencies:
            latency += self.simulated_latencies[operation]
        
        # Remove timer
        del self.timers[timer_id]
        
        # Record latency
        if operation not in self.latency_metrics:
            self.latency_metrics[operation] = deque(maxlen=self.rolling_window)
        
        self.latency_metrics[operation].append(latency)
        
        # Log latency
        logger.info(f"{operation.capitalize()} {id} completed in {latency:.2f}ms")
        
        return latency
    
    def get_latency_metrics(self, operation: str) -> Dict:
        """Get latency metrics for operation
        
        Args:
            operation: Operation name
            
        Returns:
            dict: Latency metrics
        """
        if operation not in self.latency_metrics or not self.latency_metrics[operation]:
            return {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "median": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "count": 0
            }
        
        latencies = list(self.latency_metrics[operation])
        
        return {
            "min": min(latencies),
            "max": max(latencies),
            "mean": sum(latencies) / len(latencies),
            "median": sorted(latencies)[len(latencies) // 2],
            "p95": sorted(latencies)[int(len(latencies) * 0.95)],
            "p99": sorted(latencies)[int(len(latencies) * 0.99)],
            "count": len(latencies)
        }
    
    def get_all_metrics(self) -> Dict:
        """Get all latency metrics
        
        Returns:
            dict: All latency metrics
        """
        return {operation: self.get_latency_metrics(operation) for operation in self.latency_metrics}
    
    def set_threshold(self, operation: str, threshold: float):
        """Set latency threshold for operation
        
        Args:
            operation: Operation name
            threshold: Latency threshold in microseconds
        """
        self.thresholds[operation] = threshold
        logger.info(f"Set latency threshold for {operation}: {threshold} μs")
    
    def check_threshold(self, operation: str, latency: float) -> bool:
        """Check if latency exceeds threshold
        
        Args:
            operation: Operation name
            latency: Latency in milliseconds
            
        Returns:
            bool: True if latency is below threshold, False otherwise
        """
        if operation not in self.thresholds:
            return True
        
        # Convert latency to microseconds for comparison
        latency_us = latency * 1000
        
        return latency_us <= self.thresholds[operation]
    
    def set_simulated_latency(self, operation: str, latency: float):
        """Set simulated latency for operation
        
        Args:
            operation: Operation name
            latency: Simulated latency in microseconds
        """
        # Convert microseconds to milliseconds for internal storage
        self.simulated_latencies[operation] = latency / 1000
        logger.info(f"Set simulated latency for {operation}: {latency} μs")

class OrderRouter:
    """Order router for execution optimization"""
    
    def __init__(self, 
                 max_retries: int = 3, 
                 retry_delay: float = 0.5,
                 client_instance=None,
                 latency_profiler: LatencyProfiler = None):
        """Initialize order router
        
        Args:
            max_retries: Maximum number of retries
            retry_delay: Retry delay in seconds
            client_instance: Exchange client instance
            latency_profiler: Latency profiler
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.exchanges = {}
        self.latency_profiler = latency_profiler or LatencyProfiler()
        
        # Register client if provided
        if client_instance:
            self.register_exchange("default", client_instance)
        
        logger.info(f"Initialized OrderRouter with max_retries={max_retries}, retry_delay={retry_delay}s")
    
    def register_exchange(self, name: str, client):
        """Register exchange client
        
        Args:
            name: Exchange name
            client: Exchange client
        """
        self.exchanges[name] = client
        logger.info(f"Registered exchange: {name}")
    
    def submit_order(self, order: Order, exchange: str = "default") -> Order:
        """Submit order to exchange
        
        Args:
            order: Order to submit
            exchange: Exchange name
            
        Returns:
            Order: Updated order
        """
        # Check if exchange is registered
        if exchange not in self.exchanges:
            logger.error(f"Exchange {exchange} not registered")
            order.update_status(OrderStatus.REJECTED)
            return order
        
        # Start latency timer
        self.latency_profiler.start_timer("submit", order.client_order_id)
        
        # Check latency threshold if set
        if "order_submission" in self.latency_profiler.simulated_latencies:
            simulated_latency_ms = self.latency_profiler.simulated_latencies["order_submission"]
            if "order_submission" in self.latency_profiler.thresholds:
                threshold_us = self.latency_profiler.thresholds["order_submission"]
                if simulated_latency_ms * 1000 > threshold_us:
                    logger.warning(f"High simulated latency detected ({simulated_latency_ms}ms), rejecting order")
                    order.update_status(OrderStatus.REJECTED)
                    return order
        
        # Submit order with retries
        for attempt in range(self.max_retries + 1):
            try:
                # Get exchange client
                client = self.exchanges[exchange]
                
                # Submit order based on type
                if order.order_type == OrderType.MARKET:
                    result = client.create_market_order(
                        symbol=order.symbol,
                        side=order.side.value,
                        quantity=order.quantity,
                        client_order_id=order.client_order_id
                    )
                elif order.order_type == OrderType.LIMIT:
                    result = client.create_limit_order(
                        symbol=order.symbol,
                        side=order.side.value,
                        quantity=order.quantity,
                        price=order.price,
                        time_in_force=order.time_in_force,
                        client_order_id=order.client_order_id
                    )
                elif order.order_type == OrderType.STOP_LOSS:
                    result = client.create_stop_loss_order(
                        symbol=order.symbol,
                        side=order.side.value,
                        quantity=order.quantity,
                        stop_price=order.stop_price,
                        client_order_id=order.client_order_id
                    )
                elif order.order_type == OrderType.TAKE_PROFIT:
                    result = client.create_take_profit_order(
                        symbol=order.symbol,
                        side=order.side.value,
                        quantity=order.quantity,
                        stop_price=order.stop_price,
                        client_order_id=order.client_order_id
                    )
                elif order.order_type == OrderType.ICEBERG:
                    result = self._execute_iceberg_order(order, client)
                elif order.order_type == OrderType.TWAP:
                    result = self._execute_twap_order(order, client)
                elif order.order_type == OrderType.VWAP:
                    result = self._execute_vwap_order(order, client)
                elif order.order_type == OrderType.SMART:
                    result = self._execute_smart_order(order, client)
                else:
                    logger.error(f"Unsupported order type: {order.order_type}")
                    order.update_status(OrderStatus.REJECTED)
                    return order
                
                # Update order with result
                if result:
                    if isinstance(result, dict):
                        if "status" in result:
                            order.update_status(OrderStatus(result["status"]))
                        if "filled_quantity" in result:
                            order.filled_quantity = result["filled_quantity"]
                        if "average_price" in result:
                            order.average_price = result["average_price"]
                        if "exchange_order_id" in result:
                            order.exchange_order_id = result["exchange_order_id"]
                    elif hasattr(result, "status"):
                        order.update_status(result.status)
                        if hasattr(result, "filled_quantity"):
                            order.filled_quantity = result.filled_quantity
                        if hasattr(result, "average_price"):
                            order.average_price = result.average_price
                        if hasattr(result, "exchange_order_id"):
                            order.exchange_order_id = result.exchange_order_id
                
                # Stop latency timer
                latency = self.latency_profiler.stop_timer("submit", order.client_order_id)
                order.submission_latency = latency
                
                return order
            
            except Exception as e:
                logger.error(f"Error submitting order: {str(e)}")
                
                # Retry if not last attempt
                if attempt < self.max_retries:
                    logger.info(f"Retrying in {self.retry_delay}s (attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"Max retries reached, rejecting order")
                    order.update_status(OrderStatus.REJECTED)
                    return order
        
        # Should never reach here
        order.update_status(OrderStatus.REJECTED)
        return order
    
    def cancel_order(self, order: Order, exchange: str = "default") -> Order:
        """Cancel order
        
        Args:
            order: Order to cancel
            exchange: Exchange name
            
        Returns:
            Order: Updated order
        """
        # Check if exchange is registered
        if exchange not in self.exchanges:
            logger.error(f"Exchange {exchange} not registered")
            return order
        
        # Start latency timer
        self.latency_profiler.start_timer("cancel", order.client_order_id)
        
        # Cancel order with retries
        for attempt in range(self.max_retries + 1):
            try:
                # Get exchange client
                client = self.exchanges[exchange]
                
                # Cancel order
                result = client.cancel_order(
                    symbol=order.symbol,
                    order_id=order.exchange_order_id,
                    client_order_id=order.client_order_id
                )
                
                # Update order with result
                if result:
                    if isinstance(result, dict):
                        if "status" in result:
                            order.update_status(OrderStatus(result["status"]))
                    elif hasattr(result, "status"):
                        order.update_status(result.status)
                else:
                    order.update_status(OrderStatus.CANCELED)
                
                # Stop latency timer
                latency = self.latency_profiler.stop_timer("cancel", order.client_order_id)
                
                return order
            
            except Exception as e:
                logger.error(f"Error canceling order: {str(e)}")
                
                # Retry if not last attempt
                if attempt < self.max_retries:
                    logger.info(f"Retrying in {self.retry_delay}s (attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"Max retries reached, order may not be canceled")
                    return order
        
        # Should never reach here
        return order
    
    def get_order_status(self, order: Order, exchange: str = "default") -> Order:
        """Get order status
        
        Args:
            order: Order to check
            exchange: Exchange name
            
        Returns:
            Order: Updated order
        """
        # Check if exchange is registered
        if exchange not in self.exchanges:
            logger.error(f"Exchange {exchange} not registered")
            return order
        
        # Start latency timer
        self.latency_profiler.start_timer("status", order.client_order_id)
        
        # Get order status with retries
        for attempt in range(self.max_retries + 1):
            try:
                # Get exchange client
                client = self.exchanges[exchange]
                
                # Get order status
                result = client.get_order(
                    symbol=order.symbol,
                    order_id=order.exchange_order_id,
                    client_order_id=order.client_order_id
                )
                
                # Update order with result
                if result:
                    if isinstance(result, dict):
                        if "status" in result:
                            order.update_status(OrderStatus(result["status"]))
                        if "filled_quantity" in result:
                            order.filled_quantity = result["filled_quantity"]
                        if "average_price" in result:
                            order.average_price = result["average_price"]
                    elif hasattr(result, "status"):
                        order.update_status(result.status)
                        if hasattr(result, "filled_quantity"):
                            order.filled_quantity = result.filled_quantity
                        if hasattr(result, "average_price"):
                            order.average_price = result.average_price
                
                # Stop latency timer
                latency = self.latency_profiler.stop_timer("status", order.client_order_id)
                
                return order
            
            except Exception as e:
                logger.error(f"Error getting order status: {str(e)}")
                
                # Retry if not last attempt
                if attempt < self.max_retries:
                    logger.info(f"Retrying in {self.retry_delay}s (attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"Max retries reached, order status unknown")
                    return order
        
        # Should never reach here
        return order
    
    def _execute_iceberg_order(self, order: Order, client) -> Dict:
        """Execute iceberg order
        
        Args:
            order: Iceberg order
            client: Exchange client
            
        Returns:
            dict: Order result
        """
        # Calculate chunk size
        chunk_size = order.iceberg_qty or (order.quantity * 0.1)  # Default to 10% of total quantity
        
        # Calculate number of chunks
        num_chunks = int(order.quantity / chunk_size)
        last_chunk = order.quantity % chunk_size
        
        # Execute chunks
        filled_quantity = 0.0
        total_cost = 0.0
        
        for i in range(num_chunks):
            # Create chunk order
            chunk_order = Order(
                symbol=order.symbol,
                side=order.side,
                order_type=OrderType.LIMIT,
                quantity=chunk_size,
                price=order.price,
                client_order_id=f"{order.client_order_id}_chunk_{i}",
                time_in_force=order.time_in_force
            )
            
            # Submit chunk order
            result = self.submit_order(chunk_order)
            
            # Update filled quantity and cost
            filled_quantity += result.filled_quantity
            total_cost += result.filled_quantity * result.average_price
            
            # Add chunk order to child orders
            order.child_orders.append(result)
            
            # Wait for chunk to be filled or timeout
            timeout = time.time() + 60  # 60 seconds timeout
            while result.status not in [OrderStatus.FILLED, OrderStatus.REJECTED, OrderStatus.CANCELED] and time.time() < timeout:
                time.sleep(1)
                result = self.get_order_status(result)
            
            # Cancel if not filled
            if result.status not in [OrderStatus.FILLED, OrderStatus.REJECTED, OrderStatus.CANCELED]:
                self.cancel_order(result)
        
        # Execute last chunk if any
        if last_chunk > 0:
            # Create last chunk order
            last_chunk_order = Order(
                symbol=order.symbol,
                side=order.side,
                order_type=OrderType.LIMIT,
                quantity=last_chunk,
                price=order.price,
                client_order_id=f"{order.client_order_id}_chunk_{num_chunks}",
                time_in_force=order.time_in_force
            )
            
            # Submit last chunk order
            result = self.submit_order(last_chunk_order)
            
            # Update filled quantity and cost
            filled_quantity += result.filled_quantity
            total_cost += result.filled_quantity * result.average_price
            
            # Add last chunk order to child orders
            order.child_orders.append(result)
        
        # Calculate average price
        average_price = total_cost / filled_quantity if filled_quantity > 0 else 0.0
        
        # Return result
        return {
            "status": OrderStatus.FILLED.value if filled_quantity >= order.quantity * 0.99 else OrderStatus.PARTIALLY_FILLED.value,
            "filled_quantity": filled_quantity,
            "average_price": average_price,
            "exchange_order_id": order.client_order_id
        }
    
    def _execute_twap_order(self, order: Order, client) -> Dict:
        """Execute TWAP order
        
        Args:
            order: TWAP order
            client: Exchange client
            
        Returns:
            dict: Order result
        """
        # Calculate execution window
        execution_window = order.execution_window or 3600  # Default to 1 hour
        
        # Calculate number of chunks and interval
        num_chunks = 10  # Default to 10 chunks
        interval = execution_window / num_chunks
        chunk_size = order.quantity / num_chunks
        
        # Execute chunks
        filled_quantity = 0.0
        total_cost = 0.0
        start_time = time.time()
        
        for i in range(num_chunks):
            # Wait until next interval
            next_execution = start_time + (i * interval)
            wait_time = next_execution - time.time()
            if wait_time > 0:
                time.sleep(wait_time)
            
            # Create chunk order
            chunk_order = Order(
                symbol=order.symbol,
                side=order.side,
                order_type=OrderType.MARKET,
                quantity=chunk_size,
                client_order_id=f"{order.client_order_id}_chunk_{i}"
            )
            
            # Submit chunk order
            result = self.submit_order(chunk_order)
            
            # Update filled quantity and cost
            filled_quantity += result.filled_quantity
            total_cost += result.filled_quantity * result.average_price
            
            # Add chunk order to child orders
            order.child_orders.append(result)
        
        # Calculate average price
        average_price = total_cost / filled_quantity if filled_quantity > 0 else 0.0
        
        # Return result
        return {
            "status": OrderStatus.FILLED.value if filled_quantity >= order.quantity * 0.99 else OrderStatus.PARTIALLY_FILLED.value,
            "filled_quantity": filled_quantity,
            "average_price": average_price,
            "exchange_order_id": order.client_order_id
        }
    
    def _execute_vwap_order(self, order: Order, client) -> Dict:
        """Execute VWAP order
        
        Args:
            order: VWAP order
            client: Exchange client
            
        Returns:
            dict: Order result
        """
        # Calculate execution window
        execution_window = order.execution_window or 3600  # Default to 1 hour
        
        # Get historical volume profile
        try:
            # Get historical volume data
            volume_profile = client.get_historical_volume(
                symbol=order.symbol,
                interval="5m",
                limit=12
            )
            
            # Calculate volume distribution
            total_volume = sum(v for _, v in volume_profile)
            volume_distribution = [v / total_volume for _, v in volume_profile]
            
            # Calculate number of chunks and sizes
            num_chunks = len(volume_distribution)
            chunk_sizes = [order.quantity * dist for dist in volume_distribution]
            interval = execution_window / num_chunks
        except Exception as e:
            logger.error(f"Error getting volume profile: {str(e)}")
            
            # Fall back to TWAP
            logger.info("Falling back to TWAP execution")
            return self._execute_twap_order(order, client)
        
        # Execute chunks
        filled_quantity = 0.0
        total_cost = 0.0
        start_time = time.time()
        
        for i in range(num_chunks):
            # Wait until next interval
            next_execution = start_time + (i * interval)
            wait_time = next_execution - time.time()
            if wait_time > 0:
                time.sleep(wait_time)
            
            # Create chunk order
            chunk_order = Order(
                symbol=order.symbol,
                side=order.side,
                order_type=OrderType.MARKET,
                quantity=chunk_sizes[i],
                client_order_id=f"{order.client_order_id}_chunk_{i}"
            )
            
            # Submit chunk order
            result = self.submit_order(chunk_order)
            
            # Update filled quantity and cost
            filled_quantity += result.filled_quantity
            total_cost += result.filled_quantity * result.average_price
            
            # Add chunk order to child orders
            order.child_orders.append(result)
        
        # Calculate average price
        average_price = total_cost / filled_quantity if filled_quantity > 0 else 0.0
        
        # Return result
        return {
            "status": OrderStatus.FILLED.value if filled_quantity >= order.quantity * 0.99 else OrderStatus.PARTIALLY_FILLED.value,
            "filled_quantity": filled_quantity,
            "average_price": average_price,
            "exchange_order_id": order.client_order_id
        }
    
    def _execute_smart_order(self, order: Order, client) -> Dict:
        """Execute smart order
        
        Args:
            order: Smart order
            client: Exchange client
            
        Returns:
            dict: Order result
        """
        # Get market conditions
        try:
            # Get order book
            order_book = client.get_order_book(
                symbol=order.symbol,
                limit=20
            )
            
            # Get recent trades
            recent_trades = client.get_recent_trades(
                symbol=order.symbol,
                limit=100
            )
            
            # Analyze market conditions
            bid_ask_spread = order_book["asks"][0][0] - order_book["bids"][0][0]
            market_depth = sum(q for _, q in order_book["bids"][:5]) + sum(q for _, q in order_book["asks"][:5])
            recent_volume = sum(t["quantity"] for t in recent_trades)
            
            # Determine execution strategy
            if bid_ask_spread > 0.001 * order_book["bids"][0][0]:  # Spread > 0.1%
                logger.info("Wide spread detected, using iceberg execution")
                execution_style = "iceberg"
            elif market_depth < order.quantity * 10:  # Low liquidity
                logger.info("Low liquidity detected, using TWAP execution")
                execution_style = "twap"
            elif recent_volume > order.quantity * 100:  # High volume
                logger.info("High volume detected, using VWAP execution")
                execution_style = "vwap"
            else:
                logger.info("Normal market conditions, using iceberg execution")
                execution_style = "iceberg"
        except Exception as e:
            logger.error(f"Error analyzing market conditions: {str(e)}")
            
            # Fall back to iceberg
            logger.info("Falling back to iceberg execution")
            execution_style = "iceberg"
        
        # Execute based on determined strategy
        if execution_style == "iceberg":
            # Create iceberg order
            iceberg_order = Order(
                symbol=order.symbol,
                side=order.side,
                order_type=OrderType.ICEBERG,
                quantity=order.quantity,
                price=order.price,
                client_order_id=order.client_order_id,
                time_in_force=order.time_in_force,
                iceberg_qty=order.quantity * 0.1  # 10% of total quantity
            )
            
            return self._execute_iceberg_order(iceberg_order, client)
        elif execution_style == "twap":
            # Create TWAP order
            twap_order = Order(
                symbol=order.symbol,
                side=order.side,
                order_type=OrderType.TWAP,
                quantity=order.quantity,
                client_order_id=order.client_order_id,
                execution_window=1800  # 30 minutes
            )
            
            return self._execute_twap_order(twap_order, client)
        elif execution_style == "vwap":
            # Create VWAP order
            vwap_order = Order(
                symbol=order.symbol,
                side=order.side,
                order_type=OrderType.VWAP,
                quantity=order.quantity,
                client_order_id=order.client_order_id,
                execution_window=1800  # 30 minutes
            )
            
            return self._execute_vwap_order(vwap_order, client)
        else:
            # Should never reach here
            logger.error(f"Unknown execution style: {execution_style}")
            
            # Fall back to market order
            market_order = Order(
                symbol=order.symbol,
                side=order.side,
                order_type=OrderType.MARKET,
                quantity=order.quantity,
                client_order_id=order.client_order_id
            )
            
            result = self.submit_order(market_order)
            
            return {
                "status": result.status.value,
                "filled_quantity": result.filled_quantity,
                "average_price": result.average_price,
                "exchange_order_id": result.exchange_order_id
            }

class SmartOrderRouter(OrderRouter):
    """Smart order router for execution optimization"""
    
    def __init__(self, 
                 max_retries: int = 3, 
                 retry_delay: float = 0.5,
                 client_instance=None,
                 latency_profiler: LatencyProfiler = None):
        """Initialize smart order router
        
        Args:
            max_retries: Maximum number of retries
            retry_delay: Retry delay in seconds
            client_instance: Exchange client instance
            latency_profiler: Latency profiler
        """
        super().__init__(max_retries, retry_delay, client_instance, latency_profiler)
        self.exchange_metrics = {}
        self.routing_history = []
        
        logger.info("Initialized SmartOrderRouter")
    
    def submit_order(self, order: Order, exchange: str = None) -> Order:
        """Submit order to best exchange
        
        Args:
            order: Order to submit
            exchange: Exchange name (optional)
            
        Returns:
            Order: Updated order
        """
        # If exchange is specified, use it
        if exchange:
            return super().submit_order(order, exchange)
        
        # Otherwise, find best exchange
        best_exchange = self._find_best_exchange(order)
        
        # Submit order to best exchange
        result = super().submit_order(order, best_exchange)
        
        # Update routing history
        self.routing_history.append({
            "timestamp": time.time(),
            "order_id": order.client_order_id,
            "symbol": order.symbol,
            "exchange": best_exchange,
            "latency": result.submission_latency
        })
        
        return result
    
    def _find_best_exchange(self, order: Order) -> str:
        """Find best exchange for order
        
        Args:
            order: Order to route
            
        Returns:
            str: Best exchange name
        """
        # If only one exchange is registered, use it
        if len(self.exchanges) == 1:
            return list(self.exchanges.keys())[0]
        
        # Calculate scores for each exchange
        scores = {}
        
        for name in self.exchanges:
            # Initialize score
            scores[name] = 0.0
            
            # Check if exchange supports symbol
            try:
                if not self._check_symbol_supported(name, order.symbol):
                    scores[name] = -1000  # Large negative score to avoid this exchange
                    continue
            except Exception as e:
                logger.error(f"Error checking symbol support: {str(e)}")
                scores[name] = -500  # Negative score but not as bad as confirmed unsupported
            
            # Check latency
            try:
                latency = self._get_exchange_latency(name)
                # Lower latency is better
                scores[name] += 100 - min(latency, 100)
            except Exception as e:
                logger.error(f"Error checking latency: {str(e)}")
            
            # Check fees
            try:
                fee = self._get_exchange_fee(name, order)
                # Lower fee is better
                scores[name] += 50 - min(fee * 10000, 50)
            except Exception as e:
                logger.error(f"Error checking fees: {str(e)}")
            
            # Check liquidity
            try:
                liquidity = self._get_exchange_liquidity(name, order)
                # Higher liquidity is better
                scores[name] += min(liquidity / order.quantity, 50)
            except Exception as e:
                logger.error(f"Error checking liquidity: {str(e)}")
            
            # Check historical performance
            try:
                performance = self._get_historical_performance(name)
                # Higher performance is better
                scores[name] += performance * 20
            except Exception as e:
                logger.error(f"Error checking historical performance: {str(e)}")
        
        # Find exchange with highest score
        best_exchange = max(scores, key=scores.get)
        
        # If best score is negative, use default exchange
        if scores[best_exchange] < 0:
            logger.warning("All exchanges have negative scores, using default")
            return list(self.exchanges.keys())[0]
        
        logger.info(f"Selected exchange {best_exchange} with score {scores[best_exchange]}")
        
        return best_exchange
    
    def _check_symbol_supported(self, exchange: str, symbol: str) -> bool:
        """Check if symbol is supported by exchange
        
        Args:
            exchange: Exchange name
            symbol: Symbol to check
            
        Returns:
            bool: True if symbol is supported, False otherwise
        """
        # Get exchange client
        client = self.exchanges[exchange]
        
        # Check if client has get_symbol_info method
        if hasattr(client, "get_symbol_info"):
            symbol_info = client.get_symbol_info(symbol)
            return symbol_info is not None
        
        # Check if client has get_exchange_info method
        if hasattr(client, "get_exchange_info"):
            exchange_info = client.get_exchange_info()
            symbols = [s["symbol"] for s in exchange_info["symbols"]]
            return symbol in symbols
        
        # If no method is available, assume symbol is supported
        return True
    
    def _get_exchange_latency(self, exchange: str) -> float:
        """Get exchange latency
        
        Args:
            exchange: Exchange name
            
        Returns:
            float: Exchange latency in milliseconds
        """
        # Check if we have latency metrics for this exchange
        if exchange in self.exchange_metrics and "latency" in self.exchange_metrics[exchange]:
            return self.exchange_metrics[exchange]["latency"]
        
        # Otherwise, use ping to estimate latency
        client = self.exchanges[exchange]
        
        # Check if client has ping method
        if hasattr(client, "ping"):
            start_time = time.time()
            client.ping()
            latency = (time.time() - start_time) * 1000  # Convert to milliseconds
        else:
            # If no ping method, use server time
            start_time = time.time()
            if hasattr(client, "get_server_time"):
                client.get_server_time()
            latency = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Store latency in metrics
        if exchange not in self.exchange_metrics:
            self.exchange_metrics[exchange] = {}
        self.exchange_metrics[exchange]["latency"] = latency
        
        return latency
    
    def _get_exchange_fee(self, exchange: str, order: Order) -> float:
        """Get exchange fee for order
        
        Args:
            exchange: Exchange name
            order: Order to check
            
        Returns:
            float: Exchange fee as a decimal (e.g., 0.001 for 0.1%)
        """
        # Check if we have fee metrics for this exchange
        if exchange in self.exchange_metrics and "fee" in self.exchange_metrics[exchange]:
            return self.exchange_metrics[exchange]["fee"]
        
        # Otherwise, get fee from exchange
        client = self.exchanges[exchange]
        
        # Check if client has get_trade_fee method
        if hasattr(client, "get_trade_fee"):
            fee_info = client.get_trade_fee(symbol=order.symbol)
            if order.side == OrderSide.BUY:
                fee = fee_info.get("makerFee", 0.001)  # Default to 0.1%
            else:
                fee = fee_info.get("takerFee", 0.001)  # Default to 0.1%
        else:
            # If no method, use default fee
            fee = 0.001  # Default to 0.1%
        
        # Store fee in metrics
        if exchange not in self.exchange_metrics:
            self.exchange_metrics[exchange] = {}
        self.exchange_metrics[exchange]["fee"] = fee
        
        return fee
    
    def _get_exchange_liquidity(self, exchange: str, order: Order) -> float:
        """Get exchange liquidity for order
        
        Args:
            exchange: Exchange name
            order: Order to check
            
        Returns:
            float: Exchange liquidity as a ratio to order quantity
        """
        # Get exchange client
        client = self.exchanges[exchange]
        
        # Get order book
        if hasattr(client, "get_order_book"):
            order_book = client.get_order_book(symbol=order.symbol, limit=20)
            
            # Calculate liquidity based on order side
            if order.side == OrderSide.BUY:
                liquidity = sum(float(q) for _, q in order_book["asks"][:10])
            else:
                liquidity = sum(float(q) for _, q in order_book["bids"][:10])
            
            return liquidity
        
        # If no method, return default liquidity
        return order.quantity * 10  # Assume 10x order quantity
    
    def _get_historical_performance(self, exchange: str) -> float:
        """Get historical performance for exchange
        
        Args:
            exchange: Exchange name
            
        Returns:
            float: Historical performance score (0-1)
        """
        # Check if we have performance metrics for this exchange
        if exchange in self.exchange_metrics and "performance" in self.exchange_metrics[exchange]:
            return self.exchange_metrics[exchange]["performance"]
        
        # Otherwise, calculate from routing history
        if not self.routing_history:
            return 0.5  # Default score
        
        # Filter history for this exchange
        exchange_history = [h for h in self.routing_history if h["exchange"] == exchange]
        
        if not exchange_history:
            return 0.5  # Default score
        
        # Calculate performance based on latency
        latencies = [h["latency"] for h in exchange_history if "latency" in h]
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            latency_score = max(0, min(1, 1 - (avg_latency / 1000)))  # 0-1 score, lower latency is better
        else:
            latency_score = 0.5
        
        # Store performance in metrics
        if exchange not in self.exchange_metrics:
            self.exchange_metrics[exchange] = {}
        self.exchange_metrics[exchange]["performance"] = latency_score
        
        return latency_score

class ExecutionOptimizer:
    """Execution optimizer for trading strategies"""
    
    def __init__(self, 
                 client_instance=None,
                 latency_profiler: LatencyProfiler = None,
                 use_smart_routing: bool = True):
        """Initialize execution optimizer
        
        Args:
            client_instance: Exchange client instance
            latency_profiler: Latency profiler
            use_smart_routing: Whether to use smart routing
        """
        self.latency_profiler = latency_profiler or LatencyProfiler()
        
        # Initialize order router
        if use_smart_routing:
            self.router = SmartOrderRouter(client_instance=client_instance, latency_profiler=self.latency_profiler)
        else:
            self.router = OrderRouter(client_instance=client_instance, latency_profiler=self.latency_profiler)
        
        # Initialize execution metrics
        self.execution_metrics = {
            "orders_submitted": 0,
            "orders_filled": 0,
            "orders_rejected": 0,
            "total_volume": 0.0,
            "total_cost": 0.0,
            "avg_slippage": 0.0,
            "avg_market_impact": 0.0
        }
        
        logger.info(f"Initialized ExecutionOptimizer with use_smart_routing={use_smart_routing}")
    
    def execute_order(self, order: Order, exchange: str = None) -> Order:
        """Execute order
        
        Args:
            order: Order to execute
            exchange: Exchange name (optional)
            
        Returns:
            Order: Executed order
        """
        # Submit order
        result = self.router.submit_order(order, exchange)
        
        # Update execution metrics
        self.execution_metrics["orders_submitted"] += 1
        
        if result.status == OrderStatus.FILLED:
            self.execution_metrics["orders_filled"] += 1
            self.execution_metrics["total_volume"] += result.filled_quantity
            self.execution_metrics["total_cost"] += result.filled_quantity * result.average_price
        elif result.status == OrderStatus.REJECTED:
            self.execution_metrics["orders_rejected"] += 1
        
        # Calculate slippage and market impact
        if result.status == OrderStatus.FILLED and result.price is not None:
            # Calculate slippage
            slippage = (result.average_price - result.price) / result.price
            if order.side == OrderSide.SELL:
                slippage = -slippage
            
            result.slippage = slippage
            
            # Update average slippage
            self.execution_metrics["avg_slippage"] = (
                (self.execution_metrics["avg_slippage"] * (self.execution_metrics["orders_filled"] - 1)) + slippage
            ) / self.execution_metrics["orders_filled"]
            
            # Calculate market impact (simplified)
            # In a real system, this would compare to pre-order market price
            result.market_impact = abs(slippage)
            
            # Update average market impact
            self.execution_metrics["avg_market_impact"] = (
                (self.execution_metrics["avg_market_impact"] * (self.execution_metrics["orders_filled"] - 1)) + result.market_impact
            ) / self.execution_metrics["orders_filled"]
        
        return result
    
    def execute_orders_async(self, orders: List[Order], exchange: str = None) -> List[Order]:
        """Execute multiple orders asynchronously
        
        Args:
            orders: Orders to execute
            exchange: Exchange name (optional)
            
        Returns:
            list: Executed orders
        """
        # In a real system, this would use async/await or threading
        # For simplicity, we'll just execute orders sequentially
        results = []
        
        for order in orders:
            result = self.execute_order(order, exchange)
            results.append(result)
        
        return results
    
    def create_iceberg_order(self, 
                            symbol: str, 
                            side: OrderSide, 
                            quantity: float, 
                            price: float,
                            chunk_size: float = None) -> Order:
        """Create iceberg order
        
        Args:
            symbol: Trading symbol
            side: Order side
            quantity: Order quantity
            price: Order price
            chunk_size: Chunk size
            
        Returns:
            Order: Iceberg order
        """
        # Calculate chunk size if not provided
        if chunk_size is None:
            chunk_size = quantity * 0.1  # Default to 10% of total quantity
        
        # Create iceberg order
        order = Order(
            symbol=symbol,
            side=side,
            order_type=OrderType.ICEBERG,
            quantity=quantity,
            price=price,
            iceberg_qty=chunk_size
        )
        
        return order
    
    def create_twap_order(self,
                         symbol: str,
                         side: OrderSide,
                         quantity: float,
                         execution_window: int = 3600) -> Order:
        """Create TWAP order
        
        Args:
            symbol: Trading symbol
            side: Order side
            quantity: Order quantity
            execution_window: Execution window in seconds
            
        Returns:
            Order: TWAP order
        """
        # Create TWAP order
        order = Order(
            symbol=symbol,
            side=side,
            order_type=OrderType.TWAP,
            quantity=quantity,
            execution_window=execution_window
        )
        
        return order
    
    def create_vwap_order(self,
                         symbol: str,
                         side: OrderSide,
                         quantity: float,
                         execution_window: int = 3600) -> Order:
        """Create VWAP order
        
        Args:
            symbol: Trading symbol
            side: Order side
            quantity: Order quantity
            execution_window: Execution window in seconds
            
        Returns:
            Order: VWAP order
        """
        # Create VWAP order
        order = Order(
            symbol=symbol,
            side=side,
            order_type=OrderType.VWAP,
            quantity=quantity,
            execution_window=execution_window
        )
        
        return order
    
    def create_smart_order(self,
                          symbol: str,
                          side: OrderSide,
                          quantity: float,
                          price: Optional[float] = None) -> Order:
        """Create smart order
        
        Args:
            symbol: Trading symbol
            side: Order side
            quantity: Order quantity
            price: Order price (optional)
            
        Returns:
            Order: Smart order
        """
        # Create smart order
        order = Order(
            symbol=symbol,
            side=side,
            order_type=OrderType.SMART,
            quantity=quantity,
            price=price
        )
        
        return order
    
    def get_execution_metrics(self) -> Dict:
        """Get execution metrics
        
        Returns:
            dict: Execution metrics
        """
        return self.execution_metrics
    
    def get_latency_metrics(self) -> Dict:
        """Get latency metrics
        
        Returns:
            dict: Latency metrics
        """
        return self.latency_profiler.get_all_metrics()
    
    def set_latency_threshold(self, operation: str, threshold: float):
        """Set latency threshold
        
        Args:
            operation: Operation name
            threshold: Latency threshold in microseconds
        """
        self.latency_profiler.set_threshold(operation, threshold)
    
    def set_simulated_latency(self, operation: str, latency: float):
        """Set simulated latency
        
        Args:
            operation: Operation name
            latency: Simulated latency in microseconds
        """
        self.latency_profiler.set_simulated_latency(operation, latency)
