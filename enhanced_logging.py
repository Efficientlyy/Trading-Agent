#!/usr/bin/env python
"""
Enhanced Logging System for Trading-Agent

This module provides a comprehensive logging system for the Trading-Agent
with structured logging, log rotation, and centralized configuration.
"""

import os
import sys
import json
import time
import logging
import logging.handlers
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta

# Configure base logging directory
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Log file paths
SYSTEM_LOG = os.path.join(LOG_DIR, "system.log")
SIGNAL_LOG = os.path.join(LOG_DIR, "signals.log")
ORDER_LOG = os.path.join(LOG_DIR, "orders.log")
ERROR_LOG = os.path.join(LOG_DIR, "errors.log")
API_LOG = os.path.join(LOG_DIR, "api.log")
PERFORMANCE_LOG = os.path.join(LOG_DIR, "performance.log")

# Log rotation settings
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10 MB
BACKUP_COUNT = 5

class ContextFilter(logging.Filter):
    """Filter that adds contextual information to log records"""
    
    def __init__(self, name=''):
        super().__init__(name)
        self.context = {}
    
    def filter(self, record):
        # Add context attributes to the record
        for key, value in self.context.items():
            setattr(record, key, value)
        return True

class EnhancedLogger:
    """Enhanced logging system for Trading-Agent"""
    
    def __init__(self, name="trading_agent"):
        """Initialize enhanced logger
        
        Args:
            name: Logger name
        """
        self.name = name
        self.loggers = {}
        self.context_filter = ContextFilter()
        
        # Initialize loggers
        self.setup_system_logger()
        self.setup_signal_logger()
        self.setup_order_logger()
        self.setup_error_logger()
        self.setup_api_logger()
        self.setup_performance_logger()
        
        # Set default context
        self.set_context(
            system_id=f"SYSTEM-{int(time.time())}",
            session_id=f"SESSION-{int(time.time())}",
            version="1.0.0"
        )
        
        self.system.info("Enhanced logging system initialized")
    
    def setup_system_logger(self):
        """Set up system logger"""
        self.system = logging.getLogger(f"{self.name}.system")
        self.system.setLevel(logging.INFO)
        self.system.addFilter(self.context_filter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.INFO)
        
        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            SYSTEM_LOG, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT
        )
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(system_id)s] [%(session_id)s] - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        
        # Add handlers
        self.system.addHandler(console_handler)
        self.system.addHandler(file_handler)
        
        # Store in loggers dict
        self.loggers['system'] = self.system
    
    def setup_signal_logger(self):
        """Set up signal logger"""
        self.signal = logging.getLogger(f"{self.name}.signal")
        self.signal.setLevel(logging.DEBUG)
        self.signal.addFilter(self.context_filter)
        
        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            SIGNAL_LOG, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT
        )
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - [%(system_id)s] [%(session_id)s] [%(signal_id)s] - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        
        # Add handlers
        self.signal.addHandler(file_handler)
        
        # Store in loggers dict
        self.loggers['signal'] = self.signal
    
    def setup_order_logger(self):
        """Set up order logger"""
        self.order = logging.getLogger(f"{self.name}.order")
        self.order.setLevel(logging.DEBUG)
        self.order.addFilter(self.context_filter)
        
        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            ORDER_LOG, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT
        )
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - [%(system_id)s] [%(session_id)s] [%(order_id)s] - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        
        # Add handlers
        self.order.addHandler(file_handler)
        
        # Store in loggers dict
        self.loggers['order'] = self.order
    
    def setup_error_logger(self):
        """Set up error logger"""
        self.error = logging.getLogger(f"{self.name}.error")
        self.error.setLevel(logging.ERROR)
        self.error.addFilter(self.context_filter)
        
        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            ERROR_LOG, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT
        )
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(system_id)s] [%(session_id)s] - %(message)s\n'
            'Exception: %(exc_info)s\n'
            'Stack trace: %(stack_info)s\n'
        )
        file_handler.setFormatter(file_formatter)
        
        # Add handlers
        self.error.addHandler(file_handler)
        
        # Store in loggers dict
        self.loggers['error'] = self.error
    
    def setup_api_logger(self):
        """Set up API logger"""
        self.api = logging.getLogger(f"{self.name}.api")
        self.api.setLevel(logging.INFO)
        self.api.addFilter(self.context_filter)
        
        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            API_LOG, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT
        )
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - [%(system_id)s] [%(session_id)s] [%(endpoint)s] - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        
        # Add handlers
        self.api.addHandler(file_handler)
        
        # Store in loggers dict
        self.loggers['api'] = self.api
    
    def setup_performance_logger(self):
        """Set up performance logger"""
        self.performance = logging.getLogger(f"{self.name}.performance")
        self.performance.setLevel(logging.INFO)
        self.performance.addFilter(self.context_filter)
        
        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            PERFORMANCE_LOG, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT
        )
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - [%(system_id)s] [%(session_id)s] [%(metric)s] - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        
        # Add handlers
        self.performance.addHandler(file_handler)
        
        # Store in loggers dict
        self.loggers['performance'] = self.performance
    
    def set_context(self, **kwargs):
        """Set context attributes for all loggers
        
        Args:
            **kwargs: Context attributes to set
        """
        for key, value in kwargs.items():
            self.context_filter.context[key] = value
    
    def log_signal(self, signal_id, message, level=logging.INFO, **kwargs):
        """Log signal event
        
        Args:
            signal_id: Signal ID
            message: Log message
            level: Log level
            **kwargs: Additional context attributes
        """
        # Set signal context
        temp_context = self.context_filter.context.copy()
        self.context_filter.context['signal_id'] = signal_id
        
        # Add additional context
        for key, value in kwargs.items():
            self.context_filter.context[key] = value
        
        # Log message
        self.signal.log(level, message)
        
        # Restore context
        self.context_filter.context = temp_context
    
    def log_order(self, order_id, message, level=logging.INFO, **kwargs):
        """Log order event
        
        Args:
            order_id: Order ID
            message: Log message
            level: Log level
            **kwargs: Additional context attributes
        """
        # Set order context
        temp_context = self.context_filter.context.copy()
        self.context_filter.context['order_id'] = order_id
        
        # Add additional context
        for key, value in kwargs.items():
            self.context_filter.context[key] = value
        
        # Log message
        self.order.log(level, message)
        
        # Restore context
        self.context_filter.context = temp_context
    
    def log_api(self, endpoint, message, level=logging.INFO, **kwargs):
        """Log API event
        
        Args:
            endpoint: API endpoint
            message: Log message
            level: Log level
            **kwargs: Additional context attributes
        """
        # Set API context
        temp_context = self.context_filter.context.copy()
        self.context_filter.context['endpoint'] = endpoint
        
        # Add additional context
        for key, value in kwargs.items():
            self.context_filter.context[key] = value
        
        # Log message
        self.api.log(level, message)
        
        # Restore context
        self.context_filter.context = temp_context
    
    def log_performance(self, metric, value, unit="ms", level=logging.INFO, **kwargs):
        """Log performance metric
        
        Args:
            metric: Performance metric name
            value: Performance metric value
            unit: Performance metric unit
            level: Log level
            **kwargs: Additional context attributes
        """
        # Set performance context
        temp_context = self.context_filter.context.copy()
        self.context_filter.context['metric'] = metric
        
        # Add additional context
        for key, value_item in kwargs.items():
            self.context_filter.context[key] = value_item
        
        # Log message
        self.performance.log(level, f"{value} {unit}")
        
        # Restore context
        self.context_filter.context = temp_context
    
    def log_error(self, message, exc_info=True, stack_info=True, **kwargs):
        """Log error event
        
        Args:
            message: Error message
            exc_info: Include exception info
            stack_info: Include stack info
            **kwargs: Additional context attributes
        """
        # Add additional context
        temp_context = self.context_filter.context.copy()
        for key, value in kwargs.items():
            self.context_filter.context[key] = value
        
        # Log error
        self.error.error(message, exc_info=exc_info, stack_info=stack_info)
        
        # Restore context
        self.context_filter.context = temp_context


# Example usage
if __name__ == "__main__":
    # Create enhanced logger
    logger = EnhancedLogger("trading_agent_test")
    
    # Log system events
    logger.system.info("System starting up")
    logger.system.info("Loading configuration")
    
    # Log signal events
    logger.log_signal("SIG-123456", "Signal received", symbol="BTCUSDC", type="BUY", strength=0.75)
    logger.log_signal("SIG-123456", "Signal validated", validation_score=0.85)
    logger.log_signal("SIG-123456", "Signal processed", processing_time=15.2, unit="ms")
    
    # Log order events
    logger.log_order("ORD-789012", "Order created", symbol="BTCUSDC", side="BUY", quantity=0.001, price=105000.0)
    logger.log_order("ORD-789012", "Order placed", exchange="MEXC")
    logger.log_order("ORD-789012", "Order executed", execution_time=250.5, unit="ms")
    
    # Log API events
    logger.log_api("get_ticker", "API request sent", symbol="BTCUSDC")
    logger.log_api("get_ticker", "API response received", response_time=120.3, unit="ms")
    
    # Log performance metrics
    logger.log_performance("signal_processing_time", 15.2)
    logger.log_performance("order_execution_time", 250.5)
    logger.log_performance("api_response_time", 120.3)
    
    # Log errors
    try:
        # Simulate error
        raise ValueError("Invalid signal strength")
    except Exception as e:
        logger.log_error("Error processing signal", signal_id="SIG-123456")
    
    logger.system.info("System shutdown")
    
    print(f"Logs written to {LOG_DIR}")
