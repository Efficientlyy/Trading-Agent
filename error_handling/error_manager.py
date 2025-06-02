#!/usr/bin/env python
"""
Error Manager for Trading-Agent System

This module provides comprehensive error handling and logging capabilities
for the Trading-Agent system, ensuring robust operation and detailed diagnostics.
"""

import os
import sys
import json
import time
import logging
import traceback
import threading
from enum import Enum
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("error_manager")

class ErrorSeverity(Enum):
    """Error severity levels"""
    INFO = 0
    WARNING = 1
    ERROR = 2
    CRITICAL = 3
    FATAL = 4

class ErrorCategory(Enum):
    """Error categories"""
    NETWORK = "network"
    API = "api"
    DATA = "data"
    EXECUTION = "execution"
    SIGNAL = "signal"
    PATTERN = "pattern"
    RISK = "risk"
    SYSTEM = "system"
    VISUALIZATION = "visualization"
    UNKNOWN = "unknown"

class ErrorManager:
    """Error manager for Trading-Agent system"""
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0, 
                exponential_backoff: bool = True, log_to_file: bool = True,
                error_notification_callback: Optional[Callable] = None):
        """Initialize error manager
        
        Args:
            max_retries: Maximum number of retries for recoverable errors
            retry_delay: Initial delay between retries in seconds
            exponential_backoff: Whether to use exponential backoff for retries
            log_to_file: Whether to log errors to file
            error_notification_callback: Callback function for error notifications
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.exponential_backoff = exponential_backoff
        self.log_to_file = log_to_file
        self.error_notification_callback = error_notification_callback
        
        # Initialize error counters
        self.error_counts = {category.value: 0 for category in ErrorCategory}
        self.error_history = []
        
        # Initialize error handlers
        self.error_handlers = {}
        
        # Register default error handlers
        self._register_default_handlers()
        
        logger.info(f"Initialized ErrorManager with max_retries={max_retries}, retry_delay={retry_delay}s")
    
    def _register_default_handlers(self):
        """Register default error handlers"""
        # Network errors
        self.register_handler(
            ErrorCategory.NETWORK,
            lambda error, context: self._handle_network_error(error, context)
        )
        
        # API errors
        self.register_handler(
            ErrorCategory.API,
            lambda error, context: self._handle_api_error(error, context)
        )
        
        # Data errors
        self.register_handler(
            ErrorCategory.DATA,
            lambda error, context: self._handle_data_error(error, context)
        )
        
        # Execution errors
        self.register_handler(
            ErrorCategory.EXECUTION,
            lambda error, context: self._handle_execution_error(error, context)
        )
        
        # Signal errors
        self.register_handler(
            ErrorCategory.SIGNAL,
            lambda error, context: self._handle_signal_error(error, context)
        )
        
        # Pattern errors
        self.register_handler(
            ErrorCategory.PATTERN,
            lambda error, context: self._handle_pattern_error(error, context)
        )
        
        # Risk errors
        self.register_handler(
            ErrorCategory.RISK,
            lambda error, context: self._handle_risk_error(error, context)
        )
        
        # System errors
        self.register_handler(
            ErrorCategory.SYSTEM,
            lambda error, context: self._handle_system_error(error, context)
        )
        
        # Visualization errors
        self.register_handler(
            ErrorCategory.VISUALIZATION,
            lambda error, context: self._handle_visualization_error(error, context)
        )
        
        # Unknown errors
        self.register_handler(
            ErrorCategory.UNKNOWN,
            lambda error, context: self._handle_unknown_error(error, context)
        )
    
    def register_handler(self, category: ErrorCategory, handler: Callable):
        """Register error handler
        
        Args:
            category: Error category
            handler: Error handler function
        """
        self.error_handlers[category.value] = handler
        logger.info(f"Registered error handler for {category.value}")
    
    def handle_error(self, error: Exception, category: ErrorCategory = None, 
                    severity: ErrorSeverity = None, context: Dict = None, 
                    retry_function: Callable = None, retry_args: Tuple = None, 
                    retry_kwargs: Dict = None) -> Optional[Any]:
        """Handle error
        
        Args:
            error: Exception object
            category: Error category (default: auto-detect)
            severity: Error severity (default: auto-detect)
            context: Error context
            retry_function: Function to retry on recoverable errors
            retry_args: Arguments for retry function
            retry_kwargs: Keyword arguments for retry function
            
        Returns:
            Result of retry function if successful, None otherwise
        """
        # Auto-detect category if not provided
        if category is None:
            category = self._detect_category(error)
        
        # Auto-detect severity if not provided
        if severity is None:
            severity = self._detect_severity(error, category)
        
        # Initialize context if not provided
        if context is None:
            context = {}
        
        # Add error details to context
        context["error_type"] = type(error).__name__
        context["error_message"] = str(error)
        context["error_traceback"] = traceback.format_exc()
        context["timestamp"] = datetime.now().isoformat()
        context["category"] = category.value
        context["severity"] = severity.value
        
        # Log error
        self._log_error(error, category, severity, context)
        
        # Update error counters
        self.error_counts[category.value] += 1
        
        # Add to error history
        self.error_history.append({
            "timestamp": context["timestamp"],
            "category": category.value,
            "severity": severity.value,
            "error_type": context["error_type"],
            "error_message": context["error_message"]
        })
        
        # Limit error history to 1000 entries
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-1000:]
        
        # Send error notification
        if self.error_notification_callback is not None:
            try:
                self.error_notification_callback(error, category, severity, context)
            except Exception as e:
                logger.error(f"Error in notification callback: {e}")
        
        # Handle error based on category
        if category.value in self.error_handlers:
            try:
                self.error_handlers[category.value](error, context)
            except Exception as e:
                logger.error(f"Error in error handler for {category.value}: {e}")
        
        # Retry function if provided and error is recoverable
        if retry_function is not None and self._is_recoverable(error, category, severity):
            return self._retry_function(retry_function, retry_args or (), retry_kwargs or {}, error, category, context)
        
        return None
    
    def _detect_category(self, error: Exception) -> ErrorCategory:
        """Detect error category
        
        Args:
            error: Exception object
            
        Returns:
            Error category
        """
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        # Network errors
        if any(keyword in error_type.lower() for keyword in ["connection", "timeout", "socket", "http"]) or \
           any(keyword in error_message for keyword in ["connection", "timeout", "network", "unreachable"]):
            return ErrorCategory.NETWORK
        
        # API errors
        if any(keyword in error_type.lower() for keyword in ["api", "request", "response", "http"]) or \
           any(keyword in error_message for keyword in ["api", "request", "response", "status code", "endpoint"]):
            return ErrorCategory.API
        
        # Data errors
        if any(keyword in error_type.lower() for keyword in ["data", "value", "type", "key", "index", "attribute"]) or \
           any(keyword in error_message for keyword in ["data", "value", "type", "key", "index", "attribute", "not found"]):
            return ErrorCategory.DATA
        
        # Execution errors
        if any(keyword in error_type.lower() for keyword in ["execution", "order", "trade", "position"]) or \
           any(keyword in error_message for keyword in ["execution", "order", "trade", "position", "quantity", "price"]):
            return ErrorCategory.EXECUTION
        
        # Signal errors
        if any(keyword in error_type.lower() for keyword in ["signal"]) or \
           any(keyword in error_message for keyword in ["signal", "indicator", "strategy"]):
            return ErrorCategory.SIGNAL
        
        # Pattern errors
        if any(keyword in error_type.lower() for keyword in ["pattern"]) or \
           any(keyword in error_message for keyword in ["pattern", "recognition", "detection"]):
            return ErrorCategory.PATTERN
        
        # Risk errors
        if any(keyword in error_type.lower() for keyword in ["risk"]) or \
           any(keyword in error_message for keyword in ["risk", "limit", "exposure", "margin"]):
            return ErrorCategory.RISK
        
        # System errors
        if any(keyword in error_type.lower() for keyword in ["system", "os", "io", "file", "memory", "process"]) or \
           any(keyword in error_message for keyword in ["system", "os", "io", "file", "memory", "process"]):
            return ErrorCategory.SYSTEM
        
        # Visualization errors
        if any(keyword in error_type.lower() for keyword in ["visualization", "chart", "plot", "figure"]) or \
           any(keyword in error_message for keyword in ["visualization", "chart", "plot", "figure", "display"]):
            return ErrorCategory.VISUALIZATION
        
        # Unknown errors
        return ErrorCategory.UNKNOWN
    
    def _detect_severity(self, error: Exception, category: ErrorCategory) -> ErrorSeverity:
        """Detect error severity
        
        Args:
            error: Exception object
            category: Error category
            
        Returns:
            Error severity
        """
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        # Fatal errors
        if any(keyword in error_type.lower() for keyword in ["fatal", "critical", "system", "memory", "segmentation"]) or \
           any(keyword in error_message for keyword in ["fatal", "critical", "crash", "abort", "killed"]):
            return ErrorSeverity.FATAL
        
        # Critical errors
        if category in [ErrorCategory.RISK, ErrorCategory.EXECUTION] or \
           any(keyword in error_message for keyword in ["critical", "severe", "major", "important"]):
            return ErrorSeverity.CRITICAL
        
        # Error level
        if category in [ErrorCategory.API, ErrorCategory.DATA, ErrorCategory.NETWORK]:
            return ErrorSeverity.ERROR
        
        # Warning level
        if category in [ErrorCategory.SIGNAL, ErrorCategory.PATTERN, ErrorCategory.VISUALIZATION]:
            return ErrorSeverity.WARNING
        
        # Default to ERROR
        return ErrorSeverity.ERROR
    
    def _is_recoverable(self, error: Exception, category: ErrorCategory, severity: ErrorSeverity) -> bool:
        """Check if error is recoverable
        
        Args:
            error: Exception object
            category: Error category
            severity: Error severity
            
        Returns:
            True if error is recoverable, False otherwise
        """
        # Fatal errors are not recoverable
        if severity == ErrorSeverity.FATAL:
            return False
        
        # Critical errors are generally not recoverable
        if severity == ErrorSeverity.CRITICAL:
            return False
        
        # Network errors are generally recoverable
        if category == ErrorCategory.NETWORK:
            return True
        
        # API errors are generally recoverable
        if category == ErrorCategory.API:
            return True
        
        # Data errors may be recoverable
        if category == ErrorCategory.DATA:
            # Check if error message indicates a recoverable error
            error_message = str(error).lower()
            if any(keyword in error_message for keyword in ["timeout", "temporary", "retry", "again"]):
                return True
            return False
        
        # Execution errors may be recoverable
        if category == ErrorCategory.EXECUTION:
            # Check if error message indicates a recoverable error
            error_message = str(error).lower()
            if any(keyword in error_message for keyword in ["timeout", "temporary", "retry", "again", "rate limit"]):
                return True
            return False
        
        # Other errors are generally not recoverable
        return False
    
    def _retry_function(self, func: Callable, args: Tuple, kwargs: Dict, 
                       error: Exception, category: ErrorCategory, context: Dict) -> Optional[Any]:
        """Retry function with exponential backoff
        
        Args:
            func: Function to retry
            args: Function arguments
            kwargs: Function keyword arguments
            error: Original error
            category: Error category
            context: Error context
            
        Returns:
            Result of function if successful, None otherwise
        """
        for attempt in range(1, self.max_retries + 1):
            # Calculate delay
            if self.exponential_backoff:
                delay = self.retry_delay * (2 ** (attempt - 1))
            else:
                delay = self.retry_delay
            
            # Log retry attempt
            logger.info(f"Retrying in {delay:.1f}s (attempt {attempt}/{self.max_retries})")
            
            # Wait before retry
            time.sleep(delay)
            
            try:
                # Retry function
                result = func(*args, **kwargs)
                
                # Log success
                logger.info(f"Retry successful (attempt {attempt}/{self.max_retries})")
                
                return result
            
            except Exception as e:
                # Log retry failure
                logger.error(f"Retry failed (attempt {attempt}/{self.max_retries}): {e}")
                
                # Update error
                error = e
                context["error_type"] = type(error).__name__
                context["error_message"] = str(error)
                context["error_traceback"] = traceback.format_exc()
                context["retry_attempt"] = attempt
        
        # Log max retries reached
        logger.error(f"Max retries reached, giving up")
        
        return None
    
    def _log_error(self, error: Exception, category: ErrorCategory, severity: ErrorSeverity, context: Dict):
        """Log error
        
        Args:
            error: Exception object
            category: Error category
            severity: Error severity
            context: Error context
        """
        # Determine log level based on severity
        if severity == ErrorSeverity.FATAL:
            log_level = logging.CRITICAL
        elif severity == ErrorSeverity.CRITICAL:
            log_level = logging.CRITICAL
        elif severity == ErrorSeverity.ERROR:
            log_level = logging.ERROR
        elif severity == ErrorSeverity.WARNING:
            log_level = logging.WARNING
        else:
            log_level = logging.INFO
        
        # Log error
        logger.log(log_level, f"{category.value.upper()} {severity.name}: {context['error_type']}: {context['error_message']}")
        
        # Log traceback for ERROR and above
        if log_level >= logging.ERROR:
            for line in context["error_traceback"].split("\n"):
                if line:
                    logger.log(log_level, f"  {line}")
        
        # Log to file if enabled
        if self.log_to_file:
            try:
                # Create error log directory if not exists
                os.makedirs("error_logs", exist_ok=True)
                
                # Generate filename
                filename = f"error_logs/{category.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                
                # Write error details to file
                with open(filename, "w") as f:
                    json.dump(context, f, indent=2)
                
                logger.info(f"Error details saved to {filename}")
            
            except Exception as e:
                logger.error(f"Error saving error details to file: {e}")
    
    def _handle_network_error(self, error: Exception, context: Dict):
        """Handle network error
        
        Args:
            error: Exception object
            context: Error context
        """
        logger.info(f"Handling network error: {context['error_message']}")
        
        # Network errors are generally transient, so we just log them
        pass
    
    def _handle_api_error(self, error: Exception, context: Dict):
        """Handle API error
        
        Args:
            error: Exception object
            context: Error context
        """
        logger.info(f"Handling API error: {context['error_message']}")
        
        # Check if error is due to rate limiting
        error_message = context["error_message"].lower()
        if any(keyword in error_message for keyword in ["rate limit", "too many requests", "429"]):
            logger.warning("Rate limit exceeded, consider reducing request frequency")
    
    def _handle_data_error(self, error: Exception, context: Dict):
        """Handle data error
        
        Args:
            error: Exception object
            context: Error context
        """
        logger.info(f"Handling data error: {context['error_message']}")
        
        # Data errors may indicate issues with data sources or processing
        pass
    
    def _handle_execution_error(self, error: Exception, context: Dict):
        """Handle execution error
        
        Args:
            error: Exception object
            context: Error context
        """
        logger.info(f"Handling execution error: {context['error_message']}")
        
        # Execution errors are critical and may require manual intervention
        pass
    
    def _handle_signal_error(self, error: Exception, context: Dict):
        """Handle signal error
        
        Args:
            error: Exception object
            context: Error context
        """
        logger.info(f"Handling signal error: {context['error_message']}")
        
        # Signal errors may indicate issues with signal generation
        pass
    
    def _handle_pattern_error(self, error: Exception, context: Dict):
        """Handle pattern error
        
        Args:
            error: Exception object
            context: Error context
        """
        logger.info(f"Handling pattern error: {context['error_message']}")
        
        # Pattern errors may indicate issues with pattern recognition
        pass
    
    def _handle_risk_error(self, error: Exception, context: Dict):
        """Handle risk error
        
        Args:
            error: Exception object
            context: Error context
        """
        logger.info(f"Handling risk error: {context['error_message']}")
        
        # Risk errors are critical and may require manual intervention
        pass
    
    def _handle_system_error(self, error: Exception, context: Dict):
        """Handle system error
        
        Args:
            error: Exception object
            context: Error context
        """
        logger.info(f"Handling system error: {context['error_message']}")
        
        # System errors may indicate issues with the operating system or environment
        pass
    
    def _handle_visualization_error(self, error: Exception, context: Dict):
        """Handle visualization error
        
        Args:
            error: Exception object
            context: Error context
        """
        logger.info(f"Handling visualization error: {context['error_message']}")
        
        # Visualization errors may indicate issues with chart rendering
        pass
    
    def _handle_unknown_error(self, error: Exception, context: Dict):
        """Handle unknown error
        
        Args:
            error: Exception object
            context: Error context
        """
        logger.info(f"Handling unknown error: {context['error_message']}")
        
        # Unknown errors require investigation
        pass
    
    def get_error_summary(self) -> Dict:
        """Get error summary
        
        Returns:
            Dictionary with error summary
        """
        return {
            "total_errors": sum(self.error_counts.values()),
            "error_counts": self.error_counts,
            "recent_errors": self.error_history[-10:] if self.error_history else []
        }
    
    def clear_error_history(self):
        """Clear error history"""
        self.error_history = []
        logger.info("Error history cleared")
    
    def reset_error_counts(self):
        """Reset error counters"""
        self.error_counts = {category.value: 0 for category in ErrorCategory}
        logger.info("Error counters reset")

# Global error manager instance
error_manager = ErrorManager()

def handle_error(error: Exception, category: ErrorCategory = None, 
                severity: ErrorSeverity = None, context: Dict = None, 
                retry_function: Callable = None, retry_args: Tuple = None, 
                retry_kwargs: Dict = None) -> Optional[Any]:
    """Global error handler function
    
    Args:
        error: Exception object
        category: Error category (default: auto-detect)
        severity: Error severity (default: auto-detect)
        context: Error context
        retry_function: Function to retry on recoverable errors
        retry_args: Arguments for retry function
        retry_kwargs: Keyword arguments for retry function
        
    Returns:
        Result of retry function if successful, None otherwise
    """
    return error_manager.handle_error(
        error, category, severity, context, 
        retry_function, retry_args, retry_kwargs
    )

def safe_execute(func: Callable, *args, **kwargs):
    """Execute function safely with error handling
    
    Args:
        func: Function to execute
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Result of function if successful, None otherwise
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        return handle_error(e, retry_function=func, retry_args=args, retry_kwargs=kwargs)

if __name__ == "__main__":
    # Example usage
    try:
        # Simulate network error
        raise ConnectionError("Failed to connect to API endpoint")
    except Exception as e:
        handle_error(e)
    
    # Example with retry
    def example_function(x, y):
        if x < 0:
            raise ValueError("x must be non-negative")
        return x + y
    
    result = safe_execute(example_function, -1, 5)
    print(f"Result: {result}")  # Should be None due to ValueError
    
    result = safe_execute(example_function, 10, 5)
    print(f"Result: {result}")  # Should be 15
    
    # Get error summary
    summary = error_manager.get_error_summary()
    print(f"Error summary: {summary}")
