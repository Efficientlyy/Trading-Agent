#!/usr/bin/env python
"""
Enhanced Error Handling and Logging for Trading-Agent System

This module provides comprehensive error handling and logging utilities
for the Trading-Agent system, ensuring robust operation and easier debugging.
"""

import os
import sys
import json
import logging
import traceback
import datetime
from functools import wraps
from typing import Dict, List, Any, Optional, Union, Callable

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Define log levels
LOG_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}

class LoggerFactory:
    """Factory for creating loggers with consistent configuration"""
    
    @staticmethod
    def get_logger(name, log_level='INFO', log_file=None, log_to_console=True):
        """Get logger with specified configuration
        
        Args:
            name: Logger name
            log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Log file path (optional)
            log_to_console: Whether to log to console
            
        Returns:
            logging.Logger: Configured logger
        """
        # Get logger
        logger = logging.getLogger(name)
        
        # Set log level
        logger.setLevel(LOG_LEVELS.get(log_level.upper(), logging.INFO))
        
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Add console handler if requested
        if log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            logger.addHandler(console_handler)
        
        # Add file handler if log file specified
        if log_file:
            # Ensure log directory exists
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            logger.addHandler(file_handler)
        
        return logger

class ErrorHandler:
    """Error handling utilities"""
    
    @staticmethod
    def format_exception(exc_info=None):
        """Format exception for logging
        
        Args:
            exc_info: Exception info (optional)
            
        Returns:
            str: Formatted exception
        """
        if exc_info is None:
            exc_info = sys.exc_info()
        
        if exc_info[0] is None:
            return "No exception information available"
        
        return ''.join(traceback.format_exception(*exc_info))
    
    @staticmethod
    def log_exception(logger, message=None, exc_info=None, level='ERROR'):
        """Log exception
        
        Args:
            logger: Logger to use
            message: Message to log (optional)
            exc_info: Exception info (optional)
            level: Log level (default: ERROR)
            
        Returns:
            None
        """
        if exc_info is None:
            exc_info = sys.exc_info()
        
        if message is None:
            message = "An exception occurred"
        
        formatted_exception = ErrorHandler.format_exception(exc_info)
        
        log_method = getattr(logger, level.lower())
        log_method(f"{message}: {formatted_exception}")
    
    @staticmethod
    def handle_api_error(response, logger, message=None):
        """Handle API error
        
        Args:
            response: API response
            logger: Logger to use
            message: Message to log (optional)
            
        Returns:
            dict: Error information
        """
        if message is None:
            message = "API error"
        
        try:
            error_info = {
                'status_code': response.status_code,
                'reason': response.reason,
                'message': message
            }
            
            try:
                error_info['response'] = response.json()
            except:
                error_info['response'] = response.text
            
            logger.error(f"{message}: {json.dumps(error_info)}")
            
            return error_info
        
        except Exception as e:
            logger.error(f"Error handling API error: {str(e)}")
            return {
                'status_code': 500,
                'reason': 'Internal Error',
                'message': 'Error handling API error'
            }

def retry(max_attempts=3, delay=1, backoff=2, exceptions=(Exception,), logger=None):
    """Retry decorator with exponential backoff
    
    Args:
        max_attempts: Maximum number of attempts
        delay: Initial delay between attempts (seconds)
        backoff: Backoff multiplier
        exceptions: Exceptions to catch
        logger: Logger to use (optional)
        
    Returns:
        function: Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            local_logger = logger or logging.getLogger(func.__module__)
            
            attempt = 1
            current_delay = delay
            
            while attempt <= max_attempts:
                try:
                    return func(*args, **kwargs)
                
                except exceptions as e:
                    if attempt == max_attempts:
                        local_logger.error(f"Failed after {max_attempts} attempts: {str(e)}")
                        raise
                    
                    local_logger.warning(
                        f"Attempt {attempt}/{max_attempts} failed: {str(e)}. "
                        f"Retrying in {current_delay} seconds..."
                    )
                    
                    import time
                    time.sleep(current_delay)
                    
                    attempt += 1
                    current_delay *= backoff
        
        return wrapper
    
    return decorator

def log_execution_time(logger=None, level='DEBUG'):
    """Log execution time decorator
    
    Args:
        logger: Logger to use (optional)
        level: Log level (default: DEBUG)
        
    Returns:
        function: Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            local_logger = logger or logging.getLogger(func.__module__)
            
            start_time = datetime.datetime.now()
            result = func(*args, **kwargs)
            end_time = datetime.datetime.now()
            
            execution_time = (end_time - start_time).total_seconds()
            
            log_method = getattr(local_logger, level.lower())
            log_method(f"{func.__name__} executed in {execution_time:.4f} seconds")
            
            return result
        
        return wrapper
    
    return decorator

def validate_input(validator_func):
    """Input validation decorator
    
    Args:
        validator_func: Validation function
        
    Returns:
        function: Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            validator_func(*args, **kwargs)
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator

class ErrorHandlingMiddleware:
    """Error handling middleware for Flask"""
    
    def __init__(self, app, logger=None):
        """Initialize middleware
        
        Args:
            app: Flask app
            logger: Logger to use (optional)
        """
        self.app = app
        self.logger = logger or logging.getLogger('flask.error')
        
        # Register error handlers
        self._register_error_handlers()
    
    def _register_error_handlers(self):
        """Register error handlers"""
        
        @self.app.errorhandler(Exception)
        def handle_exception(e):
            """Handle uncaught exceptions"""
            self.logger.error(f"Uncaught exception: {str(e)}")
            self.logger.error(ErrorHandler.format_exception())
            
            return {
                'error': 'Internal Server Error',
                'message': str(e)
            }, 500
        
        @self.app.errorhandler(404)
        def handle_not_found(e):
            """Handle 404 errors"""
            self.logger.warning(f"Not found: {self.app.request.path}")
            
            return {
                'error': 'Not Found',
                'message': str(e)
            }, 404
        
        @self.app.errorhandler(400)
        def handle_bad_request(e):
            """Handle 400 errors"""
            self.logger.warning(f"Bad request: {str(e)}")
            
            return {
                'error': 'Bad Request',
                'message': str(e)
            }, 400

class PerformanceMonitor:
    """Performance monitoring utilities"""
    
    def __init__(self, logger=None):
        """Initialize performance monitor
        
        Args:
            logger: Logger to use (optional)
        """
        self.logger = logger or logging.getLogger('performance')
        self.start_times = {}
    
    def start(self, operation_name):
        """Start timing operation
        
        Args:
            operation_name: Operation name
            
        Returns:
            None
        """
        self.start_times[operation_name] = datetime.datetime.now()
    
    def end(self, operation_name, log_level='DEBUG'):
        """End timing operation and log execution time
        
        Args:
            operation_name: Operation name
            log_level: Log level (default: DEBUG)
            
        Returns:
            float: Execution time in seconds
        """
        if operation_name not in self.start_times:
            self.logger.warning(f"No start time found for operation: {operation_name}")
            return None
        
        end_time = datetime.datetime.now()
        start_time = self.start_times.pop(operation_name)
        
        execution_time = (end_time - start_time).total_seconds()
        
        log_method = getattr(self.logger, log_level.lower())
        log_method(f"{operation_name} executed in {execution_time:.4f} seconds")
        
        return execution_time

class WebSocketLogger:
    """WebSocket logging utilities"""
    
    def __init__(self, logger=None):
        """Initialize WebSocket logger
        
        Args:
            logger: Logger to use (optional)
        """
        self.logger = logger or logging.getLogger('websocket')
    
    def log_connection(self, url, status='connected'):
        """Log WebSocket connection
        
        Args:
            url: WebSocket URL
            status: Connection status
            
        Returns:
            None
        """
        self.logger.info(f"WebSocket {status}: {url}")
    
    def log_message(self, message, direction='received', level='DEBUG'):
        """Log WebSocket message
        
        Args:
            message: Message
            direction: Message direction (received or sent)
            level: Log level (default: DEBUG)
            
        Returns:
            None
        """
        log_method = getattr(self.logger, level.lower())
        
        # Truncate message if too long
        message_str = str(message)
        if len(message_str) > 1000:
            message_str = message_str[:1000] + "... [truncated]"
        
        log_method(f"WebSocket message {direction}: {message_str}")
    
    def log_error(self, error, url=None):
        """Log WebSocket error
        
        Args:
            error: Error
            url: WebSocket URL (optional)
            
        Returns:
            None
        """
        if url:
            self.logger.error(f"WebSocket error for {url}: {str(error)}")
        else:
            self.logger.error(f"WebSocket error: {str(error)}")

class APILogger:
    """API logging utilities"""
    
    def __init__(self, logger=None):
        """Initialize API logger
        
        Args:
            logger: Logger to use (optional)
        """
        self.logger = logger or logging.getLogger('api')
    
    def log_request(self, method, url, params=None, headers=None, level='DEBUG'):
        """Log API request
        
        Args:
            method: HTTP method
            url: URL
            params: Request parameters (optional)
            headers: Request headers (optional)
            level: Log level (default: DEBUG)
            
        Returns:
            None
        """
        log_method = getattr(self.logger, level.lower())
        
        message = f"API request: {method} {url}"
        
        if params:
            # Mask sensitive information
            masked_params = self._mask_sensitive_data(params)
            message += f", params: {json.dumps(masked_params)}"
        
        if headers:
            # Mask sensitive headers
            masked_headers = self._mask_sensitive_headers(headers)
            message += f", headers: {json.dumps(masked_headers)}"
        
        log_method(message)
    
    def log_response(self, response, level='DEBUG'):
        """Log API response
        
        Args:
            response: Response
            level: Log level (default: DEBUG)
            
        Returns:
            None
        """
        log_method = getattr(self.logger, level.lower())
        
        message = f"API response: {response.status_code} {response.reason}"
        
        try:
            # Try to parse response as JSON
            response_json = response.json()
            
            # Truncate response if too large
            response_str = json.dumps(response_json)
            if len(response_str) > 1000:
                response_str = response_str[:1000] + "... [truncated]"
            
            message += f", body: {response_str}"
        
        except:
            # If not JSON, log text (truncated if too large)
            response_text = response.text
            if len(response_text) > 1000:
                response_text = response_text[:1000] + "... [truncated]"
            
            message += f", body: {response_text}"
        
        log_method(message)
    
    def log_error(self, error, method=None, url=None):
        """Log API error
        
        Args:
            error: Error
            method: HTTP method (optional)
            url: URL (optional)
            
        Returns:
            None
        """
        if method and url:
            self.logger.error(f"API error for {method} {url}: {str(error)}")
        else:
            self.logger.error(f"API error: {str(error)}")
    
    def _mask_sensitive_data(self, data):
        """Mask sensitive data
        
        Args:
            data: Data to mask
            
        Returns:
            dict: Masked data
        """
        if not isinstance(data, dict):
            return data
        
        masked_data = data.copy()
        
        # List of sensitive fields to mask
        sensitive_fields = [
            'api_key', 'apikey', 'api_secret', 'apisecret', 'secret',
            'password', 'token', 'access_token', 'refresh_token',
            'private_key', 'secret_key', 'signature'
        ]
        
        for field in sensitive_fields:
            if field in masked_data:
                masked_data[field] = '********'
        
        return masked_data
    
    def _mask_sensitive_headers(self, headers):
        """Mask sensitive headers
        
        Args:
            headers: Headers to mask
            
        Returns:
            dict: Masked headers
        """
        if not isinstance(headers, dict):
            return headers
        
        masked_headers = headers.copy()
        
        # List of sensitive headers to mask
        sensitive_headers = [
            'authorization', 'x-api-key', 'api-key', 'x-api-secret',
            'api-secret', 'x-auth-token', 'auth-token'
        ]
        
        for header in sensitive_headers:
            if header.lower() in masked_headers:
                masked_headers[header.lower()] = '********'
        
        return masked_headers

# Example usage
if __name__ == "__main__":
    # Create logger
    logger = LoggerFactory.get_logger(
        'example',
        log_level='DEBUG',
        log_file='example.log'
    )
    
    # Log messages
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")
    
    # Use retry decorator
    @retry(max_attempts=3, delay=1, backoff=2, logger=logger)
    def example_function():
        logger.info("Executing example function")
        raise ValueError("Example error")
    
    try:
        example_function()
    except ValueError:
        logger.info("Caught expected ValueError")
    
    # Use execution time decorator
    @log_execution_time(logger=logger)
    def slow_function():
        logger.info("Executing slow function")
        import time
        time.sleep(1)
        return "Result"
    
    result = slow_function()
    logger.info(f"Function result: {result}")
    
    # Use performance monitor
    performance_monitor = PerformanceMonitor(logger=logger)
    
    performance_monitor.start("example_operation")
    import time
    time.sleep(0.5)
    execution_time = performance_monitor.end("example_operation")
    
    logger.info(f"Operation execution time: {execution_time:.4f} seconds")
    
    # Use API logger
    api_logger = APILogger(logger=logger)
    
    api_logger.log_request(
        "GET",
        "https://api.example.com/data",
        params={"api_key": "secret_key", "limit": 10},
        headers={"Authorization": "Bearer token"}
    )
    
    # Use WebSocket logger
    ws_logger = WebSocketLogger(logger=logger)
    
    ws_logger.log_connection("wss://ws.example.com")
    ws_logger.log_message({"type": "subscribe", "channel": "trades"}, direction="sent")
    ws_logger.log_message({"type": "subscribed", "channel": "trades"}, direction="received")
    ws_logger.log_error("Connection closed", url="wss://ws.example.com")
    
    logger.info("Example completed")
