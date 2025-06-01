"""
Enhanced API request batching and caching for optimized MEXC client

This module provides optimizations for the MEXC client to improve performance
by implementing request batching and enhanced caching mechanisms.
"""

import time
import logging
from collections import defaultdict
from threading import RLock

# Configure logging
logger = logging.getLogger("performance_optimization")

class RequestBatcher:
    """Batches similar API requests to reduce network overhead"""
    
    def __init__(self, batch_window_ms=50, max_batch_size=10):
        """Initialize request batcher
        
        Args:
            batch_window_ms: Maximum time window in milliseconds to batch requests
            max_batch_size: Maximum number of requests to batch together
        """
        self.batch_window_ms = batch_window_ms
        self.max_batch_size = max_batch_size
        self.batches = defaultdict(list)
        self.batch_timers = {}
        self.lock = RLock()
    
    def add_request(self, endpoint, params, callback):
        """Add a request to the batch queue
        
        Args:
            endpoint: API endpoint
            params: Request parameters
            callback: Function to call with results
            
        Returns:
            bool: True if request was added to batch, False if it should be executed immediately
        """
        batch_key = f"{endpoint}:{self._param_signature(params)}"
        
        with self.lock:
            # If this is a new batch, start the timer
            if batch_key not in self.batch_timers:
                self.batch_timers[batch_key] = time.time() * 1000
            
            # Add request to batch
            self.batches[batch_key].append((params, callback))
            
            # Check if batch should be executed
            current_time = time.time() * 1000
            batch_age = current_time - self.batch_timers[batch_key]
            batch_size = len(self.batches[batch_key])
            
            if batch_age >= self.batch_window_ms or batch_size >= self.max_batch_size:
                return False  # Execute batch now
            
            return True  # Keep batching
    
    def get_batch(self, endpoint, params):
        """Get and clear a batch for execution
        
        Args:
            endpoint: API endpoint
            params: Request parameters
            
        Returns:
            list: List of (params, callback) tuples
        """
        batch_key = f"{endpoint}:{self._param_signature(params)}"
        
        with self.lock:
            batch = self.batches.get(batch_key, [])
            if batch_key in self.batches:
                del self.batches[batch_key]
            if batch_key in self.batch_timers:
                del self.batch_timers[batch_key]
            
            return batch
    
    def _param_signature(self, params):
        """Generate a signature for request parameters
        
        This is used to group similar requests together.
        Excludes parameters that should not be batched (e.g., symbol).
        
        Args:
            params: Request parameters
            
        Returns:
            str: Parameter signature
        """
        if not params:
            return "none"
        
        # Exclude parameters that should not be batched
        batch_params = {k: v for k, v in params.items() 
                       if k not in ['symbol', 'orderId', 'clientOrderId']}
        
        return ":".join(f"{k}={v}" for k, v in sorted(batch_params.items()))


class EnhancedCache:
    """Enhanced caching mechanism with TTL and capacity management"""
    
    def __init__(self, default_ttl_ms=1000, max_items=1000):
        """Initialize enhanced cache
        
        Args:
            default_ttl_ms: Default time-to-live in milliseconds
            max_items: Maximum number of items in cache
        """
        self.default_ttl_ms = default_ttl_ms
        self.max_items = max_items
        self.cache = {}
        self.access_times = {}
        self.expiry_times = {}
        self.lock = RLock()
    
    def get(self, key, default=None):
        """Get item from cache
        
        Args:
            key: Cache key
            default: Default value if key not found or expired
            
        Returns:
            Cached value or default
        """
        with self.lock:
            current_time = time.time() * 1000
            
            # Check if key exists and is not expired
            if key in self.cache and current_time < self.expiry_times.get(key, 0):
                # Update access time
                self.access_times[key] = current_time
                return self.cache[key]
            
            # Remove expired item
            if key in self.cache:
                self._remove_item(key)
            
            return default
    
    def set(self, key, value, ttl_ms=None):
        """Set item in cache
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_ms: Time-to-live in milliseconds (uses default if None)
            
        Returns:
            None
        """
        with self.lock:
            current_time = time.time() * 1000
            
            # Ensure we don't exceed max items
            if len(self.cache) >= self.max_items and key not in self.cache:
                self._evict_item()
            
            # Set item
            self.cache[key] = value
            self.access_times[key] = current_time
            
            # Set expiry time
            ttl = ttl_ms if ttl_ms is not None else self.default_ttl_ms
            self.expiry_times[key] = current_time + ttl
    
    def _remove_item(self, key):
        """Remove item from cache
        
        Args:
            key: Cache key
            
        Returns:
            None
        """
        if key in self.cache:
            del self.cache[key]
        if key in self.access_times:
            del self.access_times[key]
        if key in self.expiry_times:
            del self.expiry_times[key]
    
    def _evict_item(self):
        """Evict least recently used item from cache
        
        Returns:
            None
        """
        if not self.access_times:
            return
        
        # Find least recently used item
        lru_key = min(self.access_times.items(), key=lambda x: x[1])[0]
        self._remove_item(lru_key)
    
    def clear(self):
        """Clear all items from cache
        
        Returns:
            None
        """
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.expiry_times.clear()


class CircuitBreaker:
    """Circuit breaker pattern for API requests
    
    Prevents repeated failures by temporarily disabling endpoints
    that are experiencing issues.
    """
    
    # Circuit states
    CLOSED = 'closed'  # Normal operation
    OPEN = 'open'      # Circuit is open, requests fail fast
    HALF_OPEN = 'half_open'  # Testing if service is back
    
    def __init__(self, failure_threshold=5, recovery_timeout_ms=5000):
        """Initialize circuit breaker
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout_ms: Time in milliseconds before attempting recovery
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout_ms = recovery_timeout_ms
        self.circuits = {}
        self.failure_counts = defaultdict(int)
        self.last_failure_time = {}
        self.lock = RLock()
    
    def pre_request(self, endpoint):
        """Check if request should proceed
        
        Args:
            endpoint: API endpoint
            
        Returns:
            bool: True if request should proceed, False if circuit is open
        """
        with self.lock:
            circuit_state = self.circuits.get(endpoint, self.CLOSED)
            current_time = time.time() * 1000
            
            if circuit_state == self.CLOSED:
                # Normal operation
                return True
            
            elif circuit_state == self.OPEN:
                # Check if recovery timeout has elapsed
                last_failure = self.last_failure_time.get(endpoint, 0)
                if current_time - last_failure > self.recovery_timeout_ms:
                    # Try a test request
                    self.circuits[endpoint] = self.HALF_OPEN
                    logger.info(f"Circuit for {endpoint} changed from OPEN to HALF_OPEN")
                    return True
                return False
            
            elif circuit_state == self.HALF_OPEN:
                # Allow one test request
                return True
            
            return True
    
    def record_success(self, endpoint):
        """Record successful request
        
        Args:
            endpoint: API endpoint
            
        Returns:
            None
        """
        with self.lock:
            circuit_state = self.circuits.get(endpoint, self.CLOSED)
            
            if circuit_state == self.HALF_OPEN:
                # Service is back, close the circuit
                self.circuits[endpoint] = self.CLOSED
                self.failure_counts[endpoint] = 0
                logger.info(f"Circuit for {endpoint} changed from HALF_OPEN to CLOSED")
            
            # Reset failure count on success
            self.failure_counts[endpoint] = 0
    
    def record_failure(self, endpoint):
        """Record failed request
        
        Args:
            endpoint: API endpoint
            
        Returns:
            None
        """
        with self.lock:
            circuit_state = self.circuits.get(endpoint, self.CLOSED)
            current_time = time.time() * 1000
            
            # Update last failure time
            self.last_failure_time[endpoint] = current_time
            
            if circuit_state == self.HALF_OPEN:
                # Service is still down, reopen the circuit
                self.circuits[endpoint] = self.OPEN
                logger.warning(f"Circuit for {endpoint} changed from HALF_OPEN to OPEN")
                return
            
            if circuit_state == self.CLOSED:
                # Increment failure count
                self.failure_counts[endpoint] += 1
                
                # Check if threshold reached
                if self.failure_counts[endpoint] >= self.failure_threshold:
                    # Open the circuit
                    self.circuits[endpoint] = self.OPEN
                    logger.warning(f"Circuit for {endpoint} changed from CLOSED to OPEN after {self.failure_counts[endpoint]} failures")
