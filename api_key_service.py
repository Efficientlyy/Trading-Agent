#!/usr/bin/env python
"""
API Key Service module for retrieving API keys from the LLM-powered Telegram bot service.
This module provides a client for retrieving API keys securely at runtime.
"""
import os
import sys
import json
import time
import logging
import requests
from typing import Dict, Any, Optional, Union

logger = logging.getLogger(__name__)

class APIKeyService:
    """
    API Key Service client for retrieving API keys from the LLM-powered Telegram bot service.
    """
    
    def __init__(self, bot_service_url: str = None, cache_ttl: int = 300):
        """
        Initialize API Key Service client.
        
        Args:
            bot_service_url: URL of the Telegram bot service API
            cache_ttl: Cache time-to-live in seconds (default: 5 minutes)
        """
        self.bot_service_url = bot_service_url or os.environ.get(
            "BOT_SERVICE_URL", "http://trading-agent-llm-overseer:8000"
        )
        self.cache_ttl = cache_ttl
        self.cache = {}
        self.cache_timestamps = {}
        
        logger.info("API Key Service client initialized")
    
    def get_key(self, service: str, key_type: str) -> Optional[str]:
        """
        Get an API key.
        
        Args:
            service: Service name (e.g., 'mexc', 'openrouter')
            key_type: Key type (e.g., 'api_key', 'api_secret')
            
        Returns:
            Key value or None if not found
        """
        cache_key = f"{service}:{key_type}"
        
        # Check cache
        if cache_key in self.cache:
            # Check if cache is still valid
            if time.time() - self.cache_timestamps[cache_key] < self.cache_ttl:
                return self.cache[cache_key]
        
        try:
            # Make API request
            url = f"{self.bot_service_url}/api/keys/{service}/{key_type}"
            response = requests.get(url)
            
            # Check response
            if response.status_code == 200:
                data = response.json()
                value = data.get("value")
                
                # Update cache
                self.cache[cache_key] = value
                self.cache_timestamps[cache_key] = time.time()
                
                return value
            else:
                logger.error(f"Failed to get key {service}.{key_type}: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting key {service}.{key_type}: {e}")
            
            # Try to get from environment variables as fallback
            env_key = f"{service.upper()}_{key_type.upper()}"
            if env_value := os.environ.get(env_key):
                logger.info(f"Using {env_key} from environment variables as fallback")
                return env_value
            
            return None
    
    def get_service_keys(self, service: str) -> Dict[str, str]:
        """
        Get all keys for a service.
        
        Args:
            service: Service name (e.g., 'mexc', 'openrouter')
            
        Returns:
            Dictionary of key types and values
        """
        try:
            # Make API request
            url = f"{self.bot_service_url}/api/keys/{service}"
            response = requests.get(url)
            
            # Check response
            if response.status_code == 200:
                data = response.json()
                keys = data.get("keys", {})
                
                # Update cache
                for key_type, value in keys.items():
                    cache_key = f"{service}:{key_type}"
                    self.cache[cache_key] = value
                    self.cache_timestamps[cache_key] = time.time()
                
                return keys
            else:
                logger.error(f"Failed to get keys for {service}: {response.status_code}")
                return {}
                
        except Exception as e:
            logger.error(f"Error getting keys for {service}: {e}")
            
            # Try to get from environment variables as fallback
            result = {}
            prefix = f"{service.upper()}_"
            for key, value in os.environ.items():
                if key.startswith(prefix):
                    key_type = key[len(prefix):].lower()
                    result[key_type] = value
            
            if result:
                logger.info(f"Using {service} keys from environment variables as fallback")
            
            return result
    
    def clear_cache(self) -> None:
        """Clear the cache."""
        self.cache = {}
        self.cache_timestamps = {}
        logger.info("Cache cleared")
