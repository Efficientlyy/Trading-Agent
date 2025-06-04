#!/usr/bin/env python
"""
Fixed OpenRouter Integration for LLM Overseer

This module provides a fixed implementation of the OpenRouter client integration
for the LLM Overseer component, ensuring proper API access and error handling.
"""

import os
import sys
import json
import time
import logging
import requests
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import enhanced logger
from enhanced_logging_fixed import EnhancedLogger

# Initialize enhanced logger
logger = EnhancedLogger("openrouter_client")

class OpenRouterClient:
    """Fixed OpenRouter client implementation"""
    
    def __init__(self, api_key=None):
        """Initialize OpenRouter client
        
        Args:
            api_key: OpenRouter API key (optional)
        """
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        
        if not self.api_key:
            logger.system.warning("OpenRouter API key not found, using mock mode")
            self.mock_mode = True
        else:
            self.mock_mode = False
        
        self.base_url = "https://openrouter.ai/api/v1"
        self.models = {
            "default": "openai/gpt-3.5-turbo",
            "advanced": "openai/gpt-4",
            "fast": "anthropic/claude-instant-v1",
            "balanced": "anthropic/claude-2"
        }
        
        logger.system.info("OpenRouter client initialized")
    
    def chat_completion(self, messages, model="default", temperature=0.7, max_tokens=1000):
        """Get chat completion from OpenRouter
        
        Args:
            messages: List of message dictionaries
            model: Model name or key
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate
            
        Returns:
            dict: Response dictionary
        """
        if self.mock_mode:
            return self._mock_chat_completion(messages, model, temperature, max_tokens)
        
        try:
            # Get actual model name
            model_name = self.models.get(model, model)
            
            # Prepare request
            url = f"{self.base_url}/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            # Log API request
            logger.log_api(
                "openrouter_chat_completion",
                f"Requesting chat completion from {model_name}",
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Send request
            start_time = time.time()
            response = requests.post(url, headers=headers, json=data)
            end_time = time.time()
            
            # Log performance
            logger.log_performance(
                "openrouter_response_time",
                (end_time - start_time) * 1000,
                model=model_name
            )
            
            # Check response
            if response.status_code == 200:
                result = response.json()
                logger.log_api(
                    "openrouter_chat_completion",
                    "Chat completion received",
                    model=model_name,
                    tokens_used=result.get("usage", {}).get("total_tokens", 0)
                )
                return result
            else:
                logger.log_error(
                    f"OpenRouter API error: {response.status_code} - {response.text}",
                    endpoint="openrouter_chat_completion",
                    status_code=response.status_code
                )
                return {
                    "error": {
                        "message": f"API error: {response.status_code}",
                        "type": "api_error",
                        "code": response.status_code
                    }
                }
        except Exception as e:
            logger.log_error(
                f"OpenRouter request error: {str(e)}",
                endpoint="openrouter_chat_completion"
            )
            return {
                "error": {
                    "message": str(e),
                    "type": "request_error"
                }
            }
    
    def _mock_chat_completion(self, messages, model="default", temperature=0.7, max_tokens=1000):
        """Generate mock chat completion for testing
        
        Args:
            messages: List of message dictionaries
            model: Model name or key
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate
            
        Returns:
            dict: Mock response dictionary
        """
        # Get last user message
        last_message = None
        for message in reversed(messages):
            if message.get("role") == "user":
                last_message = message.get("content", "")
                break
        
        # Generate mock response based on last message
        if "market analysis" in last_message.lower():
            content = "Based on the current market conditions, I observe a slight bullish trend with increasing volume. The recent price action shows support at the current level, and momentum indicators suggest potential for upward movement. However, volatility remains high, so caution is advised."
        elif "trading decision" in last_message.lower():
            content = "Given the current market conditions, I recommend a cautious BUY position with a small allocation. Set a stop loss at 2% below entry and take profit at 5% above entry. The signal strength is moderate at 0.65, based on positive momentum and order book imbalance favoring buyers."
        elif "risk assessment" in last_message.lower():
            content = "The current risk level is MODERATE. Market volatility is at 15% annualized, which is slightly above the 30-day average. Liquidity appears adequate with bid-ask spreads within normal ranges. Consider reducing position sizes by 20% compared to your standard allocation."
        else:
            content = "I've analyzed the provided market data. The current conditions suggest a neutral stance with a slight bullish bias. Order book shows minor imbalance favoring buyers, but not enough for a strong signal. Recommend monitoring for clearer patterns before taking action."
        
        # Create mock response
        model_name = self.models.get(model, model)
        response = {
            "id": f"mock-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": sum(len(m.get("content", "").split()) * 1.3 for m in messages),
                "completion_tokens": len(content.split()) * 1.3,
                "total_tokens": sum(len(m.get("content", "").split()) * 1.3 for m in messages) + len(content.split()) * 1.3
            }
        }
        
        # Log mock response
        logger.log_api(
            "openrouter_chat_completion",
            "Mock chat completion generated",
            model=model_name,
            mock=True
        )
        
        # Add slight delay to simulate API call
        time.sleep(0.5)
        
        return response


# Example usage
if __name__ == "__main__":
    # Create OpenRouter client
    client = OpenRouterClient()
    
    # Test chat completion
    messages = [
        {"role": "system", "content": "You are a trading assistant that analyzes market data and provides insights."},
        {"role": "user", "content": "Please provide a market analysis for BTC/USDC based on the following data: price=105000, 24h_change=+2.3%, volume=1.5B, bid-ask spread=0.05%"}
    ]
    
    response = client.chat_completion(messages)
    
    # Print response
    print(json.dumps(response, indent=2))
