#!/usr/bin/env python
"""
Tiered LLM Manager for LLM Strategic Overseer.

This module provides a tiered approach to LLM model selection,
optimizing for cost and performance based on task requirements.
"""

import os
import logging
import asyncio
import json
from typing import Dict, Any, Optional, List, Union
import openrouter
import requests
from requests.exceptions import RequestException, Timeout

logger = logging.getLogger(__name__)

class TieredLLMManager:
    """
    Tiered LLM Manager for LLM Strategic Overseer.
    
    This class manages multiple LLM models across different tiers,
    optimizing for cost and performance.
    """
    
    def __init__(self, config):
        """
        Initialize tiered LLM manager.
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # Check for API key - support direct key or environment variable
        api_key = self.config.get("llm.api_key")
        if not api_key:
            # Try to get from environment
            env_var_name = self.config.get("llm.api_key_env", "OPENROUTER_API_KEY")
            api_key = os.environ.get(env_var_name)
            
        if not api_key:
            logger.warning("OpenRouter API key not found in configuration or environment")
            # For testing purposes, use a mock key
            api_key = "mock_api_key_for_testing"
            logger.warning("Using mock API key for testing")
        
        self.api_key = api_key
        
        # Initialize model configuration
        self.models = {
            1: self.config.get("llm.models.tier_1", "openai/gpt-3.5-turbo"),
            2: self.config.get("llm.models.tier_2", "anthropic/claude-3-sonnet"),
            3: self.config.get("llm.models.tier_3", "anthropic/claude-3-opus")
        }
        
        # Initialize OpenRouter client
        self.client = openrouter.OpenRouter(
            api_key=self.api_key,
            http_client=requests.Session()
        )
        
        # Initialize fallback configuration
        self.fallback_enabled = self.config.get("llm.fallback_enabled", True)
        self.max_retries = self.config.get("llm.max_retries", 3)
        self.retry_delay = self.config.get("llm.retry_delay", 1)  # seconds
        
        # Initialize cache for cost optimization
        self.cache_enabled = self.config.get("llm.cache_enabled", True)
        self.response_cache = {}
        
        logger.info("Tiered LLM Manager initialized")
    
    async def generate_response(self, tier: int, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate response from LLM.
        
        Args:
            tier: Model tier (1, 2, or 3)
            prompt: User prompt
            system_prompt: System prompt
            
        Returns:
            Response from LLM
        """
        # Validate tier
        if tier not in self.models:
            logger.warning(f"Invalid tier: {tier}, defaulting to tier 2")
            tier = 2
        
        model = self.models[tier]
        
        # Check cache if enabled
        if self.cache_enabled:
            cache_key = f"{model}:{system_prompt or ''}:{prompt}"
            if cache_key in self.response_cache:
                logger.info(f"Cache hit for tier {tier} model: {model}")
                return self.response_cache[cache_key]
        
        # Prepare messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Set model parameters based on tier
        max_tokens = self.config.get(f"llm.tiers.tier{tier}.max_tokens", 4096)
        temperature = self.config.get(f"llm.tiers.tier{tier}.temperature", 0.7)
        
        # Try to generate response with retries and fallbacks
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Generating response using tier {tier} model: {model} (attempt {attempt+1})")
                
                # Check if we're using a mock key for testing
                if self.api_key == "mock_api_key_for_testing":
                    # Simulate response for testing
                    logger.info("Using simulated response for testing")
                    response = self._simulate_response(tier, prompt, system_prompt)
                else:
                    # Make actual API call to OpenRouter
                    response = await self._call_openrouter_api(model, messages, max_tokens, temperature)
                
                # Cache response if enabled
                if self.cache_enabled:
                    self.response_cache[cache_key] = response
                
                return response
                
            except Exception as e:
                logger.error(f"Error generating response with {model}: {e}")
                
                # If this is the last attempt or fallback is disabled, raise the error
                if attempt == self.max_retries - 1 or not self.fallback_enabled:
                    return {
                        "success": False,
                        "error": str(e),
                        "model": model,
                        "tier": tier
                    }
                
                # Otherwise, wait before retrying
                logger.info(f"Retrying in {self.retry_delay} seconds...")
                await asyncio.sleep(self.retry_delay)
                
                # If this is tier 3, try tier 2 as fallback
                # If this is tier 2, try tier 1 as fallback
                if tier > 1 and attempt == self.max_retries - 2:
                    fallback_tier = tier - 1
                    fallback_model = self.models[fallback_tier]
                    logger.info(f"Falling back to tier {fallback_tier} model: {fallback_model}")
                    model = fallback_model
                    tier = fallback_tier
                    max_tokens = self.config.get(f"llm.tiers.tier{tier}.max_tokens", 4096)
                    temperature = self.config.get(f"llm.tiers.tier{tier}.temperature", 0.7)
    
    async def _call_openrouter_api(self, model: str, messages: List[Dict[str, str]], 
                                  max_tokens: int, temperature: float) -> Dict[str, Any]:
        """
        Call OpenRouter API.
        
        Args:
            model: Model name
            messages: List of messages
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            
        Returns:
            Response from OpenRouter API
        """
        try:
            # Make API call
            completion = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Extract response
            response_text = completion.choices[0].message.content
            
            # Calculate token usage
            input_tokens = completion.usage.prompt_tokens
            output_tokens = completion.usage.completion_tokens
            total_tokens = completion.usage.total_tokens
            
            return {
                "success": True,
                "response": response_text,
                "model": model,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens
                }
            }
            
        except Exception as e:
            logger.error(f"OpenRouter API error: {e}")
            raise
    
    def _simulate_response(self, tier: int, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Simulate response for testing.
        
        Args:
            tier: Model tier
            prompt: User prompt
            system_prompt: System prompt
            
        Returns:
            Simulated response
        """
        model = self.models[tier]
        
        # Simulate token counts
        input_tokens = len(prompt.split())
        if system_prompt:
            input_tokens += len(system_prompt.split())
        
        # Generate simulated response based on prompt content
        if "market" in prompt.lower():
            response = (
                f"Based on the current market conditions for BTC/USDC, I observe a bullish trend "
                f"with increasing volume. The order book shows more buy pressure than sell pressure. "
                f"I recommend maintaining current positions with a slight increase in exposure if the "
                f"uptrend continues. Set stop losses at 2% below current price."
            )
        elif "risk" in prompt.lower():
            response = (
                f"Risk assessment: Current market volatility is moderate. Position sizes should be "
                f"kept at 5-10% of portfolio. Implement trailing stops to protect profits. "
                f"Diversify across multiple trading pairs to reduce concentrated risk."
            )
        elif "performance" in prompt.lower():
            response = (
                f"Performance analysis: The trading strategy has yielded 2.3% return over the past 24 hours, "
                f"outperforming the market by 0.8%. Win rate is 62% with average profit/loss ratio of 1.5. "
                f"Recommend continuing with current parameters while monitoring slippage on larger orders."
            )
        else:
            response = (
                f"This is a simulated response from the {model} model (tier {tier}). "
                f"In a production environment, this would be a detailed analysis and recommendation "
                f"based on the specific query and available market data."
            )
        
        output_tokens = len(response.split())
        
        return {
            "success": True,
            "response": response,
            "model": model,
            "tier": tier,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            }
        }
    
    def get_model_for_tier(self, tier: int) -> str:
        """
        Get model for tier.
        
        Args:
            tier: Model tier (1, 2, or 3)
            
        Returns:
            Model name
        """
        if tier not in self.models:
            logger.warning(f"Invalid tier: {tier}, defaulting to tier 2")
            tier = 2
        
        return self.models[tier]
    
    def update_model_for_tier(self, tier: int, model: str) -> None:
        """
        Update model for tier.
        
        Args:
            tier: Model tier (1, 2, or 3)
            model: Model name
        """
        if tier not in self.models:
            logger.warning(f"Invalid tier: {tier}, ignoring model update")
            return
        
        self.models[tier] = model
        logger.info(f"Updated tier {tier} model to {model}")
    
    def clear_cache(self) -> None:
        """Clear response cache."""
        self.response_cache = {}
        logger.info("Response cache cleared")
    
    def get_usage_statistics(self) -> Dict[str, Any]:
        """
        Get usage statistics.
        
        Returns:
            Usage statistics
        """
        # In a real implementation, this would track actual usage
        # For now, we'll return a placeholder
        return {
            "total_requests": len(self.response_cache),
            "cache_hits": 0,
            "tier_usage": {
                1: {"requests": 0, "tokens": 0, "estimated_cost": 0},
                2: {"requests": 0, "tokens": 0, "estimated_cost": 0},
                3: {"requests": 0, "tokens": 0, "estimated_cost": 0}
            }
        }
