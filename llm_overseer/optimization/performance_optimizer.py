#!/usr/bin/env python
"""
Performance Optimization Module for LLM Strategic Overseer

This module provides utilities for optimizing performance and reducing costs
in the LLM Strategic Overseer system.
"""

import os
import sys
import time
import logging
import asyncio
import psutil
import threading
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("performance_optimizer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("performance_optimizer")

class PerformanceOptimizer:
    """
    Optimizes performance and reduces costs in the LLM Strategic Overseer system.
    
    This class provides utilities for monitoring resource usage, optimizing API calls,
    and implementing cost-saving strategies.
    """
    
    def __init__(self, event_bus=None):
        """
        Initialize Performance Optimizer.
        
        Args:
            event_bus: Event Bus instance
        """
        self.event_bus = event_bus
        
        # Performance metrics
        self.metrics = {
            "api_calls": {
                "total": 0,
                "by_model": {},
                "by_hour": {}
            },
            "response_times": {
                "total": 0,
                "count": 0,
                "by_model": {}
            },
            "token_usage": {
                "total_input": 0,
                "total_output": 0,
                "by_model": {}
            },
            "estimated_cost": {
                "total": 0.0,
                "by_model": {}
            },
            "memory_usage": {
                "peak": 0,
                "current": 0,
                "history": []
            },
            "cpu_usage": {
                "peak": 0,
                "current": 0,
                "history": []
            }
        }
        
        # Cost optimization settings
        self.optimization_settings = {
            "token_budget": {
                "daily": 100000,  # Maximum tokens per day
                "hourly": 10000   # Maximum tokens per hour
            },
            "model_tiers": {
                "tier1": {  # Most capable, most expensive
                    "models": ["gpt-4-turbo", "claude-3-opus"],
                    "cost_per_1k_input": 0.01,
                    "cost_per_1k_output": 0.03
                },
                "tier2": {  # Balance of capability and cost
                    "models": ["gpt-4", "claude-3-sonnet"],
                    "cost_per_1k_input": 0.005,
                    "cost_per_1k_output": 0.015
                },
                "tier3": {  # Most economical
                    "models": ["gpt-3.5-turbo", "claude-3-haiku"],
                    "cost_per_1k_input": 0.0005,
                    "cost_per_1k_output": 0.0015
                }
            },
            "tier_selection_strategy": "adaptive",  # "fixed", "adaptive", "scheduled"
            "default_tier": "tier2",
            "scheduled_tiers": {
                "00:00-08:00": "tier3",  # Use economical models during low activity
                "08:00-20:00": "tier2",  # Use balanced models during normal hours
                "20:00-24:00": "tier3"   # Use economical models during evening
            },
            "adaptive_thresholds": {
                "high_importance": 0.8,   # Use tier1 for importance >= 0.8
                "medium_importance": 0.5  # Use tier2 for importance >= 0.5, tier3 otherwise
            }
        }
        
        # Start monitoring thread
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        # Register event handlers if event bus is provided
        if self.event_bus:
            self._register_event_handlers()
        
        logger.info("Performance Optimizer initialized")
    
    def __del__(self):
        """Clean up resources when object is destroyed."""
        self.monitoring_active = False
        if hasattr(self, 'monitoring_thread') and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=1.0)
    
    def set_event_bus(self, event_bus):
        """
        Set Event Bus instance.
        
        Args:
            event_bus: Event Bus instance
        """
        self.event_bus = event_bus
        self._register_event_handlers()
        logger.info("Event Bus set for Performance Optimizer")
    
    def _register_event_handlers(self):
        """
        Register event handlers with Event Bus.
        
        Subscribes to relevant events for performance monitoring.
        """
        if self.event_bus:
            # API call events
            self.event_bus.subscribe("llm.api_call_start", self._handle_api_call_start)
            self.event_bus.subscribe("llm.api_call_complete", self._handle_api_call_complete)
            
            # System events
            self.event_bus.subscribe("system.hourly_tick", self._handle_hourly_tick)
            self.event_bus.subscribe("system.daily_tick", self._handle_daily_tick)
            
            logger.info("Performance Optimizer event handlers registered")
        else:
            logger.warning("Event Bus not set, cannot register handlers")
    
    def _monitoring_loop(self):
        """Background thread for monitoring system resources."""
        while self.monitoring_active:
            try:
                # Get current process
                process = psutil.Process(os.getpid())
                
                # Memory usage (in MB)
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)
                
                # CPU usage (percentage)
                cpu_percent = process.cpu_percent(interval=1)
                
                # Update metrics
                self.metrics["memory_usage"]["current"] = memory_mb
                self.metrics["memory_usage"]["peak"] = max(self.metrics["memory_usage"]["peak"], memory_mb)
                self.metrics["memory_usage"]["history"].append({
                    "timestamp": datetime.now().isoformat(),
                    "value": memory_mb
                })
                
                self.metrics["cpu_usage"]["current"] = cpu_percent
                self.metrics["cpu_usage"]["peak"] = max(self.metrics["cpu_usage"]["peak"], cpu_percent)
                self.metrics["cpu_usage"]["history"].append({
                    "timestamp": datetime.now().isoformat(),
                    "value": cpu_percent
                })
                
                # Trim history if too long
                if len(self.metrics["memory_usage"]["history"]) > 60:
                    self.metrics["memory_usage"]["history"] = self.metrics["memory_usage"]["history"][-60:]
                if len(self.metrics["cpu_usage"]["history"]) > 60:
                    self.metrics["cpu_usage"]["history"] = self.metrics["cpu_usage"]["history"][-60:]
                
                # Log if high resource usage
                if memory_mb > 500:  # More than 500 MB
                    logger.warning(f"High memory usage: {memory_mb:.2f} MB")
                if cpu_percent > 80:  # More than 80% CPU
                    logger.warning(f"High CPU usage: {cpu_percent:.2f}%")
                
                # Sleep for 10 seconds
                time.sleep(10)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(30)  # Sleep longer on error
    
    async def _handle_api_call_start(self, topic: str, data: Dict[str, Any]):
        """
        Handle API call start event.
        
        Args:
            topic: Event topic
            data: Event data
        """
        # Record start time for response time calculation
        call_id = data.get("call_id")
        model = data.get("model")
        
        if call_id:
            # Store start time in data for later use
            data["start_time"] = time.time()
            
            # Log API call
            logger.debug(f"API call started: {call_id} using model {model}")
    
    async def _handle_api_call_complete(self, topic: str, data: Dict[str, Any]):
        """
        Handle API call complete event.
        
        Args:
            topic: Event topic
            data: Event data
        """
        call_id = data.get("call_id")
        model = data.get("model")
        input_tokens = data.get("input_tokens", 0)
        output_tokens = data.get("output_tokens", 0)
        start_time = data.get("start_time")
        
        if not all([call_id, model, start_time]):
            logger.warning(f"Incomplete API call data: {data}")
            return
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Update metrics
        self.metrics["api_calls"]["total"] += 1
        
        # Update by model
        if model not in self.metrics["api_calls"]["by_model"]:
            self.metrics["api_calls"]["by_model"][model] = 0
            self.metrics["response_times"]["by_model"][model] = {"total": 0, "count": 0}
            self.metrics["token_usage"]["by_model"][model] = {"input": 0, "output": 0}
            self.metrics["estimated_cost"]["by_model"][model] = 0.0
        
        self.metrics["api_calls"]["by_model"][model] += 1
        
        # Update response times
        self.metrics["response_times"]["total"] += response_time
        self.metrics["response_times"]["count"] += 1
        self.metrics["response_times"]["by_model"][model]["total"] += response_time
        self.metrics["response_times"]["by_model"][model]["count"] += 1
        
        # Update token usage
        self.metrics["token_usage"]["total_input"] += input_tokens
        self.metrics["token_usage"]["total_output"] += output_tokens
        self.metrics["token_usage"]["by_model"][model]["input"] += input_tokens
        self.metrics["token_usage"]["by_model"][model]["output"] += output_tokens
        
        # Update estimated cost
        tier = self._get_model_tier(model)
        if tier:
            cost_per_1k_input = self.optimization_settings["model_tiers"][tier]["cost_per_1k_input"]
            cost_per_1k_output = self.optimization_settings["model_tiers"][tier]["cost_per_1k_output"]
            
            input_cost = (input_tokens / 1000) * cost_per_1k_input
            output_cost = (output_tokens / 1000) * cost_per_1k_output
            total_cost = input_cost + output_cost
            
            self.metrics["estimated_cost"]["total"] += total_cost
            self.metrics["estimated_cost"]["by_model"][model] += total_cost
        
        # Update by hour
        current_hour = datetime.now().strftime("%Y-%m-%d %H:00")
        if current_hour not in self.metrics["api_calls"]["by_hour"]:
            self.metrics["api_calls"]["by_hour"][current_hour] = 0
        self.metrics["api_calls"]["by_hour"][current_hour] += 1
        
        # Log API call completion
        logger.debug(f"API call completed: {call_id} using model {model} in {response_time:.2f}s with {input_tokens} input tokens and {output_tokens} output tokens")
    
    async def _handle_hourly_tick(self, topic: str, data: Dict[str, Any]):
        """
        Handle hourly tick event.
        
        Args:
            topic: Event topic
            data: Event data
        """
        # Check if we're approaching token budget
        current_hour = datetime.now().strftime("%Y-%m-%d %H:00")
        hourly_tokens = 0
        
        if current_hour in self.metrics["api_calls"]["by_hour"]:
            # Estimate tokens used this hour
            hourly_tokens = self.metrics["token_usage"]["total_input"] + self.metrics["token_usage"]["total_output"]
            hourly_budget = self.optimization_settings["token_budget"]["hourly"]
            
            if hourly_tokens > hourly_budget * 0.8:
                logger.warning(f"Approaching hourly token budget: {hourly_tokens}/{hourly_budget} tokens used")
                
                # Publish warning event
                if self.event_bus:
                    await self.event_bus.publish("system.warning", {
                        "warning_type": "token_budget",
                        "component": "performance_optimizer",
                        "message": f"Approaching hourly token budget: {hourly_tokens}/{hourly_budget} tokens used",
                        "usage_percent": (hourly_tokens / hourly_budget) * 100
                    })
    
    async def _handle_daily_tick(self, topic: str, data: Dict[str, Any]):
        """
        Handle daily tick event.
        
        Args:
            topic: Event topic
            data: Event data
        """
        # Generate daily performance report
        daily_report = self.generate_daily_report()
        
        # Publish report
        if self.event_bus:
            await self.event_bus.publish("performance.daily_report", {
                "report_type": "performance",
                "data": daily_report
            })
        
        # Reset hourly metrics
        self.metrics["api_calls"]["by_hour"] = {}
    
    def select_model_tier(self, task_importance: float = 0.5) -> str:
        """
        Select appropriate model tier based on task importance and strategy.
        
        Args:
            task_importance: Importance of the task (0.0 to 1.0)
            
        Returns:
            Selected tier name
        """
        strategy = self.optimization_settings["tier_selection_strategy"]
        
        if strategy == "fixed":
            # Always use the default tier
            return self.optimization_settings["default_tier"]
        
        elif strategy == "scheduled":
            # Select tier based on current time
            current_time = datetime.now().strftime("%H:%M")
            
            for time_range, tier in self.optimization_settings["scheduled_tiers"].items():
                start_time, end_time = time_range.split("-")
                
                if start_time <= current_time < end_time:
                    return tier
            
            # Default to default tier if no match
            return self.optimization_settings["default_tier"]
        
        elif strategy == "adaptive":
            # Select tier based on task importance
            thresholds = self.optimization_settings["adaptive_thresholds"]
            
            if task_importance >= thresholds["high_importance"]:
                return "tier1"
            elif task_importance >= thresholds["medium_importance"]:
                return "tier2"
            else:
                return "tier3"
        
        else:
            # Unknown strategy, use default
            logger.warning(f"Unknown tier selection strategy: {strategy}")
            return self.optimization_settings["default_tier"]
    
    def select_model(self, tier: str = None, task_importance: float = 0.5) -> str:
        """
        Select appropriate model based on tier and task importance.
        
        Args:
            tier: Model tier (if None, will be selected based on task_importance)
            task_importance: Importance of the task (0.0 to 1.0)
            
        Returns:
            Selected model name
        """
        if tier is None:
            tier = self.select_model_tier(task_importance)
        
        # Get models for tier
        models = self.optimization_settings["model_tiers"].get(tier, {}).get("models", [])
        
        if not models:
            # Fallback to default tier
            logger.warning(f"No models found for tier {tier}, falling back to default")
            tier = self.optimization_settings["default_tier"]
            models = self.optimization_settings["model_tiers"].get(tier, {}).get("models", [])
        
        if not models:
            # Still no models, use hardcoded fallback
            logger.error(f"No models found for default tier {tier}, using hardcoded fallback")
            return "gpt-3.5-turbo"
        
        # Return first model in tier (primary model)
        return models[0]
    
    def _get_model_tier(self, model: str) -> Optional[str]:
        """
        Get tier for a given model.
        
        Args:
            model: Model name
            
        Returns:
            Tier name or None if not found
        """
        for tier, tier_data in self.optimization_settings["model_tiers"].items():
            if model in tier_data["models"]:
                return tier
        return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current performance metrics.
        
        Returns:
            Dictionary of metrics
        """
        return self.metrics
    
    def get_optimization_settings(self) -> Dict[str, Any]:
        """
        Get current optimization settings.
        
        Returns:
            Dictionary of settings
        """
        return self.optimization_settings
    
    def update_optimization_settings(self, settings: Dict[str, Any]):
        """
        Update optimization settings.
        
        Args:
            settings: Dictionary of settings to update
        """
        # Update only provided settings
        for key, value in settings.items():
            if key in self.optimization_settings:
                if isinstance(value, dict) and isinstance(self.optimization_settings[key], dict):
                    # Merge dictionaries
                    self.optimization_settings[key].update(value)
                else:
                    # Replace value
                    self.optimization_settings[key] = value
        
        logger.info(f"Optimization settings updated: {settings}")
    
    def generate_daily_report(self) -> Dict[str, Any]:
        """
        Generate daily performance report.
        
        Returns:
            Dictionary with report data
        """
        # Calculate average response time
        avg_response_time = 0
        if self.metrics["response_times"]["count"] > 0:
            avg_response_time = self.metrics["response_times"]["total"] / self.metrics["response_times"]["count"]
        
        # Calculate average tokens per call
        avg_input_tokens = 0
        avg_output_tokens = 0
        if self.metrics["api_calls"]["total"] > 0:
            avg_input_tokens = self.metrics["token_usage"]["total_input"] / self.metrics["api_calls"]["total"]
            avg_output_tokens = self.metrics["token_usage"]["total_output"] / self.metrics["api_calls"]["total"]
        
        # Generate report
        report = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "api_calls": {
                "total": self.metrics["api_calls"]["total"],
                "by_model": self.metrics["api_calls"]["by_model"]
            },
            "response_times": {
                "average": avg_response_time
            },
            "token_usage": {
                "total_input": self.metrics["token_usage"]["total_input"],
                "total_output": self.metrics["token_usage"]["total_output"],
                "average_input": avg_input_tokens,
                "average_output": avg_output_tokens
            },
            "estimated_cost": {
                "total": self.metrics["estimated_cost"]["total"],
                "by_model": self.metrics["estimated_cost"]["by_model"]
            },
            "resource_usage": {
                "memory_peak_mb": self.metrics["memory_usage"]["peak"],
                "cpu_peak_percent": self.metrics["cpu_usage"]["peak"]
            }
        }
        
        return report
    
    def optimize_prompt(self, prompt: str, max_length: int = 4000) -> str:
        """
        Optimize prompt to reduce token usage.
        
        Args:
            prompt: Original prompt
            max_length: Maximum length in characters
            
        Returns:
            Optimized prompt
        """
        # Simple optimization: truncate if too long
        if len(prompt) > max_length:
            logger.warning(f"Prompt truncated from {len(prompt)} to {max_length} characters")
            return prompt[:max_length]
        
        return prompt
    
    def optimize_context_window(self, context: List[Dict[str, Any]], max_tokens: int = 8000) -> List[Dict[str, Any]]:
        """
        Optimize context window to reduce token usage.
        
        Args:
            context: List of context messages
            max_tokens: Maximum tokens to keep
            
        Returns:
            Optimized context window
        """
        # Simple estimation: 4 characters per token
        estimated_tokens = sum(len(str(msg.get("content", ""))) // 4 for msg in context)
        
        if estimated_tokens <= max_tokens:
            return context
        
        # Need to reduce context
        logger.warning(f"Context window optimization needed: {estimated_tokens} tokens > {max_tokens} max")
        
        # Strategy: Keep system message, recent user messages, and important assistant messages
        system_messages = [msg for msg in context if msg.get("role") == "system"]
        user_messages = [msg for msg in context if msg.get("role") == "user"]
        assistant_messages = [msg for msg in context if msg.get("role") == "assistant"]
        
        # Always keep system messages
        optimized_context = system_messages
        
        # Keep most recent user messages
        user_messages.reverse()  # Most recent first
        optimized_context.extend(user_messages[:3])  # Keep last 3 user messages
        
        # Keep most recent assistant messages
        assistant_messages.reverse()  # Most recent first
        optimized_context.extend(assistant_messages[:3])  # Keep last 3 assistant messages
        
        # Sort by original order
        optimized_context.sort(key=lambda msg: context.index(msg) if msg in context else 999)
        
        # Re-estimate tokens
        estimated_tokens = sum(len(str(msg.get("content", ""))) // 4 for msg in optimized_context)
        logger.info(f"Context window optimized: {estimated_tokens} tokens (reduced from original)")
        
        return optimized_context


# For testing
async def test():
    """Test function for PerformanceOptimizer."""
    # Create performance optimizer
    optimizer = PerformanceOptimizer()
    
    # Test model tier selection
    print(f"Model tier for importance 0.9: {optimizer.select_model_tier(0.9)}")
    print(f"Model tier for importance 0.6: {optimizer.select_model_tier(0.6)}")
    print(f"Model tier for importance 0.3: {optimizer.select_model_tier(0.3)}")
    
    # Test model selection
    print(f"Model for importance 0.9: {optimizer.select_model(task_importance=0.9)}")
    print(f"Model for importance 0.6: {optimizer.select_model(task_importance=0.6)}")
    print(f"Model for importance 0.3: {optimizer.select_model(task_importance=0.3)}")
    
    # Test prompt optimization
    long_prompt = "A" * 5000
    optimized_prompt = optimizer.optimize_prompt(long_prompt)
    print(f"Original prompt length: {len(long_prompt)}, Optimized: {len(optimized_prompt)}")
    
    # Test context window optimization
    context = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there! How can I help you today?"},
        {"role": "user", "content": "I need help with my code."},
        {"role": "assistant", "content": "Sure, I'd be happy to help. What's the issue you're facing?"},
        {"role": "user", "content": "It's not working."},
        {"role": "assistant", "content": "I'll need more details to help you effectively. Can you share the code and describe what's not working as expected?"},
        {"role": "user", "content": "Here's my code: " + "print('hello')" * 1000}
    ]
    
    optimized_context = optimizer.optimize_context_window(context)
    print(f"Original context length: {len(context)}, Optimized: {len(optimized_context)}")
    
    # Test metrics
    print("Current metrics:")
    print(optimizer.get_metrics())
    
    # Test daily report
    print("Daily report:")
    print(optimizer.generate_daily_report())
    
    # Clean up
    optimizer.monitoring_active = False

if __name__ == "__main__":
    asyncio.run(test())
