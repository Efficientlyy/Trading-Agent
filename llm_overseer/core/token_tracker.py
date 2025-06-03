#!/usr/bin/env python
"""
Token Tracker for LLM Strategic Overseer.

This module provides token usage tracking and cost estimation
for the LLM Strategic Overseer.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class TokenTracker:
    """
    Token Tracker for LLM Strategic Overseer.
    
    This class tracks token usage and estimates costs for
    the LLM Strategic Overseer.
    """
    
    def __init__(self, config=None):
        """
        Initialize token tracker.
        
        Args:
            config: Configuration object (optional)
        """
        self.config = config
        self.usage = {
            1: {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            2: {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            3: {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        }
        self.cost_per_million_tokens = {
            1: {"input": 0.20, "output": 0.20},  # Tier 1 (lightweight models)
            2: {"input": 0.50, "output": 0.75},  # Tier 2 (mid-tier models)
            3: {"input": 5.00, "output": 10.00}  # Tier 3 (advanced models)
        }
        self.last_reset = datetime.now()
        logger.info("Token tracker initialized")
    
    def track_usage(self, tier: int, input_tokens: int, output_tokens: int) -> None:
        """
        Track token usage.
        
        Args:
            tier: Model tier (1, 2, or 3)
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
        """
        if tier not in self.usage:
            logger.warning(f"Invalid tier: {tier}, defaulting to tier 2")
            tier = 2
        
        self.usage[tier]["input_tokens"] += input_tokens
        self.usage[tier]["output_tokens"] += output_tokens
        self.usage[tier]["total_tokens"] += input_tokens + output_tokens
        
        logger.debug(f"Tracked {input_tokens} input tokens and {output_tokens} output tokens for tier {tier}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get token usage statistics.
        
        Returns:
            Token usage statistics
        """
        total_cost = 0.0
        for tier, usage in self.usage.items():
            tier_cost = (
                (usage["input_tokens"] / 1_000_000) * self.cost_per_million_tokens[tier]["input"] +
                (usage["output_tokens"] / 1_000_000) * self.cost_per_million_tokens[tier]["output"]
            )
            total_cost += tier_cost
        
        days_since_reset = (datetime.now() - self.last_reset).days
        daily_cost = total_cost / max(1, days_since_reset)
        monthly_cost_estimate = daily_cost * 30
        
        return {
            "usage": self.usage,
            "total_tokens": sum(u["total_tokens"] for u in self.usage.values()),
            "cost": {
                "total_cost": total_cost,
                "daily_cost": daily_cost,
                "monthly_estimate": monthly_cost_estimate
            },
            "tracking_period": {
                "start": self.last_reset.isoformat(),
                "days": days_since_reset
            }
        }
    
    def reset_statistics(self) -> None:
        """Reset token usage statistics."""
        for tier in self.usage:
            self.usage[tier] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        
        self.last_reset = datetime.now()
        logger.info("Token usage statistics reset")
    
    def update_cost_rates(self, tier: int, input_cost: float, output_cost: float) -> None:
        """
        Update cost rates for a tier.
        
        Args:
            tier: Model tier (1, 2, or 3)
            input_cost: Cost per million input tokens
            output_cost: Cost per million output tokens
        """
        if tier not in self.cost_per_million_tokens:
            logger.warning(f"Invalid tier: {tier}, ignoring cost rate update")
            return
        
        self.cost_per_million_tokens[tier]["input"] = input_cost
        self.cost_per_million_tokens[tier]["output"] = output_cost
        
        logger.info(f"Updated cost rates for tier {tier}: input=${input_cost}, output=${output_cost} per million tokens")
