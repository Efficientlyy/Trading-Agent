#!/usr/bin/env python
"""
Tiered model selection module for LLM Strategic Overseer.

This module implements the tiered model selection strategy for balancing
cost and performance in LLM usage.
"""

import logging
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple

# Import configuration
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config

logger = logging.getLogger(__name__)

class TaskComplexity(Enum):
    """Enumeration of task complexity levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3

class TaskUrgency(Enum):
    """Enumeration of task urgency levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3

class ModelTier(Enum):
    """Enumeration of model tiers."""
    TIER1 = 1  # Lightweight models
    TIER2 = 2  # Mid-tier models
    TIER3 = 3  # Advanced models

class TieredModelSelector:
    """
    Implements the tiered model selection strategy for balancing
    cost and performance in LLM usage.
    """
    
    def __init__(self, config: Config):
        """
        Initialize tiered model selector.
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # Task categories by complexity
        self.task_categories = {
            # Tier 1 (Lightweight) tasks
            "market_data_preprocessing": TaskComplexity.LOW,
            "signal_analysis": TaskComplexity.LOW,
            "basic_pattern_recognition": TaskComplexity.LOW,
            "notification_formatting": TaskComplexity.LOW,
            "status_reporting": TaskComplexity.LOW,
            
            # Tier 2 (Mid-tier) tasks
            "strategy_adjustment": TaskComplexity.MEDIUM,
            "risk_assessment": TaskComplexity.MEDIUM,
            "performance_analysis": TaskComplexity.MEDIUM,
            "complex_pattern_recognition": TaskComplexity.MEDIUM,
            "market_trend_analysis": TaskComplexity.MEDIUM,
            
            # Tier 3 (Advanced) tasks
            "major_strategy_shift": TaskComplexity.HIGH,
            "market_regime_analysis": TaskComplexity.HIGH,
            "critical_risk_management": TaskComplexity.HIGH,
            "system_optimization": TaskComplexity.HIGH,
            "anomaly_detection": TaskComplexity.HIGH
        }
    
    def get_model_tier(self, task_type: str, context_size: Optional[int] = None, 
                      urgency: str = "normal") -> ModelTier:
        """
        Determine the appropriate model tier based on task requirements.
        
        Args:
            task_type: Type of task
            context_size: Size of context in tokens (optional)
            urgency: Urgency level ("low", "normal", "high")
            
        Returns:
            Appropriate model tier
        """
        # Convert urgency string to enum
        urgency_level = TaskUrgency.NORMAL
        if urgency.lower() == "low":
            urgency_level = TaskUrgency.LOW
        elif urgency.lower() == "high":
            urgency_level = TaskUrgency.HIGH
        
        # Get task complexity
        task_complexity = self.task_categories.get(task_type, TaskComplexity.MEDIUM)
        
        # Determine base tier from task complexity
        if task_complexity == TaskComplexity.LOW and urgency_level != TaskUrgency.HIGH:
            base_tier = ModelTier.TIER1
        elif task_complexity == TaskComplexity.MEDIUM or urgency_level == TaskUrgency.HIGH:
            base_tier = ModelTier.TIER2
        else:  # HIGH complexity
            base_tier = ModelTier.TIER3
        
        # Adjust tier based on context size if provided
        if context_size:
            if context_size > 12000:
                # Large context requires more capable models
                return ModelTier.TIER3
            elif context_size > 4000 and base_tier == ModelTier.TIER1:
                # Medium context upgrades from TIER1 to TIER2
                return ModelTier.TIER2
        
        return base_tier
    
    def select_model(self, task_type: str, context_size: Optional[int] = None, 
                    urgency: str = "normal") -> str:
        """
        Select appropriate model based on task requirements and cost efficiency.
        
        Args:
            task_type: Type of task
            context_size: Size of context in tokens (optional)
            urgency: Urgency level ("low", "normal", "high")
            
        Returns:
            Selected model name
        """
        # Get appropriate tier
        tier = self.get_model_tier(task_type, context_size, urgency)
        
        # Map tier to config key
        tier_key = f"tier{tier.value}"
        
        # Get default model for selected tier
        return self.config.get(f"llm.tiers.{tier_key}.default_model")
    
    def get_model_parameters(self, model: str) -> Dict[str, Any]:
        """
        Get parameters for a specific model.
        
        Args:
            model: Model name
            
        Returns:
            Model parameters
        """
        # Determine which tier the model belongs to
        tier_key = "tier1"  # Default
        
        for i in range(1, 4):
            tier = f"tier{i}"
            if model in self.config.get(f"llm.tiers.{tier}.models", []):
                tier_key = tier
                break
        
        # Get parameters for the tier
        return {
            "max_tokens": self.config.get(f"llm.tiers.{tier_key}.max_tokens", 4096),
            "temperature": self.config.get(f"llm.tiers.{tier_key}.temperature", 0.7)
        }
    
    def get_fallback_model(self, model: str) -> str:
        """
        Get fallback model for a given model.
        
        Args:
            model: Original model name
            
        Returns:
            Fallback model name
        """
        # Default fallback strategy: use claude-haiku as it's most reliable
        return "claude-haiku"
