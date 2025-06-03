#!/usr/bin/env python
"""
Context Manager for LLM Strategic Overseer.

This module provides a context management system for the LLM Strategic Overseer,
tracking market data, trading history, and system status.
"""

import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class ContextManager:
    """
    Context Manager for LLM Strategic Overseer.
    
    This class manages the context for the LLM Strategic Overseer,
    including market data, trading history, and system status.
    """
    
    def __init__(self, config=None):
        """
        Initialize context manager.
        
        Args:
            config: Configuration object (optional)
        """
        self.config = config
        self.context = {}
    
    def initialize_context(self, initial_context: Dict[str, Any]) -> None:
        """
        Initialize context with initial values.
        
        Args:
            initial_context: Initial context values
        """
        self.context = initial_context
        logger.info("Context initialized")
    
    def update_context(self, key: str, value: Any) -> None:
        """
        Update context value.
        
        Args:
            key: Context key
            value: Context value
        """
        self.context[key] = value
        logger.debug(f"Context updated: {key}")
    
    def get_context(self, key: str, default: Any = None) -> Any:
        """
        Get context value.
        
        Args:
            key: Context key
            default: Default value if key not found
            
        Returns:
            Context value
        """
        return self.context.get(key, default)
    
    def get_relevant_context(self, decision_type: str) -> Dict[str, Any]:
        """
        Get relevant context for decision type.
        
        Args:
            decision_type: Type of decision
            
        Returns:
            Relevant context
        """
        # Map decision types to relevant context keys
        relevance_mapping = {
            "market_monitoring": ["market_data", "system_status"],
            "market_trend_analysis": ["market_data", "trading_history"],
            "strategy_adjustment": ["market_data", "trading_history", "performance_metrics"],
            "risk_management": ["trading_history", "performance_metrics", "risk_parameters"],
            "report_generation": ["market_data", "trading_history", "performance_metrics", "system_status"],
            "emergency_response": ["market_data", "trading_history", "performance_metrics", "risk_parameters", "system_status"]
        }
        
        # Get relevant context keys for decision type
        relevant_keys = relevance_mapping.get(decision_type, ["market_data", "system_status"])
        
        # Build relevant context
        relevant_context = {}
        for key in relevant_keys:
            if key in self.context:
                # For list data, only include the most recent items
                if isinstance(self.context[key], list):
                    # Limit to last 10 items for context
                    relevant_context[key] = self.context[key][-10:]
                else:
                    relevant_context[key] = self.context[key]
        
        return relevant_context
