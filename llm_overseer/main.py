#!/usr/bin/env python
"""
Main module for LLM Strategic Overseer.

This module provides the main LLM Overseer class that coordinates
strategic decision making, context management, and token usage tracking.
"""

import os
import sys
import logging
import asyncio
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

# Configure logging
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs'), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', 'llm_overseer.log'))
    ]
)

logger = logging.getLogger(__name__)

# Import components
from .config.config import Config
from .core.llm_manager import TieredLLMManager
from .core.context_manager import ContextManager
from .core.token_tracker import TokenTracker


class LLMOverseer:
    """
    LLM Strategic Overseer for trading system.
    
    This class coordinates strategic decision making, context management,
    and token usage tracking for the trading system.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize LLM Overseer.
        
        Args:
            config_path: Path to configuration file
        """
        # Initialize configuration
        self.config = Config(config_path)
        
        # Initialize components
        self.llm_manager = TieredLLMManager(self.config)
        self.context_manager = ContextManager(self.config)
        self.token_tracker = TokenTracker(self.config)
        
        # Initialize context with default values
        self.context_manager.initialize_context({
            "system_status": {
                "status": "initializing",
                "timestamp": datetime.now().isoformat()
            },
            "market_data": [],
            "trading_history": [],
            "performance_metrics": {},
            "risk_parameters": {
                "risk_level": "medium",
                "max_position_size_pct": 0.10,
                "max_daily_drawdown_pct": 0.05,
                "stop_loss_pct": 0.02,
                "compounding_rate": 0.0
            }
        })
        
        logger.info("LLM Overseer initialized")
    
    async def make_strategic_decision(self, decision_type: str, prompt: str, urgency: str = "normal") -> Dict[str, Any]:
        """
        Make strategic decision using LLM.
        
        Args:
            decision_type: Type of decision to make
            prompt: Prompt for LLM
            urgency: Urgency level ("low", "normal", "high")
            
        Returns:
            Decision result
        """
        # Determine tier based on decision type and urgency
        tier = self._determine_tier(decision_type, urgency)
        
        # Add context to prompt
        context_data = self.context_manager.get_relevant_context(decision_type)
        enhanced_prompt = self._enhance_prompt_with_context(prompt, context_data)
        
        # Make decision using LLM
        try:
            # This would call the actual LLM via OpenRouter
            # For now, we'll simulate a response
            decision = f"Strategic decision for {decision_type}: Based on the analysis of market conditions and trading history, I recommend..."
            
            # Simulate token usage
            input_tokens = len(enhanced_prompt.split())
            output_tokens = len(decision.split())
            
            # Track token usage
            self.token_tracker.track_usage(tier, input_tokens, output_tokens)
            
            # Return decision result
            return {
                "success": True,
                "decision": decision,
                "model": f"tier_{tier}_model",
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens
                }
            }
        except Exception as e:
            logger.error(f"Error making strategic decision: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _determine_tier(self, decision_type: str, urgency: str) -> int:
        """
        Determine LLM tier based on decision type and urgency.
        
        Args:
            decision_type: Type of decision to make
            urgency: Urgency level ("low", "normal", "high")
            
        Returns:
            Tier level (1, 2, or 3)
        """
        # High urgency decisions always use tier 3
        if urgency == "high":
            return 3
        
        # Map decision types to tiers
        tier_mapping = {
            # Tier 1 (frequent, routine decisions)
            "market_monitoring": 1,
            "trade_notification": 1,
            "performance_update": 1,
            
            # Tier 2 (daily strategy adjustments)
            "market_trend_analysis": 2,
            "strategy_adjustment": 2,
            "report_generation": 2,
            
            # Tier 3 (critical decisions)
            "risk_management": 3,
            "emergency_response": 3,
            "major_strategy_shift": 3
        }
        
        # Default to tier 2 if decision type not found
        return tier_mapping.get(decision_type, 2)
    
    def _enhance_prompt_with_context(self, prompt: str, context_data: Dict[str, Any]) -> str:
        """
        Enhance prompt with context data.
        
        Args:
            prompt: Original prompt
            context_data: Context data
            
        Returns:
            Enhanced prompt
        """
        # Convert context data to string representation
        context_str = "\n".join([f"{key}: {value}" for key, value in context_data.items()])
        
        # Combine prompt with context
        return f"Context:\n{context_str}\n\nPrompt:\n{prompt}"
    
    def update_system_status(self, status: Dict[str, Any]) -> None:
        """
        Update system status in context.
        
        Args:
            status: System status
        """
        self.context_manager.update_context("system_status", status)
    
    def update_market_data(self, data: Dict[str, Any]) -> None:
        """
        Update market data in context.
        
        Args:
            data: Market data
        """
        market_data = self.context_manager.context.get("market_data", [])
        market_data.append(data)
        
        # Keep only the last 100 data points
        if len(market_data) > 100:
            market_data = market_data[-100:]
        
        self.context_manager.update_context("market_data", market_data)
    
    def update_trading_history(self, trade: Dict[str, Any]) -> None:
        """
        Update trading history in context.
        
        Args:
            trade: Trade data
        """
        trading_history = self.context_manager.context.get("trading_history", [])
        trading_history.append(trade)
        
        # Keep only the last 50 trades
        if len(trading_history) > 50:
            trading_history = trading_history[-50:]
        
        self.context_manager.update_context("trading_history", trading_history)
    
    def update_performance_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Update performance metrics in context.
        
        Args:
            metrics: Performance metrics
        """
        self.context_manager.update_context("performance_metrics", metrics)
    
    def update_risk_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Update risk parameters in context.
        
        Args:
            parameters: Risk parameters
        """
        risk_parameters = self.context_manager.context.get("risk_parameters", {})
        risk_parameters.update(parameters)
        self.context_manager.update_context("risk_parameters", risk_parameters)
    
    def get_usage_statistics(self) -> Dict[str, Any]:
        """
        Get token usage statistics.
        
        Returns:
            Token usage statistics
        """
        return self.token_tracker.get_statistics()


async def main():
    """Main function for testing."""
    # Initialize LLM Overseer
    overseer = LLMOverseer()
    
    # Update context with test data
    overseer.update_market_data({
        "timestamp": datetime.now().isoformat(),
        "price": 106739.83,
        "volume": 1234.56,
        "bid": 106730.00,
        "ask": 106750.00,
        "spread": 20.00
    })
    
    overseer.update_system_status({
        "status": "running",
        "uptime": "0 days, 0 hours, 5 minutes",
        "active_strategies": 1
    })
    
    # Test strategic decision making
    result = await overseer.make_strategic_decision(
        "market_trend_analysis",
        "Analyze the current market trend for BTC/USDC and recommend a trading strategy.",
        urgency="normal"
    )
    
    print(f"Decision: {result['decision']}")
    print(f"Model used: {result['model']}")
    print(f"Token usage: {result['usage']}")
    
    # Test token usage tracking
    usage = overseer.get_usage_statistics()
    print(f"Usage statistics: {usage}")


if __name__ == "__main__":
    asyncio.run(main())
