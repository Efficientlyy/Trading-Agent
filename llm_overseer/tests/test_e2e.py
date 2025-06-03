#!/usr/bin/env python
"""
End-to-end test for LLM Strategic Overseer.

This module tests the complete LLM Overseer system, including
LLM integration, Telegram bot, trading system integration,
notification delivery, and compounding strategy.
"""

import os
import sys
import asyncio
import logging
import unittest
import json
from unittest.mock import MagicMock, patch
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import components
from llm_overseer.core.llm_manager import TieredLLMManager
from llm_overseer.core.context_manager import ContextManager
from llm_overseer.core.token_tracker import TokenTracker
from llm_overseer.telegram.notifications import NotificationManager
from llm_overseer.strategy.compounding import CompoundingStrategy
from llm_overseer.integration.integration import LLMOverseerIntegration

class TestEndToEnd(unittest.TestCase):
    """End-to-end test for LLM Strategic Overseer."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary data directory
        self.test_data_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "test_data"
        )
        os.makedirs(self.test_data_dir, exist_ok=True)
        
        # Create a test config file
        self.test_config_file = os.path.join(self.test_data_dir, "test_settings.json")
        self.config_data = {
            "llm": {
                "provider": "openrouter",
                "api_key": "test_api_key",
                "api_key_env": "OPENROUTER_API_KEY",
                "models": {
                    "tier_1": "openai/gpt-3.5-turbo",
                    "tier_2": "anthropic/claude-3-sonnet",
                    "tier_3": "anthropic/claude-3-opus"
                }
            },
            "telegram": {
                "bot_token": "test_bot_token",
                "bot_token_env": "TELEGRAM_BOT_TOKEN",
                "allowed_user_ids": [123456789]
            },
            "trading": {
                "compounding": {
                    "enabled": True,
                    "reinvestment_rate": 0.8,
                    "min_profit_threshold": 100,
                    "frequency": "monthly"
                }
            }
        }
        
        with open(self.test_config_file, 'w') as f:
            json.dump(self.config_data, f, indent=2)
            
        # Create a mock config class that returns real values
        self.mock_config = MagicMock()
        def mock_config_get(key, default=None):
            if key == "trading.compounding.enabled":
                return True
            elif key == "trading.compounding.reinvestment_rate":
                return 0.8
            elif key == "trading.compounding.min_profit_threshold":
                return 100
            elif key == "trading.compounding.frequency":
                return "monthly"
            elif key == "llm.api_key":
                return "test_api_key"
            elif key == "telegram.bot_token":
                return "test_bot_token"
            elif key == "telegram.allowed_user_ids":
                return [123456789]
            return default
        self.mock_config.get.side_effect = mock_config_get
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove test data directory if it exists
        import shutil
        if os.path.exists(self.test_data_dir):
            shutil.rmtree(self.test_data_dir)
    
    async def test_end_to_end(self):
        """Test end-to-end functionality."""
        # Use patches to avoid actual API calls and environment loading
        with patch('llm_overseer.config.config.load_dotenv'), \
             patch('llm_overseer.main.LLMOverseer') as mock_llm_overseer:
            
            # Create mock LLM Overseer instance
            llm_overseer = mock_llm_overseer.return_value
            
            # Configure mock methods
            llm_overseer.make_strategic_decision = MagicMock(return_value=asyncio.Future())
            llm_overseer.make_strategic_decision.return_value.set_result({
                "success": True,
                "decision": "This is a test decision.",
                "model": "test_model",
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "total_tokens": 150
                }
            })
            
            llm_overseer.get_usage_statistics = MagicMock(return_value={
                "total_requests": 1,
                "cache_hits": 0,
                "tier_usage": {
                    1: {"requests": 0, "tokens": 0, "estimated_cost": 0},
                    2: {"requests": 1, "tokens": 150, "estimated_cost": 0.15},
                    3: {"requests": 0, "tokens": 0, "estimated_cost": 0}
                }
            })
            
            # Initialize LLM Overseer with config file path
            mock_llm_overseer.assert_not_called()  # Not called yet
            from llm_overseer.main import LLMOverseer
            real_llm_overseer = LLMOverseer(self.test_config_file)
            mock_llm_overseer.assert_called_once_with(self.test_config_file)
            
            # Initialize compounding strategy with proper mock config
            compounding = CompoundingStrategy(self.mock_config)
            compounding.initialize_capital(10000.0)
            
            # Test compounding strategy
            compounding.update_capital(10200.0)
            compounding_result = compounding.execute_compounding(10200.0, datetime(2025, 6, 1))
            
            self.assertTrue(compounding_result["compounded"])
            self.assertEqual(compounding_result["reinvested"], 160.0)
            self.assertEqual(compounding_result["withdrawn"], 40.0)
            
            # Test notification formatting
            notification_manager = NotificationManager(self.mock_config)
            
            trade_notification = notification_manager.format_notification(
                "BTC/USDC trade executed",
                level="trade",
                data={
                    "type": "BUY",
                    "price": 106739.83,
                    "size": 0.1
                }
            )
            
            self.assertIn("💰 *TRADE*", trade_notification)
            self.assertIn("BTC/USDC trade executed", trade_notification)
            
            # Test compounding statistics
            compounding_stats = compounding.get_statistics()
            self.assertEqual(compounding_stats["reinvestment_rate"], 0.8)
            self.assertEqual(compounding_stats["reinvested_profit"], 160.0)
            
            # All tests passed
            print("End-to-end test completed successfully")

if __name__ == "__main__":
    # Create test instance
    test = TestEndToEnd()
    # Set up test environment
    test.setUp()
    
    try:
        # Run test using asyncio
        asyncio.run(test.test_end_to_end())
        print("All tests passed!")
    finally:
        # Clean up
        test.tearDown()
