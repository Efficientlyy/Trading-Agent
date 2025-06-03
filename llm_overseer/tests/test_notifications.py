#!/usr/bin/env python
"""
Test module for notification and reporting features.

This module tests the notification formatting, delivery, and reporting
mechanisms of the LLM Strategic Overseer.
"""

import os
import sys
import unittest
import asyncio
from unittest.mock import MagicMock, patch
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use absolute imports instead of relative imports
from llm_overseer.config.config import Config
from llm_overseer.telegram.notifications import NotificationManager
from llm_overseer.core.llm_manager import TieredLLMManager

class TestNotifications(unittest.TestCase):
    """Test notification and reporting features."""
    
    def setUp(self):
        """Set up test environment."""
        # Create mock config
        self.config = MagicMock(spec=Config)
        self.config.get.return_value = {}
        
        # Create mock telegram bot
        self.telegram_bot = MagicMock()
        self.telegram_bot.send_notification = MagicMock(return_value=asyncio.Future())
        self.telegram_bot.send_notification.return_value.set_result(True)
        self.telegram_bot.broadcast_notification = MagicMock(return_value=asyncio.Future())
        self.telegram_bot.broadcast_notification.return_value.set_result({123: True})
        
        # Create notification manager
        self.notification_manager = NotificationManager(self.config, self.telegram_bot)
        
        # Create mock LLM manager
        self.llm_manager = MagicMock(spec=TieredLLMManager)
        self.llm_manager.generate_response = MagicMock(return_value=asyncio.Future())
        self.llm_manager.generate_response.return_value.set_result({
            "success": True,
            "response": "This is a test report.",
            "model": "test_model",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150
            }
        })
    
    def test_format_notification(self):
        """Test notification formatting."""
        # Test emergency notification
        emergency_msg = self.notification_manager.format_notification(
            "System is shutting down", 
            level="emergency"
        )
        self.assertIn("🚨 *EMERGENCY*", emergency_msg)
        self.assertIn("System is shutting down", emergency_msg)
        
        # Test critical notification
        critical_msg = self.notification_manager.format_notification(
            "API connection lost", 
            level="critical"
        )
        self.assertIn("⚠️ *CRITICAL*", critical_msg)
        self.assertIn("API connection lost", critical_msg)
        
        # Test trade notification
        trade_msg = self.notification_manager.format_notification(
            "BTC/USDC trade executed", 
            level="trade"
        )
        self.assertIn("💰 *TRADE*", trade_msg)
        self.assertIn("BTC/USDC trade executed", trade_msg)
        
        # Test info notification
        info_msg = self.notification_manager.format_notification(
            "System started", 
            level="info"
        )
        self.assertIn("ℹ️ *INFO*", info_msg)
        self.assertIn("System started", info_msg)
        
        # Test with additional data
        data_msg = self.notification_manager.format_notification(
            "Performance update", 
            level="info",
            data={
                "daily_pnl": 123.45,
                "win_rate": "65%"
            }
        )
        self.assertIn("*Details:*", data_msg)
        self.assertIn("daily_pnl", data_msg)
        self.assertIn("123.45", data_msg)
        self.assertIn("win_rate", data_msg)
        self.assertIn("65%", data_msg)
    
    def test_format_trade_notification(self):
        """Test trade notification formatting."""
        # Test buy trade
        buy_trade = {
            "type": "BUY",
            "symbol": "BTC/USDC",
            "price": 106739.83,
            "size": 0.1,
            "order_id": "123456",
            "timestamp": "2025-06-03T21:15:00Z",
            "fee": 0.00001,
            "fee_currency": "BTC",
            "status": "FILLED"
        }
        
        message, data = self.notification_manager.format_trade_notification(buy_trade)
        
        self.assertIn("🟢 *BUY Order Executed*", message)
        self.assertIn("0.1 BTC/USDC", message)
        self.assertIn("106739.83", message)
        self.assertEqual(data["Order ID"], "123456")
        self.assertEqual(data["Status"], "FILLED")
        
        # Test sell trade
        sell_trade = {
            "type": "SELL",
            "symbol": "BTC/USDC",
            "price": 106839.83,
            "size": 0.1,
            "order_id": "123457",
            "timestamp": "2025-06-03T21:20:00Z",
            "fee": 0.1,
            "fee_currency": "USDC",
            "status": "FILLED"
        }
        
        message, data = self.notification_manager.format_trade_notification(sell_trade)
        
        self.assertIn("🔴 *SELL Order Executed*", message)
        self.assertIn("0.1 BTC/USDC", message)
        self.assertIn("106839.83", message)
        self.assertEqual(data["Order ID"], "123457")
        self.assertEqual(data["Fee"], "0.10000000 USDC")
    
    def test_format_performance_notification(self):
        """Test performance notification formatting."""
        # Test positive performance
        positive_perf = {
            "daily_pnl": 123.45,
            "daily_pnl_pct": 0.5,
            "total_trades": 10,
            "win_rate": 70,
            "avg_trade_pnl": 12.35,
            "largest_win": 50.0,
            "largest_loss": -10.0
        }
        
        message, data = self.notification_manager.format_performance_notification(positive_perf)
        
        self.assertIn("📈 *Positive Daily Performance*", message)
        self.assertIn("+$123.45", message)
        self.assertIn("+0.50%", message)
        self.assertEqual(data["Total Trades"], 10)
        self.assertEqual(data["Win Rate"], "70.00%")
        
        # Test negative performance
        negative_perf = {
            "daily_pnl": -50.0,
            "daily_pnl_pct": -0.2,
            "total_trades": 8,
            "win_rate": 37.5,
            "avg_trade_pnl": -6.25,
            "largest_win": 20.0,
            "largest_loss": -30.0
        }
        
        message, data = self.notification_manager.format_performance_notification(negative_perf)
        
        self.assertIn("📉 *Negative Daily Performance*", message)
        self.assertIn("-$50.00", message)
        self.assertIn("-0.20%", message)
        self.assertEqual(data["Total Trades"], 8)
        self.assertEqual(data["Win Rate"], "37.50%")
    
    def test_format_risk_notification(self):
        """Test risk notification formatting."""
        # Test high risk
        high_risk = {
            "risk_level": "HIGH",
            "reason": "Position size exceeds threshold",
            "current_exposure": 25.0,
            "max_allowed": 20.0,
            "daily_drawdown": 8.5,
            "recommended_action": "Reduce position size"
        }
        
        message, data = self.notification_manager.format_risk_notification(high_risk)
        
        self.assertIn("🔴 *HIGH Risk Alert*", message)
        self.assertIn("Position size exceeds threshold", message)
        self.assertEqual(data["Current Exposure"], "25.00%")
        self.assertEqual(data["Recommended Action"], "Reduce position size")
        
        # Test medium risk
        medium_risk = {
            "risk_level": "MEDIUM",
            "reason": "Market volatility increasing",
            "current_exposure": 15.0,
            "max_allowed": 20.0,
            "daily_drawdown": 3.5,
            "recommended_action": "Monitor closely"
        }
        
        message, data = self.notification_manager.format_risk_notification(medium_risk)
        
        self.assertIn("🟠 *MEDIUM Risk Alert*", message)
        self.assertIn("Market volatility increasing", message)
        self.assertEqual(data["Current Exposure"], "15.00%")
        self.assertEqual(data["Recommended Action"], "Monitor closely")
    
    async def test_send_notification(self):
        """Test sending notification to user."""
        result = await self.notification_manager.send_notification(
            123,
            "Test message",
            level="info"
        )
        
        self.assertTrue(result)
        self.telegram_bot.send_notification.assert_called_once()
    
    async def test_broadcast_notification(self):
        """Test broadcasting notification to all users."""
        result = await self.notification_manager.broadcast_notification(
            "Broadcast test",
            level="critical"
        )
        
        self.assertEqual(result, {123: True})
        self.telegram_bot.broadcast_notification.assert_called_once()

if __name__ == "__main__":
    unittest.main()
