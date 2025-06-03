#!/usr/bin/env python
"""
Notification management module for Telegram bot integration.

This module handles formatting and sending notifications to users
through the Telegram bot.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ..config.config import Config

logger = logging.getLogger(__name__)

class NotificationManager:
    """
    Manages notifications for the Telegram bot.
    Handles formatting, prioritization, and delivery of notifications.
    """
    
    def __init__(self, config: Config, telegram_bot=None):
        """
        Initialize notification manager.
        
        Args:
            config: Configuration object
            telegram_bot: Telegram bot instance (optional, can be set later)
        """
        self.config = config
        self.telegram_bot = telegram_bot
        self.notification_levels = {
            "emergency": 1,  # Highest priority
            "critical": 2,
            "trade": 3,
            "info": 4        # Lowest priority
        }
    
    def set_telegram_bot(self, telegram_bot) -> None:
        """
        Set Telegram bot instance.
        
        Args:
            telegram_bot: Telegram bot instance
        """
        self.telegram_bot = telegram_bot
    
    def format_notification(self, message: str, level: str = "info", 
                           data: Optional[Dict[str, Any]] = None) -> str:
        """
        Format notification message.
        
        Args:
            message: Base message
            level: Notification level
            data: Additional data to include
            
        Returns:
            Formatted message
        """
        # Add emoji based on level
        if level == "emergency":
            prefix = "🚨 *EMERGENCY*\n\n"
        elif level == "critical":
            prefix = "⚠️ *CRITICAL*\n\n"
        elif level == "trade":
            prefix = "💰 *TRADE*\n\n"
        else:  # info
            prefix = "ℹ️ *INFO*\n\n"
        
        formatted_message = f"{prefix}{message}"
        
        # Add additional data if provided
        if data:
            data_str = "\n\n*Details:*\n"
            for key, value in data.items():
                data_str += f"- *{key}:* {value}\n"
            formatted_message += data_str
        
        return formatted_message
    
    async def send_notification(self, user_id: int, message: str, level: str = "info", 
                              data: Optional[Dict[str, Any]] = None) -> bool:
        """
        Send notification to user.
        
        Args:
            user_id: Telegram user ID
            message: Notification message
            level: Notification level
            data: Additional data to include
            
        Returns:
            True if notification was sent, False otherwise
        """
        if not self.telegram_bot:
            logger.error("Telegram bot not set")
            return False
        
        formatted_message = self.format_notification(message, level, data)
        
        return await self.telegram_bot.send_notification(user_id, formatted_message, level)
    
    async def broadcast_notification(self, message: str, level: str = "info", 
                                   data: Optional[Dict[str, Any]] = None) -> Dict[int, bool]:
        """
        Broadcast notification to all allowed users.
        
        Args:
            message: Notification message
            level: Notification level
            data: Additional data to include
            
        Returns:
            Dictionary mapping user IDs to success status
        """
        if not self.telegram_bot:
            logger.error("Telegram bot not set")
            return {}
        
        formatted_message = self.format_notification(message, level, data)
        
        return await self.telegram_bot.broadcast_notification(formatted_message, level)
    
    def format_trade_notification(self, trade_data: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Format trade notification.
        
        Args:
            trade_data: Trade data
            
        Returns:
            Tuple of (message, data)
        """
        trade_type = trade_data.get("type", "unknown").upper()
        symbol = trade_data.get("symbol", "unknown")
        price = trade_data.get("price", 0)
        size = trade_data.get("size", 0)
        value = price * size
        
        if trade_type == "BUY":
            message = f"🟢 *BUY Order Executed*\n\n{size} {symbol} @ ${price:.2f}\nTotal Value: ${value:.2f}"
        elif trade_type == "SELL":
            message = f"🔴 *SELL Order Executed*\n\n{size} {symbol} @ ${price:.2f}\nTotal Value: ${value:.2f}"
        else:
            message = f"⚪ *{trade_type} Order Executed*\n\n{size} {symbol} @ ${price:.2f}\nTotal Value: ${value:.2f}"
        
        # Additional data for detailed view
        data = {
            "Order ID": trade_data.get("order_id", "N/A"),
            "Timestamp": trade_data.get("timestamp", "N/A"),
            "Fee": f"{trade_data.get('fee', 0):.8f} {trade_data.get('fee_currency', 'USDC')}",
            "Status": trade_data.get("status", "N/A")
        }
        
        return message, data
    
    def format_performance_notification(self, performance_data: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Format performance notification.
        
        Args:
            performance_data: Performance data
            
        Returns:
            Tuple of (message, data)
        """
        daily_pnl = performance_data.get("daily_pnl", 0)
        daily_pnl_pct = performance_data.get("daily_pnl_pct", 0)
        
        if daily_pnl > 0:
            message = f"📈 *Positive Daily Performance*\n\nDaily P&L: +${daily_pnl:.2f} (+{daily_pnl_pct:.2f}%)"
        elif daily_pnl < 0:
            message = f"📉 *Negative Daily Performance*\n\nDaily P&L: -${abs(daily_pnl):.2f} ({daily_pnl_pct:.2f}%)"
        else:
            message = f"⏸️ *Neutral Daily Performance*\n\nDaily P&L: ${daily_pnl:.2f} ({daily_pnl_pct:.2f}%)"
        
        # Additional data for detailed view
        data = {
            "Total Trades": performance_data.get("total_trades", 0),
            "Win Rate": f"{performance_data.get('win_rate', 0):.2f}%",
            "Average Trade": f"${performance_data.get('avg_trade_pnl', 0):.2f}",
            "Largest Win": f"${performance_data.get('largest_win', 0):.2f}",
            "Largest Loss": f"${performance_data.get('largest_loss', 0):.2f}"
        }
        
        return message, data
    
    def format_risk_notification(self, risk_data: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Format risk notification.
        
        Args:
            risk_data: Risk data
            
        Returns:
            Tuple of (message, data)
        """
        risk_level = risk_data.get("risk_level", "unknown").upper()
        reason = risk_data.get("reason", "Unknown reason")
        
        if risk_level == "HIGH":
            message = f"🔴 *HIGH Risk Alert*\n\n{reason}"
        elif risk_level == "MEDIUM":
            message = f"🟠 *MEDIUM Risk Alert*\n\n{reason}"
        elif risk_level == "LOW":
            message = f"🟡 *LOW Risk Alert*\n\n{reason}"
        else:
            message = f"⚪ *{risk_level} Risk Alert*\n\n{reason}"
        
        # Additional data for detailed view
        data = {
            "Current Exposure": f"{risk_data.get('current_exposure', 0):.2f}%",
            "Max Allowed": f"{risk_data.get('max_allowed', 0):.2f}%",
            "Daily Drawdown": f"{risk_data.get('daily_drawdown', 0):.2f}%",
            "Recommended Action": risk_data.get("recommended_action", "N/A")
        }
        
        return message, data
