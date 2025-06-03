#!/usr/bin/env python
"""
Notification Manager for LLM Strategic Overseer

This module manages notifications and reporting for the LLM Strategic Overseer,
providing configurable alerts and performance reports through various channels.
"""

import os
import sys
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("notification_manager.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("notification_manager")

class NotificationManager:
    """
    Manages notifications and reporting for the LLM Strategic Overseer.
    
    This class handles the creation, prioritization, and delivery of notifications
    through various channels, including Telegram.
    """
    
    def __init__(self, event_bus=None, telegram_bot=None):
        """
        Initialize Notification Manager.
        
        Args:
            event_bus: Event Bus instance
            telegram_bot: Telegram Bot instance
        """
        self.event_bus = event_bus
        self.telegram_bot = telegram_bot
        
        # Notification settings
        self.notification_levels = {
            "emergency": 0,  # Highest priority
            "high": 1,
            "normal": 2,
            "low": 3        # Lowest priority
        }
        
        # User notification preferences
        self.user_preferences = {
            "min_notification_level": "normal",  # Minimum level to notify
            "trading_notifications": True,       # Notifications for trading events
            "performance_reports": True,         # Performance reports
            "system_alerts": True,               # System alerts
            "quiet_hours": {                     # Do not disturb during these hours
                "enabled": False,
                "start": "22:00",
                "end": "08:00",
                "timezone": "UTC"
            }
        }
        
        # Notification history
        self.notification_history = []
        
        # Register event handlers if event bus is provided
        if self.event_bus:
            self._register_event_handlers()
        
        logger.info("Notification Manager initialized")
    
    def set_event_bus(self, event_bus):
        """
        Set Event Bus instance.
        
        Args:
            event_bus: Event Bus instance
        """
        self.event_bus = event_bus
        self._register_event_handlers()
        logger.info("Event Bus set for Notification Manager")
    
    def set_telegram_bot(self, telegram_bot):
        """
        Set Telegram Bot instance.
        
        Args:
            telegram_bot: Telegram Bot instance
        """
        self.telegram_bot = telegram_bot
        logger.info("Telegram Bot set for Notification Manager")
    
    def _register_event_handlers(self):
        """
        Register event handlers with Event Bus.
        
        Subscribes to relevant events for notifications.
        """
        if self.event_bus:
            # Trading events
            self.event_bus.subscribe("llm.strategic_decision", self._handle_strategic_decision)
            self.event_bus.subscribe("llm.risk_alert", self._handle_risk_alert)
            
            # System events
            self.event_bus.subscribe("system.error", self._handle_system_error)
            self.event_bus.subscribe("system.warning", self._handle_system_warning)
            
            # Performance events
            self.event_bus.subscribe("performance.daily_report", self._handle_performance_report)
            self.event_bus.subscribe("performance.trade_completed", self._handle_trade_completed)
            
            logger.info("Notification Manager event handlers registered")
        else:
            logger.warning("Event Bus not set, cannot register handlers")
    
    def update_user_preferences(self, preferences: Dict[str, Any]):
        """
        Update user notification preferences.
        
        Args:
            preferences: Dictionary of user preferences
        """
        # Update only provided preferences
        for key, value in preferences.items():
            if key in self.user_preferences:
                if key == "quiet_hours" and isinstance(value, dict):
                    # Update quiet hours settings
                    for qh_key, qh_value in value.items():
                        if qh_key in self.user_preferences["quiet_hours"]:
                            self.user_preferences["quiet_hours"][qh_key] = qh_value
                else:
                    # Update other preferences
                    self.user_preferences[key] = value
        
        logger.info(f"User preferences updated: {self.user_preferences}")
    
    def get_user_preferences(self) -> Dict[str, Any]:
        """
        Get user notification preferences.
        
        Returns:
            Dictionary of user preferences
        """
        return self.user_preferences
    
    async def send_notification(self, message: str, level: str = "normal", data: Dict[str, Any] = None):
        """
        Send notification to user.
        
        Args:
            message: Notification message
            level: Notification level ("emergency", "high", "normal", "low")
            data: Additional data for the notification
        """
        # Check if notification level is sufficient
        if self.notification_levels.get(level, 3) > self.notification_levels.get(self.user_preferences["min_notification_level"], 2):
            logger.debug(f"Notification level {level} below minimum {self.user_preferences['min_notification_level']}, skipping")
            return
        
        # Check quiet hours
        if self._is_quiet_hours():
            # Only send emergency notifications during quiet hours
            if level != "emergency":
                logger.debug(f"Quiet hours active, skipping {level} notification")
                return
        
        # Create notification
        notification = {
            "message": message,
            "level": level,
            "timestamp": datetime.now().isoformat(),
            "data": data or {}
        }
        
        # Add to history
        self.notification_history.append(notification)
        
        # Trim history if too long
        if len(self.notification_history) > 100:
            self.notification_history = self.notification_history[-100:]
        
        # Send via Telegram if available
        if self.telegram_bot:
            try:
                # Format message based on level
                formatted_message = self._format_notification(notification)
                await self.telegram_bot.send_message(formatted_message)
                logger.info(f"Notification sent via Telegram: {level} - {message[:50]}...")
            except Exception as e:
                logger.error(f"Error sending notification via Telegram: {e}")
        else:
            logger.warning("Telegram Bot not available, notification not sent")
    
    async def send_report(self, report_type: str, data: Dict[str, Any]):
        """
        Send report to user.
        
        Args:
            report_type: Type of report ("daily", "weekly", "monthly", "trade")
            data: Report data
        """
        # Check if performance reports are enabled
        if not self.user_preferences["performance_reports"]:
            logger.debug(f"Performance reports disabled, skipping {report_type} report")
            return
        
        # Create report
        report = {
            "type": report_type,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        
        # Format report
        formatted_report = self._format_report(report)
        
        # Send via Telegram if available
        if self.telegram_bot:
            try:
                await self.telegram_bot.send_message(formatted_report)
                logger.info(f"Report sent via Telegram: {report_type}")
            except Exception as e:
                logger.error(f"Error sending report via Telegram: {e}")
        else:
            logger.warning("Telegram Bot not available, report not sent")
    
    def get_notification_history(self, count: int = 10, level: str = None) -> List[Dict[str, Any]]:
        """
        Get notification history.
        
        Args:
            count: Number of notifications to return
            level: Filter by notification level
            
        Returns:
            List of notifications
        """
        # Filter by level if provided
        if level:
            filtered_history = [n for n in self.notification_history if n["level"] == level]
        else:
            filtered_history = self.notification_history
        
        # Return most recent notifications
        return filtered_history[-count:]
    
    async def generate_daily_report(self, date: datetime = None):
        """
        Generate daily performance report.
        
        Args:
            date: Date for the report (defaults to today)
        """
        if date is None:
            date = datetime.now()
        
        # TODO: Implement actual performance data collection
        # For now, use placeholder data
        report_data = {
            "date": date.strftime("%Y-%m-%d"),
            "trades": {
                "total": 12,
                "successful": 8,
                "failed": 4
            },
            "performance": {
                "profit_loss": "+2.3%",
                "total_volume": "$15,432",
                "fees": "$12.45"
            },
            "assets": {
                "BTC/USDC": {
                    "trades": 8,
                    "profit_loss": "+3.1%"
                },
                "ETH/USDC": {
                    "trades": 4,
                    "profit_loss": "+0.8%"
                }
            }
        }
        
        await self.send_report("daily", report_data)
    
    async def generate_weekly_report(self, end_date: datetime = None):
        """
        Generate weekly performance report.
        
        Args:
            end_date: End date for the report (defaults to today)
        """
        if end_date is None:
            end_date = datetime.now()
        
        # Calculate start date (7 days ago)
        start_date = end_date - timedelta(days=7)
        
        # TODO: Implement actual performance data collection
        # For now, use placeholder data
        report_data = {
            "period": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            "trades": {
                "total": 84,
                "successful": 56,
                "failed": 28
            },
            "performance": {
                "profit_loss": "+5.7%",
                "total_volume": "$98,765",
                "fees": "$78.32"
            },
            "assets": {
                "BTC/USDC": {
                    "trades": 52,
                    "profit_loss": "+4.2%"
                },
                "ETH/USDC": {
                    "trades": 32,
                    "profit_loss": "+8.5%"
                }
            },
            "best_day": {
                "date": (start_date + timedelta(days=3)).strftime("%Y-%m-%d"),
                "profit_loss": "+2.1%"
            },
            "worst_day": {
                "date": (start_date + timedelta(days=5)).strftime("%Y-%m-%d"),
                "profit_loss": "-0.8%"
            }
        }
        
        await self.send_report("weekly", report_data)
    
    def _is_quiet_hours(self) -> bool:
        """
        Check if current time is within quiet hours.
        
        Returns:
            True if within quiet hours, False otherwise
        """
        # If quiet hours are disabled, always return False
        if not self.user_preferences["quiet_hours"]["enabled"]:
            return False
        
        # Get current time in configured timezone
        # TODO: Implement timezone conversion
        now = datetime.now()
        current_time = now.strftime("%H:%M")
        
        # Get quiet hours
        start_time = self.user_preferences["quiet_hours"]["start"]
        end_time = self.user_preferences["quiet_hours"]["end"]
        
        # Check if current time is within quiet hours
        if start_time <= end_time:
            # Simple case: start time is before end time
            return start_time <= current_time <= end_time
        else:
            # Complex case: quiet hours span midnight
            return current_time >= start_time or current_time <= end_time
    
    def _format_notification(self, notification: Dict[str, Any]) -> str:
        """
        Format notification for display.
        
        Args:
            notification: Notification data
            
        Returns:
            Formatted notification message
        """
        # Format based on level
        level = notification["level"]
        message = notification["message"]
        timestamp = datetime.fromisoformat(notification["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
        
        # Add emoji based on level
        if level == "emergency":
            emoji = "🚨"
        elif level == "high":
            emoji = "⚠️"
        elif level == "normal":
            emoji = "ℹ️"
        else:  # low
            emoji = "📝"
        
        # Format message
        formatted_message = f"{emoji} *{level.upper()}*\n{message}\n\n_Time: {timestamp}_"
        
        # Add data if available
        if notification["data"]:
            data_str = "\n".join([f"- {k}: {v}" for k, v in notification["data"].items()])
            formatted_message += f"\n\n*Details:*\n{data_str}"
        
        return formatted_message
    
    def _format_report(self, report: Dict[str, Any]) -> str:
        """
        Format report for display.
        
        Args:
            report: Report data
            
        Returns:
            Formatted report message
        """
        report_type = report["type"]
        data = report["data"]
        
        if report_type == "daily":
            return self._format_daily_report(data)
        elif report_type == "weekly":
            return self._format_weekly_report(data)
        elif report_type == "trade":
            return self._format_trade_report(data)
        else:
            return f"*{report_type.capitalize()} Report*\n\n{json.dumps(data, indent=2)}"
    
    def _format_daily_report(self, data: Dict[str, Any]) -> str:
        """
        Format daily report for display.
        
        Args:
            data: Report data
            
        Returns:
            Formatted report message
        """
        date = data["date"]
        trades = data["trades"]
        performance = data["performance"]
        assets = data["assets"]
        
        # Format message
        message = f"📊 *Daily Performance Report - {date}*\n\n"
        
        # Add performance summary
        message += f"*Performance:* {performance['profit_loss']}\n"
        message += f"*Volume:* {performance['total_volume']}\n"
        message += f"*Fees:* {performance['fees']}\n\n"
        
        # Add trade summary
        message += f"*Trades:* {trades['total']} total\n"
        message += f"✅ {trades['successful']} successful\n"
        message += f"❌ {trades['failed']} failed\n\n"
        
        # Add asset breakdown
        message += "*Asset Breakdown:*\n"
        for asset, asset_data in assets.items():
            message += f"- {asset}: {asset_data['profit_loss']} ({asset_data['trades']} trades)\n"
        
        return message
    
    def _format_weekly_report(self, data: Dict[str, Any]) -> str:
        """
        Format weekly report for display.
        
        Args:
            data: Report data
            
        Returns:
            Formatted report message
        """
        period = data["period"]
        trades = data["trades"]
        performance = data["performance"]
        assets = data["assets"]
        best_day = data["best_day"]
        worst_day = data["worst_day"]
        
        # Format message
        message = f"📈 *Weekly Performance Report*\n*{period}*\n\n"
        
        # Add performance summary
        message += f"*Performance:* {performance['profit_loss']}\n"
        message += f"*Volume:* {performance['total_volume']}\n"
        message += f"*Fees:* {performance['fees']}\n\n"
        
        # Add trade summary
        message += f"*Trades:* {trades['total']} total\n"
        message += f"✅ {trades['successful']} successful ({trades['successful']/trades['total']*100:.1f}%)\n"
        message += f"❌ {trades['failed']} failed ({trades['failed']/trades['total']*100:.1f}%)\n\n"
        
        # Add asset breakdown
        message += "*Asset Breakdown:*\n"
        for asset, asset_data in assets.items():
            message += f"- {asset}: {asset_data['profit_loss']} ({asset_data['trades']} trades)\n"
        
        # Add best and worst days
        message += f"\n*Best Day:* {best_day['date']} ({best_day['profit_loss']})\n"
        message += f"*Worst Day:* {worst_day['date']} ({worst_day['profit_loss']})\n"
        
        return message
    
    def _format_trade_report(self, data: Dict[str, Any]) -> str:
        """
        Format trade report for display.
        
        Args:
            data: Report data
            
        Returns:
            Formatted report message
        """
        trade_id = data.get("trade_id", "Unknown")
        symbol = data.get("symbol", "Unknown")
        side = data.get("side", "Unknown")
        entry_price = data.get("entry_price", "Unknown")
        exit_price = data.get("exit_price", "Unknown")
        profit_loss = data.get("profit_loss", "Unknown")
        profit_loss_pct = data.get("profit_loss_pct", "Unknown")
        duration = data.get("duration", "Unknown")
        
        # Determine emoji based on profit/loss
        if isinstance(profit_loss_pct, str) and profit_loss_pct.startswith("+"):
            emoji = "🟢"
        elif isinstance(profit_loss_pct, str) and profit_loss_pct.startswith("-"):
            emoji = "🔴"
        else:
            emoji = "⚪"
        
        # Format message
        message = f"{emoji} *Trade Completed - {symbol}*\n\n"
        message += f"*Trade ID:* {trade_id}\n"
        message += f"*Side:* {side.upper()}\n"
        message += f"*Entry:* {entry_price}\n"
        message += f"*Exit:* {exit_price}\n"
        message += f"*P/L:* {profit_loss} ({profit_loss_pct})\n"
        message += f"*Duration:* {duration}\n"
        
        # Add strategy if available
        if "strategy" in data:
            message += f"*Strategy:* {data['strategy']}\n"
        
        # Add reason if available
        if "reason" in data:
            message += f"*Reason:* {data['reason']}\n"
        
        return message
    
    async def _handle_strategic_decision(self, topic: str, data: Dict[str, Any]):
        """
        Handle strategic decision event.
        
        Args:
            topic: Event topic
            data: Event data
        """
        # Check if trading notifications are enabled
        if not self.user_preferences["trading_notifications"]:
            return
        
        decision_type = data.get("decision_type", "unknown")
        symbol = data.get("symbol", "unknown")
        direction = data.get("direction", "unknown")
        confidence = data.get("confidence", 0)
        
        # Determine notification level based on decision type
        if decision_type in ["entry", "exit"]:
            level = "high"
        else:
            level = "normal"
        
        # Create message
        if decision_type == "entry":
            message = f"New {direction} position opened for {symbol}"
        elif decision_type == "exit":
            message = f"Position closed for {symbol}"
        elif decision_type == "risk_adjustment":
            message = f"Risk adjusted for {symbol} position"
        else:
            message = f"Strategic decision: {decision_type} for {symbol}"
        
        # Add confidence
        message += f" (confidence: {confidence:.2f})"
        
        # Send notification
        await self.send_notification(message, level, data)
    
    async def _handle_risk_alert(self, topic: str, data: Dict[str, Any]):
        """
        Handle risk alert event.
        
        Args:
            topic: Event topic
            data: Event data
        """
        # Check if system alerts are enabled
        if not self.user_preferences["system_alerts"]:
            return
        
        alert_type = data.get("alert_type", "unknown")
        symbol = data.get("symbol", "unknown")
        severity = data.get("severity", "normal")
        
        # Create message
        message = f"Risk Alert: {alert_type} for {symbol}"
        
        # Send notification
        await self.send_notification(message, severity, data)
    
    async def _handle_system_error(self, topic: str, data: Dict[str, Any]):
        """
        Handle system error event.
        
        Args:
            topic: Event topic
            data: Event data
        """
        # Check if system alerts are enabled
        if not self.user_preferences["system_alerts"]:
            return
        
        error_type = data.get("error_type", "unknown")
        component = data.get("component", "unknown")
        message = data.get("message", "Unknown error")
        
        # Create notification message
        notification_message = f"System Error: {error_type} in {component}\n{message}"
        
        # Send notification
        await self.send_notification(notification_message, "high", data)
    
    async def _handle_system_warning(self, topic: str, data: Dict[str, Any]):
        """
        Handle system warning event.
        
        Args:
            topic: Event topic
            data: Event data
        """
        # Check if system alerts are enabled
        if not self.user_preferences["system_alerts"]:
            return
        
        warning_type = data.get("warning_type", "unknown")
        component = data.get("component", "unknown")
        message = data.get("message", "Unknown warning")
        
        # Create notification message
        notification_message = f"System Warning: {warning_type} in {component}\n{message}"
        
        # Send notification
        await self.send_notification(notification_message, "normal", data)
    
    async def _handle_performance_report(self, topic: str, data: Dict[str, Any]):
        """
        Handle performance report event.
        
        Args:
            topic: Event topic
            data: Event data
        """
        # Check if performance reports are enabled
        if not self.user_preferences["performance_reports"]:
            return
        
        report_type = data.get("report_type", "unknown")
        
        # Send report
        await self.send_report(report_type, data)
    
    async def _handle_trade_completed(self, topic: str, data: Dict[str, Any]):
        """
        Handle trade completed event.
        
        Args:
            topic: Event topic
            data: Event data
        """
        # Check if trading notifications are enabled
        if not self.user_preferences["trading_notifications"]:
            return
        
        # Send trade report
        await self.send_report("trade", data)


# For testing
async def test():
    """Test function for NotificationManager."""
    from datetime import timedelta
    
    # Create mock Telegram bot
    class MockTelegramBot:
        async def send_message(self, message):
            print(f"TELEGRAM: {message}")
    
    # Create notification manager
    notification_manager = NotificationManager(telegram_bot=MockTelegramBot())
    
    # Update user preferences
    notification_manager.update_user_preferences({
        "min_notification_level": "low",
        "trading_notifications": True,
        "performance_reports": True,
        "system_alerts": True
    })
    
    # Send test notifications
    await notification_manager.send_notification(
        "Emergency test notification",
        "emergency",
        {"test": True, "source": "test function"}
    )
    
    await notification_manager.send_notification(
        "High priority test notification",
        "high",
        {"test": True, "source": "test function"}
    )
    
    await notification_manager.send_notification(
        "Normal priority test notification",
        "normal",
        {"test": True, "source": "test function"}
    )
    
    await notification_manager.send_notification(
        "Low priority test notification",
        "low",
        {"test": True, "source": "test function"}
    )
    
    # Generate test reports
    await notification_manager.generate_daily_report()
    await notification_manager.generate_weekly_report()
    
    # Send test trade report
    await notification_manager.send_report("trade", {
        "trade_id": "T12345",
        "symbol": "BTC/USDC",
        "side": "buy",
        "entry_price": "50,123.45",
        "exit_price": "50,456.78",
        "profit_loss": "+$333.33",
        "profit_loss_pct": "+0.67%",
        "duration": "2h 15m",
        "strategy": "RSI Oversold Bounce",
        "reason": "RSI crossed above 30 with bullish divergence"
    })

if __name__ == "__main__":
    asyncio.run(test())
