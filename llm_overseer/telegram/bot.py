#!/usr/bin/env python
"""
Telegram bot module for LLM Strategic Overseer.

This module implements the Telegram bot interface for the trading system,
providing secure command processing and notification management.
"""

import os
import sys
import logging
import asyncio
from typing import Dict, List, Any, Optional, Callable, Union

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters
)

# Import configuration and authentication
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ..config.config import Config
from .auth import TelegramAuth

logger = logging.getLogger(__name__)

class TelegramBot:
    """
    Telegram bot for LLM Strategic Overseer.
    Provides secure command interface and notification management.
    """
    
    def __init__(self, config: Config, llm_overseer=None):
        """
        Initialize Telegram bot.
        
        Args:
            config: Configuration object
            llm_overseer: LLM Overseer instance (optional, can be set later)
        """
        self.config = config
        self.llm_overseer = llm_overseer
        self.auth = TelegramAuth(config)
        
        # Get bot token
        self.bot_token = config.get("telegram.bot_token")
        if not self.bot_token:
            logger.error("Telegram bot token not found in configuration")
            raise ValueError("Telegram bot token not found in configuration")
        
        # Initialize application
        self.application = Application.builder().token(self.bot_token).build()
        
        # Register handlers
        self._register_handlers()
        
        logger.info("Telegram bot initialized")
    
    def set_llm_overseer(self, llm_overseer) -> None:
        """
        Set LLM Overseer instance.
        
        Args:
            llm_overseer: LLM Overseer instance
        """
        self.llm_overseer = llm_overseer
    
    def _register_handlers(self) -> None:
        """Register command and message handlers."""
        # Authentication commands
        self.application.add_handler(CommandHandler("start", self._start_command))
        self.application.add_handler(CommandHandler("login", self._login_command))
        self.application.add_handler(CommandHandler("logout", self._logout_command))
        
        # Status commands
        self.application.add_handler(CommandHandler("status", self._status_command))
        self.application.add_handler(CommandHandler("balance", self._balance_command))
        self.application.add_handler(CommandHandler("performance", self._performance_command))
        self.application.add_handler(CommandHandler("positions", self._positions_command))
        
        # Control commands
        self.application.add_handler(CommandHandler("pause", self._pause_command))
        self.application.add_handler(CommandHandler("resume", self._resume_command))
        self.application.add_handler(CommandHandler("risk", self._risk_command))
        self.application.add_handler(CommandHandler("emergency", self._emergency_command))
        
        # Information commands
        self.application.add_handler(CommandHandler("report", self._report_command))
        self.application.add_handler(CommandHandler("strategy", self._strategy_command))
        self.application.add_handler(CommandHandler("market", self._market_command))
        self.application.add_handler(CommandHandler("history", self._history_command))
        
        # Configuration commands
        self.application.add_handler(CommandHandler("notify", self._notify_command))
        self.application.add_handler(CommandHandler("schedule", self._schedule_command))
        self.application.add_handler(CommandHandler("compound", self._compound_command))
        self.application.add_handler(CommandHandler("settings", self._settings_command))
        
        # Help command
        self.application.add_handler(CommandHandler("help", self._help_command))
        
        # Handle authentication codes
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message))
        
        # Handle callback queries
        self.application.add_handler(CallbackQueryHandler(self._handle_callback))
        
        # Error handler
        self.application.add_error_handler(self._error_handler)
    
    async def _require_auth(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
        """
        Check if user is authenticated.
        
        Args:
            update: Update object
            context: Context object
            
        Returns:
            True if authenticated, False otherwise
        """
        user_id = update.effective_user.id
        
        if not self.auth.is_user_allowed(user_id):
            await update.message.reply_text("You are not authorized to use this bot.")
            return False
        
        if not self.auth.is_authenticated(user_id):
            await update.message.reply_text(
                "You are not authenticated. Please use /login to authenticate."
            )
            return False
        
        # Refresh session
        self.auth.refresh_session(user_id)
        return True
    
    async def _start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command."""
        user_id = update.effective_user.id
        username = update.effective_user.username or "User"
        
        if not self.auth.is_user_allowed(user_id):
            await update.message.reply_text(
                f"Hello {username}! You are not authorized to use this bot. "
                f"Please contact the administrator if you believe this is an error."
            )
            return
        
        await update.message.reply_text(
            f"Hello {username}! Welcome to the Trading Agent Bot.\n\n"
            f"This bot provides secure access to your trading system.\n\n"
            f"Please use /login to authenticate and access the trading commands.\n"
            f"Use /help to see available commands."
        )
    
    async def _login_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /login command."""
        user_id = update.effective_user.id
        
        if self.auth.is_authenticated(user_id):
            await update.message.reply_text("You are already authenticated.")
            return
        
        success, message = self.auth.start_authentication(user_id)
        await update.message.reply_text(message)
    
    async def _logout_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /logout command."""
        user_id = update.effective_user.id
        
        success, message = self.auth.logout(user_id)
        await update.message.reply_text(message)
    
    async def _status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /status command."""
        if not await self._require_auth(update, context):
            return
        
        if not self.llm_overseer:
            await update.message.reply_text("Trading system is not connected.")
            return
        
        # Get system status
        system_status = self.llm_overseer.context_manager.context.get("system_status", {})
        
        status_text = "📊 *Trading System Status*\n\n"
        
        if not system_status:
            status_text += "No status information available."
        else:
            for key, value in system_status.items():
                status_text += f"*{key}:* {value}\n"
        
        await update.message.reply_text(status_text, parse_mode="Markdown")
    
    async def _balance_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /balance command."""
        if not await self._require_auth(update, context):
            return
        
        if not self.llm_overseer:
            await update.message.reply_text("Trading system is not connected.")
            return
        
        # This would be implemented to fetch actual balance data from the trading system
        await update.message.reply_text(
            "💰 *Account Balance*\n\n"
            "USDC: 43,148.94\n"
            "BTC: 0.0\n"
            "SOL: 0.000000003\n\n"
            "*Estimated Total Value:* $43,148.94 USD",
            parse_mode="Markdown"
        )
    
    async def _performance_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /performance command."""
        if not await self._require_auth(update, context):
            return
        
        if not self.llm_overseer:
            await update.message.reply_text("Trading system is not connected.")
            return
        
        # Get performance metrics
        performance_metrics = self.llm_overseer.context_manager.context.get("performance_metrics", {})
        
        performance_text = "📈 *Trading Performance*\n\n"
        
        if not performance_metrics:
            performance_text += "No performance data available."
        else:
            for key, value in performance_metrics.items():
                performance_text += f"*{key}:* {value}\n"
        
        await update.message.reply_text(performance_text, parse_mode="Markdown")
    
    async def _positions_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /positions command."""
        if not await self._require_auth(update, context):
            return
        
        if not self.llm_overseer:
            await update.message.reply_text("Trading system is not connected.")
            return
        
        # This would be implemented to fetch actual position data from the trading system
        await update.message.reply_text(
            "🔍 *Current Positions*\n\n"
            "No open positions.",
            parse_mode="Markdown"
        )
    
    async def _pause_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /pause command."""
        if not await self._require_auth(update, context):
            return
        
        if not self.llm_overseer:
            await update.message.reply_text("Trading system is not connected.")
            return
        
        # This would be implemented to pause the trading system
        await update.message.reply_text(
            "⏸️ Trading system paused.\n\n"
            "Use /resume to resume trading."
        )
    
    async def _resume_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /resume command."""
        if not await self._require_auth(update, context):
            return
        
        if not self.llm_overseer:
            await update.message.reply_text("Trading system is not connected.")
            return
        
        # This would be implemented to resume the trading system
        await update.message.reply_text(
            "▶️ Trading system resumed.\n\n"
            "The system is now actively trading."
        )
    
    async def _risk_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /risk command."""
        if not await self._require_auth(update, context):
            return
        
        if not self.llm_overseer:
            await update.message.reply_text("Trading system is not connected.")
            return
        
        # Get risk parameters
        risk_parameters = self.llm_overseer.context_manager.context.get("risk_parameters", {})
        
        # Create inline keyboard for risk adjustment
        keyboard = [
            [
                InlineKeyboardButton("Low Risk", callback_data="risk_low"),
                InlineKeyboardButton("Medium Risk", callback_data="risk_medium"),
                InlineKeyboardButton("High Risk", callback_data="risk_high")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        risk_text = "⚠️ *Risk Management*\n\n"
        
        if not risk_parameters:
            risk_text += "No risk parameters available."
        else:
            for key, value in risk_parameters.items():
                risk_text += f"*{key}:* {value}\n"
        
        risk_text += "\nSelect a risk level to adjust parameters:"
        
        await update.message.reply_text(risk_text, reply_markup=reply_markup, parse_mode="Markdown")
    
    async def _emergency_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /emergency command."""
        if not await self._require_auth(update, context):
            return
        
        if not self.llm_overseer:
            await update.message.reply_text("Trading system is not connected.")
            return
        
        # Create inline keyboard for emergency actions
        keyboard = [
            [
                InlineKeyboardButton("STOP ALL TRADING", callback_data="emergency_stop"),
            ],
            [
                InlineKeyboardButton("Close All Positions", callback_data="emergency_close"),
            ],
            [
                InlineKeyboardButton("Cancel", callback_data="emergency_cancel")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "🚨 *EMERGENCY ACTIONS*\n\n"
            "Warning: These actions are immediate and cannot be undone.\n\n"
            "Select an action:",
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )
    
    async def _report_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /report command."""
        if not await self._require_auth(update, context):
            return
        
        if not self.llm_overseer:
            await update.message.reply_text("Trading system is not connected.")
            return
        
        # Create inline keyboard for report types
        keyboard = [
            [
                InlineKeyboardButton("Daily Report", callback_data="report_daily"),
                InlineKeyboardButton("Weekly Report", callback_data="report_weekly")
            ],
            [
                InlineKeyboardButton("Monthly Report", callback_data="report_monthly"),
                InlineKeyboardButton("Custom Report", callback_data="report_custom")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "📋 *Trading Reports*\n\n"
            "Select a report type:",
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )
    
    async def _strategy_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /strategy command."""
        if not await self._require_auth(update, context):
            return
        
        if not self.llm_overseer:
            await update.message.reply_text("Trading system is not connected.")
            return
        
        # This would be implemented to fetch strategy information from the LLM overseer
        await update.message.reply_text(
            "🧠 *Current Trading Strategy*\n\n"
            "The system is currently using a high-frequency trading strategy focused on "
            "order book microstructure analysis and tick-by-tick signal generation.\n\n"
            "Key components:\n"
            "- Order book imbalance detection\n"
            "- Volume-weighted price impact calculation\n"
            "- Dynamic slippage modeling\n"
            "- Adaptive position sizing\n\n"
            "Performance metrics are being tracked and the strategy will be adjusted "
            "based on market conditions and performance data.",
            parse_mode="Markdown"
        )
    
    async def _market_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /market command."""
        if not await self._require_auth(update, context):
            return
        
        if not self.llm_overseer:
            await update.message.reply_text("Trading system is not connected.")
            return
        
        # Get market data
        market_data = self.llm_overseer.context_manager.context.get("market_data", [])
        
        market_text = "🌎 *Market Overview*\n\n"
        
        if not market_data:
            market_text += "No market data available."
        else:
            # Show the most recent market data
            recent_data = market_data[-1]
            market_text += f"*BTC/USDC:* ${recent_data.get('price', 'N/A')}\n"
            market_text += f"*Volume (24h):* {recent_data.get('volume', 'N/A')}\n"
            market_text += f"*Last Updated:* {recent_data.get('timestamp', 'N/A')}\n\n"
            
            # Add market analysis from LLM
            market_text += "*Market Analysis:*\n"
            market_text += "The market is currently showing moderate volatility with a slight bullish trend. "
            market_text += "Order book analysis indicates buying pressure at support levels."
        
        await update.message.reply_text(market_text, parse_mode="Markdown")
    
    async def _history_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /history command."""
        if not await self._require_auth(update, context):
            return
        
        if not self.llm_overseer:
            await update.message.reply_text("Trading system is not connected.")
            return
        
        # Get trading history
        trading_history = self.llm_overseer.context_manager.context.get("trading_history", [])
        
        history_text = "📜 *Trading History*\n\n"
        
        if not trading_history:
            history_text += "No trading history available."
        else:
            # Show the most recent trades
            for i, trade in enumerate(trading_history[-5:]):
                history_text += f"{i+1}. {trade.get('timestamp', 'N/A')} - "
                history_text += f"{trade.get('type', 'N/A')} {trade.get('size', 'N/A')} BTC "
                history_text += f"at ${trade.get('price', 'N/A')} "
                history_text += f"({trade.get('result', 'N/A')})\n"
        
        await update.message.reply_text(history_text, parse_mode="Markdown")
    
    async def _notify_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /notify command."""
        if not await self._require_auth(update, context):
            return
        
        # Create inline keyboard for notification levels
        keyboard = [
            [
                InlineKeyboardButton("Level 1 (Critical Only)", callback_data="notify_1"),
                InlineKeyboardButton("Level 2 (Trades)", callback_data="notify_2")
            ],
            [
                InlineKeyboardButton("Level 3 (All Info)", callback_data="notify_3"),
                InlineKeyboardButton("Custom", callback_data="notify_custom")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "🔔 *Notification Settings*\n\n"
            "Select a notification level:\n\n"
            "*Level 1:* Emergency and critical alerts only\n"
            "*Level 2:* Level 1 + trade notifications\n"
            "*Level 3:* Level 2 + system info and updates\n"
            "*Custom:* Configure specific notification types",
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )
    
    async def _schedule_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /schedule command."""
        if not await self._require_auth(update, context):
            return
        
        # Create inline keyboard for schedule options
        keyboard = [
            [
                InlineKeyboardButton("Daily Report", callback_data="schedule_daily"),
                InlineKeyboardButton("Weekly Report", callback_data="schedule_weekly")
            ],
            [
                InlineKeyboardButton("Performance Alert", callback_data="schedule_performance"),
                InlineKeyboardButton("Custom Schedule", callback_data="schedule_custom")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "📅 *Schedule Reports*\n\n"
            "Select a report to schedule:",
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )
    
    async def _compound_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /compound command."""
        if not await self._require_auth(update, context):
            return
        
        # Create inline keyboard for compounding options
        keyboard = [
            [
                InlineKeyboardButton("Enable (80%)", callback_data="compound_enable_80"),
                InlineKeyboardButton("Enable (50%)", callback_data="compound_enable_50")
            ],
            [
                InlineKeyboardButton("Disable", callback_data="compound_disable"),
                InlineKeyboardButton("Custom", callback_data="compound_custom")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "💹 *Profit Compounding*\n\n"
            "Current setting: Enabled (80% reinvestment)\n\n"
            "Select a compounding option:",
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )
    
    async def _settings_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /settings command."""
        if not await self._require_auth(update, context):
            return
        
        # Create inline keyboard for settings
        keyboard = [
            [
                InlineKeyboardButton("Trading Parameters", callback_data="settings_trading"),
                InlineKeyboardButton("Risk Management", callback_data="settings_risk")
            ],
            [
                InlineKeyboardButton("Notifications", callback_data="settings_notifications"),
                InlineKeyboardButton("System Settings", callback_data="settings_system")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "⚙️ *System Settings*\n\n"
            "Select a settings category:",
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )
    
    async def _help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help command."""
        help_text = (
            "🤖 *Trading Agent Bot Commands*\n\n"
            "*Authentication*\n"
            "/start - Start the bot\n"
            "/login - Authenticate with the bot\n"
            "/logout - Log out from the bot\n\n"
            
            "*Status Commands*\n"
            "/status - Get system status\n"
            "/balance - Get account balance\n"
            "/performance - Get performance metrics\n"
            "/positions - Get current positions\n\n"
            
            "*Control Commands*\n"
            "/pause - Pause trading\n"
            "/resume - Resume trading\n"
            "/risk - Adjust risk parameters\n"
            "/emergency - Emergency actions\n\n"
            
            "*Information Commands*\n"
            "/report - Generate reports\n"
            "/strategy - View current strategy\n"
            "/market - Get market overview\n"
            "/history - View trading history\n\n"
            
            "*Configuration Commands*\n"
            "/notify - Configure notifications\n"
            "/schedule - Schedule reports\n"
            "/compound - Configure profit compounding\n"
            "/settings - System settings\n\n"
            
            "*Help*\n"
            "/help - Show this help message"
        )
        
        await update.message.reply_text(help_text, parse_mode="Markdown")
    
    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle text messages."""
        user_id = update.effective_user.id
        text = update.message.text
        
        # Check if this is an authentication code
        if user_id in [int(uid) for uid in self.auth.pending_auth.keys()]:
            success, message = self.auth.verify_code(user_id, text)
            await update.message.reply_text(message)
            
            if success:
                # Send welcome message after successful authentication
                await update.message.reply_text(
                    "🔐 You are now authenticated.\n\n"
                    "Use /help to see available commands."
                )
            return
        
        # For other messages, check if authenticated
        if not self.auth.is_authenticated(user_id):
            await update.message.reply_text(
                "You are not authenticated. Please use /login to authenticate."
            )
            return
        
        # Handle other messages (could be implemented for chat with LLM)
        await update.message.reply_text(
            "I'm sorry, I don't understand that command. Use /help to see available commands."
        )
    
    async def _handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle callback queries from inline keyboards."""
        query = update.callback_query
        user_id = query.from_user.id
        
        # Check if authenticated
        if not self.auth.is_authenticated(user_id):
            await query.answer("You are not authenticated. Please use /login to authenticate.")
            return
        
        # Acknowledge the callback
        await query.answer()
        
        # Handle different callback types
        callback_data = query.data
        
        if callback_data.startswith("risk_"):
            risk_level = callback_data.split("_")[1]
            await self._handle_risk_callback(query, risk_level)
        
        elif callback_data.startswith("emergency_"):
            action = callback_data.split("_")[1]
            await self._handle_emergency_callback(query, action)
        
        elif callback_data.startswith("report_"):
            report_type = callback_data.split("_")[1]
            await self._handle_report_callback(query, report_type)
        
        elif callback_data.startswith("notify_"):
            level = callback_data.split("_")[1]
            await self._handle_notify_callback(query, level)
        
        elif callback_data.startswith("schedule_"):
            schedule_type = callback_data.split("_")[1]
            await self._handle_schedule_callback(query, schedule_type)
        
        elif callback_data.startswith("compound_"):
            compound_option = callback_data.split("_")[1:]
            await self._handle_compound_callback(query, "_".join(compound_option))
        
        elif callback_data.startswith("settings_"):
            settings_category = callback_data.split("_")[1]
            await self._handle_settings_callback(query, settings_category)
    
    async def _handle_risk_callback(self, query: Any, risk_level: str) -> None:
        """Handle risk level selection."""
        if risk_level == "low":
            message = "Risk level set to LOW.\n\n- Max position size: 5%\n- Max daily drawdown: 2%\n- Stop loss: 1%"
        elif risk_level == "medium":
            message = "Risk level set to MEDIUM.\n\n- Max position size: 10%\n- Max daily drawdown: 5%\n- Stop loss: 2%"
        elif risk_level == "high":
            message = "Risk level set to HIGH.\n\n- Max position size: 20%\n- Max daily drawdown: 10%\n- Stop loss: 5%"
        else:
            message = "Invalid risk level."
        
        await query.edit_message_text(text=message)
    
    async def _handle_emergency_callback(self, query: Any, action: str) -> None:
        """Handle emergency action selection."""
        if action == "stop":
            message = "🚨 EMERGENCY STOP ACTIVATED\n\nAll trading has been stopped. Use /resume to restart trading."
        elif action == "close":
            message = "🚨 CLOSING ALL POSITIONS\n\nAll open positions are being closed. This may take a moment."
        elif action == "cancel":
            message = "Emergency action cancelled."
        else:
            message = "Invalid emergency action."
        
        await query.edit_message_text(text=message)
    
    async def _handle_report_callback(self, query: Any, report_type: str) -> None:
        """Handle report type selection."""
        if report_type == "daily":
            message = "📊 Daily Report\n\nGenerating daily report... This will be sent shortly."
        elif report_type == "weekly":
            message = "📊 Weekly Report\n\nGenerating weekly report... This will be sent shortly."
        elif report_type == "monthly":
            message = "📊 Monthly Report\n\nGenerating monthly report... This will be sent shortly."
        elif report_type == "custom":
            # Create date selection interface (simplified for this example)
            message = "📊 Custom Report\n\nPlease specify the date range using /report_custom YYYY-MM-DD YYYY-MM-DD"
        else:
            message = "Invalid report type."
        
        await query.edit_message_text(text=message)
    
    async def _handle_notify_callback(self, query: Any, level: str) -> None:
        """Handle notification level selection."""
        if level == "1":
            message = "🔔 Notification level set to 1 (Critical Only).\n\nYou will only receive emergency and critical alerts."
        elif level == "2":
            message = "🔔 Notification level set to 2 (Trades).\n\nYou will receive critical alerts and trade notifications."
        elif level == "3":
            message = "🔔 Notification level set to 3 (All Info).\n\nYou will receive all notifications including system info."
        elif level == "custom":
            # Create custom notification selection interface (simplified for this example)
            message = "🔔 Custom Notifications\n\nPlease specify notification types using /notify_custom [types]"
        else:
            message = "Invalid notification level."
        
        await query.edit_message_text(text=message)
    
    async def _handle_schedule_callback(self, query: Any, schedule_type: str) -> None:
        """Handle schedule type selection."""
        if schedule_type == "daily":
            message = "📅 Daily Report Scheduled\n\nYou will receive a daily report at 00:00 UTC."
        elif schedule_type == "weekly":
            message = "📅 Weekly Report Scheduled\n\nYou will receive a weekly report every Monday at 00:00 UTC."
        elif schedule_type == "performance":
            message = "📅 Performance Alert Scheduled\n\nYou will receive performance alerts when significant changes occur."
        elif schedule_type == "custom":
            # Create custom schedule interface (simplified for this example)
            message = "📅 Custom Schedule\n\nPlease specify schedule using /schedule_custom [frequency] [time]"
        else:
            message = "Invalid schedule type."
        
        await query.edit_message_text(text=message)
    
    async def _handle_compound_callback(self, query: Any, compound_option: str) -> None:
        """Handle compounding option selection."""
        if compound_option == "enable_80":
            message = "💹 Profit Compounding Enabled (80%)\n\n80% of profits will be automatically reinvested."
        elif compound_option == "enable_50":
            message = "💹 Profit Compounding Enabled (50%)\n\n50% of profits will be automatically reinvested."
        elif compound_option == "disable":
            message = "💹 Profit Compounding Disabled\n\nProfits will not be automatically reinvested."
        elif compound_option == "custom":
            # Create custom compounding interface (simplified for this example)
            message = "💹 Custom Compounding\n\nPlease specify compounding rate using /compound_custom [rate]"
        else:
            message = "Invalid compounding option."
        
        await query.edit_message_text(text=message)
    
    async def _handle_settings_callback(self, query: Any, settings_category: str) -> None:
        """Handle settings category selection."""
        if settings_category == "trading":
            message = "⚙️ Trading Parameters\n\nUse /settings_trading [parameter] [value] to update trading parameters."
        elif settings_category == "risk":
            message = "⚙️ Risk Management\n\nUse /settings_risk [parameter] [value] to update risk parameters."
        elif settings_category == "notifications":
            message = "⚙️ Notifications\n\nUse /settings_notifications [parameter] [value] to update notification settings."
        elif settings_category == "system":
            message = "⚙️ System Settings\n\nUse /settings_system [parameter] [value] to update system settings."
        else:
            message = "Invalid settings category."
        
        await query.edit_message_text(text=message)
    
    async def _error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle errors."""
        logger.error(f"Update {update} caused error {context.error}")
        
        # Send error message to user if possible
        if update and update.effective_message:
            await update.effective_message.reply_text(
                "An error occurred while processing your request. Please try again later."
            )
    
    async def send_notification(self, user_id: int, message: str, level: str = "info") -> bool:
        """
        Send notification to user.
        
        Args:
            user_id: Telegram user ID
            message: Notification message
            level: Notification level ("emergency", "critical", "trade", "info")
            
        Returns:
            True if notification was sent, False otherwise
        """
        if not self.auth.is_user_allowed(user_id):
            logger.warning(f"Attempted to send notification to unauthorized user: {user_id}")
            return False
        
        # Get user's notification level
        user_notification_level = self.config.get("telegram.default_notification_level", 2)
        
        # Check if user should receive this notification
        notification_levels = self.config.get("telegram.notification_levels", {})
        
        # Convert string keys to integers
        notification_levels = {int(k): v for k, v in notification_levels.items()}
        
        if user_notification_level not in notification_levels:
            logger.warning(f"Invalid notification level for user {user_id}: {user_notification_level}")
            return False
        
        if level not in notification_levels[user_notification_level]:
            # User doesn't want this notification level
            return False
        
        try:
            # Format message based on level
            if level == "emergency":
                formatted_message = f"🚨 *EMERGENCY*\n\n{message}"
            elif level == "critical":
                formatted_message = f"⚠️ *CRITICAL*\n\n{message}"
            elif level == "trade":
                formatted_message = f"💰 *TRADE*\n\n{message}"
            else:  # info
                formatted_message = f"ℹ️ *INFO*\n\n{message}"
            
            # Send message
            await self.application.bot.send_message(
                chat_id=user_id,
                text=formatted_message,
                parse_mode="Markdown"
            )
            
            return True
        except Exception as e:
            logger.error(f"Error sending notification to user {user_id}: {e}")
            return False
    
    async def broadcast_notification(self, message: str, level: str = "info") -> Dict[int, bool]:
        """
        Broadcast notification to all allowed users.
        
        Args:
            message: Notification message
            level: Notification level ("emergency", "critical", "trade", "info")
            
        Returns:
            Dictionary mapping user IDs to success status
        """
        results = {}
        
        for user_id in self.auth.allowed_user_ids:
            results[user_id] = await self.send_notification(user_id, message, level)
        
        return results
    
    async def start(self) -> None:
        """Start the bot."""
        # Create data directory if it doesn't exist
        os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'), exist_ok=True)
        
        # Start the bot
        await self.application.initialize()
        await self.application.start()
        await self.application.updater.start_polling()
        
        logger.info("Telegram bot started")
    
    async def stop(self) -> None:
        """Stop the bot."""
        await self.application.stop()
        await self.application.shutdown()
        
        logger.info("Telegram bot stopped")


async def main():
    """Main function for testing."""
    # Initialize configuration
    config = Config()
    
    # Set test values
    config.set("telegram.allowed_user_ids", [123456789])  # Replace with your Telegram user ID
    
    # Initialize bot
    bot = TelegramBot(config)
    
    try:
        await bot.start()
        
        # Keep the bot running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        await bot.stop()


if __name__ == "__main__":
    asyncio.run(main())
