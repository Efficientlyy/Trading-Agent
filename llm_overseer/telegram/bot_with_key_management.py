#!/usr/bin/env python
"""
Extended Telegram bot module with secure API key management features.
This module enhances the existing bot with commands for managing API keys.
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
from .key_manager import KeyManager

logger = logging.getLogger(__name__)

class TelegramBotWithKeyManagement:
    """
    Enhanced Telegram bot with secure API key management.
    Extends the original bot with commands for managing API keys.
    """
    
    def __init__(self, config: Config, llm_overseer=None):
        """
        Initialize Telegram bot with key management.
        
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
        
        # Initialize key manager
        self.key_manager = KeyManager(self.bot_token)
        
        # Initialize application
        self.application = Application.builder().token(self.bot_token).build()
        
        # Register handlers
        self._register_handlers()
        
        logger.info("Telegram bot with key management initialized")
    
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
        
        # Key management commands
        self.application.add_handler(CommandHandler("setkey", self._setkey_command))
        self.application.add_handler(CommandHandler("getkey", self._getkey_command))
        self.application.add_handler(CommandHandler("listkeys", self._listkeys_command))
        self.application.add_handler(CommandHandler("deletekey", self._deletekey_command))
        self.application.add_handler(CommandHandler("rotatekeys", self._rotatekeys_command))
        
        # Help command
        self.application.add_handler(CommandHandler("help", self._help_command))
        
        # Callback query handler
        self.application.add_handler(CallbackQueryHandler(self._button_callback))
        
        # Message handler
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._message_handler))
        
        # Error handler
        self.application.add_error_handler(self._error_handler)
    
    # Authentication command handlers
    
    async def _start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command."""
        user_id = update.effective_user.id
        
        # Check if user is allowed
        if not self.auth.is_user_allowed(user_id):
            await update.message.reply_text("You are not authorized to use this bot.")
            return
        
        # Welcome message
        welcome_text = (
            f"Welcome to the Trading Agent Bot, {update.effective_user.first_name}!\n\n"
            f"This bot provides secure management of your trading system.\n\n"
            f"Available commands:\n"
            f"/login - Authenticate with the bot\n"
            f"/status - Check system status\n"
            f"/help - Show all available commands\n\n"
            f"For API key management:\n"
            f"/setkey - Set an API key\n"
            f"/getkey - Get an API key\n"
            f"/listkeys - List all available keys\n"
            f"/deletekey - Delete an API key\n"
        )
        
        await update.message.reply_text(welcome_text)
    
    async def _login_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /login command."""
        user_id = update.effective_user.id
        
        # Start authentication
        success, message = self.auth.start_authentication(user_id)
        
        await update.message.reply_text(message)
    
    async def _logout_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /logout command."""
        user_id = update.effective_user.id
        
        # Log out
        success, message = self.auth.logout(user_id)
        
        await update.message.reply_text(message)
    
    # Key management command handlers
    
    async def _setkey_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /setkey command."""
        user_id = update.effective_user.id
        
        # Check if user is authenticated
        if not self.auth.is_authenticated(user_id):
            await update.message.reply_text("You need to authenticate first. Use /login to authenticate.")
            return
        
        # Refresh session
        self.auth.refresh_session(user_id)
        
        # Check arguments
        if not context.args or len(context.args) < 3:
            await update.message.reply_text(
                "Usage: /setkey [service] [key_type] [value]\n"
                "Example: /setkey mexc api_key your_api_key_here"
            )
            return
        
        # Extract arguments
        service = context.args[0].lower()
        key_type = context.args[1].lower()
        value = context.args[2]
        
        # Set key
        success = self.key_manager.set_key(service, key_type, value, user_id)
        
        if success:
            # Log the action
            logger.info(f"User {user_id} set {service}.{key_type}")
            
            # Send confirmation
            await update.message.reply_text(f"Successfully set {service}.{key_type}")
            
            # Delete the message containing the key for security
            await update.message.delete()
        else:
            await update.message.reply_text(f"Failed to set {service}.{key_type}")
    
    async def _getkey_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /getkey command."""
        user_id = update.effective_user.id
        
        # Check if user is authenticated
        if not self.auth.is_authenticated(user_id):
            await update.message.reply_text("You need to authenticate first. Use /login to authenticate.")
            return
        
        # Refresh session
        self.auth.refresh_session(user_id)
        
        # Check arguments
        if not context.args or len(context.args) < 2:
            await update.message.reply_text(
                "Usage: /getkey [service] [key_type]\n"
                "Example: /getkey mexc api_key"
            )
            return
        
        # Extract arguments
        service = context.args[0].lower()
        key_type = context.args[1].lower()
        
        # Get key
        value = self.key_manager.get_key(service, key_type)
        
        if value:
            # Log the action
            logger.info(f"User {user_id} retrieved {service}.{key_type}")
            
            # Send key value
            await update.message.reply_text(f"{service}.{key_type}: {value}")
        else:
            await update.message.reply_text(f"Key not found: {service}.{key_type}")
    
    async def _listkeys_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /listkeys command."""
        user_id = update.effective_user.id
        
        # Check if user is authenticated
        if not self.auth.is_authenticated(user_id):
            await update.message.reply_text("You need to authenticate first. Use /login to authenticate.")
            return
        
        # Refresh session
        self.auth.refresh_session(user_id)
        
        # List keys
        keys = self.key_manager.list_keys()
        
        if keys:
            # Format keys
            keys_text = "Available keys:\n\n"
            for service, key_types in keys.items():
                keys_text += f"Service: {service}\n"
                for key_type in key_types:
                    keys_text += f"  - {key_type}\n"
                keys_text += "\n"
            
            # Log the action
            logger.info(f"User {user_id} listed keys")
            
            # Send keys
            await update.message.reply_text(keys_text)
        else:
            await update.message.reply_text("No keys found.")
    
    async def _deletekey_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /deletekey command."""
        user_id = update.effective_user.id
        
        # Check if user is authenticated
        if not self.auth.is_authenticated(user_id):
            await update.message.reply_text("You need to authenticate first. Use /login to authenticate.")
            return
        
        # Refresh session
        self.auth.refresh_session(user_id)
        
        # Check arguments
        if not context.args or len(context.args) < 2:
            await update.message.reply_text(
                "Usage: /deletekey [service] [key_type]\n"
                "Example: /deletekey mexc api_key"
            )
            return
        
        # Extract arguments
        service = context.args[0].lower()
        key_type = context.args[1].lower()
        
        # Create confirmation keyboard
        keyboard = [
            [
                InlineKeyboardButton("Yes", callback_data=f"deletekey_yes_{service}_{key_type}"),
                InlineKeyboardButton("No", callback_data="deletekey_no")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # Send confirmation message
        await update.message.reply_text(
            f"Are you sure you want to delete {service}.{key_type}?",
            reply_markup=reply_markup
        )
    
    async def _rotatekeys_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /rotatekeys command."""
        user_id = update.effective_user.id
        
        # Check if user is authenticated
        if not self.auth.is_authenticated(user_id):
            await update.message.reply_text("You need to authenticate first. Use /login to authenticate.")
            return
        
        # Refresh session
        self.auth.refresh_session(user_id)
        
        # Create confirmation keyboard
        keyboard = [
            [
                InlineKeyboardButton("Yes", callback_data="rotatekeys_yes"),
                InlineKeyboardButton("No", callback_data="rotatekeys_no")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # Send confirmation message
        await update.message.reply_text(
            "Are you sure you want to rotate all keys? This will re-encrypt all keys with the current bot token.",
            reply_markup=reply_markup
        )
    
    async def _help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help command."""
        user_id = update.effective_user.id
        
        # Check if user is allowed
        if not self.auth.is_user_allowed(user_id):
            await update.message.reply_text("You are not authorized to use this bot.")
            return
        
        # Help message
        help_text = (
            "Available commands:\n\n"
            "Authentication:\n"
            "/start - Start the bot\n"
            "/login - Authenticate with the bot\n"
            "/logout - Log out from the bot\n\n"
            "Status:\n"
            "/status - Check system status\n"
            "/balance - Check account balance\n"
            "/performance - Check performance metrics\n"
            "/positions - Check open positions\n\n"
            "Control:\n"
            "/pause - Pause trading\n"
            "/resume - Resume trading\n"
            "/risk - Adjust risk parameters\n"
            "/emergency - Emergency stop\n\n"
            "Information:\n"
            "/report - Generate report\n"
            "/strategy - View current strategy\n"
            "/market - View market analysis\n"
            "/history - View trading history\n\n"
            "Configuration:\n"
            "/notify - Configure notifications\n"
            "/schedule - Configure trading schedule\n\n"
            "Key Management:\n"
            "/setkey [service] [key_type] [value] - Set an API key\n"
            "/getkey [service] [key_type] - Get an API key\n"
            "/listkeys - List all available keys\n"
            "/deletekey [service] [key_type] - Delete an API key\n"
            "/rotatekeys - Re-encrypt all keys\n\n"
            "/help - Show this help message"
        )
        
        await update.message.reply_text(help_text)
    
    async def _button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle button callbacks."""
        query = update.callback_query
        user_id = query.from_user.id
        
        # Check if user is authenticated
        if not self.auth.is_authenticated(user_id):
            await query.answer("You need to authenticate first.")
            await query.edit_message_text("Authentication required. Use /login to authenticate.")
            return
        
        # Refresh session
        self.auth.refresh_session(user_id)
        
        # Answer callback query
        await query.answer()
        
        # Handle callback data
        data = query.data
        
        if data.startswith("deletekey_yes_"):
            # Extract service and key_type
            _, _, service, key_type = data.split("_", 3)
            
            # Delete key
            success = self.key_manager.delete_key(service, key_type)
            
            if success:
                # Log the action
                logger.info(f"User {user_id} deleted {service}.{key_type}")
                
                # Send confirmation
                await query.edit_message_text(f"Successfully deleted {service}.{key_type}")
            else:
                await query.edit_message_text(f"Failed to delete {service}.{key_type}")
                
        elif data == "deletekey_no":
            await query.edit_message_text("Key deletion cancelled.")
            
        elif data == "rotatekeys_yes":
            # Rotate keys
            success = self.key_manager.rotate_keys(self.bot_token, user_id)
            
            if success:
                # Log the action
                logger.info(f"User {user_id} rotated keys")
                
                # Send confirmation
                await query.edit_message_text("Successfully rotated all keys.")
            else:
                await query.edit_message_text("Failed to rotate keys.")
                
        elif data == "rotatekeys_no":
            await query.edit_message_text("Key rotation cancelled.")
    
    async def _message_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle text messages."""
        user_id = update.effective_user.id
        text = update.message.text
        
        # Check if user is authenticated
        if not self.auth.is_authenticated(user_id):
            # Check if this is an authentication code
            if len(text) == 6 and text.isalnum():
                # Verify code
                success, message = self.auth.verify_code(user_id, text)
                await update.message.reply_text(message)
            return
        
        # Refresh session
        self.auth.refresh_session(user_id)
        
        # Process message
        await update.message.reply_text("I don't understand that command. Use /help to see available commands.")
    
    async def _error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle errors."""
        logger.error(f"Update {update} caused error: {context.error}")
        
        # Send error message if update is available
        if update and isinstance(update, Update) and update.effective_message:
            await update.effective_message.reply_text("An error occurred. Please try again later.")
    
    # Placeholder methods for other commands
    # These would be implemented in the actual bot
    
    async def _status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /status command."""
        pass
    
    async def _balance_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /balance command."""
        pass
    
    async def _performance_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /performance command."""
        pass
    
    async def _positions_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /positions command."""
        pass
    
    async def _pause_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /pause command."""
        pass
    
    async def _resume_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /resume command."""
        pass
    
    async def _risk_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /risk command."""
        pass
    
    async def _emergency_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /emergency command."""
        pass
    
    async def _report_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /report command."""
        pass
    
    async def _strategy_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /strategy command."""
        pass
    
    async def _market_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /market command."""
        pass
    
    async def _history_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /history command."""
        pass
    
    async def _notify_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /notify command."""
        pass
    
    async def _schedule_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /schedule command."""
        pass
    
    async def run(self) -> None:
        """Run the bot."""
        await self.application.initialize()
        await self.application.start()
        await self.application.updater.start_polling()
        
        logger.info("Bot started")
        
        # Keep the bot running
        try:
            await self.application.updater.stop_polling()
            await self.application.stop()
        finally:
            await self.application.shutdown()
