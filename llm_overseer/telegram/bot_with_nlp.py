#!/usr/bin/env python
"""
Enhanced Telegram bot module with natural language API key management.
This module integrates the natural language handler with the Telegram bot.
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
from .natural_language_handler import NaturalLanguageHandler

logger = logging.getLogger(__name__)

class TelegramBotWithNLP:
    """
    Enhanced Telegram bot with natural language API key management.
    Provides a user-friendly interface for managing API keys.
    """
    
    def __init__(self, config: Config, llm_overseer=None):
        """
        Initialize Telegram bot with natural language processing.
        
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
        
        # Initialize natural language handler
        self.nlp_handler = NaturalLanguageHandler(self.key_manager)
        
        # Initialize application
        self.application = Application.builder().token(self.bot_token).build()
        
        # Register handlers
        self._register_handlers()
        
        logger.info("Telegram bot with natural language processing initialized")
    
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
        
        # Help command
        self.application.add_handler(CommandHandler("help", self._help_command))
        
        # Callback query handler
        self.application.add_handler(CallbackQueryHandler(self._button_callback))
        
        # Message handler (for natural language processing)
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
            f"I can help you manage your API keys securely using natural language.\n\n"
            f"You can simply tell me what you want to do in plain English. For example:\n"
            f"- \"Set my MEXC API key to abc123\"\n"
            f"- \"What's my OpenRouter API key?\"\n"
            f"- \"List all my keys\"\n"
            f"- \"Delete my Telegram bot token\"\n\n"
            f"Or you can use the buttons below to get started:"
        )
        
        # Create keyboard with options
        keyboard = [
            [InlineKeyboardButton("Set a key", callback_data="flow_set_key")],
            [InlineKeyboardButton("Get a key", callback_data="flow_get_key")],
            [InlineKeyboardButton("List all keys", callback_data="flow_list_keys")],
            [InlineKeyboardButton("Delete a key", callback_data="flow_delete_key")]
        ]
        
        await update.message.reply_text(
            welcome_text,
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    
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
    
    async def _help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help command."""
        user_id = update.effective_user.id
        
        # Check if user is allowed
        if not self.auth.is_user_allowed(user_id):
            await update.message.reply_text("You are not authorized to use this bot.")
            return
        
        # Help message
        help_text = (
            "I can help you manage your API keys securely using natural language.\n\n"
            "Here's what you can ask me to do:\n\n"
            "To set a key:\n"
            "- \"Set my MEXC API key to abc123\"\n"
            "- \"My OpenRouter API key is xyz789\"\n"
            "- \"Update Telegram bot token to 123:ABC\"\n\n"
            "To get a key:\n"
            "- \"What's my MEXC API key?\"\n"
            "- \"Show me my OpenRouter API key\"\n"
            "- \"Get my Telegram chat ID\"\n\n"
            "To list all keys:\n"
            "- \"List all my keys\"\n"
            "- \"What keys do I have?\"\n"
            "- \"Show me my available keys\"\n\n"
            "To delete a key:\n"
            "- \"Delete my MEXC API key\"\n"
            "- \"Remove my OpenRouter API key\"\n\n"
            "Or you can use the buttons below to get started:"
        )
        
        # Create keyboard with options
        keyboard = [
            [InlineKeyboardButton("Set a key", callback_data="flow_set_key")],
            [InlineKeyboardButton("Get a key", callback_data="flow_get_key")],
            [InlineKeyboardButton("List all keys", callback_data="flow_list_keys")],
            [InlineKeyboardButton("Delete a key", callback_data="flow_delete_key")]
        ]
        
        await update.message.reply_text(
            help_text,
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    
    async def _message_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle text messages with natural language processing."""
        user_id = update.effective_user.id
        text = update.message.text
        
        # Check if user is allowed
        if not self.auth.is_user_allowed(user_id):
            await update.message.reply_text("You are not authorized to use this bot.")
            return
        
        # Check if user is authenticated
        if not self.auth.is_authenticated(user_id):
            # Check if this is an authentication code
            if len(text) == 6 and text.isalnum():
                # Verify code
                success, message = self.auth.verify_code(user_id, text)
                await update.message.reply_text(message)
                
                # If authentication successful, show welcome message
                if success:
                    await self._start_command(update, context)
            else:
                await update.message.reply_text("You need to authenticate first. Use /login to authenticate.")
            return
        
        # Refresh session
        self.auth.refresh_session(user_id)
        
        # Process message with natural language handler
        response_text, reply_markup = await self.nlp_handler.process_message(update, context)
        
        # Send response
        await update.message.reply_text(
            response_text,
            reply_markup=reply_markup
        )
    
    async def _button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle button callbacks."""
        query = update.callback_query
        user_id = query.from_user.id
        
        # Check if user is allowed
        if not self.auth.is_user_allowed(user_id):
            await query.answer("You are not authorized to use this bot.")
            await query.edit_message_text("You are not authorized to use this bot.")
            return
        
        # Check if user is authenticated
        if not self.auth.is_authenticated(user_id):
            await query.answer("You need to authenticate first.")
            await query.edit_message_text("Authentication required. Use /login to authenticate.")
            return
        
        # Refresh session
        self.auth.refresh_session(user_id)
        
        # Answer callback query
        await query.answer()
        
        # Process callback with natural language handler
        response_text, reply_markup = await self.nlp_handler.process_callback(update, context)
        
        # Edit message with response
        await query.edit_message_text(
            response_text,
            reply_markup=reply_markup
        )
    
    async def _error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle errors."""
        logger.error(f"Update {update} caused error: {context.error}")
        
        # Send error message if update is available
        if update and isinstance(update, Update) and update.effective_message:
            await update.effective_message.reply_text("An error occurred. Please try again later.")
    
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
