#!/usr/bin/env python
"""
LLM-powered Telegram bot for natural language API key management.
This module integrates LLM for true AI-powered natural language understanding.
"""
import os
import sys
import json
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
from .llm_manager import LLMManager
from .context_manager import ContextManager

logger = logging.getLogger(__name__)

class LLMPoweredTelegramBot:
    """
    LLM-powered Telegram bot for natural language API key management.
    Provides a true AI-powered interface for managing API keys.
    """
    
    def __init__(self, config: Config, llm_overseer=None):
        """
        Initialize LLM-powered Telegram bot.
        
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
        
        # Initialize LLM manager
        self.llm_manager = LLMManager(key_manager=self.key_manager)
        
        # Initialize context manager
        self.context_manager = ContextManager()
        
        # Initialize application
        self.application = Application.builder().token(self.bot_token).build()
        
        # Register handlers
        self._register_handlers()
        
        logger.info("LLM-powered Telegram bot initialized")
    
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
        
        # Reset conversation command
        self.application.add_handler(CommandHandler("reset", self._reset_command))
        
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
            f"I'm powered by AI to help you manage your API keys securely using natural language.\n\n"
            f"You can simply tell me what you want to do in plain English. For example:\n"
            f"- \"Set my MEXC API key to abc123\"\n"
            f"- \"What's my OpenRouter API key?\"\n"
            f"- \"List all my keys\"\n"
            f"- \"Delete my Telegram bot token\"\n\n"
            f"I understand natural language, so feel free to ask in your own words!"
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
        
        # Clear conversation
        self.context_manager.clear_conversation(user_id)
        
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
            "I'm an AI-powered assistant that helps you manage your API keys securely using natural language.\n\n"
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
            "Other commands:\n"
            "/login - Authenticate with the bot\n"
            "/logout - Log out and end your session\n"
            "/reset - Reset the conversation history\n"
            "/help - Show this help message"
        )
        
        await update.message.reply_text(help_text)
    
    async def _reset_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /reset command."""
        user_id = update.effective_user.id
        
        # Check if user is allowed
        if not self.auth.is_user_allowed(user_id):
            await update.message.reply_text("You are not authorized to use this bot.")
            return
        
        # Check if user is authenticated
        if not self.auth.is_authenticated(user_id):
            await update.message.reply_text("You need to authenticate first. Use /login to authenticate.")
            return
        
        # Clear conversation history
        self.context_manager.clear_history(user_id)
        
        await update.message.reply_text("Conversation history has been reset.")
    
    async def _message_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle text messages with LLM processing."""
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
        
        # Add user message to context
        self.context_manager.add_message(user_id, "user", text)
        
        # Get conversation history
        history = self.context_manager.get_history(user_id)
        
        # Process message with LLM
        await update.message.reply_text("Processing your request...")
        
        # Send typing action
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
        
        # Process with LLM
        response_text, metadata = await self.llm_manager.process_message(text, history)
        
        # Add assistant response to context
        self.context_manager.add_message(user_id, "assistant", response_text)
        
        # Handle intent if metadata is available
        if metadata:
            intent = metadata.get("intent")
            service = metadata.get("service")
            key_type = metadata.get("key_type")
            value = metadata.get("value")
            requires_confirmation = metadata.get("requires_confirmation", False)
            
            # Update conversation state
            self.context_manager.update_state(user_id, {
                "intent": intent,
                "service": service,
                "key_type": key_type,
                "value": value,
                "requires_confirmation": requires_confirmation
            })
            
            # Handle intent
            if intent == "set_key" and service and key_type and value:
                if requires_confirmation:
                    # Create confirmation keyboard
                    keyboard = [
                        [
                            InlineKeyboardButton("Yes", callback_data=f"confirm_set_{service}_{key_type}"),
                            InlineKeyboardButton("No", callback_data="cancel_set")
                        ]
                    ]
                    
                    # Send response with confirmation keyboard
                    await update.message.reply_text(
                        response_text,
                        reply_markup=InlineKeyboardMarkup(keyboard)
                    )
                    return
                else:
                    # Set key directly
                    success = self.key_manager.set_key(service, key_type, value, user_id)
                    
                    if success:
                        # Delete the message containing the key for security
                        await update.message.delete()
                        
                        # Send success message
                        await update.message.reply_text(f"I've set your {service} {key_type.replace('_', ' ')} successfully.")
                        return
            
            elif intent == "get_key" and service and key_type:
                # Get key
                value = self.key_manager.get_key(service, key_type)
                
                if value:
                    # Send response with key value
                    await update.message.reply_text(f"Your {service} {key_type.replace('_', ' ')} is: {value}")
                    return
            
            elif intent == "list_keys":
                # List keys
                keys = self.key_manager.list_keys()
                
                if keys:
                    # Format keys
                    keys_text = "Here are your available keys:\n\n"
                    for service, key_types in keys.items():
                        keys_text += f"{service.capitalize()}:\n"
                        for key_type in key_types:
                            keys_text += f"  - {key_type.replace('_', ' ').title()}\n"
                        keys_text += "\n"
                    
                    # Send response with keys
                    await update.message.reply_text(keys_text)
                    return
            
            elif intent == "delete_key" and service and key_type:
                if requires_confirmation:
                    # Create confirmation keyboard
                    keyboard = [
                        [
                            InlineKeyboardButton("Yes", callback_data=f"confirm_delete_{service}_{key_type}"),
                            InlineKeyboardButton("No", callback_data="cancel_delete")
                        ]
                    ]
                    
                    # Send response with confirmation keyboard
                    await update.message.reply_text(
                        response_text,
                        reply_markup=InlineKeyboardMarkup(keyboard)
                    )
                    return
        
        # Send response
        await update.message.reply_text(response_text)
    
    async def _button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle button callbacks."""
        query = update.callback_query
        user_id = query.from_user.id
        data = query.data
        
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
        
        # Get conversation state
        state = self.context_manager.get_state(user_id)
        
        # Handle confirmation callbacks
        if data.startswith("confirm_set_"):
            # Extract service and key_type
            _, _, service, key_type = data.split("_", 3)
            
            # Get value from state
            value = state.get("value")
            
            if value:
                # Set key
                success = self.key_manager.set_key(service, key_type, value, user_id)
                
                if success:
                    # Clear state
                    self.context_manager.clear_state(user_id)
                    
                    # Edit message
                    await query.edit_message_text(f"I've set your {service} {key_type.replace('_', ' ')} successfully.")
                else:
                    # Edit message
                    await query.edit_message_text(f"I couldn't set your {service} {key_type.replace('_', ' ')}. Please try again.")
            else:
                # Edit message
                await query.edit_message_text("I couldn't find the key value. Please try setting the key again.")
        
        elif data.startswith("confirm_delete_"):
            # Extract service and key_type
            _, _, service, key_type = data.split("_", 3)
            
            # Delete key
            success = self.key_manager.delete_key(service, key_type)
            
            # Clear state
            self.context_manager.clear_state(user_id)
            
            if success:
                # Edit message
                await query.edit_message_text(f"I've deleted your {service} {key_type.replace('_', ' ')} successfully.")
            else:
                # Edit message
                await query.edit_message_text(f"I couldn't delete your {service} {key_type.replace('_', ' ')}. It might not exist.")
        
        elif data.startswith("cancel_"):
            # Clear state
            self.context_manager.clear_state(user_id)
            
            # Edit message
            await query.edit_message_text("Operation cancelled.")
    
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
