#!/usr/bin/env python
"""
Telegram Integration Module for LLM Overseer

This module handles the integration between the LLM Overseer and Telegram,
allowing for notifications and commands via Telegram bot.
"""

import os
import logging
import asyncio
from typing import Dict, List, Any, Optional, Callable
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Import environment configuration
import sys
sys.path.append('..')
from env_config import load_env, is_production

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TelegramIntegration:
    """
    Telegram Integration for LLM Overseer.
    
    This class handles the integration between the LLM Overseer and Telegram,
    allowing for notifications and commands via Telegram bot.
    """
    
    def __init__(self):
        """Initialize Telegram Integration."""
        # Load environment variables
        load_env()
        
        # Get Telegram credentials from environment
        self.token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        if not self.token or not self.chat_id:
            logger.error("Telegram credentials not found in environment variables")
            raise ValueError("Telegram credentials not found in environment variables")
        
        # Initialize bot
        self.bot = Bot(token=self.token)
        self.application = Application.builder().token(self.token).build()
        
        # Command handlers
        self.command_handlers = {}
        
        # Register default commands
        self.register_default_commands()
        
        logger.info("Telegram Integration initialized")
    
    def register_default_commands(self):
        """Register default command handlers."""
        self.application.add_handler(CommandHandler("start", self._start_command))
        self.application.add_handler(CommandHandler("help", self._help_command))
        self.application.add_handler(CommandHandler("status", self._status_command))
        
        # Add message handler for non-command messages
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._message_handler))
        
        logger.info("Default command handlers registered")
    
    def register_command(self, command: str, handler: Callable):
        """
        Register a custom command handler.
        
        Args:
            command: Command name (without slash)
            handler: Command handler function
        """
        self.application.add_handler(CommandHandler(command, handler))
        self.command_handlers[command] = handler
        logger.info(f"Registered command handler for /{command}")
    
    async def send_message(self, message: str) -> bool:
        """
        Send a message to the configured chat.
        
        Args:
            message: Message text to send
            
        Returns:
            True if message was sent successfully, False otherwise
        """
        try:
            await self.bot.send_message(chat_id=self.chat_id, text=message)
            return True
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return False
    
    async def send_notification(self, title: str, message: str, priority: str = "normal") -> bool:
        """
        Send a formatted notification to the configured chat.
        
        Args:
            title: Notification title
            message: Notification message
            priority: Priority level ("low", "normal", "high")
            
        Returns:
            True if notification was sent successfully, False otherwise
        """
        # Format priority emoji
        priority_emoji = "ℹ️"
        if priority == "high":
            priority_emoji = "🚨"
        elif priority == "low":
            priority_emoji = "📌"
        
        # Format notification
        formatted_message = f"{priority_emoji} *{title}*\n\n{message}"
        
        # Send notification
        return await self.send_message(formatted_message)
    
    async def start_polling(self):
        """Start polling for updates."""
        await self.application.initialize()
        await self.application.start()
        await self.application.updater.start_polling()
        
        # Send startup notification
        env_type = "Production" if is_production() else "Development"
        await self.send_notification(
            "Trading Agent Bot Started",
            f"The Trading Agent bot is now running in {env_type} mode.\n\n"
            f"Use /help to see available commands."
        )
        
        logger.info("Started polling for Telegram updates")
    
    async def stop_polling(self):
        """Stop polling for updates."""
        await self.application.updater.stop()
        await self.application.stop()
        await self.application.shutdown()
        logger.info("Stopped polling for Telegram updates")
    
    # Default command handlers
    async def _start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        await update.message.reply_text(
            "👋 Welcome to the Trading Agent Bot!\n\n"
            "This bot provides notifications and control for your automated trading system.\n\n"
            "Use /help to see available commands."
        )
    
    async def _help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command."""
        commands = [
            "/start - Start the bot",
            "/help - Show this help message",
            "/status - Show current trading status"
        ]
        
        # Add custom commands
        for command in self.command_handlers:
            commands.append(f"/{command} - Custom command")
        
        await update.message.reply_text(
            "📋 Available Commands:\n\n" + "\n".join(commands)
        )
    
    async def _status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command."""
        # This would be replaced with actual status information
        await update.message.reply_text(
            "📊 Trading Agent Status:\n\n"
            "• System: Running\n"
            "• Trading Mode: Active\n"
            "• Last Update: Just now\n\n"
            "No active alerts at this time."
        )
    
    async def _message_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle non-command messages."""
        # This would be replaced with LLM processing of messages
        await update.message.reply_text(
            "I received your message. For a list of commands, type /help."
        )


# Example usage
async def main():
    """Main function for testing."""
    telegram = TelegramIntegration()
    
    # Start polling
    await telegram.start_polling()
    
    # Send a test message
    await telegram.send_notification(
        "Test Notification",
        "This is a test notification from the Telegram Integration module.",
        "normal"
    )
    
    # Keep the bot running
    try:
        await asyncio.sleep(60)  # Run for 60 seconds
    finally:
        await telegram.stop_polling()


if __name__ == "__main__":
    asyncio.run(main())
