#!/usr/bin/env python
"""
Bot-only script for the LLM-powered Telegram bot.
This script runs only the Telegram bot without the Flask API.
"""
import os
import sys
import logging
import asyncio

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import bot and config
from llm_overseer.config.config import Config
from llm_overseer.telegram.llm_powered_bot import LLMPoweredTelegramBot
from llm_overseer.telegram.key_manager import KeyManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables
bot = None
key_manager = None

async def main():
    """Main function."""
    global bot, key_manager
    
    try:
        # Load configuration
        config = Config()
        
        # Get bot token
        bot_token = config.get("telegram.bot_token")
        if not bot_token:
            bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
            if not bot_token:
                logger.error("Telegram bot token not found")
                return
            config.set("telegram.bot_token", bot_token)
        
        # Get allowed user IDs
        allowed_user_ids = config.get("telegram.allowed_user_ids")
        if not allowed_user_ids:
            allowed_users_str = os.environ.get("TELEGRAM_ALLOWED_USERS")
            if allowed_users_str:
                allowed_user_ids = [int(user_id.strip()) for user_id in allowed_users_str.split(",")]
                config.set("telegram.allowed_user_ids", allowed_user_ids)
        
        # Initialize key manager
        key_manager = KeyManager(bot_token)
        
        # Initialize bot
        bot = LLMPoweredTelegramBot(config)
        
        logger.info("Starting LLM-powered Telegram bot")
        await bot.run()
        
    except Exception as e:
        logger.error(f"Error running bot: {e}")

if __name__ == "__main__":
    asyncio.run(main())
