#!/usr/bin/env python
"""
Run script for Telegram bot integration with LLM Strategic Overseer.

This script initializes and runs the Telegram bot with the LLM Overseer
in a standalone mode for testing and demonstration purposes.
"""

import os
import sys
import logging
import asyncio
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs', 'telegram_bot.log'))
    ]
)

logger = logging.getLogger(__name__)

# Import components
from llm_overseer.config.config import Config
from llm_overseer.core.llm_manager import TieredLLMManager
from llm_overseer.core.context_manager import ContextManager
from llm_overseer.core.token_tracker import TokenTracker
from llm_overseer.telegram.bot import TelegramBot
from llm_overseer.main import LLMOverseer

async def main():
    """Main function to run the Telegram bot with LLM Overseer."""
    try:
        # Initialize configuration
        config = Config()
        
        # Check for Telegram bot token
        bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
        if not bot_token:
            logger.error("TELEGRAM_BOT_TOKEN environment variable not set")
            print("Error: TELEGRAM_BOT_TOKEN environment variable not set")
            return
        
        # Set bot token in config
        config.set("telegram.bot_token", bot_token)
        
        # Initialize LLM Overseer
        logger.info("Initializing LLM Overseer...")
        overseer = LLMOverseer(config)
        
        # Initialize Telegram bot
        logger.info("Initializing Telegram bot...")
        bot = TelegramBot(config, overseer)
        
        # Set up the connection between bot and overseer
        bot.set_llm_overseer(overseer)
        
        # Start the bot
        logger.info("Starting Telegram bot...")
        print("Starting Telegram bot...")
        print("Press Ctrl+C to stop")
        
        # Properly initialize the application and bot before accessing properties
        await bot.application.initialize()
        
        # Now we can safely access bot properties
        bot_username = bot.application.bot.username
        print(f"Bot initialized: @{bot_username}")
        
        # Start the application and polling
        await bot.application.start()
        await bot.application.updater.start_polling()
        
        # Keep the bot running
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Stopping bot due to keyboard interrupt")
        print("Stopping bot...")
        
        # Stop the bot gracefully
        if 'bot' in locals() and hasattr(bot, 'application') and hasattr(bot.application, 'updater'):
            await bot.application.updater.stop()
        if 'bot' in locals() and hasattr(bot, 'application'):
            await bot.application.stop()
            await bot.application.shutdown()
            
    except Exception as e:
        logger.error(f"Error running bot: {e}", exc_info=True)
        print(f"Error: {e}")
        
    finally:
        logger.info("Bot stopped")
        print("Bot stopped")

if __name__ == "__main__":
    asyncio.run(main())
