#!/usr/bin/env python
"""
Bot-only script with HTTP server for the LLM-powered Telegram bot.
This script runs the Telegram bot with a simple HTTP server to keep the port open.
"""
import os
import sys
import logging
import asyncio
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

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
http_server = None

# Simple HTTP request handler
class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(b'{"status": "ok", "service": "telegram-bot"}')
    
    def log_message(self, format, *args):
        # Suppress HTTP server logs to avoid cluttering the output
        pass

# Function to run HTTP server
def run_http_server(port):
    global http_server
    try:
        server_address = ('', port)
        http_server = HTTPServer(server_address, SimpleHTTPRequestHandler)
        logger.info(f"Starting HTTP server on port {port}")
        http_server.serve_forever()
    except Exception as e:
        logger.error(f"Error running HTTP server: {e}")

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
    # Start HTTP server in a separate thread
    port = int(os.environ.get('PORT', 8000))
    http_thread = threading.Thread(target=run_http_server, args=(port,), daemon=True)
    http_thread.start()
    
    # Run the bot in the main thread
    asyncio.run(main())
