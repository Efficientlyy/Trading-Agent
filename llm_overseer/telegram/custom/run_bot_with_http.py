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
import importlib.util
from http.server import HTTPServer, BaseHTTPRequestHandler

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

# Ensure the project root is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    logger.info(f"Added project root to Python path: {project_root}")

# Try to import required modules
try:
    from llm_overseer.config.config import Config
    from llm_overseer.telegram.llm_powered_bot import LLMPoweredTelegramBot
    from llm_overseer.telegram.key_manager import KeyManager
    logger.info("Successfully imported required modules")
except ImportError as e:
    logger.error(f"Import error: {e}")
    # Try to locate the modules manually
    logger.info("Attempting to locate modules manually...")
    
    # Check if the modules exist
    config_path = os.path.join(project_root, 'llm_overseer', 'config', 'config.py')
    bot_path = os.path.join(project_root, 'llm_overseer', 'telegram', 'llm_powered_bot.py')
    key_manager_path = os.path.join(project_root, 'llm_overseer', 'telegram', 'key_manager.py')
    
    logger.info(f"Checking for config.py: {os.path.exists(config_path)}")
    logger.info(f"Checking for llm_powered_bot.py: {os.path.exists(bot_path)}")
    logger.info(f"Checking for key_manager.py: {os.path.exists(key_manager_path)}")
    
    # List directory contents to help debug
    logger.info(f"Contents of {os.path.join(project_root, 'llm_overseer')}:")
    try:
        logger.info(str(os.listdir(os.path.join(project_root, 'llm_overseer'))))
    except Exception as e:
        logger.error(f"Error listing directory: {e}")
    
    # Try to import using importlib
    try:
        spec = importlib.util.spec_from_file_location("config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        Config = config_module.Config
        
        spec = importlib.util.spec_from_file_location("llm_powered_bot", bot_path)
        bot_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bot_module)
        LLMPoweredTelegramBot = bot_module.LLMPoweredTelegramBot
        
        spec = importlib.util.spec_from_file_location("key_manager", key_manager_path)
        key_manager_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(key_manager_module)
        KeyManager = key_manager_module.KeyManager
        
        logger.info("Successfully imported modules using importlib")
    except Exception as e:
        logger.error(f"Failed to import modules using importlib: {e}")
        sys.exit(1)

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
    # Print current working directory and Python path for debugging
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Python path: {sys.path}")
    
    # Start HTTP server in a separate thread
    port = int(os.environ.get('PORT', 8000))
    logger.info(f"Using port {port} for HTTP server")
    http_thread = threading.Thread(target=run_http_server, args=(port,), daemon=True)
    http_thread.start()
    
    # Run the bot in the main thread
    asyncio.run(main())
