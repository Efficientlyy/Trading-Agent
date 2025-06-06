#!/usr/bin/env python
"""
Run script for Telegram bot with natural language key management.
This script runs the enhanced Telegram bot with NLP features.
"""
import os
import sys
import logging
import asyncio
from flask import Flask, jsonify, request
from threading import Thread

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import bot and config
from ..config.config import Config
from .bot_with_nlp import TelegramBotWithNLP
from .key_manager import KeyManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Flask app for API
app = Flask(__name__)

# Global variables
bot = None
key_manager = None

@app.route('/api/keys/<service>/<key_type>', methods=['GET'])
def get_key(service, key_type):
    """Get a specific key."""
    if not key_manager:
        return jsonify({"error": "Key manager not initialized"}), 500
    
    value = key_manager.get_key(service, key_type)
    
    if value:
        return jsonify({"service": service, "key_type": key_type, "value": value})
    else:
        return jsonify({"error": "Key not found"}), 404

@app.route('/api/keys/<service>', methods=['GET'])
def get_service_keys(service):
    """Get all keys for a service."""
    if not key_manager:
        return jsonify({"error": "Key manager not initialized"}), 500
    
    keys = {}
    service_keys = key_manager.list_keys().get(service, [])
    
    for key_type in service_keys:
        value = key_manager.get_key(service, key_type)
        if value:
            keys[key_type] = value
    
    return jsonify({"service": service, "keys": keys})

@app.route('/api/keys', methods=['GET'])
def list_keys():
    """List all available keys."""
    if not key_manager:
        return jsonify({"error": "Key manager not initialized"}), 500
    
    return jsonify({"keys": key_manager.list_keys()})

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "ok"})

def run_api_server():
    """Run the API server."""
    port = int(os.environ.get('PORT', 8000))
    host = os.environ.get('HOST', '0.0.0.0')
    app.run(host=host, port=port)

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
        bot = TelegramBotWithNLP(config)
        
        # Start API server in a separate thread
        api_thread = Thread(target=run_api_server)
        api_thread.daemon = True
        api_thread.start()
        
        logger.info("Starting Telegram bot with natural language key management")
        await bot.run()
        
    except Exception as e:
        logger.error(f"Error running bot: {e}")

if __name__ == "__main__":
    asyncio.run(main())
