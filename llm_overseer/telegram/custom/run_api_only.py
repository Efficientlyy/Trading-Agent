#!/usr/bin/env python
"""
API-only script for the LLM-powered Telegram bot.
This script runs only the Flask API without the Telegram bot.
"""
import os
import sys
import logging
from flask import Flask, jsonify, request

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import config and key manager
from llm_overseer.config.config import Config
from llm_overseer.telegram.key_manager import KeyManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Flask app for API
app = Flask(__name__)

# Global variables
key_manager = None

# Initialize key manager
def init_key_manager():
    global key_manager
    
    try:
        # Load configuration
        config = Config()
        
        # Get bot token
        bot_token = config.get("telegram.bot_token")
        if not bot_token:
            bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
            if not bot_token:
                logger.error("Telegram bot token not found")
                return False
            config.set("telegram.bot_token", bot_token)
        
        # Initialize key manager
        key_manager = KeyManager(bot_token)
        return True
    except Exception as e:
        logger.error(f"Error initializing key manager: {e}")
        return False

@app.route('/api/keys/<service>/<key_type>', methods=['GET'])
def get_key(service, key_type):
    """Get a specific key."""
    if not key_manager:
        if not init_key_manager():
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
        if not init_key_manager():
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
        if not init_key_manager():
            return jsonify({"error": "Key manager not initialized"}), 500
    
    return jsonify({"keys": key_manager.list_keys()})

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    # Initialize key manager
    init_key_manager()
    
    # Run the Flask app
    port = int(os.environ.get('PORT', 8000))
    host = os.environ.get('HOST', '0.0.0.0')
    app.run(host=host, port=port)
