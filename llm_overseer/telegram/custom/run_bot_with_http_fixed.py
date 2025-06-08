#!/usr/bin/env python
"""
Robust Telegram bot runner with HTTP server for Render deployment.
This script properly manages the bot lifecycle and HTTP server to ensure stability.
"""
import os
import sys
import logging
import asyncio
import threading
import signal
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure the project root is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    logger.info(f"Added project root to Python path: {project_root}")

# Global variables
http_server = None
bot_process = None
shutdown_event = threading.Event()

# Simple HTTP request handler
class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Handle GET requests with a simple status response."""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        if self.path == '/health':
            # Health check endpoint
            self.wfile.write(b'{"status": "ok", "service": "telegram-bot", "health": "good"}')
        else:
            # Default response
            self.wfile.write(b'{"status": "ok", "service": "telegram-bot"}')
    
    def log_message(self, format, *args):
        """Override to suppress HTTP server logs."""
        pass

def run_http_server(port):
    """Run HTTP server on specified port."""
    global http_server
    try:
        server_address = ('', port)
        http_server = HTTPServer(server_address, SimpleHTTPRequestHandler)
        logger.info(f"Starting HTTP server on port {port}")
        
        # Serve until shutdown event is set
        while not shutdown_event.is_set():
            http_server.handle_request()
            
    except Exception as e:
        logger.error(f"Error running HTTP server: {e}")
    finally:
        if http_server:
            http_server.server_close()
            logger.info("HTTP server closed")

def run_telegram_bot():
    """Run the Telegram bot in a separate process."""
    try:
        # Import here to avoid circular imports
        from llm_overseer.telegram.run_llm_powered_bot import main as bot_main
        
        logger.info("Starting Telegram bot")
        asyncio.run(bot_main())
    except Exception as e:
        if "This Application is still running" in str(e):
            logger.warning("Bot application is already running, this is expected during restarts")
        else:
            logger.error(f"Error running Telegram bot: {e}")

def signal_handler(sig, frame):
    """Handle termination signals gracefully."""
    logger.info(f"Received signal {sig}, shutting down...")
    shutdown_event.set()
    
    # Close HTTP server
    if http_server:
        logger.info("Closing HTTP server...")
        http_server.server_close()
    
    # Exit
    sys.exit(0)

def main():
    """Main function to run both HTTP server and Telegram bot."""
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Get port from environment
    port = int(os.environ.get('PORT', 8000))
    logger.info(f"Using port {port} for HTTP server")
    
    # Print current working directory and Python path for debugging
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Python path: {sys.path}")
    
    try:
        # Start HTTP server in a separate thread
        http_thread = threading.Thread(target=run_http_server, args=(port,), daemon=True)
        http_thread.start()
        logger.info("HTTP server thread started")
        
        # Start Telegram bot in the main thread
        # This is a blocking call and will keep running
        run_telegram_bot()
        
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
        shutdown_event.set()
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        # Wait for HTTP server to shut down
        if http_thread.is_alive():
            http_thread.join(timeout=5)
        logger.info("Shutdown complete")

if __name__ == "__main__":
    main()
