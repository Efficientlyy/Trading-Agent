#!/usr/bin/env python
"""
AI Trading Agent Dashboard for LLM Strategic Overseer.

This module provides a web-based dashboard for monitoring trading activities,
visualizing market data, indicators, patterns, and LLM insights.
"""

import os
import sys
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime
import traceback
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format=\'%(asctime)s - %(name)s - %(levelname)s - %(message)s\',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), \'logs\', \'dashboard.log\')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import event bus and other components
from ..core.event_bus import EventBus
from ..config.config import Config
from .chart_visualization import ChartVisualization

class TradingDashboard:
    """
    Trading Dashboard Application.
    
    Provides a FastAPI-based web dashboard with real-time updates via WebSockets.
    """
    
    def __init__(self, config: Config, event_bus: Optional[EventBus] = None):
        """
        Initialize Trading Dashboard.
        
        Args:
            config: Configuration object
            event_bus: Event bus instance (optional, will create new if None)
        """
        self.config = config
        self.event_bus = event_bus if event_bus else EventBus()
        self.app = FastAPI()
        self.chart_visualizer = ChartVisualization(config, self.event_bus)
        self.active_connections: List[WebSocket] = []
        
        # Mount static files (CSS, JS)
        static_dir = os.path.join(os.path.dirname(__file__), \'static\')
        if not os.path.exists(static_dir):
            os.makedirs(static_dir)
        self.app.mount("/static", StaticFiles(directory=static_dir), name="static")
        
        # Setup routes
        self._setup_routes()
        
        # Subscribe to relevant events
        self._subscribe_to_events()
        
        logger.info("Trading Dashboard initialized")

    def _setup_routes(self) -> None:
        """
        Setup FastAPI routes.
        """
        @self.app.get("/", response_class=HTMLResponse)
        async def get_dashboard():
            # Serve the main dashboard HTML page
            html_path = os.path.join(os.path.dirname(__file__), \'templates\', \'dashboard.html\')
            if not os.path.exists(html_path):
                 # Create a basic placeholder if file doesn\'t exist
                 placeholder_html = """
                 <!DOCTYPE html>
                 <html>
                 <head>
                     <title>Trading Dashboard</title>
                     <link href=\"/static/styles.css\" rel=\"stylesheet\">
                 </head>
                 <body>
                     <h1>Trading Dashboard</h1>
                     <div id=\"chart-container\">Loading chart...</div>
                     <div id=\"log-container\">Loading logs...</div>
                     <script src=\"/static/dashboard.js\"></script>
                 </body>
                 </html>
                 """
                 # Ensure templates directory exists
                 os.makedirs(os.path.dirname(html_path), exist_ok=True)
                 with open(html_path, "w") as f:
                     f.write(placeholder_html)
                 logger.info(f"Created placeholder dashboard.html at {html_path}")
                 return HTMLResponse(content=placeholder_html)
            else:
                with open(html_path, "r") as f:
                    html_content = f.read()
                return HTMLResponse(content=html_content)

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self.active_connections.append(websocket)
            logger.info("WebSocket connection established")
            try:
                while True:
                    # Keep connection alive, handle client messages if needed
                    data = await websocket.receive_text()
                    logger.debug(f"Received message from client: {data}")
                    # Process client messages if necessary (e.g., requests for specific data)
            except WebSocketDisconnect:
                self.active_connections.remove(websocket)
                logger.info("WebSocket connection closed")
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                if websocket in self.active_connections:
                    self.active_connections.remove(websocket)

    def _subscribe_to_events(self) -> None:
        """
        Subscribe to events from the event bus to push updates to the dashboard.
        """
        topics_to_subscribe = [
            "trading.market_data",
            "analysis.indicator",
            "analysis.pattern",
            "analysis.signal",
            "llm.decision",
            "llm.feedback",
            "system.log",
            "trading.order_update"
        ]
        
        for topic in topics_to_subscribe:
            self.event_bus.subscribe(topic, self._handle_event)
            logger.info(f"Subscribed dashboard to event topic: {topic}")

    async def _handle_event(self, topic: str, data: Dict[str, Any]) -> None:
        """
        Handle events from the event bus and broadcast to connected WebSocket clients.
        
        Args:
            topic: Event topic
            data: Event data
        """
        try:
            message = {"topic": topic, "data": data}
            message_json = json.dumps(message, default=str) # Use default=str for non-serializable types like datetime
            
            # Broadcast to all active connections
            disconnected_clients = []
            for connection in self.active_connections:
                try:
                    await connection.send_text(message_json)
                except Exception as e:
                    logger.warning(f"Failed to send message to client: {e}. Marking for removal.")
                    disconnected_clients.append(connection)
            
            # Remove disconnected clients
            for client in disconnected_clients:
                if client in self.active_connections:
                    self.active_connections.remove(client)
                    logger.info("Removed disconnected WebSocket client.")
                    
            # Additionally, update the chart visualizer if relevant
            if topic == "trading.market_data":
                await self.chart_visualizer.update_chart_data(data)
            elif topic == "analysis.indicator":
                await self.chart_visualizer.add_indicator_trace(data)
            elif topic == "analysis.pattern":
                await self.chart_visualizer.add_pattern_marker(data)
            elif topic == "llm.decision":
                await self.chart_visualizer.add_decision_marker(data)
                
        except Exception as e:
            logger.error(f"Error handling event for dashboard broadcast: {e}")
            logger.error(traceback.format_exc())

    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """
        Run the FastAPI application.
        
        Args:
            host: Host address to bind to
            port: Port number to listen on
        """
        logger.info(f"Starting Trading Dashboard server on {host}:{port}")
        # Ensure static and template directories exist
        os.makedirs(os.path.join(os.path.dirname(__file__), \'static\'), exist_ok=True)
        os.makedirs(os.path.join(os.path.dirname(__file__), \'templates\'), exist_ok=True)
        
        # Create basic static files if they don\'t exist
        css_path = os.path.join(os.path.dirname(__file__), \'static\', \'styles.css\')
        js_path = os.path.join(os.path.dirname(__file__), \'static\', \'dashboard.js\')
        
        if not os.path.exists(css_path):
            with open(css_path, "w") as f:
                f.write("/* Basic styles for dashboard */\nbody { font-family: sans-serif; }\n#chart-container { height: 500px; }\n#log-container { height: 300px; overflow-y: scroll; border: 1px solid #ccc; padding: 10px; }")
            logger.info(f"Created placeholder styles.css at {css_path}")
            
        if not os.path.exists(js_path):
            with open(js_path, "w") as f:
                f.write("// Basic JavaScript for dashboard WebSocket connection\nconst ws = new WebSocket(`ws://${window.location.host}/ws`);\nconst logContainer = document.getElementById(\'log-container\');\n\nws.onmessage = function(event) {\n    console.log(\'Message from server:\', event.data);\n    const message = JSON.parse(event.data);\n    const logEntry = document.createElement(\'div\');\n    logEntry.textContent = `${new Date().toLocaleTimeString()} [${message.topic}]: ${JSON.stringify(message.data)}`;\n    logContainer.appendChild(logEntry);\n    logContainer.scrollTop = logContainer.scrollHeight; // Auto-scroll\n    // Add logic here to update charts based on message.topic and message.data\n};\n\nws.onopen = function(event) {\n    console.log(\'WebSocket connection opened\');\n    logContainer.innerHTML = \'WebSocket connection opened.\';\n};\n\nws.onerror = function(event) {\n    console.error(\'WebSocket error observed:\', event);\n};\n\nws.onclose = function(event) {\n    console.log(\'WebSocket connection closed\');\n    logContainer.innerHTML += \'<br>WebSocket connection closed.\';\n};
")
            logger.info(f"Created placeholder dashboard.js at {js_path}")
            
        # Ensure dashboard.html exists
        html_path = os.path.join(os.path.dirname(__file__), \'templates\', \'dashboard.html\')
        if not os.path.exists(html_path):
            placeholder_html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Trading Dashboard</title>
                <link href=\"/static/styles.css\" rel=\"stylesheet\">
                <script src=\"https://cdn.plot.ly/plotly-latest.min.js\"></script> <!-- Include Plotly.js -->
            </head>
            <body>
                <h1>Trading Dashboard</h1>
                <div id=\"chart-container\">Loading chart...</div>
                <div id=\"log-container\">Loading logs...</div>
                <script src=\"/static/dashboard.js\"></script>
            </body>
            </html>
            """
            os.makedirs(os.path.dirname(html_path), exist_ok=True)
            with open(html_path, "w") as f:
                f.write(placeholder_html)
            logger.info(f"Created placeholder dashboard.html at {html_path}")
            
        try:
            uvicorn.run(self.app, host=host, port=port, log_level="info")
        except Exception as e:
            logger.error(f"Failed to start dashboard server: {e}")
            logger.error(traceback.format_exc())

# Example usage (for testing)
async def main():
    config = Config()
    event_bus = EventBus()
    dashboard = TradingDashboard(config, event_bus)
    
    # Start the dashboard in a separate task
    # Note: Running uvicorn programmatically like this might have issues with async loops.
    # It\'s generally better to run via `uvicorn llm_overseer.visualization.trading_dashboard:app --reload`
    # For testing purposes, we\'ll simulate event publishing.
    
    logger.info("Simulating events for dashboard...")
    await asyncio.sleep(2) # Give server time to start (if run differently)
    
    # Simulate market data event
    await event_bus.publish("trading.market_data", {
        "symbol": "BTC/USDC",
        "timestamp": datetime.now().isoformat(),
        "price": 51000,
        "volume_24h": 1234.5,
        "success": True
    }, "normal")
    
    await asyncio.sleep(1)
    
    # Simulate indicator event
    await event_bus.publish("analysis.indicator", {
        "symbol": "BTC/USDC",
        "indicator_type": "sma",
        "values": [50500, 50600, 50700],
        "window": 20,
        "timestamp": datetime.now().isoformat()
    }, "normal")
    
    await asyncio.sleep(1)
    
    # Simulate pattern event
    await event_bus.publish("analysis.pattern", {
        "symbol": "BTC/USDC",
        "pattern_type": "double_bottom",
        "pattern_data": {"trough1_index": 50, "trough1_price": 49000, "trough2_index": 70, "trough2_price": 49100},
        "timestamp": datetime.now().isoformat(),
        "confidence": 0.85
    }, "high")
    
    logger.info("Event simulation complete.")
    # Keep running to allow WebSocket connections
    # await asyncio.Event().wait() # Keep alive indefinitely

if __name__ == "__main__":
    # To run the dashboard server:
    # 1. Ensure uvicorn and fastapi are installed: pip install uvicorn fastapi websockets
    # 2. Run from the project root: uvicorn llm_overseer.visualization.trading_dashboard:app --host 0.0.0.0 --port 8000 --reload
    # The following is for testing event publishing, not running the server itself.
    # asyncio.run(main())
    
    # Initialize and run the dashboard server directly (alternative way)
    config = Config()
    event_bus = EventBus() # Use a shared event bus if running with other components
    dashboard_app = TradingDashboard(config, event_bus)
    # Get the FastAPI app instance
    app = dashboard_app.app 
    # Run the server (this line should ideally be run via uvicorn command)
    # dashboard_app.run() # This might block if not run carefully in an async context
    logger.info("Dashboard module loaded. Run with: uvicorn llm_overseer.visualization.trading_dashboard:app --host 0.0.0.0 --port 8000")
    # Create the FastAPI app instance for uvicorn
    config_instance = Config()
    event_bus_instance = EventBus()
    dashboard_instance = TradingDashboard(config_instance, event_bus_instance)
    app = dashboard_instance.app
