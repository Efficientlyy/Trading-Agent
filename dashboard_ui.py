#!/usr/bin/env python
"""
Advanced Trading Dashboard UI for Trading-Agent System

This module provides a Flask-based dashboard UI with advanced visualization
for multiple cryptocurrency assets (BTC, ETH, SOL) with technical indicators,
pattern recognition, and AI/ML integration.
"""

import os
import json
import time
import logging
from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("dashboard_ui")

# Import data service and chart component
try:
    from multi_asset_data_service import MultiAssetDataService
    from advanced_chart_component import AdvancedChartComponent
except ImportError as e:
    logger.error(f"Error importing required modules: {str(e)}")
    raise

class DashboardUI:
    """Advanced Trading Dashboard UI"""
    
    def __init__(self, host='0.0.0.0', port=8081, debug=True):
        """Initialize dashboard UI
        
        Args:
            host: Host to run on
            port: Port to run on
            debug: Whether to run in debug mode
        """
        self.host = host
        self.port = port
        self.debug = debug
        
        # Initialize Flask app
        self.app = Flask(__name__, static_folder='static')
        CORS(self.app)
        
        # Initialize data service
        self.data_service = MultiAssetDataService()
        
        # Initialize chart component
        self.chart_component = AdvancedChartComponent(self.data_service)
        
        # Add default indicators
        self.chart_component.add_indicator("RSI")
        self.chart_component.add_indicator("MACD")
        self.chart_component.add_indicator("BollingerBands")
        self.chart_component.add_indicator("Volume")
        
        # Initialize WebSockets for real-time data
        for asset in self.data_service.get_supported_assets():
            self.data_service.initialize_websocket(asset)
        
        # Start background data fetching
        self.data_service.start_background_fetching()
        
        # Set up routes
        self._setup_routes()
        
        logger.info("Initialized DashboardUI")
    
    def _setup_routes(self):
        """Set up Flask routes"""
        
        @self.app.route('/')
        def index():
            """Serve index page"""
            return render_template('index.html')
        
        @self.app.route('/static/<path:path>')
        def serve_static(path):
            """Serve static files"""
            return send_from_directory(self.app.static_folder, path)
        
        @self.app.route('/api/assets')
        def api_assets():
            """Get supported assets"""
            return jsonify({
                "assets": self.data_service.get_supported_assets(),
                "current": self.data_service.get_current_asset()
            })
        
        @self.app.route('/api/switch-asset', methods=['POST'])
        def api_switch_asset():
            """Switch current asset"""
            data = request.json
            asset = data.get('asset')
            
            if not asset:
                return jsonify({"success": False, "error": "Asset not specified"}), 400
            
            success = self.data_service.switch_asset(asset)
            
            return jsonify({"success": success})
        
        @self.app.route('/api/ticker')
        def api_ticker():
            """Get ticker for current asset"""
            asset = request.args.get('asset')
            return jsonify(self.data_service.get_ticker(asset))
        
        @self.app.route('/api/orderbook')
        def api_orderbook():
            """Get orderbook for current asset"""
            asset = request.args.get('asset')
            limit = request.args.get('limit', 20, type=int)
            return jsonify(self.data_service.get_orderbook(asset, limit))
        
        @self.app.route('/api/trades')
        def api_trades():
            """Get trades for current asset"""
            asset = request.args.get('asset')
            limit = request.args.get('limit', 50, type=int)
            return jsonify(self.data_service.get_trades(asset, limit))
        
        @self.app.route('/api/klines')
        def api_klines():
            """Get klines for current asset"""
            asset = request.args.get('asset')
            interval = request.args.get('interval', '1m')
            limit = request.args.get('limit', 100, type=int)
            return jsonify(self.data_service.get_klines(asset, interval, limit))
        
        @self.app.route('/api/indicators')
        def api_indicators():
            """Get available indicators"""
            return jsonify({
                "available": self.chart_component.get_available_indicators(),
                "active": list(self.chart_component.get_indicators().keys())
            })
        
        @self.app.route('/api/add-indicator', methods=['POST'])
        def api_add_indicator():
            """Add indicator"""
            data = request.json
            name = data.get('name')
            params = data.get('params')
            
            if not name:
                return jsonify({"success": False, "error": "Indicator name not specified"}), 400
            
            success = self.chart_component.add_indicator(name, params)
            
            return jsonify({"success": success})
        
        @self.app.route('/api/remove-indicator', methods=['POST'])
        def api_remove_indicator():
            """Remove indicator"""
            data = request.json
            name = data.get('name')
            
            if not name:
                return jsonify({"success": False, "error": "Indicator name not specified"}), 400
            
            success = self.chart_component.remove_indicator(name)
            
            return jsonify({"success": success})
        
        @self.app.route('/api/chart-data')
        def api_chart_data():
            """Get chart data"""
            asset = request.args.get('asset')
            interval = request.args.get('interval', '1m')
            limit = request.args.get('limit', 100, type=int)
            
            chart_data = self.chart_component.get_chart_data(asset, interval, limit)
            
            return jsonify(chart_data)
        
        @self.app.route('/api/chart-config')
        def api_chart_config():
            """Get chart configuration"""
            return jsonify(self.chart_component.get_chart_config())
        
        @self.app.route('/api/patterns')
        def api_patterns():
            """Get patterns for current asset"""
            asset = request.args.get('asset')
            return jsonify(self.data_service.get_patterns(asset))
        
        @self.app.route('/api/add-pattern', methods=['POST'])
        def api_add_pattern():
            """Add pattern"""
            data = request.json
            pattern = data.get('pattern')
            
            if not pattern:
                return jsonify({"success": False, "error": "Pattern not specified"}), 400
            
            success = self.chart_component.add_pattern(pattern)
            
            return jsonify({"success": success})
        
        @self.app.route('/api/signals')
        def api_signals():
            """Get signals"""
            return jsonify(self.chart_component.get_signals())
        
        @self.app.route('/api/add-signal', methods=['POST'])
        def api_add_signal():
            """Add signal"""
            data = request.json
            signal = data.get('signal')
            
            if not signal:
                return jsonify({"success": False, "error": "Signal not specified"}), 400
            
            success = self.chart_component.add_signal(signal)
            
            return jsonify({"success": success})
        
        @self.app.route('/api/predictions')
        def api_predictions():
            """Get predictions"""
            return jsonify(self.chart_component.get_predictions())
        
        @self.app.route('/api/add-prediction', methods=['POST'])
        def api_add_prediction():
            """Add prediction"""
            data = request.json
            prediction = data.get('prediction')
            
            if not prediction:
                return jsonify({"success": False, "error": "Prediction not specified"}), 400
            
            success = self.chart_component.add_prediction(prediction)
            
            return jsonify({"success": success})
    
    def run(self):
        """Run the dashboard UI"""
        try:
            # Ensure templates directory exists
            os.makedirs(os.path.join(os.path.dirname(__file__), 'templates'), exist_ok=True)
            
            # Create index.html template if it doesn't exist
            index_path = os.path.join(os.path.dirname(__file__), 'templates', 'index.html')
            if not os.path.exists(index_path):
                with open(index_path, 'w') as f:
                    f.write(self._generate_index_html())
            
            # Ensure static directory exists
            os.makedirs(os.path.join(os.path.dirname(__file__), 'static'), exist_ok=True)
            
            # Create CSS file if it doesn't exist
            css_path = os.path.join(os.path.dirname(__file__), 'static', 'styles.css')
            if not os.path.exists(css_path):
                with open(css_path, 'w') as f:
                    f.write(self._generate_css())
            
            # Create JS file if it doesn't exist
            js_path = os.path.join(os.path.dirname(__file__), 'static', 'dashboard.js')
            if not os.path.exists(js_path):
                with open(js_path, 'w') as f:
                    f.write(self._generate_js())
            
            # Run Flask app
            self.app.run(host=self.host, port=self.port, debug=self.debug)
        
        except Exception as e:
            logger.error(f"Error running dashboard UI: {str(e)}")
            raise
        
        finally:
            # Clean up
            self.data_service.close_all_websockets()
    
    def _generate_index_html(self):
        """Generate index.html template
        
        Returns:
            str: HTML template
        """
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trading-Agent Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="/static/styles.css">
    <script src="https://cdn.jsdelivr.net/npm/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>
</head>
<body>
    <div class="container">
        <header>
            <div class="logo">Trading-Agent Dashboard</div>
            <div class="asset-switcher" id="asset-switcher">
                <div class="asset-button active" data-asset="BTC/USDC">BTC/USDC</div>
                <div class="asset-button" data-asset="ETH/USDC">ETH/USDC</div>
                <div class="asset-button" data-asset="SOL/USDC">SOL/USDC</div>
            </div>
            <div class="ticker" id="ticker">
                <div class="ticker-price">Loading...</div>
            </div>
        </header>

        <div class="dashboard">
            <div class="main-content">
                <div class="card">
                    <div class="card-header">
                        <div class="card-title" id="chart-title">BTC/USDC Chart</div>
                        <div class="controls">
                            <div class="time-frames" id="time-frames">
                                <div class="time-frame active" data-interval="1m">1m</div>
                                <div class="time-frame" data-interval="5m">5m</div>
                                <div class="time-frame" data-interval="15m">15m</div>
                                <div class="time-frame" data-interval="1h">1h</div>
                                <div class="time-frame" data-interval="4h">4h</div>
                                <div class="time-frame" data-interval="1d">1d</div>
                            </div>
                            <div class="indicator-toggle" id="indicator-toggle">
                                <div class="indicator-button">Indicators</div>
                                <div class="indicator-dropdown" id="indicator-dropdown">
                                    <div class="indicator-item">
                                        <input type="checkbox" id="indicator-rsi" checked>
                                        <label for="indicator-rsi">RSI</label>
                                    </div>
                                    <div class="indicator-item">
                                        <input type="checkbox" id="indicator-macd" checked>
                                        <label for="indicator-macd">MACD</label>
                                    </div>
                                    <div class="indicator-item">
                                        <input type="checkbox" id="indicator-bollinger" checked>
                                        <label for="indicator-bollinger">Bollinger Bands</label>
                                    </div>
                                    <div class="indicator-item">
                                        <input type="checkbox" id="indicator-volume" checked>
                                        <label for="indicator-volume">Volume</label>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div id="chart-container"></div>
                </div>
            </div>

            <div class="sidebar">
                <div class="card">
                    <div class="card-header">
                        <div class="card-title">Market Data</div>
                    </div>
                    <div class="market-data" id="market-data">
                        <div class="data-row">
                            <div class="data-label">24h High</div>
                            <div class="data-value" id="high">Loading...</div>
                        </div>
                        <div class="data-row">
                            <div class="data-label">24h Low</div>
                            <div class="data-value" id="low">Loading...</div>
                        </div>
                        <div class="data-row">
                            <div class="data-label">24h Volume</div>
                            <div class="data-value" id="volume">Loading...</div>
                        </div>
                        <div class="data-row">
                            <div class="data-label">24h Change</div>
                            <div class="data-value" id="change">Loading...</div>
                        </div>
                    </div>
                </div>

                <div class="card">
                    <div class="card-header">
                        <div class="card-title">Detected Patterns</div>
                    </div>
                    <div class="pattern-list" id="pattern-list">
                        <div class="empty-message">No patterns detected</div>
                    </div>
                </div>

                <div class="card">
                    <div class="card-header">
                        <div class="card-title">Trading Signals</div>
                    </div>
                    <div class="signal-list" id="signal-list">
                        <div class="empty-message">No signals available</div>
                    </div>
                </div>

                <div class="card">
                    <div class="card-header">
                        <div class="card-title">Recent Trades</div>
                    </div>
                    <div class="trades-container">
                        <div class="trades-header">
                            <div>Price</div>
                            <div>Amount</div>
                            <div>Time</div>
                        </div>
                        <div id="trades-container">
                            <div class="empty-message">Loading trades...</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="/static/dashboard.js"></script>
</body>
</html>
"""
    
    def _generate_css(self):
        """Generate CSS
        
        Returns:
            str: CSS
        """
        return """:root {
    --bg-primary: #121826;
    --bg-secondary: #1a2332;
    --bg-tertiary: #232f42;
    --text-primary: #e6e9f0;
    --text-secondary: #a0aec0;
    --accent-primary: #3182ce;
    --accent-secondary: #4299e1;
    --success: #48bb78;
    --danger: #e53e3e;
    --warning: #ecc94b;
    --border-color: #2d3748;
    --chart-grid: #2d3748;
    --buy-color: #48bb78;
    --sell-color: #e53e3e;
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Inter', sans-serif;
    background-color: var(--bg-primary);
    color: var(--text-primary);
    line-height: 1.5;
    padding: 20px;
}

.container {
    max-width: 1400px;
    margin: 0 auto;
}

header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem 0;
    border-bottom: 1px solid var(--border-color);
    margin-bottom: 1rem;
}

.logo {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text-primary);
}

.asset-switcher {
    display: flex;
    gap: 0.5rem;
}

.asset-button {
    padding: 0.5rem 1rem;
    border-radius: 4px;
    background-color: var(--bg-tertiary);
    color: var(--text-secondary);
    font-size: 0.875rem;
    cursor: pointer;
    transition: all 0.2s;
}

.asset-button.active {
    background-color: var(--accent-primary);
    color: white;
}

.asset-button:hover:not(.active) {
    background-color: var(--border-color);
}

.ticker {
    display: flex;
    align-items: center;
    gap: 1rem;
}

.ticker-price {
    font-size: 1.25rem;
    font-weight: 600;
}

.dashboard {
    display: grid;
    grid-template-columns: 1fr 350px;
    gap: 1rem;
}

.card {
    background-color: var(--bg-secondary);
    border-radius: 8px;
    padding: 1rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    margin-bottom: 1rem;
}

.card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border-color);
}

.card-title {
    font-size: 1rem;
    font-weight: 600;
}

.controls {
    display: flex;
    gap: 1rem;
    align-items: center;
}

.time-frames {
    display: flex;
    gap: 0.5rem;
}

.time-frame {
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    background-color: var(--bg-tertiary);
    color: var(--text-secondary);
    font-size: 0.75rem;
    cursor: pointer;
    transition: all 0.2s;
}

.time-frame.active {
    background-color: var(--accent-primary);
    color: white;
}

.time-frame:hover:not(.active) {
    background-color: var(--border-color);
}

.indicator-toggle {
    position: relative;
}

.indicator-button {
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    background-color: var(--bg-tertiary);
    color: var(--text-secondary);
    font-size: 0.75rem;
    cursor: pointer;
    transition: all 0.2s;
}

.indicator-button:hover {
    background-color: var(--border-color);
}

.indicator-dropdown {
    position: absolute;
    top: 100%;
    right: 0;
    width: 200px;
    background-color: var(--bg-tertiary);
    border-radius: 4px;
    padding: 0.5rem;
    margin-top: 0.5rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    z-index: 10;
    display: none;
}

.indicator-dropdown.show {
    display: block;
}

.indicator-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem;
    border-radius: 4px;
    transition: all 0.2s;
}

.indicator-item:hover {
    background-color: var(--border-color);
}

.indicator-item input {
    margin: 0;
}

.indicator-item label {
    cursor: pointer;
}

#chart-container {
    height: 500px;
    width: 100%;
}

.market-data {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
}

.data-row {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
}

.data-label {
    font-size: 0.75rem;
    color: var(--text-secondary);
}

.data-value {
    font-size: 1rem;
    font-weight: 500;
}

.pattern-list, .signal-list {
    max-height: 200px;
    overflow-y: auto;
}

.pattern-item, .signal-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.5rem;
    border-radius: 4px;
    margin-bottom: 0.5rem;
    background-color: var(--bg-tertiary);
}

.pattern-type, .signal-type {
    font-weight: 500;
}

.pattern-confidence, .signal-confidence {
    font-size: 0.875rem;
    padding: 0.125rem 0.375rem;
    border-radius: 4px;
    background-color: var(--bg-primary);
}

.trades-container {
    height: 300px;
    overflow-y: auto;
}

.trades-header {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    padding: 0.5rem 0;
    font-weight: 500;
    color: var(--text-secondary);
    font-size: 0.875rem;
}

.trade-row {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    padding: 0.25rem 0;
    font-size: 0.875rem;
}

.trade-row.buy {
    color: var(--buy-color);
}

.trade-row.sell {
    color: var(--sell-color);
}

.empty-message {
    padding: 1rem;
    text-align: center;
    color: var(--text-secondary);
    font-style: italic;
}

/* Responsive design */
@media (max-width: 1200px) {
    .dashboard {
        grid-template-columns: 1fr;
    }
    
    .sidebar {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1rem;
    }
}

@media (max-width: 768px) {
    header {
        flex-direction: column;
        align-items: flex-start;
        gap: 1rem;
    }
    
    .asset-switcher {
        width: 100%;
        justify-content: space-between;
    }
    
    .ticker {
        width: 100%;
        justify-content: space-between;
    }
    
    .sidebar {
        grid-template-columns: 1fr;
    }
    
    .card-header {
        flex-direction: column;
        align-items: flex-start;
        gap: 0.5rem;
    }
    
    .controls {
        width: 100%;
        justify-content: space-between;
    }
}
"""
    
    def _generate_js(self):
        """Generate JavaScript
        
        Returns:
            str: JavaScript
        """
        return """// Dashboard JavaScript

// Global variables
let currentAsset = 'BTC/USDC';
let currentInterval = '1m';
let chartInstance = null;
let indicatorSeries = {};
let lastUpdateTime = 0;
let updateInterval = 5000; // 5 seconds

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('Initializing dashboard...');
    
    // Initialize components
    initAssetSwitcher();
    initTimeFrames();
    initIndicatorToggle();
    initChart();
    
    // Start data update loop
    updateData();
    setInterval(updateData, updateInterval);
});

// Initialize asset switcher
function initAssetSwitcher() {
    const assetButtons = document.querySelectorAll('.asset-button');
    
    assetButtons.forEach(button => {
        button.addEventListener('click', () => {
            const asset = button.getAttribute('data-asset');
            
            // Skip if already selected
            if (asset === currentAsset) return;
            
            // Update UI
            assetButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');
            
            // Update chart title
            document.getElementById('chart-title').textContent = `${asset} Chart`;
            
            // Switch asset
            switchAsset(asset);
        });
    });
    
    // Fetch current asset from API
    fetch('/api/assets')
        .then(response => response.json())
        .then(data => {
            if (data && data.current) {
                currentAsset = data.current;
                
                // Update UI
                assetButtons.forEach(btn => {
                    if (btn.getAttribute('data-asset') === currentAsset) {
                        btn.classList.add('active');
                    } else {
                        btn.classList.remove('active');
                    }
                });
                
                // Update chart title
                document.getElementById('chart-title').textContent = `${currentAsset} Chart`;
            }
        })
        .catch(error => {
            console.error('Error fetching assets:', error);
        });
}

// Switch asset
function switchAsset(asset) {
    // Update current asset
    currentAsset = asset;
    
    // Call API to switch asset
    fetch('/api/switch-asset', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ asset })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            console.log(`Switched to asset: ${asset}`);
            
            // Update data
            updateData(true);
        } else {
            console.error(`Error switching to asset: ${asset}`);
        }
    })
    .catch(error => {
        console.error('Error switching asset:', error);
    });
}

// Initialize time frames
function initTimeFrames() {
    const timeFrames = document.querySelectorAll('.time-frame');
    
    timeFrames.forEach(tf => {
        tf.addEventListener('click', () => {
            const interval = tf.getAttribute('data-interval');
            
            // Skip if already selected
            if (interval === currentInterval) return;
            
            // Update UI
            timeFrames.forEach(t => t.classList.remove('active'));
            tf.classList.add('active');
            
            // Update interval
            currentInterval = interval;
            
            // Update chart data
            updateChartData();
        });
    });
}

// Initialize indicator toggle
function initIndicatorToggle() {
    const indicatorButton = document.querySelector('.indicator-button');
    const indicatorDropdown = document.getElementById('indicator-dropdown');
    
    // Toggle dropdown
    indicatorButton.addEventListener('click', () => {
        indicatorDropdown.classList.toggle('show');
    });
    
    // Close dropdown when clicking outside
    document.addEventListener('click', (event) => {
        if (!event.target.closest('.indicator-toggle')) {
            indicatorDropdown.classList.remove('show');
        }
    });
    
    // Handle indicator toggles
    const indicatorCheckboxes = document.querySelectorAll('.indicator-item input');
    
    indicatorCheckboxes.forEach(checkbox => {
        checkbox.addEventListener('change', () => {
            const indicatorName = checkbox.id.replace('indicator-', '').toUpperCase();
            
            if (checkbox.checked) {
                // Add indicator
                addIndicator(indicatorName);
            } else {
                // Remove indicator
                removeIndicator(indicatorName);
            }
        });
    });
    
    // Fetch active indicators from API
    fetch('/api/indicators')
        .then(response => response.json())
        .then(data => {
            if (data && data.active) {
                // Update checkboxes
                indicatorCheckboxes.forEach(checkbox => {
                    const indicatorName = checkbox.id.replace('indicator-', '').toUpperCase();
                    checkbox.checked = data.active.includes(indicatorName);
                });
            }
        })
        .catch(error => {
            console.error('Error fetching indicators:', error);
        });
}

// Add indicator
function addIndicator(name) {
    fetch('/api/add-indicator', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ name })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            console.log(`Added indicator: ${name}`);
            
            // Update chart data
            updateChartData();
        } else {
            console.error(`Error adding indicator: ${name}`);
        }
    })
    .catch(error => {
        console.error('Error adding indicator:', error);
    });
}

// Remove indicator
function removeIndicator(name) {
    fetch('/api/remove-indicator', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ name })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            console.log(`Removed indicator: ${name}`);
            
            // Update chart data
            updateChartData();
        } else {
            console.error(`Error removing indicator: ${name}`);
        }
    })
    .catch(error => {
        console.error('Error removing indicator:', error);
    });
}

// Initialize chart
function initChart() {
    const chartContainer = document.getElementById('chart-container');
    
    // Fetch chart configuration
    fetch('/api/chart-config')
        .then(response => response.json())
        .then(config => {
            if (!config) {
                console.error('Error fetching chart configuration');
                return;
            }
            
            // Create chart
            const chart = LightweightCharts.createChart(chartContainer, config.chart);
            
            // Add candlestick series
            const candleSeries = chart.addCandlestickSeries(config.series.candlestick);
            
            // Store chart instance
            chartInstance = {
                chart,
                candleSeries,
                config
            };
            
            // Add indicators
            addIndicatorsToChart(chart, config.indicators);
            
            // Update chart data
            updateChartData();
            
            // Resize chart on window resize
            window.addEventListener('resize', () => {
                chart.applyOptions({
                    width: chartContainer.clientWidth,
                    height: chartContainer.clientHeight
                });
            });
        })
        .catch(error => {
            console.error('Error initializing chart:', error);
        });
}

// Add indicators to chart
function addIndicatorsToChart(chart, indicatorConfigs) {
    indicatorSeries = {};
    
    for (const config of indicatorConfigs) {
        if (config.position === 'overlay') {
            // Add overlay indicator
            if (config.type === 'bands') {
                // Add Bollinger Bands
                indicatorSeries[config.name] = {
                    middle: chart.addLineSeries({
                        color: config.colors.middle,
                        lineWidth: 1,
                        priceLineVisible: false
                    }),
                    upper: chart.addLineSeries({
                        color: config.colors.upper,
                        lineWidth: 1,
                        priceLineVisible: false
                    }),
                    lower: chart.addLineSeries({
                        color: config.colors.lower,
                        lineWidth: 1,
                        priceLineVisible: false
                    })
                };
            }
        } else {
            // Add separate indicator
            const indicatorPane = chart.addPane({
                height: config.height
            });
            
            if (config.type === 'line') {
                // Add line indicator (e.g., RSI)
                indicatorSeries[config.name] = {
                    line: indicatorPane.addLineSeries({
                        color: config.colors.line,
                        lineWidth: 2,
                        priceLineVisible: false
                    })
                };
                
                // Add overbought/oversold lines
                if (config.overbought && config.oversold) {
                    indicatorSeries[config.name].overbought = indicatorPane.addLineSeries({
                        color: config.colors.overbought,
                        lineWidth: 1,
                        priceLineVisible: false
                    });
                    
                    indicatorSeries[config.name].oversold = indicatorPane.addLineSeries({
                        color: config.colors.oversold,
                        lineWidth: 1,
                        priceLineVisible: false
                    });
                }
            } else if (config.type === 'macd') {
                // Add MACD indicator
                indicatorSeries[config.name] = {
                    macd: indicatorPane.addLineSeries({
                        color: config.colors.macd,
                        lineWidth: 2,
                        priceLineVisible: false
                    }),
                    signal: indicatorPane.addLineSeries({
                        color: config.colors.signal,
                        lineWidth: 1,
                        priceLineVisible: false
                    }),
                    histogram: indicatorPane.addHistogramSeries({
                        color: config.colors.histogram.positive,
                        priceFormat: {
                            type: 'price',
                            precision: config.precision,
                            minMove: 0.01
                        }
                    })
                };
            } else if (config.type === 'histogram') {
                // Add histogram indicator (e.g., Volume)
                indicatorSeries[config.name] = {
                    histogram: indicatorPane.addHistogramSeries({
                        color: config.colors.up,
                        priceFormat: {
                            type: 'volume',
                            precision: config.precision
                        }
                    }),
                    ma: indicatorPane.addLineSeries({
                        color: config.colors.ma,
                        lineWidth: 2,
                        priceLineVisible: false
                    })
                };
            }
        }
    }
}

// Update chart data
function updateChartData() {
    if (!chartInstance) {
        console.error('Chart not initialized');
        return;
    }
    
    // Fetch chart data
    fetch(`/api/chart-data?asset=${currentAsset}&interval=${currentInterval}`)
        .then(response => response.json())
        .then(data => {
            if (!data || !data.klines) {
                console.error('Error fetching chart data');
                return;
            }
            
            // Set candlestick data
            const candleData = data.klines.map(kline => ({
                time: Math.floor(kline.time / 1000),
                open: kline.open,
                high: kline.high,
                low: kline.low,
                close: kline.close
            }));
            
            chartInstance.candleSeries.setData(candleData);
            
            // Set indicator data
            for (const [name, values] of Object.entries(data.indicators)) {
                if (name === 'BollingerBands' && indicatorSeries[name]) {
                    // Set Bollinger Bands data
                    const middleData = values.middle.map((value, i) => ({
                        time: Math.floor(data.klines[i].time / 1000),
                        value: value
                    }));
                    
                    const upperData = values.upper.map((value, i) => ({
                        time: Math.floor(data.klines[i].time / 1000),
                        value: value
                    }));
                    
                    const lowerData = values.lower.map((value, i) => ({
                        time: Math.floor(data.klines[i].time / 1000),
                        value: value
                    }));
                    
                    indicatorSeries[name].middle.setData(middleData);
                    indicatorSeries[name].upper.setData(upperData);
                    indicatorSeries[name].lower.setData(lowerData);
                } else if (name === 'RSI' && indicatorSeries[name]) {
                    // Set RSI data
                    const lineData = values.values.map((value, i) => ({
                        time: Math.floor(data.klines[i].time / 1000),
                        value: value
                    }));
                    
                    indicatorSeries[name].line.setData(lineData);
                    
                    // Set overbought/oversold lines
                    const overboughtData = data.klines.map(kline => ({
                        time: Math.floor(kline.time / 1000),
                        value: values.overbought
                    }));
                    
                    const oversoldData = data.klines.map(kline => ({
                        time: Math.floor(kline.time / 1000),
                        value: values.oversold
                    }));
                    
                    indicatorSeries[name].overbought.setData(overboughtData);
                    indicatorSeries[name].oversold.setData(oversoldData);
                } else if (name === 'MACD' && indicatorSeries[name]) {
                    // Set MACD data
                    const macdData = values.macd.map((value, i) => ({
                        time: Math.floor(data.klines[i].time / 1000),
                        value: value
                    }));
                    
                    const signalData = values.signal.map((value, i) => ({
                        time: Math.floor(data.klines[i].time / 1000),
                        value: value
                    }));
                    
                    const histogramData = values.histogram.map((value, i) => ({
                        time: Math.floor(data.klines[i].time / 1000),
                        value: value,
                        color: value >= 0 ? 
                            chartInstance.config.indicators.find(i => i.name === 'MACD').colors.histogram.positive : 
                            chartInstance.config.indicators.find(i => i.name === 'MACD').colors.histogram.negative
                    }));
                    
                    indicatorSeries[name].macd.setData(macdData);
                    indicatorSeries[name].signal.setData(signalData);
                    indicatorSeries[name].histogram.setData(histogramData);
                } else if (name === 'Volume' && indicatorSeries[name]) {
                    // Set Volume data
                    const histogramData = values.values.map((value, i) => ({
                        time: Math.floor(data.klines[i].time / 1000),
                        value: value,
                        color: data.klines[i].close >= data.klines[i].open ? 
                            chartInstance.config.indicators.find(i => i.name === 'Volume').colors.up : 
                            chartInstance.config.indicators.find(i => i.name === 'Volume').colors.down
                    }));
                    
                    const maData = values.ma.map((value, i) => ({
                        time: Math.floor(data.klines[i].time / 1000),
                        value: value
                    }));
                    
                    indicatorSeries[name].histogram.setData(histogramData);
                    indicatorSeries[name].ma.setData(maData);
                }
            }
            
            // Update patterns
            updatePatterns(data.patterns);
            
            // Update signals
            updateSignals(data.signals);
            
            // Update predictions
            updatePredictions(data.predictions);
        })
        .catch(error => {
            console.error('Error updating chart data:', error);
        });
}

// Update patterns
function updatePatterns(patterns) {
    const patternList = document.getElementById('pattern-list');
    
    if (!patterns || patterns.length === 0) {
        patternList.innerHTML = '<div class="empty-message">No patterns detected</div>';
        return;
    }
    
    patternList.innerHTML = patterns.map(pattern => `
        <div class="pattern-item">
            <div class="pattern-type">${pattern.label}</div>
            <div class="pattern-confidence">${Math.round(pattern.confidence * 100)}%</div>
        </div>
    `).join('');
}

// Update signals
function updateSignals(signals) {
    const signalList = document.getElementById('signal-list');
    
    if (!signals || signals.length === 0) {
        signalList.innerHTML = '<div class="empty-message">No signals available</div>';
        return;
    }
    
    signalList.innerHTML = signals.map(signal => `
        <div class="signal-item">
            <div class="signal-type">${signal.label}</div>
            <div class="signal-confidence">${Math.round(signal.confidence * 100)}%</div>
        </div>
    `).join('');
}

// Update predictions
function updatePredictions(predictions) {
    // Implement prediction visualization
    // This would typically be displayed on the chart
}

// Update data
function updateData(force = false) {
    const now = Date.now();
    
    // Skip if not forced and last update was less than updateInterval ago
    if (!force && now - lastUpdateTime < updateInterval) {
        return;
    }
    
    lastUpdateTime = now;
    
    // Update ticker
    updateTicker();
    
    // Update market data
    updateMarketData();
    
    // Update trades
    updateTrades();
    
    // Update chart data
    updateChartData();
}

// Update ticker
function updateTicker() {
    fetch(`/api/ticker?asset=${currentAsset}`)
        .then(response => response.json())
        .then(data => {
            if (data && data.price) {
                document.querySelector('.ticker-price').textContent = `${data.price.toLocaleString('en-US', { style: 'currency', currency: 'USD' })}`;
            }
        })
        .catch(error => {
            console.error('Error updating ticker:', error);
        });
}

// Update market data
function updateMarketData() {
    // In a real implementation, this would fetch 24h market data
    // For now, we'll use the klines data to calculate high, low, volume, and change
    fetch(`/api/klines?asset=${currentAsset}&interval=1d&limit=2`)
        .then(response => response.json())
        .then(data => {
            if (data && data.length > 0) {
                const kline = data[0];
                const prevKline = data.length > 1 ? data[1] : null;
                
                // Update high
                document.getElementById('high').textContent = kline.high.toLocaleString('en-US', { style: 'currency', currency: 'USD' });
                
                // Update low
                document.getElementById('low').textContent = kline.low.toLocaleString('en-US', { style: 'currency', currency: 'USD' });
                
                // Update volume
                document.getElementById('volume').textContent = kline.volume.toLocaleString('en-US', { maximumFractionDigits: 2 });
                
                // Update change
                if (prevKline) {
                    const change = ((kline.close - prevKline.close) / prevKline.close) * 100;
                    const changeElement = document.getElementById('change');
                    changeElement.textContent = `${change >= 0 ? '+' : ''}${change.toFixed(2)}%`;
                    changeElement.style.color = change >= 0 ? 'var(--success)' : 'var(--danger)';
                } else {
                    document.getElementById('change').textContent = 'N/A';
                }
            }
        })
        .catch(error => {
            console.error('Error updating market data:', error);
        });
}

// Update trades
function updateTrades() {
    fetch(`/api/trades?asset=${currentAsset}`)
        .then(response => response.json())
        .then(data => {
            if (!data || data.length === 0) {
                document.getElementById('trades-container').innerHTML = '<div class="empty-message">No trades available</div>';
                return;
            }
            
            // Render trades
            const tradesContainer = document.getElementById('trades-container');
            tradesContainer.innerHTML = '';
            
            for (let i = 0; i < Math.min(data.length, 30); i++) {
                const trade = data[i];
                const row = document.createElement('div');
                row.className = `trade-row ${trade.isBuyerMaker ? 'sell' : 'buy'}`;
                
                const time = new Date(trade.time);
                const timeStr = `${time.getHours().toString().padStart(2, '0')}:${time.getMinutes().toString().padStart(2, '0')}:${time.getSeconds().toString().padStart(2, '0')}`;
                
                row.innerHTML = `
                    <div>${trade.price.toFixed(2)}</div>
                    <div>${trade.quantity.toFixed(6)}</div>
                    <div>${timeStr}</div>
                `;
                tradesContainer.appendChild(row);
            }
        })
        .catch(error => {
            console.error('Error updating trades:', error);
        });
}
"""

# Example usage
if __name__ == "__main__":
    # Initialize dashboard UI
    dashboard = DashboardUI()
    
    # Run dashboard
    dashboard.run()
