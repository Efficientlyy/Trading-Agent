#!/usr/bin/env python
"""
Monitoring Dashboard Service for Trading-Agent System

This module provides a Flask-based service to monitor the Trading-Agent system,
aggregating data from logging, performance monitoring, risk management,
and trading activity.
"""

import os
import json
import time
import logging
from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS
from threading import Lock

# Import system components (assuming they are accessible)
try:
    from error_handling_and_logging import LoggerFactory
    from performance_optimization import PerformanceMonitor, OptimizedDataCache, BatchProcessor, DataAggregator, UIUpdateOptimizer, WebSocketOptimizer
    from risk_management import RiskManager, RiskDashboard
except ImportError as e:
    # Fallback if components are not available
    logging.warning(f"Could not import all components: {e}")
    # Define dummy classes if needed for testing
    class LoggerFactory:
        @staticmethod
        def get_logger(name, log_level="INFO", log_file=None, log_to_console=True):
            return logging.getLogger(name)
    class PerformanceMonitor:
        def __init__(self, logger=None):
            pass
        def get_stats(self):
            return {}
    class OptimizedDataCache:
        def __init__(self, max_items=1000, ttl=60):
            pass
        def get_stats(self):
            return {}
    class BatchProcessor:
        def __init__(self, processor_func, batch_size=100, max_delay=1.0):
            pass
        def get_stats(self):
            return {}
    class DataAggregator:
        def __init__(self, window_size=100):
            pass
        def get_stats(self):
            return {}
    class UIUpdateOptimizer:
        def __init__(self, min_update_interval=0.1, batch_updates=True):
            pass
        def get_stats(self):
            return {}
    class WebSocketOptimizer:
        def __init__(self, compression=True, batch_size=10, max_delay=0.1):
            pass
        def get_stats(self):
            return {}
    class RiskManager:
        def __init__(self, risk_parameters=None, portfolio_value=10000.0):
            pass
        def get_risk_metrics(self):
            return {}
        def get_all_positions(self):
            return {}
        def get_trade_history(self, limit=None):
            return []
    class RiskDashboard:
        def __init__(self, risk_manager):
            pass
        def get_dashboard_data(self):
            return {
                "risk_indicators": {},
                "positions": [],
                "trade_history": [],
                "exposure_data": [],
                "volatility_data": [],
                "trend_data": []
            }
        def get_risk_alerts(self):
            return []
        def get_risk_recommendations(self):
            return []

# Configure logging
logger = LoggerFactory.get_logger(
    "monitoring_dashboard",
    log_level="INFO",
    log_file="monitoring_dashboard.log"
)

class MonitoringDashboardService:
    """Monitoring Dashboard Service"""

    def __init__(self, host="0.0.0.0", port=8082, debug=True):
        """Initialize monitoring dashboard service

        Args:
            host: Host to run on
            port: Port to run on
            debug: Whether to run in debug mode
        """
        self.host = host
        self.port = port
        self.debug = debug
        self.lock = Lock()

        # Initialize Flask app
        self.app = Flask(__name__, static_folder="static", template_folder="templates")
        CORS(self.app)

        # Initialize system components (replace with actual instances)
        # These should ideally be passed in or retrieved from a central registry
        self.risk_manager = RiskManager() # Placeholder
        self.risk_dashboard = RiskDashboard(self.risk_manager) # Placeholder
        self.performance_monitor = PerformanceMonitor(logger=logger) # Placeholder
        self.data_cache = OptimizedDataCache() # Placeholder
        self.batch_processor = BatchProcessor(lambda x: None) # Placeholder
        self.data_aggregator = DataAggregator() # Placeholder
        self.ui_optimizer = UIUpdateOptimizer() # Placeholder
        self.ws_optimizer = WebSocketOptimizer() # Placeholder

        # Set up routes
        self._setup_routes()

        logger.info("Initialized MonitoringDashboardService")

    def _setup_routes(self):
        """Set up Flask routes"""

        @self.app.route("/")
        def index():
            """Serve monitoring dashboard index page"""
            return render_template("monitoring.html")

        @self.app.route("/static/<path:path>")
        def serve_static(path):
            """Serve static files"""
            return send_from_directory(self.app.static_folder, path)

        @self.app.route("/api/system-status")
        def api_system_status():
            """Get overall system status"""
            # In a real system, check health of various components
            status = {
                "trading_engine": "Operational",
                "data_feed": "Connected",
                "risk_management": "Active",
                "order_execution": "Enabled",
                "overall_status": self.risk_manager.get_risk_metrics().get("trading_status", "NORMAL")
            }
            return jsonify(status)

        @self.app.route("/api/performance-metrics")
        def api_performance_metrics():
            """Get performance metrics"""
            # Aggregate metrics from various performance components
            metrics = {
                "performance_monitor": self.performance_monitor.get_stats(),
                "data_cache": self.data_cache.get_stats(),
                "batch_processor": self.batch_processor.get_stats(),
                "data_aggregator": self.data_aggregator.get_stats(),
                "ui_optimizer": self.ui_optimizer.get_stats(),
                "ws_optimizer": self.ws_optimizer.get_stats(),
                # Add more metrics like CPU/Memory usage if available
                "cpu_usage": "N/A",
                "memory_usage": "N/A"
            }
            return jsonify(metrics)

        @self.app.route("/api/risk-summary")
        def api_risk_summary():
            """Get risk management summary"""
            summary = self.risk_dashboard.get_dashboard_data()
            return jsonify(summary)

        @self.app.route("/api/risk-alerts")
        def api_risk_alerts():
            """Get current risk alerts"""
            alerts = self.risk_dashboard.get_risk_alerts()
            return jsonify(alerts)

        @self.app.route("/api/risk-recommendations")
        def api_risk_recommendations():
            """Get risk management recommendations"""
            recommendations = self.risk_dashboard.get_risk_recommendations()
            return jsonify(recommendations)

        @self.app.route("/api/logs")
        def api_logs():
            """Get recent log entries"""
            log_file = "monitoring_dashboard.log" # Example log file
            lines = []
            try:
                with open(log_file, "r") as f:
                    # Read last N lines (e.g., 100)
                    lines = f.readlines()[-100:]
            except FileNotFoundError:
                logger.warning(f"Log file not found: {log_file}")
            except Exception as e:
                logger.error(f"Error reading log file: {e}")
            return jsonify({"logs": lines})

        @self.app.route("/api/trading-activity")
        def api_trading_activity():
            """Get recent trading activity"""
            activity = {
                "positions": self.risk_dashboard.get_dashboard_data().get("positions", []),
                "trade_history": self.risk_dashboard.get_dashboard_data().get("trade_history", [])
            }
            return jsonify(activity)

    def run(self):
        """Run the monitoring dashboard service"""
        try:
            # Ensure templates directory exists
            template_dir = os.path.join(os.path.dirname(__file__), "templates")
            os.makedirs(template_dir, exist_ok=True)

            # Create monitoring.html template if it doesn"t exist
            index_path = os.path.join(template_dir, "monitoring.html")
            if not os.path.exists(index_path):
                with open(index_path, "w") as f:
                    f.write(self._generate_monitoring_html())

            # Ensure static directory exists
            static_dir = os.path.join(os.path.dirname(__file__), "static")
            os.makedirs(static_dir, exist_ok=True)

            # Create CSS file if it doesn"t exist
            css_path = os.path.join(static_dir, "monitoring.css")
            if not os.path.exists(css_path):
                with open(css_path, "w") as f:
                    f.write(self._generate_monitoring_css())

            # Create JS file if it doesn"t exist
            js_path = os.path.join(static_dir, "monitoring.js")
            if not os.path.exists(js_path):
                with open(js_path, "w") as f:
                    f.write(self._generate_monitoring_js())

            # Run Flask app
            self.app.run(host=self.host, port=self.port, debug=self.debug)

        except Exception as e:
            logger.error(f"Error running monitoring dashboard service: {str(e)}")
            raise

    def _generate_monitoring_html(self):
        """Generate monitoring.html template"""
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trading-Agent Monitoring</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="/static/monitoring.css">
</head>
<body>
    <div class="container">
        <header>
            <h1>Trading-Agent Monitoring Dashboard</h1>
            <div class="status-indicator" id="overall-status">SYSTEM STATUS: <span class="status-value">LOADING...</span></div>
        </header>

        <div class="dashboard-grid">
            <!-- System Status -->
            <div class="card">
                <h2>System Status</h2>
                <div class="status-grid" id="system-status-grid">
                    <div class="status-item">Trading Engine: <span class="status-value">LOADING...</span></div>
                    <div class="status-item">Data Feed: <span class="status-value">LOADING...</span></div>
                    <div class="status-item">Risk Management: <span class="status-value">LOADING...</span></div>
                    <div class="status-item">Order Execution: <span class="status-value">LOADING...</span></div>
                </div>
            </div>

            <!-- Risk Summary -->
            <div class="card">
                <h2>Risk Summary</h2>
                <div class="risk-summary-grid" id="risk-summary-grid">
                    <div class="risk-item">Portfolio Value: <span class="risk-value">LOADING...</span></div>
                    <div class="risk-item">Daily PnL: <span class="risk-value">LOADING...</span></div>
                    <div class="risk-item">Daily Trades: <span class="risk-value">LOADING...</span></div>
                    <div class="risk-item">Open Positions: <span class="risk-value">LOADING...</span></div>
                    <div class="risk-item">Total Exposure: <span class="risk-value">LOADING...</span></div>
                    <div class="risk-item">Risk Level: <span class="risk-value">LOADING...</span></div>
                </div>
            </div>

            <!-- Risk Alerts -->
            <div class="card">
                <h2>Risk Alerts</h2>
                <div class="alert-list" id="risk-alerts-list">
                    <div class="empty-message">No alerts</div>
                </div>
            </div>

            <!-- Risk Recommendations -->
            <div class="card">
                <h2>Risk Recommendations</h2>
                <div class="recommendation-list" id="risk-recommendations-list">
                    <div class="empty-message">No recommendations</div>
                </div>
            </div>

            <!-- Trading Activity -->
            <div class="card full-width">
                <h2>Trading Activity</h2>
                <div class="tabs">
                    <button class="tab-button active" data-tab="positions">Open Positions</button>
                    <button class="tab-button" data-tab="history">Trade History</button>
                </div>
                <div class="tab-content active" id="positions-content">
                    <div class="table-container">
                        <table id="positions-table">
                            <thead>
                                <tr>
                                    <th>Symbol</th>
                                    <th>Side</th>
                                    <th>Quantity</th>
                                    <th>Entry Price</th>
                                    <th>Current Price</th>
                                    <th>PnL</th>
                                    <th>PnL %</th>
                                    <th>Stop Loss</th>
                                    <th>Take Profit</th>
                                </tr>
                            </thead>
                            <tbody>
                                <!-- Position rows will be inserted here -->
                            </tbody>
                        </table>
                    </div>
                </div>
                <div class="tab-content" id="history-content">
                    <div class="table-container">
                        <table id="history-table">
                            <thead>
                                <tr>
                                    <th>Symbol</th>
                                    <th>Side</th>
                                    <th>Quantity</th>
                                    <th>Entry Price</th>
                                    <th>Exit Price</th>
                                    <th>PnL</th>
                                    <th>PnL %</th>
                                    <th>Entry Time</th>
                                    <th>Exit Time</th>
                                    <th>Reason</th>
                                </tr>
                            </thead>
                            <tbody>
                                <!-- History rows will be inserted here -->
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>

            <!-- Performance Metrics -->
            <div class="card full-width">
                <h2>Performance Metrics</h2>
                <pre id="performance-metrics-pre">Loading performance metrics...</pre>
            </div>

            <!-- Logs -->
            <div class="card full-width">
                <h2>Recent Logs</h2>
                <pre id="logs-pre">Loading logs...</pre>
            </div>
        </div>
    </div>

    <script src="/static/monitoring.js"></script>
</body>
</html>
"""

    def _generate_monitoring_css(self):
        """Generate monitoring.css"""
        # Using similar styling as dashboard_ui.py for consistency
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
    max-width: 1600px;
    margin: 0 auto;
}

header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem 0;
    border-bottom: 1px solid var(--border-color);
    margin-bottom: 1.5rem;
}

header h1 {
    font-size: 1.8rem;
    font-weight: 700;
}

.status-indicator {
    padding: 0.5rem 1rem;
    border-radius: 4px;
    font-weight: 600;
}

.status-indicator .status-value.NORMAL {
    color: var(--success);
}
.status-indicator .status-value.CAUTION {
    color: var(--warning);
}
.status-indicator .status-value.RESTRICTED {
    color: var(--danger);
}
.status-indicator .status-value.HALTED {
    color: var(--danger);
    font-weight: bold;
}

.dashboard-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
    gap: 1.5rem;
}

.card {
    background-color: var(--bg-secondary);
    border-radius: 8px;
    padding: 1.5rem;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

.card.full-width {
    grid-column: 1 / -1;
}

.card h2 {
    font-size: 1.2rem;
    font-weight: 600;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border-color);
}

.status-grid, .risk-summary-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 1rem;
}

.status-item, .risk-item {
    font-size: 0.9rem;
}

.status-value, .risk-value {
    font-weight: 600;
    margin-left: 0.5rem;
}

.status-value.Operational, .status-value.Connected, .status-value.Active, .status-value.Enabled {
    color: var(--success);
}
.status-value.Degraded, .status-value.Disconnected, .status-value.Inactive, .status-value.Disabled {
    color: var(--danger);
}

.risk-value.LOW {
    color: var(--success);
}
.risk-value.MEDIUM {
    color: var(--warning);
}
.risk-value.HIGH, .risk-value.EXTREME {
    color: var(--danger);
}

.alert-list, .recommendation-list {
    max-height: 200px;
    overflow-y: auto;
}

.alert-item, .recommendation-item {
    padding: 0.75rem;
    border-radius: 4px;
    margin-bottom: 0.5rem;
    font-size: 0.9rem;
}

.alert-item.critical {
    background-color: rgba(229, 62, 62, 0.2);
    border-left: 4px solid var(--danger);
}
.alert-item.high {
    background-color: rgba(236, 201, 75, 0.2);
    border-left: 4px solid var(--warning);
}
.alert-item.medium {
    background-color: rgba(49, 130, 206, 0.1);
    border-left: 4px solid var(--accent-primary);
}

.recommendation-item {
    background-color: var(--bg-tertiary);
    border-left: 4px solid var(--accent-secondary);
}

.tabs {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 1rem;
    border-bottom: 1px solid var(--border-color);
}

.tab-button {
    padding: 0.75rem 1.5rem;
    border: none;
    background: none;
    color: var(--text-secondary);
    font-size: 1rem;
    cursor: pointer;
    border-bottom: 2px solid transparent;
    transition: all 0.2s;
}

.tab-button.active {
    color: var(--text-primary);
    border-bottom-color: var(--accent-primary);
}

.tab-button:hover:not(.active) {
    color: var(--text-primary);
}

.tab-content {
    display: none;
}

.tab-content.active {
    display: block;
}

.table-container {
    max-height: 400px;
    overflow-y: auto;
}

table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
}

thead th {
    background-color: var(--bg-tertiary);
    padding: 0.75rem;
    text-align: left;
    font-weight: 600;
    position: sticky;
    top: 0;
}

tbody td {
    padding: 0.75rem;
    border-bottom: 1px solid var(--border-color);
}

tbody tr:last-child td {
    border-bottom: none;
}

tbody tr:hover {
    background-color: var(--bg-tertiary);
}

.pnl-positive {
    color: var(--success);
}

.pnl-negative {
    color: var(--danger);
}

pre {
    background-color: var(--bg-primary);
    padding: 1rem;
    border-radius: 4px;
    max-height: 400px;
    overflow: auto;
    font-size: 0.85rem;
    white-space: pre-wrap;
    word-wrap: break-word;
}

.empty-message {
    padding: 1rem;
    text-align: center;
    color: var(--text-secondary);
    font-style: italic;
}
"""

    def _generate_monitoring_js(self):
        """Generate monitoring.js"""
        return """// Monitoring Dashboard JavaScript

document.addEventListener('DOMContentLoaded', function() {
    console.log('Initializing monitoring dashboard...');
    setupTabs();
    startDataUpdates();
});

function setupTabs() {
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabContents = document.querySelectorAll('.tab-content');

    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tab = button.getAttribute('data-tab');

            // Update button active state
            tabButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');

            // Update content active state
            tabContents.forEach(content => {
                if (content.id === `${tab}-content`) {
                    content.classList.add('active');
                } else {
                    content.classList.remove('active');
                }
            });
        });
    });
}

function startDataUpdates() {
    // Initial data load
    updateSystemStatus();
    updateRiskSummary();
    updateRiskAlerts();
    updateRiskRecommendations();
    updateTradingActivity();
    updatePerformanceMetrics();
    updateLogs();

    // Set interval for periodic updates (e.g., every 10 seconds)
    setInterval(() => {
        updateSystemStatus();
        updateRiskSummary();
        updateRiskAlerts();
        updateRiskRecommendations();
        updateTradingActivity();
        updatePerformanceMetrics();
        updateLogs();
    }, 10000);
}

function fetchData(url, callback) {
    fetch(url)
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => callback(data))
        .catch(error => console.error(`Error fetching ${url}:`, error));
}

function updateSystemStatus() {
    fetchData('/api/system-status', data => {
        const statusGrid = document.getElementById('system-status-grid');
        statusGrid.innerHTML = `
            <div class="status-item">Trading Engine: <span class="status-value ${data.trading_engine}">${data.trading_engine}</span></div>
            <div class="status-item">Data Feed: <span class="status-value ${data.data_feed}">${data.data_feed}</span></div>
            <div class="status-item">Risk Management: <span class="status-value ${data.risk_management}">${data.risk_management}</span></div>
            <div class="status-item">Order Execution: <span class="status-value ${data.order_execution}">${data.order_execution}</span></div>
        `;
        const overallStatus = document.getElementById('overall-status').querySelector('.status-value');
        overallStatus.textContent = data.overall_status;
        overallStatus.className = `status-value ${data.overall_status}`;
    });
}

function updateRiskSummary() {
    fetchData('/api/risk-summary', data => {
        const summaryGrid = document.getElementById('risk-summary-grid');
        const indicators = data.risk_indicators || {};
        summaryGrid.innerHTML = `
            <div class="risk-item">Portfolio Value: <span class="risk-value">${formatCurrency(indicators.portfolio_value)}</span></div>
            <div class="risk-item">Daily PnL: <span class="risk-value ${indicators.daily_pnl >= 0 ? 'pnl-positive' : 'pnl-negative'}">${formatCurrency(indicators.daily_pnl)} (${formatPercent(indicators.daily_pnl_pct)})</span></div>
            <div class="risk-item">Daily Trades: <span class="risk-value">${indicators.daily_trades || '0 / 0'}</span></div>
            <div class="risk-item">Open Positions: <span class="risk-value">${indicators.open_positions || '0 / 0'}</span></div>
            <div class="risk-item">Total Exposure: <span class="risk-value">${indicators.total_exposure || '0.0% / 0.0%'}</span></div>
            <div class="risk-item">Risk Level: <span class="risk-value ${indicators.risk_level}">${indicators.risk_level || 'N/A'}</span></div>
        `;
    });
}

function updateRiskAlerts() {
    fetchData('/api/risk-alerts', data => {
        const alertList = document.getElementById('risk-alerts-list');
        if (!data || data.length === 0) {
            alertList.innerHTML = '<div class="empty-message">No alerts</div>';
            return;
        }
        alertList.innerHTML = data.map(alert => `
            <div class="alert-item ${alert.level}">${alert.message}</div>
        `).join('');
    });
}

function updateRiskRecommendations() {
    fetchData('/api/risk-recommendations', data => {
        const recommendationList = document.getElementById('risk-recommendations-list');
        if (!data || data.length === 0) {
            recommendationList.innerHTML = '<div class="empty-message">No recommendations</div>';
            return;
        }
        recommendationList.innerHTML = data.map(rec => `
            <div class="recommendation-item"><strong>[${rec.type}]</strong> ${rec.message}</div>
        `).join('');
    });
}

function updateTradingActivity() {
    fetchData('/api/trading-activity', data => {
        // Update positions table
        const positionsTableBody = document.getElementById('positions-table').querySelector('tbody');
        if (!data.positions || data.positions.length === 0) {
            positionsTableBody.innerHTML = '<tr><td colspan="9" class="empty-message">No open positions</td></tr>';
        } else {
            positionsTableBody.innerHTML = data.positions.map(pos => `
                <tr>
                    <td>${pos.symbol}</td>
                    <td>${pos.side}</td>
                    <td>${pos.quantity}</td>
                    <td>${formatCurrency(pos.entry_price)}</td>
                    <td>${formatCurrency(pos.current_price)}</td>
                    <td class="${pos.pnl >= 0 ? 'pnl-positive' : 'pnl-negative'}">${formatCurrency(pos.pnl)}</td>
                    <td class="${pos.pnl_pct >= 0 ? 'pnl-positive' : 'pnl-negative'}">${formatPercent(pos.pnl_pct)}</td>
                    <td>${formatCurrency(pos.stop_loss)}</td>
                    <td>${formatCurrency(pos.take_profit)}</td>
                </tr>
            `).join('');
        }

        // Update history table
        const historyTableBody = document.getElementById('history-table').querySelector('tbody');
        if (!data.trade_history || data.trade_history.length === 0) {
            historyTableBody.innerHTML = '<tr><td colspan="10" class="empty-message">No trade history</td></tr>';
        } else {
            historyTableBody.innerHTML = data.trade_history.map(trade => `
                <tr>
                    <td>${trade.symbol}</td>
                    <td>${trade.side}</td>
                    <td>${trade.quantity}</td>
                    <td>${formatCurrency(trade.entry_price)}</td>
                    <td>${formatCurrency(trade.exit_price)}</td>
                    <td class="${trade.pnl >= 0 ? 'pnl-positive' : 'pnl-negative'}">${formatCurrency(trade.pnl)}</td>
                    <td class="${trade.pnl_pct >= 0 ? 'pnl-positive' : 'pnl-negative'}">${formatPercent(trade.pnl_pct)}</td>
                    <td>${trade.entry_time}</td>
                    <td>${trade.exit_time}</td>
                    <td>${trade.reason}</td>
                </tr>
            `).join('');
        }
    });
}

function updatePerformanceMetrics() {
    fetchData('/api/performance-metrics', data => {
        const preElement = document.getElementById('performance-metrics-pre');
        preElement.textContent = JSON.stringify(data, null, 2);
    });
}

function updateLogs() {
    fetchData('/api/logs', data => {
        const preElement = document.getElementById('logs-pre');
        preElement.textContent = data.logs ? data.logs.join('') : 'No logs available.';
        // Auto-scroll to bottom
        preElement.scrollTop = preElement.scrollHeight;
    });
}

// Helper functions for formatting
function formatCurrency(value) {
    if (value === null || value === undefined) return 'N/A';
    return value.toLocaleString('en-US', { style: 'currency', currency: 'USD' });
}

function formatPercent(value) {
    if (value === null || value === undefined) return 'N/A';
    return `${(value * 100).toFixed(2)}%`;
}

"""

# Example usage
if __name__ == "__main__":
    # Initialize monitoring dashboard service
    monitoring_service = MonitoringDashboardService()

    # Run service
    monitoring_service.run()

