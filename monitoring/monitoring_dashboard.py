#!/usr/bin/env python
"""
Monitoring Dashboard Service for Trading-Agent System

This module provides a comprehensive monitoring dashboard for the Trading-Agent system,
including system status, risk metrics, trading activity, and performance metrics.
"""

import os
import sys
import json
import time
import logging
import threading
import numpy as np
import pandas as pd
from enum import Enum
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from flask import Flask, jsonify, request, render_template, send_from_directory
from flask_cors import CORS

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import error handling and performance optimization
from error_handling.error_manager import handle_error, ErrorCategory, ErrorSeverity, safe_execute
from performance.performance_optimizer import get_performance_summary
from risk_management.risk_controller import get_risk_metrics, get_position_summary

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("monitoring_dashboard.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("monitoring_dashboard")

# Initialize Flask app
app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# Create static and templates directories if they don't exist
os.makedirs(os.path.join(os.path.dirname(__file__), "static"), exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), "templates"), exist_ok=True)

# Create dashboard HTML template
dashboard_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trading-Agent Monitoring Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.8.1/font/bootstrap-icons.css">
    <style>
        :root {
            --bg-dark: #121212;
            --bg-card: #1e1e1e;
            --text-primary: #e0e0e0;
            --text-secondary: #a0a0a0;
            --accent-primary: #3a86ff;
            --accent-secondary: #8338ec;
            --success: #38b000;
            --warning: #ffbe0b;
            --danger: #ff006e;
            --info: #3a86ff;
        }
        
        body {
            background-color: var(--bg-dark);
            color: var(--text-primary);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        .dashboard-card {
            background-color: var(--bg-card);
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            padding: 20px;
            margin-bottom: 20px;
            transition: transform 0.3s ease;
        }
        
        .dashboard-card:hover {
            transform: translateY(-5px);
        }
        
        .card-title {
            color: var(--text-primary);
            font-weight: 600;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
        }
        
        .card-title i {
            margin-right: 10px;
            color: var(--accent-primary);
        }
        
        .metric-value {
            font-size: 24px;
            font-weight: 700;
            margin-bottom: 5px;
        }
        
        .metric-label {
            color: var(--text-secondary);
            font-size: 14px;
        }
        
        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 8px;
        }
        
        .status-active {
            background-color: var(--success);
        }
        
        .status-restricted {
            background-color: var(--warning);
        }
        
        .status-suspended {
            background-color: var(--danger);
        }
        
        .status-emergency {
            background-color: var(--danger);
            animation: blink 1s infinite;
        }
        
        @keyframes blink {
            0% { opacity: 1; }
            50% { opacity: 0.3; }
            100% { opacity: 1; }
        }
        
        .positive {
            color: var(--success);
        }
        
        .negative {
            color: var(--danger);
        }
        
        .neutral {
            color: var(--text-primary);
        }
        
        .warning {
            color: var(--warning);
        }
        
        .table {
            color: var(--text-primary);
            background-color: transparent;
        }
        
        .table thead th {
            border-color: #333;
            color: var(--text-secondary);
        }
        
        .table tbody td {
            border-color: #333;
        }
        
        .progress {
            background-color: #333;
            height: 8px;
            margin-top: 5px;
        }
        
        .chart-container {
            width: 100%;
            height: 300px;
            margin-top: 15px;
        }
        
        .nav-tabs {
            border-bottom: 1px solid #333;
        }
        
        .nav-tabs .nav-link {
            color: var(--text-secondary);
            border: none;
            border-bottom: 2px solid transparent;
            padding: 10px 15px;
        }
        
        .nav-tabs .nav-link.active {
            color: var(--accent-primary);
            background-color: transparent;
            border-bottom: 2px solid var(--accent-primary);
        }
        
        .nav-tabs .nav-link:hover {
            border-color: transparent;
            color: var(--text-primary);
        }
        
        .log-entry {
            font-family: 'Consolas', monospace;
            font-size: 12px;
            padding: 5px;
            border-bottom: 1px solid #333;
        }
        
        .log-info {
            color: var(--info);
        }
        
        .log-warning {
            color: var(--warning);
        }
        
        .log-error {
            color: var(--danger);
        }
        
        .refresh-button {
            background-color: var(--accent-primary);
            border: none;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            position: fixed;
            bottom: 20px;
            right: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease;
        }
        
        .refresh-button:hover {
            transform: rotate(180deg);
            background-color: var(--accent-secondary);
        }
        
        .asset-selector {
            background-color: var(--bg-card);
            color: var(--text-primary);
            border: 1px solid #333;
            border-radius: 5px;
            padding: 5px 10px;
        }
        
        .timeframe-button {
            background-color: var(--bg-card);
            color: var(--text-secondary);
            border: 1px solid #333;
            border-radius: 5px;
            padding: 5px 10px;
            margin-right: 5px;
            transition: all 0.3s ease;
        }
        
        .timeframe-button.active {
            background-color: var(--accent-primary);
            color: white;
            border-color: var(--accent-primary);
        }
    </style>
</head>
<body>
    <div class="container-fluid py-4">
        <div class="row mb-4">
            <div class="col-12">
                <h1 class="mb-0">Trading-Agent Monitoring Dashboard</h1>
                <p class="text-secondary">Real-time monitoring and analytics</p>
            </div>
        </div>
        
        <div class="row mb-4">
            <!-- System Status Card -->
            <div class="col-md-3">
                <div class="dashboard-card">
                    <h5 class="card-title"><i class="bi bi-activity"></i> System Status</h5>
                    <div class="d-flex align-items-center mb-3">
                        <div class="status-indicator" id="system-status-indicator"></div>
                        <div class="metric-value" id="system-status-value">Loading...</div>
                    </div>
                    <div class="metric-label" id="system-status-reason"></div>
                    <div class="mt-3">
                        <div class="d-flex justify-content-between mb-1">
                            <span>CPU Usage</span>
                            <span id="cpu-usage-value">0%</span>
                        </div>
                        <div class="progress">
                            <div class="progress-bar bg-info" id="cpu-usage-bar" role="progressbar" style="width: 0%"></div>
                        </div>
                    </div>
                    <div class="mt-3">
                        <div class="d-flex justify-content-between mb-1">
                            <span>Memory Usage</span>
                            <span id="memory-usage-value">0 MB</span>
                        </div>
                        <div class="progress">
                            <div class="progress-bar bg-info" id="memory-usage-bar" role="progressbar" style="width: 0%"></div>
                        </div>
                    </div>
                    <div class="mt-3">
                        <div class="d-flex justify-content-between mb-1">
                            <span>API Latency</span>
                            <span id="api-latency-value">0 ms</span>
                        </div>
                        <div class="progress">
                            <div class="progress-bar bg-info" id="api-latency-bar" role="progressbar" style="width: 0%"></div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Portfolio Metrics Card -->
            <div class="col-md-3">
                <div class="dashboard-card">
                    <h5 class="card-title"><i class="bi bi-wallet2"></i> Portfolio Metrics</h5>
                    <div class="row">
                        <div class="col-6 mb-3">
                            <div class="metric-value" id="portfolio-value">$0.00</div>
                            <div class="metric-label">Portfolio Value</div>
                        </div>
                        <div class="col-6 mb-3">
                            <div class="metric-value" id="daily-pnl">$0.00</div>
                            <div class="metric-label">Daily P&L</div>
                        </div>
                        <div class="col-6 mb-3">
                            <div class="metric-value" id="total-exposure">$0.00</div>
                            <div class="metric-label">Total Exposure</div>
                        </div>
                        <div class="col-6 mb-3">
                            <div class="metric-value" id="exposure-pct">0%</div>
                            <div class="metric-label">Exposure %</div>
                        </div>
                    </div>
                    <div class="mt-2">
                        <div class="d-flex justify-content-between mb-1">
                            <span>Drawdown</span>
                            <span id="drawdown-value">0%</span>
                        </div>
                        <div class="progress">
                            <div class="progress-bar bg-warning" id="drawdown-bar" role="progressbar" style="width: 0%"></div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Asset Exposure Card -->
            <div class="col-md-3">
                <div class="dashboard-card">
                    <h5 class="card-title"><i class="bi bi-pie-chart"></i> Asset Exposure</h5>
                    <div id="asset-exposure-container">
                        <div class="text-center py-4">
                            <div class="spinner-border text-primary" role="status">
                                <span class="visually-hidden">Loading...</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Trading Activity Card -->
            <div class="col-md-3">
                <div class="dashboard-card">
                    <h5 class="card-title"><i class="bi bi-graph-up"></i> Trading Activity</h5>
                    <div class="row">
                        <div class="col-6 mb-3">
                            <div class="metric-value" id="open-positions">0</div>
                            <div class="metric-label">Open Positions</div>
                        </div>
                        <div class="col-6 mb-3">
                            <div class="metric-value" id="closed-positions">0</div>
                            <div class="metric-label">Closed Positions</div>
                        </div>
                        <div class="col-6 mb-3">
                            <div class="metric-value" id="signals-today">0</div>
                            <div class="metric-label">Signals Today</div>
                        </div>
                        <div class="col-6 mb-3">
                            <div class="metric-value" id="trades-today">0</div>
                            <div class="metric-label">Trades Today</div>
                        </div>
                    </div>
                    <div class="mt-2">
                        <div class="d-flex justify-content-between mb-1">
                            <span>Win Rate</span>
                            <span id="win-rate-value">0%</span>
                        </div>
                        <div class="progress">
                            <div class="progress-bar bg-success" id="win-rate-bar" role="progressbar" style="width: 0%"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row mb-4">
            <!-- Market Data Card -->
            <div class="col-md-8">
                <div class="dashboard-card">
                    <div class="d-flex justify-content-between align-items-center mb-3">
                        <h5 class="card-title mb-0"><i class="bi bi-bar-chart"></i> Market Data</h5>
                        <div>
                            <select class="asset-selector me-2" id="asset-selector">
                                <option value="BTC/USDC">BTC/USDC</option>
                                <option value="ETH/USDC">ETH/USDC</option>
                                <option value="SOL/USDC">SOL/USDC</option>
                            </select>
                            <button class="timeframe-button active" data-timeframe="5m">5m</button>
                            <button class="timeframe-button" data-timeframe="15m">15m</button>
                            <button class="timeframe-button" data-timeframe="1h">1h</button>
                            <button class="timeframe-button" data-timeframe="4h">4h</button>
                            <button class="timeframe-button" data-timeframe="1d">1d</button>
                        </div>
                    </div>
                    <div class="chart-container" id="price-chart-container">
                        <canvas id="price-chart"></canvas>
                    </div>
                </div>
            </div>
            
            <!-- Circuit Breakers Card -->
            <div class="col-md-4">
                <div class="dashboard-card">
                    <h5 class="card-title"><i class="bi bi-exclamation-triangle"></i> Circuit Breakers</h5>
                    <div id="circuit-breakers-container">
                        <div class="alert alert-success">No active circuit breakers</div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row mb-4">
            <!-- Open Positions Card -->
            <div class="col-md-6">
                <div class="dashboard-card">
                    <h5 class="card-title"><i class="bi bi-list-check"></i> Open Positions</h5>
                    <div class="table-responsive">
                        <table class="table table-sm">
                            <thead>
                                <tr>
                                    <th>Symbol</th>
                                    <th>Quantity</th>
                                    <th>Entry Price</th>
                                    <th>Current Price</th>
                                    <th>P&L</th>
                                </tr>
                            </thead>
                            <tbody id="open-positions-table">
                                <tr>
                                    <td colspan="5" class="text-center">No open positions</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
            
            <!-- Recent Trades Card -->
            <div class="col-md-6">
                <div class="dashboard-card">
                    <h5 class="card-title"><i class="bi bi-clock-history"></i> Recent Trades</h5>
                    <div class="table-responsive">
                        <table class="table table-sm">
                            <thead>
                                <tr>
                                    <th>Symbol</th>
                                    <th>Side</th>
                                    <th>Quantity</th>
                                    <th>Price</th>
                                    <th>P&L</th>
                                    <th>Time</th>
                                </tr>
                            </thead>
                            <tbody id="recent-trades-table">
                                <tr>
                                    <td colspan="6" class="text-center">No recent trades</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row mb-4">
            <!-- Performance Metrics Card -->
            <div class="col-md-6">
                <div class="dashboard-card">
                    <h5 class="card-title"><i class="bi bi-speedometer2"></i> Performance Metrics</h5>
                    <ul class="nav nav-tabs mb-3" id="performanceTab" role="tablist">
                        <li class="nav-item" role="presentation">
                            <button class="nav-link active" id="execution-tab" data-bs-toggle="tab" data-bs-target="#execution" type="button" role="tab">Execution Time</button>
                        </li>
                        <li class="nav-item" role="presentation">
                            <button class="nav-link" id="api-tab" data-bs-toggle="tab" data-bs-target="#api" type="button" role="tab">API Performance</button>
                        </li>
                        <li class="nav-item" role="presentation">
                            <button class="nav-link" id="resources-tab" data-bs-toggle="tab" data-bs-target="#resources" type="button" role="tab">Resources</button>
                        </li>
                    </ul>
                    <div class="tab-content" id="performanceTabContent">
                        <div class="tab-pane fade show active" id="execution" role="tabpanel">
                            <div class="table-responsive">
                                <table class="table table-sm">
                                    <thead>
                                        <tr>
                                            <th>Operation</th>
                                            <th>Avg Time (ms)</th>
                                            <th>Min (ms)</th>
                                            <th>Max (ms)</th>
                                            <th>Count</th>
                                        </tr>
                                    </thead>
                                    <tbody id="execution-time-table">
                                        <tr>
                                            <td colspan="5" class="text-center">No data available</td>
                                        </tr>
                                    </tbody>
                                </table>
                            </div>
                        </div>
                        <div class="tab-pane fade" id="api" role="tabpanel">
                            <div class="table-responsive">
                                <table class="table table-sm">
                                    <thead>
                                        <tr>
                                            <th>Endpoint</th>
                                            <th>Avg Latency (ms)</th>
                                            <th>Min (ms)</th>
                                            <th>Max (ms)</th>
                                            <th>Count</th>
                                        </tr>
                                    </thead>
                                    <tbody id="api-latency-table">
                                        <tr>
                                            <td colspan="5" class="text-center">No data available</td>
                                        </tr>
                                    </tbody>
                                </table>
                            </div>
                        </div>
                        <div class="tab-pane fade" id="resources" role="tabpanel">
                            <div class="chart-container" id="resources-chart-container">
                                <canvas id="resources-chart"></canvas>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- System Logs Card -->
            <div class="col-md-6">
                <div class="dashboard-card">
                    <h5 class="card-title"><i class="bi bi-journal-text"></i> System Logs</h5>
                    <div class="d-flex justify-content-between mb-2">
                        <div>
                            <button class="btn btn-sm btn-outline-secondary me-1" id="log-level-all">All</button>
                            <button class="btn btn-sm btn-outline-info me-1" id="log-level-info">Info</button>
                            <button class="btn btn-sm btn-outline-warning me-1" id="log-level-warning">Warning</button>
                            <button class="btn btn-sm btn-outline-danger" id="log-level-error">Error</button>
                        </div>
                        <div>
                            <button class="btn btn-sm btn-outline-secondary" id="refresh-logs">
                                <i class="bi bi-arrow-clockwise"></i> Refresh
                            </button>
                        </div>
                    </div>
                    <div class="log-container p-2 bg-dark" style="height: 300px; overflow-y: auto;">
                        <div id="log-entries">
                            <div class="log-entry log-info">
                                [INFO] System initialized
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <button class="refresh-button" id="refresh-dashboard">
        <i class="bi bi-arrow-clockwise text-white"></i>
    </button>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    
    <script>
        // Dashboard refresh interval (in milliseconds)
        const REFRESH_INTERVAL = 5000;
        
        // Initialize charts
        let priceChart = null;
        let resourcesChart = null;
        
        // Dashboard data
        let dashboardData = {
            systemStatus: {},
            riskMetrics: {},
            positionSummary: {},
            performanceSummary: {}
        };
        
        // Initialize dashboard
        function initDashboard() {
            // Set up event listeners
            document.getElementById('refresh-dashboard').addEventListener('click', refreshDashboard);
            document.getElementById('refresh-logs').addEventListener('click', fetchLogs);
            document.getElementById('asset-selector').addEventListener('change', updatePriceChart);
            
            // Set up timeframe buttons
            document.querySelectorAll('.timeframe-button').forEach(button => {
                button.addEventListener('click', (e) => {
                    document.querySelectorAll('.timeframe-button').forEach(btn => btn.classList.remove('active'));
                    e.target.classList.add('active');
                    updatePriceChart();
                });
            });
            
            // Set up log level filters
            document.getElementById('log-level-all').addEventListener('click', () => filterLogs('all'));
            document.getElementById('log-level-info').addEventListener('click', () => filterLogs('info'));
            document.getElementById('log-level-warning').addEventListener('click', () => filterLogs('warning'));
            document.getElementById('log-level-error').addEventListener('click', () => filterLogs('error'));
            
            // Initialize charts
            initCharts();
            
            // Fetch initial data
            refreshDashboard();
            fetchLogs();
            
            // Set up refresh interval
            setInterval(refreshDashboard, REFRESH_INTERVAL);
        }
        
        // Initialize charts
        function initCharts() {
            // Price chart
            const priceCtx = document.getElementById('price-chart').getContext('2d');
            priceChart = new Chart(priceCtx, {
                type: 'candlestick',
                data: {
                    datasets: [{
                        label: 'BTC/USDC',
                        data: []
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            type: 'time',
                            time: {
                                unit: 'hour'
                            },
                            grid: {
                                color: '#333'
                            },
                            ticks: {
                                color: '#a0a0a0'
                            }
                        },
                        y: {
                            grid: {
                                color: '#333'
                            },
                            ticks: {
                                color: '#a0a0a0'
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            display: false
                        },
                        tooltip: {
                            mode: 'index',
                            intersect: false
                        }
                    }
                }
            });
            
            // Resources chart
            const resourcesCtx = document.getElementById('resources-chart').getContext('2d');
            resourcesChart = new Chart(resourcesCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        {
                            label: 'CPU Usage (%)',
                            data: [],
                            borderColor: '#3a86ff',
                            backgroundColor: 'rgba(58, 134, 255, 0.1)',
                            tension: 0.4,
                            fill: true
                        },
                        {
                            label: 'Memory Usage (MB)',
                            data: [],
                            borderColor: '#8338ec',
                            backgroundColor: 'rgba(131, 56, 236, 0.1)',
                            tension: 0.4,
                            fill: true
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            grid: {
                                color: '#333'
                            },
                            ticks: {
                                color: '#a0a0a0'
                            }
                        },
                        y: {
                            grid: {
                                color: '#333'
                            },
                            ticks: {
                                color: '#a0a0a0'
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            labels: {
                                color: '#a0a0a0'
                            }
                        }
                    }
                }
            });
        }
        
        // Refresh dashboard data
        function refreshDashboard() {
            // Fetch system status
            fetch('/api/system_status')
                .then(response => response.json())
                .then(data => {
                    dashboardData.systemStatus = data;
                    updateSystemStatus();
                })
                .catch(error => console.error('Error fetching system status:', error));
            
            // Fetch risk metrics
            fetch('/api/risk_metrics')
                .then(response => response.json())
                .then(data => {
                    dashboardData.riskMetrics = data;
                    updateRiskMetrics();
                })
                .catch(error => console.error('Error fetching risk metrics:', error));
            
            // Fetch position summary
            fetch('/api/position_summary')
                .then(response => response.json())
                .then(data => {
                    dashboardData.positionSummary = data;
                    updatePositionSummary();
                })
                .catch(error => console.error('Error fetching position summary:', error));
            
            // Fetch performance summary
            fetch('/api/performance_summary')
                .then(response => response.json())
                .then(data => {
                    dashboardData.performanceSummary = data;
                    updatePerformanceSummary();
                })
                .catch(error => console.error('Error fetching performance summary:', error));
            
            // Fetch market data for chart
            updatePriceChart();
        }
        
        // Update system status display
        function updateSystemStatus() {
            const data = dashboardData.systemStatus;
            
            // Update status indicator
            const statusIndicator = document.getElementById('system-status-indicator');
            const statusValue = document.getElementById('system-status-value');
            const statusReason = document.getElementById('system-status-reason');
            
            statusIndicator.className = 'status-indicator';
            
            if (data.trading_status === 'active') {
                statusIndicator.classList.add('status-active');
                statusValue.textContent = 'Active';
                statusValue.className = 'metric-value positive';
            } else if (data.trading_status === 'restricted') {
                statusIndicator.classList.add('status-restricted');
                statusValue.textContent = 'Restricted';
                statusValue.className = 'metric-value warning';
            } else if (data.trading_status === 'suspended') {
                statusIndicator.classList.add('status-suspended');
                statusValue.textContent = 'Suspended';
                statusValue.className = 'metric-value negative';
            } else if (data.trading_status === 'emergency_stop') {
                statusIndicator.classList.add('status-emergency');
                statusValue.textContent = 'Emergency Stop';
                statusValue.className = 'metric-value negative';
            }
            
            statusReason.textContent = data.status_reason || '';
            
            // Update resource usage
            if (data.resources) {
                const cpuUsage = data.resources.cpu_percent || 0;
                const memoryUsage = data.resources.memory_mb || 0;
                const apiLatency = data.resources.api_latency_ms || 0;
                
                document.getElementById('cpu-usage-value').textContent = `${cpuUsage.toFixed(1)}%`;
                document.getElementById('cpu-usage-bar').style.width = `${cpuUsage}%`;
                
                document.getElementById('memory-usage-value').textContent = `${memoryUsage.toFixed(1)} MB`;
                document.getElementById('memory-usage-bar').style.width = `${Math.min(memoryUsage / 10, 100)}%`;
                
                document.getElementById('api-latency-value').textContent = `${apiLatency.toFixed(1)} ms`;
                document.getElementById('api-latency-bar').style.width = `${Math.min(apiLatency / 5, 100)}%`;
            }
        }
        
        // Update risk metrics display
        function updateRiskMetrics() {
            const data = dashboardData.riskMetrics;
            
            // Update portfolio metrics
            document.getElementById('portfolio-value').textContent = formatCurrency(data.portfolio_value);
            
            const dailyPnl = document.getElementById('daily-pnl');
            dailyPnl.textContent = formatCurrency(data.daily_pnl);
            dailyPnl.className = 'metric-value ' + (data.daily_pnl >= 0 ? 'positive' : 'negative');
            
            document.getElementById('total-exposure').textContent = formatCurrency(data.total_exposure);
            
            const exposurePct = document.getElementById('exposure-pct');
            exposurePct.textContent = formatPercent(data.exposure_pct);
            
            // Update drawdown
            const drawdownValue = document.getElementById('drawdown-value');
            drawdownValue.textContent = formatPercent(data.drawdown);
            
            const drawdownBar = document.getElementById('drawdown-bar');
            drawdownBar.style.width = `${data.drawdown * 100}%`;
            
            if (data.drawdown > 0.05) {
                drawdownBar.className = 'progress-bar bg-danger';
            } else if (data.drawdown > 0.02) {
                drawdownBar.className = 'progress-bar bg-warning';
            } else {
                drawdownBar.className = 'progress-bar bg-success';
            }
            
            // Update asset exposure
            const assetExposureContainer = document.getElementById('asset-exposure-container');
            
            if (data.asset_exposures && Object.keys(data.asset_exposures).length > 0) {
                let html = '';
                
                for (const [asset, exposure] of Object.entries(data.asset_exposures)) {
                    const exposurePct = data.asset_exposure_pcts[asset] || 0;
                    
                    html += `
                        <div class="mb-3">
                            <div class="d-flex justify-content-between mb-1">
                                <span>${asset}</span>
                                <span>${formatCurrency(exposure)} (${formatPercent(exposurePct)})</span>
                            </div>
                            <div class="progress">
                                <div class="progress-bar bg-info" role="progressbar" style="width: ${exposurePct * 100}%"></div>
                            </div>
                        </div>
                    `;
                }
                
                assetExposureContainer.innerHTML = html;
            } else {
                assetExposureContainer.innerHTML = '<div class="alert alert-info">No asset exposure</div>';
            }
            
            // Update circuit breakers
            const circuitBreakersContainer = document.getElementById('circuit-breakers-container');
            
            if (data.circuit_breakers && Object.keys(data.circuit_breakers).length > 0) {
                let html = '';
                
                for (const [symbol, cb] of Object.entries(data.circuit_breakers)) {
                    const triggeredAt = new Date(cb.triggered_at).toLocaleTimeString();
                    const resetAt = new Date(cb.reset_at).toLocaleTimeString();
                    
                    html += `
                        <div class="alert alert-warning mb-2">
                            <div class="d-flex justify-content-between">
                                <strong>${symbol}</strong>
                                <span>Reset at ${resetAt}</span>
                            </div>
                            <div>${cb.reason}</div>
                            <small class="text-muted">Triggered at ${triggeredAt}</small>
                        </div>
                    `;
                }
                
                circuitBreakersContainer.innerHTML = html;
            } else {
                circuitBreakersContainer.innerHTML = '<div class="alert alert-success">No active circuit breakers</div>';
            }
            
            // Update trading activity
            document.getElementById('open-positions').textContent = data.open_positions_count || 0;
            document.getElementById('closed-positions').textContent = data.closed_positions_count || 0;
            
            // Placeholder values for signals and trades today
            document.getElementById('signals-today').textContent = data.signals_today || 0;
            document.getElementById('trades-today').textContent = data.trades_today || 0;
            
            // Update win rate
            const winRate = data.win_rate || 0;
            document.getElementById('win-rate-value').textContent = formatPercent(winRate);
            document.getElementById('win-rate-bar').style.width = `${winRate * 100}%`;
        }
        
        // Update position summary display
        function updatePositionSummary() {
            const data = dashboardData.positionSummary;
            
            // Update open positions table
            const openPositionsTable = document.getElementById('open-positions-table');
            
            if (data.open_positions && data.open_positions.length > 0) {
                let html = '';
                
                for (const position of data.open_positions) {
                    const pnlClass = position.unrealized_pnl >= 0 ? 'positive' : 'negative';
                    
                    html += `
                        <tr>
                            <td>${position.symbol}</td>
                            <td>${position.quantity}</td>
                            <td>${formatCurrency(position.entry_price)}</td>
                            <td>${formatCurrency(position.last_price)}</td>
                            <td class="${pnlClass}">${formatCurrency(position.unrealized_pnl)} (${formatPercent(position.unrealized_pnl_pct)})</td>
                        </tr>
                    `;
                }
                
                openPositionsTable.innerHTML = html;
            } else {
                openPositionsTable.innerHTML = '<tr><td colspan="5" class="text-center">No open positions</td></tr>';
            }
            
            // Update recent trades table
            const recentTradesTable = document.getElementById('recent-trades-table');
            
            if (data.recent_closed_positions && data.recent_closed_positions.length > 0) {
                let html = '';
                
                for (const trade of data.recent_closed_positions) {
                    const pnlClass = trade.realized_pnl >= 0 ? 'positive' : 'negative';
                    const exitTime = new Date(trade.exit_time).toLocaleTimeString();
                    
                    html += `
                        <tr>
                            <td>${trade.symbol}</td>
                            <td>BUY</td>
                            <td>${trade.quantity}</td>
                            <td>${formatCurrency(trade.exit_price)}</td>
                            <td class="${pnlClass}">${formatCurrency(trade.realized_pnl)} (${formatPercent(trade.realized_pnl_pct)})</td>
                            <td>${exitTime}</td>
                        </tr>
                    `;
                }
                
                recentTradesTable.innerHTML = html;
            } else {
                recentTradesTable.innerHTML = '<tr><td colspan="6" class="text-center">No recent trades</td></tr>';
            }
        }
        
        // Update performance summary display
        function updatePerformanceSummary() {
            const data = dashboardData.performanceSummary;
            
            if (!data.metrics) return;
            
            // Update execution time table
            const executionTimeTable = document.getElementById('execution-time-table');
            
            if (data.metrics.execution_times && Object.keys(data.metrics.execution_times).length > 0) {
                let html = '';
                
                for (const [operation, metrics] of Object.entries(data.metrics.execution_times)) {
                    html += `
                        <tr>
                            <td>${operation}</td>
                            <td>${(metrics.avg * 1000).toFixed(2)}</td>
                            <td>${(metrics.min * 1000).toFixed(2)}</td>
                            <td>${(metrics.max * 1000).toFixed(2)}</td>
                            <td>${metrics.count}</td>
                        </tr>
                    `;
                }
                
                executionTimeTable.innerHTML = html;
            } else {
                executionTimeTable.innerHTML = '<tr><td colspan="5" class="text-center">No data available</td></tr>';
            }
            
            // Update API latency table
            const apiLatencyTable = document.getElementById('api-latency-table');
            
            if (data.metrics.api_latencies && Object.keys(data.metrics.api_latencies).length > 0) {
                let html = '';
                
                for (const [endpoint, metrics] of Object.entries(data.metrics.api_latencies)) {
                    html += `
                        <tr>
                            <td>${endpoint}</td>
                            <td>${(metrics.avg * 1000).toFixed(2)}</td>
                            <td>${(metrics.min * 1000).toFixed(2)}</td>
                            <td>${(metrics.max * 1000).toFixed(2)}</td>
                            <td>${metrics.count}</td>
                        </tr>
                    `;
                }
                
                apiLatencyTable.innerHTML = html;
            } else {
                apiLatencyTable.innerHTML = '<tr><td colspan="5" class="text-center">No data available</td></tr>';
            }
            
            // Update resources chart
            if (resourcesChart) {
                // Add current timestamp
                const now = new Date().toLocaleTimeString();
                
                if (resourcesChart.data.labels.length >= 20) {
                    resourcesChart.data.labels.shift();
                    resourcesChart.data.datasets[0].data.shift();
                    resourcesChart.data.datasets[1].data.shift();
                }
                
                resourcesChart.data.labels.push(now);
                resourcesChart.data.datasets[0].data.push(data.metrics.cpu_usage?.current || 0);
                resourcesChart.data.datasets[1].data.push(data.metrics.memory_usage?.current || 0);
                
                resourcesChart.update();
            }
        }
        
        // Update price chart
        function updatePriceChart() {
            const symbol = document.getElementById('asset-selector').value;
            const timeframe = document.querySelector('.timeframe-button.active').dataset.timeframe;
            
            fetch(`/api/market_data?symbol=${symbol}&timeframe=${timeframe}`)
                .then(response => response.json())
                .then(data => {
                    if (priceChart) {
                        priceChart.data.datasets[0].label = symbol;
                        priceChart.data.datasets[0].data = data.map(candle => ({
                            x: new Date(candle.timestamp),
                            o: candle.open,
                            h: candle.high,
                            l: candle.low,
                            c: candle.close
                        }));
                        
                        priceChart.update();
                    }
                })
                .catch(error => console.error('Error fetching market data:', error));
        }
        
        // Fetch system logs
        function fetchLogs() {
            fetch('/api/logs')
                .then(response => response.json())
                .then(data => {
                    const logEntries = document.getElementById('log-entries');
                    
                    if (data.logs && data.logs.length > 0) {
                        let html = '';
                        
                        for (const log of data.logs) {
                            let logClass = 'log-info';
                            
                            if (log.level === 'WARNING') {
                                logClass = 'log-warning';
                            } else if (log.level === 'ERROR' || log.level === 'CRITICAL') {
                                logClass = 'log-error';
                            }
                            
                            html += `
                                <div class="log-entry ${logClass}" data-level="${log.level}">
                                    [${log.timestamp}] [${log.level}] ${log.message}
                                </div>
                            `;
                        }
                        
                        logEntries.innerHTML = html;
                    } else {
                        logEntries.innerHTML = '<div class="log-entry log-info">[INFO] No logs available</div>';
                    }
                })
                .catch(error => console.error('Error fetching logs:', error));
        }
        
        // Filter logs by level
        function filterLogs(level) {
            const logEntries = document.querySelectorAll('.log-entry');
            
            logEntries.forEach(entry => {
                if (level === 'all') {
                    entry.style.display = 'block';
                } else if (level === 'info' && entry.classList.contains('log-info')) {
                    entry.style.display = 'block';
                } else if (level === 'warning' && entry.classList.contains('log-warning')) {
                    entry.style.display = 'block';
                } else if (level === 'error' && entry.classList.contains('log-error')) {
                    entry.style.display = 'block';
                } else {
                    entry.style.display = 'none';
                }
            });
        }
        
        // Format currency
        function formatCurrency(value) {
            return new Intl.NumberFormat('en-US', {
                style: 'currency',
                currency: 'USD'
            }).format(value);
        }
        
        // Format percent
        function formatPercent(value) {
            return new Intl.NumberFormat('en-US', {
                style: 'percent',
                minimumFractionDigits: 2,
                maximumFractionDigits: 2
            }).format(value);
        }
        
        // Initialize dashboard when DOM is loaded
        document.addEventListener('DOMContentLoaded', initDashboard);
    </script>
</body>
</html>
"""

# Create dashboard HTML file
with open(os.path.join(os.path.dirname(__file__), "templates", "dashboard.html"), "w") as f:
    f.write(dashboard_html)

# Mock data for testing
mock_data = {
    "system_status": {
        "trading_status": "active",
        "status_reason": "",
        "resources": {
            "cpu_percent": 15.2,
            "memory_mb": 256.8,
            "api_latency_ms": 42.5
        }
    },
    "risk_metrics": {
        "portfolio_value": 10000.0,
        "peak_portfolio_value": 10500.0,
        "total_exposure": 2500.0,
        "exposure_pct": 0.25,
        "asset_exposures": {
            "BTC": 1500.0,
            "ETH": 750.0,
            "SOL": 250.0
        },
        "asset_exposure_pcts": {
            "BTC": 0.15,
            "ETH": 0.075,
            "SOL": 0.025
        },
        "daily_pnl": 125.0,
        "daily_pnl_pct": 0.0125,
        "drawdown": 0.02,
        "trading_status": "active",
        "status_reason": "",
        "open_positions_count": 3,
        "closed_positions_count": 5,
        "signals_today": 12,
        "trades_today": 8,
        "win_rate": 0.65,
        "circuit_breakers": {}
    },
    "position_summary": {
        "open_positions": [
            {
                "position_id": "pos_1654321_BTC_USDC",
                "symbol": "BTC/USDC",
                "quantity": 0.05,
                "entry_price": 35000.0,
                "entry_time": "2025-06-02T12:00:00",
                "stop_loss": 34300.0,
                "take_profit": 36750.0,
                "trailing_stop": 34800.0,
                "last_price": 35500.0,
                "highest_price": 35800.0,
                "lowest_price": 34900.0,
                "unrealized_pnl": 25.0,
                "unrealized_pnl_pct": 0.0143
            },
            {
                "position_id": "pos_1654322_ETH_USDC",
                "symbol": "ETH/USDC",
                "quantity": 0.5,
                "entry_price": 1500.0,
                "entry_time": "2025-06-02T13:30:00",
                "stop_loss": 1470.0,
                "take_profit": 1575.0,
                "trailing_stop": None,
                "last_price": 1520.0,
                "highest_price": 1525.0,
                "lowest_price": 1495.0,
                "unrealized_pnl": 10.0,
                "unrealized_pnl_pct": 0.0133
            },
            {
                "position_id": "pos_1654323_SOL_USDC",
                "symbol": "SOL/USDC",
                "quantity": 5.0,
                "entry_price": 50.0,
                "entry_time": "2025-06-02T14:45:00",
                "stop_loss": 48.5,
                "take_profit": 53.0,
                "trailing_stop": None,
                "last_price": 49.5,
                "highest_price": 50.5,
                "lowest_price": 49.0,
                "unrealized_pnl": -2.5,
                "unrealized_pnl_pct": -0.01
            }
        ],
        "recent_closed_positions": [
            {
                "position_id": "pos_1654320_BTC_USDC",
                "symbol": "BTC/USDC",
                "quantity": 0.03,
                "entry_price": 34800.0,
                "entry_time": "2025-06-02T10:15:00",
                "exit_price": 35200.0,
                "exit_time": "2025-06-02T11:30:00",
                "realized_pnl": 12.0,
                "realized_pnl_pct": 0.0115,
                "close_reason": "take_profit"
            },
            {
                "position_id": "pos_1654319_ETH_USDC",
                "symbol": "ETH/USDC",
                "quantity": 0.3,
                "entry_price": 1520.0,
                "entry_time": "2025-06-02T09:45:00",
                "exit_price": 1490.0,
                "exit_time": "2025-06-02T10:30:00",
                "realized_pnl": -9.0,
                "realized_pnl_pct": -0.0197,
                "close_reason": "stop_loss"
            }
        ]
    },
    "performance_summary": {
        "metrics": {
            "execution_times": {
                "pattern_recognition": {
                    "avg": 0.125,
                    "min": 0.085,
                    "max": 0.350,
                    "count": 120
                },
                "signal_generation": {
                    "avg": 0.045,
                    "min": 0.020,
                    "max": 0.120,
                    "count": 240
                },
                "order_execution": {
                    "avg": 0.085,
                    "min": 0.050,
                    "max": 0.250,
                    "count": 15
                }
            },
            "api_latencies": {
                "get_ticker": {
                    "avg": 0.035,
                    "min": 0.020,
                    "max": 0.120,
                    "count": 350
                },
                "get_klines": {
                    "avg": 0.085,
                    "min": 0.050,
                    "max": 0.250,
                    "count": 120
                },
                "create_order": {
                    "avg": 0.150,
                    "min": 0.100,
                    "max": 0.350,
                    "count": 15
                }
            },
            "cpu_usage": {
                "avg": 12.5,
                "current": 15.2
            },
            "memory_usage": {
                "avg": 245.0,
                "current": 256.8
            }
        }
    },
    "logs": [
        {
            "timestamp": "2025-06-02 15:30:45",
            "level": "INFO",
            "message": "System initialized"
        },
        {
            "timestamp": "2025-06-02 15:31:12",
            "level": "INFO",
            "message": "Connected to MEXC API"
        },
        {
            "timestamp": "2025-06-02 15:32:05",
            "level": "INFO",
            "message": "Pattern recognition initialized"
        },
        {
            "timestamp": "2025-06-02 15:33:20",
            "level": "WARNING",
            "message": "API rate limit approaching (80% used)"
        },
        {
            "timestamp": "2025-06-02 15:35:10",
            "level": "INFO",
            "message": "Signal generated for BTC/USDC: BUY"
        },
        {
            "timestamp": "2025-06-02 15:35:15",
            "level": "INFO",
            "message": "Position opened: BTC/USDC 0.05 @ 35000.0"
        },
        {
            "timestamp": "2025-06-02 15:40:30",
            "level": "ERROR",
            "message": "Failed to fetch order book: Connection timeout"
        },
        {
            "timestamp": "2025-06-02 15:40:45",
            "level": "INFO",
            "message": "Reconnected to MEXC API"
        }
    ],
    "market_data": {
        "BTC/USDC": {
            "5m": [
                {"timestamp": "2025-06-02T15:00:00", "open": 35000.0, "high": 35100.0, "low": 34950.0, "close": 35050.0, "volume": 10.5},
                {"timestamp": "2025-06-02T15:05:00", "open": 35050.0, "high": 35150.0, "low": 35000.0, "close": 35100.0, "volume": 12.3},
                {"timestamp": "2025-06-02T15:10:00", "open": 35100.0, "high": 35200.0, "low": 35050.0, "close": 35150.0, "volume": 15.7},
                {"timestamp": "2025-06-02T15:15:00", "open": 35150.0, "high": 35250.0, "low": 35100.0, "close": 35200.0, "volume": 18.2},
                {"timestamp": "2025-06-02T15:20:00", "open": 35200.0, "high": 35300.0, "low": 35150.0, "close": 35250.0, "volume": 14.5},
                {"timestamp": "2025-06-02T15:25:00", "open": 35250.0, "high": 35350.0, "low": 35200.0, "close": 35300.0, "volume": 11.8},
                {"timestamp": "2025-06-02T15:30:00", "open": 35300.0, "high": 35400.0, "low": 35250.0, "close": 35350.0, "volume": 13.2},
                {"timestamp": "2025-06-02T15:35:00", "open": 35350.0, "high": 35450.0, "low": 35300.0, "close": 35400.0, "volume": 16.9},
                {"timestamp": "2025-06-02T15:40:00", "open": 35400.0, "high": 35500.0, "low": 35350.0, "close": 35450.0, "volume": 19.3},
                {"timestamp": "2025-06-02T15:45:00", "open": 35450.0, "high": 35550.0, "low": 35400.0, "close": 35500.0, "volume": 17.1}
            ]
        },
        "ETH/USDC": {
            "5m": [
                {"timestamp": "2025-06-02T15:00:00", "open": 1500.0, "high": 1505.0, "low": 1495.0, "close": 1502.0, "volume": 25.5},
                {"timestamp": "2025-06-02T15:05:00", "open": 1502.0, "high": 1508.0, "low": 1500.0, "close": 1505.0, "volume": 28.3},
                {"timestamp": "2025-06-02T15:10:00", "open": 1505.0, "high": 1510.0, "low": 1502.0, "close": 1508.0, "volume": 30.7},
                {"timestamp": "2025-06-02T15:15:00", "open": 1508.0, "high": 1515.0, "low": 1505.0, "close": 1512.0, "volume": 35.2},
                {"timestamp": "2025-06-02T15:20:00", "open": 1512.0, "high": 1518.0, "low": 1510.0, "close": 1515.0, "volume": 32.5},
                {"timestamp": "2025-06-02T15:25:00", "open": 1515.0, "high": 1520.0, "low": 1512.0, "close": 1518.0, "volume": 29.8},
                {"timestamp": "2025-06-02T15:30:00", "open": 1518.0, "high": 1525.0, "low": 1515.0, "close": 1520.0, "volume": 31.2},
                {"timestamp": "2025-06-02T15:35:00", "open": 1520.0, "high": 1528.0, "low": 1518.0, "close": 1525.0, "volume": 36.9},
                {"timestamp": "2025-06-02T15:40:00", "open": 1525.0, "high": 1530.0, "low": 1522.0, "close": 1528.0, "volume": 39.3},
                {"timestamp": "2025-06-02T15:45:00", "open": 1528.0, "high": 1535.0, "low": 1525.0, "close": 1530.0, "volume": 37.1}
            ]
        },
        "SOL/USDC": {
            "5m": [
                {"timestamp": "2025-06-02T15:00:00", "open": 50.0, "high": 50.5, "low": 49.8, "close": 50.2, "volume": 150.5},
                {"timestamp": "2025-06-02T15:05:00", "open": 50.2, "high": 50.8, "low": 50.0, "close": 50.5, "volume": 165.3},
                {"timestamp": "2025-06-02T15:10:00", "open": 50.5, "high": 51.0, "low": 50.2, "close": 50.8, "volume": 180.7},
                {"timestamp": "2025-06-02T15:15:00", "open": 50.8, "high": 51.2, "low": 50.5, "close": 51.0, "volume": 195.2},
                {"timestamp": "2025-06-02T15:20:00", "open": 51.0, "high": 51.5, "low": 50.8, "close": 51.2, "volume": 175.5},
                {"timestamp": "2025-06-02T15:25:00", "open": 51.2, "high": 51.8, "low": 51.0, "close": 51.5, "volume": 160.8},
                {"timestamp": "2025-06-02T15:30:00", "open": 51.5, "high": 52.0, "low": 51.2, "close": 51.8, "volume": 170.2},
                {"timestamp": "2025-06-02T15:35:00", "open": 51.8, "high": 52.2, "low": 51.5, "close": 52.0, "volume": 185.9},
                {"timestamp": "2025-06-02T15:40:00", "open": 52.0, "high": 52.5, "low": 51.8, "close": 52.2, "volume": 200.3},
                {"timestamp": "2025-06-02T15:45:00", "open": 52.2, "high": 52.8, "low": 52.0, "close": 52.5, "volume": 190.1}
            ]
        }
    }
}

# API routes
@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/api/system_status')
def api_system_status():
    try:
        # Get real system status if available
        status = {
            "trading_status": "active",
            "status_reason": "",
            "resources": {
                "cpu_percent": 0,
                "memory_mb": 0,
                "api_latency_ms": 0
            }
        }
        
        # Try to get real risk metrics
        try:
            risk_metrics = get_risk_metrics()
            if risk_metrics:
                status["trading_status"] = risk_metrics.get("trading_status", "active")
                status["status_reason"] = risk_metrics.get("status_reason", "")
        except Exception as e:
            logger.error(f"Error getting risk metrics: {e}")
        
        # Try to get real performance metrics
        try:
            performance_summary = get_performance_summary()
            if performance_summary and "metrics" in performance_summary:
                metrics = performance_summary["metrics"]
                if "cpu_usage" in metrics:
                    status["resources"]["cpu_percent"] = metrics["cpu_usage"].get("current", 0)
                if "memory_usage" in metrics:
                    status["resources"]["memory_mb"] = metrics["memory_usage"].get("current", 0)
                
                # Calculate average API latency
                if "api_latencies" in metrics:
                    api_latencies = metrics["api_latencies"]
                    if api_latencies:
                        avg_latency = 0
                        count = 0
                        for endpoint, latency in api_latencies.items():
                            avg_latency += latency.get("avg", 0) * 1000  # Convert to ms
                            count += 1
                        
                        if count > 0:
                            status["resources"]["api_latency_ms"] = avg_latency / count
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
        
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error in system_status API: {e}")
        return jsonify(mock_data["system_status"])

@app.route('/api/risk_metrics')
def api_risk_metrics():
    try:
        # Get real risk metrics if available
        try:
            risk_metrics = get_risk_metrics()
            if risk_metrics:
                return jsonify(risk_metrics)
        except Exception as e:
            logger.error(f"Error getting risk metrics: {e}")
        
        return jsonify(mock_data["risk_metrics"])
    except Exception as e:
        logger.error(f"Error in risk_metrics API: {e}")
        return jsonify(mock_data["risk_metrics"])

@app.route('/api/position_summary')
def api_position_summary():
    try:
        # Get real position summary if available
        try:
            position_summary = get_position_summary()
            if position_summary:
                return jsonify(position_summary)
        except Exception as e:
            logger.error(f"Error getting position summary: {e}")
        
        return jsonify(mock_data["position_summary"])
    except Exception as e:
        logger.error(f"Error in position_summary API: {e}")
        return jsonify(mock_data["position_summary"])

@app.route('/api/performance_summary')
def api_performance_summary():
    try:
        # Get real performance summary if available
        try:
            performance_summary = get_performance_summary()
            if performance_summary:
                return jsonify({"metrics": performance_summary})
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
        
        return jsonify(mock_data["performance_summary"])
    except Exception as e:
        logger.error(f"Error in performance_summary API: {e}")
        return jsonify(mock_data["performance_summary"])

@app.route('/api/logs')
def api_logs():
    try:
        # Get real logs if available
        logs = []
        
        try:
            # Read last 100 lines from log file
            with open("trading_agent.log", "r") as f:
                lines = f.readlines()[-100:]
                
                for line in lines:
                    parts = line.split(" - ")
                    if len(parts) >= 4:
                        timestamp = parts[0]
                        level = parts[2]
                        message = " - ".join(parts[3:]).strip()
                        
                        logs.append({
                            "timestamp": timestamp,
                            "level": level,
                            "message": message
                        })
        except Exception as e:
            logger.error(f"Error reading log file: {e}")
        
        if not logs:
            logs = mock_data["logs"]
        
        return jsonify({"logs": logs})
    except Exception as e:
        logger.error(f"Error in logs API: {e}")
        return jsonify({"logs": mock_data["logs"]})

@app.route('/api/market_data')
def api_market_data():
    try:
        symbol = request.args.get('symbol', 'BTC/USDC')
        timeframe = request.args.get('timeframe', '5m')
        
        # Get real market data if available
        # TODO: Implement real market data retrieval
        
        # Use mock data for now
        if symbol in mock_data["market_data"] and timeframe in mock_data["market_data"][symbol]:
            return jsonify(mock_data["market_data"][symbol][timeframe])
        
        return jsonify([])
    except Exception as e:
        logger.error(f"Error in market_data API: {e}")
        return jsonify([])

def run_dashboard(host='0.0.0.0', port=5000, debug=False):
    """Run monitoring dashboard
    
    Args:
        host: Host to listen on
        port: Port to listen on
        debug: Whether to run in debug mode
    """
    logger.info(f"Starting monitoring dashboard on {host}:{port}")
    app.run(host=host, port=port, debug=debug)

if __name__ == "__main__":
    run_dashboard()
