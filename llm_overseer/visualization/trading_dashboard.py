#!/usr/bin/env python
"""
AI Trading Dashboard for Monitoring and Feedback

This module provides a web-based dashboard for monitoring trading activities,
visualizing chart data, and displaying AI insights and feedback.
"""

import os
import sys
import json
import logging
import asyncio
import threading
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_dashboard.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("trading_dashboard")

# Try to import Dash components
try:
    import dash
    from dash import dcc, html, callback, Input, Output, State
    import dash_bootstrap_components as dbc
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    logger.error("Dash components not found. Please install with: pip install dash dash-bootstrap-components")
    raise

class TradingDashboard:
    """
    Web-based dashboard for monitoring trading activities and AI insights.
    
    This class provides a Dash-based web interface for visualizing chart data,
    displaying AI insights, and monitoring trading activities.
    """
    
    def __init__(self, event_bus=None, data_pipeline=None, chart_visualization=None):
        """
        Initialize Trading Dashboard.
        
        Args:
            event_bus: Event Bus instance
            data_pipeline: Unified Data Pipeline instance
            chart_visualization: Chart Visualization instance
        """
        self.event_bus = event_bus
        self.data_pipeline = data_pipeline
        self.chart_visualization = chart_visualization
        
        # Dashboard state
        self.active_symbol = "BTC/USDC"
        self.active_timeframe = "1h"
        self.ai_insights = []
        self.recent_trades = []
        self.performance_metrics = {}
        self.system_status = {
            "flash_trading_active": False,
            "paper_trading_active": False,
            "last_update": datetime.now().isoformat()
        }
        
        # Dashboard app
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.DARKLY],
            suppress_callback_exceptions=True
        )
        
        # Configure dashboard layout
        self._configure_layout()
        
        # Configure callbacks
        self._configure_callbacks()
        
        # Register event handlers if event bus is provided
        if self.event_bus:
            self._register_event_handlers()
        
        # Dashboard server
        self.server = self.app.server
        
        logger.info("Trading Dashboard initialized")
    
    def set_event_bus(self, event_bus):
        """
        Set Event Bus instance.
        
        Args:
            event_bus: Event Bus instance
        """
        self.event_bus = event_bus
        self._register_event_handlers()
        
        logger.info("Event Bus set")
    
    def set_data_pipeline(self, data_pipeline):
        """
        Set Unified Data Pipeline instance.
        
        Args:
            data_pipeline: Unified Data Pipeline instance
        """
        self.data_pipeline = data_pipeline
        
        logger.info("Unified Data Pipeline set")
    
    def set_chart_visualization(self, chart_visualization):
        """
        Set Chart Visualization instance.
        
        Args:
            chart_visualization: Chart Visualization instance
        """
        self.chart_visualization = chart_visualization
        
        logger.info("Chart Visualization set")
    
    def _register_event_handlers(self):
        """Register event handlers with Event Bus."""
        self.event_bus.subscribe("trading.order_update", self._handle_order_update)
        self.event_bus.subscribe("trading.status_update", self._handle_status_update)
        self.event_bus.subscribe("llm.strategic_decision", self._handle_strategic_decision)
        self.event_bus.subscribe("indicator.signal", self._handle_indicator_signal)
        self.event_bus.subscribe("visualization.pattern_detected", self._handle_pattern_detected)
        self.event_bus.subscribe("performance.metrics_update", self._handle_performance_update)
        
        logger.info("Event handlers registered")
    
    def _configure_layout(self):
        """Configure dashboard layout."""
        # Header
        header = dbc.Navbar(
            dbc.Container(
                [
                    html.A(
                        dbc.Row(
                            [
                                dbc.Col(html.Img(src="/assets/logo.png", height="30px"), width="auto"),
                                dbc.Col(dbc.NavbarBrand("AI Trading Dashboard", className="ms-2")),
                            ],
                            align="center",
                            className="g-0",
                        ),
                        href="/",
                        style={"textDecoration": "none"},
                    ),
                    dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
                    dbc.Collapse(
                        dbc.Nav(
                            [
                                dbc.NavItem(dbc.NavLink("Charts", href="#")),
                                dbc.NavItem(dbc.NavLink("AI Insights", href="#")),
                                dbc.NavItem(dbc.NavLink("Performance", href="#")),
                                dbc.NavItem(dbc.NavLink("Settings", href="#")),
                            ],
                            className="ms-auto",
                            navbar=True,
                        ),
                        id="navbar-collapse",
                        navbar=True,
                    ),
                ]
            ),
            color="dark",
            dark=True,
            className="mb-4",
        )
        
        # Chart controls
        chart_controls = dbc.Card(
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Label("Symbol"),
                                    dcc.Dropdown(
                                        id="symbol-dropdown",
                                        options=[
                                            {"label": "BTC/USDC", "value": "BTC/USDC"},
                                            {"label": "ETH/USDC", "value": "ETH/USDC"},
                                            {"label": "SOL/USDC", "value": "SOL/USDC"},
                                        ],
                                        value="BTC/USDC",
                                        clearable=False,
                                    ),
                                ],
                                width=4,
                            ),
                            dbc.Col(
                                [
                                    html.Label("Timeframe"),
                                    dcc.Dropdown(
                                        id="timeframe-dropdown",
                                        options=[
                                            {"label": "1 minute", "value": "1m"},
                                            {"label": "5 minutes", "value": "5m"},
                                            {"label": "15 minutes", "value": "15m"},
                                            {"label": "1 hour", "value": "1h"},
                                            {"label": "4 hours", "value": "4h"},
                                            {"label": "1 day", "value": "1d"},
                                        ],
                                        value="1h",
                                        clearable=False,
                                    ),
                                ],
                                width=4,
                            ),
                            dbc.Col(
                                [
                                    html.Label("Indicators"),
                                    dcc.Dropdown(
                                        id="indicators-dropdown",
                                        options=[
                                            {"label": "SMA", "value": "sma"},
                                            {"label": "EMA", "value": "ema"},
                                            {"label": "MACD", "value": "macd"},
                                            {"label": "RSI", "value": "rsi"},
                                            {"label": "Bollinger Bands", "value": "bb"},
                                        ],
                                        value=["sma", "macd"],
                                        multi=True,
                                    ),
                                ],
                                width=4,
                            ),
                        ]
                    ),
                ]
            ),
            className="mb-4",
        )
        
        # Chart
        chart = dbc.Card(
            dbc.CardBody(
                [
                    dcc.Loading(
                        id="loading-chart",
                        type="circle",
                        children=[
                            dcc.Graph(
                                id="price-chart",
                                style={"height": "500px"},
                                config={"displayModeBar": True, "scrollZoom": True},
                            ),
                        ],
                    ),
                ]
            ),
            className="mb-4",
        )
        
        # AI Insights
        ai_insights = dbc.Card(
            [
                dbc.CardHeader("AI Strategic Insights"),
                dbc.CardBody(
                    [
                        html.Div(id="ai-insights-content"),
                        dcc.Interval(
                            id="ai-insights-interval",
                            interval=5000,  # 5 seconds
                            n_intervals=0,
                        ),
                    ]
                ),
            ],
            className="mb-4",
        )
        
        # Recent Trades
        recent_trades = dbc.Card(
            [
                dbc.CardHeader("Recent Trades"),
                dbc.CardBody(
                    [
                        html.Div(id="recent-trades-content"),
                        dcc.Interval(
                            id="recent-trades-interval",
                            interval=5000,  # 5 seconds
                            n_intervals=0,
                        ),
                    ]
                ),
            ],
            className="mb-4",
        )
        
        # Performance Metrics
        performance_metrics = dbc.Card(
            [
                dbc.CardHeader("Performance Metrics"),
                dbc.CardBody(
                    [
                        html.Div(id="performance-metrics-content"),
                        dcc.Interval(
                            id="performance-metrics-interval",
                            interval=10000,  # 10 seconds
                            n_intervals=0,
                        ),
                    ]
                ),
            ],
            className="mb-4",
        )
        
        # System Status
        system_status = dbc.Card(
            [
                dbc.CardHeader("System Status"),
                dbc.CardBody(
                    [
                        html.Div(id="system-status-content"),
                        dcc.Interval(
                            id="system-status-interval",
                            interval=2000,  # 2 seconds
                            n_intervals=0,
                        ),
                    ]
                ),
            ],
            className="mb-4",
        )
        
        # Main layout
        self.app.layout = html.Div(
            [
                header,
                dbc.Container(
                    [
                        chart_controls,
                        chart,
                        dbc.Row(
                            [
                                dbc.Col(ai_insights, width=6),
                                dbc.Col(recent_trades, width=6),
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(performance_metrics, width=6),
                                dbc.Col(system_status, width=6),
                            ]
                        ),
                        # Hidden div for storing data
                        html.Div(id="chart-data-store", style={"display": "none"}),
                        # Interval for updating chart
                        dcc.Interval(
                            id="chart-update-interval",
                            interval=5000,  # 5 seconds
                            n_intervals=0,
                        ),
                    ],
                    fluid=True,
                ),
            ]
        )
    
    def _configure_callbacks(self):
        """Configure dashboard callbacks."""
        # Symbol and timeframe selection callback
        @self.app.callback(
            Output("chart-data-store", "children"),
            [
                Input("symbol-dropdown", "value"),
                Input("timeframe-dropdown", "value"),
                Input("chart-update-interval", "n_intervals"),
            ],
        )
        def update_chart_data(symbol, timeframe, n_intervals):
            """Update chart data based on selected symbol and timeframe."""
            self.active_symbol = symbol
            self.active_timeframe = timeframe
            
            # Get chart data
            if self.chart_visualization:
                chart_data = self.chart_visualization.get_chart_data(symbol, timeframe)
                return json.dumps(chart_data)
            
            # Fallback to empty data
            return json.dumps({
                "symbol": symbol,
                "timeframe": timeframe,
                "klines": [],
                "indicators": {},
                "markers": []
            })
        
        # Chart update callback
        @self.app.callback(
            Output("price-chart", "figure"),
            [
                Input("chart-data-store", "children"),
                Input("indicators-dropdown", "value"),
            ],
        )
        def update_price_chart(chart_data_json, indicators):
            """Update price chart with selected indicators."""
            if not chart_data_json:
                return go.Figure()
            
            try:
                chart_data = json.loads(chart_data_json)
                
                # Extract data
                symbol = chart_data.get("symbol", self.active_symbol)
                timeframe = chart_data.get("timeframe", self.active_timeframe)
                klines = chart_data.get("klines", [])
                indicators_data = chart_data.get("indicators", {})
                markers = chart_data.get("markers", [])
                
                # Create figure with secondary y-axis for volume
                fig = make_subplots(
                    rows=2,
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    row_heights=[0.8, 0.2],
                    specs=[
                        [{"secondary_y": True}],
                        [{"secondary_y": False}]
                    ]
                )
                
                # Check if we have klines data
                if not klines:
                    fig.add_annotation(
                        text="No data available",
                        xref="paper",
                        yref="paper",
                        x=0.5,
                        y=0.5,
                        showarrow=False,
                        font=dict(size=20)
                    )
                    
                    fig.update_layout(
                        title=f"{symbol} - {timeframe}",
                        template="plotly_dark",
                        showlegend=True,
                        height=500
                    )
                    
                    return fig
                
                # Convert klines to DataFrame for easier processing
                df = pd.DataFrame(klines)
                
                # Convert timestamp to datetime
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                
                # Add candlestick chart
                fig.add_trace(
                    go.Candlestick(
                        x=df["timestamp"],
                        open=df["open"],
                        high=df["high"],
                        low=df["low"],
                        close=df["close"],
                        name="Price",
                    ),
                    row=1,
                    col=1,
                )
                
                # Add volume bars
                fig.add_trace(
                    go.Bar(
                        x=df["timestamp"],
                        y=df["volume"],
                        name="Volume",
                        marker_color="rgba(128, 128, 128, 0.5)",
                    ),
                    row=2,
                    col=1,
                )
                
                # Add selected indicators
                if indicators:
                    if "sma" in indicators:
                        # Add SMA indicators
                        for period, color in [(20, "rgba(255, 255, 0, 0.7)"), (50, "rgba(255, 165, 0, 0.7)"), (200, "rgba(255, 0, 0, 0.7)")]:
                            indicator_key = f"sma_{period}"
                            if indicator_key in indicators_data:
                                sma_data = indicators_data[indicator_key]
                                timestamps = [pd.to_datetime(ts) for ts in sma_data.keys()]
                                values = list(sma_data.values())
                                
                                fig.add_trace(
                                    go.Scatter(
                                        x=timestamps,
                                        y=values,
                                        mode="lines",
                                        name=f"SMA {period}",
                                        line=dict(color=color, width=1),
                                    ),
                                    row=1,
                                    col=1,
                                )
                    
                    if "ema" in indicators:
                        # Add EMA indicators
                        for period, color in [(12, "rgba(0, 255, 255, 0.7)"), (26, "rgba(128, 0, 128, 0.7)")]:
                            indicator_key = f"ema_{period}"
                            if indicator_key in indicators_data:
                                ema_data = indicators_data[indicator_key]
                                timestamps = [pd.to_datetime(ts) for ts in ema_data.keys()]
                                values = list(ema_data.values())
                                
                                fig.add_trace(
                                    go.Scatter(
                                        x=timestamps,
                                        y=values,
                                        mode="lines",
                                        name=f"EMA {period}",
                                        line=dict(color=color, width=1),
                                    ),
                                    row=1,
                                    col=1,
                                )
                    
                    if "bb" in indicators:
                        # Add Bollinger Bands
                        for band, color in [("bb_upper", "rgba(0, 128, 0, 0.5)"), ("bb_middle", "rgba(0, 128, 0, 0.7)"), ("bb_lower", "rgba(0, 128, 0, 0.5)")]:
                            if band in indicators_data:
                                bb_data = indicators_data[band]
                                timestamps = [pd.to_datetime(ts) for ts in bb_data.keys()]
                                values = list(bb_data.values())
                                
                                fig.add_trace(
                                    go.Scatter(
                                        x=timestamps,
                                        y=values,
                                        mode="lines",
                                        name=band.replace("bb_", "BB ").title(),
                                        line=dict(color=color, width=1),
                                    ),
                                    row=1,
                                    col=1,
                                )
                    
                    if "macd" in indicators:
                        # Add MACD
                        for macd_component, color in [("macd", "rgba(0, 0, 255, 0.7)"), ("macd_signal", "rgba(255, 0, 0, 0.7)")]:
                            if macd_component in indicators_data:
                                macd_data = indicators_data[macd_component]
                                timestamps = [pd.to_datetime(ts) for ts in macd_data.keys()]
                                values = list(macd_data.values())
                                
                                fig.add_trace(
                                    go.Scatter(
                                        x=timestamps,
                                        y=values,
                                        mode="lines",
                                        name=macd_component.replace("_", " ").upper(),
                                        line=dict(color=color, width=1),
                                    ),
                                    row=2,
                                    col=1,
                                )
                        
                        # Add MACD histogram
                        if "macd_histogram" in indicators_data:
                            hist_data = indicators_data["macd_histogram"]
                            timestamps = [pd.to_datetime(ts) for ts in hist_data.keys()]
                            values = list(hist_data.values())
                            
                            # Create colors based on values
                            colors = ["rgba(0, 255, 0, 0.7)" if val >= 0 else "rgba(255, 0, 0, 0.7)" for val in values]
                            
                            fig.add_trace(
                                go.Bar(
                                    x=timestamps,
                                    y=values,
                                    name="MACD Histogram",
                                    marker_color=colors,
                                ),
                                row=2,
                                col=1,
                            )
                    
                    if "rsi" in indicators:
                        # Add RSI
                        if "rsi" in indicators_data:
                            rsi_data = indicators_data["rsi"]
                            timestamps = [pd.to_datetime(ts) for ts in rsi_data.keys()]
                            values = list(rsi_data.values())
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=timestamps,
                                    y=values,
                                    mode="lines",
                                    name="RSI",
                                    line=dict(color="rgba(255, 165, 0, 0.7)", width=1),
                                ),
                                row=2,
                                col=1,
                            )
                            
                            # Add RSI overbought/oversold lines
                            fig.add_shape(
                                type="line",
                                x0=min(timestamps),
                                x1=max(timestamps),
                                y0=70,
                                y1=70,
                                line=dict(color="rgba(255, 0, 0, 0.5)", width=1, dash="dash"),
                                row=2,
                                col=1,
                            )
                            
                            fig.add_shape(
                                type="line",
                                x0=min(timestamps),
                                x1=max(timestamps),
                                y0=30,
                                y1=30,
                                line=dict(color="rgba(0, 255, 0, 0.5)", width=1, dash="dash"),
                                row=2,
                                col=1,
                            )
                
                # Add markers for strategic decisions, patterns, etc.
                for marker in markers:
                    marker_type = marker.get("type")
                    timestamp = pd.to_datetime(marker.get("timestamp"))
                    price = marker.get("price")
                    
                    # Skip markers without price (can't place on chart)
                    if price is None:
                        continue
                    
                    # Determine marker properties based on type
                    if marker_type == "decision":
                        direction = marker.get("direction", "neutral")
                        decision_type = marker.get("decision_type", "unknown")
                        
                        # Set color based on direction
                        color = {
                            "bullish": "rgba(0, 255, 0, 1)",
                            "bearish": "rgba(255, 0, 0, 1)",
                            "neutral": "rgba(255, 255, 0, 1)"
                        }.get(direction, "rgba(128, 128, 128, 1)")
                        
                        # Set symbol based on decision type
                        symbol = {
                            "entry": "triangle-up",
                            "exit": "triangle-down",
                            "risk_adjustment": "circle",
                            "position_sizing": "square",
                            "market_trend": "star",
                            "pattern_confirmation": "diamond",
                            "emergency": "x"
                        }.get(decision_type, "circle")
                        
                        fig.add_trace(
                            go.Scatter(
                                x=[timestamp],
                                y=[price],
                                mode="markers",
                                marker=dict(
                                    symbol=symbol,
                                    size=12,
                                    color=color,
                                    line=dict(width=1, color="rgba(0, 0, 0, 1)")
                                ),
                                name=f"{decision_type.replace('_', ' ').title()} ({direction.title()})",
                                text=marker.get("text", ""),
                                hoverinfo="text+name"
                            ),
                            row=1,
                            col=1
                        )
                    
                    elif marker_type == "pattern":
                        pattern_type = marker.get("pattern_type", "unknown")
                        direction = marker.get("direction", "neutral")
                        confidence = marker.get("confidence", 0.5)
                        
                        # Set color based on direction and confidence
                        alpha = min(1.0, 0.5 + confidence * 0.5)
                        if direction == "bullish":
                            color = f"rgba(0, 255, 0, {alpha})"
                        elif direction == "bearish":
                            color = f"rgba(255, 0, 0, {alpha})"
                        else:
                            color = f"rgba(255, 255, 0, {alpha})"
                        
                        fig.add_trace(
                            go.Scatter(
                                x=[timestamp],
                                y=[price],
                                mode="markers+text",
                                marker=dict(
                                    symbol="star",
                                    size=16,
                                    color=color,
                                    line=dict(width=1, color="rgba(0, 0, 0, 1)")
                                ),
                                text=pattern_type.replace("_", " ").title(),
                                textposition="top center",
                                name=f"{pattern_type.replace('_', ' ').title()} ({confidence:.2f})",
                                hoverinfo="text+name"
                            ),
                            row=1,
                            col=1
                        )
                    
                    elif marker_type == "alert":
                        alert_type = marker.get("alert_type", "unknown")
                        severity = marker.get("severity", "medium")
                        
                        # Set color based on severity
                        color = {
                            "low": "rgba(0, 0, 255, 1)",
                            "medium": "rgba(255, 165, 0, 1)",
                            "high": "rgba(255, 0, 0, 1)",
                            "critical": "rgba(128, 0, 0, 1)"
                        }.get(severity, "rgba(128, 128, 128, 1)")
                        
                        fig.add_trace(
                            go.Scatter(
                                x=[timestamp],
                                y=[price],
                                mode="markers",
                                marker=dict(
                                    symbol="x",
                                    size=14,
                                    color=color,
                                    line=dict(width=1, color="rgba(0, 0, 0, 1)")
                                ),
                                name=f"{alert_type.replace('_', ' ').title()} ({severity.title()})",
                                text=marker.get("text", ""),
                                hoverinfo="text+name"
                            ),
                            row=1,
                            col=1
                        )
                
                # Update layout
                fig.update_layout(
                    title=f"{symbol} - {timeframe}",
                    template="plotly_dark",
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    height=500,
                    margin=dict(l=50, r=50, t=50, b=50),
                    xaxis_rangeslider_visible=False
                )
                
                # Update y-axis labels
                fig.update_yaxes(title_text="Price", row=1, col=1)
                fig.update_yaxes(title_text="Volume", row=2, col=1)
                
                return fig
            
            except Exception as e:
                logger.error(f"Error updating price chart: {e}")
                
                # Return empty figure with error message
                fig = go.Figure()
                fig.add_annotation(
                    text=f"Error updating chart: {str(e)}",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=14, color="red")
                )
                
                fig.update_layout(
                    title=f"{self.active_symbol} - {self.active_timeframe}",
                    template="plotly_dark",
                    height=500
                )
                
                return fig
        
        # AI Insights update callback
        @self.app.callback(
            Output("ai-insights-content", "children"),
            [Input("ai-insights-interval", "n_intervals")],
        )
        def update_ai_insights(n_intervals):
            """Update AI insights content."""
            if not self.ai_insights:
                return html.Div("No AI insights available")
            
            # Sort insights by timestamp (newest first)
            sorted_insights = sorted(
                self.ai_insights,
                key=lambda x: x.get("timestamp", ""),
                reverse=True
            )
            
            # Limit to 10 most recent insights
            recent_insights = sorted_insights[:10]
            
            # Create cards for each insight
            cards = []
            for insight in recent_insights:
                timestamp = insight.get("timestamp", "")
                if timestamp:
                    try:
                        dt = datetime.fromisoformat(timestamp)
                        timestamp = dt.strftime("%Y-%m-%d %H:%M:%S")
                    except (ValueError, TypeError):
                        pass
                
                insight_type = insight.get("type", "unknown")
                symbol = insight.get("symbol", "")
                content = insight.get("content", "")
                confidence = insight.get("confidence", 0)
                
                # Determine card color based on insight type
                color = {
                    "strategic_decision": "primary",
                    "pattern_recognition": "success",
                    "risk_alert": "danger",
                    "market_analysis": "info"
                }.get(insight_type, "secondary")
                
                card = dbc.Card(
                    [
                        dbc.CardHeader(
                            [
                                html.Span(f"{insight_type.replace('_', ' ').title()} - {symbol}", className="fw-bold"),
                                html.Span(f" ({timestamp})", className="text-muted ms-2 small"),
                            ]
                        ),
                        dbc.CardBody(
                            [
                                html.P(content),
                                dbc.Progress(
                                    value=int(confidence * 100),
                                    color="success" if confidence >= 0.7 else "warning" if confidence >= 0.4 else "danger",
                                    className="mb-0",
                                    label=f"Confidence: {confidence:.2f}",
                                    style={"height": "20px"}
                                ),
                            ]
                        ),
                    ],
                    color=color,
                    outline=True,
                    className="mb-2",
                )
                
                cards.append(card)
            
            return html.Div(cards)
        
        # Recent Trades update callback
        @self.app.callback(
            Output("recent-trades-content", "children"),
            [Input("recent-trades-interval", "n_intervals")],
        )
        def update_recent_trades(n_intervals):
            """Update recent trades content."""
            if not self.recent_trades:
                return html.Div("No recent trades available")
            
            # Sort trades by timestamp (newest first)
            sorted_trades = sorted(
                self.recent_trades,
                key=lambda x: x.get("timestamp", ""),
                reverse=True
            )
            
            # Limit to 10 most recent trades
            recent_trades = sorted_trades[:10]
            
            # Create table for trades
            table_header = [
                html.Thead(
                    html.Tr(
                        [
                            html.Th("Time"),
                            html.Th("Symbol"),
                            html.Th("Side"),
                            html.Th("Price"),
                            html.Th("Quantity"),
                            html.Th("Status"),
                        ]
                    )
                )
            ]
            
            rows = []
            for trade in recent_trades:
                timestamp = trade.get("timestamp", "")
                if timestamp:
                    try:
                        dt = datetime.fromisoformat(timestamp)
                        timestamp = dt.strftime("%H:%M:%S")
                    except (ValueError, TypeError):
                        pass
                
                symbol = trade.get("symbol", "")
                side = trade.get("side", "")
                price = trade.get("price", 0)
                quantity = trade.get("quantity", 0)
                status = trade.get("status", "")
                
                # Determine row color based on side and status
                row_color = ""
                if status == "filled":
                    row_color = "table-success" if side == "buy" else "table-danger"
                elif status == "cancelled":
                    row_color = "table-warning"
                elif status == "rejected":
                    row_color = "table-danger"
                
                row = html.Tr(
                    [
                        html.Td(timestamp),
                        html.Td(symbol),
                        html.Td(side.upper(), className="text-success" if side == "buy" else "text-danger"),
                        html.Td(f"{price:.2f}"),
                        html.Td(f"{quantity:.6f}"),
                        html.Td(status.title()),
                    ],
                    className=row_color,
                )
                
                rows.append(row)
            
            table_body = [html.Tbody(rows)]
            
            table = dbc.Table(
                table_header + table_body,
                striped=True,
                bordered=True,
                hover=True,
                size="sm",
                className="mb-0",
            )
            
            return table
        
        # Performance Metrics update callback
        @self.app.callback(
            Output("performance-metrics-content", "children"),
            [Input("performance-metrics-interval", "n_intervals")],
        )
        def update_performance_metrics(n_intervals):
            """Update performance metrics content."""
            if not self.performance_metrics:
                return html.Div("No performance metrics available")
            
            # Extract metrics
            total_trades = self.performance_metrics.get("total_trades", 0)
            win_rate = self.performance_metrics.get("win_rate", 0)
            profit_loss = self.performance_metrics.get("profit_loss", 0)
            profit_factor = self.performance_metrics.get("profit_factor", 0)
            max_drawdown = self.performance_metrics.get("max_drawdown", 0)
            sharpe_ratio = self.performance_metrics.get("sharpe_ratio", 0)
            
            # Create metrics cards
            cards = [
                dbc.Card(
                    [
                        dbc.CardBody(
                            [
                                html.H5("Total Trades", className="card-title"),
                                html.H3(f"{total_trades}", className="card-text text-center"),
                            ]
                        ),
                    ],
                    className="mb-2",
                ),
                dbc.Card(
                    [
                        dbc.CardBody(
                            [
                                html.H5("Win Rate", className="card-title"),
                                html.H3(
                                    f"{win_rate:.2%}",
                                    className=f"card-text text-center {'text-success' if win_rate >= 0.5 else 'text-danger'}",
                                ),
                            ]
                        ),
                    ],
                    className="mb-2",
                ),
                dbc.Card(
                    [
                        dbc.CardBody(
                            [
                                html.H5("Profit/Loss", className="card-title"),
                                html.H3(
                                    f"{profit_loss:.2f} USDC",
                                    className=f"card-text text-center {'text-success' if profit_loss >= 0 else 'text-danger'}",
                                ),
                            ]
                        ),
                    ],
                    className="mb-2",
                ),
                dbc.Card(
                    [
                        dbc.CardBody(
                            [
                                html.H5("Profit Factor", className="card-title"),
                                html.H3(
                                    f"{profit_factor:.2f}",
                                    className=f"card-text text-center {'text-success' if profit_factor >= 1 else 'text-danger'}",
                                ),
                            ]
                        ),
                    ],
                    className="mb-2",
                ),
                dbc.Card(
                    [
                        dbc.CardBody(
                            [
                                html.H5("Max Drawdown", className="card-title"),
                                html.H3(
                                    f"{max_drawdown:.2%}",
                                    className="card-text text-center text-danger",
                                ),
                            ]
                        ),
                    ],
                    className="mb-2",
                ),
                dbc.Card(
                    [
                        dbc.CardBody(
                            [
                                html.H5("Sharpe Ratio", className="card-title"),
                                html.H3(
                                    f"{sharpe_ratio:.2f}",
                                    className=f"card-text text-center {'text-success' if sharpe_ratio >= 1 else 'text-warning' if sharpe_ratio >= 0 else 'text-danger'}",
                                ),
                            ]
                        ),
                    ],
                    className="mb-2",
                ),
            ]
            
            return dbc.Row([dbc.Col(card, width=4) for card in cards])
        
        # System Status update callback
        @self.app.callback(
            Output("system-status-content", "children"),
            [Input("system-status-interval", "n_intervals")],
        )
        def update_system_status(n_intervals):
            """Update system status content."""
            # Extract status
            flash_trading_active = self.system_status.get("flash_trading_active", False)
            paper_trading_active = self.system_status.get("paper_trading_active", False)
            last_update = self.system_status.get("last_update", "")
            
            if last_update:
                try:
                    dt = datetime.fromisoformat(last_update)
                    last_update = dt.strftime("%Y-%m-%d %H:%M:%S")
                except (ValueError, TypeError):
                    pass
            
            # Create status indicators
            status_items = [
                dbc.ListGroupItem(
                    [
                        html.Div(
                            [
                                html.Span("Flash Trading: ", className="fw-bold"),
                                html.Span(
                                    "Active" if flash_trading_active else "Inactive",
                                    className=f"{'text-success' if flash_trading_active else 'text-danger'}",
                                ),
                            ]
                        ),
                        html.Div(
                            dbc.Badge(
                                "ON" if flash_trading_active else "OFF",
                                color="success" if flash_trading_active else "danger",
                                className="ms-1",
                            )
                        ),
                    ],
                    className="d-flex justify-content-between align-items-center",
                ),
                dbc.ListGroupItem(
                    [
                        html.Div(
                            [
                                html.Span("Paper Trading: ", className="fw-bold"),
                                html.Span(
                                    "Active" if paper_trading_active else "Inactive",
                                    className=f"{'text-success' if paper_trading_active else 'text-danger'}",
                                ),
                            ]
                        ),
                        html.Div(
                            dbc.Badge(
                                "ON" if paper_trading_active else "OFF",
                                color="success" if paper_trading_active else "danger",
                                className="ms-1",
                            )
                        ),
                    ],
                    className="d-flex justify-content-between align-items-center",
                ),
                dbc.ListGroupItem(
                    [
                        html.Div(
                            [
                                html.Span("Last Update: ", className="fw-bold"),
                                html.Span(last_update, className="text-muted"),
                            ]
                        ),
                    ]
                ),
            ]
            
            return dbc.ListGroup(status_items)
    
    async def _handle_order_update(self, topic: str, data: Dict[str, Any]):
        """
        Handle order update event.
        
        Args:
            topic: Event topic
            data: Event data
        """
        # Add to recent trades
        self.recent_trades.append(data)
        
        # Limit to 100 recent trades
        if len(self.recent_trades) > 100:
            self.recent_trades = self.recent_trades[-100:]
        
        logger.info(f"Order update received: {data.get('symbol')} {data.get('side')} {data.get('status')}")
    
    async def _handle_status_update(self, topic: str, data: Dict[str, Any]):
        """
        Handle status update event.
        
        Args:
            topic: Event topic
            data: Event data
        """
        # Update system status
        self.system_status.update(data)
        
        logger.info(f"Status update received: {data}")
    
    async def _handle_strategic_decision(self, topic: str, data: Dict[str, Any]):
        """
        Handle strategic decision event.
        
        Args:
            topic: Event topic
            data: Event data
        """
        # Create AI insight from strategic decision
        insight = {
            "type": "strategic_decision",
            "symbol": data.get("symbol", ""),
            "timestamp": data.get("timestamp", datetime.now().isoformat()),
            "content": data.get("summary", ""),
            "confidence": data.get("confidence", 0.5),
            "direction": data.get("direction", "neutral"),
            "decision_type": data.get("decision_type", "unknown")
        }
        
        # Add to AI insights
        self.ai_insights.append(insight)
        
        # Limit to 100 insights
        if len(self.ai_insights) > 100:
            self.ai_insights = self.ai_insights[-100:]
        
        logger.info(f"Strategic decision received: {data.get('decision_type')} for {data.get('symbol')}")
    
    async def _handle_indicator_signal(self, topic: str, data: Dict[str, Any]):
        """
        Handle indicator signal event.
        
        Args:
            topic: Event topic
            data: Event data
        """
        # Create AI insight from indicator signal
        indicator = data.get("indicator", "unknown")
        signal = data.get("signal", "unknown")
        direction = data.get("direction", "neutral")
        value = data.get("value", 0)
        
        content = f"{indicator} {signal} signal detected with value {value:.2f}"
        
        insight = {
            "type": "market_analysis",
            "symbol": data.get("symbol", ""),
            "timestamp": data.get("timestamp", datetime.now().isoformat()),
            "content": content,
            "confidence": data.get("confidence", 0.5),
            "direction": direction,
            "indicator": indicator
        }
        
        # Add to AI insights
        self.ai_insights.append(insight)
        
        # Limit to 100 insights
        if len(self.ai_insights) > 100:
            self.ai_insights = self.ai_insights[-100:]
        
        logger.info(f"Indicator signal received: {indicator} {signal} for {data.get('symbol')}")
    
    async def _handle_pattern_detected(self, topic: str, data: Dict[str, Any]):
        """
        Handle pattern detected event.
        
        Args:
            topic: Event topic
            data: Event data
        """
        # Create AI insight from pattern detection
        pattern_type = data.get("pattern_type", "unknown")
        direction = data.get("direction", "neutral")
        confidence = data.get("confidence", 0.5)
        
        content = f"{pattern_type.replace('_', ' ').title()} pattern detected with {confidence:.2f} confidence"
        
        insight = {
            "type": "pattern_recognition",
            "symbol": data.get("symbol", ""),
            "timestamp": data.get("timestamp", datetime.now().isoformat()),
            "content": content,
            "confidence": confidence,
            "direction": direction,
            "pattern_type": pattern_type
        }
        
        # Add to AI insights
        self.ai_insights.append(insight)
        
        # Limit to 100 insights
        if len(self.ai_insights) > 100:
            self.ai_insights = self.ai_insights[-100:]
        
        logger.info(f"Pattern detected: {pattern_type} for {data.get('symbol')}")
    
    async def _handle_performance_update(self, topic: str, data: Dict[str, Any]):
        """
        Handle performance metrics update event.
        
        Args:
            topic: Event topic
            data: Event data
        """
        # Update performance metrics
        self.performance_metrics.update(data)
        
        logger.info(f"Performance metrics updated: {data}")
    
    def run(self, host="0.0.0.0", port=8050, debug=False):
        """
        Run the dashboard server.
        
        Args:
            host: Host to run the server on
            port: Port to run the server on
            debug: Whether to run in debug mode
        """
        logger.info(f"Starting dashboard server on {host}:{port}")
        self.app.run_server(host=host, port=port, debug=debug)


# For testing
def test():
    """Test function."""
    # Create dashboard
    dashboard = TradingDashboard()
    
    # Add some sample data
    dashboard.ai_insights = [
        {
            "type": "strategic_decision",
            "symbol": "BTC/USDC",
            "timestamp": datetime.now().isoformat(),
            "content": "Enter long position based on bullish pattern",
            "confidence": 0.85,
            "direction": "bullish",
            "decision_type": "entry"
        },
        {
            "type": "pattern_recognition",
            "symbol": "BTC/USDC",
            "timestamp": (datetime.now() - timedelta(minutes=5)).isoformat(),
            "content": "Double Bottom pattern detected with 0.75 confidence",
            "confidence": 0.75,
            "direction": "bullish",
            "pattern_type": "double_bottom"
        }
    ]
    
    dashboard.recent_trades = [
        {
            "symbol": "BTC/USDC",
            "side": "buy",
            "price": 50000.0,
            "quantity": 0.1,
            "status": "filled",
            "timestamp": datetime.now().isoformat()
        },
        {
            "symbol": "ETH/USDC",
            "side": "sell",
            "price": 3000.0,
            "quantity": 1.0,
            "status": "filled",
            "timestamp": (datetime.now() - timedelta(minutes=10)).isoformat()
        }
    ]
    
    dashboard.performance_metrics = {
        "total_trades": 100,
        "win_rate": 0.65,
        "profit_loss": 1250.75,
        "profit_factor": 1.8,
        "max_drawdown": 0.15,
        "sharpe_ratio": 1.2
    }
    
    dashboard.system_status = {
        "flash_trading_active": True,
        "paper_trading_active": False,
        "last_update": datetime.now().isoformat()
    }
    
    # Run dashboard
    dashboard.run(debug=True)


if __name__ == "__main__":
    # Run test
    test()
