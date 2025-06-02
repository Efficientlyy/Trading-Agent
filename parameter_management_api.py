#!/usr/bin/env python
"""
Parameter Management API - Backend implementation for the Trading-Agent system.

This module provides API endpoints for managing configurable parameters across
all modules of the Trading-Agent system. It handles parameter retrieval, updates,
validation, presets, and security.
"""

import os
import json
import logging
import datetime
from typing import Dict, Any, List, Optional, Union
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import copy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Constants
PARAMETERS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config')
DEFAULT_PARAMETERS_FILE = os.path.join(PARAMETERS_DIR, 'default_parameters.json')
CURRENT_PARAMETERS_FILE = os.path.join(PARAMETERS_DIR, 'current_parameters.json')
PRESETS_DIR = os.path.join(PARAMETERS_DIR, 'presets')
AUDIT_LOG_FILE = os.path.join(PARAMETERS_DIR, 'parameter_audit.log')

# Ensure directories exist
os.makedirs(PARAMETERS_DIR, exist_ok=True)
os.makedirs(PRESETS_DIR, exist_ok=True)

# Parameter validation rules
VALIDATION_RULES = {
    "numeric_range": lambda value, min_val, max_val: isinstance(value, (int, float)) and min_val <= value <= max_val,
    "option_valid": lambda value, options: value in options,
    "boolean_valid": lambda value: isinstance(value, bool),
    "string_valid": lambda value: isinstance(value, str) and len(value) > 0,
    "array_valid": lambda value, options: isinstance(value, list) and all(item in options for item in value),
    "dependency_valid": lambda value, dependent_param, dependent_values: True if dependent_param not in request.json else request.json[dependent_param] in dependent_values
}

# Default parameter presets
DEFAULT_PRESETS = {
    "conservative": {
        "description": "Low risk settings with smaller position sizes and fewer trades",
        "risk_level": "low",
        "max_portfolio_risk_percent": 1.0,
        "max_position_size_usd": 500,
        "max_position_size_percent": 2.5,
        "stop_loss_percent": 1.5,
        "take_profit_percent": 3.0,
        "max_trades_per_day": 3,
        "decision_confidence_threshold": 0.9
    },
    "balanced": {
        "description": "Moderate risk and reward with balanced settings",
        "risk_level": "medium",
        "max_portfolio_risk_percent": 2.0,
        "max_position_size_usd": 1000,
        "max_position_size_percent": 5.0,
        "stop_loss_percent": 2.0,
        "take_profit_percent": 4.0,
        "max_trades_per_day": 5,
        "decision_confidence_threshold": 0.85
    },
    "aggressive": {
        "description": "Higher risk tolerance with larger position sizes and more trades",
        "risk_level": "high",
        "max_portfolio_risk_percent": 3.0,
        "max_position_size_usd": 2000,
        "max_position_size_percent": 10.0,
        "stop_loss_percent": 3.0,
        "take_profit_percent": 6.0,
        "max_trades_per_day": 10,
        "decision_confidence_threshold": 0.75
    },
    "high_frequency": {
        "description": "Optimized for frequent trading with shorter timeframes",
        "risk_level": "medium",
        "max_portfolio_risk_percent": 1.5,
        "max_position_size_usd": 500,
        "max_position_size_percent": 3.0,
        "stop_loss_percent": 1.0,
        "take_profit_percent": 2.0,
        "max_trades_per_day": 20,
        "primary_timeframes": ["1m", "5m", "15m"],
        "decision_confidence_threshold": 0.8
    },
    "swing_trading": {
        "description": "Optimized for longer-term positions with wider stops",
        "risk_level": "medium",
        "max_portfolio_risk_percent": 2.5,
        "max_position_size_usd": 1500,
        "max_position_size_percent": 7.5,
        "stop_loss_percent": 4.0,
        "take_profit_percent": 8.0,
        "max_trades_per_day": 2,
        "primary_timeframes": ["1h", "4h", "1d"],
        "decision_confidence_threshold": 0.9
    }
}

# Default parameters
DEFAULT_PARAMETERS = {
    "market_data": {
        "active_trading_pairs": ["BTC/USDC", "ETH/USDC", "SOL/USDC"],
        "primary_timeframes": ["5m", "15m", "1h", "4h"],
        "historical_candles_count": 1000,
        "websocket_enabled": True,
        "data_update_interval_sec": 60,
        "enable_order_book_data": True,
        "order_book_depth": 20,
        "enable_trade_data": True,
        "recent_trades_limit": 50,
        "data_caching_enabled": True,
        "cache_ttl_sec": 300,
        "retry_attempts": 3,
        "retry_delay_ms": 1000
    },
    "pattern_recognition": {
        "enabled_patterns": ["head_and_shoulders", "double_top", "double_bottom", "triangle", "wedge", "channel", "support_resistance"],
        "min_pattern_confidence": 0.75,
        "pattern_lookback_periods": 50,
        "enable_ml_detection": True,
        "ml_confidence_threshold": 0.8,
        "ml_model_update_interval_hours": 24,
        "enable_adaptive_thresholds": True,
        "volatility_adjustment_factor": 0.5,
        "enable_multi_timeframe_confirmation": True,
        "confirmation_timeframes": ["15m", "1h"],
        "pattern_visualization_enabled": True,
        "max_concurrent_patterns": 5
    },
    "signal_generation": {
        "min_signal_confidence": 0.8,
        "signal_expiry_periods": 3,
        "enable_signal_filtering": True,
        "enable_signal_stacking": True,
        "max_stacked_signals": 3,
        "signal_reinforcement_factor": 1.2,
        "enable_contrarian_signals": False,
        "overbought_threshold": 70,
        "oversold_threshold": 30,
        "enable_volume_confirmation": True,
        "min_volume_percentile": 60,
        "enable_trend_filter": True,
        "trend_determination_method": "ema_cross",
        "trend_lookback_periods": 20
    },
    "decision_making": {
        "decision_confidence_threshold": 0.85,
        "enable_position_sizing": True,
        "max_trades_per_day": 5,
        "min_time_between_trades_min": 60,
        "enable_reinforcement_learning": True,
        "exploration_rate": 0.1,
        "learning_rate": 0.001,
        "reward_discount_factor": 0.95,
        "use_market_state_features": True,
        "use_technical_indicators": True,
        "use_sentiment_data": False,
        "enable_decision_explanations": True,
        "decision_history_size": 100,
        "enable_auto_training": True,
        "training_interval_hours": 24
    },
    "order_execution": {
        "default_order_type": "market",
        "enable_smart_routing": True,
        "slippage_tolerance_percent": 0.5,
        "enable_iceberg_orders": False,
        "iceberg_order_threshold": 5000,
        "iceberg_display_size_percent": 10,
        "enable_twap_execution": False,
        "twap_order_threshold": 10000,
        "twap_interval_minutes": 30,
        "twap_slices": 5,
        "enable_retry_on_failure": True,
        "max_retry_attempts": 3,
        "retry_delay_seconds": 5,
        "enable_partial_fills": True,
        "min_fill_percent": 90,
        "cancel_after_seconds": 300
    },
    "risk_management": {
        "risk_level": "medium",
        "max_portfolio_risk_percent": 2.0,
        "max_position_size_usd": 1000,
        "max_position_size_percent": 5.0,
        "max_total_exposure_percent": 25.0,
        "stop_loss_percent": 2.0,
        "take_profit_percent": 4.0,
        "enable_trailing_stop": True,
        "trailing_stop_activation_percent": 1.0,
        "trailing_stop_distance_percent": 1.5,
        "max_daily_drawdown_percent": 5.0,
        "max_trades_per_asset_daily": 3,
        "enable_circuit_breakers": True,
        "price_circuit_breaker_percent": 5.0,
        "volume_circuit_breaker_factor": 3.0,
        "circuit_breaker_cooldown_minutes": 60,
        "enable_correlation_risk_control": True,
        "max_correlation_exposure": 15.0,
        "enable_volatility_adjustment": True,
        "volatility_lookback_periods": 20,
        "enable_overnight_position_control": True,
        "max_overnight_exposure_percent": 15.0
    },
    "visualization": {
        "default_chart_timeframe": "1h",
        "default_chart_type": "candlestick",
        "default_chart_theme": "dark",
        "enable_technical_indicators": True,
        "default_indicators": ["MA", "RSI", "MACD"],
        "max_indicators": 5,
        "enable_pattern_visualization": True,
        "enable_signal_markers": True,
        "enable_trade_markers": True,
        "enable_price_alerts": True,
        "enable_volume_profile": True,
        "enable_depth_chart": True,
        "chart_update_interval_ms": 1000,
        "max_visible_candles": 100,
        "enable_multi_chart_view": True,
        "enable_drawing_tools": True,
        "enable_chart_animations": True
    },
    "monitoring_dashboard": {
        "dashboard_refresh_interval_sec": 5,
        "enable_system_status_monitoring": True,
        "enable_risk_metrics_display": True,
        "enable_performance_metrics": True,
        "enable_trade_history": True,
        "trade_history_limit": 50,
        "enable_log_display": True,
        "log_display_limit": 100,
        "enable_alert_notifications": True,
        "alert_notification_methods": ["dashboard", "email"],
        "critical_alert_threshold": "error",
        "enable_email_reports": False,
        "email_report_frequency": "daily",
        "enable_performance_charts": True,
        "enable_resource_monitoring": True,
        "resource_alert_threshold_percent": 80
    },
    "performance_optimization": {
        "enable_data_caching": True,
        "cache_size_mb": 100,
        "enable_batch_processing": True,
        "batch_size": 100,
        "enable_parallel_processing": True,
        "max_parallel_threads": 4,
        "enable_websocket_optimization": True,
        "websocket_reconnect_interval_sec": 5,
        "enable_lazy_loading": True,
        "enable_memory_optimization": True,
        "memory_cleanup_interval_min": 30,
        "enable_query_optimization": True,
        "query_cache_size": 50,
        "enable_compression": True,
        "compression_level": 6
    },
    "system_settings": {
        "trading_enabled": False,
        "paper_trading_mode": True,
        "base_currency": "USDC",
        "log_level": "info",
        "enable_telegram_notifications": False,
        "enable_email_notifications": False,
        "notification_level": "warning",
        "timezone": "UTC",
        "enable_auto_updates": True,
        "backup_interval_hours": 24,
        "backup_retention_days": 30,
        "enable_api_access": False,
        "api_access_ips": ["127.0.0.1"],
        "session_timeout_minutes": 60,
        "max_login_attempts": 5,
        "enable_2fa": False
    }
}

# Parameter metadata for validation and UI rendering
PARAMETER_METADATA = {
    "market_data.active_trading_pairs": {
        "type": "array",
        "options": ["BTC/USDC", "ETH/USDC", "SOL/USDC", "BTC/USDT", "ETH/USDT", "SOL/USDT"],
        "category": "basic",
        "description": "Trading pairs to monitor and trade"
    },
    "market_data.primary_timeframes": {
        "type": "array",
        "options": ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "3d", "1w"],
        "category": "basic",
        "description": "Primary timeframes for data collection"
    },
    "market_data.historical_candles_count": {
        "type": "numeric",
        "min": 100,
        "max": 5000,
        "category": "advanced",
        "description": "Number of historical candles to retrieve"
    },
    "risk_management.risk_level": {
        "type": "option",
        "options": ["low", "medium", "high"],
        "category": "basic",
        "description": "Overall risk level for trading operations"
    },
    "risk_management.max_portfolio_risk_percent": {
        "type": "numeric",
        "min": 0.1,
        "max": 5.0,
        "category": "basic",
        "description": "Maximum percentage of portfolio value at risk"
    },
    "risk_management.max_position_size_usd": {
        "type": "numeric",
        "min": 10,
        "max": 5000,
        "category": "basic",
        "description": "Maximum position size in USD"
    },
    "risk_management.max_position_size_percent": {
        "type": "numeric",
        "min": 0.1,
        "max": 20.0,
        "category": "basic",
        "description": "Maximum position size as percentage of portfolio"
    },
    "risk_management.max_total_exposure_percent": {
        "type": "numeric",
        "min": 1.0,
        "max": 100.0,
        "category": "advanced",
        "description": "Maximum total exposure as percentage of portfolio"
    },
    "risk_management.stop_loss_percent": {
        "type": "numeric",
        "min": 0.1,
        "max": 10.0,
        "category": "basic",
        "description": "Default stop loss percentage"
    },
    "risk_management.take_profit_percent": {
        "type": "numeric",
        "min": 0.1,
        "max": 20.0,
        "category": "basic",
        "description": "Default take profit percentage"
    },
    "risk_management.enable_trailing_stop": {
        "type": "boolean",
        "category": "basic",
        "description": "Enable trailing stop loss"
    },
    "risk_management.trailing_stop_activation_percent": {
        "type": "numeric",
        "min": 0.1,
        "max": 10.0,
        "category": "advanced",
        "description": "Profit percentage to activate trailing stop"
    },
    "risk_management.trailing_stop_distance_percent": {
        "type": "numeric",
        "min": 0.1,
        "max": 10.0,
        "category": "advanced",
        "description": "Trailing stop distance as percentage"
    },
    "risk_management.max_daily_drawdown_percent": {
        "type": "numeric",
        "min": 0.1,
        "max": 20.0,
        "category": "advanced",
        "description": "Maximum daily drawdown percentage"
    },
    "risk_management.max_trades_per_asset_daily": {
        "type": "numeric",
        "min": 1,
        "max": 50,
        "category": "advanced",
        "description": "Maximum trades per asset per day"
    },
    "risk_management.enable_circuit_breakers": {
        "type": "boolean",
        "category": "advanced",
        "description": "Enable circuit breakers for extreme market conditions"
    },
    "risk_management.price_circuit_breaker_percent": {
        "type": "numeric",
        "min": 1.0,
        "max": 20.0,
        "category": "expert",
        "description": "Price movement percentage to trigger circuit breaker"
    },
    "risk_management.volume_circuit_breaker_factor": {
        "type": "numeric",
        "min": 1.0,
        "max": 10.0,
        "category": "expert",
        "description": "Volume multiple factor to trigger circuit breaker"
    },
    "risk_management.circuit_breaker_cooldown_minutes": {
        "type": "numeric",
        "min": 1,
        "max": 1440,
        "category": "expert",
        "description": "Cooldown period after circuit breaker in minutes"
    },
    "risk_management.enable_correlation_risk_control": {
        "type": "boolean",
        "category": "advanced",
        "description": "Enable correlation-based risk control"
    },
    "risk_management.max_correlation_exposure": {
        "type": "numeric",
        "min": 1.0,
        "max": 50.0,
        "category": "expert",
        "description": "Maximum exposure to correlated assets"
    },
    "risk_management.enable_volatility_adjustment": {
        "type": "boolean",
        "category": "advanced",
        "description": "Enable volatility-based position sizing"
    },
    "risk_management.volatility_lookback_periods": {
        "type": "numeric",
        "min": 5,
        "max": 100,
        "category": "expert",
        "description": "Lookback periods for volatility calculation"
    },
    "risk_management.enable_overnight_position_control": {
        "type": "boolean",
        "category": "advanced",
        "description": "Enable overnight position control"
    },
    "risk_management.max_overnight_exposure_percent": {
        "type": "numeric",
        "min": 0.0,
        "max": 100.0,
        "category": "advanced",
        "description": "Maximum overnight exposure percentage"
    }
    # Additional metadata entries for all parameters...
}

# Helper functions
def load_parameters() -> Dict[str, Any]:
    """Load current parameters from file or create from defaults if not exists."""
    try:
        if os.path.exists(CURRENT_PARAMETERS_FILE):
            with open(CURRENT_PARAMETERS_FILE, 'r') as f:
                return json.load(f)
        else:
            # Create default parameters file if it doesn't exist
            save_parameters(DEFAULT_PARAMETERS)
            return copy.deepcopy(DEFAULT_PARAMETERS)
    except Exception as e:
        logger.error(f"Error loading parameters: {e}")
        return copy.deepcopy(DEFAULT_PARAMETERS)

def save_parameters(parameters: Dict[str, Any]) -> bool:
    """Save parameters to file."""
    try:
        os.makedirs(os.path.dirname(CURRENT_PARAMETERS_FILE), exist_ok=True)
        with open(CURRENT_PARAMETERS_FILE, 'w') as f:
            json.dump(parameters, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving parameters: {e}")
        return False

def load_presets() -> Dict[str, Dict[str, Any]]:
    """Load parameter presets."""
    presets = copy.deepcopy(DEFAULT_PRESETS)
    
    # Load custom presets from files
    try:
        for filename in os.listdir(PRESETS_DIR):
            if filename.endswith('.json'):
                preset_name = os.path.splitext(filename)[0]
                with open(os.path.join(PRESETS_DIR, filename), 'r') as f:
                    presets[preset_name] = json.load(f)
    except Exception as e:
        logger.error(f"Error loading presets: {e}")
    
    return presets

def save_preset(name: str, parameters: Dict[str, Any]) -> bool:
    """Save a parameter preset."""
    try:
        os.makedirs(PRESETS_DIR, exist_ok=True)
        with open(os.path.join(PRESETS_DIR, f"{name}.json"), 'w') as f:
            json.dump(parameters, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving preset: {e}")
        return False

def log_parameter_change(module: str, user: str, changes: Dict[str, Any]) -> None:
    """Log parameter changes to audit log."""
    try:
        timestamp = datetime.datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "user": user,
            "module": module,
            "changes": changes
        }
        
        with open(AUDIT_LOG_FILE, 'a') as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        logger.error(f"Error logging parameter change: {e}")

def validate_parameters(module: str, parameters: Dict[str, Any]) -> Dict[str, List[str]]:
    """Validate parameters against rules."""
    errors = {}
    current_params = load_parameters()
    
    # Only validate parameters for the specified module
    for param_name, param_value in parameters.items():
        full_param_name = f"{module}.{param_name}"
        
        if full_param_name in PARAMETER_METADATA:
            metadata = PARAMETER_METADATA[full_param_name]
            param_errors = []
            
            # Validate based on parameter type
            if metadata["type"] == "numeric":
                if not isinstance(param_value, (int, float)):
                    param_errors.append("Must be a number")
                elif "min" in metadata and "max" in metadata and not VALIDATION_RULES["numeric_range"](param_value, metadata["min"], metadata["max"]):
                    param_errors.append(f"Must be between {metadata['min']} and {metadata['max']}")
            
            elif metadata["type"] == "boolean":
                if not isinstance(param_value, bool):
                    param_errors.append("Must be a boolean (true/false)")
            
            elif metadata["type"] == "option":
                if not VALIDATION_RULES["option_valid"](param_value, metadata["options"]):
                    param_errors.append(f"Must be one of: {', '.join(metadata['options'])}")
            
            elif metadata["type"] == "array":
                if not isinstance(param_value, list):
                    param_errors.append("Must be an array")
                elif "options" in metadata and not VALIDATION_RULES["array_valid"](param_value, metadata["options"]):
                    param_errors.append(f"All items must be one of: {', '.join(metadata['options'])}")
            
            # Check dependencies if any
            if "depends_on" in metadata:
                dep_param = metadata["depends_on"]["param"]
                dep_values = metadata["depends_on"]["values"]
                dep_module, dep_name = dep_param.split(".")
                
                # Get the current value of the dependency parameter
                dep_current_value = None
                if dep_module in current_params and dep_name in current_params[dep_module]:
                    dep_current_value = current_params[dep_module][dep_name]
                
                # Check if the dependency parameter is being updated
                if dep_module == module and dep_name in parameters:
                    dep_current_value = parameters[dep_name]
                
                if dep_current_value is not None and dep_current_value not in dep_values:
                    param_errors.append(f"Only valid when {dep_param} is one of: {', '.join(dep_values)}")
            
            if param_errors:
                errors[param_name] = param_errors
    
    return errors

def apply_preset_to_parameters(preset_name: str) -> Dict[str, Any]:
    """Apply a preset to the current parameters."""
    presets = load_presets()
    current_params = load_parameters()
    
    if preset_name not in presets:
        return current_params
    
    preset = presets[preset_name]
    
    # Apply preset values to appropriate modules
    for module in current_params:
        for param_name, param_value in current_params[module].items():
            # If the parameter exists in the preset, update it
            if param_name in preset:
                current_params[module][param_name] = preset[param_name]
    
    # Save the updated parameters
    save_parameters(current_params)
    
    return current_params

# API Routes
@app.route('/api/parameters', methods=['GET'])
def get_all_parameters():
    """Get all parameters."""
    return jsonify(load_parameters())

@app.route('/api/parameters/<module>', methods=['GET'])
def get_module_parameters(module):
    """Get parameters for a specific module."""
    parameters = load_parameters()
    
    if module in parameters:
        return jsonify(parameters[module])
    else:
        return jsonify({"error": f"Module '{module}' not found"}), 404

@app.route('/api/parameters/<module>', methods=['PUT'])
def update_module_parameters(module):
    """Update parameters for a specific module."""
    parameters = load_parameters()
    
    if module not in parameters:
        return jsonify({"error": f"Module '{module}' not found"}), 404
    
    # Get the updated parameters from the request
    updated_params = request.json
    
    # Validate the parameters
    validation_errors = validate_parameters(module, updated_params)
    if validation_errors:
        return jsonify({"error": "Validation failed", "details": validation_errors}), 400
    
    # Track changes for audit log
    changes = {}
    
    # Update the parameters
    for param_name, param_value in updated_params.items():
        if param_name in parameters[module]:
            # Record the change if different
            if parameters[module][param_name] != param_value:
                changes[param_name] = {
                    "old": parameters[module][param_name],
                    "new": param_value
                }
            
            # Update the parameter
            parameters[module][param_name] = param_value
    
    # Save the updated parameters
    if save_parameters(parameters):
        # Log the changes
        if changes:
            user = request.headers.get('X-User-ID', 'unknown')
            log_parameter_change(module, user, changes)
        
        return jsonify({"success": True, "message": "Parameters updated successfully"})
    else:
        return jsonify({"error": "Failed to save parameters"}), 500

@app.route('/api/parameters/reset', methods=['POST'])
def reset_parameters():
    """Reset parameters to defaults."""
    # Save default parameters
    if save_parameters(DEFAULT_PARAMETERS):
        # Log the reset
        user = request.headers.get('X-User-ID', 'unknown')
        log_parameter_change("all", user, {"action": "reset_to_defaults"})
        
        return jsonify({"success": True, "message": "Parameters reset to defaults"})
    else:
        return jsonify({"error": "Failed to reset parameters"}), 500

@app.route('/api/parameters/presets', methods=['GET'])
def get_parameter_presets():
    """Get available parameter presets."""
    return jsonify(load_presets())

@app.route('/api/parameters/presets/<preset>', methods=['POST'])
def apply_parameter_preset(preset):
    """Apply a parameter preset."""
    presets = load_presets()
    
    if preset not in presets:
        return jsonify({"error": f"Preset '{preset}' not found"}), 404
    
    # Apply the preset
    updated_params = apply_preset_to_parameters(preset)
    
    # Log the preset application
    user = request.headers.get('X-User-ID', 'unknown')
    log_parameter_change("all", user, {"action": f"applied_preset_{preset}"})
    
    return jsonify({"success": True, "message": f"Preset '{preset}' applied successfully", "parameters": updated_params})

@app.route('/api/parameters/presets/custom', methods=['POST'])
def save_custom_preset():
    """Save current parameters as a custom preset."""
    preset_name = request.json.get('name', 'custom')
    description = request.json.get('description', 'Custom user preset')
    
    # Get current parameters
    current_params = load_parameters()
    
    # Create a flattened version for the preset
    preset_params = {}
    for module, params in current_params.items():
        for param_name, param_value in params.items():
            preset_params[param_name] = param_value
    
    # Add description
    preset_params['description'] = description
    
    # Save the preset
    if save_preset(preset_name, preset_params):
        return jsonify({"success": True, "message": f"Custom preset '{preset_name}' saved successfully"})
    else:
        return jsonify({"error": "Failed to save custom preset"}), 500

@app.route('/api/parameters/validate', methods=['POST'])
def validate_parameter_update():
    """Validate parameters without applying them."""
    module = request.json.get('module')
    parameters = request.json.get('parameters', {})
    
    if not module:
        return jsonify({"error": "Module is required"}), 400
    
    # Validate the parameters
    validation_errors = validate_parameters(module, parameters)
    
    if validation_errors:
        return jsonify({"valid": False, "errors": validation_errors})
    else:
        return jsonify({"valid": True})

@app.route('/api/parameters/metadata', methods=['GET'])
def get_parameter_metadata():
    """Get metadata for all parameters."""
    return jsonify(PARAMETER_METADATA)

@app.route('/api/parameters/audit-log', methods=['GET'])
def get_audit_log():
    """Get parameter change audit log."""
    try:
        if not os.path.exists(AUDIT_LOG_FILE):
            return jsonify([])
        
        with open(AUDIT_LOG_FILE, 'r') as f:
            log_entries = [json.loads(line) for line in f]
        
        # Apply filters if provided
        module = request.args.get('module')
        user = request.args.get('user')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        if module:
            log_entries = [entry for entry in log_entries if entry['module'] == module]
        
        if user:
            log_entries = [entry for entry in log_entries if entry['user'] == user]
        
        if start_date:
            start_dt = datetime.datetime.fromisoformat(start_date)
            log_entries = [entry for entry in log_entries if datetime.datetime.fromisoformat(entry['timestamp']) >= start_dt]
        
        if end_date:
            end_dt = datetime.datetime.fromisoformat(end_date)
            log_entries = [entry for entry in log_entries if datetime.datetime.fromisoformat(entry['timestamp']) <= end_dt]
        
        return jsonify(log_entries)
    except Exception as e:
        logger.error(f"Error retrieving audit log: {e}")
        return jsonify({"error": "Failed to retrieve audit log"}), 500

# Initialize the application
def init_app():
    """Initialize the application."""
    # Create default parameters file if it doesn't exist
    if not os.path.exists(CURRENT_PARAMETERS_FILE):
        save_parameters(DEFAULT_PARAMETERS)
    
    # Create default presets if they don't exist
    for preset_name, preset_params in DEFAULT_PRESETS.items():
        preset_file = os.path.join(PRESETS_DIR, f"{preset_name}.json")
        if not os.path.exists(preset_file):
            save_preset(preset_name, preset_params)
    
    logger.info("Parameter Management API initialized")

# Initialize on import
init_app()

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=5001, debug=True)
