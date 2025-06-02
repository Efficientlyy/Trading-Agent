# Trading-Agent Configurable Parameters

This document outlines all user-configurable parameters for each module/engine in the Trading-Agent system. These parameters can be exposed via the frontend interface to allow users to customize the system's behavior while maintaining safe default values.

## Table of Contents

1. [Market Data Collection](#market-data-collection)
2. [Pattern Recognition](#pattern-recognition)
3. [Signal Generation](#signal-generation)
4. [Decision Making (RL Agent)](#decision-making-rl-agent)
5. [Order Execution](#order-execution)
6. [Risk Management](#risk-management)
7. [Visualization](#visualization)
8. [Monitoring Dashboard](#monitoring-dashboard)
9. [Performance Optimization](#performance-optimization)
10. [System-Wide Settings](#system-wide-settings)

## Market Data Collection

| Parameter | Description | Default | Range/Options | Category |
|-----------|-------------|---------|--------------|----------|
| `active_trading_pairs` | Trading pairs to monitor and trade | `["BTC/USDC", "ETH/USDC", "SOL/USDC"]` | Any valid trading pair on MEXC | Basic |
| `primary_timeframes` | Primary timeframes for data collection | `["5m", "15m", "1h", "4h"]` | `["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "3d", "1w"]` | Basic |
| `historical_candles_count` | Number of historical candles to retrieve | `1000` | `100-5000` | Advanced |
| `websocket_enabled` | Enable real-time data via WebSocket | `true` | `true/false` | Basic |
| `data_update_interval_sec` | Interval for REST API data updates (seconds) | `60` | `10-3600` | Advanced |
| `enable_order_book_data` | Collect order book data | `true` | `true/false` | Advanced |
| `order_book_depth` | Depth of order book to collect | `20` | `5-100` | Advanced |
| `enable_trade_data` | Collect recent trades data | `true` | `true/false` | Advanced |
| `recent_trades_limit` | Number of recent trades to collect | `50` | `10-500` | Advanced |
| `data_caching_enabled` | Enable data caching | `true` | `true/false` | Advanced |
| `cache_ttl_sec` | Cache time-to-live in seconds | `300` | `10-3600` | Advanced |
| `retry_attempts` | Number of retry attempts for API calls | `3` | `0-10` | Advanced |
| `retry_delay_ms` | Delay between retry attempts (ms) | `1000` | `100-10000` | Advanced |

## Pattern Recognition

| Parameter | Description | Default | Range/Options | Category |
|-----------|-------------|---------|--------------|----------|
| `enabled_patterns` | Technical patterns to detect | `["head_and_shoulders", "double_top", "double_bottom", "triangle", "wedge", "channel", "support_resistance"]` | Multiple selection from available patterns | Basic |
| `min_pattern_confidence` | Minimum confidence threshold for pattern detection | `0.75` | `0.5-0.95` | Basic |
| `pattern_lookback_periods` | Number of periods to look back for pattern detection | `50` | `20-200` | Advanced |
| `enable_ml_detection` | Enable machine learning pattern detection | `true` | `true/false` | Basic |
| `ml_confidence_threshold` | Confidence threshold for ML pattern detection | `0.8` | `0.5-0.99` | Advanced |
| `ml_model_update_interval_hours` | Interval for ML model updates (hours) | `24` | `1-168` | Advanced |
| `enable_adaptive_thresholds` | Dynamically adjust thresholds based on market volatility | `true` | `true/false` | Advanced |
| `volatility_adjustment_factor` | Factor for volatility-based threshold adjustment | `0.5` | `0.1-2.0` | Expert |
| `enable_multi_timeframe_confirmation` | Require pattern confirmation across multiple timeframes | `true` | `true/false` | Advanced |
| `confirmation_timeframes` | Timeframes required for confirmation | `["15m", "1h"]` | Multiple selection from available timeframes | Advanced |
| `pattern_visualization_enabled` | Enable pattern visualization on charts | `true` | `true/false` | Basic |
| `max_concurrent_patterns` | Maximum number of patterns to track concurrently | `5` | `1-20` | Advanced |

## Signal Generation

| Parameter | Description | Default | Range/Options | Category |
|-----------|-------------|---------|--------------|----------|
| `min_signal_confidence` | Minimum confidence for signal generation | `0.8` | `0.5-0.99` | Basic |
| `signal_expiry_periods` | Number of periods before signal expires | `3` | `1-20` | Advanced |
| `enable_signal_filtering` | Filter signals based on market conditions | `true` | `true/false` | Basic |
| `enable_signal_stacking` | Allow multiple signals to stack/combine | `true` | `true/false` | Advanced |
| `max_stacked_signals` | Maximum number of stacked signals | `3` | `1-10` | Advanced |
| `signal_reinforcement_factor` | Factor for reinforcing signals with multiple confirmations | `1.2` | `1.0-2.0` | Advanced |
| `enable_contrarian_signals` | Generate contrarian signals in overbought/oversold conditions | `false` | `true/false` | Advanced |
| `overbought_threshold` | RSI threshold for overbought condition | `70` | `60-90` | Advanced |
| `oversold_threshold` | RSI threshold for oversold condition | `30` | `10-40` | Advanced |
| `enable_volume_confirmation` | Require volume confirmation for signals | `true` | `true/false` | Basic |
| `min_volume_percentile` | Minimum volume percentile for confirmation | `60` | `0-100` | Advanced |
| `enable_trend_filter` | Filter signals against the prevailing trend | `true` | `true/false` | Basic |
| `trend_determination_method` | Method to determine trend | `"ema_cross"` | `["ema_cross", "higher_highs", "macd", "supertrend"]` | Advanced |
| `trend_lookback_periods` | Periods to look back for trend determination | `20` | `5-100` | Advanced |

## Decision Making (RL Agent)

| Parameter | Description | Default | Range/Options | Category |
|-----------|-------------|---------|--------------|----------|
| `decision_confidence_threshold` | Minimum confidence for trade decisions | `0.85` | `0.5-0.99` | Basic |
| `enable_position_sizing` | Enable dynamic position sizing | `true` | `true/false` | Basic |
| `max_trades_per_day` | Maximum number of trades per day | `5` | `1-100` | Basic |
| `min_time_between_trades_min` | Minimum time between trades (minutes) | `60` | `0-1440` | Advanced |
| `enable_reinforcement_learning` | Use RL for decision making | `true` | `true/false` | Basic |
| `exploration_rate` | Exploration rate for RL agent | `0.1` | `0.01-0.5` | Advanced |
| `learning_rate` | Learning rate for RL agent | `0.001` | `0.0001-0.01` | Expert |
| `reward_discount_factor` | Discount factor for future rewards | `0.95` | `0.8-0.99` | Expert |
| `use_market_state_features` | Include market state in decision features | `true` | `true/false` | Advanced |
| `use_technical_indicators` | Include technical indicators in decision features | `true` | `true/false` | Advanced |
| `use_sentiment_data` | Include sentiment data in decision features | `false` | `true/false` | Advanced |
| `enable_decision_explanations` | Generate explanations for decisions | `true` | `true/false` | Basic |
| `decision_history_size` | Number of past decisions to retain | `100` | `10-1000` | Advanced |
| `enable_auto_training` | Automatically train model on new data | `true` | `true/false` | Advanced |
| `training_interval_hours` | Interval for model training (hours) | `24` | `1-168` | Advanced |

## Order Execution

| Parameter | Description | Default | Range/Options | Category |
|-----------|-------------|---------|--------------|----------|
| `default_order_type` | Default order type | `"market"` | `["market", "limit"]` | Basic |
| `enable_smart_routing` | Enable smart order routing | `true` | `true/false` | Basic |
| `slippage_tolerance_percent` | Maximum allowed slippage | `0.5` | `0.1-5.0` | Basic |
| `enable_iceberg_orders` | Enable iceberg orders for large positions | `false` | `true/false` | Advanced |
| `iceberg_order_threshold` | Threshold for using iceberg orders (USD) | `5000` | `1000-100000` | Advanced |
| `iceberg_display_size_percent` | Percentage of order to display in iceberg | `10` | `5-50` | Advanced |
| `enable_twap_execution` | Enable TWAP execution for large orders | `false` | `true/false` | Advanced |
| `twap_order_threshold` | Threshold for using TWAP (USD) | `10000` | `1000-100000` | Advanced |
| `twap_interval_minutes` | Interval for TWAP execution (minutes) | `30` | `5-240` | Advanced |
| `twap_slices` | Number of slices for TWAP execution | `5` | `2-20` | Advanced |
| `enable_retry_on_failure` | Retry failed orders | `true` | `true/false` | Basic |
| `max_retry_attempts` | Maximum retry attempts for failed orders | `3` | `0-10` | Advanced |
| `retry_delay_seconds` | Delay between retry attempts (seconds) | `5` | `1-60` | Advanced |
| `enable_partial_fills` | Allow partial fills for orders | `true` | `true/false` | Basic |
| `min_fill_percent` | Minimum fill percentage to consider successful | `90` | `50-100` | Advanced |
| `cancel_after_seconds` | Cancel unfilled orders after seconds | `300` | `10-3600` | Advanced |

## Risk Management

| Parameter | Description | Default | Range/Options | Category |
|-----------|-------------|---------|--------------|----------|
| `risk_level` | Overall risk level | `"medium"` | `["very_low", "low", "medium", "high", "very_high"]` | Basic |
| `max_portfolio_risk_percent` | Maximum portfolio at risk | `2.0` | `0.5-10.0` | Basic |
| `max_position_size_usd` | Maximum position size in USD | `1000` | `100-100000` | Basic |
| `max_position_size_percent` | Maximum position size as % of portfolio | `5.0` | `1.0-50.0` | Basic |
| `max_total_exposure_percent` | Maximum total exposure as % of portfolio | `25.0` | `5.0-100.0` | Basic |
| `stop_loss_percent` | Default stop loss percentage | `2.0` | `0.5-10.0` | Basic |
| `take_profit_percent` | Default take profit percentage | `4.0` | `1.0-20.0` | Basic |
| `enable_trailing_stop` | Enable trailing stop loss | `true` | `true/false` | Basic |
| `trailing_stop_activation_percent` | Profit % to activate trailing stop | `1.0` | `0.5-5.0` | Advanced |
| `trailing_stop_distance_percent` | Trailing stop distance percentage | `1.5` | `0.5-5.0` | Advanced |
| `max_daily_drawdown_percent` | Maximum daily drawdown allowed | `5.0` | `1.0-20.0` | Basic |
| `max_trades_per_asset_daily` | Maximum trades per asset per day | `3` | `1-20` | Basic |
| `enable_circuit_breakers` | Enable circuit breakers | `true` | `true/false` | Basic |
| `price_circuit_breaker_percent` | Price movement % to trigger circuit breaker | `5.0` | `1.0-20.0` | Advanced |
| `volume_circuit_breaker_factor` | Volume spike factor to trigger circuit breaker | `3.0` | `1.5-10.0` | Advanced |
| `circuit_breaker_cooldown_minutes` | Cooldown period after circuit breaker (minutes) | `60` | `15-1440` | Advanced |
| `enable_correlation_risk_control` | Control risk based on asset correlations | `true` | `true/false` | Advanced |
| `max_correlation_exposure` | Maximum exposure to correlated assets | `15.0` | `5.0-50.0` | Advanced |
| `enable_volatility_adjustment` | Adjust position size based on volatility | `true` | `true/false` | Advanced |
| `volatility_lookback_periods` | Periods to calculate volatility | `20` | `5-100` | Advanced |
| `enable_overnight_position_control` | Control overnight positions | `true` | `true/false` | Basic |
| `max_overnight_exposure_percent` | Maximum overnight exposure | `15.0` | `0.0-100.0` | Basic |

## Visualization

| Parameter | Description | Default | Range/Options | Category |
|-----------|-------------|---------|--------------|----------|
| `default_chart_timeframe` | Default timeframe for charts | `"1h"` | `["1m", "5m", "15m", "30m", "1h", "4h", "1d"]` | Basic |
| `default_chart_type` | Default chart type | `"candlestick"` | `["candlestick", "line", "area", "bar"]` | Basic |
| `default_chart_theme` | Default chart theme | `"dark"` | `["dark", "light"]` | Basic |
| `enable_technical_indicators` | Show technical indicators on charts | `true` | `true/false` | Basic |
| `default_indicators` | Default indicators to display | `["MA", "RSI", "MACD"]` | Multiple selection from available indicators | Basic |
| `max_indicators` | Maximum number of indicators to display | `5` | `0-10` | Advanced |
| `enable_pattern_visualization` | Show detected patterns on charts | `true` | `true/false` | Basic |
| `enable_signal_markers` | Show signal markers on charts | `true` | `true/false` | Basic |
| `enable_trade_markers` | Show trade markers on charts | `true` | `true/false` | Basic |
| `enable_price_alerts` | Enable price alerts | `true` | `true/false` | Basic |
| `enable_volume_profile` | Show volume profile | `true` | `true/false` | Advanced |
| `enable_depth_chart` | Show market depth chart | `true` | `true/false` | Advanced |
| `chart_update_interval_ms` | Chart update interval (ms) | `1000` | `100-10000` | Advanced |
| `max_visible_candles` | Maximum candles visible on chart | `100` | `50-500` | Advanced |
| `enable_multi_chart_view` | Enable multi-chart view | `true` | `true/false` | Basic |
| `enable_drawing_tools` | Enable drawing tools | `true` | `true/false` | Basic |
| `enable_chart_animations` | Enable chart animations | `true` | `true/false` | Basic |

## Monitoring Dashboard

| Parameter | Description | Default | Range/Options | Category |
|-----------|-------------|---------|--------------|----------|
| `dashboard_refresh_interval_sec` | Dashboard refresh interval (seconds) | `5` | `1-60` | Basic |
| `enable_system_status_monitoring` | Monitor system status | `true` | `true/false` | Basic |
| `enable_risk_metrics_display` | Display risk metrics | `true` | `true/false` | Basic |
| `enable_performance_metrics` | Display performance metrics | `true` | `true/false` | Basic |
| `enable_trade_history` | Display trade history | `true` | `true/false` | Basic |
| `trade_history_limit` | Number of trades to display in history | `50` | `10-500` | Advanced |
| `enable_log_display` | Display system logs | `true` | `true/false` | Basic |
| `log_display_limit` | Number of log entries to display | `100` | `10-1000` | Advanced |
| `enable_alert_notifications` | Enable alert notifications | `true` | `true/false` | Basic |
| `alert_notification_methods` | Methods for alert notifications | `["dashboard", "email"]` | `["dashboard", "email", "sms", "webhook"]` | Advanced |
| `critical_alert_threshold` | Threshold for critical alerts | `"error"` | `["warning", "error", "critical"]` | Advanced |
| `enable_email_reports` | Enable email reports | `false` | `true/false` | Advanced |
| `email_report_frequency` | Frequency of email reports | `"daily"` | `["hourly", "daily", "weekly"]` | Advanced |
| `enable_performance_charts` | Enable performance charts | `true` | `true/false` | Basic |
| `enable_resource_monitoring` | Monitor system resources | `true` | `true/false` | Advanced |
| `resource_alert_threshold_percent` | Resource usage alert threshold | `80` | `50-95` | Advanced |

## Performance Optimization

| Parameter | Description | Default | Range/Options | Category |
|-----------|-------------|---------|--------------|----------|
| `enable_data_caching` | Enable data caching | `true` | `true/false` | Advanced |
| `cache_size_mb` | Maximum cache size (MB) | `100` | `10-1000` | Advanced |
| `enable_batch_processing` | Enable batch processing | `true` | `true/false` | Advanced |
| `batch_size` | Batch size for processing | `100` | `10-1000` | Advanced |
| `enable_parallel_processing` | Enable parallel processing | `true` | `true/false` | Advanced |
| `max_parallel_threads` | Maximum parallel threads | `4` | `1-16` | Advanced |
| `enable_websocket_optimization` | Optimize WebSocket connections | `true` | `true/false` | Advanced |
| `websocket_reconnect_interval_sec` | WebSocket reconnect interval (seconds) | `5` | `1-60` | Advanced |
| `enable_lazy_loading` | Enable lazy loading of data | `true` | `true/false` | Advanced |
| `enable_memory_optimization` | Enable memory usage optimization | `true` | `true/false` | Advanced |
| `memory_cleanup_interval_min` | Memory cleanup interval (minutes) | `30` | `5-240` | Expert |
| `enable_query_optimization` | Enable database query optimization | `true` | `true/false` | Advanced |
| `query_cache_size` | Query cache size | `50` | `10-500` | Expert |
| `enable_compression` | Enable data compression | `true` | `true/false` | Advanced |
| `compression_level` | Compression level | `6` | `1-9` | Expert |

## System-Wide Settings

| Parameter | Description | Default | Range/Options | Category |
|-----------|-------------|---------|--------------|----------|
| `trading_enabled` | Enable trading | `false` | `true/false` | Basic |
| `paper_trading_mode` | Use paper trading mode | `true` | `true/false` | Basic |
| `base_currency` | Base currency for portfolio | `"USDC"` | `["USDC", "USDT", "USD"]` | Basic |
| `log_level` | Logging level | `"info"` | `["debug", "info", "warning", "error", "critical"]` | Basic |
| `enable_telegram_notifications` | Enable Telegram notifications | `false` | `true/false` | Basic |
| `enable_email_notifications` | Enable email notifications | `false` | `true/false` | Basic |
| `notification_level` | Minimum level for notifications | `"warning"` | `["info", "warning", "error", "critical"]` | Basic |
| `timezone` | Timezone for timestamps | `"UTC"` | List of timezones | Basic |
| `enable_auto_updates` | Enable automatic updates | `true` | `true/false` | Basic |
| `backup_interval_hours` | Interval for system backups (hours) | `24` | `1-168` | Advanced |
| `backup_retention_days` | Days to retain backups | `30` | `1-365` | Advanced |
| `enable_api_access` | Enable API access | `false` | `true/false` | Advanced |
| `api_access_ips` | Allowed IP addresses for API access | `["127.0.0.1"]` | List of IP addresses | Advanced |
| `session_timeout_minutes` | Session timeout (minutes) | `60` | `5-1440` | Basic |
| `max_login_attempts` | Maximum login attempts | `5` | `1-10` | Basic |
| `enable_2fa` | Enable two-factor authentication | `false` | `true/false` | Basic |

## Frontend Parameter Organization

For the frontend interface, parameters should be organized into the following categories:

1. **Basic Settings**: Essential parameters that most users will want to configure
2. **Advanced Settings**: Parameters for users who want more control
3. **Expert Settings**: Parameters that should only be modified by experienced users

The interface should include:

- Tooltips explaining each parameter
- Visual indicators for safe/risky values
- Reset to default buttons
- Parameter validation
- Save/cancel functionality
- Parameter search functionality
- Parameter export/import
- Preset configurations (Conservative, Balanced, Aggressive)

## API Endpoints

The following API endpoints should be implemented for parameter management:

- `GET /api/parameters` - Get all parameters
- `GET /api/parameters/{module}` - Get parameters for a specific module
- `PUT /api/parameters/{module}` - Update parameters for a specific module
- `POST /api/parameters/reset` - Reset parameters to defaults
- `GET /api/parameters/presets` - Get available parameter presets
- `POST /api/parameters/presets/{preset}` - Apply a parameter preset

## Parameter Validation

All parameters should be validated before being applied:

- Range validation for numeric parameters
- Option validation for selection parameters
- Dependency validation for related parameters
- Risk validation to prevent dangerous configurations

## Default Parameter Profiles

The system should include the following default parameter profiles:

1. **Conservative**: Minimizes risk with smaller position sizes, tighter stop losses, and fewer trades
2. **Balanced**: Default settings with moderate risk and reward
3. **Aggressive**: Higher risk tolerance with larger position sizes and more trades
4. **High-Frequency**: Optimized for frequent trading with shorter timeframes
5. **Swing Trading**: Optimized for longer-term positions with wider stops

Each profile should be a complete set of parameters across all modules.
