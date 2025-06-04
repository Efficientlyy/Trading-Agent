# LLM Strategic Overseer Visualization Guide

This guide provides comprehensive instructions on using, configuring, and extending the real-time visualization components of the LLM Strategic Overseer system.

## Overview

The visualization system provides real-time charting for BTC, ETH, and SOL trading pairs with technical indicators, pattern recognition markers, and strategic decision highlights. The system is built on an event-driven architecture that ensures seamless updates across all components.

## Components

The visualization system consists of the following key components:

1. **Market Data Service** (`data/market_data_service.py`): Connects to cryptocurrency exchanges and provides real-time market data for BTC, ETH, and SOL.

2. **Chart Visualization** (`visualization/chart_visualization.py`): Renders charts with technical indicators and pattern markers.

3. **LLM-Visualization Bridge** (`visualization/bridge.py`): Provides bidirectional communication between the LLM Strategic Overseer and visualization components.

4. **Trading Dashboard** (`visualization/trading_dashboard.py`): Web-based dashboard for monitoring trading activities and visualizing market data.

5. **Pattern Recognition** (`analysis/pattern_recognition.py`): Detects chart patterns and calculates technical indicators.

## Getting Started

### Prerequisites

- Python 3.11+
- Required packages: `matplotlib`, `pandas`, `numpy`, `fastapi`, `uvicorn`, `websockets`

### Setup

1. Install required packages:
   ```bash
   pip install matplotlib pandas numpy fastapi uvicorn websockets
   ```

2. Configure the system in `config/settings.json`:
   ```json
   {
     "market_data": {
       "exchange": "mexc",
       "api_key": "your_api_key",
       "api_secret": "your_api_secret",
       "mock_mode": true,
       "update_interval": 1.0,
       "timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"]
     },
     "trading": {
       "symbols": ["BTC/USDC", "ETH/USDC", "SOL/USDC"]
     },
     "visualization": {
       "timeframe": "1h",
       "max_points": 100,
       "indicators": ["sma", "rsi", "macd", "bollinger"],
       "theme": "dark",
       "show_patterns": true,
       "show_signals": true
     },
     "analysis": {
       "patterns_enabled": true,
       "indicators_enabled": true,
       "min_data_points": 30,
       "detection_interval": 5,
       "calculation_interval": 1,
       "patterns": [
         "double_top", "double_bottom", "head_and_shoulders", 
         "inverse_head_and_shoulders", "triangle", "wedge"
       ],
       "indicators": ["sma", "ema", "rsi", "macd", "bollinger"],
       "sma_window": 20,
       "ema_window": 20,
       "rsi_window": 14,
       "macd_fast": 12,
       "macd_slow": 26,
       "macd_signal": 9,
       "bollinger_window": 20,
       "bollinger_std": 2.0
     }
   }
   ```

3. Set up API keys (if using real exchange data):
   - For MEXC, obtain API keys from your account settings
   - Add the keys to your configuration or environment variables

### Running the Visualization System

#### Option 1: Run the Complete LLM Strategic Overseer

The visualization components are integrated into the main LLM Strategic Overseer system. To run the complete system:

```bash
python -m llm_overseer.main
```

This will start all components, including the visualization system.

#### Option 2: Run the Trading Dashboard Standalone

To run just the trading dashboard for visualization:

```bash
python -m llm_overseer.visualization.trading_dashboard
```

This will start a web server at http://localhost:8000 where you can access the dashboard.

#### Option 3: Run the Market Data Service Standalone

To run just the market data service for testing:

```bash
python -m llm_overseer.data.market_data_service
```

This will start generating market data events that can be consumed by other components.

## Features

### Real-Time Charts

The system provides real-time charts for BTC, ETH, and SOL with the following features:

- Price and volume visualization
- Candlestick charts for different timeframes (1m, 5m, 15m, 1h, 4h, 1d)
- Automatic updates based on market data events

### Technical Indicators

The following technical indicators are supported:

- **Simple Moving Average (SMA)**: Shows average price over a specified period
- **Exponential Moving Average (EMA)**: Gives more weight to recent prices
- **Relative Strength Index (RSI)**: Measures momentum and overbought/oversold conditions
- **Moving Average Convergence Divergence (MACD)**: Shows trend direction and momentum
- **Bollinger Bands**: Shows volatility and potential reversal points

### Pattern Recognition

The system automatically detects the following chart patterns:

- **Double Top/Bottom**: Reversal patterns indicating potential trend changes
- **Head and Shoulders**: Complex reversal pattern
- **Inverse Head and Shoulders**: Bullish reversal pattern
- **Triangle**: Continuation pattern showing consolidation
- **Wedge**: Pattern indicating potential breakout

### Strategic Decision Visualization

The system visualizes LLM Strategic Overseer decisions on the charts:

- **Entry/Exit Points**: Marks where the system decides to enter or exit trades
- **Risk Adjustments**: Highlights when risk parameters are adjusted
- **Pattern Confirmations**: Shows when patterns are confirmed by the LLM

## Configuration Options

### Market Data Service

| Option | Description | Default |
|--------|-------------|---------|
| `exchange` | Exchange to connect to (mexc) | `"mexc"` |
| `api_key` | API key for exchange | `""` |
| `api_secret` | API secret for exchange | `""` |
| `mock_mode` | Use mock data instead of real exchange | `true` |
| `update_interval` | Interval between updates in seconds | `1.0` |
| `timeframes` | Timeframes to fetch data for | `["1m", "5m", "15m", "1h", "4h", "1d"]` |

### Chart Visualization

| Option | Description | Default |
|--------|-------------|---------|
| `timeframe` | Default timeframe to display | `"1h"` |
| `max_points` | Maximum number of data points to display | `100` |
| `indicators` | Indicators to display | `["sma", "rsi", "macd", "bollinger"]` |
| `theme` | Chart theme (dark/light) | `"dark"` |
| `show_patterns` | Show pattern markers | `true` |
| `show_signals` | Show strategic decision signals | `true` |

### Pattern Recognition

| Option | Description | Default |
|--------|-------------|---------|
| `patterns_enabled` | Enable pattern detection | `true` |
| `indicators_enabled` | Enable indicator calculation | `true` |
| `min_data_points` | Minimum data points required for pattern detection | `30` |
| `detection_interval` | Run detection every N data points | `5` |
| `calculation_interval` | Run calculation every N data points | `1` |
| `patterns` | Patterns to detect | `["double_top", "double_bottom", ...]` |
| `indicators` | Indicators to calculate | `["sma", "ema", "rsi", "macd", "bollinger"]` |
| `sma_window` | Window size for SMA | `20` |
| `ema_window` | Window size for EMA | `20` |
| `rsi_window` | Window size for RSI | `14` |
| `macd_fast` | Fast period for MACD | `12` |
| `macd_slow` | Slow period for MACD | `26` |
| `macd_signal` | Signal period for MACD | `9` |
| `bollinger_window` | Window size for Bollinger Bands | `20` |
| `bollinger_std` | Standard deviation for Bollinger Bands | `2.0` |

## Extending the System

### Adding New Indicators

To add a new technical indicator:

1. Add the indicator to the `indicators` list in the configuration
2. Implement the calculation method in `pattern_recognition.py`
3. Add visualization support in `chart_visualization.py`

Example for adding a new "Stochastic Oscillator" indicator:

```python
# In pattern_recognition.py
def _calculate_stochastic(self, prices: List[float], highs: List[float], lows: List[float], window: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> tuple:
    """Calculate Stochastic Oscillator."""
    # Implementation here
    return k_values, d_values

# In chart_visualization.py
def _plot_stochastic(self, ax, symbol: str, timestamps):
    """Plot Stochastic Oscillator."""
    # Implementation here
```

### Adding New Chart Patterns

To add a new chart pattern:

1. Add the pattern to the `patterns` list in the configuration
2. Implement the detection method in `pattern_recognition.py`
3. Add visualization support in `chart_visualization.py`

Example for adding a new "Flag" pattern:

```python
# In pattern_recognition.py
def _detect_flag(self, prices: List[float], timestamps: List[datetime]) -> Optional[Dict[str, Any]]:
    """Detect Flag pattern."""
    # Implementation here
    return pattern_data

# In chart_visualization.py
# Add to _plot_patterns method
elif pattern_type == "flag":
    ax.scatter(pattern_timestamp, pattern_price, marker="f", color="orange", s=100, label=f"Flag")
```

### Connecting to Different Exchanges

To add support for a new exchange:

1. Add the exchange to the `exchange` option in the configuration
2. Implement the exchange-specific methods in `market_data_service.py`

Example for adding Binance support:

```python
# In market_data_service.py
async def _connect_to_exchange(self) -> bool:
    """Connect to cryptocurrency exchange."""
    if self.exchange == "binance":
        # Implement Binance connection
        # ...
        return True
    elif self.exchange == "mexc":
        # Existing MEXC implementation
        # ...
        return True
    else:
        logger.error(f"Unsupported exchange: {self.exchange}")
        return False
```

## Troubleshooting

### Common Issues

#### No Data Displayed in Charts

- Check if the market data service is running
- Verify that the event bus is properly connected
- Ensure the symbols in configuration match what you're trying to visualize

#### Missing Technical Indicators

- Check if the indicator is enabled in configuration
- Verify there are enough data points for calculation
- Check logs for calculation errors

#### Pattern Detection Not Working

- Ensure enough data points are available (min_data_points setting)
- Check if pattern detection is enabled
- Verify the pattern is in the enabled patterns list

#### Slow Performance

- Reduce the number of indicators being calculated
- Increase the calculation_interval and detection_interval
- Reduce the max_points setting for chart visualization

### Logs

The system generates logs in the following locations:

- `logs/market_data_service.log`: Market data service logs
- `logs/chart_visualization.log`: Chart visualization logs
- `logs/pattern_recognition.log`: Pattern recognition logs
- `logs/dashboard.log`: Trading dashboard logs

Check these logs for detailed error messages and debugging information.

## Advanced Usage

### Custom Event Handlers

You can subscribe to specific events for custom processing:

```python
from llm_overseer.core.event_bus import EventBus

event_bus = EventBus()

async def custom_pattern_handler(topic: str, data: Dict[str, Any]):
    # Custom processing for pattern events
    print(f"Pattern detected: {data['pattern_type']} for {data['symbol']}")

# Subscribe to pattern events
subscription_id = event_bus.subscribe("analysis.pattern", custom_pattern_handler)
```

### Backtesting with Historical Data

For backtesting with historical data:

1. Create a CSV file with historical data
2. Load the data into the market data service:

```python
import pandas as pd
from llm_overseer.data.market_data_service import MarketDataService

# Load historical data
historical_data = pd.read_csv("historical_data.csv")

# Create market data service
market_data_service = MarketDataService(config, event_bus)

# Load historical data
for index, row in historical_data.iterrows():
    market_data = {
        "success": True,
        "symbol": row["symbol"],
        "price": row["price"],
        "volume_24h": row["volume"],
        "timestamp": row["timestamp"]
    }
    asyncio.run(event_bus.publish("trading.market_data", market_data))
```

### Custom Visualization Themes

To create a custom visualization theme:

1. Create a new theme in `chart_visualization.py`:

```python
def _set_theme(self, theme: str):
    """Set chart theme."""
    if theme == "dark":
        plt.style.use("dark_background")
    elif theme == "light":
        plt.style.use("default")
    elif theme == "custom":
        # Custom theme settings
        plt.rcParams.update({
            "figure.facecolor": "#1E1E1E",
            "axes.facecolor": "#1E1E1E",
            "axes.edgecolor": "#808080",
            "axes.labelcolor": "#D0D0D0",
            "xtick.color": "#D0D0D0",
            "ytick.color": "#D0D0D0",
            "grid.color": "#404040",
            "text.color": "#D0D0D0"
        })
    else:
        plt.style.use("default")
```

2. Update the `theme` option in configuration to use your custom theme

## API Reference

The visualization system exposes the following key classes and methods:

### MarketDataService

- `start()`: Start the market data service
- `stop()`: Stop the market data service
- `get_market_data(symbol)`: Get current market data for a symbol
- `get_klines_data(symbol, timeframe)`: Get klines data for a symbol and timeframe

### ChartVisualization

- `update_chart(symbol, strategy_decision=None)`: Update chart for a symbol
- `calculate_indicators(symbol)`: Calculate technical indicators for a symbol
- `get_chart_path(symbol)`: Get path to chart image for a symbol

### PatternRecognition

- `detect_patterns(symbol)`: Detect patterns for a symbol
- `calculate_indicators(symbol)`: Calculate indicators for a symbol

### LLMVisualizationBridge

- `subscribe_to_strategic_decisions(callback)`: Subscribe to strategic decisions
- `subscribe_to_pattern_recognition(callback)`: Subscribe to pattern recognition results
- `publish_strategic_decision(decision)`: Publish strategic decision
- `publish_pattern_recognition(pattern)`: Publish pattern recognition result

### TradingDashboard

- `run(host, port)`: Run the trading dashboard server

## Conclusion

The visualization system provides powerful real-time charting capabilities for the LLM Strategic Overseer, enabling effective monitoring and analysis of trading activities. By leveraging the event-driven architecture, the system ensures that all components stay synchronized with the latest market data and strategic decisions.

For further assistance or to report issues, please contact the development team or submit an issue on GitHub.
