# LLM Strategic Overseer Visualization

This directory contains the visualization components for the LLM Strategic Overseer system, providing real-time chart visualization for BTC, ETH, and SOL trading pairs.

## Components

- **bridge.py**: Bidirectional communication bridge between LLM and visualization
- **chart_visualization.py**: Real-time chart rendering with technical indicators
- **trading_dashboard.py**: Web-based dashboard for monitoring trading activities

## Features

- Real-time price and volume charts
- Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
- Pattern recognition visualization
- Strategic decision markers
- Event-driven architecture for instant updates

## Usage

### Running the Dashboard

```bash
python -m llm_overseer.visualization.trading_dashboard
```

Then access the dashboard at http://localhost:8000

### Integrating with LLM Overseer

The visualization components are automatically integrated when running the full system:

```bash
python -m llm_overseer.main
```

## Configuration

Configuration options are available in `config/settings.json` under the `visualization` section:

```json
"visualization": {
  "timeframe": "1h",
  "max_points": 100,
  "indicators": ["sma", "rsi", "macd", "bollinger"],
  "theme": "dark",
  "show_patterns": true,
  "show_signals": true
}
```

## Documentation

For detailed documentation, see [visualization_guide.md](../docs/visualization_guide.md) in the docs directory.
