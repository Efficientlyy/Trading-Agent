# LLM Strategic Overseer

## Overview

The LLM Strategic Overseer is an advanced AI-powered system that provides strategic decision-making, risk management, and performance optimization for the Trading-Agent platform. It integrates with the trading system core to provide real-time insights, pattern recognition, and automated decision support.

## Features

- **Tiered LLM Integration**: Cost-optimized approach using OpenRouter with multiple model tiers
- **Strategic Decision Making**: AI-powered analysis and trading recommendations
- **Pattern Recognition**: Advanced technical analysis and chart pattern detection
- **Real-time Visualization**: Interactive charts with pattern markers and decision indicators
- **Secure Telegram Integration**: Authenticated command interface and notifications
- **Capital Allocation Strategy**: Configurable percentage-based allocation (default 80%)
- **Performance Optimization**: Resource monitoring and cost reduction strategies
- **Comprehensive Notifications**: Configurable alerts and performance reports

## Architecture

The LLM Strategic Overseer is built with a modular architecture:

```
llm_overseer/
├── analysis/              # Pattern recognition and technical analysis
├── config/                # Configuration and settings management
├── core/                  # Core components (LLM, context, event bus)
├── data/                  # Data management and unified pipeline
├── integration/           # Integration with trading system
├── notifications/         # Notification and reporting system
├── optimization/          # Performance optimization and cost reduction
├── strategy/              # Trading strategy components
├── telegram/              # Telegram bot and authentication
├── tests/                 # Test suite
├── visualization/         # Chart visualization and dashboard
└── main.py                # Main entry point
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Efficientlyy/Trading-Agent.git
cd Trading-Agent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and settings
```

## Configuration

The system is configured through the `config/settings.json` file and environment variables:

```json
{
  "llm": {
    "provider": "openrouter",
    "api_key_env": "OPENROUTER_API_KEY",
    "default_model": "gpt-3.5-turbo",
    "timeout": 30
  },
  "telegram": {
    "bot_token_env": "TELEGRAM_BOT_TOKEN",
    "allowed_user_ids": [123456789],
    "admin_user_ids": [123456789]
  },
  "trading": {
    "capital_allocation_percentage": 80,
    "minimum_reserve": 100,
    "supported_assets": ["BTC/USDC", "ETH/USDC", "SOL/USDC"]
  }
}
```

## Usage

### Starting the System

```bash
python -m llm_overseer.main
```

### Telegram Commands

- `/start` - Start the bot and verify authentication
- `/help` - Show available commands
- `/status` - Check system status
- `/allocate <percentage>` - Set capital allocation percentage
- `/analyze <symbol> <timeframe>` - Request analysis for a specific asset
- `/report <daily|weekly>` - Generate performance report
- `/settings` - View current settings
- `/update_settings <setting> <value>` - Update a setting

## Development

### Running Tests

```bash
# Run all tests
python -m llm_overseer.tests.run_tests

# Run specific test
python -m llm_overseer.tests.test_visualization
```

### Adding New Components

1. Create a new module in the appropriate directory
2. Update `__init__.py` to expose the module
3. Add tests in the `tests/` directory
4. Update documentation

## Integration

The LLM Strategic Overseer integrates with the Trading-Agent system through the `integration` module. It connects to:

- Trading system core for execution
- Visualization components for chart display
- Telegram for user interaction
- Data pipeline for market data

## License

Copyright (c) 2025 Efficientlyy - All Rights Reserved
