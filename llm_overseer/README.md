# LLM Strategic Overseer

## Overview

The LLM Strategic Overseer is an advanced AI-powered decision-making system for the Trading-Agent platform. It leverages large language models (LLMs) to provide strategic insights, optimize trading parameters, and manage risk for cryptocurrency trading operations.

## Key Features

- **Tiered LLM Integration**: Cost-optimized approach using different model tiers based on task complexity
- **Secure Telegram Interface**: Authenticated command interface with session management
- **Strategic Decision Engine**: AI-powered analysis and decision making for trading optimization
- **Market Analysis**: Advanced order book and market microstructure analysis
- **Risk Management**: Sophisticated position sizing and risk control
- **Capital Allocation**: Configurable percentage of available USDC for trading (default 80%)
- **Notification System**: Real-time alerts and performance reporting

## Architecture

The LLM Strategic Overseer consists of the following core components:

```
llm_overseer/
├── config/               # Configuration management
├── core/                 # Core LLM functionality
├── strategy/             # Trading strategy components
├── telegram/             # Secure Telegram interface
├── integration/          # Trading system integration
├── tests/                # Test suite
└── main.py               # Main entry point
```

### Core Components

- **LLM Manager**: Handles API calls to OpenRouter with tiered model selection
- **Context Manager**: Maintains market context and trading history
- **Token Tracker**: Monitors token usage and costs
- **Decision Engine**: Generates strategic trading decisions
- **Market Analyzer**: Analyzes market microstructure and order book
- **Risk Manager**: Controls position sizing and risk parameters
- **Capital Allocation Strategy**: Manages trading capital allocation

## Configuration

Configuration is managed through a combination of:

1. **settings.json**: Default configuration values
2. **Environment Variables**: Secure API keys and tokens
3. **Command Interface**: Runtime parameter adjustments

### Example Configuration

```json
{
  "llm": {
    "provider": "openrouter",
    "api_key_env": "OPENROUTER_API_KEY",
    "models": {
      "tier_1": "openai/gpt-3.5-turbo",
      "tier_2": "anthropic/claude-3-sonnet",
      "tier_3": "anthropic/claude-3-opus"
    }
  },
  "telegram": {
    "bot_token_env": "TELEGRAM_BOT_TOKEN",
    "allowed_user_ids": [123456789]
  },
  "trading": {
    "allocation": {
      "enabled": true,
      "percentage": 0.8,
      "min_reserve": 100
    },
    "risk": {
      "max_position_size": 0.1,
      "max_total_exposure": 0.5,
      "risk_reward_ratio": 2.0
    }
  }
}
```

## Capital Allocation Strategy

The system implements a configurable capital allocation strategy that:

1. Uses a specified percentage (default 80%) of available USDC for trading
2. Maintains a minimum reserve amount (default $100)
3. Provides easy configuration through settings or commands
4. Tracks allocation history for analysis

This approach ensures that trading operations consistently use the specified portion of available capital while maintaining a safety reserve.

## Security

The system implements multiple security layers:

1. **Authentication**: Secure Telegram authentication with verification codes
2. **Session Management**: Time-limited sessions with automatic expiration
3. **Access Control**: Whitelisted user IDs for command access
4. **Secure Storage**: Environment variables for sensitive API keys

## Usage

### Prerequisites

- Python 3.11+
- OpenRouter API key
- Telegram Bot token
- Trading system integration

### Installation

```bash
# Clone the repository
git clone https://github.com/Efficientlyy/Trading-Agent.git
cd Trading-Agent

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Running the System

```bash
# Start the LLM Strategic Overseer
python -m llm_overseer.main
```

### Telegram Commands

Once the system is running, you can interact with it via Telegram:

- `/start` - Begin authentication process
- `/status` - Get system status
- `/analyze <symbol>` - Analyze market for symbol
- `/strategy <symbol>` - Generate trading strategy
- `/risk` - Get risk metrics
- `/allocation` - View capital allocation settings
- `/set_allocation <percentage>` - Set allocation percentage (0-100)
- `/performance` - Get performance metrics
- `/settings` - View current settings
- `/help` - Show available commands

## Development

### Running Tests

```bash
# Run all tests
python -m unittest discover -s llm_overseer/tests

# Run specific test
python -m llm_overseer.tests.test_compounding
```

### Adding New Features

1. Implement new functionality in the appropriate module
2. Add tests to verify behavior
3. Update configuration if needed
4. Document changes in this README

## Cost Optimization

The tiered LLM approach optimizes costs by:

1. Using cheaper models for routine tasks (Tier 1)
2. Using mid-range models for analysis tasks (Tier 2)
3. Reserving expensive models for critical strategy decisions (Tier 3)

This approach reduces monthly costs from $500+ to $100-150 while maintaining high-quality decision making.

## License

This project is proprietary and confidential. All rights reserved.

## Contributors

- Efficientlyy Team
