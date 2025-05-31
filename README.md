# Flash Trading System for MEXC

This repository contains an ultra-fast flash trading system designed for the MEXC exchange, with a focus on zero-fee trading pairs (BTCUSDC, ETHUSDC).

## Key Features

- **Ultra-Low Latency**: Optimized for microsecond-level response times
- **Zero-Fee Trading**: Focused on BTCUSDC and ETHUSDC pairs for maximum cost efficiency
- **Paper Trading**: Test strategies with real market data but no financial risk
- **Advanced Signal Generation**: Multiple signal strategies including order book imbalance, momentum, and volatility
- **Modular Architecture**: Clean separation between data acquisition, processing, signal generation, and execution

## System Architecture

The system is built with a modular architecture consisting of six distinct layers:

1. **Data Acquisition Layer**: Handles market data from MEXC and external sources
2. **Data Processing Layer**: Processes and stores market data
3. **Signal Generation Layer**: Analyzes data to generate trading signals
4. **Decision Making Layer**: Makes trading decisions based on signals
5. **Execution Layer**: Executes trades and manages positions
6. **Visualization Layer**: Provides interactive dashboards and visualizations

## Components

### Optimized MEXC Client

The `OptimizedMexcClient` provides ultra-fast connectivity to the MEXC API with features like:

- Connection pooling for reduced latency
- Request caching for frequently accessed data
- Asynchronous operations for non-blocking execution
- Robust error handling and retry mechanisms

### Paper Trading System

The `PaperTradingSystem` allows testing strategies with real market data but without financial risk:

- Simulates order placement and execution
- Tracks virtual balances and positions
- Applies realistic slippage and partial fills
- Maintains order and trade history

### Signal Generator

The `SignalGenerator` analyzes market data to identify trading opportunities:

- Order book imbalance detection
- Price momentum analysis
- Volatility breakout signals
- Multi-factor signal aggregation

### Flash Trading Integration

The `FlashTradingSystem` integrates all components for end-to-end operation:

- Configurable trading parameters
- Real-time signal processing
- Paper trading execution
- Performance monitoring and statistics

## Getting Started

### Prerequisites

- Python 3.8+
- Access to MEXC API (API key and secret)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Efficientlyy/Trading-Agent.git
cd Trading-Agent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
mkdir -p .env-secure
echo "MEXC_API_KEY=your_api_key" > .env-secure/.env
echo "MEXC_API_SECRET=your_api_secret" >> .env-secure/.env
```

### Configuration

The system is configured through `flash_trading_config.py`, which creates a default configuration file:

```bash
python flash_trading_config.py
```

This generates `flash_trading_config.json` with settings for:
- Trading pairs (BTCUSDC, ETHUSDC)
- Paper trading parameters
- Signal generation thresholds
- Execution rules

### Running the System

To run the flash trading system with paper trading:

```bash
python flash_trading.py --duration 3600 --reset
```

Options:
- `--duration`: Run time in seconds
- `--reset`: Reset paper trading balances to initial values
- `--env`: Path to environment file (default: `.env-secure/.env`)
- `--config`: Path to configuration file (default: `flash_trading_config.json`)

## Development

### Testing Components Individually

Test the optimized MEXC client:
```bash
python optimized_mexc_client.py --benchmark
```

Test the paper trading system:
```bash
python paper_trading.py --test
```

Test the signal generator:
```bash
python flash_trading_signals.py --test --duration 60
```

### Adding New Signal Strategies

To add a new signal strategy:
1. Extend the `generate_signals` method in `flash_trading_signals.py`
2. Add configuration parameters in `flash_trading_config.py`
3. Test the strategy with paper trading before live deployment

## Security Notes

- API keys are stored in `.env-secure/.env` which is excluded from git
- Never commit API keys or secrets to the repository
- Use paper trading for all testing and development

## License

This project is proprietary and confidential.

## Acknowledgments

- MEXC API documentation and examples
- Contributors to the Trading-Agent project
