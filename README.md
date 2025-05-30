# MEXC Trading System

A complete trading system for MEXC exchange with real-time BTC/USDC data, charts, and paper trading functionality.

## Features

- Real-time BTC/USDC market data from MEXC exchange
- Interactive price charts with technical indicators
- Live order book visualization
- Real-time trade history
- Paper trading with simulated account
- System monitoring with Prometheus and Grafana

## Architecture

The system follows a modular architecture with the following components:

- **Market Data Processor (Rust)**: Connects to MEXC API and processes market data
- **Dashboard (React)**: User interface for trading and market analysis
- **Paper Trading Engine (Rust)**: Simulates trading with real market data
- **Monitoring (Prometheus/Grafana)**: System metrics and performance monitoring

## Getting Started

### Prerequisites

- Docker and Docker Compose
- MEXC API credentials (optional for public data)

### Environment Setup

1. Clone this repository:
```bash
git clone https://github.com/yourusername/mexc-trading-system.git
cd mexc-trading-system
```

2. Create a `.env` file with your MEXC API credentials (optional):
```
MEXC_API_KEY=your_api_key
MEXC_API_SECRET=your_api_secret
```

### Running the System

Start the entire system using Docker Compose:

```bash
docker-compose up -d
```

This will start all components:
- Market Data Processor: http://localhost:8080
- Grafana Dashboard: http://localhost:3000 (login: admin/trading123)
- Prometheus: http://localhost:9090

### Accessing the Trading Dashboard

Open your browser and navigate to:

```
http://localhost:8080
```

## Development

### Building the Market Data Processor

```bash
cd boilerplate/rust/market-data-processor
cargo build --release
```

### Building the Dashboard

```bash
cd boilerplate/rust/market-data-processor/dashboard
npm install
npm run build
```

### Running Tests

```bash
cd boilerplate/rust/market-data-processor
cargo test
```

## Configuration

The system can be configured through environment variables or a `config.json` file:

- `MEXC_DEFAULT_PAIR`: Default trading pair (default: BTCUSDC)
- `MEXC_API_KEY`: Your MEXC API key
- `MEXC_API_SECRET`: Your MEXC API secret
- `PAPER_TRADING_INITIAL_USDC`: Initial USDC balance for paper trading
- `PAPER_TRADING_INITIAL_BTC`: Initial BTC balance for paper trading

## Troubleshooting

### Windows Docker Issues

If you encounter issues with Docker on Windows:

1. Use WSL2 for Docker Desktop
2. Ensure proper volume mounting with correct paths
3. Check network connectivity between containers

### WebSocket Connection Issues

If the dashboard is not receiving real-time updates:

1. Check browser console for WebSocket errors
2. Verify the Market Data Processor is running
3. Ensure no firewall is blocking WebSocket connections

## License

This project is licensed under the MIT License - see the LICENSE file for details.
