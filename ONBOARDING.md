# MEXC Trading System - New Developer Onboarding Guide

Welcome to the MEXC Trading System project! This guide will help you get up and running quickly with the latest implementation. The system provides real-time BTC/USDC trading capabilities with a complete dashboard, paper trading functionality, and comprehensive monitoring.

## Table of Contents

1. [Getting Started](#getting-started)
2. [System Architecture](#system-architecture)
3. [Development Workflow](#development-workflow)
4. [Key Components](#key-components)
5. [Testing](#testing)
6. [Troubleshooting](#troubleshooting)
7. [Next Steps](#next-steps)

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Efficientlyy/Trading-Agent.git
cd Trading-Agent
```

### 2. Review Documentation

Before diving into the code, familiarize yourself with:
- `README.md` - Overview and basic setup instructions
- `/docs` directory - Detailed documentation on system components
- `/architecture` directory - System architecture diagrams and explanations

### 3. Set Up Environment

Create a `.env` file in the root directory with your MEXC API credentials:

```
MEXC_API_KEY=your_api_key_here
MEXC_API_SECRET=your_api_secret_here
```

Note: The system will work with public data even without API keys, but trading functionality will be limited.

### 4. Start the System

```bash
docker-compose up -d
```

This will start all components:
- Market Data Processor: http://localhost:8080
- Grafana Dashboard: http://localhost:3000 (login: admin/trading123)
- Prometheus: http://localhost:9090

### 5. Verify Installation

Open your browser and navigate to:
- http://localhost:8080 - You should see the trading dashboard with BTC/USDC charts
- http://localhost:3000 - Grafana monitoring (login: admin/trading123)

## System Architecture

The MEXC Trading System follows a modular architecture with these key components:

1. **Market Data Processor (Rust)**
   - Connects to MEXC API for real-time market data
   - Processes and distributes data to other components
   - Located in `/boilerplate/rust/market-data-processor/`

2. **Dashboard (React)**
   - User interface for trading and market analysis
   - Displays real-time charts, order book, and trade history
   - Located in `/boilerplate/rust/market-data-processor/dashboard/`

3. **Paper Trading Engine (Rust)**
   - Simulates trading with real market data
   - Manages paper trading accounts and positions
   - Located in `/boilerplate/rust/market-data-processor/src/paper_trading/`

4. **Monitoring (Prometheus/Grafana)**
   - System metrics and performance monitoring
   - Located in `/boilerplate/monitoring/`

## Development Workflow

### Backend Development (Rust)

1. Navigate to the Market Data Processor directory:
   ```bash
   cd boilerplate/rust/market-data-processor
   ```

2. Build the project:
   ```bash
   cargo build
   ```

3. Run tests:
   ```bash
   cargo test
   ```

4. Run the service locally:
   ```bash
   cargo run
   ```

### Frontend Development (React)

1. Navigate to the dashboard directory:
   ```bash
   cd boilerplate/rust/market-data-processor/dashboard
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start development server:
   ```bash
   npm start
   ```

4. Build for production:
   ```bash
   npm run build
   ```

## Key Components

### MEXC API Integration

The system integrates with MEXC exchange through:

1. **REST API Client** (`src/api/mexc_client.rs`)
   - Handles market data requests
   - Manages order placement and account information

2. **WebSocket Client** (`src/api/websocket.rs`)
   - Provides real-time market data streaming
   - Handles reconnection and error recovery

### Dashboard Components

1. **Price Chart** (`dashboard/src/components/PriceChart.js`)
   - Displays BTC/USDC price charts with technical indicators
   - Supports multiple timeframes

2. **Order Book** (`dashboard/src/components/OrderBook.js`)
   - Shows real-time market depth
   - Visualizes bid/ask spread

3. **Trade History** (`dashboard/src/components/TradeHistory.js`)
   - Displays recent market trades
   - Updates in real-time

4. **Paper Trading Interface** (`dashboard/src/components/PaperTrading.js`)
   - Allows simulated trading with real market data
   - Manages paper trading account and positions

### Data Flow

1. Market Data Processor connects to MEXC API
2. Real-time data is processed and stored
3. WebSocket server streams data to frontend
4. Dashboard components visualize data
5. Paper trading engine simulates order execution

## Testing

### Backend Testing

```bash
cd boilerplate/rust/market-data-processor
cargo test
```

Key test files:
- `src/api/mexc_client_test.rs` - Tests for MEXC API client
- `src/paper_trading/mod_test.rs` - Tests for paper trading engine

### Frontend Testing

```bash
cd boilerplate/rust/market-data-processor/dashboard
npm test
```

### End-to-End Testing

See the test plan and results in the `/test_results` directory.

## Troubleshooting

### Common Issues

1. **WebSocket Connection Failures**
   - Check network connectivity
   - Verify WebSocket server is running
   - Check browser console for errors

2. **Docker Deployment Issues**
   - Ensure Docker and Docker Compose are installed
   - Check container logs: `docker-compose logs`
   - Verify port availability

3. **API Connection Issues**
   - Verify API credentials in `.env` file
   - Check MEXC API status
   - Review rate limiting considerations

### Debugging Tools

1. **Logs**
   - Backend logs: `docker-compose logs market-data-processor`
   - Frontend logs: Browser developer console

2. **Metrics**
   - Prometheus: http://localhost:9090
   - Grafana dashboards: http://localhost:3000

## Next Steps

1. **Explore the Codebase**
   - Review the implementation of key components
   - Understand the data flow between components

2. **Run the Test Suite**
   - Verify all components are working correctly
   - Understand test coverage and expectations

3. **Make a Small Change**
   - Fix a minor issue or add a small feature
   - Submit a pull request to get familiar with the workflow

4. **Explore Extension Points**
   - Additional trading pairs
   - New technical indicators
   - Enhanced paper trading features

## Getting Help

If you encounter any issues or have questions, please:
1. Check the documentation in the `/docs` directory
2. Review existing issues on GitHub
3. Reach out to the team for assistance

Welcome aboard, and happy coding!
