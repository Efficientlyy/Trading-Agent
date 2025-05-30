# Rust Market Data Processor Boilerplate

This is a boilerplate for the Market Data Processor component of the MEXC trading system, implemented in Rust.

## Project Structure

```
market-data-processor/
├── .cargo/
│   └── config.toml       # Cargo configuration
├── src/
│   ├── api/              # gRPC API implementation
│   │   ├── grpc.rs       # gRPC service implementation
│   │   ├── mod.rs        # API module exports
│   │   └── server.rs     # API server implementation
│   ├── models/           # Data models
│   │   ├── common.rs     # Common data types
│   │   ├── market.rs     # Market data models
│   │   ├── mod.rs        # Models module exports
│   │   ├── order_book.rs # Order book models
│   │   ├── ticker.rs     # Ticker models
│   │   ├── trade.rs      # Trade models
│   │   └── websocket.rs  # WebSocket message models
│   ├── services/         # Core services
│   │   ├── market_data_service.rs # Market data processing service
│   │   ├── message_parser.rs      # WebSocket message parser
│   │   ├── mod.rs                 # Services module exports
│   │   └── order_book_manager.rs  # Order book state manager
│   ├── utils/            # Utility functions
│   │   ├── config.rs     # Configuration handling
│   │   ├── error.rs      # Error types and handling
│   │   ├── logging.rs    # Logging setup
│   │   ├── metrics.rs    # Metrics collection
│   │   ├── mod.rs        # Utils module exports
│   │   └── websocket.rs  # WebSocket client utilities
│   ├── lib.rs            # Library entry point
│   └── main.rs           # Application entry point
├── tests/                # Integration tests
│   └── integration_test.rs # Integration test suite
├── benches/              # Performance benchmarks
│   └── benchmarks.rs     # Benchmark suite
├── proto/                # Protocol Buffers definitions
│   └── market_data.proto # Market data service definition
├── .gitignore           # Git ignore file
├── build.rs             # Build script for code generation
├── Cargo.toml           # Cargo manifest
├── Dockerfile           # Docker build configuration
├── README.md            # Project documentation
└── rustfmt.toml         # Rust formatting configuration
```

## Getting Started

### Prerequisites

- Rust toolchain (1.68 or later)
- Protocol Buffers compiler
- Docker and Docker Compose (for containerized development)

### Development Setup

1. Clone the repository
2. Install Protocol Buffers compiler (protoc)
3. Build the project:
   ```bash
   cargo build
   ```
4. Run tests:
   ```bash
   cargo test
   ```
5. Run the service:
   ```bash
   cargo run
   ```

### Configuration

The Market Data Processor uses a hierarchical configuration system:

1. Default configuration in `config/default.toml`
2. Environment-specific configuration (e.g., `config/development.toml`)
3. Local overrides in `config/local.toml` (git-ignored for your personal settings)
4. Environment variables prefixed with `APP__` (using double underscore as separator)

Key configuration options:

| Setting | Description | Default |
|---------|-------------|----------|
| `grpc_server_addr` | Address for the gRPC server | "0.0.0.0:50051" |
| `mexc_ws_url` | MEXC WebSocket API URL | "wss://wbs.mexc.com/ws" |
| `is_paper_trading` | Whether to use paper trading mode | true |
| `trading_pairs` | Trading pairs to subscribe to | ["BTCUSDT", "ETHUSDT"] |
| `log_level` | Logging level | "info" |
| `enable_telemetry` | Enable OpenTelemetry tracing | false |

You can set environment variables to override these settings:

```bash
# Example: Override server address and enable debug logging
APP__GRPC_SERVER_ADDR=0.0.0.0:5000 APP__LOG_LEVEL=debug cargo run
```

### Paper Trading Mode

The Market Data Processor is configured to run in paper trading mode by default, which:

1. Connects to the real MEXC WebSocket API to receive market data
2. Processes real market data for accurate testing
3. Does not require API credentials for read-only market data
4. Does not place actual trades

### Using the gRPC API

The Market Data Processor exposes a gRPC API for other components to consume market data:

```protobuf
service MarketDataService {
  rpc GetOrderBook (OrderBookRequest) returns (OrderBookResponse);
  rpc GetTicker (TickerRequest) returns (TickerResponse);
  rpc GetRecentTrades (RecentTradesRequest) returns (RecentTradesResponse);
  rpc SubscribeToOrderBookUpdates (OrderBookSubscriptionRequest) returns (stream OrderBookUpdate);
  rpc SubscribeToTrades (TradeSubscriptionRequest) returns (stream TradeUpdate);
  rpc SubscribeToTickers (TickerSubscriptionRequest) returns (stream TickerUpdate);
}
```

### Docker Development

For containerized development:

```bash
# Build the docker image
docker build -t market-data-processor .

# Run the container
docker run -p 50051:50051 market-data-processor
```

## Windows Development Notes

When developing on Windows:

- Use WSL2 for better Docker performance
- When editing Proto files, ensure they use LF line endings
- You may need to manually install the Protocol Buffers compiler:
  ```bash
  # Using Chocolatey
  choco install protoc
  ```
- Use the VSCode Remote - Containers extension for a consistent development experience
