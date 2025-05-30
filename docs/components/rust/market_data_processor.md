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
2. Build the project:
   ```bash
   cargo build
   ```
3. Run tests:
   ```bash
   cargo test
   ```
4. Run benchmarks:
   ```bash
   cargo bench
   ```
5. Run the service:
   ```bash
   cargo run
   ```

### Docker Development

For containerized development:

```bash
docker-compose up market-data-processor
```

## Configuration

The service is configured via environment variables:

- `RUST_LOG`: Log level (e.g., `info`, `debug`)
- `MEXC_WS_URL`: MEXC WebSocket URL
- `MEXC_REST_URL`: MEXC REST API URL
- `GRPC_SERVER_ADDR`: gRPC server address
- `DATABASE_URL`: PostgreSQL connection string
- `RABBITMQ_URL`: RabbitMQ connection string
- `REDIS_URL`: Redis connection string

## Windows Development Notes

When developing on Windows, consider the following:

- Use WSL2 for better Docker performance
- Use forward slashes in paths, even on Windows
- Set Git to use LF line endings:
  ```bash
  git config --global core.autocrlf input
  ```
- Use the VSCode Remote - Containers extension for a consistent development experience
