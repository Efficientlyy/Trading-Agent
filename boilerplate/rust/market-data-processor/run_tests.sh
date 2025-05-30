#!/bin/bash
# Test runner script for paper trading module
set -e

echo "=== Paper Trading Module Integration Tests ==="
echo "Starting tests at $(date)"

# Build and run tests in Docker container
cd "$(dirname "$0")"
PROJECT_DIR="$(pwd)"

echo "Project directory: $PROJECT_DIR"

# Run Docker-based tests
if command -v docker &> /dev/null; then
    echo "Using Docker for consistent test environment"
    
    # Build test image
    echo "Building test Docker image..."
    docker build -t mexc-trading-agent-test -f Dockerfile.test .
    
    # Run tests in container
    echo "Running tests in Docker container..."
    docker run --rm \
        -v "$PROJECT_DIR:/app" \
        -e RUST_BACKTRACE=1 \
        -e PAPER_TRADING=true \
        -e PAPER_TRADING_INITIAL_BALANCE_USDT=10000 \
        -e PAPER_TRADING_INITIAL_BALANCE_BTC=1 \
        -e MAX_POSITION_SIZE=1.0 \
        -e DEFAULT_ORDER_SIZE=0.1 \
        -e MAX_DRAWDOWN_PERCENT=10 \
        -e TRADING_PAIRS=BTCUSDT,ETHUSDT \
        mexc-trading-agent-test \
        --test market_data_processor_tests --test-threads=1
    
    echo "Docker tests completed"
else
    echo "Docker not available, running tests directly"
    # Set environment variables
    export RUST_BACKTRACE=1
    export PAPER_TRADING=true
    export PAPER_TRADING_INITIAL_BALANCE_USDT=10000
    export PAPER_TRADING_INITIAL_BALANCE_BTC=1
    export MAX_POSITION_SIZE=1.0
    export DEFAULT_ORDER_SIZE=0.1
    export MAX_DRAWDOWN_PERCENT=10
    export TRADING_PAIRS=BTCUSDT,ETHUSDT
    
    # Run tests
    cargo test --test market_data_processor_tests --test-threads=1
fi

echo "Tests completed at $(date)"
echo "=== End of Test Run ==="
