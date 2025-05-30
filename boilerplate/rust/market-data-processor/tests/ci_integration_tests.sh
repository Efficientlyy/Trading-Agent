#!/bin/bash
# CI/CD Integration Test Runner
# This script is designed to be run in a CI/CD pipeline to validate the paper trading module

set -e

echo "=== CI/CD Paper Trading Module Integration Tests ==="
echo "Starting CI tests at $(date)"

# Set environment variables for testing
export RUST_BACKTRACE=1
export PAPER_TRADING=true
export PAPER_TRADING_INITIAL_BALANCE_USDT=10000
export PAPER_TRADING_INITIAL_BALANCE_BTC=1
export MAX_POSITION_SIZE=1.0
export DEFAULT_ORDER_SIZE=0.1
export MAX_DRAWDOWN_PERCENT=10
export TRADING_PAIRS=BTCUSDT,ETHUSDT
export TEST_MODE=ci

# Directory setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_DIR/data/historical"
REPORT_DIR="$PROJECT_DIR/test-reports"

echo "Project directory: $PROJECT_DIR"
echo "Data directory: $DATA_DIR"
echo "Report directory: $REPORT_DIR"

# Create directories if they don't exist
mkdir -p "$DATA_DIR"
mkdir -p "$REPORT_DIR"

# Run tests in Docker container
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
    -e TEST_MODE=ci \
    -e TEST_DATA_DIR=/app/data/historical \
    -e TEST_REPORT_DIR=/app/test-reports \
    mexc-trading-agent-test \
    --test "*" --test-threads=1

# Generate HTML report from test results
echo "Generating HTML test report..."
docker run --rm \
    -v "$PROJECT_DIR:/app" \
    -e TEST_REPORT_DIR=/app/test-reports \
    mexc-trading-agent-test \
    /bin/bash -c "cd /app && cargo run --bin generate_test_report -- --input /app/test-reports/test_results.json --output /app/test-reports/test_report.html --format html"

# Generate Markdown report for CI systems
echo "Generating Markdown test report..."
docker run --rm \
    -v "$PROJECT_DIR:/app" \
    -e TEST_REPORT_DIR=/app/test-reports \
    mexc-trading-agent-test \
    /bin/bash -c "cd /app && cargo run --bin generate_test_report -- --input /app/test-reports/test_results.json --output /app/test-reports/test_report.md --format markdown"

echo "Test reports generated at: $REPORT_DIR"
echo "Tests completed at $(date)"
echo "=== End of CI/CD Test Run ==="
