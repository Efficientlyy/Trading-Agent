#!/bin/bash
# Integration Test Runner for Paper Trading Module
# Compatible with Docker containerized environment

set -e

# Print header
echo "========================================================"
echo "Paper Trading Module - Integration Test Runner"
echo "========================================================"
echo "Starting tests at $(date)"
echo "Using containerized test environment"

# Directory setup
TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$TEST_DIR/.." && pwd)"
RESULTS_DIR="$TEST_DIR/results"

# Ensure results directory exists
mkdir -p "$RESULTS_DIR"

# Set environment variables for testing
export RUST_LOG=info
export PAPER_TRADING=true
export PAPER_TRADING_INITIAL_BALANCE_USDT=10000
export PAPER_TRADING_INITIAL_BALANCE_BTC=1
export MAX_POSITION_SIZE=1.0
export DEFAULT_ORDER_SIZE=0.1
export MAX_DRAWDOWN_PERCENT=10
export TRADING_PAIRS="BTCUSDT,ETHUSDT"

echo "Building test binary..."
cargo build --bin integration_test --release

echo "Running integration tests..."
RUST_BACKTRACE=1 cargo run --bin integration_test --release

# Check exit code
if [ $? -eq 0 ]; then
    echo "✅ All integration tests passed!"
else
    echo "❌ Integration tests failed!"
    exit 1
fi

# Generate final report
if [ -f "$RESULTS_DIR/latest_metrics.json" ]; then
    echo "Generating test report..."
    cargo run --bin generate_test_report --release -- --input "$RESULTS_DIR/latest_metrics.json" --output "$RESULTS_DIR/test_report.md"
    
    echo "Test report generated at $RESULTS_DIR/test_report.md"
fi

echo "Tests completed at $(date)"
echo "========================================================"
