#!/bin/bash
# Unified Launcher Script for Trading Agent System
# This script starts all system components in the correct order

set -e

# Banner
echo "=========================================================="
echo "       Trading Agent System - Unified Launcher"
echo "=========================================================="

# Detect environment
if [ -f ".env" ]; then
    echo "Loading environment variables from .env file"
    export $(grep -v '^#' .env | xargs)
fi

# Default configuration
ENVIRONMENT=${ENVIRONMENT:-"development"}
USE_DOCKER=${USE_DOCKER:-"false"}
LOG_LEVEL=${LOG_LEVEL:-"info"}
PAPER_TRADING=${PAPER_TRADING:-"true"}
SERVE_DASHBOARD=${SERVE_DASHBOARD:-"true"}
HTTP_PORT=${HTTP_PORT:-"8080"}
GRPC_PORT=${GRPC_PORT:-"50051"}

echo "Detected environment: $ENVIRONMENT"
echo "Paper trading mode: $PAPER_TRADING"
echo "Serve dashboard: $SERVE_DASHBOARD"

# Create required directories
mkdir -p logs
mkdir -p data
mkdir -p data/historical

# Function to check if Docker is available
check_docker() {
    if ! command -v docker &> /dev/null; then
        echo "Docker is not installed. Cannot use Docker mode."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        echo "Docker Compose is not installed. Cannot use Docker mode."
        exit 1
    fi
    
    echo "Docker and Docker Compose are available."
}

# Function to check if dashboard is built
check_dashboard() {
    if [ ! -d "dashboard/build" ]; then
        echo "Dashboard is not built. Building now..."
        cd dashboard
        npm install
        npm run build
        cd ..
        echo "Dashboard built successfully."
    else
        echo "Dashboard already built."
    fi
}

# Function to start system in Docker mode
start_docker_mode() {
    echo "Starting system in Docker mode..."
    docker-compose up --build
}

# Function to start system in native mode
start_native_mode() {
    echo "Starting system in native mode..."
    
    # Export environment variables
    export RUST_BACKTRACE=1
    export PAPER_TRADING=$PAPER_TRADING
    export SERVE_DASHBOARD=$SERVE_DASHBOARD
    export HTTP_SERVER_ADDR="0.0.0.0:$HTTP_PORT"
    export DASHBOARD_PATH="./dashboard/build"
    export LOG_LEVEL=$LOG_LEVEL
    
    # Start market data processor
    echo "Starting Market Data Processor..."
    cargo run --release &
    
    # Save the process ID
    MDP_PID=$!
    
    # Wait for system to start
    echo "Waiting for system to start..."
    sleep 5
    
    # Display access information
    echo "=========================================================="
    echo "System started successfully!"
    echo "Dashboard: http://localhost:$HTTP_PORT"
    echo "gRPC API: localhost:$GRPC_PORT"
    echo "Logs: ./logs"
    echo "=========================================================="
    echo "Press Ctrl+C to stop the system"
    
    # Wait for Ctrl+C
    trap "kill $MDP_PID; echo 'System stopped.'; exit 0" INT
    wait $MDP_PID
}

# Main logic
case $USE_DOCKER in
    true)
        check_docker
        start_docker_mode
        ;;
    *)
        if [ "$SERVE_DASHBOARD" = "true" ]; then
            check_dashboard
        fi
        start_native_mode
        ;;
esac
