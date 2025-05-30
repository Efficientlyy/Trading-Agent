#!/bin/bash
set -e

# This script runs after the devcontainer is created
# It sets up the development environment for all components

echo "Setting up development environment..."

# Install common dependencies
apt-get update
apt-get install -y curl wget git build-essential pkg-config libssl-dev

# Setup for Rust components
if [ -d "/workspace/rust" ]; then
  echo "Setting up Rust environment..."
  cd /workspace/rust
  
  # Install Rust toolchain if not already installed
  if ! command -v rustup &> /dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source $HOME/.cargo/env
  fi
  
  # Install additional Rust tools
  rustup component add rustfmt clippy
  cargo install cargo-watch cargo-expand cargo-criterion
  
  # Build Rust components to pre-cache dependencies
  if [ -d "/workspace/rust/market-data-processor" ]; then
    cd /workspace/rust/market-data-processor
    cargo build
  fi
  
  if [ -d "/workspace/rust/order-execution" ]; then
    cd /workspace/rust/order-execution
    cargo build
  fi
fi

# Setup for Node.js components
if [ -d "/workspace/nodejs" ]; then
  echo "Setting up Node.js environment..."
  cd /workspace/nodejs
  
  # Install Node.js dependencies
  if [ -d "/workspace/nodejs/decision-service" ]; then
    cd /workspace/nodejs/decision-service
    npm install
  fi
  
  if [ -d "/workspace/nodejs/api-gateway" ]; then
    cd /workspace/nodejs/api-gateway
    npm install
  fi
fi

# Setup for Python components
if [ -d "/workspace/python" ]; then
  echo "Setting up Python environment..."
  cd /workspace/python
  
  # Create and activate virtual environment if not exists
  if [ -d "/workspace/python/signal-generator" ]; then
    cd /workspace/python/signal-generator
    if [ ! -d ".venv" ]; then
      python -m venv .venv
    fi
    source .venv/bin/activate
    pip install -r requirements.txt
    deactivate
  fi
fi

# Setup for frontend components
if [ -d "/workspace/frontend" ]; then
  echo "Setting up frontend environment..."
  cd /workspace/frontend
  
  if [ -d "/workspace/frontend/dashboard" ]; then
    cd /workspace/frontend/dashboard
    npm install
  fi
fi

echo "Development environment setup complete!"
