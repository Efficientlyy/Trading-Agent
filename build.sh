#!/bin/bash
# Render build script for Trading-Agent

echo "Starting build process for Trading-Agent..."

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p .env-secure

# Copy environment template if needed
if [ ! -f .env-secure/.env ]; then
  echo "Creating environment file from template..."
  cp .env-secure/.env.template .env-secure/.env
fi

echo "Build process completed successfully."
