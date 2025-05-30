# Fix Docker build issues for Market Data Processor
# This script addresses Windows-specific Docker build problems

# Stop any running containers
Write-Host "Stopping existing containers..." -ForegroundColor Cyan
docker-compose down -v

# Step 1: Fix the .dockerignore file to ensure proper context
Write-Host "Creating proper .dockerignore..." -ForegroundColor Cyan
@"
# Generic files to ignore
.git/
.github/
.vscode/
**/*.rs.bk
**/.DS_Store

# Don't ignore dashboard directory
!dashboard/
"@ | Out-File -FilePath ".dockerignore" -Encoding utf8

# Step 2: Create a fixed Dockerfile that works in Windows
Write-Host "Creating Windows-compatible Dockerfile..." -ForegroundColor Cyan
@"
# syntax=docker/dockerfile:1

# Stage 1: Node.js builder for the dashboard
FROM node:18-slim as dashboard-builder

WORKDIR /app/dashboard

# Copy package.json and package-lock.json
COPY dashboard/package*.json ./

# Install dependencies with proper Windows file paths handling
RUN npm ci

# Copy dashboard source code
COPY dashboard/ ./

# Build the dashboard
RUN npm run build

# Stage 2: Rust builder for the backend
FROM rust:1.68-slim-bullseye as backend-builder

WORKDIR /app

# Install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    protobuf-compiler \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Cargo files for dependency caching
COPY Cargo.toml ./
COPY Cargo.lock* ./

# Create dummy source files to build dependencies
RUN mkdir -p src && \
    echo "fn main() {}" > src/main.rs && \
    echo "pub fn dummy() {}" > src/lib.rs

# Build dependencies
RUN cargo build --release

# Remove the dummy source files
RUN rm -rf src

# Copy the actual source code (excluding dashboard directory)
COPY src/ src/
COPY config/ config/
COPY tests/ tests/
COPY build.rs ./

# Build the application
RUN cargo build --release

# Stage 3: Runtime stage
FROM debian:bullseye-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl1.1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the binary from the backend-builder stage
COPY --from=backend-builder /app/target/release/market-data-processor /app/market-data-processor

# Create dashboard directory
RUN mkdir -p /app/dashboard/build

# Copy the dashboard from the dashboard-builder stage
COPY --from=dashboard-builder /app/dashboard/build /app/dashboard/build

# Create required directories
RUN mkdir -p /app/config /app/logs /app/data

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# Add environment variables
ENV PAPER_TRADING=true
ENV SERVE_DASHBOARD=true
ENV DASHBOARD_PATH=/app/dashboard/build
ENV HTTP_SERVER_ADDR=0.0.0.0:8080
ENV GRPC_SERVER_ADDR=0.0.0.0:50051
ENV LOG_LEVEL=info
ENV ENABLE_TELEMETRY=true

# Expose HTTP port for dashboard
EXPOSE 8080

# Expose gRPC port
EXPOSE 50051

# Set the entrypoint
ENTRYPOINT ["/app/market-data-processor"]
"@ | Out-File -FilePath "Dockerfile.windows" -Encoding utf8

# Step 3: Create a fixed docker-compose file
Write-Host "Creating fixed docker-compose.yml..." -ForegroundColor Cyan
@"
version: '3.8'
name: trading-agent-dashboard

services:
  market-data-processor:
    build:
      context: .
      dockerfile: Dockerfile.windows
    networks:
      - trading-network
    ports:
      - "8080:8080"  # HTTP/Dashboard port
      - "50051:50051" # gRPC port
    environment:
      # Paper Trading configuration
      - PAPER_TRADING=true
      - SERVE_DASHBOARD=true
      - PAPER_TRADING_INITIAL_BALANCE_USDT=10000
      - PAPER_TRADING_INITIAL_BALANCE_BTC=1
      - PAPER_TRADING_SLIPPAGE_MODEL=REALISTIC
      - PAPER_TRADING_LATENCY_MODEL=NORMAL
      - PAPER_TRADING_FEE_RATE=0.001
      
      # Trading configuration
      - TRADING_PAIRS=BTCUSDT,ETHUSDT
      - DEFAULT_ORDER_SIZE=0.1
      
      # Server configuration
      - HTTP_SERVER_ADDR=0.0.0.0:8080
      - GRPC_SERVER_ADDR=0.0.0.0:50051
      
      # Logging and telemetry
      - LOG_LEVEL=info
      - ENABLE_TELEMETRY=true
    volumes:
      - ./logs:/app/logs
      - ./config:/app/config
      - ./data:/app/data
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    networks:
      - trading-network
    volumes:
      - ./monitoring/prometheus:/etc/prometheus
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'
    ports:
      - "9090:9090"
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    networks:
      - trading-network
    volumes:
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning
      - grafana-data:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD=trading123
      - GF_USERS_ALLOW_SIGN_UP=false
    ports:
      - "3000:3000"
    restart: unless-stopped

volumes:
  prometheus-data:
  grafana-data:

networks:
  trading-network:
    driver: bridge
"@ | Out-File -FilePath "docker-compose.fixed.yml" -Encoding utf8

# Step 4: Make sure prometheus config exists and is valid
Write-Host "Ensuring Prometheus configuration is valid..." -ForegroundColor Cyan
if (-not (Test-Path "monitoring/prometheus/prometheus.yml")) {
    Write-Host "Creating Prometheus configuration..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path "monitoring/prometheus" -Force | Out-Null
    @"
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'market-data-processor'
    static_configs:
      - targets: ['market-data-processor:8080']
    metrics_path: '/metrics'
"@ | Out-File -FilePath "monitoring/prometheus/prometheus.yml" -Encoding utf8
}

# Step 5: Verify required dashboard directories
Write-Host "Verifying dashboard directory structure..." -ForegroundColor Cyan
if (-not (Test-Path "dashboard/src")) {
    Write-Host "WARNING: dashboard/src directory not found!" -ForegroundColor Red
    Write-Host "Please make sure your dashboard source code is in place" -ForegroundColor Yellow
}

# Step 6: Build and start containers
Write-Host "Building and starting containers..." -ForegroundColor Cyan
docker-compose -f docker-compose.fixed.yml build --no-cache
docker-compose -f docker-compose.fixed.yml up -d

# Step 7: Monitor container startup
Write-Host "Monitoring container startup..." -ForegroundColor Cyan
Start-Sleep -Seconds 10
docker-compose -f docker-compose.fixed.yml ps
Write-Host "Checking logs for market-data-processor..." -ForegroundColor Cyan
docker-compose -f docker-compose.fixed.yml logs market-data-processor

Write-Host "`nComplete system deployment started. Access your dashboard at:" -ForegroundColor Green
Write-Host "- Trading Dashboard: http://localhost:8080" -ForegroundColor Yellow
Write-Host "- Grafana: http://localhost:3000 (admin/trading123)" -ForegroundColor Yellow
Write-Host "- Prometheus: http://localhost:9090" -ForegroundColor Yellow
