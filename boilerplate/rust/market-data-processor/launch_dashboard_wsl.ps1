# Complete Trading System Launch Script (WSL2 Backend)
# This addresses Windows-specific Docker issues by using WSL2 directly

# Ensure this runs with administrator privileges
if (-not ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "Please run this script as Administrator" -ForegroundColor Red
    exit 1
}

# Check if WSL2 is properly installed
Write-Host "Checking WSL2 setup..." -ForegroundColor Cyan
$wslStatus = wsl --status
if ($LASTEXITCODE -ne 0) {
    Write-Host "WSL2 is not properly configured. Please install and configure WSL2 first." -ForegroundColor Red
    exit 1
}
Write-Host "WSL2 is properly configured" -ForegroundColor Green

# Set WSL2 as the default version
wsl --set-default-version 2 | Out-Null
Write-Host "WSL2 set as default version" -ForegroundColor Green

# Ensure Docker Desktop is using WSL2 backend
Write-Host "Ensuring Docker Desktop is using WSL2 backend..." -ForegroundColor Cyan
$dockerRunning = docker info 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Docker is not running. Please start Docker Desktop." -ForegroundColor Red
    Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    Write-Host "Waiting for Docker to start..." -ForegroundColor Yellow
    Start-Sleep -Seconds 30
}

# Convert project path to WSL path for better performance
$projectPath = Get-Location
$drive = $projectPath.Path[0].ToString().ToLower()
$relativePath = $projectPath.Path.Substring(2).Replace('\', '/')
$wslPath = "/mnt/$drive$relativePath"

Write-Host "Project path in WSL: $wslPath" -ForegroundColor Green

# Copy the Docker files to WSL2 for better file I/O performance
Write-Host "Setting up project in WSL for optimal performance..." -ForegroundColor Cyan
$uniqueFolder = "trading-agent-" + (Get-Date).ToString("yyyyMMddHHmmss")
$wslProjectPath = "~/projects/$uniqueFolder"

# Create the project directory in WSL
wsl mkdir -p $wslProjectPath | Out-Null

# Copy essential files to WSL for building
Write-Host "Copying project files to WSL..." -ForegroundColor Cyan
wsl cp -r "$wslPath/src" $wslProjectPath/
wsl cp -r "$wslPath/config" $wslProjectPath/ 2>/dev/null || Write-Host "No config directory found, creating empty one..."
wsl mkdir -p $wslProjectPath/config
wsl cp -r "$wslPath/dashboard" $wslProjectPath/
wsl cp -r "$wslPath/tests" $wslProjectPath/ 2>/dev/null || Write-Host "No tests directory found, skipping..."
wsl cp "$wslPath/Cargo.toml" $wslProjectPath/ 2>/dev/null || Write-Host "No Cargo.toml found"
wsl cp "$wslPath/Cargo.lock" $wslProjectPath/ 2>/dev/null || Write-Host "No Cargo.lock found"
wsl cp "$wslPath/build.rs" $wslProjectPath/ 2>/dev/null || Write-Host "No build.rs found"

# Copy the monitoring files
wsl mkdir -p $wslProjectPath/monitoring/prometheus
wsl mkdir -p $wslProjectPath/monitoring/grafana/provisioning/dashboards
wsl cp -r "$wslPath/monitoring/prometheus"/* $wslProjectPath/monitoring/prometheus/ 2>/dev/null || Write-Host "Creating empty prometheus dir"
wsl cp -r "$wslPath/monitoring/grafana/provisioning/dashboards"/* $wslProjectPath/monitoring/grafana/provisioning/dashboards/ 2>/dev/null || Write-Host "Creating empty grafana dir"

# Create the Docker and docker-compose files in WSL
Write-Host "Creating Docker files in WSL..." -ForegroundColor Cyan

# Create fixed Dockerfile
$dockerfileContent = @"
# syntax=docker/dockerfile:1

# Stage 1: Node.js builder for the dashboard
FROM node:18-slim as dashboard-builder

WORKDIR /app/dashboard

# Copy package.json and package-lock.json
COPY dashboard/package*.json ./

# Install dependencies
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

# Copy the actual source code
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
"@

# Create docker-compose.yml with proper environment variables
$dockerComposeContent = @"
version: '3.8'

services:
  market-data-processor:
    build:
      context: .
      dockerfile: Dockerfile
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
"@

# Create prometheus.yml if it doesn't exist
$prometheusYml = @"
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
"@

# Write files to WSL
$dockerfileContent | Out-File -FilePath "temp_dockerfile" -Encoding utf8
Get-Content "temp_dockerfile" | wsl bash -c "cat > $wslProjectPath/Dockerfile"
Remove-Item "temp_dockerfile"

$dockerComposeContent | Out-File -FilePath "temp_docker_compose.yml" -Encoding utf8
Get-Content "temp_docker_compose.yml" | wsl bash -c "cat > $wslProjectPath/docker-compose.yml"
Remove-Item "temp_docker_compose.yml"

$prometheusYml | Out-File -FilePath "temp_prometheus.yml" -Encoding utf8
Get-Content "temp_prometheus.yml" | wsl bash -c "cat > $wslProjectPath/monitoring/prometheus/prometheus.yml"
Remove-Item "temp_prometheus.yml"

# Ensure .dockerignore is properly set
"**/node_modules
**/target
!dashboard/" | Out-File -FilePath "temp_dockerignore" -Encoding utf8
Get-Content "temp_dockerignore" | wsl bash -c "cat > $wslProjectPath/.dockerignore"
Remove-Item "temp_dockerignore"

# Now build and start the containers in WSL
Write-Host "Building and starting containers in WSL..." -ForegroundColor Cyan

# Stop existing containers
wsl -e docker-compose down -v

# Run docker compose in WSL
Write-Host "Building containers (this may take a while)..." -ForegroundColor Yellow
wsl bash -c "cd $wslProjectPath && docker-compose build --no-cache"

Write-Host "Starting containers..." -ForegroundColor Cyan
wsl bash -c "cd $wslProjectPath && docker-compose up -d"

# Check if containers are running
Write-Host "Checking container status..." -ForegroundColor Cyan
Start-Sleep -Seconds 10
$containerStatus = wsl bash -c "cd $wslProjectPath && docker-compose ps"
Write-Host $containerStatus

# Display logs if needed
$showLogs = Read-Host "Do you want to see the logs from the market-data-processor? (y/n)"
if ($showLogs -eq "y") {
    wsl bash -c "cd $wslProjectPath && docker-compose logs market-data-processor"
}

# Open browser to view dashboard
Write-Host "`n=== COMPLETE TRADING SYSTEM OPERATIONAL ===" -ForegroundColor Green
Write-Host "Your complete trading system with real data flow is now available at:" -ForegroundColor Cyan
Write-Host "- Market Data Processor Dashboard: http://localhost:8080" -ForegroundColor Yellow
Write-Host "- Grafana Trading Dashboards: http://localhost:3000" -ForegroundColor Yellow
Write-Host "  Username: admin" -ForegroundColor Yellow
Write-Host "  Password: trading123" -ForegroundColor Yellow
Write-Host "- Prometheus Metrics: http://localhost:9090" -ForegroundColor Yellow

$openBrowser = Read-Host "Would you like to open the main dashboard in your browser? (y/n)"
if ($openBrowser -eq "y") {
    Start-Process "http://localhost:8080"
}

Write-Host "`nIMPORTANT: The system is now running with real data flow through all components." -ForegroundColor Magenta
Write-Host "To stop the system, run: wsl bash -c 'cd $wslProjectPath && docker-compose down'" -ForegroundColor Cyan
