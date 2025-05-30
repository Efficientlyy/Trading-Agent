# Debugging script for Docker in WSL2
# This directly addresses the known issues with the Trading Agent system in Windows

# Stop any existing containers first
Write-Host "Stopping any existing containers..." -ForegroundColor Cyan
docker-compose down

# Create required directories
Write-Host "Creating required directories..." -ForegroundColor Cyan
$dirs = @("./logs", "./data", "./config", "./monitoring/prometheus", "./monitoring/grafana/provisioning/dashboards")
foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force
        Write-Host "Created $dir" -ForegroundColor Green
    }
}

# Fix path issues by setting absolute paths
Write-Host "Creating Docker Compose file with fixed paths..." -ForegroundColor Cyan
$currentDir = (Get-Location).Path.Replace('\', '/')

$dockerComposeContent = @"
version: '3.8'
name: trading-agent-dashboard

services:
  # Main application: Market Data Processor with Paper Trading
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
      - $currentDir/logs:/app/logs
      - $currentDir/config:/app/config
      - $currentDir/data:/app/data
    # Removed dependencies on healthchecks to allow startup
    restart: unless-stopped

  # Prometheus for metrics collection
  prometheus:
    image: prom/prometheus:latest
    networks:
      - trading-network
    volumes:
      - $currentDir/monitoring/prometheus:/etc/prometheus
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

  # Grafana for metrics visualization
  grafana:
    image: grafana/grafana:latest
    networks:
      - trading-network
    volumes:
      - $currentDir/monitoring/grafana/provisioning:/etc/grafana/provisioning
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

$dockerComposeContent | Out-File -FilePath "docker-compose.debug.yml" -Encoding utf8

# Create simplified Prometheus config
Write-Host "Creating simplified Prometheus config..." -ForegroundColor Cyan
$prometheusConfig = @"
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
"@

if (-not (Test-Path "monitoring/prometheus")) {
    New-Item -ItemType Directory -Path "monitoring/prometheus" -Force
}
$prometheusConfig | Out-File -FilePath "monitoring/prometheus/prometheus.yml" -Encoding utf8

# Start with Docker Compose in debug mode
Write-Host "Starting Docker services (debug mode)..." -ForegroundColor Cyan
docker-compose -f docker-compose.debug.yml up -d

# Check service status
Write-Host "Checking service status..." -ForegroundColor Cyan
docker-compose -f docker-compose.debug.yml ps

# Offer to show logs
Write-Host "`nDo you want to see the logs? (y/n)" -ForegroundColor Yellow
$showLogs = Read-Host
if ($showLogs -eq "y") {
    docker-compose -f docker-compose.debug.yml logs
}

Write-Host "`n=== DASHBOARD ACCESS INFORMATION ===" -ForegroundColor Green
Write-Host "If services started successfully, access your dashboard at:" -ForegroundColor Cyan
Write-Host "- Grafana: http://localhost:3000 (admin/trading123)" -ForegroundColor Yellow
Write-Host "- Prometheus: http://localhost:9090" -ForegroundColor Yellow
Write-Host "- Market Data API: http://localhost:8080" -ForegroundColor Yellow
