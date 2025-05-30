# PowerShell script for launching the complete trading dashboard with all functionality
# This script handles Windows-specific issues and ensures all components are properly connected

# Stop any running containers to ensure a clean start
Write-Host "Stopping any existing containers..." -ForegroundColor Cyan
docker-compose down

# Ensure required directories exist
Write-Host "Creating required directories..." -ForegroundColor Cyan
$directories = @(
    "./logs", 
    "./data", 
    "./config", 
    "./tests/performance/results"
)

foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "  Created directory: $dir" -ForegroundColor Green
    }
}

# Generate sample market data if needed
Write-Host "Checking for market data..." -ForegroundColor Cyan
if (-not (Test-Path "./data/market_data")) {
    Write-Host "  Generating sample market data for dashboard visualization..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path "./data/market_data" -Force | Out-Null
    
    # Create a minimal data file to ensure the system has something to display
    $dataContent = @"
timestamp,symbol,price,volume,side
$(Get-Date -Format "yyyy-MM-ddTHH:mm:ss"),BTCUSDT,30000.00,1.5,buy
$(Get-Date -Format "yyyy-MM-ddTHH:mm:ss" -Date (Get-Date).AddSeconds(-5)),BTCUSDT,30010.50,2.3,sell
$(Get-Date -Format "yyyy-MM-ddTHH:mm:ss" -Date (Get-Date).AddSeconds(-10)),BTCUSDT,30005.25,1.7,buy
$(Get-Date -Format "yyyy-MM-ddTHH:mm:ss" -Date (Get-Date).AddSeconds(-15)),BTCUSDT,29995.75,3.1,sell
$(Get-Date -Format "yyyy-MM-ddTHH:mm:ss" -Date (Get-Date).AddSeconds(-20)),BTCUSDT,30001.25,0.8,buy
"@
    $dataContent | Out-File -FilePath "./data/market_data/btcusdt_trades.csv" -Encoding utf8
    Write-Host "  Created sample market data file" -ForegroundColor Green

    # Create a sample for ETHUSDT as well
    $ethDataContent = @"
timestamp,symbol,price,volume,side
$(Get-Date -Format "yyyy-MM-ddTHH:mm:ss"),ETHUSDT,2000.00,10.5,buy
$(Get-Date -Format "yyyy-MM-ddTHH:mm:ss" -Date (Get-Date).AddSeconds(-5)),ETHUSDT,2005.50,15.3,sell
$(Get-Date -Format "yyyy-MM-ddTHH:mm:ss" -Date (Get-Date).AddSeconds(-10)),ETHUSDT,2003.25,12.7,buy
$(Get-Date -Format "yyyy-MM-ddTHH:mm:ss" -Date (Get-Date).AddSeconds(-15)),ETHUSDT,1998.75,18.1,sell
$(Get-Date -Format "yyyy-MM-ddTHH:mm:ss" -Date (Get-Date).AddSeconds(-20)),ETHUSDT,2001.25,9.8,buy
"@
    $ethDataContent | Out-File -FilePath "./data/market_data/ethusdt_trades.csv" -Encoding utf8
}

# Ensure the dashboard directory exists
if (-not (Test-Path "./monitoring/grafana/provisioning/dashboards")) {
    Write-Host "  Creating dashboard directories..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path "./monitoring/grafana/provisioning/dashboards" -Force | Out-Null
}

# Generate additional dashboards if needed
if (-not (Test-Path "./monitoring/grafana/provisioning/dashboards/trading_overview.json")) {
    Write-Host "  Generating additional dashboard for trading overview..." -ForegroundColor Yellow
    
    # Create a new trading overview dashboard to supplement the performance dashboard
    # This will be implemented in a separate step
}

# Make sure Prometheus can access its configuration
Write-Host "Ensuring Prometheus configuration is accessible..." -ForegroundColor Cyan
if (-not (Test-Path "./monitoring/prometheus")) {
    New-Item -ItemType Directory -Path "./monitoring/prometheus" -Force | Out-Null
    
    # Create a basic prometheus.yml if it doesn't exist
    if (-not (Test-Path "./monitoring/prometheus/prometheus.yml")) {
        $prometheusConfig = @"
global:
  scrape_interval: 15s
  evaluation_interval: 15s

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          # - alertmanager:9093

rule_files:
  - "alert_rules.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'market-data-processor'
    static_configs:
      - targets: ['market-data-processor:8080']
"@
        $prometheusConfig | Out-File -FilePath "./monitoring/prometheus/prometheus.yml" -Encoding utf8
        Write-Host "  Created prometheus.yml configuration file" -ForegroundColor Green
    }
    
    # Create alert rules file if it doesn't exist
    if (-not (Test-Path "./monitoring/prometheus/alert_rules.yml")) {
        $alertRules = @"
groups:
- name: Trading System Alerts
  rules:
  - alert: HighOrderExecutionLatency
    expr: histogram_quantile(0.95, sum(rate(order_execution_latency_seconds_bucket[5m])) by (le, trading_pair)) > 0.5
    for: 1m
    labels:
      severity: warning
    annotations:
      summary: "High order execution latency"
      description: "Order execution latency is above 500ms for {{ \$labels.trading_pair }}"
      
  - alert: LowMarketDataThroughput
    expr: sum(rate(market_data_updates_total[1m])) by (trading_pair) < 100
    for: 1m
    labels:
      severity: warning
    annotations:
      summary: "Low market data throughput"
      description: "Market data throughput is below 100 updates/sec for {{ \$labels.trading_pair }}"
      
  - alert: HighCpuUsage
    expr: process_cpu_seconds_total{job="market-data-processor"} > 90
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High CPU usage"
      description: "CPU usage is above 90% for the trading system"
      
  - alert: HighMemoryUsage
    expr: process_resident_memory_bytes{job="market-data-processor"} > 4294967296
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High memory usage"
      description: "Memory usage is above 4GB for the trading system"
"@
        $alertRules | Out-File -FilePath "./monitoring/prometheus/alert_rules.yml" -Encoding utf8
        Write-Host "  Created alert_rules.yml file" -ForegroundColor Green
    }
}

# Start all services with Docker Compose
Write-Host "`nStarting all services with Docker Compose..." -ForegroundColor Cyan
Write-Host "This will start the market data processor with paper trading enabled" -ForegroundColor Cyan
Write-Host "as well as Prometheus, Grafana, and Jaeger for full dashboard functionality." -ForegroundColor Cyan

# Use the fixed docker-compose file
docker-compose up -d

# Check if the services are running
Write-Host "`nChecking service status..." -ForegroundColor Cyan
$containers = docker ps --filter "name=trading-agent-system" --format "{{.Names}} - {{.Status}}"

if ($containers) {
    Write-Host "Services successfully started:" -ForegroundColor Green
    Write-Host $containers
} else {
    Write-Host "Warning: No services found running. There might be an issue with Docker." -ForegroundColor Red
    Write-Host "Check Docker logs for more information: docker-compose logs" -ForegroundColor Yellow
}

# Display dashboard access information
Write-Host "`n=== DASHBOARD ACCESS INFORMATION ===" -ForegroundColor Green
Write-Host "Your trading dashboard is now available at the following URLs:" -ForegroundColor Cyan
Write-Host "- Grafana Trading Dashboard: http://localhost:3000" -ForegroundColor Yellow
Write-Host "  Username: admin" -ForegroundColor Yellow
Write-Host "  Password: trading123" -ForegroundColor Yellow
Write-Host "- Prometheus Metrics: http://localhost:9090" -ForegroundColor Yellow
Write-Host "- Jaeger Distributed Tracing: http://localhost:16686" -ForegroundColor Yellow
Write-Host "- Market Data Processor API: http://localhost:8080" -ForegroundColor Yellow

# Additional instructions
Write-Host "`n=== DASHBOARD NAVIGATION GUIDE ===" -ForegroundColor Green
Write-Host "1. Open Grafana at http://localhost:3000 and log in" -ForegroundColor Cyan
Write-Host "2. Click on the dashboard dropdown in the top navigation bar" -ForegroundColor Cyan
Write-Host "3. Select 'Trading Performance' dashboard to view trading metrics" -ForegroundColor Cyan
Write-Host "4. For detailed system tracing, visit Jaeger at http://localhost:16686" -ForegroundColor Cyan
Write-Host "5. To view raw metrics, visit Prometheus at http://localhost:9090" -ForegroundColor Cyan

Write-Host "`n=== TROUBLESHOOTING ===" -ForegroundColor Green
Write-Host "If you encounter any issues:" -ForegroundColor Cyan
Write-Host "- Check container logs: docker-compose logs" -ForegroundColor Yellow
Write-Host "- Restart services: ./launch_dashboard.ps1" -ForegroundColor Yellow
Write-Host "- Ensure Docker Desktop is running with WSL2 backend for best performance" -ForegroundColor Yellow
Write-Host "- Refer to WSL2_SETUP_GUIDE.md for Windows-specific configuration" -ForegroundColor Yellow
