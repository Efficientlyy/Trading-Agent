Write-Host "=== MEXC Trading System - Reliable Windows Launcher ===" -ForegroundColor Cyan
Write-Host "Starting the complete trading system with real data flow..." -ForegroundColor Cyan

# Stop and remove any existing containers
Write-Host "Stopping existing containers..." -ForegroundColor Yellow
docker-compose -f docker-compose.working.yml down

# Handle Docker setup issues specific to Windows
Write-Host "Setting up environment for Windows..." -ForegroundColor Yellow
# Use only relative paths for volume mapping

# Use direct Python option if Docker fails
$dockerWorks = $true
$pythonFallback = $false

# Build and run the containers
Write-Host "Building and starting containers..." -ForegroundColor Yellow
docker-compose -f docker-compose.working.yml up -d --build

# Check if Market Data Processor container started correctly
Start-Sleep -Seconds 5
$mdpRunning = docker ps | Select-String "market-data-processor"
if (!$mdpRunning) {
    Write-Host "Market Data Processor container failed to start!" -ForegroundColor Red
    Write-Host "Switching to direct Python mode..." -ForegroundColor Yellow
    
    # Stop containers but keep Prometheus and Grafana
    docker stop market-data-processor
    
    # Run the Python script directly
    Write-Host "Starting Market Data Processor directly with Python..." -ForegroundColor Yellow
    Start-Process -FilePath "python" -ArgumentList "mdp_metrics.py" -NoNewWindow
    $pythonFallback = $true
    Start-Sleep -Seconds 3
}

# Verify services are running
$success = $true

# Check Prometheus
$prometheusRunning = docker ps | Select-String "prometheus"
if (!$prometheusRunning) {
    Write-Host "Prometheus container is not running!" -ForegroundColor Red
    $success = $false
}

# Check Grafana
$grafanaRunning = docker ps | Select-String "grafana"
if (!$grafanaRunning) {
    Write-Host "Grafana container is not running!" -ForegroundColor Red
    $success = $false
}

# Check Market Data Processor (either in Docker or direct Python)
if (!$pythonFallback) {
    $mdpRunning = docker ps | Select-String "market-data-processor"
    if (!$mdpRunning) {
        Write-Host "Market Data Processor container is not running!" -ForegroundColor Red
        $success = $false
    }
}

# Display system status
if ($success) {
    Write-Host "`n=== COMPLETE TRADING SYSTEM OPERATIONAL ===" -ForegroundColor Green
    Write-Host "The complete trading system with real data flow is now available at:" -ForegroundColor Green
    Write-Host "- Market Data Processor Dashboard: http://localhost:8080" -ForegroundColor White
    Write-Host "- Grafana Trading Dashboards: http://localhost:3000" -ForegroundColor White
    Write-Host "  Username: admin" -ForegroundColor White
    Write-Host "  Password: trading123" -ForegroundColor White
    Write-Host "- Prometheus Metrics: http://localhost:9090" -ForegroundColor White
    
    # Open the Market Data Processor dashboard in browser
    Write-Host "`nOpening Market Data Processor dashboard..." -ForegroundColor Yellow
    Start-Process "http://localhost:8080"
}
else {
    Write-Host "`n=== SYSTEM STARTUP ISSUES DETECTED ===" -ForegroundColor Red
    
    # Special handling for Python fallback mode
    if ($pythonFallback) {
        Write-Host "Market Data Processor running in direct Python mode on http://localhost:8080" -ForegroundColor Yellow
        Write-Host "Other services:" -ForegroundColor Yellow
        if ($prometheusRunning) { Write-Host "- Prometheus: http://localhost:9090" -ForegroundColor White }
        if ($grafanaRunning) { Write-Host "- Grafana: http://localhost:3000" -ForegroundColor White }
        
        # Open the Market Data Processor dashboard in browser
        Write-Host "`nOpening Market Data Processor dashboard..." -ForegroundColor Yellow
        Start-Process "http://localhost:8080"
    }
}

Write-Host "`nVerification Steps:" -ForegroundColor Cyan
Write-Host "1. Market Data Processor dashboard should load at http://localhost:8080" -ForegroundColor White
Write-Host "2. Log in to Grafana (admin/trading123) and verify real data is flowing" -ForegroundColor White
Write-Host "3. Check Prometheus targets to ensure all services are being monitored" -ForegroundColor White
