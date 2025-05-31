# Windows PowerShell script for running the Trading Agent system
# This script handles common Windows-specific Docker issues

Write-Host "=========================================="
Write-Host "  BTC/USDC Trading System Startup Script  " -ForegroundColor Green
Write-Host "=========================================="

# Check if Docker is running
$dockerRunning = $false
try {
    $dockerStatus = docker info 2>&1
    $dockerRunning = $true
} catch {
    $dockerRunning = $false
}

if (-not $dockerRunning) {
    Write-Host "Docker is not running. Please start Docker Desktop and try again." -ForegroundColor Red
    exit 1
}

Write-Host "✓ Docker is running" -ForegroundColor Green

# Create necessary directories if they don't exist
$directories = @(
    ".\boilerplate\rust\market-data-processor\logs",
    ".\boilerplate\rust\market-data-processor\data"
)

foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        Write-Host "Creating directory: $dir"
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}

Write-Host "✓ Directories created" -ForegroundColor Green

# Check if .env file exists and has required values
if (-not (Test-Path ".\.env")) {
    Write-Host "Creating .env file with placeholders"
    @"
MEXC_API_KEY=your_api_key_here
MEXC_API_SECRET=your_api_secret_here
PAPER_TRADING=true
"@ | Out-File -FilePath ".\.env" -Encoding utf8
}

Write-Host "✓ Environment file checked" -ForegroundColor Green

# Remove any old containers to prevent conflicts
Write-Host "Cleaning up old containers..."
docker-compose -f docker-compose.fixed.yml down 2>&1 | Out-Null

# Check if the containers exist and remove them individually if needed
$containersToRemove = @("market-data-processor", "prometheus", "grafana", "jaeger")
foreach ($container in $containersToRemove) {
    $containerExists = docker ps -a --filter "name=$container" --format "{{.Names}}"
    if ($containerExists) {
        Write-Host "Removing container: $container"
        docker rm -f $container 2>&1 | Out-Null
    }
}

Write-Host "✓ Old containers removed" -ForegroundColor Green

# Pull the latest images
Write-Host "Pulling latest Docker images..."
docker-compose -f docker-compose.fixed.yml pull 2>&1 | Out-Null

Write-Host "✓ Docker images updated" -ForegroundColor Green

# Start the containers with the fixed configuration
Write-Host "Starting the Trading Agent system..."
docker-compose -f docker-compose.fixed.yml up -d

# Check if containers started successfully
Start-Sleep -Seconds 5
$containersRunning = docker ps --filter "name=market-data-processor" --format "{{.Names}}"

if ($containersRunning) {
    Write-Host "=========================================="
    Write-Host "  Trading Agent System Started Successfully!  " -ForegroundColor Green
    Write-Host "=========================================="
    Write-Host "Dashboard URL: http://localhost:8080"
    Write-Host "Prometheus: http://localhost:9090"
    Write-Host "Grafana: http://localhost:3000 (admin/trading123)"
    Write-Host "Jaeger: http://localhost:16686"
    
    # Open the dashboard in the default browser
    Write-Host "Opening dashboard in browser..."
    Start-Process "http://localhost:8080"
} else {
    Write-Host "Failed to start containers. Checking logs..." -ForegroundColor Red
    docker-compose -f docker-compose.fixed.yml logs
}

Write-Host ""
Write-Host "To stop the system, run: docker-compose -f docker-compose.fixed.yml down"
