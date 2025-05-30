# PowerShell script for running the validation framework
# Takes into account Windows-specific issues and PowerShell syntax requirements

# Create required directories
Write-Host "Creating required directories..." -ForegroundColor Cyan
if (-not (Test-Path ".\tests\performance\results")) {
    New-Item -ItemType Directory -Path ".\tests\performance\results" -Force | Out-Null
}

# Function to check if Docker is available
function Test-DockerAvailable {
    try {
        $null = docker ps
        return $true
    }
    catch {
        return $false
    }
}

# Validate Environment
Write-Host "`n=== Environment Validation ===" -ForegroundColor Green
Write-Host "Checking system environment and configuration..."

# Detect if running in WSL
$isWsl = $false
if (Test-Path "\\wsl$") {
    Write-Host "WSL detected on this system" -ForegroundColor Yellow
    # We're not in WSL, but WSL is available
}

# Check Docker availability
$dockerAvailable = Test-DockerAvailable
if ($dockerAvailable) {
    Write-Host "Docker is available" -ForegroundColor Green
    
    # Check Docker Desktop backend
    $dockerInfo = docker info
    if ($dockerInfo -match "WSL") {
        Write-Host "Docker Desktop is using WSL2 backend (recommended)" -ForegroundColor Green
    }
    elseif ($dockerInfo -match "Hyper-V") {
        Write-Host "Docker Desktop is using Hyper-V backend" -ForegroundColor Yellow
        Write-Host "  Note: WSL2 backend is recommended for better performance" -ForegroundColor Yellow
    }
} 
else {
    Write-Host "Docker is not available" -ForegroundColor Red
    Write-Host "  Some validation tests will be skipped" -ForegroundColor Red
}

# Run performance baseline tests
Write-Host "`n=== Performance Baseline Tests ===" -ForegroundColor Green
Write-Host "Running performance baseline tests..."

# Run Rust tests (needs to be executed separately in PowerShell)
try {
    Set-Location -Path "."
    Write-Host "Running baseline tests..."
    cargo test --package market-data-processor --test baseline_test -- --nocapture

    Write-Host "Running integration tests..."
    cargo test --package market-data-processor --test integration_test -- --nocapture

    Write-Host "Running environment validation tests..."
    cargo test --package market-data-processor --test environment_validation -- --nocapture
}
catch {
    Write-Host "Error running tests: $_" -ForegroundColor Red
}

# If Docker is available, test Docker configuration
if ($dockerAvailable) {
    Write-Host "`n=== Docker Configuration Tests ===" -ForegroundColor Green
    
    # Special handling for Windows volume paths
    Write-Host "Testing Docker volume configuration..."
    $testVolumeOutput = docker run --rm -v "${PWD}/tests:/tests" alpine ls -la /tests
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Docker volume mounts are working correctly" -ForegroundColor Green
    }
    else {
        Write-Host "Docker volume mount test failed" -ForegroundColor Red
        Write-Host "  This is a common issue on Windows systems" -ForegroundColor Red
        Write-Host "  Solutions:" -ForegroundColor Yellow
        Write-Host "  1. Ensure Docker has permission to access the local filesystem" -ForegroundColor Yellow
        Write-Host "  2. Check Docker Desktop settings > Resources > File Sharing" -ForegroundColor Yellow
        Write-Host "  3. Consider using WSL2 as recommended in WSL2_SETUP_GUIDE.md" -ForegroundColor Yellow
    }
    
    # Test simplified Docker Compose configuration
    Write-Host "`nTesting simplified Docker Compose configuration..."
    try {
        docker-compose -f docker-compose.simple.v2.yml config
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Docker Compose configuration is valid" -ForegroundColor Green
        }
        else {
            Write-Host "Docker Compose configuration has errors" -ForegroundColor Red
        }
    }
    catch {
        Write-Host "Error validating Docker Compose configuration: $_" -ForegroundColor Red
    }
}

# Dashboard Usability Testing instructions
Write-Host "`n=== Dashboard Usability Testing ===" -ForegroundColor Green
Write-Host "To conduct dashboard usability testing:"
Write-Host "1. Start the system using: docker-compose -f docker-compose.simple.v2.yml up -d"
Write-Host "2. Access Grafana at: http://localhost:3000 (admin/trading123)"
Write-Host "3. Follow the testing scenarios in tests/performance/dashboard_usability_test.md"
Write-Host "4. Record results using the provided template"

# Display test result summary if available
Write-Host "`n=== Results Summary ===" -ForegroundColor Green
$resultFiles = Get-ChildItem -Path ".\tests\performance\results" -Filter "*.json" -ErrorAction SilentlyContinue

if ($resultFiles.Count -gt 0) {
    Write-Host "Found $($resultFiles.Count) result files:"
    foreach ($file in $resultFiles) {
        Write-Host "- $($file.Name)"
    }
    
    Write-Host "`nTo view detailed results, examine the JSON files in the tests/performance/results directory"
}
else {
    Write-Host "No test result files found. Run the tests first to generate results."
}

# Next steps
Write-Host "`n=== Next Steps ===" -ForegroundColor Green
Write-Host "1. Review the test results in the tests/performance/results directory"
Write-Host "2. Conduct dashboard usability testing with multiple users"
Write-Host "3. Compile findings and implement improvements"
Write-Host "4. Proceed to Documentation Expansion phase"

Write-Host "`nValidation framework execution complete!" -ForegroundColor Cyan
