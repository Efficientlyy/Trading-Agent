# WSL2 Dashboard Deployment Script for MEXC Trading System
# This script handles WSL2 setup and Docker deployment for the trading dashboard

# Define colors for better readability
function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    }
    else {
        $input | Write-Output
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

function Write-Success($message) {
    Write-ColorOutput Green "[SUCCESS] $message"
}

function Write-Info($message) {
    Write-ColorOutput Cyan "[INFO] $message" 
}

function Write-Warning($message) {
    Write-ColorOutput Yellow "[WARNING] $message"
}

function Write-Error($message) {
    Write-ColorOutput Red "[ERROR] $message"
}

# Check if WSL2 is installed and configured
function Check-WSL2 {
    Write-Info "Checking WSL2 installation..."
    
    try {
        $wslInfo = wsl --status
        if ($wslInfo -match "Default Version: 2") {
            Write-Success "WSL2 is installed and set as default"
            return $true
        } else {
            Write-Warning "WSL is installed but WSL2 may not be the default version"
            $confirmation = Read-Host "Would you like to set WSL2 as the default version? (y/n)"
            if ($confirmation -eq 'y') {
                wsl --set-default-version 2
                Write-Success "WSL2 is now the default version"
                return $true
            } else {
                Write-Warning "Proceeding without changing WSL default version"
                return $true
            }
        }
    } catch {
        Write-Error "WSL2 is not installed or an error occurred while checking"
        Write-Info "Please follow the WSL2 installation guide in docs/WSL2_SETUP_GUIDE.md"
        Start-Process "https://docs.microsoft.com/en-us/windows/wsl/install"
        return $false
    }
}

# Check if Docker Desktop is installed and configured for WSL2
function Check-DockerDesktop {
    Write-Info "Checking Docker Desktop installation..."
    
    try {
        $dockerVersion = docker --version
        if ($dockerVersion) {
            Write-Success "Docker is installed: $dockerVersion"
            
            # Try to check if it's using WSL2 backend by running a command in WSL
            try {
                $wslDocker = wsl -d Ubuntu -e docker --version 2>$null
                if ($wslDocker) {
                    Write-Success "Docker is accessible from WSL: $wslDocker"
                    return $true
                } else {
                    Write-Warning "Docker may not be configured to use WSL2 backend"
                    Write-Info "Please enable WSL2 integration in Docker Desktop settings"
                    $confirmation = Read-Host "Continue anyway? (y/n)"
                    return ($confirmation -eq 'y')
                }
            } catch {
                Write-Warning "Unable to check Docker in WSL"
                Write-Info "Please ensure Docker Desktop is running and WSL2 integration is enabled"
                $confirmation = Read-Host "Continue anyway? (y/n)"
                return ($confirmation -eq 'y')
            }
        } else {
            Write-Error "Docker not found"
            Write-Info "Please install Docker Desktop from https://www.docker.com/products/docker-desktop"
            Start-Process "https://www.docker.com/products/docker-desktop"
            return $false
        }
    } catch {
        Write-Error "Docker is not installed or an error occurred while checking"
        Write-Info "Please install Docker Desktop from https://www.docker.com/products/docker-desktop"
        Start-Process "https://www.docker.com/products/docker-desktop"
        return $false
    }
}

# Create .wslconfig if it doesn't exist
function Configure-WSL {
    Write-Info "Checking WSL2 configuration..."
    
    $wslConfigPath = "$env:USERPROFILE\.wslconfig"
    
    if (-not (Test-Path $wslConfigPath)) {
        Write-Info "Creating .wslconfig file for optimal performance..."
        $wslConfig = @"
[wsl2]
memory=8GB
processors=4
swap=2GB
"@
        $wslConfig | Out-File -FilePath $wslConfigPath -Encoding ASCII
        Write-Success "Created .wslconfig file at $wslConfigPath"
        
        Write-Warning "WSL2 configuration has been updated. A restart of WSL is recommended."
        $restart = Read-Host "Restart WSL now? (y/n)"
        if ($restart -eq 'y') {
            wsl --shutdown
            Write-Success "WSL has been restarted. New configuration is now active."
        }
    } else {
        Write-Success ".wslconfig already exists at $wslConfigPath"
    }
    
    return $true
}

# Check if Ubuntu distribution is installed
function Check-Ubuntu {
    Write-Info "Checking Ubuntu distribution..."
    
    try {
        $wslList = wsl -l
        if ($wslList -match "Ubuntu") {
            Write-Success "Ubuntu distribution is installed"
            return $true
        } else {
            Write-Warning "Ubuntu distribution not found"
            Write-Info "Installing Ubuntu from Microsoft Store..."
            Start-Process "ms-windows-store://pdp/?productid=9PDXGNCFSCZV"
            Write-Info "After installation completes, please run this script again"
            return $false
        }
    } catch {
        Write-Error "An error occurred while checking Ubuntu distribution"
        return $false
    }
}

# Ensure project is in WSL filesystem for optimal performance
function Prepare-ProjectInWSL {
    Write-Info "Preparing project in WSL filesystem for optimal performance..."
    
    $projectPath = Get-Location
    $projectName = (Get-Item $projectPath).Name
    
    # Check if we're in Windows path and offer to move to WSL filesystem
    if ($projectPath -match "^[A-Z]:\\") {
        Write-Warning "Project is currently on Windows filesystem ($projectPath)"
        Write-Info "For optimal performance, it's recommended to clone the project to the WSL filesystem"
        
        $moveToWSL = Read-Host "Would you like to clone the project to WSL filesystem? (y/n)"
        if ($moveToWSL -eq 'y') {
            # Clone to WSL filesystem
            $wslHome = wsl -d Ubuntu -e bash -c "echo ~" | Out-String
            $wslHome = $wslHome.Trim()
            $wslProjectPath = "$wslHome/projects/$projectName"
            
            Write-Info "Creating directory $wslProjectPath in WSL..."
            wsl -d Ubuntu -e mkdir -p ~/projects
            
            # Check if the directory already exists in WSL
            $dirExists = wsl -d Ubuntu -e test -d $wslProjectPath -a -e $wslProjectPath/docker-compose.dashboard.yml 2>$null
            if ($LASTEXITCODE -eq 0) {
                Write-Warning "Project already exists in WSL at $wslProjectPath"
                $updateExisting = Read-Host "Update existing project? (y/n)"
                if ($updateExisting -eq 'y') {
                    # Sync only key files
                    Write-Info "Syncing key files to WSL filesystem..."
                    Copy-WSLFiles -sourcePath $projectPath -destPath $wslProjectPath
                } else {
                    Write-Info "Using existing project in WSL"
                }
            } else {
                # Copy project to WSL
                Write-Info "Copying project to WSL filesystem..."
                Copy-WSLFiles -sourcePath $projectPath -destPath $wslProjectPath
            }
            
            Write-Success "Project prepared in WSL at $wslProjectPath"
            return $wslProjectPath
        } else {
            Write-Warning "Continuing with project on Windows filesystem. Performance may be affected."
            return $projectPath
        }
    } else {
        Write-Success "Project is already on WSL filesystem"
        return $projectPath
    }
}

function Copy-WSLFiles {
    param (
        [string]$sourcePath,
        [string]$destPath
    )
    
    # Create necessary directories in WSL
    wsl -d Ubuntu -e mkdir -p $destPath
    wsl -d Ubuntu -e mkdir -p $destPath/monitoring/grafana/provisioning/dashboards
    wsl -d Ubuntu -e mkdir -p $destPath/monitoring/prometheus
    
    # Copy key files to WSL
    Write-Info "Copying docker-compose.dashboard.yml..."
    Get-Content "$sourcePath\docker-compose.dashboard.yml" | wsl -d Ubuntu -e bash -c "cat > $destPath/docker-compose.dashboard.yml"
    
    Write-Info "Copying Grafana dashboards..."
    Get-Content "$sourcePath\monitoring\grafana\provisioning\dashboards\trading_performance.json" | wsl -d Ubuntu -e bash -c "cat > $destPath/monitoring/grafana/provisioning/dashboards/trading_performance.json"
    Get-Content "$sourcePath\monitoring\grafana\provisioning\dashboards\trading_overview.json" | wsl -d Ubuntu -e bash -c "cat > $destPath/monitoring/grafana/provisioning/dashboards/trading_overview.json"
    Get-Content "$sourcePath\monitoring\grafana\provisioning\dashboards\dashboard.yml" | wsl -d Ubuntu -e bash -c "cat > $destPath/monitoring/grafana/provisioning/dashboards/dashboard.yml"
    
    Write-Info "Copying Prometheus configuration..."
    Get-Content "$sourcePath\monitoring\prometheus\prometheus.yml" | wsl -d Ubuntu -e bash -c "cat > $destPath/monitoring/prometheus/prometheus.yml"
    Get-Content "$sourcePath\monitoring\prometheus\alert_rules.yml" | wsl -d Ubuntu -e bash -c "cat > $destPath/monitoring/prometheus/alert_rules.yml"
    
    Write-Success "Project files copied to WSL filesystem"
}

# Deploy the dashboard using Docker in WSL
function Deploy-Dashboard {
    param (
        [string]$projectPath
    )
    
    Write-Info "Deploying dashboard using Docker in WSL..."
    
    # If project is on Windows filesystem, convert path to WSL format
    $wslPath = $projectPath
    if ($projectPath -match "^[A-Z]:\\") {
        $drive = $projectPath[0].ToString().ToLower()
        $relativePath = $projectPath.Substring(3).Replace('\', '/')
        $wslPath = "/mnt/$drive/$relativePath"
        Write-Info "Converted Windows path to WSL path: $wslPath"
    }
    
    # Run Docker commands in WSL
    try {
        # Pull required images first
        Write-Info "Pulling required Docker images..."
        wsl -d Ubuntu -e bash -c "cd $wslPath && docker-compose -f docker-compose.dashboard.yml pull"
        
        # Stop any running containers
        Write-Info "Stopping any existing dashboard containers..."
        wsl -d Ubuntu -e bash -c "cd $wslPath && docker-compose -f docker-compose.dashboard.yml down"
        
        # Start services
        Write-Info "Starting dashboard services..."
        wsl -d Ubuntu -e bash -c "cd $wslPath && docker-compose -f docker-compose.dashboard.yml up -d"
        
        # Check if services are running
        Start-Sleep -Seconds 5
        $services = wsl -d Ubuntu -e bash -c "cd $wslPath && docker-compose -f docker-compose.dashboard.yml ps"
        
        if ($services -match "running") {
            Write-Success "Dashboard services are running successfully"
            return $true
        } else {
            Write-Warning "Some services may not be running properly"
            Write-Info "Services status:"
            Write-Output $services
            
            $checkLogs = Read-Host "Would you like to view service logs? (y/n)"
            if ($checkLogs -eq 'y') {
                wsl -d Ubuntu -e bash -c "cd $wslPath && docker-compose -f docker-compose.dashboard.yml logs"
            }
            
            return $false
        }
    } catch {
        Write-Error "An error occurred while deploying the dashboard: $_"
        return $false
    }
}

# Run container-specific healthchecks
function Check-ContainerHealth {
    param (
        [string]$projectPath
    )
    
    Write-Info "Checking container health..."
    
    # If project is on Windows filesystem, convert path to WSL format
    $wslPath = $projectPath
    if ($projectPath -match "^[A-Z]:\\") {
        $drive = $projectPath[0].ToString().ToLower()
        $relativePath = $projectPath.Substring(3).Replace('\', '/')
        $wslPath = "/mnt/$drive/$relativePath"
    }
    
    # Check health status of each container
    $healthStatus = wsl -d Ubuntu -e bash -c "cd $wslPath && docker-compose -f docker-compose.dashboard.yml ps"
    
    # Parse health status from output
    $prometheusHealth = if ($healthStatus -match "prometheus.*Up.*\(healthy\)") { "Healthy" } else { "Unhealthy" }
    $grafanaHealth = if ($healthStatus -match "grafana.*Up.*\(healthy\)") { "Healthy" } else { "Unhealthy" }
    $marketDataHealth = if ($healthStatus -match "market-data-processor.*Up.*\(healthy\)") { "Healthy" } else { "Unhealthy" }
    
    Write-Info "Container Health Status:"
    Write-Info "- Prometheus: $prometheusHealth"
    Write-Info "- Grafana: $grafanaHealth"
    Write-Info "- Market Data Processor: $marketDataHealth"
    
    # If any containers are unhealthy, offer to view logs
    if ($prometheusHealth -eq "Unhealthy" -or $grafanaHealth -eq "Unhealthy" -or $marketDataHealth -eq "Unhealthy") {
        Write-Warning "Some containers are not healthy"
        $viewLogs = Read-Host "Would you like to view logs for unhealthy containers? (y/n)"
        if ($viewLogs -eq 'y') {
            if ($prometheusHealth -eq "Unhealthy") {
                Write-Info "Prometheus logs:"
                wsl -d Ubuntu -e bash -c "cd $wslPath && docker-compose -f docker-compose.dashboard.yml logs prometheus"
            }
            if ($grafanaHealth -eq "Unhealthy") {
                Write-Info "Grafana logs:"
                wsl -d Ubuntu -e bash -c "cd $wslPath && docker-compose -f docker-compose.dashboard.yml logs grafana"
            }
            if ($marketDataHealth -eq "Unhealthy") {
                Write-Info "Market Data Processor logs:"
                wsl -d Ubuntu -e bash -c "cd $wslPath && docker-compose -f docker-compose.dashboard.yml logs market-data-processor"
            }
        }
        return $false
    } else {
        Write-Success "All containers are healthy"
        return $true
    }
}

# Display dashboard access information
function Show-DashboardInfo {
    Write-Info "`n=== DASHBOARD ACCESS INFORMATION ===`n"
    Write-Info "Your trading dashboard is now available at the following URLs:"
    Write-ColorOutput Yellow "- Grafana Trading Dashboard: http://localhost:3000"
    Write-ColorOutput Yellow "  Username: admin"
    Write-ColorOutput Yellow "  Password: trading123"
    Write-ColorOutput Yellow "- Prometheus Metrics: http://localhost:9090"
    Write-ColorOutput Yellow "- Market Data Processor API: http://localhost:8080"
    
    Write-Info "`n=== DASHBOARD NAVIGATION GUIDE ===`n"
    Write-Info "1. Open Grafana at http://localhost:3000 and log in"
    Write-Info "2. Click on the dashboard dropdown in the top navigation bar"
    Write-Info "3. Select 'Trading Performance' or 'Trading Overview' dashboard"
    Write-Info "4. For detailed metrics, visit Prometheus at http://localhost:9090"
    
    Write-Info "`n=== TROUBLESHOOTING ===`n"
    Write-Info "If you encounter any issues:"
    Write-ColorOutput Yellow "- Check container logs: wsl -d Ubuntu -e docker-compose -f /path/to/docker-compose.dashboard.yml logs"
    Write-ColorOutput Yellow "- Restart services: wsl -d Ubuntu -e docker-compose -f /path/to/docker-compose.dashboard.yml restart"
    Write-ColorOutput Yellow "- For more help, refer to docs/WSL2_SETUP_GUIDE.md"
    
    # Offer to open dashboard in browser
    $openBrowser = Read-Host "Would you like to open the Grafana dashboard in your browser? (y/n)"
    if ($openBrowser -eq 'y') {
        Start-Process "http://localhost:3000"
    }
}

# Main execution
Write-ColorOutput Magenta "`n=== MEXC Trading Dashboard - WSL2 Deployment ===`n"

# Check prerequisites
$wslOk = Check-WSL2
$dockerOk = Check-DockerDesktop
$ubuntuOk = Check-Ubuntu

if (-not ($wslOk -and $dockerOk -and $ubuntuOk)) {
    Write-Error "Prerequisites check failed. Please fix the issues and run the script again."
    exit 1
}

# Configure WSL
$wslConfigured = Configure-WSL
if (-not $wslConfigured) {
    Write-Error "WSL configuration failed. Please fix the issues and run the script again."
    exit 1
}

# Prepare project in WSL filesystem
$projectPath = Prepare-ProjectInWSL

# Deploy dashboard
$deployed = Deploy-Dashboard -projectPath $projectPath
if (-not $deployed) {
    Write-Warning "Dashboard deployment might have issues. Proceeding with health checks..."
}

# Check container health
$healthy = Check-ContainerHealth -projectPath $projectPath
if (-not $healthy) {
    Write-Warning "Some containers are not healthy. The dashboard may not function correctly."
    $continue = Read-Host "Continue anyway? (y/n)"
    if ($continue -ne 'y') {
        exit 1
    }
}

# Show dashboard information
Show-DashboardInfo

Write-Success "`nDashboard deployment completed successfully!"
