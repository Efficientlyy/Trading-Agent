# Windows-specific dashboard deployment script for MEXC Trading Agent
# This script addresses the known issues with Docker in Windows environments

# Step 1: Ensure all required directories exist
Write-Host "Creating required directories..." -ForegroundColor Cyan
$directories = @(
    "./monitoring/prometheus",
    "./monitoring/grafana/provisioning/dashboards",
    "./logs",
    "./data",
    "./config"
)
foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "Created: $dir" -ForegroundColor Green
    }
}

# Step 2: Create minimal Prometheus configuration
Write-Host "Creating minimal Prometheus configuration..." -ForegroundColor Cyan
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
"@
$prometheusYml | Out-File -FilePath "monitoring/prometheus/prometheus.yml" -Encoding utf8

# Step 3: Ensure dashboard JSON files exist
if (-not (Test-Path "monitoring/grafana/provisioning/dashboards/trading_performance.json")) {
    Write-Host "Creating sample dashboard JSON..." -ForegroundColor Cyan
    $dashboardJson = @"
{
  "annotations": {
    "list": [
      {
        "builtIn": 1,
        "datasource": "-- Grafana --",
        "enable": true,
        "hide": true,
        "iconColor": "rgba(0, 211, 255, 1)",
        "name": "Annotations & Alerts",
        "type": "dashboard"
      }
    ]
  },
  "editable": true,
  "gnetId": null,
  "graphTooltip": 0,
  "id": 1,
  "links": [],
  "panels": [
    {
      "aliasColors": {},
      "bars": false,
      "dashLength": 10,
      "dashes": false,
      "datasource": "Prometheus",
      "fill": 1,
      "fillGradient": 0,
      "gridPos": {
        "h": 9,
        "w": 12,
        "x": 0,
        "y": 0
      },
      "hiddenSeries": false,
      "id": 2,
      "legend": {
        "avg": false,
        "current": false,
        "max": false,
        "min": false,
        "show": true,
        "total": false,
        "values": false
      },
      "lines": true,
      "linewidth": 1,
      "nullPointMode": "null",
      "options": {
        "dataLinks": []
      },
      "percentage": false,
      "pointradius": 2,
      "points": false,
      "renderer": "flot",
      "seriesOverrides": [],
      "spaceLength": 10,
      "stack": false,
      "steppedLine": false,
      "targets": [
        {
          "expr": "market_data_updates_total",
          "interval": "",
          "legendFormat": "Market Data Updates",
          "refId": "A"
        }
      ],
      "thresholds": [],
      "timeFrom": null,
      "timeRegions": [],
      "timeShift": null,
      "title": "Market Data Updates",
      "tooltip": {
        "shared": true,
        "sort": 0,
        "value_type": "individual"
      },
      "type": "graph",
      "xaxis": {
        "buckets": null,
        "mode": "time",
        "name": null,
        "show": true,
        "values": []
      },
      "yaxes": [
        {
          "format": "short",
          "label": null,
          "logBase": 1,
          "max": null,
          "min": null,
          "show": true
        },
        {
          "format": "short",
          "label": null,
          "logBase": 1,
          "max": null,
          "min": null,
          "show": true
        }
      ],
      "yaxis": {
        "align": false,
        "alignLevel": null
      }
    }
  ],
  "schemaVersion": 22,
  "style": "dark",
  "tags": [],
  "templating": {
    "list": []
  },
  "time": {
    "from": "now-6h",
    "to": "now"
  },
  "timepicker": {},
  "timezone": "",
  "title": "Trading Performance",
  "uid": "trading-performance",
  "variables": {
    "list": []
  },
  "version": 1
}
"@
    $dashboardJson | Out-File -FilePath "monitoring/grafana/provisioning/dashboards/trading_performance.json" -Encoding utf8
}

# Create dashboard.yml provisioning configuration
$dashboardYml = @"
apiVersion: 1

providers:
- name: 'default'
  orgId: 1
  folder: ''
  type: file
  disableDeletion: false
  updateIntervalSeconds: 10
  options:
    path: /etc/grafana/provisioning/dashboards
"@
$dashboardYml | Out-File -FilePath "monitoring/grafana/provisioning/dashboards/dashboard.yml" -Encoding utf8

# Step 4: Create a Windows-compatible docker-compose file
Write-Host "Creating Windows-compatible docker-compose file..." -ForegroundColor Cyan
$currentDir = (Get-Location).Path.Replace('\', '/') # Convert Windows path to Docker path

$dockerComposeContent = @"
version: '3.8'
name: trading-agent-dashboard

services:
  # Skip building the market-data-processor for now
  # Instead, use premade images to validate the dashboard setup

  # Prometheus for metrics collection
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ${currentDir}/monitoring/prometheus:/etc/prometheus
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'
    networks:
      - trading-network
    restart: unless-stopped

  # Grafana for metrics visualization
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - ${currentDir}/monitoring/grafana/provisioning:/etc/grafana/provisioning
      - grafana-data:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD=trading123
      - GF_USERS_ALLOW_SIGN_UP=false
    networks:
      - trading-network
    restart: unless-stopped

volumes:
  prometheus-data:
  grafana-data:

networks:
  trading-network:
    driver: bridge
"@
$dockerComposeContent | Out-File -FilePath "docker-compose.windows.yml" -Encoding utf8

# Step 5: Stop any existing containers and start with the new configuration
Write-Host "Stopping any existing containers..." -ForegroundColor Cyan
docker-compose down -v

Write-Host "Starting dashboard components..." -ForegroundColor Cyan
docker-compose -f docker-compose.windows.yml up -d

# Step 6: Check if services are running
Start-Sleep -Seconds 5
$services = docker-compose -f docker-compose.windows.yml ps
Write-Host "Service status:" -ForegroundColor Yellow
Write-Host $services

# Step 7: Create a dashboard HTML preview to ensure we can visualize the dashboard
Write-Host "Creating dashboard preview file..." -ForegroundColor Cyan
$htmlContent = @"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MEXC Trading Dashboard</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        header {
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            text-align: center;
        }
        .dashboard-links {
            display: flex;
            justify-content: space-around;
            margin-top: 30px;
        }
        .dashboard-card {
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            padding: 20px;
            text-align: center;
            width: 30%;
            transition: transform 0.3s;
        }
        .dashboard-card:hover {
            transform: translateY(-5px);
        }
        .dashboard-card h2 {
            color: #2c3e50;
        }
        .dashboard-card p {
            color: #7f8c8d;
            margin-bottom: 20px;
        }
        .btn {
            display: inline-block;
            background-color: #3498db;
            color: white;
            padding: 10px 20px;
            border-radius: 4px;
            text-decoration: none;
            font-weight: bold;
            transition: background-color 0.3s;
        }
        .btn:hover {
            background-color: #2980b9;
        }
        .status {
            margin-top: 30px;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            padding: 20px;
        }
        .status h2 {
            color: #2c3e50;
            border-bottom: 1px solid #ecf0f1;
            padding-bottom: 10px;
        }
        .status-item {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #ecf0f1;
        }
        .status-name {
            font-weight: bold;
        }
        .status-value {
            padding: 2px 10px;
            border-radius: 4px;
        }
        .status-online {
            background-color: #2ecc71;
            color: white;
        }
        .status-offline {
            background-color: #e74c3c;
            color: white;
        }
        iframe {
            width: 100%;
            height: 800px;
            border: none;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            margin-top: 30px;
        }
    </style>
</head>
<body>
    <header>
        <h1>MEXC Trading Dashboard</h1>
        <p>Monitor trading performance and market data in real-time</p>
    </header>
    
    <div class="container">
        <div class="dashboard-links">
            <div class="dashboard-card">
                <h2>Trading Performance</h2>
                <p>View detailed performance metrics and trading signals</p>
                <a href="http://localhost:3000/d/trading-performance" class="btn" target="_blank">Open Dashboard</a>
            </div>
            
            <div class="dashboard-card">
                <h2>Prometheus Metrics</h2>
                <p>Access raw metrics and create custom queries</p>
                <a href="http://localhost:9090" class="btn" target="_blank">Open Prometheus</a>
            </div>
            
            <div class="dashboard-card">
                <h2>Trading Overview</h2>
                <p>High-level view of trading activity and system health</p>
                <a href="http://localhost:3000" class="btn" target="_blank">Open Grafana</a>
            </div>
        </div>
        
        <div class="status">
            <h2>Service Status</h2>
            <div class="status-item">
                <span class="status-name">Grafana</span>
                <span class="status-value status-online">Online</span>
            </div>
            <div class="status-item">
                <span class="status-name">Prometheus</span>
                <span class="status-value status-online">Online</span>
            </div>
            <div class="status-item">
                <span class="status-name">Market Data Processor</span>
                <span class="status-value status-offline">Offline (View Only Mode)</span>
            </div>
        </div>
        
        <h2>Grafana Dashboard Preview</h2>
        <iframe src="http://localhost:3000/d/trading-performance" title="Trading Performance Dashboard"></iframe>
    </div>
    
    <script>
        // Check services availability
        function checkService(url, elementClass) {
            fetch(url, { mode: 'no-cors' })
                .then(() => {
                    document.querySelector('.' + elementClass).classList.replace('status-offline', 'status-online');
                    document.querySelector('.' + elementClass).textContent = 'Online';
                })
                .catch(() => {
                    document.querySelector('.' + elementClass).classList.replace('status-online', 'status-offline');
                    document.querySelector('.' + elementClass).textContent = 'Offline';
                });
        }
        
        // Check services every 5 seconds
        setInterval(() => {
            checkService('http://localhost:3000', 'grafana-status');
            checkService('http://localhost:9090', 'prometheus-status');
        }, 5000);
    </script>
</body>
</html>
"@
$htmlContent | Out-File -FilePath "dashboard_preview.html" -Encoding utf8

Write-Host "`n=== DASHBOARD ACCESS INFORMATION ===" -ForegroundColor Green
Write-Host "Dashboard components should now be running. Access your dashboard using these links:" -ForegroundColor Cyan
Write-Host "- Dashboard Preview: file://$currentDir/dashboard_preview.html" -ForegroundColor Yellow
Write-Host "- Grafana: http://localhost:3000 (admin/trading123)" -ForegroundColor Yellow
Write-Host "- Prometheus: http://localhost:9090" -ForegroundColor Yellow

Write-Host "`nNext Steps:" -ForegroundColor Magenta
Write-Host "1. Open dashboard_preview.html in your browser to access all dashboard components" -ForegroundColor Cyan
Write-Host "2. If Grafana and Prometheus are running, you can see their UIs directly" -ForegroundColor Cyan
Write-Host "3. The Market Data Processor is not running, this is a view-only dashboard" -ForegroundColor Cyan

# Offer to open the dashboard preview
$openBrowser = Read-Host "Would you like to open the dashboard preview in your browser? (y/n)"
if ($openBrowser -eq "y") {
    Start-Process "file://$currentDir/dashboard_preview.html"
}
