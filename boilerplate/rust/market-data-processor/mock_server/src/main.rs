use prometheus::{Encoder, TextEncoder};
use tiny_http::{Method, Response, Server, StatusCode};
use std::thread;

mod metrics;

fn main() {
    println!("Starting Market Data Processor (Metrics Simulation Mode)");
    
    // Start metrics simulation
    metrics::start_metrics_simulation();
    
    // Start HTTP server
    let server = Server::http("0.0.0.0:8080").unwrap();
    
    for request in server.incoming_requests() {
        match (request.method(), request.url()) {
            (Method::Get, "/metrics") => {
                // Handle Prometheus metrics endpoint
                let encoder = TextEncoder::new();
                let metric_families = prometheus::gather();
                let mut buffer = vec![];
                encoder.encode(&metric_families, &mut buffer).unwrap();
                
                let response = Response::from_data(buffer)
                    .with_header("Content-Type: text/plain".parse().unwrap());
                request.respond(response).unwrap();
            },
            (Method::Get, "/health") => {
                // Simple health check
                let response = Response::from_string("healthy")
                    .with_status_code(StatusCode(200));
                request.respond(response).unwrap();
            },
            (Method::Get, "/") => {
                // Main dashboard page
                let html = r#"<!DOCTYPE html>
<html>
<head>
    <title>MEXC Trading Dashboard</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
        }
        .header {
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            text-align: center;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        .card {
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            padding: 20px;
        }
        .dashboard-link {
            display: block;
            background-color: #3498db;
            color: white;
            text-align: center;
            padding: 15px;
            text-decoration: none;
            font-weight: bold;
            border-radius: 5px;
            margin: 10px 0;
        }
        .dashboard-link:hover {
            background-color: #2980b9;
        }
        .metrics {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
        }
        .metric-card {
            flex-basis: 48%;
            margin-bottom: 15px;
        }
        .metric-title {
            font-weight: bold;
            margin-bottom: 5px;
        }
        .metric-value {
            font-size: 24px;
            color: #2c3e50;
        }
        .status {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 3px;
            font-weight: bold;
        }
        .status-ok {
            background-color: #2ecc71;
            color: white;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>MEXC Trading Dashboard</h1>
        <p>Real-time trading system monitoring</p>
    </div>
    
    <div class="container">
        <div class="card">
            <h2>System Status</h2>
            <p>All services are <span class="status status-ok">OPERATIONAL</span></p>
            
            <h3>Quick Links</h3>
            <a href="http://localhost:3000" class="dashboard-link">Grafana Dashboards</a>
            <a href="http://localhost:9090" class="dashboard-link">Prometheus Metrics</a>
        </div>
        
        <div class="card">
            <h2>Current Trading Performance</h2>
            <div class="metrics">
                <div class="metric-card">
                    <div class="metric-title">BTC Balance</div>
                    <div class="metric-value">1.02 BTC</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">USDT Balance</div>
                    <div class="metric-value">10,234.56 USDT</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">24h Trading Volume</div>
                    <div class="metric-value">5.34 BTC</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Open Orders</div>
                    <div class="metric-value">3</div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>System Health</h2>
            <div class="metrics">
                <div class="metric-card">
                    <div class="metric-title">CPU Usage</div>
                    <div class="metric-value">23%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Memory Usage</div>
                    <div class="metric-value">286 MB</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Avg. Order Execution</div>
                    <div class="metric-value">126 ms</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Market Data Updates</div>
                    <div class="metric-value">1,245/min</div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Update metrics periodically
        setInterval(() => {
            // This would normally fetch real-time data
            document.querySelector('.metrics').innerHTML = 
                <div class="metric-card">
                    <div class="metric-title">BTC Balance</div>
                    <div class="metric-value"> BTC</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">USDT Balance</div>
                    <div class="metric-value"> USDT</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">24h Trading Volume</div>
                    <div class="metric-value"> BTC</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Open Orders</div>
                    <div class="metric-value"></div>
                </div>
            ;
        }, 5000);
    </script>
</body>
</html>"#;
                
                let response = Response::from_string(html)
                    .with_header("Content-Type: text/html".parse().unwrap());
                request.respond(response).unwrap();
            },
            _ => {
                // 404 for anything else
                let response = Response::from_string("Not Found")
                    .with_status_code(StatusCode(404));
                request.respond(response).unwrap();
            }
        }
    }
}
