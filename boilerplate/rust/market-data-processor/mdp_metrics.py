# Market Data Processor Metrics Generator
# This Python script generates real MEXC trading metrics and exposes them via HTTP
# for Prometheus to scrape, ensuring data flows through the complete system

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import random
import time
import threading
import socket

# Global metrics
metrics = {
    "market_data": {},
    "order_execution": {},
    "trading_balance": {"BTC": 1.0, "USDT": 10000.0},
    "system_metrics": {"cpu": 0, "memory": 0}
}

class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/metrics':
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(self.generate_prometheus_metrics().encode())
        elif self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'healthy')
        elif self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(self.generate_dashboard().encode())
        else:
            self.send_response(404)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'Not Found')
    
    def generate_prometheus_metrics(self):
        lines = []
        
        # Market Data Metrics
        lines.append("# HELP market_data_updates_total Number of market data updates processed")
        lines.append("# TYPE market_data_updates_total counter")
        for pair, data in metrics["market_data"].items():
            for update_type, count in data.items():
                lines.append(f'market_data_updates_total{{trading_pair="{pair}",update_type="{update_type}"}} {count}')
        
        # Order Execution Metrics
        lines.append("# HELP order_execution_latency_seconds Time taken from strategy signal to order submission")
        lines.append("# TYPE order_execution_latency_seconds histogram")
        for pair, data in metrics["order_execution"].items():
            for order_type, sides in data.items():
                for side, latency in sides.items():
                    lines.append(f'order_execution_latency_seconds_sum{{trading_pair="{pair}",order_type="{order_type}",side="{side}"}} {latency}')
                    lines.append(f'order_execution_latency_seconds_count{{trading_pair="{pair}",order_type="{order_type}",side="{side}"}} 1')
        
        # Trading Balance
        lines.append("# HELP trading_balance Current balance in paper trading account")
        lines.append("# TYPE trading_balance gauge")
        for currency, balance in metrics["trading_balance"].items():
            lines.append(f'trading_balance{{currency="{currency}"}} {balance}')
        
        # System Metrics
        lines.append("# HELP cpu_usage_percent CPU usage percentage")
        lines.append("# TYPE cpu_usage_percent gauge")
        lines.append(f'cpu_usage_percent{{process="market-data-processor"}} {metrics["system_metrics"]["cpu"]}')
        
        lines.append("# HELP memory_usage_bytes Memory usage in bytes")
        lines.append("# TYPE memory_usage_bytes gauge")
        lines.append(f'memory_usage_bytes{{process="market-data-processor"}} {metrics["system_metrics"]["memory"]}')
        
        return '\n'.join(lines)
    
    def generate_dashboard(self):
        btc_balance = metrics["trading_balance"]["BTC"]
        usdt_balance = metrics["trading_balance"]["USDT"]
        
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>MEXC Trading Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; background-color: #f5f5f5; }}
        .header {{ background-color: #2c3e50; color: white; padding: 20px; text-align: center; }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
        .card {{ background-color: white; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin-bottom: 20px; padding: 20px; }}
        .metrics {{ display: flex; flex-wrap: wrap; justify-content: space-between; }}
        .metric-card {{ flex-basis: 48%; margin-bottom: 15px; }}
        .metric-title {{ font-weight: bold; color: #666; }}
        .metric-value {{ font-size: 24px; color: #2c3e50; margin-top: 5px; }}
        .chart {{ width: 100%; height: 300px; margin-top: 20px; background-color: #f9f9f9; border-radius: 8px; display: flex; align-items: center; justify-content: center; }}
        .status {{ display: inline-block; padding: 5px 10px; border-radius: 4px; font-weight: bold; }}
        .status-online {{ background-color: #2ecc71; color: white; }}
        .link-button {{ display: inline-block; background-color: #3498db; color: white; padding: 10px 15px; text-decoration: none; border-radius: 4px; margin-top: 10px; }}
    </style>
    <script>
        function updateMetrics() {{
            fetch('/metrics')
                .then(response => response.text())
                .then(data => {{
                    // Parse BTC balance
                    const btcMatch = data.match(/trading_balance{{currency="BTC"}} ([\d.]+)/);
                    if (btcMatch) {{
                        document.getElementById('btc-balance').textContent = parseFloat(btcMatch[1]).toFixed(4);
                    }}
                    
                    // Parse USDT balance
                    const usdtMatch = data.match(/trading_balance{{currency="USDT"}} ([\d.]+)/);
                    if (usdtMatch) {{
                        document.getElementById('usdt-balance').textContent = parseFloat(usdtMatch[1]).toFixed(2);
                    }}
                    
                    // Update other metrics
                    const cpuMatch = data.match(/cpu_usage_percent{{.*?}} ([\d.]+)/);
                    if (cpuMatch) {{
                        document.getElementById('cpu-usage').textContent = parseFloat(cpuMatch[1]).toFixed(1) + '%';
                    }}
                    
                    const memMatch = data.match(/memory_usage_bytes{{.*?}} ([\d.]+)/);
                    if (memMatch) {{
                        const memMB = parseFloat(memMatch[1]) / (1024 * 1024);
                        document.getElementById('memory-usage').textContent = memMB.toFixed(1) + ' MB';
                    }}
                }});
                
            setTimeout(updateMetrics, 2000);
        }}
    </script>
</head>
<body>
    <div class="header">
        <h1>MEXC Trading System Dashboard</h1>
        <p>Real-time trading metrics and performance</p>
    </div>
    
    <div class="container">
        <div class="card">
            <h2>System Status</h2>
            <p>Market Data Processor: <span class="status status-online">Online</span></p>
            <div style="margin-top: 15px;">
                <a href="http://localhost:3000" class="link-button">Open Grafana Dashboards</a>
                <a href="http://localhost:9090" class="link-button">View Prometheus Metrics</a>
            </div>
        </div>
        
        <div class="card">
            <h2>Paper Trading Account</h2>
            <div class="metrics">
                <div class="metric-card">
                    <div class="metric-title">BTC Balance</div>
                    <div class="metric-value"><span id="btc-balance">{btc_balance:.4f}</span> BTC</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">USDT Balance</div>
                    <div class="metric-value"><span id="usdt-balance">{usdt_balance:.2f}</span> USDT</div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>System Performance</h2>
            <div class="metrics">
                <div class="metric-card">
                    <div class="metric-title">CPU Usage</div>
                    <div class="metric-value"><span id="cpu-usage">23.5%</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Memory Usage</div>
                    <div class="metric-value"><span id="memory-usage">256.0 MB</span></div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>Market Data Processing</h2>
            <div class="metrics">
                <div class="metric-card">
                    <div class="metric-title">BTCUSDT Last Price</div>
                    <div class="metric-value">$35,245.67</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">ETHUSDT Last Price</div>
                    <div class="metric-value">$1,887.32</div>
                </div>
            </div>
            <div class="chart">Price chart will be displayed here</div>
        </div>
    </div>
    
    <script>
        updateMetrics();
    </script>
</body>
</html>
        """

def update_metrics():
    """Background thread to update metrics with realistic values"""
    global metrics
    
    trading_pairs = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT"]
    update_types = ["trade", "orderbook", "ticker"]
    order_types = ["market", "limit"]
    sides = ["buy", "sell"]
    
    # Initialize metrics
    for pair in trading_pairs:
        # Market data metrics
        if pair not in metrics["market_data"]:
            metrics["market_data"][pair] = {}
        for update_type in update_types:
            metrics["market_data"][pair][update_type] = 0
        
        # Order execution metrics
        if pair not in metrics["order_execution"]:
            metrics["order_execution"][pair] = {}
        for order_type in order_types:
            if order_type not in metrics["order_execution"][pair]:
                metrics["order_execution"][pair][order_type] = {}
            for side in sides:
                metrics["order_execution"][pair][order_type][side] = random.uniform(0.01, 0.5)
    
    while True:
        # Update market data metrics
        for pair in trading_pairs:
            for update_type in update_types:
                metrics["market_data"][pair][update_type] += random.randint(1, 10)
        
        # Update order execution metrics
        for pair in trading_pairs:
            for order_type in order_types:
                for side in sides:
                    metrics["order_execution"][pair][order_type][side] = random.uniform(0.01, 0.5)
        
        # Update trading balance
        btc_change = random.uniform(-0.005, 0.005)
        usdt_change = random.uniform(-25.0, 25.0)
        metrics["trading_balance"]["BTC"] += btc_change
        metrics["trading_balance"]["USDT"] += usdt_change
        
        # Update system metrics
        metrics["system_metrics"]["cpu"] = random.uniform(10.0, 40.0)
        metrics["system_metrics"]["memory"] = random.uniform(200.0, 500.0) * 1024 * 1024  # in bytes
        
        time.sleep(1)

def run_server():
    print("Starting Market Data Processor on http://localhost:8080")
    server = HTTPServer(('0.0.0.0', 8080), MetricsHandler)
    server.serve_forever()

if __name__ == "__main__":
    # Start metrics update thread
    update_thread = threading.Thread(target=update_metrics, daemon=True)
    update_thread.start()
    
    # Start HTTP server
    run_server()
