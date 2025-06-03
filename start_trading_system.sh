#!/bin/bash
# Unified startup script for Trading-Agent system
# This script launches all components in the correct order

# Parse command line arguments
MOCK_MODE=false
for arg in "$@"
do
    if [ "$arg" == "--mock" ] || [ "$arg" == "-m" ]; then
        MOCK_MODE=true
    fi
done

echo "Starting Trading-Agent System..."
echo "================================"

# Create log directory if it doesn't exist
mkdir -p logs

# Function to check if a port is in use
port_in_use() {
  lsof -i:$1 >/dev/null 2>&1
  return $?
}

# Kill any existing processes on our ports
for port in 8080 5001 5002; do
  if port_in_use $port; then
    echo "Port $port is in use. Stopping existing process..."
    fuser -k $port/tcp >/dev/null 2>&1
  fi
done

# Start the core trading engine (BTC/USDC Trading System)
echo "Starting BTC/USDC Trading System..."
if [ "$MOCK_MODE" = true ]; then
  echo "Running in MOCK MODE - no real trades will be executed"
  python start_btc_usdc_trading.py --mock > logs/trading_engine.log 2>&1 &
else
  echo "Running in REAL MODE with API credentials"
  python start_btc_usdc_trading.py > logs/trading_engine.log 2>&1 &
fi
TRADING_ENGINE_PID=$!
echo "BTC/USDC Trading System started with PID: $TRADING_ENGINE_PID"
sleep 2

# Start the parameter management API
echo "Starting Parameter Management API..."
python parameter_management_api.py > logs/parameter_api.log 2>&1 &
PARAM_API_PID=$!
echo "Parameter Management API started with PID: $PARAM_API_PID"
sleep 2

# Start the visualization dashboard
echo "Starting Visualization Dashboard..."
python visualization/chart_component.py > logs/visualization.log 2>&1 &
VISUALIZATION_PID=$!
echo "Visualization Dashboard started with PID: $VISUALIZATION_PID"
sleep 2

# Start the monitoring dashboard
echo "Starting Monitoring Dashboard..."
python monitoring/monitoring_dashboard.py > logs/monitoring.log 2>&1 &
MONITORING_PID=$!
echo "Monitoring Dashboard started with PID: $MONITORING_PID"
sleep 2

echo ""
echo "All components started successfully!"
echo "================================"
echo "Access the system at:"
echo "- Trading Dashboard: http://localhost:8080/"
echo "- Parameter Management: http://localhost:8080/parameters"
echo "- Monitoring Dashboard: http://localhost:5001/"
echo "- Chart Visualization: http://localhost:5002/"
echo ""
echo "Log files are available in the logs directory"
echo ""
echo "To stop all components, press Ctrl+C or run: ./stop_trading_system.sh"

# Create a stop script
cat > stop_trading_system.sh << EOF
#!/bin/bash
echo "Stopping Trading-Agent System..."
kill $TRADING_ENGINE_PID $PARAM_API_PID $VISUALIZATION_PID $MONITORING_PID 2>/dev/null
echo "All components stopped."
EOF
chmod +x stop_trading_system.sh

# Try to open the dashboard in the default browser if available
if command -v xdg-open >/dev/null 2>&1; then
  xdg-open http://localhost:8080/ >/dev/null 2>&1 &
elif command -v open >/dev/null 2>&1; then
  open http://localhost:8080/ >/dev/null 2>&1 &
fi

# Wait for all processes to finish (or until the user presses Ctrl+C)
wait
