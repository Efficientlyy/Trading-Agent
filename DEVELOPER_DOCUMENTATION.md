# Trading-Agent Developer Documentation

## Overview

This document provides comprehensive instructions for new developers to deploy, launch, and work with the Trading-Agent system. The Trading-Agent is a sophisticated algorithmic trading platform with support for BTC, ETH, and SOL trading, advanced visualization, risk management, and configurable parameters.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Quick Start](#quick-start)
3. [Detailed Deployment Options](#detailed-deployment-options)
4. [Project Structure](#project-structure)
5. [Configuration](#configuration)
6. [Development Workflow](#development-workflow)
7. [Testing](#testing)
8. [Troubleshooting](#troubleshooting)

## System Requirements

- **Operating System**: Linux (Ubuntu 20.04+ recommended), macOS, or Windows with WSL2
- **Docker**: Docker Engine 20.10+ and Docker Compose 2.0+ (for containerized deployment)
- **Python**: Python 3.9+ (for local development)
- **Memory**: Minimum 4GB RAM, 8GB recommended
- **Storage**: Minimum 10GB free space
- **Network**: Stable internet connection for market data

## Quick Start

### Option 1: Docker Deployment (Recommended for Production)

1. Clone the repository:
   ```bash
   git clone https://github.com/Efficientlyy/Trading-Agent.git
   cd Trading-Agent
   ```

2. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env file with your MEXC API credentials
   ```

3. Deploy with Docker:
   ```bash
   docker-compose up -d
   ```

4. Access the system:
   - Trading Dashboard: http://localhost:5000/
   - Parameter Management: http://localhost:5000/parameters
   - Monitoring Dashboard: http://localhost:5001/
   - Chart Visualization: http://localhost:5002/

### Option 2: Single Command Script (Development)

1. Clone the repository:
   ```bash
   git clone https://github.com/Efficientlyy/Trading-Agent.git
   cd Trading-Agent
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env file with your MEXC API credentials
   ```

4. Launch the system:
   ```bash
   ./start_trading_system.sh
   ```

5. Access the system at the same URLs as in the Docker deployment.

## Detailed Deployment Options

### Docker Deployment

The Docker deployment uses three containers:
- `trading-engine`: Core trading logic, parameter management, and API
- `visualization`: Chart visualization for BTC, ETH, and SOL
- `monitoring`: System monitoring and risk management dashboard

#### Customizing Docker Deployment

To modify the Docker deployment:

1. Edit `docker-compose.yml` to change ports, volumes, or environment variables
2. Edit Dockerfiles in the project root to modify container builds
3. Rebuild and restart:
   ```bash
   docker-compose down
   docker-compose build
   docker-compose up -d
   ```

#### Docker Logs and Debugging

```bash
# View logs from all containers
docker-compose logs

# View logs from a specific container
docker-compose logs trading-engine

# Follow logs in real-time
docker-compose logs -f
```

### Local Development Deployment

The local deployment uses the `start_trading_system.sh` script to launch all components:

#### Components Started

1. Core Trading Engine (`trading_engine.py`)
2. Parameter Management API (`parameter_management_api.py`)
3. Visualization Dashboard (`visualization/chart_component.py`)
4. Monitoring Dashboard (`monitoring/monitoring_dashboard.py`)

#### Logs and Debugging

All logs are stored in the `logs/` directory:
- `logs/trading_engine.log`
- `logs/parameter_api.log`
- `logs/visualization.log`
- `logs/monitoring.log`

To stop all components:
```bash
./stop_trading_system.sh
```

## Project Structure

```
Trading-Agent/
├── .env                      # Environment variables
├── docker-compose.yml        # Docker configuration
├── requirements.txt          # Python dependencies
├── start_trading_system.sh   # Unified startup script
├── trading_engine.py         # Core trading engine
├── parameter_management_api.py # Parameter management API
├── execution_optimization.py # Order execution logic
├── mock_exchange_client.py   # Mock exchange for testing
├── optimized_mexc_client.py  # MEXC exchange client
├── enhanced_dl_integration_fixed.py # Deep learning integration
├── enhanced_dl_model_fixed.py # Deep learning model
├── enhanced_feature_adapter_fixed.py # Feature adapter
├── enhanced_flash_trading_signals.py # Trading signals
├── rl_agent_fixed_v4.py      # Reinforcement learning agent
├── risk_management/          # Risk management components
├── visualization/            # Visualization components
├── monitoring/               # Monitoring components
├── error_handling/           # Error handling components
├── performance/              # Performance optimization
├── config/                   # Configuration files
│   └── presets/              # Parameter presets
├── data/                     # Data storage
└── logs/                     # Log files
```

## Configuration

### Environment Variables

Key environment variables in `.env`:

```
MEXC_API_KEY=your_api_key
MEXC_API_SECRET=your_api_secret
LOG_LEVEL=INFO
ENABLE_PAPER_TRADING=true
MAX_PORTFOLIO_RISK_PERCENT=1.5
```

### Parameter Management

The system includes a comprehensive parameter management system with:

1. **Web Interface**: Access at http://localhost:5000/parameters
2. **API Endpoints**:
   - GET `/parameters` - List all parameters
   - GET `/parameters/{module}` - Get parameters for a specific module
   - POST `/parameters/{module}` - Update parameters for a module
   - GET `/parameters/presets` - List available presets
   - POST `/parameters/presets/apply` - Apply a preset
   - POST `/parameters/presets/save` - Save a custom preset

3. **Parameter Categories**:
   - Basic: Essential parameters for typical operation
   - Advanced: Fine-tuning parameters for experienced users
   - Expert: Detailed parameters for system optimization

## Development Workflow

### Setting Up Development Environment

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

3. Set up pre-commit hooks:
   ```bash
   pre-commit install
   ```

### Making Changes

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and test thoroughly
3. Run the validation tests:
   ```bash
   python validate_parameter_system.py
   python integration_test.py
   ```

4. Commit and push your changes:
   ```bash
   git add .
   git commit -m "Add your feature description"
   git push origin feature/your-feature-name
   ```

5. Create a pull request on GitHub

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest test_enhanced_signals_mock.py

# Run with coverage
pytest --cov=.
```

### Test Environment

The system includes a mock exchange client for testing without real API credentials:

```python
from mock_exchange_client import MockExchangeClient
client = MockExchangeClient()
```

## Troubleshooting

### Common Issues

1. **API Connection Errors**:
   - Verify MEXC API credentials in `.env`
   - Check internet connection
   - Ensure MEXC API is operational

2. **Component Startup Failures**:
   - Check logs in `logs/` directory
   - Verify all dependencies are installed
   - Check for port conflicts

3. **Parameter Validation Errors**:
   - Review parameter values against allowed ranges
   - Check parameter metadata in `parameter_management_api.py`

### Getting Help

- Review the logs in the `logs/` directory
- Check the GitHub repository issues
- Contact the development team

---

## Additional Resources

- [MEXC API Documentation](https://mxcdevelop.github.io/apidocs/spot_v3_en/)
- [Trading Strategy Documentation](./TRADING_STRATEGIES.md)
- [Risk Management Documentation](./RISK_MANAGEMENT.md)
- [Parameter Configuration Guide](./CONFIGURABLE_PARAMETERS.md)

---

Last Updated: June 2, 2025
