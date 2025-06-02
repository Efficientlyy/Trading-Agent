# Getting Started with Trading-Agent

This quick-start guide provides essential information for new developers to get up and running with the Trading-Agent system quickly.

## Prerequisites

- **Docker** (recommended): Docker Engine 20.10+ and Docker Compose 2.0+
- **Python**: Python 3.9+ with pip
- **Git**: For cloning the repository
- **MEXC Account**: For API credentials (optional for development with mock data)

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/Efficientlyy/Trading-Agent.git
cd Trading-Agent
```

### Step 2: Set Up Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit the .env file with your credentials
# For development, you can use mock data without real credentials
```

## Deployment Options

### Option 1: One-Click Docker Deployment (Recommended)

```bash
# Start all services
docker-compose up -d

# Access the system at http://localhost:5000
```

### Option 2: Single Command Script

```bash
# Install dependencies
pip install -r requirements.txt

# Start all services
./start_trading_system.sh

# Access the system at http://localhost:5000
```

## Key Features

- **Zero-fee trading** for BTC/USDC on MEXC
- **Advanced visualization** for BTC, ETH, and SOL
- **Comprehensive risk management** with configurable parameters
- **Pattern recognition** with deep learning integration
- **Paper trading** mode for testing strategies

## Next Steps

- Review the full [Developer Documentation](./DEVELOPER_DOCUMENTATION.md)
- Explore the [Parameter Configuration Guide](./CONFIGURABLE_PARAMETERS.md)
- Check out the [System Architecture](./SYSTEM_DOCUMENTATION_UPDATED.md)

## Need Help?

- Check the logs in the `logs/` directory
- Review the troubleshooting section in the developer documentation
- Contact the development team

---

Last Updated: June 2, 2025
