Hi there,

I'm reaching out regarding the MEXC Trading System project. We've just completed a significant update to the codebase with a focus on BTC/USDC integration. Here's what you need to know:

## Latest Updates

We've implemented:
- Real-time market data integration with MEXC API
- WebSocket client for live data streaming
- Paper trading functionality with account management
- Updated dashboard UI with real-time charts and order book
- Fixed Docker configuration for reliable deployment
- Comprehensive monitoring with Prometheus and Grafana

## Getting Started

1. Pull the latest changes from the repository:
```bash
git pull origin master
```

2. Check out the new ONBOARDING.md file I've created specifically to help you get up to speed quickly. It contains:
   - Complete setup instructions
   - System architecture overview
   - Development workflow guidance
   - Key component explanations
   - Troubleshooting tips

3. Follow the Docker setup instructions in the README.md to get the system running locally:
```bash
docker-compose up -d
```

## Key Focus Areas

Please pay special attention to:
1. The BTC/USDC integration in the Market Data Processor
2. The WebSocket implementation for real-time data
3. The paper trading functionality

## Testing the System

Once you have the system running:
1. Access the trading dashboard at http://localhost:8080
2. Check the Grafana monitoring at http://localhost:3000 (login: admin/trading123)
3. Verify that real-time BTC/USDC data is flowing through the system

If you encounter any issues or have questions, please don't hesitate to reach out. The ONBOARDING.md file should address most common questions, but I'm here to help if you need additional guidance.

Looking forward to your contributions to the project!

Best regards,
