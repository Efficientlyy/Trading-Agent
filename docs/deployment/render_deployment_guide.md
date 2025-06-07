# Render Deployment Guide for Trading-Agent System

This guide documents the deployment process for the Trading-Agent system on Render, including troubleshooting steps and correct configuration for each service.

## System Architecture

The Trading-Agent system consists of three main services:

1. **LLM Overseer (Telegram Bot)**: A Flask-based web service that provides a Telegram bot interface for API key management
2. **Dashboard**: A Flask-based web interface for monitoring trading activities and market data
3. **Trading Engine**: A background worker that executes trading strategies

## Deployment Configuration

### Environment Variables

The following environment variables must be set for the LLM overseer service:

- `TELEGRAM_BOT_TOKEN`: The Telegram bot token (e.g., 7707340014:AAFzwftqwWijHaDQ50vG8RYLz63BoLGV7qk)
- `TELEGRAM_ALLOWED_USERS`: Comma-separated list of allowed Telegram user IDs (e.g., 1888718908)

### Service Configuration

#### LLM Overseer (Telegram Bot)

- **Build Command**: `pip install cryptography python-telegram-bot && pip install -r requirements.txt`
- **Start Command**: `pip install gunicorn && gunicorn --bind 0.0.0.0:$PORT llm_overseer.telegram.run_llm_powered_bot:app`
- **Type**: Web Service

#### Dashboard

- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `pip install gunicorn && gunicorn --bind 0.0.0.0:$PORT mexc_dashboard_production:app`
- **Type**: Web Service

#### Trading Engine

- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `python flash_trading.py --env production`
- **Type**: Background Worker

## Common Issues and Solutions

### Port Binding Issues

**Issue**: Services fail with "Port scan timeout reached, failed to detect open port 10000 from PORT environment variable"

**Solution**:
- Ensure web services bind to the PORT environment variable provided by Render
- Use `--bind 0.0.0.0:$PORT` with Gunicorn to listen on all interfaces
- Do not hardcode port numbers in the start command

### Development Server in Production

**Issue**: Using Flask's built-in development server in production

**Solution**:
- Use Gunicorn as a production WSGI server
- Install Gunicorn in the start command: `pip install gunicorn && gunicorn...`
- Specify the correct app module path for each service

### Module Path Errors

**Issue**: "ModuleNotFoundError: No module named 'src'" or similar

**Solution**:
- Use the correct module path that matches your repository structure
- For the LLM overseer: `llm_overseer.telegram.run_llm_powered_bot:app`
- For the dashboard: `mexc_dashboard_production:app`

### Service Type Confusion

**Issue**: Trying to run background workers as web services

**Solution**:
- Use appropriate start commands for each service type
- Web services should use Gunicorn and bind to a port
- Background workers should use direct Python execution

## Verification Steps

After deployment, verify each service:

1. **LLM Overseer**: Test the Telegram bot by sending the "/login" command to @JVLS123
2. **Dashboard**: Access the dashboard at https://trading-agent-dashboard.onrender.com
3. **Trading Engine**: Check the logs for successful initialization and trading activity

## Troubleshooting

If services fail to deploy:

1. Check the logs for specific error messages
2. Verify that the correct start commands are being used
3. Ensure all required environment variables are set
4. Confirm that web services are binding to the correct port
5. Verify that the module paths match your repository structure

## Maintenance

To update the deployed services:

1. Push changes to the GitHub repository
2. Trigger a manual redeploy in the Render dashboard
3. Monitor the logs for any deployment issues
4. Verify that the services are functioning correctly after deployment
