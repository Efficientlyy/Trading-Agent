import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
def load_env():
    """Load environment variables from .env file or .env-secure/.env if specified"""
    env_path = None
    
    # Check if --env flag is provided
    if "--env" in sys.argv:
        try:
            env_index = sys.argv.index("--env")
            if env_index + 1 < len(sys.argv):
                env_path = sys.argv[env_index + 1]
                print(f"Loading environment from: {env_path}")
        except ValueError:
            pass
    
    # Load from specified path or default
    if env_path:
        load_dotenv(env_path)
    else:
        # Try .env-secure/.env first, then fall back to .env
        if os.path.exists(".env-secure/.env"):
            load_dotenv(".env-secure/.env")
        else:
            load_dotenv()
    
    # Verify critical environment variables
    critical_vars = [
        'MEXC_API_KEY', 
        'MEXC_API_SECRET', 
        'OPENROUTER_API_KEY',
        'TELEGRAM_BOT_TOKEN',
        'TELEGRAM_CHAT_ID'
    ]
    
    missing = []
    for var in critical_vars:
        if not os.getenv(var):
            missing.append(var)
    
    if missing:
        print(f"WARNING: Missing critical environment variables: {', '.join(missing)}")
        return False
    
    return True

# Check if running in production mode
def is_production():
    """Check if running in production mode"""
    return os.getenv('ENVIRONMENT', '').lower() == 'production'

# Check if paper trading is enabled
def is_paper_trading():
    """Check if paper trading is enabled"""
    paper_trading = os.getenv('PAPER_TRADING', 'True')
    return paper_trading.lower() in ('true', '1', 't', 'yes')

# Get log level from environment
def get_log_level():
    """Get log level from environment"""
    return os.getenv('LOG_LEVEL', 'INFO')

# Get port for web services
def get_port():
    """Get port for web services"""
    return int(os.getenv('PORT', 10000))

# Get host for web services
def get_host():
    """Get host for web services"""
    return os.getenv('HOST', '0.0.0.0')

# Get trading pairs
def get_trading_pairs():
    """Get trading pairs from environment"""
    pairs = os.getenv('TRADING_PAIRS', 'BTCUSDC,ETHUSDC')
    return pairs.split(',')

# Get base order sizes
def get_base_order_size(symbol):
    """Get base order size for a symbol"""
    if symbol.startswith('BTC'):
        return float(os.getenv('BASE_ORDER_SIZE_BTC', 0.001))
    elif symbol.startswith('ETH'):
        return float(os.getenv('BASE_ORDER_SIZE_ETH', 0.01))
    else:
        return 0.001  # Default

# Get max open orders
def get_max_open_orders():
    """Get maximum number of open orders"""
    return int(os.getenv('MAX_OPEN_ORDERS', 5))

# Check if metrics are enabled
def is_metrics_enabled():
    """Check if metrics are enabled"""
    metrics = os.getenv('ENABLE_METRICS', 'True')
    return metrics.lower() in ('true', '1', 't', 'yes')

# Get metrics port
def get_metrics_port():
    """Get port for metrics server"""
    return int(os.getenv('METRICS_PORT', 9090))

# Get state persistence settings
def is_state_persistence_enabled():
    """Check if state persistence is enabled"""
    persist = os.getenv('PERSIST_STATE', 'True')
    return persist.lower() in ('true', '1', 't', 'yes')

def get_state_file_path():
    """Get path for state file"""
    return os.getenv('STATE_FILE_PATH', '/tmp/trading_state.json')

# Print environment summary (for debugging)
def print_env_summary():
    """Print summary of environment configuration"""
    print("\n=== Environment Configuration ===")
    print(f"Environment: {os.getenv('ENVIRONMENT', 'development')}")
    print(f"Paper Trading: {is_paper_trading()}")
    print(f"Log Level: {get_log_level()}")
    print(f"Web Port: {get_port()}")
    print(f"Trading Pairs: {get_trading_pairs()}")
    print(f"Metrics Enabled: {is_metrics_enabled()}")
    print(f"State Persistence: {is_state_persistence_enabled()}")
    print("================================\n")

if __name__ == "__main__":
    # Test environment loading
    if load_env():
        print_env_summary()
    else:
        print("Failed to load environment variables")
