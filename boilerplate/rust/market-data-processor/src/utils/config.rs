use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Read;
use std::env;

/// Configuration for the application
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// MEXC API key
    pub api_key: Option<String>,
    
    /// MEXC API secret
    pub api_secret: Option<String>,
    
    /// Default trading pair
    pub default_pair: String,
    
    /// WebSocket base URL
    pub websocket_url: String,
    
    /// REST API base URL
    pub rest_api_url: String,
    
    /// WebSocket reconnect interval in seconds
    pub websocket_reconnect_interval: u64,
    
    /// Initial paper trading USDC balance
    pub initial_balance_usdc: f64,
    
    /// Initial paper trading BTC balance
    pub initial_balance_btc: f64,
    
    /// Order book depth (number of levels)
    pub orderbook_depth: usize,
    
    /// Enable paper trading
    pub paper_trading_enabled: bool,
    
    /// Enable metrics collection
    pub metrics_enabled: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            api_key: None,
            api_secret: None,
            default_pair: "BTCUSDC".to_string(),
            websocket_url: "wss://wbs.mexc.com/ws".to_string(),
            rest_api_url: "https://api.mexc.com".to_string(),
            websocket_reconnect_interval: 5,
            initial_balance_usdc: 10000.0,
            initial_balance_btc: 0.0,
            orderbook_depth: 20,
            paper_trading_enabled: true,
            metrics_enabled: true,
        }
    }
}

impl Config {
    /// Load configuration from file
    pub fn load(path: Option<String>) -> Self {
        let path = path.unwrap_or_else(|| "config.json".to_string());
        
        match File::open(&path) {
            Ok(mut file) => {
                let mut contents = String::new();
                if file.read_to_string(&mut contents).is_ok() {
                    match serde_json::from_str(&contents) {
                        Ok(config) => {
                            log::info!("Loaded configuration from {}", path);
                            config
                        }
                        Err(e) => {
                            log::error!("Failed to parse configuration file: {}", e);
                            Self::from_env()
                        }
                    }
                } else {
                    log::error!("Failed to read configuration file");
                    Self::from_env()
                }
            }
            Err(e) => {
                log::error!("Failed to open configuration file: {}", e);
                Self::from_env()
            }
        }
    }
    
    /// Create configuration from environment variables
    pub fn from_env() -> Self {
        let mut config = Self::default();
        
        if let Ok(api_key) = env::var("MEXC_API_KEY") {
            config.api_key = Some(api_key);
        }
        
        if let Ok(api_secret) = env::var("MEXC_API_SECRET") {
            config.api_secret = Some(api_secret);
        }
        
        if let Ok(default_pair) = env::var("MEXC_DEFAULT_PAIR") {
            config.default_pair = default_pair;
        }
        
        if let Ok(websocket_url) = env::var("MEXC_WEBSOCKET_URL") {
            config.websocket_url = websocket_url;
        }
        
        if let Ok(rest_api_url) = env::var("MEXC_REST_API_URL") {
            config.rest_api_url = rest_api_url;
        }
        
        if let Ok(interval) = env::var("MEXC_WEBSOCKET_RECONNECT_INTERVAL") {
            if let Ok(interval) = interval.parse() {
                config.websocket_reconnect_interval = interval;
            }
        }
        
        if let Ok(balance) = env::var("PAPER_TRADING_INITIAL_USDC") {
            if let Ok(balance) = balance.parse() {
                config.initial_balance_usdc = balance;
            }
        }
        
        if let Ok(balance) = env::var("PAPER_TRADING_INITIAL_BTC") {
            if let Ok(balance) = balance.parse() {
                config.initial_balance_btc = balance;
            }
        }
        
        if let Ok(depth) = env::var("ORDERBOOK_DEPTH") {
            if let Ok(depth) = depth.parse() {
                config.orderbook_depth = depth;
            }
        }
        
        if let Ok(enabled) = env::var("PAPER_TRADING_ENABLED") {
            config.paper_trading_enabled = enabled.to_lowercase() == "true";
        }
        
        if let Ok(enabled) = env::var("METRICS_ENABLED") {
            config.metrics_enabled = enabled.to_lowercase() == "true";
        }
        
        config
    }
    
    /// Save configuration to file
    pub fn save(&self, path: Option<String>) -> Result<(), Box<dyn std::error::Error>> {
        let path = path.unwrap_or_else(|| "config.json".to_string());
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    use std::io::Write;
    
    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.default_pair, "BTCUSDC");
        assert_eq!(config.initial_balance_usdc, 10000.0);
    }
    
    #[test]
    fn test_config_from_env() {
        env::set_var("MEXC_DEFAULT_PAIR", "ETHUSDC");
        env::set_var("PAPER_TRADING_INITIAL_USDC", "5000.0");
        
        let config = Config::from_env();
        assert_eq!(config.default_pair, "ETHUSDC");
        assert_eq!(config.initial_balance_usdc, 5000.0);
        
        // Clean up
        env::remove_var("MEXC_DEFAULT_PAIR");
        env::remove_var("PAPER_TRADING_INITIAL_USDC");
    }
    
    #[test]
    fn test_save_and_load_config() {
        let mut config = Config::default();
        config.default_pair = "ETHUSDC".to_string();
        config.initial_balance_usdc = 5000.0;
        
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path().to_str().unwrap().to_string();
        
        config.save(Some(path.clone())).unwrap();
        let loaded_config = Config::load(Some(path));
        
        assert_eq!(loaded_config.default_pair, "ETHUSDC");
        assert_eq!(loaded_config.initial_balance_usdc, 5000.0);
    }
}
