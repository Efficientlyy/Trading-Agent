use serde::{Deserialize, Serialize};
use config::{Config as ConfigCrate, File, Environment};
use std::env;
use lazy_static::lazy_static;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Config {
    // Server configuration
    pub grpc_server_addr: String,
    pub rest_server_addr: String,
    
    // MEXC API configuration
    pub mexc_ws_url: String,
    pub mexc_rest_url: String,
    pub mexc_api_key: String,
    pub mexc_api_secret: String,
    
    // Trading configuration
    pub is_paper_trading: bool,
    pub trading_pairs: Vec<String>,
    pub default_order_size: f64,
    pub max_position_size: f64,
    
    // Logging configuration
    pub log_level: String,
    pub enable_telemetry: bool,
    pub jaeger_endpoint: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            grpc_server_addr: "0.0.0.0:50051".to_string(),
            rest_server_addr: "0.0.0.0:8080".to_string(),
            
            mexc_ws_url: "wss://wbs.mexc.com/ws".to_string(),
            mexc_rest_url: "https://api.mexc.com".to_string(),
            mexc_api_key: "".to_string(),
            mexc_api_secret: "".to_string(),
            
            is_paper_trading: true,
            trading_pairs: vec!["BTCUSDT".to_string(), "ETHUSDT".to_string()],
            default_order_size: 0.01,
            max_position_size: 1.0,
            
            log_level: "info".to_string(),
            enable_telemetry: false,
            jaeger_endpoint: "http://localhost:14268/api/traces".to_string(),
        }
    }
}

impl Config {
    pub fn load() -> Result<Self, Box<dyn std::error::Error>> {
        // Load .env file if present
        let _ = dotenv::dotenv();
        
        let config_path = env::var("CONFIG_PATH").unwrap_or_else(|_| "config".to_string());
        
        let builder = ConfigCrate::builder()
            // Start with default settings
            .add_source(File::with_name(&format!("{}/default", config_path)).required(false))
            // Add environment specific settings
            .add_source(File::with_name(&format!("{}/{}", config_path, env::var("RUN_MODE").unwrap_or_else(|_| "development".to_string()))).required(false))
            // Add local settings
            .add_source(File::with_name(&format!("{}/local", config_path)).required(false))
            // Add settings from environment variables prefixed with APP_
            .add_source(Environment::with_prefix("app").separator("__"));
        
        // Build and deserialize
        let config = builder.build()?;
        let config: Config = config.try_deserialize()?;
        
        Ok(config)
    }
    
    pub fn is_paper_trading(&self) -> bool {
        self.is_paper_trading
    }
}
