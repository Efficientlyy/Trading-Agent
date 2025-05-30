use config::{Config as ConfigCrate, ConfigError, Environment, File};
use serde::Deserialize;
use std::collections::HashMap;
use std::env;
use std::path::Path;
use tracing::{debug, info, warn};

/// Enhanced configuration with flexible data source settings
#[derive(Debug, Deserialize, Clone)]
pub struct EnhancedConfig {
    // Server configuration
    pub grpc_server_addr: String,
    pub rest_server_addr: String,
    
    // MEXC API configuration
    pub mexc_ws_url: String,
    pub mexc_rest_url: String,
    pub mexc_api_key: String,
    pub mexc_api_secret: String,
    
    // Data source configuration
    pub use_websocket: bool,
    pub use_rest_fallback: bool,
    pub rest_polling_interval_ms: u64,
    
    // WebSocket connection settings
    pub ws_reconnect_backoff_enabled: bool,
    pub ws_circuit_breaker_enabled: bool,
    pub ws_max_reconnect_attempts: u32,
    
    // Data consistency validation
    pub validation_enabled: bool,
    pub validation_interval_ms: u64,
    pub reconciliation_enabled: bool,
    
    // Trading configuration
    pub is_paper_trading: bool,
    pub trading_pairs: Vec<String>,
    pub default_order_size: f64,
    pub max_position_size: f64,
    
    // Logging configuration
    pub log_level: String,
    pub enable_telemetry: bool,
    pub jaeger_endpoint: String,
    
    // Additional settings
    pub settings: HashMap<String, String>,
}

impl EnhancedConfig {
    /// Load configuration from files and environment variables
    pub fn load() -> Result<Self, ConfigError> {
        let mut s = ConfigCrate::default();
        
        // Set default values
        s.set_default("grpc_server_addr", "0.0.0.0:50051")?;
        s.set_default("rest_server_addr", "0.0.0.0:8080")?;
        s.set_default("mexc_ws_url", "wss://wbs.mexc.com/ws")?;
        s.set_default("mexc_rest_url", "https://api.mexc.com")?;
        s.set_default("mexc_api_key", "")?;
        s.set_default("mexc_api_secret", "")?;
        s.set_default("use_websocket", true)?;
        s.set_default("use_rest_fallback", true)?;
        s.set_default("rest_polling_interval_ms", 5000)?; // 5 seconds
        s.set_default("ws_reconnect_backoff_enabled", true)?;
        s.set_default("ws_circuit_breaker_enabled", true)?;
        s.set_default("ws_max_reconnect_attempts", 10)?;
        s.set_default("validation_enabled", true)?;
        s.set_default("validation_interval_ms", 30000)?; // 30 seconds
        s.set_default("reconciliation_enabled", true)?;
        s.set_default("is_paper_trading", true)?;
        s.set_default("trading_pairs", vec!["BTCUSDC"])?; // Default to BTCUSDC as requested
        s.set_default("default_order_size", 0.01)?;
        s.set_default("max_position_size", 1.0)?;
        s.set_default("log_level", "info")?;
        s.set_default("enable_telemetry", false)?;
        s.set_default("jaeger_endpoint", "http://localhost:14268/api/traces")?;
        
        // Load configuration in order:
        // 1. Default config file
        let default_config = Path::new("config/default.toml");
        if default_config.exists() {
            info!("Loading default configuration from config/default.toml");
            s.merge(File::with_name("config/default.toml"))?;
        } else {
            warn!("Default configuration file config/default.toml not found");
        }
        
        // 2. Environment-specific config (development, production, etc.)
        let env_name = env::var("RUN_ENV").unwrap_or_else(|_| "development".into());
        let env_config = format!("config/{}.toml", env_name);
        let env_config_path = Path::new(&env_config);
        
        if env_config_path.exists() {
            info!("Loading {} configuration from {}", env_name, env_config);
            s.merge(File::with_name(&env_config))?;
        } else {
            warn!("Environment configuration file {} not found", env_config);
        }
        
        // 3. Local config (git-ignored for personal settings)
        let local_config = Path::new("config/local.toml");
        if local_config.exists() {
            info!("Loading local configuration from config/local.toml");
            s.merge(File::with_name("config/local.toml"))?;
        } else {
            debug!("Local configuration file config/local.toml not found");
        }
        
        // 4. Environment variables with prefix
        info!("Loading configuration from environment variables with prefix APP__");
        s.merge(Environment::with_prefix("APP").separator("__"))?;
        
        // Build the config
        let config: EnhancedConfig = s.try_into()?;
        
        // Validate the configuration
        config.validate()?;
        
        Ok(config)
    }
    
    /// Validate the configuration values
    fn validate(&self) -> Result<(), ConfigError> {
        // Validate that API credentials are present if WebSocket is enabled
        if self.use_websocket && (self.mexc_api_key.is_empty() || self.mexc_api_secret.is_empty()) {
            warn!("WebSocket is enabled but API credentials are not set. Some endpoints may be blocked.");
        }
        
        // Validate that at least one data source is enabled
        if !self.use_websocket && !self.use_rest_fallback {
            return Err(ConfigError::Message("At least one data source (WebSocket or REST) must be enabled".into()));
        }
        
        // Validate trading pairs
        if self.trading_pairs.is_empty() {
            return Err(ConfigError::Message("At least one trading pair must be specified".into()));
        }
        
        // Log important configuration settings
        info!("gRPC server will listen on {}", self.grpc_server_addr);
        info!("REST server will listen on {}", self.rest_server_addr);
        info!("Using MEXC {} URL: {}", 
            if self.use_websocket { "WebSocket" } else { "REST" }, 
            if self.use_websocket { &self.mexc_ws_url } else { &self.mexc_rest_url }
        );
        info!("API key is {}", if self.mexc_api_key.is_empty() { "not set" } else { "set" });
        info!("REST fallback is {}", if self.use_rest_fallback { "enabled" } else { "disabled" });
        info!("Paper trading is {}", if self.is_paper_trading { "enabled" } else { "disabled" });
        info!("Trading pairs: {}", self.trading_pairs.join(", "));
        
        Ok(())
    }
    
    /// Get a configuration value from the additional settings
    pub fn get_setting(&self, key: &str) -> Option<&String> {
        self.settings.get(key)
    }
    
    /// Check if a configuration value is true
    pub fn is_enabled(&self, key: &str) -> bool {
        self.get_setting(key)
            .map(|v| v.to_lowercase() == "true" || v == "1")
            .unwrap_or(false)
    }
}
