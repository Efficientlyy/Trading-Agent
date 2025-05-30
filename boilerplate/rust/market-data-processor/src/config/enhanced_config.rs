use std::collections::HashMap;
use std::env;
use std::path::Path;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use crate::utils::config::Config;

/// Enhanced configuration for the Market Data Processor
/// This configuration supports paper trading and dashboard features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedConfig {
    // Server configuration
    pub grpc_server_addr: String,
    pub http_server_addr: String,
    
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
    
    // Paper trading configuration
    pub paper_trading_enabled: bool,
    pub paper_trading_initial_balances: HashMap<String, f64>,
    pub paper_trading_slippage_model: String,  // NONE, MINIMAL, REALISTIC, HIGH
    pub paper_trading_latency_model: String,   // NONE, LOW, NORMAL, HIGH
    pub paper_trading_fee_rate: f64,           // e.g., 0.001 = 0.1%
    
    // Trading configuration
    pub trading_pairs: Vec<String>,
    pub default_order_size: f64,
    pub max_position_size: f64,
    pub max_drawdown_percent: f64,
    
    // Dashboard configuration
    pub serve_dashboard: bool,
    pub dashboard_path: Option<String>,
    
    // Logging configuration
    pub log_level: String,
    pub enable_telemetry: bool,
}

impl EnhancedConfig {
    /// Create an enhanced configuration from the base config
    pub fn from_config(config: &Config) -> Self {
        // Initialize with default values
        let mut enhanced_config = EnhancedConfig {
            grpc_server_addr: config.grpc_server_addr.clone(),
            http_server_addr: "0.0.0.0:8080".to_string(),
            mexc_ws_url: "wss://wbs.mexc.com/ws".to_string(),
            mexc_rest_url: "https://api.mexc.com".to_string(),
            mexc_api_key: String::new(),
            mexc_api_secret: String::new(),
            use_websocket: true,
            use_rest_fallback: true,
            rest_polling_interval_ms: 5000,
            ws_reconnect_backoff_enabled: true,
            ws_circuit_breaker_enabled: true,
            ws_max_reconnect_attempts: 10,
            paper_trading_enabled: true,
            paper_trading_initial_balances: HashMap::new(),
            paper_trading_slippage_model: "REALISTIC".to_string(),
            paper_trading_latency_model: "NORMAL".to_string(),
            paper_trading_fee_rate: 0.001,
            trading_pairs: vec!["BTCUSDT".to_string(), "ETHUSDT".to_string()],
            default_order_size: 0.1,
            max_position_size: 1.0,
            max_drawdown_percent: 10.0,
            serve_dashboard: true,
            dashboard_path: None,
            log_level: "info".to_string(),
            enable_telemetry: false,
        };
        
        // Set initial balances (default values)
        enhanced_config.paper_trading_initial_balances.insert("USDT".to_string(), 10000.0);
        enhanced_config.paper_trading_initial_balances.insert("BTC".to_string(), 0.5);
        enhanced_config.paper_trading_initial_balances.insert("ETH".to_string(), 5.0);
        
        // Override with environment variables
        enhanced_config.load_from_env();
        
        enhanced_config
    }
    
    /// Load configuration from environment variables
    fn load_from_env(&mut self) {
        // Server configuration
        if let Ok(val) = env::var("HTTP_SERVER_ADDR") {
            self.http_server_addr = val;
        }
        
        // MEXC API configuration
        if let Ok(val) = env::var("MEXC_WS_URL") {
            self.mexc_ws_url = val;
        }
        if let Ok(val) = env::var("MEXC_REST_URL") {
            self.mexc_rest_url = val;
        }
        if let Ok(val) = env::var("MEXC_API_KEY") {
            self.mexc_api_key = val;
        }
        if let Ok(val) = env::var("MEXC_API_SECRET") {
            self.mexc_api_secret = val;
        }
        
        // Data source configuration
        if let Ok(val) = env::var("USE_WEBSOCKET") {
            self.use_websocket = val.to_lowercase() == "true";
        }
        if let Ok(val) = env::var("USE_REST_FALLBACK") {
            self.use_rest_fallback = val.to_lowercase() == "true";
        }
        if let Ok(val) = env::var("REST_POLLING_INTERVAL_MS") {
            if let Ok(interval) = val.parse::<u64>() {
                self.rest_polling_interval_ms = interval;
            }
        }
        
        // WebSocket connection settings
        if let Ok(val) = env::var("WS_RECONNECT_BACKOFF_ENABLED") {
            self.ws_reconnect_backoff_enabled = val.to_lowercase() == "true";
        }
        if let Ok(val) = env::var("WS_CIRCUIT_BREAKER_ENABLED") {
            self.ws_circuit_breaker_enabled = val.to_lowercase() == "true";
        }
        if let Ok(val) = env::var("WS_MAX_RECONNECT_ATTEMPTS") {
            if let Ok(attempts) = val.parse::<u32>() {
                self.ws_max_reconnect_attempts = attempts;
            }
        }
        
        // Paper trading configuration
        if let Ok(val) = env::var("PAPER_TRADING") {
            self.paper_trading_enabled = val.to_lowercase() == "true";
        }
        
        // Paper trading initial balances
        if let Ok(val) = env::var("PAPER_TRADING_INITIAL_BALANCE_USDT") {
            if let Ok(balance) = val.parse::<f64>() {
                self.paper_trading_initial_balances.insert("USDT".to_string(), balance);
            }
        }
        if let Ok(val) = env::var("PAPER_TRADING_INITIAL_BALANCE_BTC") {
            if let Ok(balance) = val.parse::<f64>() {
                self.paper_trading_initial_balances.insert("BTC".to_string(), balance);
            }
        }
        if let Ok(val) = env::var("PAPER_TRADING_INITIAL_BALANCE_ETH") {
            if let Ok(balance) = val.parse::<f64>() {
                self.paper_trading_initial_balances.insert("ETH".to_string(), balance);
            }
        }
        
        // Paper trading models
        if let Ok(val) = env::var("PAPER_TRADING_SLIPPAGE_MODEL") {
            self.paper_trading_slippage_model = val.to_uppercase();
        }
        if let Ok(val) = env::var("PAPER_TRADING_LATENCY_MODEL") {
            self.paper_trading_latency_model = val.to_uppercase();
        }
        if let Ok(val) = env::var("PAPER_TRADING_FEE_RATE") {
            if let Ok(rate) = val.parse::<f64>() {
                self.paper_trading_fee_rate = rate;
            }
        }
        
        // Trading configuration
        if let Ok(val) = env::var("TRADING_PAIRS") {
            self.trading_pairs = val.split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
        }
        if let Ok(val) = env::var("DEFAULT_ORDER_SIZE") {
            if let Ok(size) = val.parse::<f64>() {
                self.default_order_size = size;
            }
        }
        if let Ok(val) = env::var("MAX_POSITION_SIZE") {
            if let Ok(size) = val.parse::<f64>() {
                self.max_position_size = size;
            }
        }
        if let Ok(val) = env::var("MAX_DRAWDOWN_PERCENT") {
            if let Ok(percent) = val.parse::<f64>() {
                self.max_drawdown_percent = percent;
            }
        }
        
        // Dashboard configuration
        if let Ok(val) = env::var("SERVE_DASHBOARD") {
            self.serve_dashboard = val.to_lowercase() == "true";
        }
        if let Ok(val) = env::var("DASHBOARD_PATH") {
            self.dashboard_path = Some(val);
        }
        
        // Logging configuration
        if let Ok(val) = env::var("LOG_LEVEL") {
            self.log_level = val.to_lowercase();
        }
        if let Ok(val) = env::var("ENABLE_TELEMETRY") {
            self.enable_telemetry = val.to_lowercase() == "true";
        }
        
        // Log important configuration settings
        debug!("Enhanced configuration loaded from environment variables");
        debug!("Paper trading: {}", if self.paper_trading_enabled { "enabled" } else { "disabled" });
        debug!("Dashboard: {}", if self.serve_dashboard { "enabled" } else { "disabled" });
    }
    
    /// Update paper trading settings
    pub fn update_paper_trading_settings(
        &mut self,
        initial_balances: HashMap<String, f64>,
        trading_pairs: Vec<String>,
        max_position_size: f64,
        default_order_size: f64,
        max_drawdown_percent: f64,
        slippage_model: String,
        latency_model: String,
        trading_fees: f64,
    ) {
        self.paper_trading_initial_balances = initial_balances;
        self.trading_pairs = trading_pairs;
        self.max_position_size = max_position_size;
        self.default_order_size = default_order_size;
        self.max_drawdown_percent = max_drawdown_percent;
        self.paper_trading_slippage_model = slippage_model;
        self.paper_trading_latency_model = latency_model;
        self.paper_trading_fee_rate = trading_fees;
    }
}
