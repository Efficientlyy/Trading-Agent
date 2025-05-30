// Library entry point for the Market Data Processor
pub mod api;
pub mod models;
pub mod services;
pub mod utils;

// Re-export commonly used items
pub use crate::api::server::ApiServer;
pub use crate::services::market_data_service::MarketDataService;
pub use crate::services::order_execution::OrderExecutionService;
pub use crate::services::paper_trading::PaperTradingService;
pub use crate::services::signal_generator::SignalGenerator;
pub use crate::utils::config::Config;
pub use crate::utils::enhanced_config::EnhancedConfig;
pub use crate::utils::error::Error;
