// API module for Market Data Processor
pub mod grpc;
pub mod server;
pub mod visualization;
pub mod paper_trading_api;

// Re-export visualization components
pub use visualization::TradingDashboard;

// Re-export paper trading API components
pub use paper_trading_api::paper_trading_routes;
