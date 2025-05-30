use prometheus::{register_histogram_vec, register_counter_vec, register_gauge_vec, HistogramVec, CounterVec, GaugeVec};
use lazy_static::lazy_static;

// Re-export prometheus for user convenience
pub use prometheus;

// Performance monitoring metrics for the trading system
lazy_static! {
    // Order execution metrics
    pub static ref ORDER_EXECUTION_LATENCY: HistogramVec = register_histogram_vec!(
        "order_execution_latency_seconds",
        "Time taken from strategy signal to order submission",
        &["trading_pair", "order_type", "side"],
        vec![0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    ).unwrap();
    
    pub static ref ORDERS_EXECUTED_TOTAL: CounterVec = register_counter_vec!(
        "orders_executed_total",
        "Total number of orders executed",
        &["trading_pair", "order_type", "side", "status"]
    ).unwrap();
    
    pub static ref OPEN_ORDERS_COUNT: GaugeVec = register_gauge_vec!(
        "open_orders_count",
        "Number of open orders",
        &["trading_pair"]
    ).unwrap();

    // Market data processing metrics
    pub static ref MARKET_DATA_THROUGHPUT: CounterVec = register_counter_vec!(
        "market_data_updates_total",
        "Number of market data updates processed",
        &["trading_pair", "update_type"]
    ).unwrap();
    
    pub static ref MARKET_PRICE: GaugeVec = register_gauge_vec!(
        "market_price",
        "Current market price for trading pairs",
        &["trading_pair"]
    ).unwrap();
    
    pub static ref MARKET_VOLUME_24H: GaugeVec = register_gauge_vec!(
        "market_volume_24h",
        "24-hour trading volume",
        &["trading_pair"]
    ).unwrap();

    // Signal generation metrics
    pub static ref SIGNAL_GENERATION_TIME: HistogramVec = register_histogram_vec!(
        "signal_generation_seconds",
        "Time taken to generate trading signals",
        &["strategy", "trading_pair"],
        vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
    ).unwrap();
    
    pub static ref SIGNALS_GENERATED_TOTAL: CounterVec = register_counter_vec!(
        "signals_generated_total",
        "Total number of trading signals generated",
        &["strategy", "trading_pair", "signal_type"]
    ).unwrap();

    // API metrics
    pub static ref API_REQUEST_DURATION: HistogramVec = register_histogram_vec!(
        "api_request_duration_seconds",
        "Duration of API requests",
        &["endpoint", "method", "status"],
        vec![0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0]
    ).unwrap();

    pub static ref WEBSOCKET_MESSAGE_DURATION: HistogramVec = register_histogram_vec!(
        "websocket_message_duration_seconds",
        "Time taken to process WebSocket messages",
        &["message_type"],
        vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5]
    ).unwrap();
}

// Performance monitoring utility functions
pub mod utils;
pub mod collectors;

/// Initialize metrics for Prometheus scraping
pub fn initialize() {
    // This function should be called at application startup
    // to ensure Prometheus metrics are properly initialized
    // and a metrics endpoint is registered
    
    // Ensure metrics registry is initialized
    prometheus::default_registry();
}
