//! Order book microstructure analysis module
//! 
//! This module provides high-performance order book analytics for HFT signal generation,
//! including order flow imbalance detection, liquidity analysis, and pattern recognition.

mod types;
mod analytics;
mod visualization;

pub use types::{OrderBook, OrderBookUpdate, PriceLevel, Side};
pub use analytics::{OrderFlowImbalance, LiquidityMetrics, MarketPressure};

/// Order book depth levels to track for analysis
pub const DEFAULT_DEPTH: usize = 20;

/// Minimum update frequency in milliseconds
pub const MIN_UPDATE_FREQUENCY_MS: u64 = 10;
