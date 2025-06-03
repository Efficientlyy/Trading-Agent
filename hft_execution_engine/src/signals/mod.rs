//! High-frequency trading signal generation framework
//! 
//! This module provides tick-by-tick signal generation for ultra-fast trading,
//! including momentum signals, mean reversion, and order book pattern recognition.

mod tick_processor;
mod momentum_signals;
mod mean_reversion;
mod pattern_recognition;
mod signal_aggregator;

pub use tick_processor::{TickData, TickProcessor};
pub use momentum_signals::{MomentumSignal, MomentumConfig};
pub use mean_reversion::{MeanReversionSignal, MeanReversionConfig};
pub use pattern_recognition::{PatternSignal, PatternConfig};
pub use signal_aggregator::{SignalStrength, SignalAggregator, TradingSignal};

/// Minimum tick interval in microseconds for signal generation
pub const MIN_TICK_INTERVAL_US: u64 = 100;

/// Default lookback window for tick analysis
pub const DEFAULT_LOOKBACK_WINDOW: usize = 100;
