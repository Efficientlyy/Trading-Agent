//! Pattern recognition for high-frequency trading
//! 
//! This module provides ultra-fast pattern recognition for order book
//! microstructure and price action patterns.

use std::time::{SystemTime, UNIX_EPOCH};
use serde::{Serialize, Deserialize};
use super::tick_processor::TickProcessor;
use super::tick_processor::TickData;
use crate::orderbook::types::OrderBook;

/// Configuration for pattern recognition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternConfig {
    /// Minimum pattern confidence threshold
    pub min_confidence: f64,
    /// Lookback window for pattern detection
    pub lookback_window: usize,
    /// Price threshold for significant moves (%)
    pub price_threshold_pct: f64,
    /// Volume threshold for significant activity
    pub volume_threshold_multiplier: f64,
}

impl Default for PatternConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.7,
            lookback_window: 30,
            price_threshold_pct: 0.1,
            volume_threshold_multiplier: 2.0,
        }
    }
}

/// Pattern types that can be detected
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PatternType {
    /// Sudden price spike up
    PriceSpikeUp,
    /// Sudden price spike down
    PriceSpikeDown,
    /// Volume spike with price stability
    VolumeSpikeNeutral,
    /// Volume spike with price increase
    VolumeSpikeUp,
    /// Volume spike with price decrease
    VolumeSpikeDown,
    /// Order book imbalance favoring buys
    OrderBookImbalanceBuy,
    /// Order book imbalance favoring sells
    OrderBookImbalanceSell,
    /// Rapid spread widening
    SpreadWidening,
    /// Rapid spread narrowing
    SpreadNarrowing,
    /// Iceberg order detection
    IcebergOrder,
    /// No significant pattern detected
    NoPattern,
}

/// Pattern signal for trading
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternSignal {
    /// Symbol this signal is for
    pub symbol: String,
    /// Detected pattern type
    pub pattern: PatternType,
    /// Confidence in pattern detection (0.0 to 1.0)
    pub confidence: f64,
    /// Signal strength (-1.0 to 1.0, negative for sell)
    pub signal_strength: f64,
    /// Expected duration of pattern effect in milliseconds
    pub expected_duration_ms: u64,
    /// Timestamp in microseconds
    pub timestamp_us: u64,
}

impl PatternSignal {
    /// Detect patterns from tick processor and order book
    pub fn from_data(
        processor: &TickProcessor,
        order_book: Option<&OrderBook>,
        config: &PatternConfig,
    ) -> Self {
        let timestamp_us = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;
            
        let ticks = processor.ticks();
        
        if ticks.len() < config.lookback_window {
            return Self {
                symbol: processor.symbol().to_string(),
                pattern: PatternType::NoPattern,
                confidence: 0.0,
                signal_strength: 0.0,
                expected_duration_ms: 0,
                timestamp_us,
            };
        }
        
        // Get recent ticks for analysis
        let recent_ticks: Vec<&TickData> = ticks.iter()
            .rev()
            .take(config.lookback_window)
            .collect();
            
        // Check for price spikes
        let (price_pattern, price_confidence) = detect_price_pattern(&recent_ticks, config);
        
        // Check for volume spikes
        let (volume_pattern, volume_confidence) = detect_volume_pattern(&recent_ticks, config);
        
        // Check for order book patterns if available
        let (book_pattern, book_confidence) = if let Some(book) = order_book {
            detect_order_book_pattern(book, processor, config)
        } else {
            (PatternType::NoPattern, 0.0)
        };
        
        // Select the pattern with highest confidence
        let (pattern, confidence) = [
            (price_pattern, price_confidence),
            (volume_pattern, volume_confidence),
            (book_pattern, book_confidence),
        ]
        .iter()
        .max_by(|(_, a_conf), (_, b_conf)| a_conf.partial_cmp(b_conf).unwrap())
        .copied()
        .unwrap();
        
        // Calculate signal strength and expected duration based on pattern
        let (signal_strength, expected_duration_ms) = match pattern {
            PatternType::PriceSpikeUp => (confidence * 0.8, 500),
            PatternType::PriceSpikeDown => (-confidence * 0.8, 500),
            PatternType::VolumeSpikeNeutral => (0.0, 1000),
            PatternType::VolumeSpikeUp => (confidence * 0.6, 800),
            PatternType::VolumeSpikeDown => (-confidence * 0.6, 800),
            PatternType::OrderBookImbalanceBuy => (confidence * 0.9, 300),
            PatternType::OrderBookImbalanceSell => (-confidence * 0.9, 300),
            PatternType::SpreadWidening => (0.0, 400),
            PatternType::SpreadNarrowing => (0.0, 400),
            PatternType::IcebergOrder => (0.0, 2000),
            PatternType::NoPattern => (0.0, 0),
        };
        
        Self {
            symbol: processor.symbol().to_string(),
            pattern,
            confidence,
            signal_strength,
            expected_duration_ms,
            timestamp_us,
        }
    }
    
    /// Get the trading signal direction (-1.0 to 1.0)
    pub fn signal(&self) -> f64 {
        self.signal_strength
    }
    
    /// Check if this is a buy signal
    pub fn is_buy_signal(&self, config: &PatternConfig) -> bool {
        self.signal_strength > 0.0 && self.confidence >= config.min_confidence
    }
    
    /// Check if this is a sell signal
    pub fn is_sell_signal(&self, config: &PatternConfig) -> bool {
        self.signal_strength < 0.0 && self.confidence >= config.min_confidence
    }
    
    /// Check if this pattern is significant enough to act on
    pub fn is_significant(&self, config: &PatternConfig) -> bool {
        self.confidence >= config.min_confidence && self.pattern != PatternType::NoPattern
    }
}

/// Detect price patterns from recent ticks
fn detect_price_pattern(ticks: &[&TickData], config: &PatternConfig) -> (PatternType, f64) {
    if ticks.len() < 5 {
        return (PatternType::NoPattern, 0.0);
    }
    
    // Calculate recent price changes
    let mut price_changes = Vec::with_capacity(ticks.len() - 1);
    for i in 0..ticks.len() - 1 {
        let pct_change = (ticks[i].price - ticks[i + 1].price) / ticks[i + 1].price * 100.0;
        price_changes.push(pct_change);
    }
    
    // Calculate average and standard deviation of price changes
    let avg_change: f64 = price_changes.iter().sum::<f64>() / price_changes.len() as f64;
    let variance: f64 = price_changes.iter()
        .map(|c| (c - avg_change).powi(2))
        .sum::<f64>() / price_changes.len() as f64;
    let std_dev = variance.sqrt();
    
    // Check for price spikes (changes greater than threshold)
    let threshold = config.price_threshold_pct;
    let recent_change = price_changes[0];
    
    if recent_change > threshold && recent_change > 2.0 * std_dev {
        // Significant up move
        let confidence = (recent_change / (threshold * 2.0)).min(1.0);
        (PatternType::PriceSpikeUp, confidence)
    } else if recent_change < -threshold && recent_change < -2.0 * std_dev {
        // Significant down move
        let confidence = (-recent_change / (threshold * 2.0)).min(1.0);
        (PatternType::PriceSpikeDown, confidence)
    } else {
        (PatternType::NoPattern, 0.0)
    }
}

/// Detect volume patterns from recent ticks
fn detect_volume_pattern(ticks: &[&TickData], config: &PatternConfig) -> (PatternType, f64) {
    if ticks.len() < 5 {
        return (PatternType::NoPattern, 0.0);
    }
    
    // Calculate average volume
    let volumes: Vec<f64> = ticks.iter().map(|t| t.volume).collect();
    let avg_volume = volumes.iter().sum::<f64>() / volumes.len() as f64;
    
    // Check for volume spike
    let recent_volume = volumes[0];
    let threshold = avg_volume * config.volume_threshold_multiplier;
    
    if recent_volume > threshold {
        // Volume spike detected, check price action
        let recent_price_change = if ticks.len() >= 3 {
            (ticks[0].price - ticks[2].price) / ticks[2].price * 100.0
        } else {
            0.0
        };
        
        let confidence = (recent_volume / threshold).min(1.0);
        
        if recent_price_change > config.price_threshold_pct {
            (PatternType::VolumeSpikeUp, confidence)
        } else if recent_price_change < -config.price_threshold_pct {
            (PatternType::VolumeSpikeDown, confidence)
        } else {
            (PatternType::VolumeSpikeNeutral, confidence * 0.7)
        }
    } else {
        (PatternType::NoPattern, 0.0)
    }
}

/// Detect order book patterns
fn detect_order_book_pattern(
    book: &OrderBook,
    processor: &TickProcessor,
    config: &PatternConfig,
) -> (PatternType, f64) {
    // Calculate bid/ask ratio
    let bid_value = book.total_bid_value(10);
    let ask_value = book.total_ask_value(10);
    
    if bid_value > 0.0 && ask_value > 0.0 {
        let ratio = bid_value / ask_value;
        
        // Check for significant imbalance
        if ratio > 2.0 {
            let confidence = (ratio / 4.0).min(1.0);
            (PatternType::OrderBookImbalanceBuy, confidence)
        } else if ratio < 0.5 {
            let confidence = (1.0 / ratio / 4.0).min(1.0);
            (PatternType::OrderBookImbalanceSell, confidence)
        } else {
            // Check for spread patterns
            let current_spread = processor.spread_pct();
            let avg_spread = 0.05; // This would ideally be calculated from historical data
            
            if current_spread > avg_spread * 2.0 {
                let confidence = (current_spread / (avg_spread * 3.0)).min(1.0);
                (PatternType::SpreadWidening, confidence)
            } else if current_spread < avg_spread * 0.5 {
                let confidence = (avg_spread / current_spread / 3.0).min(1.0);
                (PatternType::SpreadNarrowing, confidence)
            } else {
                (PatternType::NoPattern, 0.0)
            }
        }
    } else {
        (PatternType::NoPattern, 0.0)
    }
}
