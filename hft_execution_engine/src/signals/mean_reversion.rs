//! Mean reversion signals for high-frequency trading
//! 
//! This module provides ultra-fast mean reversion signal generation for
//! identifying short-term price reversals and overbought/oversold conditions.

use std::time::{SystemTime, UNIX_EPOCH};
use serde::{Serialize, Deserialize};
use super::tick_processor::TickProcessor;
use super::tick_processor::TickData;

/// Configuration for mean reversion signal generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeanReversionConfig {
    /// Lookback period for calculating mean
    pub lookback_period: usize,
    /// Standard deviation multiplier for overbought/oversold
    pub std_dev_multiplier: f64,
    /// Threshold for strong mean reversion signal
    pub strong_threshold: f64,
    /// Threshold for weak mean reversion signal
    pub weak_threshold: f64,
}

impl Default for MeanReversionConfig {
    fn default() -> Self {
        Self {
            lookback_period: 50,
            std_dev_multiplier: 2.0,
            strong_threshold: 0.8,
            weak_threshold: 0.5,
        }
    }
}

/// Mean reversion trading signal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeanReversionSignal {
    /// Symbol this signal is for
    pub symbol: String,
    /// Current price
    pub price: f64,
    /// Moving average price
    pub mean_price: f64,
    /// Standard deviation of price
    pub std_dev: f64,
    /// Z-score (how many std devs from mean)
    pub z_score: f64,
    /// Mean reversion signal strength (-1.0 to 1.0)
    pub signal_strength: f64,
    /// Timestamp in microseconds
    pub timestamp_us: u64,
}

impl MeanReversionSignal {
    /// Calculate mean reversion signals from tick processor
    pub fn from_tick_processor(processor: &TickProcessor, config: &MeanReversionConfig) -> Option<Self> {
        let timestamp_us = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;
            
        let ticks = processor.ticks();
        let tick_count = ticks.len();
        
        if tick_count < config.lookback_period {
            return None;
        }
        
        // Calculate mean price
        let recent_ticks: Vec<&TickData> = ticks.iter().rev().take(config.lookback_period).collect();
        let price_sum: f64 = recent_ticks.iter().map(|t| t.price).sum();
        let mean_price = price_sum / recent_ticks.len() as f64;
        
        // Calculate standard deviation
        let variance_sum: f64 = recent_ticks.iter()
            .map(|t| (t.price - mean_price).powi(2))
            .sum();
        let std_dev = (variance_sum / recent_ticks.len() as f64).sqrt();
        
        // Current price and z-score
        let current_price = processor.price();
        let z_score = if std_dev > 0.0 {
            (current_price - mean_price) / std_dev
        } else {
            0.0
        };
        
        // Calculate signal strength based on z-score
        // Negative z-score means price is below mean (potential buy)
        // Positive z-score means price is above mean (potential sell)
        let raw_signal = -z_score / config.std_dev_multiplier;
        let signal_strength = raw_signal.max(-1.0).min(1.0);
        
        Some(Self {
            symbol: processor.symbol().to_string(),
            price: current_price,
            mean_price,
            std_dev,
            z_score,
            signal_strength,
            timestamp_us,
        })
    }
    
    /// Get the trading signal direction (-1.0 to 1.0)
    pub fn signal(&self) -> f64 {
        self.signal_strength
    }
    
    /// Check if this is a strong buy signal (oversold)
    pub fn is_strong_buy(&self, config: &MeanReversionConfig) -> bool {
        self.signal_strength > config.strong_threshold
    }
    
    /// Check if this is a weak buy signal (slightly oversold)
    pub fn is_weak_buy(&self, config: &MeanReversionConfig) -> bool {
        self.signal_strength > config.weak_threshold
    }
    
    /// Check if this is a strong sell signal (overbought)
    pub fn is_strong_sell(&self, config: &MeanReversionConfig) -> bool {
        self.signal_strength < -config.strong_threshold
    }
    
    /// Check if this is a weak sell signal (slightly overbought)
    pub fn is_weak_sell(&self, config: &MeanReversionConfig) -> bool {
        self.signal_strength < -config.weak_threshold
    }
    
    /// Calculate distance from mean as percentage
    pub fn distance_from_mean_pct(&self) -> f64 {
        if self.mean_price > 0.0 {
            (self.price - self.mean_price) / self.mean_price * 100.0
        } else {
            0.0
        }
    }
}
