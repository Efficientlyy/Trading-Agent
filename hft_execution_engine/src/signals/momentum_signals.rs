//! Momentum-based signals for high-frequency trading
//! 
//! This module provides ultra-fast momentum signal generation for
//! identifying short-term price trends and breakouts.

use std::time::{SystemTime, UNIX_EPOCH};
use serde::{Serialize, Deserialize};
use super::tick_processor::TickProcessor;

/// Configuration for momentum signal generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MomentumConfig {
    /// Short lookback period in ticks
    pub short_period: usize,
    /// Medium lookback period in ticks
    pub medium_period: usize,
    /// Long lookback period in ticks
    pub long_period: usize,
    /// Threshold for strong momentum signal
    pub strong_threshold: f64,
    /// Threshold for weak momentum signal
    pub weak_threshold: f64,
}

impl Default for MomentumConfig {
    fn default() -> Self {
        Self {
            short_period: 10,
            medium_period: 30,
            long_period: 60,
            strong_threshold: 0.5,
            weak_threshold: 0.2,
        }
    }
}

/// Momentum-based trading signal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MomentumSignal {
    /// Symbol this signal is for
    pub symbol: String,
    /// Short-term momentum score (-1.0 to 1.0)
    pub short_momentum: f64,
    /// Medium-term momentum score (-1.0 to 1.0)
    pub medium_momentum: f64,
    /// Long-term momentum score (-1.0 to 1.0)
    pub long_momentum: f64,
    /// Combined momentum score (-1.0 to 1.0)
    pub combined_momentum: f64,
    /// Signal strength (0.0 to 1.0)
    pub strength: f64,
    /// Timestamp in microseconds
    pub timestamp_us: u64,
}

impl MomentumSignal {
    /// Calculate momentum signals from tick processor
    pub fn from_tick_processor(processor: &TickProcessor, config: &MomentumConfig) -> Self {
        let timestamp_us = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;
            
        let ticks = processor.ticks();
        let tick_count = ticks.len();
        
        // Calculate momentum over different periods
        let short_momentum = if tick_count >= config.short_period {
            calculate_momentum(&ticks.iter().rev().take(config.short_period).collect::<Vec<_>>())
        } else {
            0.0
        };
        
        let medium_momentum = if tick_count >= config.medium_period {
            calculate_momentum(&ticks.iter().rev().take(config.medium_period).collect::<Vec<_>>())
        } else {
            0.0
        };
        
        let long_momentum = if tick_count >= config.long_period {
            calculate_momentum(&ticks.iter().rev().take(config.long_period).collect::<Vec<_>>())
        } else {
            0.0
        };
        
        // Calculate combined momentum (weighted average)
        let combined_momentum = if tick_count >= config.short_period {
            short_momentum * 0.5 + medium_momentum * 0.3 + long_momentum * 0.2
        } else {
            0.0
        };
        
        // Calculate signal strength based on consistency across timeframes
        let consistency = calculate_consistency(short_momentum, medium_momentum, long_momentum);
        let magnitude = combined_momentum.abs();
        
        let strength = consistency * magnitude;
        
        Self {
            symbol: processor.symbol().to_string(),
            short_momentum,
            medium_momentum,
            long_momentum,
            combined_momentum,
            strength,
            timestamp_us,
        }
    }
    
    /// Get the trading signal direction (-1.0 to 1.0)
    pub fn signal(&self) -> f64 {
        self.combined_momentum
    }
    
    /// Check if this is a strong buy signal
    pub fn is_strong_buy(&self, config: &MomentumConfig) -> bool {
        self.combined_momentum > config.strong_threshold && self.strength > 0.7
    }
    
    /// Check if this is a weak buy signal
    pub fn is_weak_buy(&self, config: &MomentumConfig) -> bool {
        self.combined_momentum > config.weak_threshold && self.strength > 0.5
    }
    
    /// Check if this is a strong sell signal
    pub fn is_strong_sell(&self, config: &MomentumConfig) -> bool {
        self.combined_momentum < -config.strong_threshold && self.strength > 0.7
    }
    
    /// Check if this is a weak sell signal
    pub fn is_weak_sell(&self, config: &MomentumConfig) -> bool {
        self.combined_momentum < -config.weak_threshold && self.strength > 0.5
    }
}

/// Calculate momentum score from a sequence of ticks
fn calculate_momentum(ticks: &[&TickData]) -> f64 {
    if ticks.len() < 2 {
        return 0.0;
    }
    
    // Linear regression slope calculation
    let n = ticks.len() as f64;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xy = 0.0;
    let mut sum_xx = 0.0;
    
    for (i, tick) in ticks.iter().enumerate() {
        let x = i as f64;
        let y = tick.price;
        
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_xx += x * x;
    }
    
    let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
    
    // Normalize slope to -1.0 to 1.0 range
    // This is a simple normalization that can be improved
    let avg_price = sum_y / n;
    let normalized_slope = slope / (avg_price * 0.01);
    
    normalized_slope.max(-1.0).min(1.0)
}

/// Calculate consistency between different timeframe signals
fn calculate_consistency(short: f64, medium: f64, long: f64) -> f64 {
    let short_sign = short.signum();
    let medium_sign = medium.signum();
    let long_sign = long.signum();
    
    if short_sign == medium_sign && medium_sign == long_sign {
        // All timeframes agree
        1.0
    } else if short_sign == medium_sign || medium_sign == long_sign || short_sign == long_sign {
        // Two timeframes agree
        0.7
    } else {
        // No agreement
        0.3
    }
}

/// Tick data for momentum calculation
use super::tick_processor::TickData;
