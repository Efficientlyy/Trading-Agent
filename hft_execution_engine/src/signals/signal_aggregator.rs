//! Signal aggregation for high-frequency trading
//! 
//! This module provides a framework for combining multiple trading signals
//! into a unified decision with confidence scoring.

use std::time::{SystemTime, UNIX_EPOCH};
use serde::{Serialize, Deserialize};
use super::momentum_signals::MomentumSignal;
use super::mean_reversion::MeanReversionSignal;
use super::pattern_recognition::PatternSignal;

/// Signal strength enum for categorizing signal intensity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignalStrength {
    /// Strong buy signal
    StrongBuy,
    /// Weak buy signal
    WeakBuy,
    /// Neutral signal (no action)
    Neutral,
    /// Weak sell signal
    WeakSell,
    /// Strong sell signal
    StrongSell,
}

impl SignalStrength {
    /// Convert signal value to enum
    pub fn from_value(value: f64) -> Self {
        if value > 0.7 {
            Self::StrongBuy
        } else if value > 0.3 {
            Self::WeakBuy
        } else if value < -0.7 {
            Self::StrongSell
        } else if value < -0.3 {
            Self::WeakSell
        } else {
            Self::Neutral
        }
    }
    
    /// Convert to numeric value
    pub fn to_value(&self) -> f64 {
        match self {
            Self::StrongBuy => 1.0,
            Self::WeakBuy => 0.5,
            Self::Neutral => 0.0,
            Self::WeakSell => -0.5,
            Self::StrongSell => -1.0,
        }
    }
}

/// Unified trading signal with confidence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingSignal {
    /// Symbol this signal is for
    pub symbol: String,
    /// Signal direction and strength
    pub strength: SignalStrength,
    /// Signal value (-1.0 to 1.0)
    pub value: f64,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Expected duration in milliseconds
    pub expected_duration_ms: u64,
    /// Timestamp in microseconds
    pub timestamp_us: u64,
}

impl TradingSignal {
    /// Create a new trading signal
    pub fn new(symbol: &str, value: f64, confidence: f64, duration_ms: u64) -> Self {
        let timestamp_us = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;
            
        Self {
            symbol: symbol.to_string(),
            strength: SignalStrength::from_value(value),
            value,
            confidence,
            expected_duration_ms: duration_ms,
            timestamp_us,
        }
    }
    
    /// Check if this is an actionable signal
    pub fn is_actionable(&self, min_confidence: f64) -> bool {
        self.confidence >= min_confidence && self.strength != SignalStrength::Neutral
    }
    
    /// Check if this is a buy signal
    pub fn is_buy(&self) -> bool {
        matches!(self.strength, SignalStrength::StrongBuy | SignalStrength::WeakBuy)
    }
    
    /// Check if this is a sell signal
    pub fn is_sell(&self) -> bool {
        matches!(self.strength, SignalStrength::StrongSell | SignalStrength::WeakSell)
    }
}

/// Signal aggregator for combining multiple signal sources
#[derive(Debug)]
pub struct SignalAggregator {
    /// Symbol this aggregator is for
    symbol: String,
    /// Minimum confidence threshold for actionable signals
    min_confidence: f64,
    /// Weight for momentum signals
    momentum_weight: f64,
    /// Weight for mean reversion signals
    mean_reversion_weight: f64,
    /// Weight for pattern recognition signals
    pattern_weight: f64,
}

impl SignalAggregator {
    /// Create a new signal aggregator
    pub fn new(symbol: &str) -> Self {
        Self {
            symbol: symbol.to_string(),
            min_confidence: 0.6,
            momentum_weight: 0.4,
            mean_reversion_weight: 0.3,
            pattern_weight: 0.3,
        }
    }
    
    /// Create a new signal aggregator with custom weights
    pub fn with_weights(
        symbol: &str,
        momentum_weight: f64,
        mean_reversion_weight: f64,
        pattern_weight: f64,
    ) -> Self {
        Self {
            symbol: symbol.to_string(),
            min_confidence: 0.6,
            momentum_weight,
            mean_reversion_weight,
            pattern_weight,
        }
    }
    
    /// Set minimum confidence threshold
    pub fn with_min_confidence(mut self, min_confidence: f64) -> Self {
        self.min_confidence = min_confidence;
        self
    }
    
    /// Aggregate signals from multiple sources
    pub fn aggregate(
        &self,
        momentum: Option<&MomentumSignal>,
        mean_reversion: Option<&MeanReversionSignal>,
        pattern: Option<&PatternSignal>,
    ) -> TradingSignal {
        let timestamp_us = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;
            
        // Extract signal values and confidences
        let (momentum_value, momentum_conf) = momentum
            .map(|s| (s.combined_momentum, s.strength))
            .unwrap_or((0.0, 0.0));
            
        let (mean_rev_value, mean_rev_conf) = mean_reversion
            .map(|s| (s.signal_strength, s.z_score.abs() / 3.0))
            .unwrap_or((0.0, 0.0));
            
        let (pattern_value, pattern_conf, pattern_duration) = pattern
            .map(|s| (s.signal_strength, s.confidence, s.expected_duration_ms))
            .unwrap_or((0.0, 0.0, 0));
            
        // Calculate weighted signal value
        let total_weight = 
            (if momentum.is_some() { self.momentum_weight } else { 0.0 }) +
            (if mean_reversion.is_some() { self.mean_reversion_weight } else { 0.0 }) +
            (if pattern.is_some() { self.pattern_weight } else { 0.0 });
            
        let value = if total_weight > 0.0 {
            (momentum_value * self.momentum_weight * momentum_conf +
             mean_rev_value * self.mean_reversion_weight * mean_rev_conf +
             pattern_value * self.pattern_weight * pattern_conf) / total_weight
        } else {
            0.0
        };
        
        // Calculate overall confidence
        let confidence = if total_weight > 0.0 {
            (momentum_conf * self.momentum_weight +
             mean_rev_conf * self.mean_reversion_weight +
             pattern_conf * self.pattern_weight) / total_weight
        } else {
            0.0
        };
        
        // Determine expected duration (use shortest non-zero duration)
        let momentum_duration = 500; // Default duration for momentum signals
        let mean_rev_duration = 1000; // Default duration for mean reversion signals
        
        let duration_ms = [
            if momentum.is_some() { momentum_duration } else { u64::MAX },
            if mean_reversion.is_some() { mean_rev_duration } else { u64::MAX },
            if pattern.is_some() { pattern_duration } else { u64::MAX },
        ]
        .iter()
        .filter(|&&d| d < u64::MAX)
        .min()
        .copied()
        .unwrap_or(500);
        
        TradingSignal {
            symbol: self.symbol.clone(),
            strength: SignalStrength::from_value(value),
            value,
            confidence,
            expected_duration_ms: duration_ms,
            timestamp_us,
        }
    }
}
