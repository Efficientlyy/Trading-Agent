//! Order book analytics for high-frequency trading
//! 
//! This module provides advanced analytics for order book microstructure,
//! including order flow imbalance detection, liquidity analysis, and market pressure metrics.

use std::collections::VecDeque;
use std::time::{SystemTime, UNIX_EPOCH};
use serde::{Serialize, Deserialize};
use super::types::{OrderBook, Side};

/// Window size for rolling analytics calculations
const DEFAULT_WINDOW_SIZE: usize = 20;

/// Order flow imbalance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderFlowImbalance {
    /// Symbol this analysis is for
    pub symbol: String,
    /// Ratio of bid to ask volume in the order book
    pub volume_ratio: f64,
    /// Imbalance score (-1.0 to 1.0, negative means more selling pressure)
    pub imbalance_score: f64,
    /// Recent price trend direction (-1.0 to 1.0)
    pub price_trend: f64,
    /// Timestamp of this analysis
    pub timestamp: u64,
}

impl OrderFlowImbalance {
    /// Calculate order flow imbalance from an order book
    pub fn from_orderbook(book: &OrderBook, depth: usize) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
            
        let bid_value = book.total_bid_value(depth);
        let ask_value = book.total_ask_value(depth);
        
        // Calculate volume ratio (bid/ask)
        let volume_ratio = if ask_value > 0.0 {
            bid_value / ask_value
        } else {
            1.0
        };
        
        // Calculate imbalance score (-1.0 to 1.0)
        let imbalance_score = if bid_value + ask_value > 0.0 {
            (bid_value - ask_value) / (bid_value + ask_value)
        } else {
            0.0
        };
        
        // For price trend, we'd need historical data
        // This is a placeholder that would be replaced with actual trend calculation
        let price_trend = 0.0;
        
        Self {
            symbol: book.symbol.clone(),
            volume_ratio,
            imbalance_score,
            price_trend,
            timestamp,
        }
    }
    
    /// Determine if there's significant buying pressure
    pub fn is_buying_pressure(&self) -> bool {
        self.imbalance_score > 0.2
    }
    
    /// Determine if there's significant selling pressure
    pub fn is_selling_pressure(&self) -> bool {
        self.imbalance_score < -0.2
    }
}

/// Liquidity metrics for market analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiquidityMetrics {
    /// Symbol this analysis is for
    pub symbol: String,
    /// Current spread in quote currency
    pub spread: f64,
    /// Current spread as percentage of mid price
    pub spread_pct: f64,
    /// Depth available within 0.1% of mid price on bid side
    pub bid_depth_bps_10: f64,
    /// Depth available within 0.1% of mid price on ask side
    pub ask_depth_bps_10: f64,
    /// Depth available within 0.5% of mid price on bid side
    pub bid_depth_bps_50: f64,
    /// Depth available within 0.5% of mid price on ask side
    pub ask_depth_bps_50: f64,
    /// Timestamp of this analysis
    pub timestamp: u64,
}

impl LiquidityMetrics {
    /// Calculate liquidity metrics from an order book
    pub fn from_orderbook(book: &OrderBook) -> Option<Self> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
            
        let mid_price = book.mid_price()?;
        let spread = book.spread()?;
        let spread_pct = book.spread_pct()?;
        
        // Calculate depth within 0.1% (10 bps) of mid price
        let bps_10 = mid_price * 0.001;
        let bid_threshold_10 = mid_price - bps_10;
        let ask_threshold_10 = mid_price + bps_10;
        
        // Calculate depth within 0.5% (50 bps) of mid price
        let bps_50 = mid_price * 0.005;
        let bid_threshold_50 = mid_price - bps_50;
        let ask_threshold_50 = mid_price + bps_50;
        
        // Sum up liquidity within thresholds
        let mut bid_depth_bps_10 = 0.0;
        let mut ask_depth_bps_10 = 0.0;
        let mut bid_depth_bps_50 = 0.0;
        let mut ask_depth_bps_50 = 0.0;
        
        for level in book.bids.values() {
            if level.price >= bid_threshold_50 {
                bid_depth_bps_50 += level.quantity;
                
                if level.price >= bid_threshold_10 {
                    bid_depth_bps_10 += level.quantity;
                }
            }
        }
        
        for level in book.asks.values() {
            if level.price <= ask_threshold_50 {
                ask_depth_bps_50 += level.quantity;
                
                if level.price <= ask_threshold_10 {
                    ask_depth_bps_10 += level.quantity;
                }
            }
        }
        
        Some(Self {
            symbol: book.symbol.clone(),
            spread,
            spread_pct,
            bid_depth_bps_10,
            ask_depth_bps_10,
            bid_depth_bps_50,
            ask_depth_bps_50,
            timestamp,
        })
    }
    
    /// Calculate the liquidity imbalance ratio (bid depth / ask depth)
    pub fn liquidity_imbalance_ratio(&self) -> f64 {
        if self.ask_depth_bps_50 > 0.0 {
            self.bid_depth_bps_50 / self.ask_depth_bps_50
        } else {
            1.0
        }
    }
    
    /// Determine if the market is liquid enough for trading
    pub fn is_liquid_enough(&self, min_depth: f64) -> bool {
        self.bid_depth_bps_10 >= min_depth && self.ask_depth_bps_10 >= min_depth
    }
}

/// Market pressure analysis for short-term price prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketPressure {
    /// Symbol this analysis is for
    pub symbol: String,
    /// Current buying pressure score (-1.0 to 1.0)
    pub buying_pressure: f64,
    /// Current selling pressure score (-1.0 to 1.0)
    pub selling_pressure: f64,
    /// Net pressure score (-1.0 to 1.0)
    pub net_pressure: f64,
    /// Confidence score for the pressure analysis (0.0 to 1.0)
    pub confidence: f64,
    /// Timestamp of this analysis
    pub timestamp: u64,
}

impl MarketPressure {
    /// Calculate market pressure from order book and recent updates
    pub fn from_orderbook(
        book: &OrderBook,
        imbalance: &OrderFlowImbalance,
        liquidity: &LiquidityMetrics,
    ) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
            
        // Calculate buying pressure based on order book imbalance and liquidity
        let buying_pressure = imbalance.imbalance_score.max(0.0) * 0.7 + 
            (liquidity.liquidity_imbalance_ratio() - 1.0).min(1.0).max(0.0) * 0.3;
            
        // Calculate selling pressure based on order book imbalance and liquidity
        let selling_pressure = (-imbalance.imbalance_score).max(0.0) * 0.7 + 
            (1.0 - liquidity.liquidity_imbalance_ratio()).min(1.0).max(0.0) * 0.3;
            
        // Calculate net pressure
        let net_pressure = buying_pressure - selling_pressure;
        
        // Calculate confidence based on spread and depth
        let confidence = (1.0 - liquidity.spread_pct / 1.0).max(0.0).min(1.0) * 0.5 +
            (liquidity.bid_depth_bps_10 + liquidity.ask_depth_bps_10).min(10.0) / 10.0 * 0.5;
            
        Self {
            symbol: book.symbol.clone(),
            buying_pressure,
            selling_pressure,
            net_pressure,
            confidence,
            timestamp,
        }
    }
    
    /// Get the trading signal from market pressure (-1.0 to 1.0)
    pub fn trading_signal(&self) -> f64 {
        self.net_pressure * self.confidence
    }
    
    /// Determine if there's a strong buy signal
    pub fn is_strong_buy(&self) -> bool {
        self.trading_signal() > 0.5
    }
    
    /// Determine if there's a strong sell signal
    pub fn is_strong_sell(&self) -> bool {
        self.trading_signal() < -0.5
    }
}
