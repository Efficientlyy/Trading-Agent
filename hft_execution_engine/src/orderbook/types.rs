//! Order book data structures and types
//! 
//! This module defines the core data structures for representing and
//! manipulating order book data with high performance.

use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};
use serde::{Serialize, Deserialize};

/// Side of the order book (bid or ask)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Side {
    /// Buy side (bids)
    Bid,
    /// Sell side (asks)
    Ask,
}

/// A price level in the order book with price and quantity
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PriceLevel {
    /// Price of this level
    pub price: f64,
    /// Quantity available at this price
    pub quantity: f64,
    /// Number of orders at this price level (if available)
    pub order_count: Option<u32>,
    /// Timestamp when this level was last updated
    pub last_update: u64,
}

impl PriceLevel {
    /// Create a new price level
    pub fn new(price: f64, quantity: f64) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
            
        Self {
            price,
            quantity,
            order_count: None,
            last_update: timestamp,
        }
    }
    
    /// Create a new price level with order count
    pub fn with_order_count(price: f64, quantity: f64, order_count: u32) -> Self {
        let mut level = Self::new(price, quantity);
        level.order_count = Some(order_count);
        level
    }
    
    /// Total value at this price level (price * quantity)
    pub fn value(&self) -> f64 {
        self.price * self.quantity
    }
}

/// Complete order book snapshot with bids and asks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    /// Symbol this order book represents (e.g., "BTCUSDC")
    pub symbol: String,
    /// Bid side of the book (buy orders), sorted by price descending
    pub bids: BTreeMap<u64, PriceLevel>,
    /// Ask side of the book (sell orders), sorted by price ascending
    pub asks: BTreeMap<u64, PriceLevel>,
    /// Last update timestamp in milliseconds
    pub last_update_time: u64,
    /// Sequence number or update ID if available
    pub sequence: Option<u64>,
}

impl OrderBook {
    /// Create a new empty order book for the given symbol
    pub fn new(symbol: &str) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
            
        Self {
            symbol: symbol.to_string(),
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            last_update_time: timestamp,
            sequence: None,
        }
    }
    
    /// Get the best bid price
    pub fn best_bid(&self) -> Option<&PriceLevel> {
        self.bids.values().next()
    }
    
    /// Get the best ask price
    pub fn best_ask(&self) -> Option<&PriceLevel> {
        self.asks.values().next()
    }
    
    /// Get the current spread (best ask - best bid)
    pub fn spread(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some(ask.price - bid.price),
            _ => None,
        }
    }
    
    /// Get the current spread as a percentage of the mid price
    pub fn spread_pct(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => {
                let mid = (bid.price + ask.price) / 2.0;
                Some((ask.price - bid.price) / mid * 100.0)
            },
            _ => None,
        }
    }
    
    /// Get the mid price
    pub fn mid_price(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some((bid.price + ask.price) / 2.0),
            _ => None,
        }
    }
    
    /// Update the order book with a new price level
    pub fn update(&mut self, side: Side, price: f64, quantity: f64) {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
            
        // Convert price to sortable integer (price * 10^8)
        let price_key = (price * 100_000_000.0) as u64;
        
        let map = match side {
            Side::Bid => &mut self.bids,
            Side::Ask => &mut self.asks,
        };
        
        if quantity > 0.0 {
            map.insert(price_key, PriceLevel {
                price,
                quantity,
                order_count: None,
                last_update: timestamp,
            });
        } else {
            map.remove(&price_key);
        }
        
        self.last_update_time = timestamp;
    }
    
    /// Calculate the total bid value up to a certain depth
    pub fn total_bid_value(&self, depth: usize) -> f64 {
        self.bids.values()
            .take(depth)
            .map(|level| level.value())
            .sum()
    }
    
    /// Calculate the total ask value up to a certain depth
    pub fn total_ask_value(&self, depth: usize) -> f64 {
        self.asks.values()
            .take(depth)
            .map(|level| level.value())
            .sum()
    }
    
    /// Calculate the bid/ask ratio (total bid value / total ask value)
    pub fn bid_ask_ratio(&self, depth: usize) -> Option<f64> {
        let bid_value = self.total_bid_value(depth);
        let ask_value = self.total_ask_value(depth);
        
        if ask_value > 0.0 {
            Some(bid_value / ask_value)
        } else {
            None
        }
    }
}

/// Order book update event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookUpdate {
    /// Symbol this update is for
    pub symbol: String,
    /// Side of the book being updated
    pub side: Side,
    /// Price level being updated
    pub price: f64,
    /// New quantity (0 for deletion)
    pub quantity: f64,
    /// Update timestamp in milliseconds
    pub timestamp: u64,
}

impl OrderBookUpdate {
    /// Create a new order book update
    pub fn new(symbol: &str, side: Side, price: f64, quantity: f64) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
            
        Self {
            symbol: symbol.to_string(),
            side,
            price,
            quantity,
            timestamp,
        }
    }
    
    /// Apply this update to an order book
    pub fn apply_to(&self, book: &mut OrderBook) {
        book.update(self.side, self.price, self.quantity);
    }
}
