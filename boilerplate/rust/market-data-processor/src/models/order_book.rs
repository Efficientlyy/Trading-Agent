use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookEntry {
    pub price: f64,
    pub quantity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    pub symbol: String,
    pub last_update_id: u64,
    pub bids: BTreeMap<f64, f64>,  // price -> quantity
    pub asks: BTreeMap<f64, f64>,  // price -> quantity
    pub timestamp: u64,
}

// Additional methods for OrderBook
impl OrderBook {
    pub fn new(symbol: &str) -> Self {
        Self {
            symbol: symbol.to_string(),
            last_update_id: 0,
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            timestamp: 0,
        }
    }
    
    pub fn update_bid(&mut self, price: f64, quantity: f64) {
        if quantity > 0.0 {
            self.bids.insert(price, quantity);
        } else {
            self.bids.remove(&price);
        }
    }
    
    pub fn update_ask(&mut self, price: f64, quantity: f64) {
        if quantity > 0.0 {
            self.asks.insert(price, quantity);
        } else {
            self.asks.remove(&price);
        }
    }
    
    // Additional methods for order book analysis
    pub fn best_bid(&self) -> Option<(f64, f64)> {
        self.bids.iter().next_back().map(|(k, v)| (*k, *v))
    }
    
    pub fn best_ask(&self) -> Option<(f64, f64)> {
        self.asks.iter().next().map(|(k, v)| (*k, *v))
    }
    
    pub fn spread(&self) -> Option<f64> {
        match (self.best_ask(), self.best_bid()) {
            (Some((ask, _)), Some((bid, _))) => Some(ask - bid),
            _ => None,
        }
    }
    
    // Get mid price
    pub fn mid_price(&self) -> Option<f64> {
        match (self.best_ask(), self.best_bid()) {
            (Some((ask, _)), Some((bid, _))) => Some((ask + bid) / 2.0),
            _ => None,
        }
    }
    
    // Calculate depth at specific price levels
    pub fn bid_depth(&self, levels: usize) -> Vec<(f64, f64)> {
        self.bids.iter()
            .rev()
            .take(levels)
            .map(|(k, v)| (*k, *v))
            .collect()
    }
    
    pub fn ask_depth(&self, levels: usize) -> Vec<(f64, f64)> {
        self.asks.iter()
            .take(levels)
            .map(|(k, v)| (*k, *v))
            .collect()
    }
}
