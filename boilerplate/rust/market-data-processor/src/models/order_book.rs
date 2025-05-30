use serde::{Deserialize, Serialize};

/// Represents an order book with bids and asks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    /// Trading pair symbol (e.g., "BTCUSDC")
    pub symbol: String,
    
    /// List of bids as (price, quantity) tuples
    pub bids: Vec<(f64, f64)>,
    
    /// List of asks as (price, quantity) tuples
    pub asks: Vec<(f64, f64)>,
    
    /// Timestamp in milliseconds
    pub timestamp: u64,
}

impl OrderBook {
    /// Create a new order book instance
    pub fn new(symbol: String, bids: Vec<(f64, f64)>, asks: Vec<(f64, f64)>, timestamp: u64) -> Self {
        Self {
            symbol,
            bids,
            asks,
            timestamp,
        }
    }
    
    /// Get the best bid price
    pub fn best_bid(&self) -> Option<f64> {
        self.bids.first().map(|(price, _)| *price)
    }
    
    /// Get the best ask price
    pub fn best_ask(&self) -> Option<f64> {
        self.asks.first().map(|(price, _)| *price)
    }
    
    /// Get the spread (difference between best ask and best bid)
    pub fn spread(&self) -> Option<f64> {
        match (self.best_ask(), self.best_bid()) {
            (Some(ask), Some(bid)) => Some(ask - bid),
            _ => None,
        }
    }
    
    /// Calculate the total bid volume up to a certain price level
    pub fn bid_volume_at_price(&self, price: f64) -> f64 {
        self.bids
            .iter()
            .filter(|(bid_price, _)| *bid_price >= price)
            .map(|(_, quantity)| *quantity)
            .sum()
    }
    
    /// Calculate the total ask volume up to a certain price level
    pub fn ask_volume_at_price(&self, price: f64) -> f64 {
        self.asks
            .iter()
            .filter(|(ask_price, _)| *ask_price <= price)
            .map(|(_, quantity)| *quantity)
            .sum()
    }
    
    /// Calculate order book imbalance (bid volume - ask volume) / (bid volume + ask volume)
    pub fn imbalance(&self, levels: usize) -> Option<f64> {
        let bid_levels = self.bids.iter().take(levels);
        let ask_levels = self.asks.iter().take(levels);
        
        let bid_volume: f64 = bid_levels.map(|(_, qty)| *qty).sum();
        let ask_volume: f64 = ask_levels.map(|(_, qty)| *qty).sum();
        
        let total_volume = bid_volume + ask_volume;
        
        if total_volume > 0.0 {
            Some((bid_volume - ask_volume) / total_volume)
        } else {
            None
        }
    }
    
    /// Convert order book to JSON string
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
    
    /// Create order book from JSON string
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_orderbook_best_prices() {
        let bids = vec![(34900.0, 1.0), (34800.0, 2.0), (34700.0, 3.0)];
        let asks = vec![(35000.0, 1.0), (35100.0, 2.0), (35200.0, 3.0)];
        
        let orderbook = OrderBook::new(
            "BTCUSDC".to_string(),
            bids,
            asks,
            1621500000000,
        );
        
        assert_eq!(orderbook.best_bid(), Some(34900.0));
        assert_eq!(orderbook.best_ask(), Some(35000.0));
        assert_eq!(orderbook.spread(), Some(100.0));
    }
    
    #[test]
    fn test_orderbook_volume_at_price() {
        let bids = vec![(34900.0, 1.0), (34800.0, 2.0), (34700.0, 3.0)];
        let asks = vec![(35000.0, 1.0), (35100.0, 2.0), (35200.0, 3.0)];
        
        let orderbook = OrderBook::new(
            "BTCUSDC".to_string(),
            bids,
            asks,
            1621500000000,
        );
        
        assert_eq!(orderbook.bid_volume_at_price(34800.0), 3.0); // 1.0 + 2.0
        assert_eq!(orderbook.ask_volume_at_price(35100.0), 3.0); // 1.0 + 2.0
    }
    
    #[test]
    fn test_orderbook_imbalance() {
        let bids = vec![(34900.0, 2.0), (34800.0, 2.0), (34700.0, 3.0)];
        let asks = vec![(35000.0, 1.0), (35100.0, 2.0), (35200.0, 3.0)];
        
        let orderbook = OrderBook::new(
            "BTCUSDC".to_string(),
            bids,
            asks,
            1621500000000,
        );
        
        // For top 2 levels: (2.0 + 2.0) - (1.0 + 2.0) / (2.0 + 2.0 + 1.0 + 2.0) = 1.0 / 7.0
        assert_eq!(orderbook.imbalance(2), Some(1.0 / 7.0));
    }
}
