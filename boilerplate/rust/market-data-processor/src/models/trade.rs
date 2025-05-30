use serde::{Deserialize, Serialize};

/// Represents a trade execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    /// Trade ID
    pub id: String,
    
    /// Trading pair symbol (e.g., "BTCUSDC")
    pub symbol: String,
    
    /// Trade price
    pub price: f64,
    
    /// Trade quantity
    pub quantity: f64,
    
    /// Whether the buyer was the maker
    pub is_buyer_maker: bool,
    
    /// Timestamp in milliseconds
    pub timestamp: u64,
}

impl Trade {
    /// Create a new trade instance
    pub fn new(
        id: String,
        symbol: String,
        price: f64,
        quantity: f64,
        is_buyer_maker: bool,
        timestamp: u64,
    ) -> Self {
        Self {
            id,
            symbol,
            price,
            quantity,
            is_buyer_maker,
            timestamp,
        }
    }
    
    /// Get the trade side (buy or sell)
    pub fn side(&self) -> &str {
        if self.is_buyer_maker {
            "sell"
        } else {
            "buy"
        }
    }
    
    /// Get the trade value (price * quantity)
    pub fn value(&self) -> f64 {
        self.price * self.quantity
    }
    
    /// Convert trade to JSON string
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
    
    /// Create trade from JSON string
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_trade_side() {
        let buy_trade = Trade::new(
            "123".to_string(),
            "BTCUSDC".to_string(),
            35000.0,
            0.1,
            false,
            1621500000000,
        );
        
        let sell_trade = Trade::new(
            "456".to_string(),
            "BTCUSDC".to_string(),
            35000.0,
            0.1,
            true,
            1621500000000,
        );
        
        assert_eq!(buy_trade.side(), "buy");
        assert_eq!(sell_trade.side(), "sell");
    }
    
    #[test]
    fn test_trade_value() {
        let trade = Trade::new(
            "123".to_string(),
            "BTCUSDC".to_string(),
            35000.0,
            0.1,
            false,
            1621500000000,
        );
        
        assert_eq!(trade.value(), 3500.0);
    }
    
    #[test]
    fn test_trade_serialization() {
        let trade = Trade::new(
            "123".to_string(),
            "BTCUSDC".to_string(),
            35000.0,
            0.1,
            false,
            1621500000000,
        );
        
        let json = trade.to_json().unwrap();
        let deserialized = Trade::from_json(&json).unwrap();
        
        assert_eq!(deserialized.id, "123");
        assert_eq!(deserialized.symbol, "BTCUSDC");
        assert_eq!(deserialized.price, 35000.0);
        assert_eq!(deserialized.quantity, 0.1);
        assert_eq!(deserialized.is_buyer_maker, false);
        assert_eq!(deserialized.timestamp, 1621500000000);
    }
}
