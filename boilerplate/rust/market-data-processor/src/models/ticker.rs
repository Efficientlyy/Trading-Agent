use serde::{Deserialize, Serialize};

/// Represents market ticker data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ticker {
    /// Trading pair symbol (e.g., "BTCUSDC")
    pub symbol: String,
    
    /// Current price
    pub price: f64,
    
    /// 24-hour trading volume
    pub volume: f64,
    
    /// 24-hour high price
    pub high: f64,
    
    /// 24-hour low price
    pub low: f64,
    
    /// Timestamp in milliseconds
    pub timestamp: u64,
}

impl Ticker {
    /// Create a new ticker instance
    pub fn new(symbol: String, price: f64, volume: f64, high: f64, low: f64, timestamp: u64) -> Self {
        Self {
            symbol,
            price,
            volume,
            high,
            low,
            timestamp,
        }
    }
    
    /// Convert ticker to JSON string
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
    
    /// Create ticker from JSON string
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ticker_serialization() {
        let ticker = Ticker::new(
            "BTCUSDC".to_string(),
            35000.0,
            1000.0,
            36000.0,
            34000.0,
            1621500000000,
        );
        
        let json = ticker.to_json().unwrap();
        let deserialized = Ticker::from_json(&json).unwrap();
        
        assert_eq!(deserialized.symbol, "BTCUSDC");
        assert_eq!(deserialized.price, 35000.0);
        assert_eq!(deserialized.volume, 1000.0);
        assert_eq!(deserialized.high, 36000.0);
        assert_eq!(deserialized.low, 34000.0);
        assert_eq!(deserialized.timestamp, 1621500000000);
    }
}
