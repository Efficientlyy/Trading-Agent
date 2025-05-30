use serde::{Deserialize, Serialize};

/// WebSocket message types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WebSocketMessage {
    /// Ticker update
    Ticker {
        symbol: String,
        price: f64,
        volume: f64,
        high: f64,
        low: f64,
        timestamp: u64,
    },
    
    /// Order book update
    OrderBook {
        symbol: String,
        bids: Vec<(f64, f64)>,
        asks: Vec<(f64, f64)>,
        timestamp: u64,
    },
    
    /// Trade update
    Trade {
        id: String,
        symbol: String,
        price: f64,
        quantity: f64,
        is_buyer_maker: bool,
        timestamp: u64,
    },
    
    /// Subscription response
    SubscriptionResponse {
        id: u64,
        result: bool,
    },
    
    /// Error message
    Error {
        code: i32,
        message: String,
    },
    
    /// Ping message
    Ping {
        timestamp: u64,
    },
    
    /// Pong message
    Pong {
        timestamp: u64,
    },
}

impl WebSocketMessage {
    /// Convert message to JSON string
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
    
    /// Create message from JSON string
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_websocket_message_serialization() {
        let ticker = WebSocketMessage::Ticker {
            symbol: "BTCUSDC".to_string(),
            price: 35000.0,
            volume: 1000.0,
            high: 36000.0,
            low: 34000.0,
            timestamp: 1621500000000,
        };
        
        let json = ticker.to_json().unwrap();
        let deserialized = WebSocketMessage::from_json(&json).unwrap();
        
        match deserialized {
            WebSocketMessage::Ticker { symbol, price, .. } => {
                assert_eq!(symbol, "BTCUSDC");
                assert_eq!(price, 35000.0);
            }
            _ => panic!("Deserialized to wrong variant"),
        }
    }
}
