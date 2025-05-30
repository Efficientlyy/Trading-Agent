use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebSocketMessage {
    pub c: String,  // Channel
    pub s: String,  // Symbol
    pub d: Value,   // Data (varies by message type)
    pub t: u64,     // Timestamp
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookUpdateData {
    pub s: String,           // Symbol
    pub t: u64,              // Timestamp
    pub v: u64,              // Version
    #[serde(default)]
    pub b: Vec<[String; 2]>, // Bids [price, quantity]
    #[serde(default)]
    pub a: Vec<[String; 2]>, // Asks [price, quantity]
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeData {
    pub s: String,  // Symbol
    pub p: String,  // Price
    pub q: String,  // Quantity
    pub v: String,  // Value
    pub t: u64,     // Timestamp
    pub m: bool,    // Is buyer maker
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickerData {
    pub s: String,  // Symbol
    pub o: String,  // Open price
    pub h: String,  // High price
    pub l: String,  // Low price
    pub c: String,  // Close price
    pub v: String,  // Volume
    pub qv: String, // Quote volume
    pub t: u64,     // Timestamp
}

// MEXC-specific message types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MexcSubscriptionRequest {
    pub method: String,
    pub params: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MexcSubscriptionResponse {
    pub code: i32,
    pub msg: String,
}
