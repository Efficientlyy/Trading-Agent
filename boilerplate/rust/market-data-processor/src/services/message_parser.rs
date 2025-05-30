use crate::models::{
    order_book::{OrderBook, OrderBookEntry},
    trade::Trade,
    ticker::Ticker,
    websocket::{WebSocketMessage, OrderBookUpdateData, TradeData, TickerData},
};
use serde_json::Value;
use thiserror::Error;
use tracing::{debug, error};

#[derive(Debug, Error)]
pub enum ParserError {
    #[error("JSON parsing error: {0}")]
    JsonError(#[from] serde_json::Error),
    #[error("Invalid message format: {0}")]
    InvalidFormat(String),
    #[error("Unsupported channel: {0}")]
    UnsupportedChannel(String),
}

pub struct MessageParser;

impl MessageParser {
    pub fn parse_message(message: &str) -> Result<WebSocketMessage, ParserError> {
        debug!("Parsing WebSocket message: {}", message);
        let ws_message: WebSocketMessage = serde_json::from_str(message)?;
        Ok(ws_message)
    }
    
    pub fn parse_order_book_update(message: &WebSocketMessage) -> Result<OrderBookUpdateData, ParserError> {
        if !message.c.contains("depth") {
            return Err(ParserError::UnsupportedChannel(message.c.clone()));
        }
        
        let data: OrderBookUpdateData = serde_json::from_value(message.d.clone())?;
        Ok(data)
    }
    
    pub fn parse_trade(message: &WebSocketMessage) -> Result<TradeData, ParserError> {
        if !message.c.contains("deals") {
            return Err(ParserError::UnsupportedChannel(message.c.clone()));
        }
        
        let data: TradeData = serde_json::from_value(message.d.clone())?;
        Ok(data)
    }
    
    pub fn parse_ticker(message: &WebSocketMessage) -> Result<TickerData, ParserError> {
        if !message.c.contains("ticker") {
            return Err(ParserError::UnsupportedChannel(message.c.clone()));
        }
        
        let data: TickerData = serde_json::from_value(message.d.clone())?;
        Ok(data)
    }
    
    // Convert parsed data to domain models
    pub fn order_book_update_to_entries(data: &OrderBookUpdateData) -> (Vec<OrderBookEntry>, Vec<OrderBookEntry>) {
        let bids = data.b.iter().map(|[price, qty]| {
            OrderBookEntry {
                price: price.parse().unwrap_or(0.0),
                quantity: qty.parse().unwrap_or(0.0),
            }
        }).collect();
        
        let asks = data.a.iter().map(|[price, qty]| {
            OrderBookEntry {
                price: price.parse().unwrap_or(0.0),
                quantity: qty.parse().unwrap_or(0.0),
            }
        }).collect();
        
        (bids, asks)
    }
    
    pub fn trade_data_to_trade(data: &TradeData, symbol: &str) -> Trade {
        Trade {
            id: 0, // MEXC doesn't provide trade ID in WebSocket
            symbol: symbol.to_string(),
            price: data.p.parse().unwrap_or(0.0),
            quantity: data.q.parse().unwrap_or(0.0),
            buyer_order_id: 0, // Not provided
            seller_order_id: 0, // Not provided
            timestamp: data.t,
            is_buyer_maker: data.m,
        }
    }
    
    pub fn ticker_data_to_ticker(data: &TickerData, symbol: &str) -> Ticker {
        Ticker {
            symbol: symbol.to_string(),
            price_change: 0.0, // Not provided in this message
            price_change_percent: 0.0, // Not provided
            weighted_avg_price: 0.0, // Not provided
            last_price: data.c.parse().unwrap_or(0.0),
            last_quantity: 0.0, // Not provided
            bid_price: 0.0, // Not provided
            bid_quantity: 0.0, // Not provided
            ask_price: 0.0, // Not provided
            ask_quantity: 0.0, // Not provided
            open_price: data.o.parse().unwrap_or(0.0),
            high_price: data.h.parse().unwrap_or(0.0),
            low_price: data.l.parse().unwrap_or(0.0),
            volume: data.v.parse().unwrap_or(0.0),
            quote_volume: data.qv.parse().unwrap_or(0.0),
            open_time: 0, // Not provided
            close_time: data.t,
            first_id: 0, // Not provided
            last_id: 0, // Not provided
            count: 0, // Not provided
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_parse_order_book_update() {
        let message = WebSocketMessage {
            c: "spot@public.depth.v3.api@BTCUSDT".to_string(),
            s: "BTCUSDT".to_string(),
            d: json!({
                "s": "BTCUSDT",
                "t": 1622185591123,
                "v": 12345,
                "b": [
                    ["39000.5", "1.25"],
                    ["39000.0", "2.5"]
                ],
                "a": [
                    ["39001.0", "0.5"],
                    ["39001.5", "1.0"]
                ]
            }),
            t: 1622185591123,
        };

        let result = MessageParser::parse_order_book_update(&message);
        assert!(result.is_ok());
        
        let data = result.unwrap();
        assert_eq!(data.s, "BTCUSDT");
        assert_eq!(data.t, 1622185591123);
        assert_eq!(data.v, 12345);
        assert_eq!(data.b.len(), 2);
        assert_eq!(data.a.len(), 2);
        
        let (bids, asks) = MessageParser::order_book_update_to_entries(&data);
        assert_eq!(bids.len(), 2);
        assert_eq!(asks.len(), 2);
        
        assert_eq!(bids[0].price, 39000.5);
        assert_eq!(bids[0].quantity, 1.25);
        assert_eq!(asks[0].price, 39001.0);
        assert_eq!(asks[0].quantity, 0.5);
    }
}
