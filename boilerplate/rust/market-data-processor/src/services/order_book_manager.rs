use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tokio::sync::broadcast;

use crate::models::{
    order_book::OrderBook,
    websocket::OrderBookUpdateData,
};
use crate::services::message_parser::MessageParser;
use tracing::{debug, info, warn};

#[derive(Debug, Clone)]
pub struct OrderBookUpdate {
    pub symbol: String,
    pub is_snapshot: bool,
    pub order_book: Arc<OrderBook>,
}

pub struct OrderBookManager {
    order_books: Arc<RwLock<HashMap<String, Arc<RwLock<OrderBook>>>>>,
    update_sender: broadcast::Sender<OrderBookUpdate>,
}

impl OrderBookManager {
    pub fn new() -> Self {
        let (update_sender, _) = broadcast::channel(1000);
        Self {
            order_books: Arc::new(RwLock::new(HashMap::new())),
            update_sender,
        }
    }
    
    pub fn get_update_receiver(&self) -> broadcast::Receiver<OrderBookUpdate> {
        self.update_sender.subscribe()
    }
    
    pub fn get_order_book(&self, symbol: &str) -> Option<Arc<RwLock<OrderBook>>> {
        let order_books = self.order_books.read().unwrap();
        order_books.get(symbol).cloned()
    }
    
    pub fn process_snapshot(&self, symbol: &str, data: &OrderBookUpdateData) {
        info!("Processing order book snapshot for {}", symbol);
        let mut order_books = self.order_books.write().unwrap();
        let order_book = order_books.entry(symbol.to_string())
            .or_insert_with(|| Arc::new(RwLock::new(OrderBook::new(symbol))))
            .clone();
        
        let mut book = order_book.write().unwrap();
        book.last_update_id = data.v;
        book.timestamp = data.t;
        
        // Clear existing entries
        book.bids.clear();
        book.asks.clear();
        
        // Add new entries
        let (bids, asks) = MessageParser::order_book_update_to_entries(data);
        for entry in bids {
            book.update_bid(entry.price, entry.quantity);
        }
        
        for entry in asks {
            book.update_ask(entry.price, entry.quantity);
        }
        
        debug!("Order book snapshot processed: {} entries (bids: {}, asks: {})", 
            book.bids.len() + book.asks.len(), book.bids.len(), book.asks.len());
        
        // Notify subscribers
        let _ = self.update_sender.send(OrderBookUpdate {
            symbol: symbol.to_string(),
            is_snapshot: true,
            order_book: Arc::new(book.clone()),
        });
    }
    
    pub fn process_update(&self, symbol: &str, data: &OrderBookUpdateData) -> bool {
        let order_books = self.order_books.read().unwrap();
        let order_book = match order_books.get(symbol) {
            Some(book) => book.clone(),
            None => {
                warn!("Received update for unknown symbol {}, ignoring", symbol);
                return false; // No snapshot received yet
            }
        };
        
        let mut book = order_book.write().unwrap();
        
        // Verify sequence
        if data.v <= book.last_update_id {
            debug!("Outdated update received (current: {}, received: {}), ignoring", 
                book.last_update_id, data.v);
            return false; // Outdated update
        }
        
        book.last_update_id = data.v;
        book.timestamp = data.t;
        
        // Process updates
        let (bids, asks) = MessageParser::order_book_update_to_entries(data);
        for entry in bids {
            book.update_bid(entry.price, entry.quantity);
        }
        
        for entry in asks {
            book.update_ask(entry.price, entry.quantity);
        }
        
        // Notify subscribers
        let _ = self.update_sender.send(OrderBookUpdate {
            symbol: symbol.to_string(),
            is_snapshot: false,
            order_book: Arc::new(book.clone()),
        });
        
        true
    }
    
    // Additional methods for order book analysis
    pub fn get_market_depth(&self, symbol: &str, levels: usize) -> Option<(Vec<(f64, f64)>, Vec<(f64, f64)>)> {
        let order_books = self.order_books.read().unwrap();
        let order_book = match order_books.get(symbol) {
            Some(book) => book.clone(),
            None => return None,
        };
        
        let book = order_book.read().unwrap();
        
        let bids: Vec<(f64, f64)> = book.bids.iter()
            .rev()
            .take(levels)
            .map(|(k, v)| (*k, *v))
            .collect();
            
        let asks: Vec<(f64, f64)> = book.asks.iter()
            .take(levels)
            .map(|(k, v)| (*k, *v))
            .collect();
            
        Some((bids, asks))
    }
    
    pub fn calculate_imbalance(&self, symbol: &str, depth: usize) -> Option<f64> {
        if let Some((bids, asks)) = self.get_market_depth(symbol, depth) {
            let bid_volume: f64 = bids.iter().map(|(_, qty)| qty).sum();
            let ask_volume: f64 = asks.iter().map(|(_, qty)| qty).sum();
            
            if bid_volume + ask_volume > 0.0 {
                return Some((bid_volume - ask_volume) / (bid_volume + ask_volume));
            }
        }
        
        None
    }
    
    // Get all symbols with active order books
    pub fn get_active_symbols(&self) -> Vec<String> {
        let order_books = self.order_books.read().unwrap();
        order_books.keys().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_book_creation() {
        let manager = OrderBookManager::new();
        assert!(manager.get_order_book("BTCUSDT").is_none());
    }

    #[test]
    fn test_process_snapshot() {
        let manager = OrderBookManager::new();
        
        let data = OrderBookUpdateData {
            s: "BTCUSDT".to_string(),
            t: 1622185591123,
            v: 12345,
            b: vec![
                ["39000.5".to_string(), "1.25".to_string()],
                ["39000.0".to_string(), "2.5".to_string()],
            ],
            a: vec![
                ["39001.0".to_string(), "0.5".to_string()],
                ["39001.5".to_string(), "1.0".to_string()],
            ],
        };
        
        // Process snapshot
        manager.process_snapshot("BTCUSDT", &data);
        
        // Check that order book was created
        let order_book = manager.get_order_book("BTCUSDT");
        assert!(order_book.is_some());
        
        let book = order_book.unwrap().read().unwrap();
        assert_eq!(book.last_update_id, 12345);
        assert_eq!(book.bids.len(), 2);
        assert_eq!(book.asks.len(), 2);
        
        // Check bid prices
        assert_eq!(book.bids.get(&39000.5), Some(&1.25));
        assert_eq!(book.bids.get(&39000.0), Some(&2.5));
        
        // Check ask prices
        assert_eq!(book.asks.get(&39001.0), Some(&0.5));
        assert_eq!(book.asks.get(&39001.5), Some(&1.0));
    }
    
    #[test]
    fn test_process_update() {
        let manager = OrderBookManager::new();
        
        // First create a snapshot
        let snapshot = OrderBookUpdateData {
            s: "BTCUSDT".to_string(),
            t: 1622185591123,
            v: 12345,
            b: vec![
                ["39000.5".to_string(), "1.25".to_string()],
                ["39000.0".to_string(), "2.5".to_string()],
            ],
            a: vec![
                ["39001.0".to_string(), "0.5".to_string()],
                ["39001.5".to_string(), "1.0".to_string()],
            ],
        };
        
        manager.process_snapshot("BTCUSDT", &snapshot);
        
        // Then process an update
        let update = OrderBookUpdateData {
            s: "BTCUSDT".to_string(),
            t: 1622185591124,
            v: 12346, // Incremented sequence number
            b: vec![
                ["39000.5".to_string(), "1.5".to_string()], // Modified bid
                ["38999.5".to_string(), "0.75".to_string()], // New bid
            ],
            a: vec![
                ["39001.0".to_string(), "0.0".to_string()], // Remove ask
                ["39002.0".to_string(), "1.2".to_string()], // New ask
            ],
        };
        
        let result = manager.process_update("BTCUSDT", &update);
        assert!(result);
        
        // Verify the updates were applied
        let order_book = manager.get_order_book("BTCUSDT").unwrap();
        let book = order_book.read().unwrap();
        
        // Check updated sequence
        assert_eq!(book.last_update_id, 12346);
        
        // Check bids
        assert_eq!(book.bids.len(), 3); // One new bid added
        assert_eq!(book.bids.get(&39000.5), Some(&1.5)); // Modified
        assert_eq!(book.bids.get(&38999.5), Some(&0.75)); // Added
        
        // Check asks
        assert_eq!(book.asks.len(), 2); // One removed, one added
        assert_eq!(book.asks.get(&39001.0), None); // Removed
        assert_eq!(book.asks.get(&39002.0), Some(&1.2)); // Added
    }
    
    #[test]
    fn test_reject_outdated_update() {
        let manager = OrderBookManager::new();
        
        // Create a snapshot
        let snapshot = OrderBookUpdateData {
            s: "BTCUSDT".to_string(),
            t: 1622185591123,
            v: 12345,
            b: vec![
                ["39000.5".to_string(), "1.25".to_string()],
            ],
            a: vec![
                ["39001.0".to_string(), "0.5".to_string()],
            ],
        };
        
        manager.process_snapshot("BTCUSDT", &snapshot);
        
        // Try to process an outdated update
        let outdated = OrderBookUpdateData {
            s: "BTCUSDT".to_string(),
            t: 1622185591122, // Older timestamp
            v: 12344, // Older sequence
            b: vec![
                ["39000.0".to_string(), "1.0".to_string()],
            ],
            a: vec![
                ["39002.0".to_string(), "0.8".to_string()],
            ],
        };
        
        let result = manager.process_update("BTCUSDT", &outdated);
        assert!(!result); // Should reject the update
        
        // Verify the original data is unchanged
        let order_book = manager.get_order_book("BTCUSDT").unwrap();
        let book = order_book.read().unwrap();
        
        assert_eq!(book.last_update_id, 12345); // Still original
        assert_eq!(book.bids.len(), 1);
        assert_eq!(book.asks.len(), 1);
        assert_eq!(book.bids.get(&39000.5), Some(&1.25));
        assert_eq!(book.asks.get(&39001.0), Some(&0.5));
    }
    
    #[test]
    fn test_unknown_symbol_update() {
        let manager = OrderBookManager::new();
        
        let update = OrderBookUpdateData {
            s: "UNKNOWN".to_string(),
            t: 1622185591123,
            v: 12345,
            b: vec![["100.0".to_string(), "1.0".to_string()]],
            a: vec![["101.0".to_string(), "1.0".to_string()]],
        };
        
        let result = manager.process_update("UNKNOWN", &update);
        assert!(!result); // Should reject the update since no snapshot exists
    }
    
    #[test]
    fn test_market_depth() {
        let manager = OrderBookManager::new();
        
        // Create an order book with multiple entries
        let snapshot = OrderBookUpdateData {
            s: "BTCUSDT".to_string(),
            t: 1622185591123,
            v: 12345,
            b: vec![
                ["39000.5".to_string(), "1.25".to_string()],
                ["39000.0".to_string(), "2.5".to_string()],
                ["38999.5".to_string(), "3.0".to_string()],
                ["38999.0".to_string(), "4.0".to_string()],
                ["38998.5".to_string(), "5.0".to_string()],
            ],
            a: vec![
                ["39001.0".to_string(), "0.5".to_string()],
                ["39001.5".to_string(), "1.0".to_string()],
                ["39002.0".to_string(), "1.5".to_string()],
                ["39002.5".to_string(), "2.0".to_string()],
                ["39003.0".to_string(), "2.5".to_string()],
            ],
        };
        
        manager.process_snapshot("BTCUSDT", &snapshot);
        
        // Test getting market depth with limited level
        let depth = manager.get_market_depth("BTCUSDT", 3);
        assert!(depth.is_some());
        
        let (bids, asks) = depth.unwrap();
        assert_eq!(bids.len(), 3);
        assert_eq!(asks.len(), 3);
        
        // Bids should be in descending order
        assert_eq!(bids[0].0, 39000.5); // Highest bid
        assert_eq!(bids[1].0, 39000.0);
        assert_eq!(bids[2].0, 38999.5);
        
        // Asks should be in ascending order
        assert_eq!(asks[0].0, 39001.0); // Lowest ask
        assert_eq!(asks[1].0, 39001.5);
        assert_eq!(asks[2].0, 39002.0);
    }
    
    #[test]
    fn test_order_book_imbalance() {
        let manager = OrderBookManager::new();
        
        // Create an order book with imbalance
        let snapshot = OrderBookUpdateData {
            s: "BTCUSDT".to_string(),
            t: 1622185591123,
            v: 12345,
            b: vec![
                ["39000.5".to_string(), "10.0".to_string()], // Total bid volume: 25.0
                ["39000.0".to_string(), "15.0".to_string()],
            ],
            a: vec![
                ["39001.0".to_string(), "5.0".to_string()], // Total ask volume: 15.0
                ["39001.5".to_string(), "10.0".to_string()],
            ],
        };
        
        manager.process_snapshot("BTCUSDT", &snapshot);
        
        // Calculate imbalance
        let imbalance = manager.calculate_imbalance("BTCUSDT", 2);
        assert!(imbalance.is_some());
        
        // Expected imbalance: (25 - 15) / (25 + 15) = 10 / 40 = 0.25
        assert_eq!(imbalance.unwrap(), 0.25);
    }
}
