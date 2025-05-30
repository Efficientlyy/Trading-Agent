#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::websocket::OrderBookUpdateData;

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
