use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

use crate::models::order_book::OrderBook;
use crate::models::common::OrderBookUpdate;

/// Maximum acceptable deviation between WebSocket and REST order book data
const MAX_PRICE_DEVIATION_PERCENT: f64 = 0.5; // 0.5%
const MAX_DEPTH_DEVIATION_PERCENT: f64 = 5.0; // 5%

/// Window size for tracking sequence numbers
const SEQUENCE_WINDOW_SIZE: usize = 100;

/// Data consistency validator for order book data
pub struct DataConsistencyValidator {
    /// Track the last N sequence numbers for each symbol to detect missed updates
    sequence_numbers: Arc<RwLock<HashMap<String, VecDeque<u64>>>>,
    
    /// Track the last validation timestamp for each symbol
    last_validation: Arc<RwLock<HashMap<String, Instant>>>,
    
    /// Track the number of inconsistencies detected for each symbol
    inconsistency_count: Arc<RwLock<HashMap<String, u32>>>,
    
    /// Track the number of missed sequence numbers for each symbol
    missed_sequences: Arc<RwLock<HashMap<String, u32>>>,
    
    /// Reference data from REST API for cross-validation
    rest_data: Arc<RwLock<HashMap<String, Arc<RwLock<OrderBook>>>>>,
}

impl DataConsistencyValidator {
    pub fn new() -> Self {
        Self {
            sequence_numbers: Arc::new(RwLock::new(HashMap::new())),
            last_validation: Arc::new(RwLock::new(HashMap::new())),
            inconsistency_count: Arc::new(RwLock::new(HashMap::new())),
            missed_sequences: Arc::new(RwLock::new(HashMap::new())),
            rest_data: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Track a sequence number for a symbol to detect gaps
    pub async fn track_sequence_number(&self, symbol: &str, sequence: u64) -> bool {
        let mut sequences = self.sequence_numbers.write().await;
        
        // Get or create the sequence queue for this symbol
        let sequence_queue = sequences
            .entry(symbol.to_string())
            .or_insert_with(|| VecDeque::with_capacity(SEQUENCE_WINDOW_SIZE));
        
        // Check for sequence gap if we have previous sequences
        let mut gap_detected = false;
        if !sequence_queue.is_empty() {
            let last_sequence = *sequence_queue.back().unwrap();
            
            // Detect gaps (assuming sequences should be consecutive)
            if sequence > last_sequence + 1 {
                let gap_size = sequence - last_sequence - 1;
                warn!(
                    symbol = symbol,
                    last_sequence = last_sequence,
                    current_sequence = sequence,
                    gap_size = gap_size,
                    "Sequence gap detected in order book updates"
                );
                
                // Record the missed sequences
                let mut missed = self.missed_sequences.write().await;
                let count = missed.entry(symbol.to_string()).or_insert(0);
                *count += gap_size as u32;
                
                gap_detected = true;
            } else if sequence < last_sequence {
                // Out of order sequence - could be a replay or duplicate
                warn!(
                    symbol = symbol,
                    last_sequence = last_sequence,
                    current_sequence = sequence,
                    "Out of order sequence detected"
                );
            }
        }
        
        // Add the sequence to the queue
        sequence_queue.push_back(sequence);
        
        // Maintain the window size
        if sequence_queue.len() > SEQUENCE_WINDOW_SIZE {
            sequence_queue.pop_front();
        }
        
        gap_detected
    }
    
    /// Update the REST API reference data for a symbol
    pub async fn update_rest_data(&self, symbol: &str, order_book: Arc<RwLock<OrderBook>>) {
        let mut rest_data = self.rest_data.write().await;
        rest_data.insert(symbol.to_string(), order_book);
        
        // Update the last validation timestamp
        let mut last_validation = self.last_validation.write().await;
        last_validation.insert(symbol.to_string(), Instant::now());
    }
    
    /// Validate WebSocket order book against REST reference data
    pub async fn validate_order_book(&self, symbol: &str, ws_order_book: &OrderBook) -> bool {
        let rest_data = self.rest_data.read().await;
        
        // Check if we have REST data for this symbol
        if let Some(rest_order_book) = rest_data.get(symbol) {
            let rest_book = rest_order_book.read().await;
            
            // Calculate mid price for both data sources
            let ws_mid_price = ws_order_book.mid_price();
            let rest_mid_price = rest_book.mid_price();
            
            if let (Some(ws_mid), Some(rest_mid)) = (ws_mid_price, rest_mid_price) {
                // Calculate the deviation as a percentage
                let deviation_percent = ((ws_mid - rest_mid).abs() / rest_mid) * 100.0;
                
                // Log the validation result
                debug!(
                    symbol = symbol,
                    ws_mid_price = ws_mid,
                    rest_mid_price = rest_mid,
                    deviation_percent = deviation_percent,
                    "Order book validation"
                );
                
                // Check if the deviation exceeds the threshold
                if deviation_percent > MAX_PRICE_DEVIATION_PERCENT {
                    warn!(
                        symbol = symbol,
                        ws_mid_price = ws_mid,
                        rest_mid_price = rest_mid,
                        deviation_percent = deviation_percent,
                        threshold = MAX_PRICE_DEVIATION_PERCENT,
                        "Order book price deviation exceeds threshold"
                    );
                    
                    // Record the inconsistency
                    let mut inconsistencies = self.inconsistency_count.write().await;
                    let count = inconsistencies.entry(symbol.to_string()).or_insert(0);
                    *count += 1;
                    
                    return false;
                }
                
                // Validate depth (number of price levels)
                let ws_depth = ws_order_book.bids.len() + ws_order_book.asks.len();
                let rest_depth = rest_book.bids.len() + rest_book.asks.len();
                
                // Only compare if both have some depth
                if ws_depth > 0 && rest_depth > 0 {
                    let depth_ratio = ws_depth as f64 / rest_depth as f64;
                    let depth_deviation_percent = ((depth_ratio - 1.0).abs()) * 100.0;
                    
                    if depth_deviation_percent > MAX_DEPTH_DEVIATION_PERCENT {
                        warn!(
                            symbol = symbol,
                            ws_depth = ws_depth,
                            rest_depth = rest_depth,
                            depth_deviation_percent = depth_deviation_percent,
                            threshold = MAX_DEPTH_DEVIATION_PERCENT,
                            "Order book depth deviation exceeds threshold"
                        );
                        
                        // Record the inconsistency
                        let mut inconsistencies = self.inconsistency_count.write().await;
                        let count = inconsistencies.entry(symbol.to_string()).or_insert(0);
                        *count += 1;
                        
                        return false;
                    }
                }
                
                // Both checks passed
                return true;
            }
        }
        
        // No REST data to validate against
        false
    }
    
    /// Check if a REST API validation is needed (based on time since last validation)
    pub async fn needs_validation(&self, symbol: &str, interval: Duration) -> bool {
        let last_validation = self.last_validation.read().await;
        
        if let Some(last_time) = last_validation.get(symbol) {
            last_time.elapsed() > interval
        } else {
            // No validation has been done yet
            true
        }
    }
    
    /// Get validation metrics for a symbol
    pub async fn get_metrics(&self, symbol: &str) -> serde_json::Value {
        let missed = self.missed_sequences.read().await;
        let inconsistencies = self.inconsistency_count.read().await;
        let last_validation = self.last_validation.read().await;
        
        let missed_count = missed.get(symbol).cloned().unwrap_or(0);
        let inconsistency_count = inconsistencies.get(symbol).cloned().unwrap_or(0);
        let last_validation_time = last_validation.get(symbol)
            .map(|t| t.elapsed().as_secs())
            .unwrap_or(u64::MAX);
        
        serde_json::json!({
            "symbol": symbol,
            "missed_updates": missed_count,
            "inconsistencies": inconsistency_count,
            "seconds_since_last_validation": last_validation_time,
            "validation_status": if inconsistency_count > 0 { "inconsistent" } else { "consistent" }
        })
    }
    
    /// Reconcile WebSocket and REST data when inconsistencies are detected
    pub async fn reconcile(&self, symbol: &str, ws_order_book: &mut OrderBook) -> bool {
        let rest_data = self.rest_data.read().await;
        
        if let Some(rest_order_book) = rest_data.get(symbol) {
            let rest_book = rest_order_book.read().await;
            
            // Strategy: take the mid point from both sources and selectively apply price levels
            if let (Some(ws_mid), Some(rest_mid)) = (ws_order_book.mid_price(), rest_book.mid_price()) {
                // If mid prices are very different, use REST data as the source of truth
                let deviation_percent = ((ws_mid - rest_mid).abs() / rest_mid) * 100.0;
                
                if deviation_percent > MAX_PRICE_DEVIATION_PERCENT * 2.0 {
                    // Major inconsistency - replace the entire book
                    info!(
                        symbol = symbol,
                        ws_mid = ws_mid,
                        rest_mid = rest_mid,
                        deviation = deviation_percent,
                        "Major inconsistency detected, replacing WebSocket order book with REST data"
                    );
                    
                    // Clear the WebSocket order book
                    ws_order_book.bids.clear();
                    ws_order_book.asks.clear();
                    
                    // Copy price levels from REST data
                    for (price, qty) in &rest_book.bids {
                        ws_order_book.update_bid(*price, *qty);
                    }
                    
                    for (price, qty) in &rest_book.asks {
                        ws_order_book.update_ask(*price, *qty);
                    }
                    
                    return true;
                } else {
                    // Minor inconsistency - selectively update price levels
                    debug!(
                        symbol = symbol,
                        "Performing selective reconciliation of order book data"
                    );
                    
                    // Identify price levels that differ significantly
                    for (price, rest_qty) in &rest_book.bids {
                        if let Some(ws_qty) = ws_order_book.bids.get(price) {
                            let qty_diff_percent = ((ws_qty - rest_qty).abs() / rest_qty) * 100.0;
                            if qty_diff_percent > 20.0 {  // 20% threshold for quantity difference
                                ws_order_book.update_bid(*price, *rest_qty);
                            }
                        } else {
                            // Add missing price level
                            ws_order_book.update_bid(*price, *rest_qty);
                        }
                    }
                    
                    for (price, rest_qty) in &rest_book.asks {
                        if let Some(ws_qty) = ws_order_book.asks.get(price) {
                            let qty_diff_percent = ((ws_qty - rest_qty).abs() / rest_qty) * 100.0;
                            if qty_diff_percent > 20.0 {  // 20% threshold for quantity difference
                                ws_order_book.update_ask(*price, *rest_qty);
                            }
                        } else {
                            // Add missing price level
                            ws_order_book.update_ask(*price, *rest_qty);
                        }
                    }
                    
                    return true;
                }
            }
        }
        
        false
    }
}
