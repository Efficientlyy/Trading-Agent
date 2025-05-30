use reqwest;
use serde_json::Value;
use std::sync::{Arc, RwLock};
use tracing::{info, error, Level};
use tracing_subscriber::FmtSubscriber;

use market_data_processor::models::order_book::OrderBook;
use market_data_processor::services::order_book_manager::OrderBookManager;
use market_data_processor::models::common::OrderBookUpdate;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;
    
    info!("Testing Order Book via MEXC REST API");
    
    // Create an OrderBookManager
    let order_book_manager = OrderBookManager::new();
    
    // Symbol to fetch
    let symbol = "BTCUSDT";
    
    // MEXC REST API endpoint for order book
    let url = format!("https://api.mexc.com/api/v3/depth?symbol={}&limit=100", symbol);
    
    info!("Fetching order book for {} from REST API", symbol);
    let client = reqwest::Client::new();
    let response = client.get(&url)
        .header("User-Agent", "Mozilla/5.0")
        .send()
        .await?;
    
    if response.status().is_success() {
        let order_book_data: Value = response.json().await?;
        info!("Successfully received order book data via REST API");
        
        // Extract bids and asks
        if let (Some(bids), Some(asks)) = (order_book_data.get("bids"), order_book_data.get("asks")) {
            if let (Some(bids_arr), Some(asks_arr)) = (bids.as_array(), asks.as_array()) {
                info!("Order book contains {} bids and {} asks", bids_arr.len(), asks_arr.len());
                
                // Process top bids and asks
                info!("Top 5 bids:");
                for (i, bid) in bids_arr.iter().take(5).enumerate() {
                    if let (Some(price), Some(qty)) = (bid.get(0), bid.get(1)) {
                        info!("  {}: {} @ {}", i+1, qty, price);
                    }
                }
                
                info!("Top 5 asks:");
                for (i, ask) in asks_arr.iter().take(5).enumerate() {
                    if let (Some(price), Some(qty)) = (ask.get(0), ask.get(1)) {
                        info!("  {}: {} @ {}", i+1, qty, price);
                    }
                }
                
                // Create an OrderBook and populate it
                let mut order_book = OrderBook::new(symbol);
                
                // Add bids
                for bid in bids_arr {
                    if let (Some(price), Some(qty)) = (bid.get(0), bid.get(1)) {
                        if let (Some(price_str), Some(qty_str)) = (price.as_str(), qty.as_str()) {
                            if let (Ok(price_val), Ok(qty_val)) = (price_str.parse::<f64>(), qty_str.parse::<f64>()) {
                                order_book.update_bid(price_val, qty_val);
                            }
                        }
                    }
                }
                
                // Add asks
                for ask in asks_arr {
                    if let (Some(price), Some(qty)) = (ask.get(0), ask.get(1)) {
                        if let (Some(price_str), Some(qty_str)) = (price.as_str(), qty.as_str()) {
                            if let (Ok(price_val), Ok(qty_val)) = (price_str.parse::<f64>(), qty_str.parse::<f64>()) {
                                order_book.update_ask(price_val, qty_val);
                            }
                        }
                    }
                }
                
                // Calculate market metrics
                if let Some(spread) = order_book.spread() {
                    info!("Spread: {}", spread);
                }
                
                if let Some(mid_price) = order_book.mid_price() {
                    info!("Mid price: {}", mid_price);
                }
                
                // Update the OrderBookManager
                if let Err(e) = order_book_manager.process_snapshot(symbol.to_string(), Arc::new(RwLock::new(order_book))) {
                    error!("Failed to process order book snapshot: {}", e);
                }
                
                // Calculate order book imbalance
                if let Some(imbalance) = order_book_manager.calculate_imbalance(symbol, 10) {
                    info!("Order book imbalance (depth 10): {:.4f}", imbalance);
                    
                    if imbalance > 0.2 {
                        info!("Significant buying pressure detected");
                    } else if imbalance < -0.2 {
                        info!("Significant selling pressure detected");
                    }
                }
                
                info!("REST API order book test completed successfully");
                return Ok(());
            }
        }
        
        error!("Failed to parse order book data: {}", order_book_data);
    } else {
        error!("Failed to fetch order book: HTTP {}", response.status());
    }
    
    Ok(())
}
