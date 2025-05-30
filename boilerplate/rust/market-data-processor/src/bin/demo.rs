use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, watch};
use tracing::{info, error, Level};
use tracing_subscriber::FmtSubscriber;

use market_data_processor::services::{
    websocket_client::WebSocketClient,
    market_data_service::MarketDataService,
    order_book_manager::OrderBookManager,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;
    
    info!("Starting MEXC Market Data Processor Demo");
    
    // Symbol to monitor
    let symbol = "BTCUSDT";
    
    // Create channels
    let (message_tx, message_rx) = mpsc::channel(10000);
    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    
    // Create MarketDataService
    let market_data_service = Arc::new(MarketDataService::new(message_rx, shutdown_rx.clone()));
    let order_book_manager = market_data_service.get_order_book_manager();
    
    // Create WebSocket client
    let mut ws_client = WebSocketClient::new(
        "wss://wbs.mexc.com/ws".to_string(),
        message_tx,
        shutdown_rx.clone(),
    );
    
    // Add subscriptions
    ws_client.add_depth_subscription(symbol);
    ws_client.add_trade_subscription(symbol);
    ws_client.add_ticker_subscription(symbol);
    
    // Connect to WebSocket
    info!("Connecting to MEXC WebSocket...");
    match ws_client.connect_and_subscribe().await {
        Ok(_) => info!("Successfully connected to MEXC WebSocket"),
        Err(e) => {
            error!("Failed to connect to MEXC WebSocket: {}", e);
            return Err(e);
        }
    }
    
    // Start market data processing
    let market_data_handle = tokio::spawn(async move {
        market_data_service.run().await;
    });
    
    // Wait for some data to be received
    info!("Waiting for market data...");
    let mut ticker_received = false;
    let mut order_book_received = false;
    let mut consecutive_empty_cycles = 0;
    let max_empty_cycles = 30; // Maximum number of empty cycles before timeout
    
    while consecutive_empty_cycles < max_empty_cycles {
        // Check if ticker has been received
        if !ticker_received {
            let ticker = market_data_service.get_ticker(symbol);
            if let Some(ticker) = ticker {
                info!("Received ticker for {}: Last price = {}, 24h volume = {}", 
                    symbol, ticker.last_price, ticker.volume);
                ticker_received = true;
            }
        }
        
        // Check if order book has been received
        if !order_book_received {
            let order_book = order_book_manager.get_order_book(symbol);
            if let Some(order_book) = order_book {
                let book = order_book.read().unwrap();
                if !book.bids.is_empty() && !book.asks.is_empty() {
                    info!("Received order book for {}: {} bids, {} asks", 
                        symbol, book.bids.len(), book.asks.len());
                    
                    // Display top 5 bids and asks
                    info!("Top 5 bids:");
                    for (i, (price, qty)) in book.bids.iter().rev().take(5).enumerate() {
                        info!("  {}: {} @ {}", i + 1, qty, price);
                    }
                    
                    info!("Top 5 asks:");
                    for (i, (price, qty)) in book.asks.iter().take(5).enumerate() {
                        info!("  {}: {} @ {}", i + 1, qty, price);
                    }
                    
                    if let Some(spread) = book.spread() {
                        info!("Spread: {}", spread);
                    }
                    
                    order_book_received = true;
                }
            }
        }
        
        // If both ticker and order book received, break the loop
        if ticker_received && order_book_received {
            info!("Successfully received both ticker and order book data");
            break;
        }
        
        // If neither were received, increment the empty cycle counter
        if !ticker_received && !order_book_received {
            consecutive_empty_cycles += 1;
        } else {
            consecutive_empty_cycles = 0; // Reset if we received some data
        }
        
        // Wait a bit before checking again
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
    
    if consecutive_empty_cycles >= max_empty_cycles {
        error!("Timed out waiting for market data");
    }
    
    // Get recent trades
    let trades = market_data_service.get_recent_trades(symbol, 5);
    if !trades.is_empty() {
        info!("Recent trades for {}:", symbol);
        for (i, trade) in trades.iter().enumerate() {
            info!("  {}: {} {} at {}", 
                i + 1, 
                if trade.is_buyer_maker { "SELL" } else { "BUY" }, 
                trade.quantity, 
                trade.price);
        }
    } else {
        info!("No trades received yet");
    }
    
    // Show mid price if available
    if let Some(order_book) = order_book_manager.get_order_book(symbol) {
        let book = order_book.read().unwrap();
        if let Some(mid_price) = book.mid_price() {
            info!("Current mid price: {}", mid_price);
        }
    }
    
    // Calculate and display order book imbalance
    if let Some(imbalance) = order_book_manager.calculate_imbalance(symbol, 10) {
        info!("Order book imbalance (depth 10): {:.2}%", imbalance * 100.0);
    }
    
    // Demonstrate that we have successfully connected to MEXC and processed data
    if ticker_received || order_book_received {
        info!("Demo completed successfully - Real-time market data received and processed");
    } else {
        error!("Demo failed - No market data received");
    }
    
    // Initiate shutdown
    info!("Shutting down...");
    shutdown_tx.send(true)?;
    
    // Wait for tasks to complete
    market_data_handle.await?;
    
    info!("Demo finished");
    Ok(())
}
