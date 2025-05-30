use crate::models::{OrderBook, Ticker, Trade, WebSocketMessage};
use crate::utils::{config::Config, metrics, Result};
use futures_util::{SinkExt, StreamExt};
use std::sync::Arc;
use tokio::sync::mpsc::{self, Sender};
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};
use url::Url;

/// MEXC WebSocket client
pub struct MexcWebsocketClient {
    /// Configuration
    config: Arc<Config>,
    
    /// Channel for sending ticker updates
    ticker_tx: Option<Sender<Ticker>>,
    
    /// Channel for sending order book updates
    orderbook_tx: Option<Sender<OrderBook>>,
    
    /// Channel for sending trade updates
    trades_tx: Option<Sender<Trade>>,
}

impl MexcWebsocketClient {
    /// Create a new WebSocket client
    pub fn new(config: Arc<Config>) -> Self {
        Self {
            config,
            ticker_tx: None,
            orderbook_tx: None,
            trades_tx: None,
        }
    }
    
    /// Create channels for data distribution
    pub fn create_channels() -> (
        (Sender<Ticker>, mpsc::Receiver<Ticker>),
        (Sender<OrderBook>, mpsc::Receiver<OrderBook>),
        (Sender<Trade>, mpsc::Receiver<Trade>),
    ) {
        let (ticker_tx, ticker_rx) = mpsc::channel(100);
        let (orderbook_tx, orderbook_rx) = mpsc::channel(100);
        let (trades_tx, trades_rx) = mpsc::channel(100);
        
        ((ticker_tx, ticker_rx), (orderbook_tx, orderbook_rx), (trades_tx, trades_rx))
    }
    
    /// Set ticker channel
    pub fn set_ticker_channel(&mut self, tx: Sender<Ticker>) {
        self.ticker_tx = Some(tx);
    }
    
    /// Set order book channel
    pub fn set_orderbook_channel(&mut self, tx: Sender<OrderBook>) {
        self.orderbook_tx = Some(tx);
    }
    
    /// Set trades channel
    pub fn set_trades_channel(&mut self, tx: Sender<Trade>) {
        self.trades_tx = Some(tx);
    }
    
    /// Connect to WebSocket and subscribe to channels
    pub async fn connect(&mut self, symbol: &str) -> Result<()> {
        let ws_url = Url::parse(&self.config.websocket_url)?;
        
        log::info!("Connecting to WebSocket at {}", ws_url);
        
        let (ws_stream, _) = connect_async(ws_url).await?;
        let (mut write, mut read) = ws_stream.split();
        
        log::info!("WebSocket connected, subscribing to channels for {}", symbol);
        
        // Subscribe to ticker channel
        let ticker_sub = serde_json::json!({
            "method": "SUBSCRIPTION",
            "params": [format!("spot@public.ticker.v3.api@{}", symbol)],
            "id": 1
        });
        
        write.send(Message::Text(ticker_sub.to_string())).await?;
        
        // Subscribe to order book channel
        let orderbook_sub = serde_json::json!({
            "method": "SUBSCRIPTION",
            "params": [format!("spot@public.depth.v3.api@{}", symbol)],
            "id": 2
        });
        
        write.send(Message::Text(orderbook_sub.to_string())).await?;
        
        // Subscribe to trades channel
        let trades_sub = serde_json::json!({
            "method": "SUBSCRIPTION",
            "params": [format!("spot@public.deals.v3.api@{}", symbol)],
            "id": 3
        });
        
        write.send(Message::Text(trades_sub.to_string())).await?;
        
        log::info!("Subscribed to all channels, processing messages");
        
        // Process incoming messages
        while let Some(msg) = read.next().await {
            match msg {
                Ok(Message::Text(text)) => {
                    metrics::record_websocket_message();
                    
                    if let Err(e) = self.process_message(&text).await {
                        log::error!("Error processing WebSocket message: {}", e);
                    }
                }
                Ok(Message::Ping(data)) => {
                    // Respond to ping with pong
                    if let Err(e) = write.send(Message::Pong(data)).await {
                        log::error!("Error sending pong: {}", e);
                        break;
                    }
                }
                Ok(Message::Close(_)) => {
                    log::info!("WebSocket connection closed by server");
                    break;
                }
                Err(e) => {
                    log::error!("WebSocket error: {}", e);
                    metrics::record_websocket_error();
                    break;
                }
                _ => {}
            }
        }
        
        log::info!("WebSocket connection closed, will reconnect");
        metrics::record_websocket_reconnect();
        
        Ok(())
    }
    
    /// Process a WebSocket message
    async fn process_message(&self, text: &str) -> Result<()> {
        let data: serde_json::Value = serde_json::from_str(text)?;
        
        // Check if it's a subscription response
        if data["id"].is_number() {
            log::debug!("Received subscription response: {}", text);
            return Ok(());
        }
        
        // Check if it's a ticker update
        if let Some(ticker_data) = data["d"].as_object() {
            if data["c"].as_str() == Some("spot@public.ticker.v3.api") {
                if let (Some(symbol), Some(price_str), Some(volume_str), Some(high_str), Some(low_str)) = (
                    data["s"].as_str(),
                    ticker_data.get("c").and_then(|v| v.as_str()),
                    ticker_data.get("v").and_then(|v| v.as_str()),
                    ticker_data.get("h").and_then(|v| v.as_str()),
                    ticker_data.get("l").and_then(|v| v.as_str()),
                ) {
                    let price = price_str.parse().unwrap_or(0.0);
                    let volume = volume_str.parse().unwrap_or(0.0);
                    let high = high_str.parse().unwrap_or(0.0);
                    let low = low_str.parse().unwrap_or(0.0);
                    
                    let timestamp = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_millis() as u64;
                        
                    let ticker = Ticker::new(
                        symbol.to_string(),
                        price,
                        volume,
                        high,
                        low,
                        timestamp,
                    );
                    
                    metrics::update_ticker_metrics(symbol, price);
                    
                    if let Some(tx) = &self.ticker_tx {
                        if let Err(e) = tx.send(ticker).await {
                            log::error!("Error sending ticker update: {}", e);
                        }
                    }
                    
                    return Ok(());
                }
            }
        }
        
        // Check if it's an order book update
        if data["c"].as_str() == Some("spot@public.depth.v3.api") {
            if let (Some(symbol), Some(bids_data), Some(asks_data)) = (
                data["s"].as_str(),
                data["d"]["b"].as_array(),
                data["d"]["a"].as_array(),
            ) {
                let timestamp = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as u64;
                    
                let bids = bids_data
                    .iter()
                    .filter_map(|bid| {
                        if let (Some(price_str), Some(qty_str)) = (bid[0].as_str(), bid[1].as_str()) {
                            let price = price_str.parse().ok()?;
                            let quantity = qty_str.parse().ok()?;
                            Some((price, quantity))
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<(f64, f64)>>();
                    
                let asks = asks_data
                    .iter()
                    .filter_map(|ask| {
                        if let (Some(price_str), Some(qty_str)) = (ask[0].as_str(), ask[1].as_str()) {
                            let price = price_str.parse().ok()?;
                            let quantity = qty_str.parse().ok()?;
                            Some((price, quantity))
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<(f64, f64)>>();
                    
                let orderbook = OrderBook::new(
                    symbol.to_string(),
                    bids,
                    asks,
                    timestamp,
                );
                
                metrics::update_orderbook_metrics(symbol, &orderbook.bids, &orderbook.asks);
                
                if let Some(tx) = &self.orderbook_tx {
                    if let Err(e) = tx.send(orderbook).await {
                        log::error!("Error sending order book update: {}", e);
                    }
                }
                
                return Ok(());
            }
        }
        
        // Check if it's a trade update
        if data["c"].as_str() == Some("spot@public.deals.v3.api") {
            if let (Some(symbol), Some(trades_data)) = (
                data["s"].as_str(),
                data["d"].as_array(),
            ) {
                for trade_data in trades_data {
                    if let (Some(id), Some(price_str), Some(qty_str), Some(is_buyer_maker), Some(time)) = (
                        trade_data["i"].as_str(),
                        trade_data["p"].as_str(),
                        trade_data["q"].as_str(),
                        trade_data["m"].as_bool(),
                        trade_data["t"].as_u64(),
                    ) {
                        let price = price_str.parse().unwrap_or(0.0);
                        let quantity = qty_str.parse().unwrap_or(0.0);
                        
                        let trade = Trade::new(
                            id.to_string(),
                            symbol.to_string(),
                            price,
                            quantity,
                            is_buyer_maker,
                            time,
                        );
                        
                        metrics::update_trade_metrics(symbol);
                        
                        if let Some(tx) = &self.trades_tx {
                            if let Err(e) = tx.send(trade).await {
                                log::error!("Error sending trade update: {}", e);
                            }
                        }
                    }
                }
                
                return Ok(());
            }
        }
        
        log::debug!("Unhandled WebSocket message: {}", text);
        Ok(())
    }
}
