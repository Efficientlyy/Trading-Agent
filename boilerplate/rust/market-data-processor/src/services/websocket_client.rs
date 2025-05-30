use std::sync::Arc;
use tokio::sync::{mpsc, watch};
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};
use futures::{SinkExt, StreamExt};
use serde_json::{json, Value};
use url::Url;
use tracing::{info, error, warn, debug};

use crate::models::websocket::{MexcSubscriptionRequest, MexcSubscriptionResponse};
use crate::utils::config::Config;

pub struct WebSocketClient {
    url: String,
    message_sender: mpsc::Sender<String>,
    shutdown: watch::Receiver<bool>,
    subscriptions: Vec<String>,
}

impl WebSocketClient {
    pub fn new(
        url: String,
        message_sender: mpsc::Sender<String>,
        shutdown: watch::Receiver<bool>,
    ) -> Self {
        Self {
            url,
            message_sender,
            shutdown,
            subscriptions: Vec::new(),
        }
    }
    
    pub fn from_config(
        config: &Config,
        message_sender: mpsc::Sender<String>,
        shutdown: watch::Receiver<bool>,
    ) -> Self {
        Self::new(
            config.mexc_ws_url.clone(),
            message_sender,
            shutdown,
        )
    }
    
    pub fn add_subscription(&mut self, channel: &str, symbol: &str) {
        let subscription = format!("{}@{}", channel, symbol);
        self.subscriptions.push(subscription);
    }
    
    pub fn add_depth_subscription(&mut self, symbol: &str) {
        // Subscribe to order book snapshot and updates
        self.add_subscription("spot@public.depth.v3.api", symbol);
        self.add_subscription("spot@public.limit.depth.v3.api", symbol);
    }
    
    pub fn add_trade_subscription(&mut self, symbol: &str) {
        // Subscribe to trade updates
        self.add_subscription("spot@public.deals.v3.api", symbol);
    }
    
    pub fn add_ticker_subscription(&mut self, symbol: &str) {
        // Subscribe to ticker updates
        self.add_subscription("spot@public.ticker.v3.api", symbol);
    }
    
    pub fn add_kline_subscription(&mut self, symbol: &str, interval: &str) {
        // Subscribe to kline/candlestick updates
        self.add_subscription(&format!("spot@public.kline.v3.api@{}", interval), symbol);
    }
    
    pub async fn connect_and_subscribe(&self) -> Result<(), Box<dyn std::error::Error>> {
        info!("Connecting to MEXC WebSocket at {}", self.url);
        let url = Url::parse(&self.url)?;
        let (ws_stream, _) = connect_async(url).await?;
        info!("WebSocket connection established");
        
        let (mut write, mut read) = ws_stream.split();
        
        // Send subscription requests
        for subscription in &self.subscriptions {
            let (channel, symbol) = if let Some(idx) = subscription.find('@') {
                (&subscription[..idx], &subscription[(idx + 1)..])
            } else {
                (subscription.as_str(), "")
            };
            
            debug!("Subscribing to channel {} for symbol {}", channel, symbol);
            
            let request = MexcSubscriptionRequest {
                method: "SUBSCRIPTION".to_string(),
                params: vec![subscription.clone()],
            };
            
            let request_json = serde_json::to_string(&request)?;
            write.send(Message::Text(request_json)).await?;
        }
        
        // Tokio task for receiving messages
        let message_sender = self.message_sender.clone();
        let mut shutdown = self.shutdown.clone();
        
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = shutdown.changed() => {
                        if *shutdown.borrow() {
                            info!("Shutdown signal received, closing WebSocket connection");
                            let _ = write.send(Message::Close(None)).await;
                            break;
                        }
                    }
                    Some(msg) = read.next() => {
                        match msg {
                            Ok(Message::Text(text)) => {
                                // Handle different message types
                                if let Ok(value) = serde_json::from_str::<Value>(&text) {
                                    if let Some(c) = value.get("c") {
                                        // This is a data message
                                        if let Err(e) = message_sender.send(text).await {
                                            error!("Failed to forward WebSocket message: {}", e);
                                        }
                                    } else if value.get("code").is_some() {
                                        // This is a subscription response
                                        if let Ok(response) = serde_json::from_value::<MexcSubscriptionResponse>(value) {
                                            if response.code == 0 {
                                                debug!("Subscription successful: {}", response.msg);
                                            } else {
                                                warn!("Subscription failed: {} (code: {})", response.msg, response.code);
                                            }
                                        }
                                    }
                                } else {
                                    warn!("Received non-JSON message: {}", text);
                                }
                            }
                            Ok(Message::Ping(data)) => {
                                // Respond to ping with pong
                                debug!("Received ping, sending pong");
                                if let Err(e) = write.send(Message::Pong(data)).await {
                                    error!("Failed to send pong: {}", e);
                                }
                            }
                            Ok(Message::Close(_)) => {
                                info!("WebSocket connection closed by server");
                                break;
                            }
                            Err(e) => {
                                error!("WebSocket error: {}", e);
                                break;
                            }
                            _ => {} // Ignore other message types
                        }
                    }
                }
            }
            
            // Attempt reconnection after delay
            info!("Attempting to reconnect WebSocket in 5 seconds");
        });
        
        Ok(())
    }
}

// Helper function to create a WebSocket client for paper trading
pub async fn create_paper_trading_client(
    config: &Config,
    symbols: Vec<String>,
    message_sender: mpsc::Sender<String>,
    shutdown: watch::Receiver<bool>,
) -> Result<WebSocketClient, Box<dyn std::error::Error>> {
    let mut client = WebSocketClient::from_config(config, message_sender, shutdown);
    
    for symbol in symbols {
        // Add all necessary subscriptions for each symbol
        client.add_depth_subscription(&symbol);
        client.add_trade_subscription(&symbol);
        client.add_ticker_subscription(&symbol);
        client.add_kline_subscription(&symbol, "1m");
    }
    
    Ok(client)
}
