use futures::{SinkExt, StreamExt};
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, watch, RwLock};
use tokio::time::sleep;
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};
use tracing::{debug, error, info, warn};
use url::Url;

use crate::models::websocket::{MexcSubscriptionRequest, MexcSubscriptionResponse};
use crate::utils::config::Config;
use crate::utils::error::Error;

// Connection health score range: 0.0 (unhealthy) to 1.0 (perfect health)
const HEALTH_SCORE_THRESHOLD: f64 = 0.6;
const MAX_RECONNECT_DELAY_MS: u64 = 30000; // 30 seconds
const INITIAL_RECONNECT_DELAY_MS: u64 = 1000; // 1 second
const HEALTH_WINDOW_SIZE: usize = 100; // Number of events to track for health calculation
const PING_INTERVAL_MS: u64 = 15000; // 15 seconds

/// WebSocket client with enhanced error handling, health monitoring, and circuit breaker pattern
pub struct EnhancedWebSocketClient {
    url: String,
    api_key: Option<String>,
    api_secret: Option<String>,
    message_sender: mpsc::Sender<String>,
    shutdown: watch::Receiver<bool>,
    subscriptions: Vec<String>,
    
    // Health monitoring
    health_score: Arc<RwLock<f64>>,
    message_latencies: Arc<RwLock<VecDeque<u64>>>, // in milliseconds
    error_count: Arc<RwLock<u32>>,
    last_successful_message: Arc<RwLock<Instant>>,
    
    // Circuit breaker
    using_fallback: Arc<RwLock<bool>>,
    reconnect_attempt: Arc<RwLock<u32>>,
    last_reconnect_time: Arc<RwLock<Instant>>,
    
    // Metrics
    received_message_count: Arc<RwLock<u64>>,
    sent_message_count: Arc<RwLock<u64>>,
    disconnect_count: Arc<RwLock<u32>>,
    
    // Connection settings
    reconnect_backoff_enabled: bool,
    circuit_breaker_enabled: bool,
    max_reconnect_attempts: u32,
}

impl EnhancedWebSocketClient {
    pub fn new(
        url: String,
        message_sender: mpsc::Sender<String>,
        shutdown: watch::Receiver<bool>,
    ) -> Self {
        Self {
            url,
            api_key: None,
            api_secret: None,
            message_sender,
            shutdown,
            subscriptions: Vec::new(),
            health_score: Arc::new(RwLock::new(1.0)), // Start with perfect health
            message_latencies: Arc::new(RwLock::new(VecDeque::with_capacity(HEALTH_WINDOW_SIZE))),
            error_count: Arc::new(RwLock::new(0)),
            last_successful_message: Arc::new(RwLock::new(Instant::now())),
            using_fallback: Arc::new(RwLock::new(false)),
            reconnect_attempt: Arc::new(RwLock::new(0)),
            last_reconnect_time: Arc::new(RwLock::new(Instant::now())),
            received_message_count: Arc::new(RwLock::new(0)),
            sent_message_count: Arc::new(RwLock::new(0)),
            disconnect_count: Arc::new(RwLock::new(0)),
            reconnect_backoff_enabled: true,
            circuit_breaker_enabled: true,
            max_reconnect_attempts: 10,
        }
    }
    
    pub fn from_config(
        config: &Config,
        message_sender: mpsc::Sender<String>,
        shutdown: watch::Receiver<bool>,
    ) -> Self {
        let mut client = Self::new(
            config.mexc_ws_url.clone(),
            message_sender,
            shutdown,
        );
        
        // Set API credentials if available
        if !config.mexc_api_key.is_empty() && !config.mexc_api_secret.is_empty() {
            client.set_api_credentials(
                config.mexc_api_key.clone(),
                config.mexc_api_secret.clone(),
            );
        }
        
        // Configure behavior from config
        client.reconnect_backoff_enabled = config.ws_reconnect_backoff_enabled;
        client.circuit_breaker_enabled = config.ws_circuit_breaker_enabled;
        client.max_reconnect_attempts = config.ws_max_reconnect_attempts;
        
        client
    }
    
    pub fn set_api_credentials(&mut self, api_key: String, api_secret: String) {
        self.api_key = Some(api_key);
        self.api_secret = Some(api_secret);
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
        self.add_subscription("spot@public.deals.v3.api", symbol);
    }
    
    pub fn add_ticker_subscription(&mut self, symbol: &str) {
        self.add_subscription("spot@public.ticker.v3.api", symbol);
    }
    
    pub fn add_kline_subscription(&mut self, symbol: &str, interval: &str) {
        self.add_subscription(&format!("spot@public.kline.v3.api@{}", interval), symbol);
    }
    
    /// Calculate the current connection health score (0.0 to 1.0)
    async fn calculate_health_score(&self) -> f64 {
        let error_count = *self.error_count.read().await;
        let latencies = self.message_latencies.read().await;
        let time_since_last_message = self.last_successful_message.read().await.elapsed().as_millis() as f64;
        
        // Calculate average latency
        let avg_latency = if latencies.is_empty() {
            0.0
        } else {
            latencies.iter().sum::<u64>() as f64 / latencies.len() as f64
        };
        
        // Normalize factors to 0.0-1.0 range
        // 1. Error penalty (0.0 = many errors, 1.0 = no errors)
        let error_factor = (1.0 / (1.0 + error_count as f64)).min(1.0);
        
        // 2. Latency penalty (0.0 = high latency, 1.0 = low latency)
        // Assume latency over 2000ms is very poor
        let latency_factor = (1.0 - (avg_latency / 2000.0)).max(0.0);
        
        // 3. Recent message penalty (0.0 = long time since last message, 1.0 = recent message)
        // Consider anything over 60 seconds to be a problem
        let recency_factor = (1.0 - (time_since_last_message / 60000.0)).max(0.0);
        
        // Calculate overall health score with weighted factors
        let health_score = (
            (error_factor * 0.4) + 
            (latency_factor * 0.3) + 
            (recency_factor * 0.3)
        ).min(1.0).max(0.0);
        
        // Update the stored health score
        *self.health_score.write().await = health_score;
        
        health_score
    }
    
    /// Determine if the circuit breaker should be tripped
    async fn should_use_fallback(&self) -> bool {
        if !self.circuit_breaker_enabled {
            return false;
        }
        
        let health_score = self.calculate_health_score().await;
        let reconnect_attempts = *self.reconnect_attempt.read().await;
        
        // Trip the circuit breaker if:
        // 1. Health score is below threshold, or
        // 2. Too many reconnect attempts
        health_score < HEALTH_SCORE_THRESHOLD || reconnect_attempts > self.max_reconnect_attempts
    }
    
    /// Calculate reconnect delay with exponential backoff
    async fn get_reconnect_delay(&self) -> Duration {
        if !self.reconnect_backoff_enabled {
            return Duration::from_millis(INITIAL_RECONNECT_DELAY_MS);
        }
        
        let attempt = *self.reconnect_attempt.read().await;
        let base_delay = INITIAL_RECONNECT_DELAY_MS;
        
        // Calculate exponential backoff: base_delay * 2^attempt
        let delay = base_delay * (1 << attempt.min(10)); // Cap at 2^10 to avoid overflow
        
        Duration::from_millis(delay.min(MAX_RECONNECT_DELAY_MS))
    }
    
    /// Record a successful message receipt
    async fn record_message_success(&self) {
        // Update last successful message time
        *self.last_successful_message.write().await = Instant::now();
        
        // Increment message count
        let mut count = self.received_message_count.write().await;
        *count += 1;
        
        // Reset error counter on successful messages
        if *self.error_count.read().await > 0 {
            let mut errors = self.error_count.write().await;
            *errors = (*errors).saturating_sub(1); // Gradually decrease error count
        }
    }
    
    /// Record a message latency measurement
    async fn record_latency(&self, latency_ms: u64) {
        let mut latencies = self.message_latencies.write().await;
        
        // Add the new latency and maintain the window size
        latencies.push_back(latency_ms);
        if latencies.len() > HEALTH_WINDOW_SIZE {
            latencies.pop_front();
        }
    }
    
    /// Record an error occurrence
    async fn record_error(&self, error_type: &str) {
        // Increment error count
        let mut errors = self.error_count.write().await;
        *errors += 1;
        
        // Log the error with structured information
        error!(
            error_type = error_type,
            error_count = *errors,
            "WebSocket error occurred"
        );
    }
    
    /// Record a disconnect event
    async fn record_disconnect(&self) {
        let mut disconnects = self.disconnect_count.write().await;
        *disconnects += 1;
        
        let mut attempts = self.reconnect_attempt.write().await;
        *attempts += 1;
        
        *self.last_reconnect_time.write().await = Instant::now();
    }
    
    /// Generate a signature for MEXC API authentication
    fn generate_signature(&self, timestamp: &str) -> Result<String, Error> {
        use hmac::{Hmac, Mac};
        use sha2::Sha256;
        
        let api_secret = self.api_secret.as_ref()
            .ok_or_else(|| Error::MissingCredentials("API secret is not set".into()))?;
            
        let mut mac = Hmac::<Sha256>::new_from_slice(api_secret.as_bytes())
            .map_err(|_| Error::SignatureError("Failed to create HMAC".into()))?;
            
        mac.update(timestamp.as_bytes());
        
        let result = mac.finalize();
        let signature = hex::encode(result.into_bytes());
        
        Ok(signature)
    }
    
    /// Attempt to authenticate with MEXC API using credentials
    async fn authenticate(&self, write: &mut futures::stream::SplitSink<tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>, Message>) -> Result<(), Error> {
        if self.api_key.is_none() || self.api_secret.is_none() {
            debug!("No API credentials provided, skipping authentication");
            return Ok(());
        }
        
        // Generate timestamp and signature
        let timestamp = format!("{}", chrono::Utc::now().timestamp_millis());
        let signature = self.generate_signature(&timestamp)?;
        
        // Create authentication message
        let auth_msg = json!({
            "method": "api_key",
            "api_key": self.api_key.as_ref().unwrap(),
            "sign": signature,
            "reqTime": timestamp
        });
        
        info!("Sending authentication request to MEXC");
        write.send(Message::Text(auth_msg.to_string())).await
            .map_err(|e| Error::WebSocketError(format!("Failed to send authentication message: {}", e)))?;
            
        // Increment sent message count
        let mut sent = self.sent_message_count.write().await;
        *sent += 1;
        
        Ok(())
    }
    
    /// Connect to WebSocket, authenticate, and subscribe to channels
    pub async fn connect_and_subscribe(&self) -> Result<(), Error> {
        // Check if we should use fallback due to circuit breaker
        if self.should_use_fallback().await {
            let mut using_fallback = self.using_fallback.write().await;
            *using_fallback = true;
            info!("Circuit breaker tripped, using fallback data source");
            return Err(Error::CircuitBreakerTripped);
        }
        
        info!("Connecting to MEXC WebSocket at {}", self.url);
        let url = Url::parse(&self.url)
            .map_err(|e| Error::ConfigError(format!("Invalid WebSocket URL: {}", e)))?;
            
        let (ws_stream, _) = connect_async(url).await
            .map_err(|e| Error::WebSocketError(format!("Failed to connect: {}", e)))?;
            
        info!("WebSocket connection established");
        
        let (mut write, mut read) = ws_stream.split();
        
        // Attempt to authenticate if credentials are provided
        if let Err(e) = self.authenticate(&mut write).await {
            warn!("Authentication failed: {}", e);
            // Continue anyway, as some operations don't require authentication
        }
        
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
            
            let request_json = serde_json::to_string(&request)
                .map_err(|e| Error::SerializationError(format!("Failed to serialize subscription request: {}", e)))?;
                
            write.send(Message::Text(request_json)).await
                .map_err(|e| Error::WebSocketError(format!("Failed to send subscription request: {}", e)))?;
                
            // Increment sent message count
            let mut sent = self.sent_message_count.write().await;
            *sent += 1;
        }
        
        // Tokio task for receiving messages
        let message_sender = self.message_sender.clone();
        let mut shutdown = self.shutdown.clone();
        
        // Health monitoring
        let health_score = self.health_score.clone();
        let message_latencies = self.message_latencies.clone();
        let error_count = self.error_count.clone();
        let last_successful_message = self.last_successful_message.clone();
        let received_message_count = self.received_message_count.clone();
        let disconnect_count = self.disconnect_count.clone();
        
        // Spawn a ping task to keep the connection alive
        let ping_task = {
            let mut write_ping = write.clone();
            tokio::spawn(async move {
                let mut interval = tokio::time::interval(Duration::from_millis(PING_INTERVAL_MS));
                loop {
                    interval.tick().await;
                    if *shutdown.borrow() {
                        break;
                    }
                    
                    // Send ping
                    if let Err(e) = write_ping.send(Message::Ping(vec![])).await {
                        error!("Failed to send ping: {}", e);
                        break;
                    }
                }
            })
        };
        
        // Main message processing task
        tokio::spawn(async move {
            let mut last_pong = Instant::now();
            
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
                                // Record successful message
                                record_message_success(&received_message_count, &last_successful_message).await;
                                
                                // Measure and record latency (assume 10ms network latency for simplicity)
                                record_latency(&message_latencies, 10).await;
                                
                                // Handle different message types
                                if let Ok(value) = serde_json::from_str::<Value>(&text) {
                                    if let Some(c) = value.get("c") {
                                        // This is a data message
                                        if let Err(e) = message_sender.send(text).await {
                                            error!("Failed to forward WebSocket message: {}", e);
                                            record_error(&error_count, "message_forward_error").await;
                                        }
                                    } else if value.get("code").is_some() {
                                        // This is a subscription response
                                        if let Ok(response) = serde_json::from_value::<MexcSubscriptionResponse>(value.clone()) {
                                            if response.code == 0 {
                                                debug!("Subscription successful: {}", response.msg);
                                            } else {
                                                warn!("Subscription failed: {} (code: {})", response.msg, response.code);
                                                
                                                // Check for blocked subscriptions
                                                if response.msg.contains("Blocked") {
                                                    error!("Subscription blocked: {}", response.msg);
                                                    record_error(&error_count, "subscription_blocked").await;
                                                }
                                            }
                                        }
                                    }
                                } else {
                                    warn!("Received non-JSON message: {}", text);
                                    record_error(&error_count, "invalid_json").await;
                                }
                            }
                            Ok(Message::Ping(data)) => {
                                // Respond to ping with pong
                                debug!("Received ping, sending pong");
                                if let Err(e) = write.send(Message::Pong(data)).await {
                                    error!("Failed to send pong: {}", e);
                                    record_error(&error_count, "pong_error").await;
                                }
                            }
                            Ok(Message::Pong(_)) => {
                                // Reset last pong time
                                last_pong = Instant::now();
                                debug!("Received pong response");
                            }
                            Ok(Message::Close(_)) => {
                                info!("WebSocket connection closed by server");
                                record_disconnect(&disconnect_count).await;
                                break;
                            }
                            Err(e) => {
                                error!("WebSocket error: {}", e);
                                record_error(&error_count, "websocket_error").await;
                                record_disconnect(&disconnect_count).await;
                                break;
                            }
                            _ => {} // Ignore other message types
                        }
                    }
                    // Check for stale connection (no pong response in 30 seconds)
                    _ = tokio::time::sleep(Duration::from_secs(5)) => {
                        if last_pong.elapsed() > Duration::from_secs(30) {
                            error!("No pong response received in 30 seconds, connection is stale");
                            record_error(&error_count, "stale_connection").await;
                            record_disconnect(&disconnect_count).await;
                            break;
                        }
                    }
                }
            }
            
            // Cancel the ping task when the main task exits
            ping_task.abort();
        });
        
        Ok(())
    }
    
    /// Get current connection health metrics
    pub async fn get_health_metrics(&self) -> serde_json::Value {
        json!({
            "health_score": *self.health_score.read().await,
            "error_count": *self.error_count.read().await,
            "reconnect_attempts": *self.reconnect_attempt.read().await,
            "disconnect_count": *self.disconnect_count.read().await,
            "received_messages": *self.received_message_count.read().await,
            "sent_messages": *self.sent_message_count.read().await,
            "using_fallback": *self.using_fallback.read().await,
            "time_since_last_message_ms": self.last_successful_message.read().await.elapsed().as_millis(),
        })
    }
}

// Helper functions for the async task
async fn record_message_success(received_count: &Arc<RwLock<u64>>, last_message_time: &Arc<RwLock<Instant>>) {
    *last_message_time.write().await = Instant::now();
    let mut count = received_count.write().await;
    *count += 1;
}

async fn record_latency(latencies: &Arc<RwLock<VecDeque<u64>>>, latency_ms: u64) {
    let mut lats = latencies.write().await;
    lats.push_back(latency_ms);
    if lats.len() > HEALTH_WINDOW_SIZE {
        lats.pop_front();
    }
}

async fn record_error(error_count: &Arc<RwLock<u32>>, error_type: &str) {
    let mut errors = error_count.write().await;
    *errors += 1;
    
    error!(
        error_type = error_type,
        error_count = *errors,
        "WebSocket error occurred"
    );
}

async fn record_disconnect(disconnect_count: &Arc<RwLock<u32>>) {
    let mut disconnects = disconnect_count.write().await;
    *disconnects += 1;
}
