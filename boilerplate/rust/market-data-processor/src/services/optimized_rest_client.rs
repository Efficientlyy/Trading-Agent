use crate::models::order_book::{OrderBook, OrderBookSide};
use crate::utils::enhanced_config::EnhancedConfig;
use async_trait::async_trait;
use futures::future::join_all;
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, RwLock};
use tokio::time::{interval, sleep};
use tracing::{debug, error, info, warn};

/// Maximum number of recent order books to cache
const MAX_CACHE_SIZE: usize = 10;

/// Minimum polling interval in milliseconds
const MIN_POLLING_INTERVAL_MS: u64 = 100;

/// Maximum polling interval in milliseconds
const MAX_POLLING_INTERVAL_MS: u64 = 10_000;

/// Market volatility thresholds for adaptive polling
const LOW_VOLATILITY_THRESHOLD: f64 = 0.0001; // 0.01%
const HIGH_VOLATILITY_THRESHOLD: f64 = 0.001; // 0.1%

/// Consecutive errors threshold for circuit breaker
const CIRCUIT_BREAKER_THRESHOLD: u32 = 5;

/// Circuit breaker reset time in seconds
const CIRCUIT_BREAKER_RESET_SEC: u64 = 60;

/// MEXC REST API response for order book
#[derive(Debug, Deserialize)]
pub struct MexcOrderBookResponse {
    pub bids: Vec<[String; 2]>,
    pub asks: Vec<[String; 2]>,
    pub time: Option<u64>,
}

/// Performance metrics for API requests
#[derive(Debug, Clone)]
pub struct ApiMetrics {
    pub request_count: u64,
    pub error_count: u64,
    pub last_latency_ms: u64,
    pub avg_latency_ms: f64,
    pub min_latency_ms: u64,
    pub max_latency_ms: u64,
    pub last_update_time: Instant,
    pub consecutive_errors: u32,
    pub circuit_open: bool,
    pub circuit_open_time: Option<Instant>,
}

impl Default for ApiMetrics {
    fn default() -> Self {
        Self {
            request_count: 0,
            error_count: 0,
            last_latency_ms: 0,
            avg_latency_ms: 0.0,
            min_latency_ms: u64::MAX,
            max_latency_ms: 0,
            last_update_time: Instant::now(),
            consecutive_errors: 0,
            circuit_open: false,
            circuit_open_time: None,
        }
    }
}

impl ApiMetrics {
    /// Record a successful request
    pub fn record_success(&mut self, latency_ms: u64) {
        self.request_count += 1;
        self.last_latency_ms = latency_ms;
        self.min_latency_ms = self.min_latency_ms.min(latency_ms);
        self.max_latency_ms = self.max_latency_ms.max(latency_ms);
        
        // Update moving average
        self.avg_latency_ms = (self.avg_latency_ms * (self.request_count - 1) as f64 + latency_ms as f64) 
            / self.request_count as f64;
        
        self.last_update_time = Instant::now();
        self.consecutive_errors = 0;
        
        // Reset circuit breaker if it was open
        if self.circuit_open {
            self.circuit_open = false;
            self.circuit_open_time = None;
            info!("Circuit breaker reset after successful request");
        }
    }
    
    /// Record a failed request
    pub fn record_error(&mut self) {
        self.request_count += 1;
        self.error_count += 1;
        self.consecutive_errors += 1;
        
        // Check if circuit breaker should open
        if self.consecutive_errors >= CIRCUIT_BREAKER_THRESHOLD && !self.circuit_open {
            self.circuit_open = true;
            self.circuit_open_time = Some(Instant::now());
            warn!("Circuit breaker opened after {} consecutive errors", self.consecutive_errors);
        }
    }
    
    /// Check if circuit breaker should auto-reset
    pub fn check_circuit_breaker(&mut self) {
        if self.circuit_open {
            if let Some(open_time) = self.circuit_open_time {
                if open_time.elapsed() > Duration::from_secs(CIRCUIT_BREAKER_RESET_SEC) {
                    self.circuit_open = false;
                    self.circuit_open_time = None;
                    self.consecutive_errors = 0;
                    info!("Circuit breaker auto-reset after {} seconds", CIRCUIT_BREAKER_RESET_SEC);
                }
            }
        }
    }
    
    /// Get health score from 0.0 (worst) to 1.0 (best)
    pub fn health_score(&self) -> f64 {
        if self.request_count == 0 {
            return 0.5; // Neutral when no data
        }
        
        if self.circuit_open {
            return 0.0; // Worst score when circuit is open
        }
        
        // Calculate error rate (0.0 to 1.0)
        let error_rate = self.error_count as f64 / self.request_count as f64;
        
        // Calculate latency score (higher latency = lower score)
        let latency_score = if self.avg_latency_ms <= 100.0 {
            1.0 // Great latency
        } else if self.avg_latency_ms <= 500.0 {
            0.75 // Good latency
        } else if self.avg_latency_ms <= 1000.0 {
            0.5 // Acceptable latency
        } else {
            0.25 // Poor latency
        };
        
        // Weight: 70% error rate, 30% latency
        (1.0 - error_rate) * 0.7 + latency_score * 0.3
    }
}

/// Order book diff calculation result
#[derive(Debug)]
pub struct OrderBookDiff {
    pub symbol: String,
    pub sequence: u64,
    pub bids_added: Vec<(String, String)>,
    pub bids_removed: Vec<(String, String)>,
    pub bids_updated: Vec<(String, String, String)>, // price, old_qty, new_qty
    pub asks_added: Vec<(String, String)>,
    pub asks_removed: Vec<(String, String)>,
    pub asks_updated: Vec<(String, String, String)>, // price, old_qty, new_qty
    pub spread_change: Option<f64>,
    pub midpoint_change: Option<f64>,
}

/// REST API client trait for abstraction
#[async_trait]
pub trait RestApiClient: Send + Sync {
    async fn get_order_book(&self, symbol: &str, limit: usize) -> Result<OrderBook, String>;
    async fn get_ticker(&self, symbol: &str) -> Result<Value, String>;
    async fn get_metrics(&self) -> ApiMetrics;
}

/// Optimized REST API client with adaptive polling and circuit breaker
pub struct OptimizedRestClient {
    client: Client,
    base_url: String,
    api_key: String,
    api_secret: String,
    metrics: Arc<RwLock<ApiMetrics>>,
    cache: Arc<RwLock<HashMap<String, VecDeque<(Instant, OrderBook)>>>>,
    volatility_map: Arc<RwLock<HashMap<String, f64>>>,
    polling_intervals: Arc<RwLock<HashMap<String, u64>>>,
}

impl OptimizedRestClient {
    /// Create a new optimized REST client
    pub fn new(config: &EnhancedConfig) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .unwrap_or_else(|_| Client::new());
        
        Self {
            client,
            base_url: config.mexc_rest_url.clone(),
            api_key: config.mexc_api_key.clone(),
            api_secret: config.mexc_api_secret.clone(),
            metrics: Arc::new(RwLock::new(ApiMetrics::default())),
            cache: Arc::new(RwLock::new(HashMap::new())),
            volatility_map: Arc::new(RwLock::new(HashMap::new())),
            polling_intervals: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Start adaptive polling for order books
    pub async fn start_adaptive_polling(
        &self, 
        symbols: Vec<String>,
        sender: mpsc::Sender<Result<OrderBook, String>>,
        initial_interval_ms: u64,
    ) {
        let client = self.clone();
        let metrics = self.metrics.clone();
        let cache = self.cache.clone();
        let volatility_map = self.volatility_map.clone();
        let polling_intervals = self.polling_intervals.clone();
        
        // Initialize polling intervals
        {
            let mut intervals = polling_intervals.write().await;
            for symbol in &symbols {
                intervals.insert(symbol.clone(), initial_interval_ms);
            }
        }
        
        // Start polling task
        tokio::spawn(async move {
            info!("Starting adaptive polling for symbols: {:?}", symbols);
            
            // Create a map of last poll times
            let mut last_poll_times = HashMap::new();
            for symbol in &symbols {
                last_poll_times.insert(symbol.clone(), Instant::now());
            }
            
            // Poll at 100ms interval to check which symbols need updating
            let mut check_interval = interval(Duration::from_millis(100));
            
            loop {
                check_interval.tick().await;
                
                // Check if circuit breaker is open
                {
                    let mut metrics_lock = metrics.write().await;
                    metrics_lock.check_circuit_breaker();
                    
                    if metrics_lock.circuit_open {
                        // Skip this iteration if circuit breaker is open
                        debug!("Circuit breaker is open, skipping poll iteration");
                        continue;
                    }
                }
                
                // Check which symbols need polling
                let mut symbols_to_poll = Vec::new();
                
                {
                    let intervals = polling_intervals.read().await;
                    
                    for symbol in &symbols {
                        let interval_ms = *intervals.get(symbol).unwrap_or(&initial_interval_ms);
                        let last_poll = last_poll_times.get(symbol).unwrap_or(&Instant::now());
                        
                        if last_poll.elapsed() >= Duration::from_millis(interval_ms) {
                            symbols_to_poll.push(symbol.clone());
                            last_poll_times.insert(symbol.clone(), Instant::now());
                        }
                    }
                }
                
                // Poll the symbols that need updating
                if !symbols_to_poll.is_empty() {
                    let futures = symbols_to_poll.iter().map(|symbol| {
                        let client_clone = client.clone();
                        let symbol_clone = symbol.clone();
                        async move {
                            (symbol_clone.clone(), client_clone.get_order_book(&symbol_clone, 100).await)
                        }
                    });
                    
                    let results = join_all(futures).await;
                    
                    // Process results
                    for (symbol, result) in results {
                        match result {
                            Ok(order_book) => {
                                // Cache the order book
                                {
                                    let mut cache_lock = cache.write().await;
                                    let symbol_cache = cache_lock.entry(symbol.clone()).or_insert_with(VecDeque::new);
                                    
                                    // Calculate volatility if we have previous data
                                    if !symbol_cache.is_empty() {
                                        let prev_order_book = &symbol_cache.back().unwrap().1;
                                        let volatility = calculate_volatility(prev_order_book, &order_book);
                                        
                                        // Update volatility map
                                        {
                                            let mut volatility_lock = volatility_map.write().await;
                                            volatility_lock.insert(symbol.clone(), volatility);
                                        }
                                        
                                        // Adjust polling interval based on volatility
                                        {
                                            let mut intervals_lock = polling_intervals.write().await;
                                            let current_interval = intervals_lock.get(&symbol).unwrap_or(&initial_interval_ms);
                                            let new_interval = adapt_polling_interval(*current_interval, volatility);
                                            intervals_lock.insert(symbol.clone(), new_interval);
                                            
                                            if new_interval != *current_interval {
                                                debug!("Adjusted polling interval for {}: {} -> {} ms (volatility: {:.6})", 
                                                    symbol, current_interval, new_interval, volatility);
                                            }
                                        }
                                        
                                        // Calculate diff for debugging
                                        let diff = calculate_order_book_diff(prev_order_book, &order_book);
                                        debug!("Order book diff for {}: {} bids changed, {} asks changed", 
                                            symbol, 
                                            diff.bids_added.len() + diff.bids_removed.len() + diff.bids_updated.len(),
                                            diff.asks_added.len() + diff.asks_removed.len() + diff.asks_updated.len());
                                    }
                                    
                                    // Add to cache and maintain max size
                                    symbol_cache.push_back((Instant::now(), order_book.clone()));
                                    if symbol_cache.len() > MAX_CACHE_SIZE {
                                        symbol_cache.pop_front();
                                    }
                                }
                                
                                // Send the order book to the channel
                                if let Err(e) = sender.send(Ok(order_book)).await {
                                    error!("Failed to send order book for {}: {}", symbol, e);
                                }
                            },
                            Err(e) => {
                                error!("Failed to get order book for {}: {}", symbol, e);
                                
                                // Record error in metrics
                                {
                                    let mut metrics_lock = metrics.write().await;
                                    metrics_lock.record_error();
                                }
                                
                                // Send error to the channel
                                if let Err(e) = sender.send(Err(format!("Failed to get order book for {}: {}", symbol, e))).await {
                                    error!("Failed to send error for {}: {}", symbol, e);
                                }
                                
                                // Back off on error
                                {
                                    let mut intervals_lock = polling_intervals.write().await;
                                    let current_interval = intervals_lock.get(&symbol).unwrap_or(&initial_interval_ms);
                                    let new_interval = (*current_interval * 2).min(MAX_POLLING_INTERVAL_MS);
                                    intervals_lock.insert(symbol.clone(), new_interval);
                                    debug!("Backing off polling for {} to {} ms due to error", symbol, new_interval);
                                }
                            }
                        }
                    }
                }
            }
        });
    }
    
    /// Get the latest cached order book if available
    pub async fn get_cached_order_book(&self, symbol: &str) -> Option<OrderBook> {
        let cache = self.cache.read().await;
        cache.get(symbol)
            .and_then(|queue| queue.back())
            .map(|(_, order_book)| order_book.clone())
    }
    
    /// Get volatility for a symbol
    pub async fn get_volatility(&self, symbol: &str) -> f64 {
        let volatility_map = self.volatility_map.read().await;
        *volatility_map.get(symbol).unwrap_or(&0.0)
    }
    
    /// Get current polling interval for a symbol
    pub async fn get_polling_interval(&self, symbol: &str) -> u64 {
        let polling_intervals = self.polling_intervals.read().await;
        *polling_intervals.get(symbol).unwrap_or(&0)
    }
}

impl Clone for OptimizedRestClient {
    fn clone(&self) -> Self {
        Self {
            client: self.client.clone(),
            base_url: self.base_url.clone(),
            api_key: self.api_key.clone(),
            api_secret: self.api_secret.clone(),
            metrics: self.metrics.clone(),
            cache: self.cache.clone(),
            volatility_map: self.volatility_map.clone(),
            polling_intervals: self.polling_intervals.clone(),
        }
    }
}

#[async_trait]
impl RestApiClient for OptimizedRestClient {
    async fn get_order_book(&self, symbol: &str, limit: usize) -> Result<OrderBook, String> {
        // Check circuit breaker
        {
            let metrics = self.metrics.read().await;
            if metrics.circuit_open {
                return Err(format!("Circuit breaker is open for REST API"));
            }
        }
        
        let start_time = Instant::now();
        
        // Check cache first for very recent data (less than 100ms old)
        {
            let cache = self.cache.read().await;
            if let Some(queue) = cache.get(symbol) {
                if let Some((timestamp, order_book)) = queue.back() {
                    if timestamp.elapsed() < Duration::from_millis(100) {
                        debug!("Using cached order book for {} (age: {:?})", symbol, timestamp.elapsed());
                        return Ok(order_book.clone());
                    }
                }
            }
        }
        
        // Build URL
        let url = format!(
            "{}/api/v3/depth?symbol={}&limit={}",
            self.base_url, symbol, limit
        );
        
        // Make request
        let response = match self.client
            .get(&url)
            .header("X-MEXC-APIKEY", &self.api_key)
            .send()
            .await
        {
            Ok(resp) => resp,
            Err(e) => {
                let mut metrics = self.metrics.write().await;
                metrics.record_error();
                return Err(format!("Request error: {}", e));
            }
        };
        
        // Check status code
        if response.status() != StatusCode::OK {
            let mut metrics = self.metrics.write().await;
            metrics.record_error();
            
            let status = response.status();
            let body = response.text().await.unwrap_or_else(|_| "Unable to read response body".to_string());
            return Err(format!("HTTP error {}: {}", status, body));
        }
        
        // Parse response
        let resp: MexcOrderBookResponse = match response.json().await {
            Ok(data) => data,
            Err(e) => {
                let mut metrics = self.metrics.write().await;
                metrics.record_error();
                return Err(format!("Failed to parse order book: {}", e));
            }
        };
        
        // Calculate latency
        let latency_ms = start_time.elapsed().as_millis() as u64;
        
        // Record metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.record_success(latency_ms);
        }
        
        // Convert to our OrderBook model
        let timestamp = resp.time.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64
        });
        
        let mut bids = Vec::new();
        for bid in resp.bids {
            if bid.len() >= 2 {
                let price = bid[0].parse::<f64>().unwrap_or_default();
                let quantity = bid[1].parse::<f64>().unwrap_or_default();
                bids.push((price, quantity));
            }
        }
        
        let mut asks = Vec::new();
        for ask in resp.asks {
            if ask.len() >= 2 {
                let price = ask[0].parse::<f64>().unwrap_or_default();
                let quantity = ask[1].parse::<f64>().unwrap_or_default();
                asks.push((price, quantity));
            }
        }
        
        // Sort bids (highest first) and asks (lowest first)
        bids.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        asks.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        
        let order_book = OrderBook {
            symbol: symbol.to_string(),
            timestamp,
            bids: OrderBookSide::new(bids),
            asks: OrderBookSide::new(asks),
            sequence: None,
        };
        
        Ok(order_book)
    }
    
    async fn get_ticker(&self, symbol: &str) -> Result<Value, String> {
        // Check circuit breaker
        {
            let metrics = self.metrics.read().await;
            if metrics.circuit_open {
                return Err(format!("Circuit breaker is open for REST API"));
            }
        }
        
        let start_time = Instant::now();
        
        // Build URL
        let url = format!(
            "{}/api/v3/ticker/24hr?symbol={}",
            self.base_url, symbol
        );
        
        // Make request
        let response = match self.client
            .get(&url)
            .header("X-MEXC-APIKEY", &self.api_key)
            .send()
            .await
        {
            Ok(resp) => resp,
            Err(e) => {
                let mut metrics = self.metrics.write().await;
                metrics.record_error();
                return Err(format!("Request error: {}", e));
            }
        };
        
        // Check status code
        if response.status() != StatusCode::OK {
            let mut metrics = self.metrics.write().await;
            metrics.record_error();
            
            let status = response.status();
            let body = response.text().await.unwrap_or_else(|_| "Unable to read response body".to_string());
            return Err(format!("HTTP error {}: {}", status, body));
        }
        
        // Parse response
        let resp: Value = match response.json().await {
            Ok(data) => data,
            Err(e) => {
                let mut metrics = self.metrics.write().await;
                metrics.record_error();
                return Err(format!("Failed to parse ticker: {}", e));
            }
        };
        
        // Calculate latency
        let latency_ms = start_time.elapsed().as_millis() as u64;
        
        // Record metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.record_success(latency_ms);
        }
        
        Ok(resp)
    }
    
    async fn get_metrics(&self) -> ApiMetrics {
        self.metrics.read().await.clone()
    }
}

/// Calculate volatility between two order books
fn calculate_volatility(prev: &OrderBook, current: &OrderBook) -> f64 {
    // Calculate mid-price for both order books
    let prev_mid = calculate_mid_price(prev);
    let current_mid = calculate_mid_price(current);
    
    // Calculate relative price change
    if let (Some(prev_mid), Some(current_mid)) = (prev_mid, current_mid) {
        (current_mid - prev_mid).abs() / prev_mid
    } else {
        0.0 // Default to no volatility if we can't calculate
    }
}

/// Calculate mid price from order book
fn calculate_mid_price(order_book: &OrderBook) -> Option<f64> {
    if order_book.bids.levels.is_empty() || order_book.asks.levels.is_empty() {
        return None;
    }
    
    let best_bid = order_book.bids.levels[0].0;
    let best_ask = order_book.asks.levels[0].0;
    
    Some((best_bid + best_ask) / 2.0)
}

/// Adapt polling interval based on volatility
fn adapt_polling_interval(current_interval: u64, volatility: f64) -> u64 {
    if volatility > HIGH_VOLATILITY_THRESHOLD {
        // High volatility - poll faster
        (current_interval / 2).max(MIN_POLLING_INTERVAL_MS)
    } else if volatility < LOW_VOLATILITY_THRESHOLD {
        // Low volatility - poll slower
        (current_interval * 2).min(MAX_POLLING_INTERVAL_MS)
    } else {
        // Moderate volatility - keep current interval
        current_interval
    }
}

/// Calculate difference between two order books
fn calculate_order_book_diff(prev: &OrderBook, current: &OrderBook) -> OrderBookDiff {
    // Create price -> quantity maps for easier comparison
    let mut prev_bids_map = HashMap::new();
    let mut prev_asks_map = HashMap::new();
    
    for &(price, qty) in &prev.bids.levels {
        prev_bids_map.insert(price.to_string(), qty.to_string());
    }
    
    for &(price, qty) in &prev.asks.levels {
        prev_asks_map.insert(price.to_string(), qty.to_string());
    }
    
    let mut current_bids_map = HashMap::new();
    let mut current_asks_map = HashMap::new();
    
    for &(price, qty) in &current.bids.levels {
        current_bids_map.insert(price.to_string(), qty.to_string());
    }
    
    for &(price, qty) in &current.asks.levels {
        current_asks_map.insert(price.to_string(), qty.to_string());
    }
    
    // Find differences
    let mut bids_added = Vec::new();
    let mut bids_removed = Vec::new();
    let mut bids_updated = Vec::new();
    
    // Bids added or updated
    for (price, qty) in &current_bids_map {
        if let Some(prev_qty) = prev_bids_map.get(price) {
            if prev_qty != qty {
                bids_updated.push((price.clone(), prev_qty.clone(), qty.clone()));
            }
        } else {
            bids_added.push((price.clone(), qty.clone()));
        }
    }
    
    // Bids removed
    for (price, qty) in &prev_bids_map {
        if !current_bids_map.contains_key(price) {
            bids_removed.push((price.clone(), qty.clone()));
        }
    }
    
    // Asks changes
    let mut asks_added = Vec::new();
    let mut asks_removed = Vec::new();
    let mut asks_updated = Vec::new();
    
    // Asks added or updated
    for (price, qty) in &current_asks_map {
        if let Some(prev_qty) = prev_asks_map.get(price) {
            if prev_qty != qty {
                asks_updated.push((price.clone(), prev_qty.clone(), qty.clone()));
            }
        } else {
            asks_added.push((price.clone(), qty.clone()));
        }
    }
    
    // Asks removed
    for (price, qty) in &prev_asks_map {
        if !current_asks_map.contains_key(price) {
            asks_removed.push((price.clone(), qty.clone()));
        }
    }
    
    // Calculate spread and midpoint changes
    let prev_spread = if prev.bids.levels.is_empty() || prev.asks.levels.is_empty() {
        None
    } else {
        Some(prev.asks.levels[0].0 - prev.bids.levels[0].0)
    };
    
    let current_spread = if current.bids.levels.is_empty() || current.asks.levels.is_empty() {
        None
    } else {
        Some(current.asks.levels[0].0 - current.bids.levels[0].0)
    };
    
    let spread_change = match (prev_spread, current_spread) {
        (Some(prev), Some(current)) => Some(current - prev),
        _ => None,
    };
    
    let prev_midpoint = calculate_mid_price(prev);
    let current_midpoint = calculate_mid_price(current);
    
    let midpoint_change = match (prev_midpoint, current_midpoint) {
        (Some(prev), Some(current)) => Some(current - prev),
        _ => None,
    };
    
    OrderBookDiff {
        symbol: current.symbol.clone(),
        sequence: current.sequence.unwrap_or(0),
        bids_added,
        bids_removed,
        bids_updated,
        asks_added,
        asks_removed,
        asks_updated,
        spread_change,
        midpoint_change,
    }
}
