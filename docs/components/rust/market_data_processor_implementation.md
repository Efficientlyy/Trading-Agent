# Market Data Processor: Rust Implementation Plan

## Overview

This document outlines a step-by-step implementation plan for the Market Data Processor component in Rust. The Market Data Processor is a critical performance component responsible for efficiently processing real-time market data from MEXC exchange, maintaining order book state, and providing processed data to other system components.

## Phase 1: Foundation (Weeks 1-2)

### Step 1: Project Setup and Environment (Days 1-2)

**Objective**: Set up the Rust project structure and development environment.

**Tasks**:
1. Create new Rust project using Cargo
   ```bash
   cargo new market-data-processor --lib
   ```
2. Set up project directory structure
   ```
   /market-data-processor
     /src
       /models      # Data models
       /services    # Core services
       /utils       # Utility functions
       /api         # API interfaces
       lib.rs       # Library entry point
     /tests         # Integration tests
     /benches       # Performance benchmarks
     Cargo.toml     # Dependencies and configuration
     .github        # CI/CD workflows
   ```
3. Configure development environment
   - Set up Rust toolchain with rustup
   - Configure VS Code or other IDE with Rust extensions
   - Set up git repository with .gitignore
4. Add essential dependencies to Cargo.toml
   ```toml
   [dependencies]
   tokio = { version = "1.28", features = ["full"] }
   serde = { version = "1.0", features = ["derive"] }
   serde_json = "1.0"
   thiserror = "1.0"
   tracing = "0.1"
   tracing-subscriber = "0.3"
   
   [dev-dependencies]
   criterion = "0.5"
   mockall = "0.11"
   tokio-test = "0.4"
   
   [profile.release]
   lto = true
   codegen-units = 1
   panic = "abort"
   ```
5. Set up logging and error handling utilities
6. Create basic README with project overview and setup instructions

**Deliverables**:
- Initialized Rust project with proper structure
- Development environment configuration
- Basic utility modules for logging and error handling
- Project documentation

**Integration Points**:
- None at this stage

### Step 2: Core Data Models (Days 3-5)

**Objective**: Implement the core data models for market data.

**Tasks**:
1. Define market data models in `src/models/mod.rs`
   ```rust
   // Example model structure
   pub mod order_book;
   pub mod trade;
   pub mod ticker;
   pub mod candle;
   pub mod common;
   ```

2. Implement order book models in `src/models/order_book.rs`
   ```rust
   use serde::{Deserialize, Serialize};
   use std::collections::BTreeMap;
   
   #[derive(Debug, Clone, Serialize, Deserialize)]
   pub struct OrderBookEntry {
       pub price: f64,
       pub quantity: f64,
   }
   
   #[derive(Debug, Clone, Serialize, Deserialize)]
   pub struct OrderBook {
       pub symbol: String,
       pub last_update_id: u64,
       pub bids: BTreeMap<f64, f64>,  // price -> quantity
       pub asks: BTreeMap<f64, f64>,  // price -> quantity
       pub timestamp: u64,
   }
   
   // Additional methods for OrderBook
   impl OrderBook {
       pub fn new(symbol: &str) -> Self {
           Self {
               symbol: symbol.to_string(),
               last_update_id: 0,
               bids: BTreeMap::new(),
               asks: BTreeMap::new(),
               timestamp: 0,
           }
       }
       
       pub fn update_bid(&mut self, price: f64, quantity: f64) {
           if quantity > 0.0 {
               self.bids.insert(price, quantity);
           } else {
               self.bids.remove(&price);
           }
       }
       
       pub fn update_ask(&mut self, price: f64, quantity: f64) {
           if quantity > 0.0 {
               self.asks.insert(price, quantity);
           } else {
               self.asks.remove(&price);
           }
       }
       
       // Additional methods for order book analysis
       pub fn best_bid(&self) -> Option<(f64, f64)> {
           self.bids.iter().next_back().map(|(k, v)| (*k, *v))
       }
       
       pub fn best_ask(&self) -> Option<(f64, f64)> {
           self.asks.iter().next().map(|(k, v)| (*k, *v))
       }
       
       pub fn spread(&self) -> Option<f64> {
           match (self.best_ask(), self.best_bid()) {
               (Some((ask, _)), Some((bid, _))) => Some(ask - bid),
               _ => None,
           }
       }
   }
   ```

3. Implement trade models in `src/models/trade.rs`
   ```rust
   use serde::{Deserialize, Serialize};
   
   #[derive(Debug, Clone, Serialize, Deserialize)]
   pub struct Trade {
       pub id: u64,
       pub symbol: String,
       pub price: f64,
       pub quantity: f64,
       pub buyer_order_id: u64,
       pub seller_order_id: u64,
       pub timestamp: u64,
       pub is_buyer_maker: bool,
   }
   ```

4. Implement ticker models in `src/models/ticker.rs`
   ```rust
   use serde::{Deserialize, Serialize};
   
   #[derive(Debug, Clone, Serialize, Deserialize)]
   pub struct Ticker {
       pub symbol: String,
       pub price_change: f64,
       pub price_change_percent: f64,
       pub weighted_avg_price: f64,
       pub last_price: f64,
       pub last_quantity: f64,
       pub bid_price: f64,
       pub bid_quantity: f64,
       pub ask_price: f64,
       pub ask_quantity: f64,
       pub open_price: f64,
       pub high_price: f64,
       pub low_price: f64,
       pub volume: f64,
       pub quote_volume: f64,
       pub open_time: u64,
       pub close_time: u64,
       pub first_id: u64,
       pub last_id: u64,
       pub count: u64,
   }
   ```

5. Implement candlestick models in `src/models/candle.rs`
   ```rust
   use serde::{Deserialize, Serialize};
   
   #[derive(Debug, Clone, Serialize, Deserialize)]
   pub struct Candle {
       pub symbol: String,
       pub interval: String,
       pub open_time: u64,
       pub open: f64,
       pub high: f64,
       pub low: f64,
       pub close: f64,
       pub volume: f64,
       pub close_time: u64,
       pub quote_asset_volume: f64,
       pub number_of_trades: u64,
       pub taker_buy_base_asset_volume: f64,
       pub taker_buy_quote_asset_volume: f64,
   }
   ```

6. Implement common types in `src/models/common.rs`
   ```rust
   use serde::{Deserialize, Serialize};
   
   #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
   pub enum Interval {
       #[serde(rename = "1m")]
       OneMinute,
       #[serde(rename = "3m")]
       ThreeMinutes,
       #[serde(rename = "5m")]
       FiveMinutes,
       #[serde(rename = "15m")]
       FifteenMinutes,
       #[serde(rename = "30m")]
       ThirtyMinutes,
       #[serde(rename = "1h")]
       OneHour,
       #[serde(rename = "2h")]
       TwoHours,
       #[serde(rename = "4h")]
       FourHours,
       #[serde(rename = "6h")]
       SixHours,
       #[serde(rename = "8h")]
       EightHours,
       #[serde(rename = "12h")]
       TwelveHours,
       #[serde(rename = "1d")]
       OneDay,
       #[serde(rename = "3d")]
       ThreeDays,
       #[serde(rename = "1w")]
       OneWeek,
       #[serde(rename = "1M")]
       OneMonth,
   }
   
   impl Interval {
       pub fn as_millis(&self) -> u64 {
           match self {
               Interval::OneMinute => 60_000,
               Interval::ThreeMinutes => 180_000,
               Interval::FiveMinutes => 300_000,
               Interval::FifteenMinutes => 900_000,
               Interval::ThirtyMinutes => 1_800_000,
               Interval::OneHour => 3_600_000,
               Interval::TwoHours => 7_200_000,
               Interval::FourHours => 14_400_000,
               Interval::SixHours => 21_600_000,
               Interval::EightHours => 28_800_000,
               Interval::TwelveHours => 43_200_000,
               Interval::OneDay => 86_400_000,
               Interval::ThreeDays => 259_200_000,
               Interval::OneWeek => 604_800_000,
               Interval::OneMonth => 2_592_000_000,
           }
       }
   }
   ```

7. Create unit tests for all models
8. Implement serialization/deserialization tests

**Deliverables**:
- Complete set of data models for market data
- Serialization/deserialization support
- Unit tests for all models

**Integration Points**:
- Models will be shared with other Rust components
- Serialization formats will be compatible with Node.js and Python components

### Step 3: WebSocket Message Parsing (Days 6-8)

**Objective**: Implement parsers for MEXC WebSocket messages.

**Tasks**:
1. Create WebSocket message models in `src/models/websocket.rs`
   ```rust
   use serde::{Deserialize, Serialize};
   use serde_json::Value;
   
   #[derive(Debug, Clone, Serialize, Deserialize)]
   pub struct WebSocketMessage {
       pub c: String,  // Channel
       pub s: String,  // Symbol
       pub d: Value,   // Data (varies by message type)
       pub t: u64,     // Timestamp
   }
   
   #[derive(Debug, Clone, Serialize, Deserialize)]
   pub struct OrderBookUpdateData {
       pub s: String,           // Symbol
       pub t: u64,              // Timestamp
       pub v: u64,              // Version
       #[serde(default)]
       pub b: Vec<[String; 2]>, // Bids [price, quantity]
       #[serde(default)]
       pub a: Vec<[String; 2]>, // Asks [price, quantity]
   }
   
   #[derive(Debug, Clone, Serialize, Deserialize)]
   pub struct TradeData {
       pub s: String,  // Symbol
       pub p: String,  // Price
       pub q: String,  // Quantity
       pub v: String,  // Value
       pub t: u64,     // Timestamp
       pub m: bool,    // Is buyer maker
   }
   
   #[derive(Debug, Clone, Serialize, Deserialize)]
   pub struct TickerData {
       pub s: String,  // Symbol
       pub o: String,  // Open price
       pub h: String,  // High price
       pub l: String,  // Low price
       pub c: String,  // Close price
       pub v: String,  // Volume
       pub qv: String, // Quote volume
       pub t: u64,     // Timestamp
   }
   ```

2. Implement message parsers in `src/services/message_parser.rs`
   ```rust
   use crate::models::{
       order_book::{OrderBook, OrderBookEntry},
       trade::Trade,
       ticker::Ticker,
       websocket::{WebSocketMessage, OrderBookUpdateData, TradeData, TickerData},
   };
   use serde_json::Value;
   use thiserror::Error;
   
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
   }
   ```

3. Create unit tests for message parsing
4. Implement error handling for malformed messages
5. Add benchmarks for parsing performance

**Deliverables**:
- WebSocket message models
- Message parsing service
- Unit tests for message parsing
- Performance benchmarks

**Integration Points**:
- Will receive raw WebSocket messages from WebSocket client
- Will provide parsed data to order book manager

### Step 4: Order Book Manager (Days 9-12)

**Objective**: Implement an efficient order book manager to maintain real-time order book state.

**Tasks**:
1. Create order book manager in `src/services/order_book_manager.rs`
   ```rust
   use std::collections::HashMap;
   use std::sync::{Arc, RwLock};
   use tokio::sync::broadcast;
   
   use crate::models::{
       order_book::OrderBook,
       websocket::OrderBookUpdateData,
   };
   use crate::services::message_parser::MessageParser;
   
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
               None => return false, // No snapshot received yet
           };
           
           let mut book = order_book.write().unwrap();
           
           // Verify sequence
           if data.v <= book.last_update_id {
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
   }
   ```

2. Implement thread-safe access to order books
3. Create efficient update mechanism for order book changes
4. Implement order book analysis methods
5. Add unit tests for order book management
6. Create benchmarks for order book updates

**Deliverables**:
- Order book manager implementation
- Thread-safe order book state management
- Order book analysis utilities
- Unit tests and benchmarks

**Integration Points**:
- Will receive parsed order book updates from message parser
- Will provide order book state to API and analysis components

## Phase 2: Core Functionality (Weeks 3-4)

### Step 5: Market Data Service (Days 13-16)

**Objective**: Implement the core market data service to process and distribute market data.

**Tasks**:
1. Create market data service in `src/services/market_data_service.rs`
   ```rust
   use std::collections::HashMap;
   use std::sync::{Arc, RwLock};
   use tokio::sync::{broadcast, mpsc};
   use tokio::task;
   
   use crate::models::{
       order_book::OrderBook,
       trade::Trade,
       ticker::Ticker,
       websocket::{WebSocketMessage, OrderBookUpdateData, TradeData, TickerData},
   };
   use crate::services::{
       message_parser::MessageParser,
       order_book_manager::OrderBookManager,
   };
   
   pub struct MarketDataService {
       order_book_manager: Arc<OrderBookManager>,
       trades: Arc<RwLock<HashMap<String, Vec<Trade>>>>,
       tickers: Arc<RwLock<HashMap<String, Ticker>>>,
       message_receiver: mpsc::Receiver<String>,
       shutdown: tokio::sync::watch::Receiver<bool>,
   }
   
   impl MarketDataService {
       pub fn new(
           message_receiver: mpsc::Receiver<String>,
           shutdown: tokio::sync::watch::Receiver<bool>,
       ) -> Self {
           Self {
               order_book_manager: Arc::new(OrderBookManager::new()),
               trades: Arc::new(RwLock::new(HashMap::new())),
               tickers: Arc::new(RwLock::new(HashMap::new())),
               message_receiver,
               shutdown,
           }
       }
       
       pub fn get_order_book_manager(&self) -> Arc<OrderBookManager> {
           self.order_book_manager.clone()
       }
       
       pub async fn run(&mut self) {
           while !*self.shutdown.borrow() {
               tokio::select! {
                   Some(message) = self.message_receiver.recv() => {
                       self.process_message(&message).await;
                   }
                   _ = self.shutdown.changed() => {
                       break;
                   }
               }
           }
       }
       
       async fn process_message(&self, message: &str) {
           // Parse the WebSocket message
           let ws_message = match MessageParser::parse_message(message) {
               Ok(msg) => msg,
               Err(e) => {
                   tracing::error!("Failed to parse WebSocket message: {}", e);
                   return;
               }
           };
           
           // Process based on channel type
           if ws_message.c.contains("depth") {
               self.process_order_book_message(&ws_message).await;
           } else if ws_message.c.contains("deals") {
               self.process_trade_message(&ws_message).await;
           } else if ws_message.c.contains("ticker") {
               self.process_ticker_message(&ws_message).await;
           } else {
               tracing::debug!("Ignoring message from unsupported channel: {}", ws_message.c);
           }
       }
       
       async fn process_order_book_message(&self, message: &WebSocketMessage) {
           let data = match MessageParser::parse_order_book_update(message) {
               Ok(data) => data,
               Err(e) => {
                   tracing::error!("Failed to parse order book update: {}", e);
                   return;
               }
           };
           
           let symbol = &message.s;
           let is_snapshot = message.c.contains("snapshot");
           
           if is_snapshot {
               self.order_book_manager.process_snapshot(symbol, &data);
           } else {
               self.order_book_manager.process_update(symbol, &data);
           }
       }
       
       async fn process_trade_message(&self, message: &WebSocketMessage) {
           let data = match MessageParser::parse_trade(message) {
               Ok(data) => data,
               Err(e) => {
                   tracing::error!("Failed to parse trade data: {}", e);
                   return;
               }
           };
           
           let trade = Trade {
               id: 0, // MEXC doesn't provide trade ID in WebSocket
               symbol: message.s.clone(),
               price: data.p.parse().unwrap_or(0.0),
               quantity: data.q.parse().unwrap_or(0.0),
               buyer_order_id: 0, // Not provided
               seller_order_id: 0, // Not provided
               timestamp: data.t,
               is_buyer_maker: data.m,
           };
           
           // Store trade and notify subscribers
           let mut trades = self.trades.write().unwrap();
           let symbol_trades = trades.entry(message.s.clone()).or_insert_with(Vec::new);
           
           // Keep only the last 1000 trades
           if symbol_trades.len() >= 1000 {
               symbol_trades.remove(0);
           }
           
           symbol_trades.push(trade);
       }
       
       async fn process_ticker_message(&self, message: &WebSocketMessage) {
           let data = match MessageParser::parse_ticker(message) {
               Ok(data) => data,
               Err(e) => {
                   tracing::error!("Failed to parse ticker data: {}", e);
                   return;
               }
           };
           
           let ticker = Ticker {
               symbol: message.s.clone(),
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
           };
           
           // Store ticker and notify subscribers
           let mut tickers = self.tickers.write().unwrap();
           tickers.insert(message.s.clone(), ticker);
       }
       
       // Public API methods
       pub fn get_ticker(&self, symbol: &str) -> Option<Ticker> {
           let tickers = self.tickers.read().unwrap();
           tickers.get(symbol).cloned()
       }
       
       pub fn get_recent_trades(&self, symbol: &str, limit: usize) -> Vec<Trade> {
           let trades = self.trades.read().unwrap();
           match trades.get(symbol) {
               Some(symbol_trades) => {
                   let start = symbol_trades.len().saturating_sub(limit);
                   symbol_trades[start..].to_vec()
               }
               None => Vec::new(),
           }
       }
   }
   ```

2. Implement message processing pipeline
3. Create efficient data distribution mechanisms
4. Add thread-safe access to market data
5. Implement unit tests for market data service
6. Create benchmarks for message processing throughput

**Deliverables**:
- Market data service implementation
- Message processing pipeline
- Data distribution mechanisms
- Unit tests and benchmarks

**Integration Points**:
- Will receive raw WebSocket messages from WebSocket client
- Will provide processed market data to API layer

### Step 6: gRPC API Service (Days 17-20)

**Objective**: Implement a gRPC API for exposing market data to other system components.

**Tasks**:
1. Add gRPC dependencies to Cargo.toml
   ```toml
   [dependencies]
   tonic = "0.9"
   prost = "0.11"
   
   [build-dependencies]
   tonic-build = "0.9"
   ```

2. Create Protocol Buffers definitions in `proto/market_data.proto`
   ```protobuf
   syntax = "proto3";
   package market_data;
   
   service MarketDataService {
     rpc GetOrderBook (OrderBookRequest) returns (OrderBookResponse);
     rpc GetTicker (TickerRequest) returns (TickerResponse);
     rpc GetRecentTrades (RecentTradesRequest) returns (RecentTradesResponse);
     rpc SubscribeToOrderBookUpdates (OrderBookSubscriptionRequest) returns (stream OrderBookUpdate);
     rpc SubscribeToTrades (TradeSubscriptionRequest) returns (stream TradeUpdate);
     rpc SubscribeToTickers (TickerSubscriptionRequest) returns (stream TickerUpdate);
   }
   
   message OrderBookRequest {
     string symbol = 1;
     int32 depth = 2;
   }
   
   message OrderBookResponse {
     string symbol = 1;
     uint64 last_update_id = 2;
     repeated OrderBookEntry bids = 3;
     repeated OrderBookEntry asks = 4;
     uint64 timestamp = 5;
   }
   
   message OrderBookEntry {
     double price = 1;
     double quantity = 2;
   }
   
   message TickerRequest {
     string symbol = 1;
   }
   
   message TickerResponse {
     string symbol = 1;
     double price_change = 2;
     double price_change_percent = 3;
     double weighted_avg_price = 4;
     double last_price = 5;
     double last_quantity = 6;
     double bid_price = 7;
     double bid_quantity = 8;
     double ask_price = 9;
     double ask_quantity = 10;
     double open_price = 11;
     double high_price = 12;
     double low_price = 13;
     double volume = 14;
     double quote_volume = 15;
     uint64 open_time = 16;
     uint64 close_time = 17;
     uint64 first_id = 18;
     uint64 last_id = 19;
     uint64 count = 20;
   }
   
   message RecentTradesRequest {
     string symbol = 1;
     int32 limit = 2;
   }
   
   message RecentTradesResponse {
     repeated Trade trades = 1;
   }
   
   message Trade {
     uint64 id = 1;
     string symbol = 2;
     double price = 3;
     double quantity = 4;
     uint64 buyer_order_id = 5;
     uint64 seller_order_id = 6;
     uint64 timestamp = 7;
     bool is_buyer_maker = 8;
   }
   
   message OrderBookSubscriptionRequest {
     string symbol = 1;
     int32 depth = 2;
     int32 update_speed = 3; // in milliseconds
   }
   
   message OrderBookUpdate {
     string symbol = 1;
     bool is_snapshot = 2;
     OrderBookResponse order_book = 3;
   }
   
   message TradeSubscriptionRequest {
     string symbol = 1;
   }
   
   message TradeUpdate {
     Trade trade = 1;
   }
   
   message TickerSubscriptionRequest {
     string symbol = 1;
   }
   
   message TickerUpdate {
     TickerResponse ticker = 1;
   }
   ```

3. Create build.rs file for code generation
   ```rust
   fn main() -> Result<(), Box<dyn std::error::Error>> {
       tonic_build::compile_protos("proto/market_data.proto")?;
       Ok(())
   }
   ```

4. Implement gRPC service in `src/api/grpc.rs`
   ```rust
   use std::pin::Pin;
   use std::sync::Arc;
   use std::time::Duration;
   
   use tokio::sync::mpsc;
   use tokio_stream::{Stream, StreamExt};
   use tonic::{Request, Response, Status};
   
   use crate::services::market_data_service::MarketDataService;
   
   // Import the generated code
   pub mod market_data {
       tonic::include_proto!("market_data");
   }
   
   use market_data::{
       market_data_service_server::{MarketDataService as GrpcMarketDataService, MarketDataServiceServer},
       OrderBookRequest, OrderBookResponse, OrderBookEntry, OrderBookUpdate,
       TickerRequest, TickerResponse, TickerUpdate,
       RecentTradesRequest, RecentTradesResponse, Trade, TradeUpdate,
       OrderBookSubscriptionRequest, TradeSubscriptionRequest, TickerSubscriptionRequest,
   };
   
   pub struct GrpcService {
       market_data_service: Arc<MarketDataService>,
   }
   
   impl GrpcService {
       pub fn new(market_data_service: Arc<MarketDataService>) -> Self {
           Self { market_data_service }
       }
       
       pub fn into_server(self) -> MarketDataServiceServer<Self> {
           MarketDataServiceServer::new(self)
       }
   }
   
   #[tonic::async_trait]
   impl GrpcMarketDataService for GrpcService {
       async fn get_order_book(
           &self,
           request: Request<OrderBookRequest>,
       ) -> Result<Response<OrderBookResponse>, Status> {
           let req = request.into_inner();
           let symbol = req.symbol;
           let depth = req.depth as usize;
           
           let order_book_manager = self.market_data_service.get_order_book_manager();
           
           match order_book_manager.get_market_depth(&symbol, depth) {
               Some((bids, asks)) => {
                   let order_book = OrderBookResponse {
                       symbol: symbol.clone(),
                       last_update_id: 0, // Not exposed in this API
                       bids: bids.into_iter().map(|(price, qty)| OrderBookEntry { price, quantity: qty }).collect(),
                       asks: asks.into_iter().map(|(price, qty)| OrderBookEntry { price, quantity: qty }).collect(),
                       timestamp: 0, // Not exposed in this API
                   };
                   
                   Ok(Response::new(order_book))
               }
               None => Err(Status::not_found(format!("Order book for symbol {} not found", symbol))),
           }
       }
       
       async fn get_ticker(
           &self,
           request: Request<TickerRequest>,
       ) -> Result<Response<TickerResponse>, Status> {
           let req = request.into_inner();
           let symbol = req.symbol;
           
           match self.market_data_service.get_ticker(&symbol) {
               Some(ticker) => {
                   let response = TickerResponse {
                       symbol: ticker.symbol,
                       price_change: ticker.price_change,
                       price_change_percent: ticker.price_change_percent,
                       weighted_avg_price: ticker.weighted_avg_price,
                       last_price: ticker.last_price,
                       last_quantity: ticker.last_quantity,
                       bid_price: ticker.bid_price,
                       bid_quantity: ticker.bid_quantity,
                       ask_price: ticker.ask_price,
                       ask_quantity: ticker.ask_quantity,
                       open_price: ticker.open_price,
                       high_price: ticker.high_price,
                       low_price: ticker.low_price,
                       volume: ticker.volume,
                       quote_volume: ticker.quote_volume,
                       open_time: ticker.open_time,
                       close_time: ticker.close_time,
                       first_id: ticker.first_id,
                       last_id: ticker.last_id,
                       count: ticker.count,
                   };
                   
                   Ok(Response::new(response))
               }
               None => Err(Status::not_found(format!("Ticker for symbol {} not found", symbol))),
           }
       }
       
       async fn get_recent_trades(
           &self,
           request: Request<RecentTradesRequest>,
       ) -> Result<Response<RecentTradesResponse>, Status> {
           let req = request.into_inner();
           let symbol = req.symbol;
           let limit = req.limit as usize;
           
           let trades = self.market_data_service.get_recent_trades(&symbol, limit);
           
           let response = RecentTradesResponse {
               trades: trades.into_iter().map(|t| Trade {
                   id: t.id,
                   symbol: t.symbol,
                   price: t.price,
                   quantity: t.quantity,
                   buyer_order_id: t.buyer_order_id,
                   seller_order_id: t.seller_order_id,
                   timestamp: t.timestamp,
                   is_buyer_maker: t.is_buyer_maker,
               }).collect(),
           };
           
           Ok(Response::new(response))
       }
       
       type SubscribeToOrderBookUpdatesStream = Pin<Box<dyn Stream<Item = Result<OrderBookUpdate, Status>> + Send + 'static>>;
       
       async fn subscribe_to_order_book_updates(
           &self,
           request: Request<OrderBookSubscriptionRequest>,
       ) -> Result<Response<Self::SubscribeToOrderBookUpdatesStream>, Status> {
           let req = request.into_inner();
           let symbol = req.symbol;
           let depth = req.depth as usize;
           let update_speed = req.update_speed as u64;
           
           let order_book_manager = self.market_data_service.get_order_book_manager();
           let mut receiver = order_book_manager.get_update_receiver();
           
           let (tx, rx) = mpsc::channel(100);
           
           tokio::spawn(async move {
               let mut last_update_time = std::time::Instant::now();
               
               while let Ok(update) = receiver.recv().await {
                   if update.symbol != symbol {
                       continue;
                   }
                   
                   // Rate limit updates
                   let now = std::time::Instant::now();
                   if !update.is_snapshot && now.duration_since(last_update_time).as_millis() < update_speed as u128 {
                       continue;
                   }
                   
                   last_update_time = now;
                   
                   // Get market depth with specified depth
                   let order_book = update.order_book.read().unwrap();
                   
                   let bids: Vec<OrderBookEntry> = order_book.bids.iter()
                       .rev()
                       .take(depth)
                       .map(|(price, qty)| OrderBookEntry { price: *price, quantity: *qty })
                       .collect();
                       
                   let asks: Vec<OrderBookEntry> = order_book.asks.iter()
                       .take(depth)
                       .map(|(price, qty)| OrderBookEntry { price: *price, quantity: *qty })
                       .collect();
                   
                   let response = OrderBookUpdate {
                       symbol: symbol.clone(),
                       is_snapshot: update.is_snapshot,
                       order_book: Some(OrderBookResponse {
                           symbol: symbol.clone(),
                           last_update_id: order_book.last_update_id,
                           bids,
                           asks,
                           timestamp: order_book.timestamp,
                       }),
                   };
                   
                   if tx.send(Ok(response)).await.is_err() {
                       break;
                   }
               }
           });
           
           let output_stream = tokio_stream::wrappers::ReceiverStream::new(rx);
           Ok(Response::new(Box::pin(output_stream) as Self::SubscribeToOrderBookUpdatesStream))
       }
       
       // Implement other streaming methods similarly
       // ...
   }
   ```

5. Create server implementation in `src/api/server.rs`
   ```rust
   use std::sync::Arc;
   use std::net::SocketAddr;
   use tonic::transport::Server;
   
   use crate::api::grpc::GrpcService;
   use crate::services::market_data_service::MarketDataService;
   
   pub struct ApiServer {
       market_data_service: Arc<MarketDataService>,
       addr: SocketAddr,
   }
   
   impl ApiServer {
       pub fn new(market_data_service: Arc<MarketDataService>, addr: SocketAddr) -> Self {
           Self {
               market_data_service,
               addr,
           }
       }
       
       pub async fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
           let grpc_service = GrpcService::new(self.market_data_service.clone());
           
           Server::builder()
               .add_service(grpc_service.into_server())
               .serve(self.addr)
               .await?;
               
           Ok(())
       }
   }
   ```

6. Implement unit and integration tests for gRPC API
7. Create benchmarks for API performance

**Deliverables**:
- Protocol Buffers definitions
- gRPC service implementation
- API server
- Unit and integration tests
- Performance benchmarks

**Integration Points**:
- Will expose market data to other system components
- Will be consumed by Node.js and Python services

### Step 7: Performance Optimization (Days 21-24)

**Objective**: Optimize the market data processor for maximum performance.

**Tasks**:
1. Implement memory pooling for frequently allocated objects
   ```rust
   // In src/utils/memory_pool.rs
   use std::sync::{Arc, Mutex};
   
   pub struct Pool<T> {
       items: Mutex<Vec<T>>,
       create: Box<dyn Fn() -> T + Send + Sync>,
   }
   
   impl<T> Pool<T> {
       pub fn new<F>(capacity: usize, create: F) -> Self
       where
           F: Fn() -> T + Send + Sync + 'static,
       {
           let mut items = Vec::with_capacity(capacity);
           for _ in 0..capacity {
               items.push(create());
           }
           
           Self {
               items: Mutex::new(items),
               create: Box::new(create),
           }
       }
       
       pub fn get(&self) -> PoolItem<T> {
           let item = {
               let mut items = self.items.lock().unwrap();
               if items.is_empty() {
                   (self.create)()
               } else {
                   items.pop().unwrap()
               }
           };
           
           PoolItem {
               item: Some(item),
               pool: self,
           }
       }
       
       fn put_back(&self, item: T) {
           let mut items = self.items.lock().unwrap();
           items.push(item);
       }
   }
   
   pub struct PoolItem<'a, T> {
       item: Option<T>,
       pool: &'a Pool<T>,
   }
   
   impl<'a, T> std::ops::Deref for PoolItem<'a, T> {
       type Target = T;
       
       fn deref(&self) -> &Self::Target {
           self.item.as_ref().unwrap()
       }
   }
   
   impl<'a, T> std::ops::DerefMut for PoolItem<'a, T> {
       fn deref_mut(&mut self) -> &mut Self::Target {
           self.item.as_mut().unwrap()
       }
   }
   
   impl<'a, T> Drop for PoolItem<'a, T> {
       fn drop(&mut self) {
           if let Some(item) = self.item.take() {
               self.pool.put_back(item);
           }
       }
   }
   ```

2. Implement lock-free data structures for high-contention paths
   ```rust
   // In src/utils/lock_free.rs
   use std::sync::atomic::{AtomicUsize, Ordering};
   
   pub struct Counter {
       value: AtomicUsize,
   }
   
   impl Counter {
       pub fn new(initial: usize) -> Self {
           Self {
               value: AtomicUsize::new(initial),
           }
       }
       
       pub fn increment(&self) -> usize {
           self.value.fetch_add(1, Ordering::SeqCst)
       }
       
       pub fn decrement(&self) -> usize {
           self.value.fetch_sub(1, Ordering::SeqCst)
       }
       
       pub fn get(&self) -> usize {
           self.value.load(Ordering::SeqCst)
       }
   }
   ```

3. Optimize message parsing with zero-copy techniques
4. Implement batch processing for multiple messages
5. Add performance metrics collection
6. Create comprehensive benchmarks for all critical paths
7. Optimize memory usage patterns

**Deliverables**:
- Memory pooling implementation
- Lock-free data structures
- Optimized message parsing
- Batch processing implementation
- Performance metrics collection
- Comprehensive benchmarks

**Integration Points**:
- Will improve performance across all components
- Will provide metrics for monitoring

### Step 8: Integration and Testing (Days 25-28)

**Objective**: Integrate all components and perform comprehensive testing.

**Tasks**:
1. Create main application entry point in `src/main.rs`
   ```rust
   use std::net::SocketAddr;
   use std::sync::Arc;
   use tokio::sync::{mpsc, watch};
   
   use market_data_processor::services::market_data_service::MarketDataService;
   use market_data_processor::api::server::ApiServer;
   
   #[tokio::main]
   async fn main() -> Result<(), Box<dyn std::error::Error>> {
       // Initialize logging
       tracing_subscriber::fmt::init();
       
       // Create channels
       let (message_tx, message_rx) = mpsc::channel(10000);
       let (shutdown_tx, shutdown_rx) = watch::channel(false);
       
       // Create market data service
       let market_data_service = Arc::new(MarketDataService::new(message_rx, shutdown_rx.clone()));
       
       // Create API server
       let addr = "0.0.0.0:50051".parse::<SocketAddr>()?;
       let api_server = ApiServer::new(market_data_service.clone(), addr);
       
       // Run services
       let market_data_handle = tokio::spawn(async move {
           market_data_service.run().await;
       });
       
       let api_server_handle = tokio::spawn(async move {
           api_server.run().await.unwrap();
       });
       
       // Wait for shutdown signal
       tokio::signal::ctrl_c().await?;
       
       // Initiate graceful shutdown
       shutdown_tx.send(true)?;
       
       // Wait for services to shut down
       let _ = tokio::join!(market_data_handle, api_server_handle);
       
       Ok(())
   }
   ```

2. Implement integration tests for the entire pipeline
3. Create Docker configuration for containerization
   ```dockerfile
   # Dockerfile
   FROM rust:1.68 as builder
   
   WORKDIR /usr/src/market-data-processor
   COPY . .
   
   RUN cargo build --release
   
   FROM debian:bullseye-slim
   
   RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
   
   COPY --from=builder /usr/src/market-data-processor/target/release/market-data-processor /usr/local/bin/market-data-processor
   
   EXPOSE 50051
   
   CMD ["market-data-processor"]
   ```

4. Set up CI/CD pipeline with GitHub Actions
   ```yaml
   # .github/workflows/ci.yml
   name: CI
   
   on:
     push:
       branches: [ main ]
     pull_request:
       branches: [ main ]
   
   jobs:
     build:
       runs-on: ubuntu-latest
       steps:
       - uses: actions/checkout@v3
       - name: Install Rust
         uses: actions-rs/toolchain@v1
         with:
           toolchain: stable
           override: true
       - name: Build
         uses: actions-rs/cargo@v1
         with:
           command: build
       - name: Run tests
         uses: actions-rs/cargo@v1
         with:
           command: test
       - name: Run clippy
         uses: actions-rs/cargo@v1
         with:
           command: clippy
           args: -- -D warnings
   ```

5. Create comprehensive documentation
6. Implement health checks and monitoring
7. Set up logging and error reporting

**Deliverables**:
- Main application entry point
- Integration tests
- Docker configuration
- CI/CD pipeline
- Comprehensive documentation
- Health checks and monitoring
- Logging and error reporting

**Integration Points**:
- Will integrate all components into a cohesive system
- Will provide deployment configuration for production

## Conclusion

This implementation plan provides a step-by-step approach to building the Market Data Processor component in Rust. The plan is designed to be incremental, with each step building on the previous ones to create a high-performance, reliable system for processing market data from the MEXC exchange.

The implementation focuses on:

1. **Performance**: Utilizing Rust's zero-cost abstractions and memory safety for high-throughput data processing
2. **Reliability**: Implementing robust error handling and recovery mechanisms
3. **Scalability**: Designing for horizontal scaling with clean separation of concerns
4. **Interoperability**: Providing clear APIs for integration with other system components

By following this plan, you will create a Market Data Processor that forms the foundation of your trading system, providing accurate, real-time market data for decision-making and execution.
