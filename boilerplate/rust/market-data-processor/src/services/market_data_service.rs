use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tokio::sync::{broadcast, mpsc, watch};
use tokio::task;
use tracing::{debug, error, info, warn};

use crate::models::{
    order_book::OrderBook,
    trade::Trade,
    ticker::Ticker,
    websocket::{WebSocketMessage, OrderBookUpdateData, TradeData, TickerData},
};
use crate::services::{
    message_parser::MessageParser,
    order_book_manager::OrderBookManager,
    websocket_client::{WebSocketClient, create_paper_trading_client},
};
use crate::utils::config::Config;

pub struct MarketDataService {
    order_book_manager: Arc<OrderBookManager>,
    trades: Arc<RwLock<HashMap<String, Vec<Trade>>>>,
    tickers: Arc<RwLock<HashMap<String, Ticker>>>,
    message_receiver: mpsc::Receiver<String>,
    shutdown: watch::Receiver<bool>,
    config: Arc<Config>,
    is_paper_trading: bool,
}

impl MarketDataService {
    pub fn new(
        message_receiver: mpsc::Receiver<String>,
        shutdown: watch::Receiver<bool>,
    ) -> Self {
        // Load config
        let config = Config::load().expect("Failed to load configuration");
        
        Self {
            order_book_manager: Arc::new(OrderBookManager::new()),
            trades: Arc::new(RwLock::new(HashMap::new())),
            tickers: Arc::new(RwLock::new(HashMap::new())),
            message_receiver,
            shutdown,
            config: Arc::new(config),
            is_paper_trading: true, // Default to paper trading mode
        }
    }
    
    pub fn with_paper_trading(mut self, is_paper_trading: bool) -> Self {
        self.is_paper_trading = is_paper_trading;
        self
    }
    
    pub fn get_order_book_manager(&self) -> Arc<OrderBookManager> {
        self.order_book_manager.clone()
    }
    
    pub async fn run(&self) {
        info!("Starting Market Data Service in {} mode", 
            if self.is_paper_trading { "paper trading" } else { "live trading" });
            
        while !*self.shutdown.borrow() {
            tokio::select! {
                Some(message) = self.message_receiver.recv() => {
                    self.process_message(&message).await;
                }
                _ = self.shutdown.changed() => {
                    if *self.shutdown.borrow() {
                        info!("Shutdown signal received, stopping Market Data Service");
                        break;
                    }
                }
            }
        }
        
        info!("Market Data Service stopped");
    }
    
    async fn process_message(&self, message: &str) {
        // Parse the WebSocket message
        let ws_message = match MessageParser::parse_message(message) {
            Ok(msg) => msg,
            Err(e) => {
                error!("Failed to parse WebSocket message: {}", e);
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
            debug!("Ignoring message from unsupported channel: {}", ws_message.c);
        }
    }
    
    async fn process_order_book_message(&self, message: &WebSocketMessage) {
        let data = match MessageParser::parse_order_book_update(message) {
            Ok(data) => data,
            Err(e) => {
                error!("Failed to parse order book update: {}", e);
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
                error!("Failed to parse trade data: {}", e);
                return;
            }
        };
        
        let trade = MessageParser::trade_data_to_trade(&data, &message.s);
        
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
                error!("Failed to parse ticker data: {}", e);
                return;
            }
        };
        
        let ticker = MessageParser::ticker_data_to_ticker(&data, &message.s);
        
        // Store ticker
        let mut tickers = self.tickers.write().unwrap();
        tickers.insert(message.s.clone(), ticker);
    }
    
    // Connect to MEXC WebSocket API and start processing market data
    pub async fn connect_to_exchange(&self, symbols: Vec<String>) -> Result<(), Box<dyn std::error::Error>> {
        let (ws_message_tx, _) = mpsc::channel(10000);
        
        let client = create_paper_trading_client(
            &self.config,
            symbols,
            ws_message_tx,
            self.shutdown.clone()
        ).await?;
        
        client.connect_and_subscribe().await?;
        
        Ok(())
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
    
    pub fn is_paper_trading_mode(&self) -> bool {
        self.is_paper_trading
    }
}
