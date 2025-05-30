use crate::models::order_book::OrderBook;
use crate::models::signal::{Signal, SignalSource, SignalStrength, SignalType};
use crate::services::optimized_rest_client::OptimizedRestClient;
use crate::utils::enhanced_config::EnhancedConfig;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use ta::{indicators::{RelativeStrengthIndex, MovingAverageConvergenceDivergence, BollingerBands}, Next};
use tokio::sync::{mpsc, RwLock};
use tokio::time::interval;
use tracing::{debug, error, info, warn};

/// Maximum number of signals to store in history
const MAX_SIGNAL_HISTORY: usize = 100;

/// Maximum number of price points to store for technical analysis
const MAX_PRICE_HISTORY: usize = 500;

/// Data structure to hold price history for technical analysis
#[derive(Debug, Clone)]
struct PriceHistory {
    symbol: String,
    timestamps: VecDeque<u64>,
    close_prices: VecDeque<f64>,
    high_prices: VecDeque<f64>,
    low_prices: VecDeque<f64>,
    volumes: VecDeque<f64>,
    last_update: Instant,
}

impl PriceHistory {
    /// Create a new price history
    fn new(symbol: &str) -> Self {
        Self {
            symbol: symbol.to_string(),
            timestamps: VecDeque::with_capacity(MAX_PRICE_HISTORY),
            close_prices: VecDeque::with_capacity(MAX_PRICE_HISTORY),
            high_prices: VecDeque::with_capacity(MAX_PRICE_HISTORY),
            low_prices: VecDeque::with_capacity(MAX_PRICE_HISTORY),
            volumes: VecDeque::with_capacity(MAX_PRICE_HISTORY),
            last_update: Instant::now(),
        }
    }
    
    /// Add a new price point
    fn add_price_point(&mut self, timestamp: u64, close: f64, high: f64, low: f64, volume: f64) {
        self.timestamps.push_back(timestamp);
        self.close_prices.push_back(close);
        self.high_prices.push_back(high);
        self.low_prices.push_back(low);
        self.volumes.push_back(volume);
        
        // Maintain maximum size
        if self.timestamps.len() > MAX_PRICE_HISTORY {
            self.timestamps.pop_front();
            self.close_prices.pop_front();
            self.high_prices.pop_front();
            self.low_prices.pop_front();
            self.volumes.pop_front();
        }
        
        self.last_update = Instant::now();
    }
    
    /// Get the last price
    fn last_price(&self) -> Option<f64> {
        self.close_prices.back().copied()
    }
    
    /// Get price history vector for technical analysis
    fn get_close_prices(&self) -> Vec<f64> {
        self.close_prices.iter().copied().collect()
    }
    
    /// Check if price history is stale
    fn is_stale(&self, max_age: Duration) -> bool {
        self.last_update.elapsed() > max_age
    }
}

/// Signal generator service
pub struct SignalGenerator {
    config: EnhancedConfig,
    rest_client: Arc<OptimizedRestClient>,
    price_history: Arc<RwLock<HashMap<String, PriceHistory>>>,
    signal_history: Arc<RwLock<HashMap<String, VecDeque<Signal>>>>,
    signal_sender: mpsc::Sender<Signal>,
}

impl SignalGenerator {
    /// Create a new signal generator
    pub fn new(
        config: EnhancedConfig,
        rest_client: Arc<OptimizedRestClient>,
        signal_sender: mpsc::Sender<Signal>,
    ) -> Self {
        Self {
            config,
            rest_client,
            price_history: Arc::new(RwLock::new(HashMap::new())),
            signal_history: Arc::new(RwLock::new(HashMap::new())),
            signal_sender,
        }
    }
    
    /// Start the signal generator
    pub async fn start(&self) {
        info!("Starting signal generator service");
        
        // Start price history collection for each trading pair
        self.start_price_collection().await;
        
        // Start technical analysis for each trading pair
        self.start_technical_analysis().await;
        
        // Start order book analysis for each trading pair
        self.start_order_book_analysis().await;
    }
    
    /// Start collecting price history
    async fn start_price_collection(&self) {
        let config = self.config.clone();
        let rest_client = self.rest_client.clone();
        let price_history = self.price_history.clone();
        
        // Initialize price history for each trading pair
        {
            let mut history = price_history.write().await;
            for pair in &config.trading_pairs {
                history.insert(pair.clone(), PriceHistory::new(pair));
            }
        }
        
        // Start collection task
        tokio::spawn(async move {
            let mut interval_timer = interval(Duration::from_secs(10));
            
            loop {
                interval_timer.tick().await;
                
                for pair in &config.trading_pairs {
                    match rest_client.get_ticker(pair).await {
                        Ok(ticker) => {
                            // Extract price data from ticker
                            let close_price = ticker["lastPrice"]
                                .as_str()
                                .and_then(|p| p.parse::<f64>().ok())
                                .unwrap_or_default();
                                
                            let high_price = ticker["highPrice"]
                                .as_str()
                                .and_then(|p| p.parse::<f64>().ok())
                                .unwrap_or(close_price);
                                
                            let low_price = ticker["lowPrice"]
                                .as_str()
                                .and_then(|p| p.parse::<f64>().ok())
                                .unwrap_or(close_price);
                                
                            let volume = ticker["volume"]
                                .as_str()
                                .and_then(|p| p.parse::<f64>().ok())
                                .unwrap_or_default();
                                
                            let timestamp = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_millis() as u64;
                            
                            // Update price history
                            let mut history = price_history.write().await;
                            if let Some(price_history) = history.get_mut(pair) {
                                price_history.add_price_point(timestamp, close_price, high_price, low_price, volume);
                                debug!("Updated price history for {}: price={}, volume={}", pair, close_price, volume);
                            }
                        },
                        Err(e) => {
                            error!("Failed to get ticker for {}: {}", pair, e);
                        }
                    }
                }
            }
        });
    }
    
    /// Start technical analysis
    async fn start_technical_analysis(&self) {
        let config = self.config.clone();
        let price_history = self.price_history.clone();
        let signal_history = self.signal_history.clone();
        let signal_sender = self.signal_sender.clone();
        
        // Initialize signal history for each trading pair
        {
            let mut history = signal_history.write().await;
            for pair in &config.trading_pairs {
                history.insert(pair.clone(), VecDeque::with_capacity(MAX_SIGNAL_HISTORY));
            }
        }
        
        // Start analysis task
        tokio::spawn(async move {
            let mut interval_timer = interval(Duration::from_secs(30));
            
            loop {
                interval_timer.tick().await;
                
                for pair in &config.trading_pairs {
                    let prices_opt = {
                        let history = price_history.read().await;
                        history.get(pair).map(|h| h.get_close_prices())
                    };
                    
                    if let Some(prices) = prices_opt {
                        if prices.len() < 30 {
                            debug!("Not enough price history for {} to generate signals", pair);
                            continue;
                        }
                        
                        // Generate signals using technical indicators
                        let rsi_signal = generate_rsi_signal(pair, &prices);
                        let macd_signal = generate_macd_signal(pair, &prices);
                        let bb_signal = generate_bollinger_bands_signal(pair, &prices);
                        
                        // Combine signals
                        let combined_signal = combine_signals(pair, &[rsi_signal, macd_signal, bb_signal]);
                        
                        if combined_signal.signal_type != SignalType::Neutral {
                            info!("Generated {} signal for {}: {:?}", 
                                combined_signal.signal_type, pair, combined_signal);
                                
                            // Add to history
                            {
                                let mut history = signal_history.write().await;
                                if let Some(signals) = history.get_mut(pair) {
                                    signals.push_back(combined_signal.clone());
                                    
                                    // Maintain maximum size
                                    if signals.len() > MAX_SIGNAL_HISTORY {
                                        signals.pop_front();
                                    }
                                }
                            }
                            
                            // Send signal
                            if let Err(e) = signal_sender.send(combined_signal).await {
                                error!("Failed to send signal for {}: {}", pair, e);
                            }
                        }
                    }
                }
            }
        });
    }
    
    /// Start order book analysis
    async fn start_order_book_analysis(&self) {
        let config = self.config.clone();
        let rest_client = self.rest_client.clone();
        let signal_history = self.signal_history.clone();
        let signal_sender = self.signal_sender.clone();
        
        // Start analysis task
        tokio::spawn(async move {
            let mut interval_timer = interval(Duration::from_secs(15));
            
            loop {
                interval_timer.tick().await;
                
                for pair in &config.trading_pairs {
                    match rest_client.get_order_book(pair, 100).await {
                        Ok(order_book) => {
                            // Analyze order book for signals
                            let order_book_signal = analyze_order_book(pair, &order_book);
                            
                            if order_book_signal.signal_type != SignalType::Neutral {
                                info!("Generated order book signal for {}: {:?}", pair, order_book_signal);
                                
                                // Add to history
                                {
                                    let mut history = signal_history.write().await;
                                    if let Some(signals) = history.get_mut(pair) {
                                        signals.push_back(order_book_signal.clone());
                                        
                                        // Maintain maximum size
                                        if signals.len() > MAX_SIGNAL_HISTORY {
                                            signals.pop_front();
                                        }
                                    }
                                }
                                
                                // Send signal
                                if let Err(e) = signal_sender.send(order_book_signal).await {
                                    error!("Failed to send order book signal for {}: {}", pair, e);
                                }
                            }
                        },
                        Err(e) => {
                            error!("Failed to get order book for {}: {}", pair, e);
                        }
                    }
                }
            }
        });
    }
    
    /// Get recent signals for a trading pair
    pub async fn get_recent_signals(&self, symbol: &str) -> Vec<Signal> {
        let history = self.signal_history.read().await;
        history
            .get(symbol)
            .map(|signals| signals.iter().cloned().collect())
            .unwrap_or_default()
    }
}

/// Generate RSI signal
fn generate_rsi_signal(symbol: &str, prices: &[f64]) -> Signal {
    // Initialize RSI with period 14
    let mut rsi = RelativeStrengthIndex::new(14).unwrap();
    
    // Calculate RSI for all prices
    let mut rsi_values = Vec::with_capacity(prices.len());
    for &price in prices {
        rsi_values.push(rsi.next(price));
    }
    
    // Get latest RSI value
    let latest_rsi = *rsi_values.last().unwrap_or(&50.0);
    
    // Generate signal based on RSI value
    let (signal_type, strength) = if latest_rsi < 30.0 {
        // Oversold - potential buy
        (SignalType::Buy, SignalStrength::from_f64((30.0 - latest_rsi) / 30.0))
    } else if latest_rsi > 70.0 {
        // Overbought - potential sell
        (SignalType::Sell, SignalStrength::from_f64((latest_rsi - 70.0) / 30.0))
    } else {
        // Neutral
        (SignalType::Neutral, SignalStrength::Weak)
    };
    
    Signal {
        symbol: symbol.to_string(),
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64,
        signal_type,
        strength,
        source: SignalSource::RSI,
        price: prices.last().copied(),
        metadata: Some(format!("RSI(14): {:.2}", latest_rsi)),
    }
}

/// Generate MACD signal
fn generate_macd_signal(symbol: &str, prices: &[f64]) -> Signal {
    // Initialize MACD with standard parameters (12, 26, 9)
    let mut macd = MovingAverageConvergenceDivergence::new(12, 26, 9).unwrap();
    
    // Calculate MACD for all prices
    let mut macd_values = Vec::with_capacity(prices.len());
    let mut signal_values = Vec::with_capacity(prices.len());
    let mut histogram_values = Vec::with_capacity(prices.len());
    
    for &price in prices {
        let macd_result = macd.next(price);
        macd_values.push(macd_result.macd);
        signal_values.push(macd_result.signal);
        histogram_values.push(macd_result.histogram);
    }
    
    // Get latest values
    let latest_macd = *macd_values.last().unwrap_or(&0.0);
    let latest_signal = *signal_values.last().unwrap_or(&0.0);
    let latest_histogram = *histogram_values.last().unwrap_or(&0.0);
    
    // Check for MACD crossover
    let (signal_type, strength) = if macd_values.len() >= 2 && signal_values.len() >= 2 {
        let prev_macd = macd_values[macd_values.len() - 2];
        let prev_signal = signal_values[signal_values.len() - 2];
        
        if prev_macd <= prev_signal && latest_macd > latest_signal {
            // Bullish crossover
            (SignalType::Buy, SignalStrength::from_f64(latest_histogram.abs()))
        } else if prev_macd >= prev_signal && latest_macd < latest_signal {
            // Bearish crossover
            (SignalType::Sell, SignalStrength::from_f64(latest_histogram.abs()))
        } else {
            // No crossover
            (SignalType::Neutral, SignalStrength::Weak)
        }
    } else {
        (SignalType::Neutral, SignalStrength::Weak)
    };
    
    Signal {
        symbol: symbol.to_string(),
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64,
        signal_type,
        strength,
        source: SignalSource::MACD,
        price: prices.last().copied(),
        metadata: Some(format!("MACD: {:.2}, Signal: {:.2}, Histogram: {:.2}", 
            latest_macd, latest_signal, latest_histogram)),
    }
}

/// Generate Bollinger Bands signal
fn generate_bollinger_bands_signal(symbol: &str, prices: &[f64]) -> Signal {
    // Initialize Bollinger Bands with period 20 and 2 standard deviations
    let mut bb = BollingerBands::new(20, 2.0).unwrap();
    
    // Calculate Bollinger Bands for all prices
    let mut bb_values = Vec::with_capacity(prices.len());
    for &price in prices {
        bb_values.push(bb.next(price));
    }
    
    // Get latest values
    let latest_bb = bb_values.last().unwrap_or(&bb.next(prices.last().copied().unwrap_or(0.0)));
    let latest_price = prices.last().copied().unwrap_or(0.0);
    
    // Generate signal based on price position relative to bands
    let (signal_type, strength) = if latest_price < latest_bb.lower {
        // Price below lower band - potential buy (oversold)
        let strength_factor = (latest_bb.lower - latest_price) / (latest_bb.upper - latest_bb.lower);
        (SignalType::Buy, SignalStrength::from_f64(strength_factor))
    } else if latest_price > latest_bb.upper {
        // Price above upper band - potential sell (overbought)
        let strength_factor = (latest_price - latest_bb.upper) / (latest_bb.upper - latest_bb.lower);
        (SignalType::Sell, SignalStrength::from_f64(strength_factor))
    } else {
        // Price within bands - neutral
        (SignalType::Neutral, SignalStrength::Weak)
    };
    
    Signal {
        symbol: symbol.to_string(),
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64,
        signal_type,
        strength,
        source: SignalSource::BollingerBands,
        price: Some(latest_price),
        metadata: Some(format!("BB: Upper={:.2}, Middle={:.2}, Lower={:.2}", 
            latest_bb.upper, latest_bb.middle, latest_bb.lower)),
    }
}

/// Analyze order book for signals
fn analyze_order_book(symbol: &str, order_book: &OrderBook) -> Signal {
    // Calculate total bid and ask volume
    let total_bid_volume: f64 = order_book.bids.levels.iter().map(|&(_, qty)| qty).sum();
    let total_ask_volume: f64 = order_book.asks.levels.iter().map(|&(_, qty)| qty).sum();
    
    // Calculate volume imbalance
    let total_volume = total_bid_volume + total_ask_volume;
    let bid_ratio = if total_volume > 0.0 { total_bid_volume / total_volume } else { 0.5 };
    
    // Calculate bid-ask spread
    let spread = if !order_book.bids.levels.is_empty() && !order_book.asks.levels.is_empty() {
        order_book.asks.levels[0].0 - order_book.bids.levels[0].0
    } else {
        0.0
    };
    
    // Calculate mid price
    let mid_price = if !order_book.bids.levels.is_empty() && !order_book.asks.levels.is_empty() {
        (order_book.asks.levels[0].0 + order_book.bids.levels[0].0) / 2.0
    } else {
        0.0
    };
    
    // Generate signal based on volume imbalance
    let (signal_type, strength) = if bid_ratio > 0.6 {
        // More bids than asks - potential buy
        (SignalType::Buy, SignalStrength::from_f64((bid_ratio - 0.5) * 2.0))
    } else if bid_ratio < 0.4 {
        // More asks than bids - potential sell
        (SignalType::Sell, SignalStrength::from_f64((0.5 - bid_ratio) * 2.0))
    } else {
        // Balanced - neutral
        (SignalType::Neutral, SignalStrength::Weak)
    };
    
    Signal {
        symbol: symbol.to_string(),
        timestamp: order_book.timestamp,
        signal_type,
        strength,
        source: SignalSource::OrderBook,
        price: Some(mid_price),
        metadata: Some(format!("Bid Vol: {:.2}, Ask Vol: {:.2}, Spread: {:.2}, Imbalance: {:.2}%", 
            total_bid_volume, total_ask_volume, spread, (bid_ratio - 0.5) * 100.0)),
    }
}

/// Combine multiple signals into a single signal
fn combine_signals(symbol: &str, signals: &[Signal]) -> Signal {
    let mut buy_strength = 0.0;
    let mut sell_strength = 0.0;
    let mut total_signals = 0;
    let mut latest_price = None;
    
    // Combine signal strengths
    for signal in signals {
        if signal.symbol == symbol {
            match signal.signal_type {
                SignalType::Buy => {
                    buy_strength += signal.strength.as_f64();
                    total_signals += 1;
                },
                SignalType::Sell => {
                    sell_strength += signal.strength.as_f64();
                    total_signals += 1;
                },
                _ => {}
            }
            
            // Use the latest available price
            if signal.price.is_some() {
                latest_price = signal.price;
            }
        }
    }
    
    // Calculate average strengths
    if total_signals > 0 {
        buy_strength /= total_signals as f64;
        sell_strength /= total_signals as f64;
    }
    
    // Determine overall signal
    let (signal_type, strength) = if buy_strength > sell_strength && buy_strength > 0.3 {
        (SignalType::Buy, SignalStrength::from_f64(buy_strength))
    } else if sell_strength > buy_strength && sell_strength > 0.3 {
        (SignalType::Sell, SignalStrength::from_f64(sell_strength))
    } else {
        (SignalType::Neutral, SignalStrength::Weak)
    };
    
    // Create metadata string with all component signals
    let mut metadata = String::new();
    for signal in signals {
        if !metadata.is_empty() {
            metadata.push_str(" | ");
        }
        metadata.push_str(&format!("{:?}: {:?}", signal.source, signal.signal_type));
        if let Some(ref meta) = signal.metadata {
            metadata.push_str(&format!(" ({})", meta));
        }
    }
    
    Signal {
        symbol: symbol.to_string(),
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64,
        signal_type,
        strength,
        source: SignalSource::Combined,
        price: latest_price,
        metadata: Some(metadata),
    }
}
