use crate::models::order::{Order, OrderFill, OrderSide, OrderStatus, OrderType, TimeInForce};
use crate::models::order_book::OrderBook;
use crate::models::position::{Position, PositionDirection, PositionTrade};
use crate::models::signal::{Signal, SignalType};
use crate::services::optimized_rest_client::OptimizedRestClient;
use crate::services::paper_trading_metrics;
use crate::utils::enhanced_config::EnhancedConfig;
use chrono::{DateTime, Duration, Utc};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{mpsc, RwLock};
use tokio::time::interval;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// MEXC fee structure (taker fee)
const MEXC_TAKER_FEE: f64 = 0.001; // 0.1%

/// MEXC fee structure (maker fee)
const MEXC_MAKER_FEE: f64 = 0.0008; // 0.08%

/// Default latency range for order simulation (milliseconds)
const DEFAULT_MIN_LATENCY_MS: u64 = 50;
const DEFAULT_MAX_LATENCY_MS: u64 = 200;

/// Default slippage factors
const DEFAULT_MIN_SLIPPAGE: f64 = 0.0001; // 0.01%
const DEFAULT_MAX_SLIPPAGE: f64 = 0.001;  // 0.1%

/// Performance record for a trading session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingPerformance {
    pub symbol: String,
    pub start_time: DateTime<Utc>,
    pub end_time: Option<DateTime<Utc>>,
    pub initial_balance: f64,
    pub final_balance: Option<f64>,
    pub total_trades: usize,
    pub winning_trades: usize,
    pub losing_trades: usize,
    pub total_pnl: f64,
    pub total_fees: f64,
    pub max_drawdown: f64,
    pub max_position_size: f64,
    pub roi: f64,
    pub sharpe_ratio: Option<f64>,
}

/// Paper trading service for simulating trades against real market data
pub struct PaperTradingService {
    config: EnhancedConfig,
    rest_client: Arc<OptimizedRestClient>,
    account_balance: Arc<RwLock<HashMap<String, f64>>>,
    prices: Arc<RwLock<HashMap<String, f64>>>,
    volumes: Arc<RwLock<HashMap<String, f64>>>,
    positions: Arc<RwLock<HashMap<String, Position>>>,
    open_orders: Arc<RwLock<HashMap<String, Order>>>,
    completed_orders: Arc<RwLock<Vec<Order>>>,
    performance: Arc<RwLock<HashMap<String, TradingPerformance>>>,
    signal_receiver: mpsc::Receiver<Signal>,
    min_latency_ms: u64,
    max_latency_ms: u64,
    min_slippage: f64,
    max_slippage: f64,
}

impl PaperTradingService {
    /// Create a new paper trading service
    pub fn new(
        config: EnhancedConfig,
        rest_client: Arc<OptimizedRestClient>,
        signal_receiver: mpsc::Receiver<Signal>,
    ) -> Self {
        let mut account_balance = HashMap::new();
        
        // Initialize with default balances
        // We'll use USDT instead of USDC for consistency with dashboard
        account_balance.insert("BTC".to_string(), 1.0); 
        account_balance.insert("ETH".to_string(), 10.0);
        account_balance.insert("USDT".to_string(), 100000.0);
        
        Self {
            config: config.clone(),
            rest_client,
            account_balance: Arc::new(RwLock::new(account_balance)),
            prices: Arc::new(RwLock::new(HashMap::new())),
            volumes: Arc::new(RwLock::new(HashMap::new())),
            positions: Arc::new(RwLock::new(HashMap::new())),
            open_orders: Arc::new(RwLock::new(HashMap::new())),
            completed_orders: Arc::new(RwLock::new(Vec::new())),
            performance: Arc::new(RwLock::new(HashMap::new())),
            signal_receiver,
            min_latency_ms: DEFAULT_MIN_LATENCY_MS,
            max_latency_ms: DEFAULT_MAX_LATENCY_MS,
            min_slippage: DEFAULT_MIN_SLIPPAGE,
            max_slippage: DEFAULT_MAX_SLIPPAGE,
        }
    }
    
    /// Start the paper trading service
    pub async fn start(&mut self) {
        info!("Starting paper trading service");
        
        // Initialize performance tracking and metrics
        self.initialize_performance().await;
        
        // Start the order processing loop
        let order_processor = self.start_order_processor();
        
        // Start the signal processing loop
        let signal_processor = self.start_signal_processor();
        
        // Start the price update loop
        let price_updater = self.start_price_updater();
        
        // Wait for all tasks to complete (they should run indefinitely)
        tokio::select! {
            _ = order_processor => error!("Order processor task ended unexpectedly"),
            _ = signal_processor => error!("Signal processor task ended unexpectedly"),
            _ = price_updater => error!("Price updater task ended unexpectedly"),
        }
    }
    
    /// Initialize performance tracking and metrics
    async fn initialize_performance(&self) {
        let mut performance = self.performance.write().await;
        
        for pair in &self.config.trading_pairs {
            let now = Utc::now();
            let initial_balance = 100000.0; // Example initial balance (USDC)
            
            performance.insert(pair.clone(), TradingPerformance {
                symbol: pair.clone(),
                start_time: now,
                end_time: None,
                initial_balance,
                final_balance: None,
                total_trades: 0,
                winning_trades: 0,
                losing_trades: 0,
                total_pnl: 0.0,
                total_fees: 0.0,
                max_drawdown: 0.0,
                max_position_size: 0.0,
                roi: 0.0,
                sharpe_ratio: None,
            });
        }
        
        // Initialize metrics for dashboard
        let account_balance = self.account_balance.read().await;
        paper_trading_metrics::initialize_metrics(&self.config.trading_pairs, &account_balance);
    }
    
    /// Start the order processing loop
    async fn start_order_processor(&self) -> tokio::task::JoinHandle<()> {
        let open_orders = self.open_orders.clone();
        let completed_orders = self.completed_orders.clone();
        let positions = self.positions.clone();
        let rest_client = self.rest_client.clone();
        let account_balance = self.account_balance.clone();
        
        tokio::spawn(async move {
            let mut interval_timer = interval(std::time::Duration::from_millis(100));
            
            loop {
                interval_timer.tick().await;
                
                // Process orders that are due for execution
                let orders_to_process = {
                    let orders = open_orders.read().await;
                    orders
                        .values()
                        .filter(|order| order.is_active() && order.is_execution_due())
                        .cloned()
                        .collect::<Vec<Order>>()
                };
                
                for mut order in orders_to_process {
                    // Get the latest order book for the symbol
                    match rest_client.get_order_book(&order.symbol, 100).await {
                        Ok(order_book) => {
                            // Execute the order against the order book
                            Self::execute_order_against_book(&mut order, &order_book);
                            
                            // Update position
                            if order.executed_qty > 0.0 {
                                let mut positions_lock = positions.write().await;
                                let position = positions_lock
                                    .entry(order.symbol.clone())
                                    .or_insert_with(|| Position::new(&order.symbol));
                                
                                // Process each fill
                                for fill in &order.fills {
                                    let trade = PositionTrade::from_order(
                                        &order,
                                        fill.quantity,
                                        fill.price,
                                        fill.commission,
                                        &fill.commission_asset,
                                    );
                                    position.add_trade(trade);
                                }
                                
                                // Update account balance
                                if order.is_filled() {
                                    let mut balance = account_balance.write().await;
                                    
                                    // Extract base and quote asset from symbol (e.g., BTCUSDC -> BTC, USDC)
                                    let base_asset = order.symbol.split("USDC").next().unwrap_or("BTC");
                                    let quote_asset = "USDC";
                                    
                                    // Update balances based on order side
                                    match order.side {
                                        OrderSide::Buy => {
                                            // Increase base asset
                                            *balance.entry(base_asset.to_string()).or_insert(0.0) += order.executed_qty;
                                            // Decrease quote asset (including fees)
                                            let cost = order.executed_qty * order.avg_execution_price.unwrap_or(0.0);
                                            *balance.entry(quote_asset.to_string()).or_insert(0.0) -= (cost + order.cumulative_commission);
                                        },
                                        OrderSide::Sell => {
                                            // Decrease base asset
                                            *balance.entry(base_asset.to_string()).or_insert(0.0) -= order.executed_qty;
                                            // Increase quote asset (minus fees)
                                            let revenue = order.executed_qty * order.avg_execution_price.unwrap_or(0.0);
                                            *balance.entry(quote_asset.to_string()).or_insert(0.0) += (revenue - order.cumulative_commission);
                                        },
                                    }
                                }
                            }
                            
                            // Move completed order
                            if order.is_filled() || order.status == OrderStatus::Canceled || order.status == OrderStatus::Rejected {
                                let mut open_orders_lock = open_orders.write().await;
                                open_orders_lock.remove(&order.id);
                                
                                let mut completed_orders_lock = completed_orders.write().await;
                                completed_orders_lock.push(order.clone());
                                
                                info!("Order {} completed with status {:?}", order.id, order.status);
                            } else {
                                // Update order in open orders
                                let mut open_orders_lock = open_orders.write().await;
                                open_orders_lock.insert(order.id.clone(), order);
                            }
                        },
                        Err(e) => {
                            error!("Failed to get order book for {}: {}", order.symbol, e);
                            // Mark order as rejected if we can't get the order book
                            order.reject(&format!("Failed to get order book: {}", e));
                            
                            let mut open_orders_lock = open_orders.write().await;
                            open_orders_lock.remove(&order.id);
                            
                            let mut completed_orders_lock = completed_orders.write().await;
                            completed_orders_lock.push(order);
                        }
                    }
                }
            }
        })
    }
    
    /// Execute an order against the order book
    fn execute_order_against_book(order: &mut Order, order_book: &OrderBook) {
        match order.order_type {
            OrderType::Market => {
                // Simulate market order execution
                Self::execute_market_order(order, order_book);
            },
            OrderType::Limit => {
                // Simulate limit order execution
                Self::execute_limit_order(order, order_book);
            },
            OrderType::StopLoss => {
                // Check if stop price is reached
                if let Some(stop_price) = order.stop_price {
                    let triggered = match order.side {
                        OrderSide::Buy => {
                            if !order_book.asks.levels.is_empty() {
                                let best_ask = order_book.asks.levels[0].0;
                                best_ask >= stop_price
                            } else {
                                false
                            }
                        },
                        OrderSide::Sell => {
                            if !order_book.bids.levels.is_empty() {
                                let best_bid = order_book.bids.levels[0].0;
                                best_bid <= stop_price
                            } else {
                                false
                            }
                        },
                    };
                    
                    if triggered {
                        // Convert to market order and execute
                        order.order_type = OrderType::Market;
                        Self::execute_market_order(order, order_book);
                    }
                }
            },
            _ => {
                // Other order types not implemented yet
                warn!("Order type {:?} not implemented yet", order.order_type);
            }
        }
    }
    
    /// Execute a market order against the order book
    fn execute_market_order(order: &mut Order, order_book: &OrderBook) {
        // Remaining quantity to execute
        let mut remaining_qty = order.remaining_qty();
        
        // Get the relevant side of the order book
        let levels = match order.side {
            OrderSide::Buy => &order_book.asks.levels,
            OrderSide::Sell => &order_book.bids.levels,
        };
        
        // Apply slippage to simulate real-world conditions
        let slippage_factor = 1.0 + if order.side == OrderSide::Buy {
            order.slippage_factor
        } else {
            -order.slippage_factor
        };
        
        // Execute against each level until filled or no more liquidity
        let mut total_executed = 0.0;
        let mut total_value = 0.0;
        
        for &(price, available_qty) in levels {
            // Apply slippage
            let execution_price = price * slippage_factor;
            
            // Calculate fill quantity
            let fill_qty = remaining_qty.min(available_qty);
            
            if fill_qty > 0.0 {
                // Calculate fill value and commission
                let fill_value = fill_qty * execution_price;
                let commission = fill_value * MEXC_TAKER_FEE;
                
                // Add fill to order
                order.add_fill(execution_price, fill_qty, commission, "USDC".to_string());
                
                // Update totals
                total_executed += fill_qty;
                total_value += fill_value;
                
                // Update remaining quantity
                remaining_qty -= fill_qty;
                
                if remaining_qty <= 0.0 {
                    break;
                }
            }
        }
        
        // If not fully filled, mark as partially filled
        if remaining_qty > 0.0 && total_executed > 0.0 {
            order.status = OrderStatus::PartiallyFilled;
        } else if total_executed == 0.0 {
            // No fills due to insufficient liquidity
            order.status = OrderStatus::Rejected;
            debug!("Market order rejected due to insufficient liquidity");
        }
    }
    
    /// Execute a limit order against the order book
    fn execute_limit_order(order: &mut Order, order_book: &OrderBook) {
        if let Some(limit_price) = order.price {
            // Check if limit price can be executed
            let can_execute = match order.side {
                OrderSide::Buy => {
                    if !order_book.asks.levels.is_empty() {
                        let best_ask = order_book.asks.levels[0].0;
                        best_ask <= limit_price
                    } else {
                        false
                    }
                },
                OrderSide::Sell => {
                    if !order_book.bids.levels.is_empty() {
                        let best_bid = order_book.bids.levels[0].0;
                        best_bid >= limit_price
                    } else {
                        false
                    }
                },
            };
            
            if can_execute {
                // Remaining quantity to execute
                let mut remaining_qty = order.remaining_qty();
                
                // Get the relevant side of the order book
                let levels = match order.side {
                    OrderSide::Buy => &order_book.asks.levels,
                    OrderSide::Sell => &order_book.bids.levels,
                };
                
                // Execute against each level until filled or price exceeds limit
                let mut total_executed = 0.0;
                let mut total_value = 0.0;
                
                for &(price, available_qty) in levels {
                    if (order.side == OrderSide::Buy && price <= limit_price) || 
                       (order.side == OrderSide::Sell && price >= limit_price) {
                        // Calculate fill quantity
                        let fill_qty = remaining_qty.min(available_qty);
                        
                        if fill_qty > 0.0 {
                            // Calculate fill value and commission
                            let fill_value = fill_qty * price;
                            let commission = fill_value * MEXC_MAKER_FEE;
                            
                            // Add fill to order
                            order.add_fill(price, fill_qty, commission, "USDC".to_string());
                            
                            // Update totals
                            total_executed += fill_qty;
                            total_value += fill_value;
                            
                            // Update remaining quantity
                            remaining_qty -= fill_qty;
                            
                            if remaining_qty <= 0.0 {
                                break;
                            }
                        }
                    } else {
                        // Price exceeds limit
                        break;
                    }
                }
                
                // If not fully filled, keep as partially filled
                if remaining_qty > 0.0 && total_executed > 0.0 {
                    order.status = OrderStatus::PartiallyFilled;
                } else if total_executed == 0.0 {
                    // No fills but order is valid
                    if order.status == OrderStatus::Created {
                        order.set_new();
                    }
                }
            } else {
                // Price not yet reached, set as new if just created
                if order.status == OrderStatus::Created {
                    order.set_new();
                }
            }
        }
    }
    
    /// Start the signal processing loop
    fn start_signal_processor(&mut self) -> tokio::task::JoinHandle<()> {
        let signal_receiver = std::mem::replace(&mut self.signal_receiver, mpsc::channel(100).1);
        let open_orders = self.open_orders.clone();
        let positions = self.positions.clone();
        let account_balance = self.account_balance.clone();
        let performance = self.performance.clone();
        let prices = self.prices.clone();
        let volumes = self.volumes.clone();
        let min_latency_ms = self.min_latency_ms;
        let max_latency_ms = self.max_latency_ms;
        let min_slippage = self.min_slippage;
        let max_slippage = self.max_slippage;
        
        tokio::spawn(async move {
            let mut rng = rand::thread_rng();
            
            while let Some(signal) = signal_receiver.recv().await {
                debug!("Received signal: {:?}", signal);
                
                // Only process Buy/Sell signals with sufficient strength
                if signal.signal_type == SignalType::Neutral || signal.strength.as_f64() < 0.5 {
                    continue;
                }
                
                // Determine order side
                let side = match signal.signal_type {
                    SignalType::Buy => OrderSide::Buy,
                    SignalType::Sell => OrderSide::Sell,
                    _ => continue,
                };
                
                // Check if we have a position for this symbol
                let position_opt = {
                    let positions_lock = positions.read().await;
                    positions_lock.get(&signal.symbol).cloned()
                };
                
                // Determine order size based on position or default size
                let size = match position_opt {
                    Some(position) if position.is_open() => {
                        if (position.direction == PositionDirection::Long && side == OrderSide::Sell) ||
                           (position.direction == PositionDirection::Short && side == OrderSide::Buy) {
                            // Closing position
                            position.quantity
                        } else {
                            // Adding to position - use default size
                            self.config.default_order_size
                        }
                    },
                    _ => self.config.default_order_size,
                };
                
                // Generate random latency and slippage
                let latency_ms = rng.gen_range(min_latency_ms..=max_latency_ms);
                let slippage = rng.gen_range(min_slippage..=max_slippage);
                
                info!("Received signal: {:?} for {}", signal.signal_type, signal.symbol);
                
                // Record the signal for dashboard metrics
                let signal_type_str = match signal.signal_type {
                    SignalType::Buy => "buy",
                    SignalType::Sell => "sell",
                    SignalType::Close => "close",
                    SignalType::StopLoss => "stop_loss",
                };
                paper_trading_metrics::record_trading_signal("main", &signal.symbol, signal_type_str);
                
                match signal.signal_type {
                    SignalType::Buy | SignalType::Sell => {
                        // Create a market order
                        let order = Order::new_market_order(
                            signal.symbol.clone(),
                            side,
                            size,
                            latency_ms,
                            slippage,
                        );
                        
                        // Add to open orders
                        {
                            let mut open_orders_lock = open_orders.write().await;
                            open_orders_lock.insert(order.id.clone(), order.clone());
                        }
                        
                        // Record in metrics for dashboard
                        let side_str = match side {
                            OrderSide::Buy => "buy",
                            OrderSide::Sell => "sell",
                        };
                        paper_trading_metrics::record_executed_order(signal.symbol, "market", side_str, "new");
                        
                        info!("Created order {} from signal: {} {:?} {} @ market (latency: {}ms, slippage: {:.4}%)",
                            order.id, order.symbol, side, size, latency_ms, slippage * 100.0);
                    },
                    _ => {
                        // Other signal types not implemented yet
                        warn!("Signal type {:?} not implemented yet", signal.signal_type);
                    }
                }
            }
        })
    }
    
    /// Start the price updater loop
    fn start_price_updater(&self) -> tokio::task::JoinHandle<()> {
        let prices = self.prices.clone();
        let volumes = self.volumes.clone();
        let open_orders = self.open_orders.clone();
        let positions = self.positions.clone();
        let account_balance = self.account_balance.clone();
        let completed_orders = self.completed_orders.clone();
        let trading_pairs = self.config.trading_pairs.clone();
        let rest_client = self.rest_client.clone();
        
        tokio::spawn(async move {
            let mut interval = interval(tokio::time::Duration::from_secs(2));
            
            loop {
                interval.tick().await;
                
                for pair in &trading_pairs {
                    match rest_client.get_ticker(pair).await {
                        Ok(ticker) => {
                            let price = ticker.last_price;
                            let volume = ticker.volume_24h;
                            debug!("Updated price for {}: {}, 24h volume: {}", pair, price, volume);
                            
                            // Update price and volume
                            {
                                let mut prices_lock = prices.write().await;
                                prices_lock.insert(pair.clone(), price);
                                
                                let mut volumes_lock = volumes.write().await;
                                volumes_lock.insert(pair.clone(), volume);
                                
                                // Update metrics for dashboard
                                paper_trading_metrics::update_market_price(pair, price);
                                paper_trading_metrics::update_market_volume(pair, volume);
                            }
                            
                            // Check for limit orders that can be filled
                            {
                                let mut orders_lock = open_orders.write().await;
                                
                                for (_, order) in orders_lock.iter_mut() {
                                    if order.symbol == *pair && order.order_type == OrderType::Limit && !order.is_complete() {
                                        if let Some(limit_price) = order.price {
                                            let can_fill = match order.side {
                                                OrderSide::Buy => price <= limit_price,
                                                OrderSide::Sell => price >= limit_price,
                                            };
                                            
                                            if can_fill {
                                                debug!("Limit order can be filled: {}", order.id);
                                                order.update_status(OrderStatus::Filled);
                                                order.fill_at_price(price);
                                                
                                                // Record executed order for dashboard
                                                let side_str = match order.side {
                                                    OrderSide::Buy => "buy",
                                                    OrderSide::Sell => "sell",
                                                };
                                                paper_trading_metrics::record_executed_order(
                                                    &order.symbol, 
                                                    "limit", 
                                                    side_str, 
                                                    "filled"
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            error!("Failed to update price for {}: {}", pair, e);
                        }
                    }
                }
                
                // Update all dashboard metrics
                let prices_snapshot = prices.read().await.clone();
                let open_orders_snapshot = open_orders.read().await.clone();
                let positions_snapshot = positions.read().await.clone();
                let account_balance_snapshot = account_balance.read().await.clone();
                let completed_orders_snapshot = completed_orders.read().await.clone();
                
                paper_trading_metrics::update_all_paper_trading_metrics(
                    &account_balance_snapshot,
                    &open_orders_snapshot,
                    &positions_snapshot,
                    &prices_snapshot,
                    &completed_orders_snapshot
                );
            }
        })
    }
    
    /// Create a new market order
    pub async fn create_market_order(
        &self,
        symbol: &str,
        side: OrderSide,
        quantity: f64,
    ) -> Result<String, String> {
        // Generate random latency and slippage
        let mut rng = rand::thread_rng();
        let latency_ms = rng.gen_range(self.min_latency_ms..=self.max_latency_ms);
        let slippage = rng.gen_range(self.min_slippage..=self.max_slippage);
        
        // Create order
        let order = Order::new_market_order(
            symbol.to_string(),
            side,
            quantity,
            latency_ms,
            slippage,
        );
        
        // Add to open orders
        {
            let mut open_orders = self.open_orders.write().await;
            open_orders.insert(order.id.clone(), order.clone());
        }
        
        // Record in metrics for dashboard
        let side_str = match side {
            OrderSide::Buy => "buy",
            OrderSide::Sell => "sell",
        };
        paper_trading_metrics::record_executed_order(symbol, "market", side_str, "new");
        
        info!("Created market order: {} {:?} {} @ market (latency: {}ms, slippage: {:.4}%)",
            symbol, side, quantity, latency_ms, slippage * 100.0);
        
        Ok(order.id)
    }
    
    /// Create a new limit order
    pub async fn create_limit_order(
        &self,
        symbol: &str,
        side: OrderSide,
        quantity: f64,
        price: f64,
        time_in_force: TimeInForce,
    ) -> Result<String, String> {
        // Generate random latency and slippage
        let mut rng = rand::thread_rng();
        let latency_ms = rng.gen_range(self.min_latency_ms..=self.max_latency_ms);
        let slippage = rng.gen_range(self.min_slippage..=self.max_slippage);
        
        // Create order
        let order = Order::new_limit_order(
            symbol.to_string(),
            side,
            quantity,
            price,
            time_in_force,
            latency_ms,
            slippage,
        );
        
        // Add to open orders
        {
            let mut open_orders = self.open_orders.write().await;
            open_orders.insert(order.id.clone(), order.clone());
        }
        
        // Record in metrics for dashboard
        let side_str = match side {
            OrderSide::Buy => "buy",
            OrderSide::Sell => "sell",
        };
        paper_trading_metrics::record_executed_order(symbol, "limit", side_str, "new");
        
        info!("Created limit order: {} {:?} {} @ {} (latency: {}ms, slippage: {:.4}%)",
            symbol, side, quantity, price, latency_ms, slippage * 100.0);
        
        Ok(order.id)
    }
    
    /// Cancel an order
    pub async fn cancel_order(&self, order_id: &str) -> Result<bool, String> {
        let mut result = false;
        
        // Find and cancel the order
        {
            let mut open_orders = self.open_orders.write().await;
            if let Some(order) = open_orders.get_mut(order_id) {
                result = order.cancel();
                if result {
                    info!("Canceled order {}", order_id);
                }
            }
        }
        
        if result {
            Ok(true)
        } else {
            Err(format!("Order {} not found or cannot be canceled", order_id))
        }
    }
    
    /// Get position for a symbol
    pub async fn get_position(&self, symbol: &str) -> Option<Position> {
        let positions = self.positions.read().await;
        positions.get(symbol).cloned()
    }
    
    /// Get all open positions
    pub async fn get_open_positions(&self) -> Vec<Position> {
        let positions = self.positions.read().await;
        positions
            .values()
            .filter(|p| p.is_open())
            .cloned()
            .collect()
    }
    
    /// Get all open orders
    pub async fn get_open_orders(&self) -> Vec<Order> {
        let open_orders = self.open_orders.read().await;
        open_orders.values().cloned().collect()
    }
    
    /// Get account balance
    pub async fn get_balance(&self, asset: &str) -> f64 {
        let balance = self.account_balance.read().await;
        *balance.get(asset).unwrap_or(&0.0)
    }
    
    /// Get performance metrics for a symbol
    pub async fn get_performance(&self, symbol: &str) -> Option<TradingPerformance> {
        let performance = self.performance.read().await;
        performance.get(symbol).cloned()
    }
}
