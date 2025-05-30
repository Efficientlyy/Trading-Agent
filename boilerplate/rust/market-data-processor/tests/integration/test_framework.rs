use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tokio::time;
use tracing::{debug, error, info, warn};

use market_data_processor::models::order::{Order, OrderSide, OrderStatus, OrderType};
use market_data_processor::models::position::Position;
use market_data_processor::services::decision_module::{DecisionAction, DecisionModule, DecisionOutput};
use market_data_processor::services::order_execution::{OrderExecutionService, OrderRequest};
use market_data_processor::services::paper_trading::{
    MatchingEngine, PaperTradingService, VirtualAccount
};
use market_data_processor::services::signal_generator::{Signal, SignalGenerator, SignalType};
use market_data_processor::utils::enhanced_config::EnhancedConfig;

/// Market data snapshot for testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketDataSnapshot {
    pub timestamp: DateTime<Utc>,
    pub symbol: String,
    pub price: f64,
    pub bids: Vec<(f64, f64)>, // (price, quantity)
    pub asks: Vec<(f64, f64)>, // (price, quantity)
    pub volume_24h: f64,
}

/// Test scenario configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestScenario {
    pub name: String,
    pub description: String,
    pub market_condition: MarketCondition,
    pub initial_balances: HashMap<String, f64>,
    pub signals: Vec<TestSignal>,
    pub expected_outcomes: ExpectedOutcomes,
    pub max_test_duration_seconds: u64,
}

/// Market conditions for different test scenarios
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MarketCondition {
    Trending,
    Ranging,
    Volatile,
    LowLiquidity,
    Normal,
}

/// Test signal to inject into the system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestSignal {
    pub timestamp: DateTime<Utc>,
    pub symbol: String,
    pub signal_type: SignalType,
    pub strength: f64,
    pub source: String,
    pub expected_action: Option<DecisionAction>,
}

/// Expected outcomes of the test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectedOutcomes {
    pub final_balances: HashMap<String, (f64, f64)>, // (min, max) ranges
    pub order_count: (usize, usize),                 // (min, max) ranges
    pub filled_order_count: (usize, usize),          // (min, max) ranges
    pub pnl_range: (f64, f64),                       // (min, max) ranges
    pub max_drawdown: f64,                           // Maximum allowable drawdown
}

/// Test result metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResults {
    pub scenario_name: String,
    pub passed: bool,
    pub execution_time_ms: u64,
    pub final_balances: HashMap<String, f64>,
    pub order_count: usize,
    pub filled_order_count: usize,
    pub total_pnl: f64,
    pub max_drawdown: f64,
    pub win_rate: f64,
    pub sharpe_ratio: Option<f64>,
    pub failures: Vec<String>,
}

/// Mock market data service for testing
pub struct MockMarketDataService {
    data: Vec<MarketDataSnapshot>,
    current_index: usize,
}

impl MockMarketDataService {
    /// Create a new mock market data service
    pub fn new(data: Vec<MarketDataSnapshot>) -> Self {
        Self {
            data,
            current_index: 0,
        }
    }

    /// Get the current market data
    pub fn current(&self) -> Option<&MarketDataSnapshot> {
        self.data.get(self.current_index)
    }

    /// Advance to the next market data snapshot
    pub fn advance(&mut self) -> Option<&MarketDataSnapshot> {
        if self.current_index < self.data.len() - 1 {
            self.current_index += 1;
            self.current()
        } else {
            None
        }
    }

    /// Reset to the beginning
    pub fn reset(&mut self) {
        self.current_index = 0;
    }

    /// Get current order book for a symbol
    pub fn get_order_book(&self, symbol: &str) -> Option<(Vec<(f64, f64)>, Vec<(f64, f64)>)> {
        self.current().and_then(|snapshot| {
            if snapshot.symbol == symbol {
                Some((snapshot.bids.clone(), snapshot.asks.clone()))
            } else {
                None
            }
        })
    }

    /// Get current price for a symbol
    pub fn get_price(&self, symbol: &str) -> Option<f64> {
        self.current().and_then(|snapshot| {
            if snapshot.symbol == symbol {
                Some(snapshot.price)
            } else {
                None
            }
        })
    }
}

/// Integration test runner
pub struct IntegrationTestRunner {
    config: Arc<EnhancedConfig>,
    mock_market_data: Arc<Mutex<MockMarketDataService>>,
    paper_trading: Arc<PaperTradingService>,
    order_execution: Arc<OrderExecutionService>,
    test_scenarios: Vec<TestScenario>,
}

impl IntegrationTestRunner {
    /// Create a new integration test runner
    pub async fn new(
        config: EnhancedConfig,
        test_scenarios: Vec<TestScenario>,
        market_data: Vec<MarketDataSnapshot>,
    ) -> Self {
        let config = Arc::new(config);
        let mock_market_data = Arc::new(Mutex::new(MockMarketDataService::new(market_data)));
        
        // Create custom matching engine that uses our mock market data
        let matching_engine = Arc::new(MockMatchingEngine::new(mock_market_data.clone()));
        
        // Create paper trading service with mock components
        let paper_trading = Arc::new(PaperTradingService::new_with_components(
            config.clone(),
            matching_engine,
        ));
        
        // Create order execution service
        let order_execution = Arc::new(OrderExecutionService::new(
            config.clone(),
            paper_trading.clone(),
            None, // No live trading for tests
        ));
        
        Self {
            config,
            mock_market_data,
            paper_trading,
            order_execution,
            test_scenarios,
        }
    }
    
    /// Run all test scenarios
    pub async fn run_all_tests(&self) -> Vec<TestResults> {
        let mut results = Vec::new();
        
        for scenario in &self.test_scenarios {
            info!("Running test scenario: {}", scenario.name);
            let result = self.run_scenario(scenario).await;
            results.push(result);
        }
        
        results
    }
    
    /// Run a single test scenario
    pub async fn run_scenario(&self, scenario: &TestScenario) -> TestResults {
        let start_time = std::time::Instant::now();
        let mut failures = Vec::new();
        
        // Reset market data
        self.mock_market_data.lock().await.reset();
        
        // Initialize virtual account with scenario balances
        self.paper_trading.set_initial_balances(scenario.initial_balances.clone()).await;
        
        // Process test signals according to their timestamps
        let mut current_time = Utc::now();
        let mut signals_processed = 0;
        let mut orders_placed = 0;
        let mut orders_filled = 0;
        let mut max_drawdown = 0.0;
        let mut peak_balance = 0.0;
        let mut current_drawdown = 0.0;
        
        // Track profits and losses for performance metrics
        let mut trade_results = Vec::new();
        
        for signal in &scenario.signals {
            // Advance market data to signal timestamp
            while let Some(snapshot) = self.mock_market_data.lock().await.current() {
                if snapshot.timestamp >= signal.timestamp {
                    break;
                }
                
                // Update positions with current market data
                self.paper_trading.update_positions().await;
                
                // Check for drawdown
                let current_balance = self.paper_trading.get_total_balance_usdt().await.unwrap_or(0.0);
                if current_balance > peak_balance {
                    peak_balance = current_balance;
                } else {
                    current_drawdown = (peak_balance - current_balance) / peak_balance * 100.0;
                    if current_drawdown > max_drawdown {
                        max_drawdown = current_drawdown;
                    }
                }
                
                self.mock_market_data.lock().await.advance();
                
                // Simulate time passing
                time::sleep(Duration::from_millis(10)).await;
            }
            
            // Convert test signal to system signal
            let system_signal = Signal {
                symbol: signal.symbol.clone(),
                signal_type: signal.signal_type,
                strength: signal.strength.into(),
                price: self.mock_market_data.lock().await.get_price(&signal.symbol),
                timestamp: signal.timestamp,
                source: signal.source.clone(),
                metadata: None,
            };
            
            // Process signal
            match self.process_signal(system_signal).await {
                Ok(order_id) => {
                    orders_placed += 1;
                    
                    // Check if order was filled
                    if let Ok(order) = self.paper_trading.get_order(&order_id).await {
                        if order.status == OrderStatus::Filled {
                            orders_filled += 1;
                            
                            // Record trade result
                            if let Some(executed_price) = order.avg_execution_price {
                                let pnl = match order.side {
                                    OrderSide::Buy => 0.0, // We record PnL on sell orders
                                    OrderSide::Sell => {
                                        order.executed_qty * executed_price - order.executed_qty * order.price.unwrap_or(executed_price)
                                    }
                                };
                                
                                if pnl != 0.0 {
                                    trade_results.push(pnl);
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    failures.push(format!("Failed to process signal: {}", e));
                }
            }
            
            signals_processed += 1;
            
            // Check if max test duration exceeded
            if start_time.elapsed().as_secs() > scenario.max_test_duration_seconds {
                failures.push(format!(
                    "Test exceeded maximum duration of {} seconds. Processed {} of {} signals.",
                    scenario.max_test_duration_seconds, signals_processed, scenario.signals.len()
                ));
                break;
            }
        }
        
        // Validate test results against expected outcomes
        let final_balances = self.paper_trading.get_balances().await.unwrap_or_default();
        
        // Check final balances
        for (currency, (min, max)) in &scenario.expected_outcomes.final_balances {
            if let Some(balance) = final_balances.get(currency) {
                if *balance < *min || *balance > *max {
                    failures.push(format!(
                        "Final balance for {} is {}, expected between {} and {}",
                        currency, balance, min, max
                    ));
                }
            } else {
                failures.push(format!("Missing final balance for {}", currency));
            }
        }
        
        // Check order counts
        let (min_orders, max_orders) = scenario.expected_outcomes.order_count;
        if orders_placed < min_orders || orders_placed > max_orders {
            failures.push(format!(
                "Order count is {}, expected between {} and {}",
                orders_placed, min_orders, max_orders
            ));
        }
        
        // Check filled order counts
        let (min_filled, max_filled) = scenario.expected_outcomes.filled_order_count;
        if orders_filled < min_filled || orders_filled > max_filled {
            failures.push(format!(
                "Filled order count is {}, expected between {} and {}",
                orders_filled, min_filled, max_filled
            ));
        }
        
        // Check PnL
        let total_pnl: f64 = trade_results.iter().sum();
        let (min_pnl, max_pnl) = scenario.expected_outcomes.pnl_range;
        if total_pnl < min_pnl || total_pnl > max_pnl {
            failures.push(format!(
                "Total PnL is {}, expected between {} and {}",
                total_pnl, min_pnl, max_pnl
            ));
        }
        
        // Check max drawdown
        if max_drawdown > scenario.expected_outcomes.max_drawdown {
            failures.push(format!(
                "Max drawdown is {}%, expected to be less than {}%",
                max_drawdown, scenario.expected_outcomes.max_drawdown
            ));
        }
        
        // Calculate win rate
        let win_count = trade_results.iter().filter(|&&r| r > 0.0).count();
        let loss_count = trade_results.iter().filter(|&&r| r < 0.0).count();
        let win_rate = if win_count + loss_count > 0 {
            win_count as f64 / (win_count + loss_count) as f64 * 100.0
        } else {
            0.0
        };
        
        // Calculate Sharpe ratio if we have enough trades
        let sharpe_ratio = if trade_results.len() >= 10 {
            let mean = total_pnl / trade_results.len() as f64;
            let variance = trade_results.iter()
                .map(|&r| (r - mean).powi(2))
                .sum::<f64>() / trade_results.len() as f64;
            let std_dev = variance.sqrt();
            
            if std_dev > 0.0 {
                Some(mean / std_dev)
            } else {
                None
            }
        } else {
            None
        };
        
        // Create test results
        TestResults {
            scenario_name: scenario.name.clone(),
            passed: failures.is_empty(),
            execution_time_ms: start_time.elapsed().as_millis() as u64,
            final_balances,
            order_count: orders_placed,
            filled_order_count: orders_filled,
            total_pnl,
            max_drawdown,
            win_rate,
            sharpe_ratio,
            failures,
        }
    }
    
    /// Process a signal and return the order ID if an order was placed
    async fn process_signal(&self, signal: Signal) -> Result<String, String> {
        // Simulate decision making
        let decision = self.simulate_decision(signal).await;
        
        // Skip if action is Hold
        if decision.action == DecisionAction::Hold {
            return Err("Signal resulted in HOLD decision".to_string());
        }
        
        // Create order request
        let side = match decision.action {
            DecisionAction::Buy => OrderSide::Buy,
            DecisionAction::Sell => OrderSide::Sell,
            DecisionAction::Hold => return Err("Signal resulted in HOLD decision".to_string()),
        };
        
        let order_request = OrderRequest {
            symbol: decision.symbol.clone(),
            side,
            order_type: OrderType::Market,
            quantity: decision.position_size,
            price: None,
            time_in_force: None,
            client_order_id: None,
        };
        
        // Place order
        match self.order_execution.place_order(order_request).await {
            Ok(response) => Ok(response.order_id),
            Err(e) => Err(format!("Failed to place order: {}", e)),
        }
    }
    
    /// Simulate a decision from a signal
    async fn simulate_decision(&self, signal: Signal) -> DecisionOutput {
        // Convert signal type to decision action
        let action = match signal.signal_type {
            SignalType::Buy => DecisionAction::Buy,
            SignalType::Sell => DecisionAction::Sell,
            _ => DecisionAction::Hold,
        };
        
        // Calculate confidence based on signal strength
        let confidence = signal.strength.as_f64();
        
        // Generate reasoning
        let reasoning = match action {
            DecisionAction::Buy => format!(
                "Buy signal detected from {} with strength {}. Technical indicators suggest upward momentum.",
                signal.source, confidence
            ),
            DecisionAction::Sell => format!(
                "Sell signal detected from {} with strength {}. Technical indicators suggest downward momentum.",
                signal.source, confidence
            ),
            DecisionAction::Hold => format!(
                "No clear signal detected from {}. Strength {} is insufficient for action.",
                signal.source, confidence
            ),
        };
        
        // Calculate risk score based on signal volatility
        let risk_score = 0.5; // Mid-level risk (0.0 to 1.0)
        
        // Calculate position size based on risk score and config
        let position_size = self.calculate_position_size(signal.symbol.as_str(), risk_score);
        
        DecisionOutput {
            symbol: signal.symbol,
            action,
            confidence,
            reasoning,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            risk_score,
            position_size,
        }
    }
    
    /// Calculate position size based on risk parameters
    fn calculate_position_size(&self, symbol: &str, risk_score: f64) -> f64 {
        // Base size from config
        let base_size = self.config.default_order_size;
        
        // Adjust based on risk score (lower risk = larger position)
        let risk_factor = 1.0 - (risk_score * 0.5); // 0.5 to 1.0
        
        // Ensure position size doesn't exceed max
        (base_size * risk_factor).min(self.config.max_position_size)
    }
}

/// Mock matching engine for testing
pub struct MockMatchingEngine {
    mock_market_data: Arc<Mutex<MockMarketDataService>>,
}

impl MockMatchingEngine {
    /// Create a new mock matching engine
    pub fn new(mock_market_data: Arc<Mutex<MockMarketDataService>>) -> Self {
        Self { mock_market_data }
    }
    
    /// Execute a market order against mock market data
    pub async fn execute_market_order(
        &self,
        symbol: &str,
        side: OrderSide,
        quantity: f64,
    ) -> Result<(f64, f64), String> {
        let mock_data = self.mock_market_data.lock().await;
        
        let (bids, asks) = mock_data.get_order_book(symbol)
            .ok_or_else(|| format!("No order book data available for {}", symbol))?;
        
        match side {
            OrderSide::Buy => {
                // For buy orders, we match against asks
                let mut remaining_qty = quantity;
                let mut total_cost = 0.0;
                let mut executed_qty = 0.0;
                
                for (price, qty) in asks.iter() {
                    let matched_qty = remaining_qty.min(*qty);
                    total_cost += matched_qty * price;
                    executed_qty += matched_qty;
                    remaining_qty -= matched_qty;
                    
                    if remaining_qty <= 0.0 {
                        break;
                    }
                }
                
                if executed_qty > 0.0 {
                    let avg_price = total_cost / executed_qty;
                    Ok((executed_qty, avg_price))
                } else {
                    Err("No liquidity available to execute buy order".to_string())
                }
            },
            OrderSide::Sell => {
                // For sell orders, we match against bids
                let mut remaining_qty = quantity;
                let mut total_value = 0.0;
                let mut executed_qty = 0.0;
                
                for (price, qty) in bids.iter() {
                    let matched_qty = remaining_qty.min(*qty);
                    total_value += matched_qty * price;
                    executed_qty += matched_qty;
                    remaining_qty -= matched_qty;
                    
                    if remaining_qty <= 0.0 {
                        break;
                    }
                }
                
                if executed_qty > 0.0 {
                    let avg_price = total_value / executed_qty;
                    Ok((executed_qty, avg_price))
                } else {
                    Err("No liquidity available to execute sell order".to_string())
                }
            },
        }
    }
    
    /// Execute a limit order against mock market data
    pub async fn execute_limit_order(
        &self,
        symbol: &str,
        side: OrderSide,
        quantity: f64,
        limit_price: f64,
    ) -> Result<(f64, f64), String> {
        let mock_data = self.mock_market_data.lock().await;
        
        let (bids, asks) = mock_data.get_order_book(symbol)
            .ok_or_else(|| format!("No order book data available for {}", symbol))?;
        
        match side {
            OrderSide::Buy => {
                // For buy limit orders, we match against asks that are <= limit_price
                let mut remaining_qty = quantity;
                let mut total_cost = 0.0;
                let mut executed_qty = 0.0;
                
                for (price, qty) in asks.iter().filter(|(p, _)| *p <= limit_price) {
                    let matched_qty = remaining_qty.min(*qty);
                    total_cost += matched_qty * price;
                    executed_qty += matched_qty;
                    remaining_qty -= matched_qty;
                    
                    if remaining_qty <= 0.0 {
                        break;
                    }
                }
                
                if executed_qty > 0.0 {
                    let avg_price = total_cost / executed_qty;
                    Ok((executed_qty, avg_price))
                } else {
                    Err("No matching liquidity for buy limit order".to_string())
                }
            },
            OrderSide::Sell => {
                // For sell limit orders, we match against bids that are >= limit_price
                let mut remaining_qty = quantity;
                let mut total_value = 0.0;
                let mut executed_qty = 0.0;
                
                for (price, qty) in bids.iter().filter(|(p, _)| *p >= limit_price) {
                    let matched_qty = remaining_qty.min(*qty);
                    total_value += matched_qty * price;
                    executed_qty += matched_qty;
                    remaining_qty -= matched_qty;
                    
                    if remaining_qty <= 0.0 {
                        break;
                    }
                }
                
                if executed_qty > 0.0 {
                    let avg_price = total_value / executed_qty;
                    Ok((executed_qty, avg_price))
                } else {
                    Err("No matching liquidity for sell limit order".to_string())
                }
            },
        }
    }
}
