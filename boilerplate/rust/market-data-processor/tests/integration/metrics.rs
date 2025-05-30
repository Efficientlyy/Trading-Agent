use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

/// Performance metrics for a test run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Timestamp when metrics were collected
    pub timestamp: DateTime<Utc>,
    
    /// Test scenario name
    pub scenario_name: String,
    
    /// Account metrics
    pub account: AccountMetrics,
    
    /// Trading metrics
    pub trading: TradingMetrics,
    
    /// Risk metrics
    pub risk: RiskMetrics,
    
    /// Per-symbol metrics
    pub symbols: HashMap<String, SymbolMetrics>,
}

/// Account-level metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccountMetrics {
    /// Initial account value in USDT
    pub initial_value_usdt: f64,
    
    /// Final account value in USDT
    pub final_value_usdt: f64,
    
    /// Total profit/loss in USDT
    pub total_pnl_usdt: f64,
    
    /// Total profit/loss percentage
    pub total_pnl_percent: f64,
    
    /// Balances by currency
    pub balances: HashMap<String, f64>,
}

/// Trading performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingMetrics {
    /// Total number of orders placed
    pub total_orders: usize,
    
    /// Number of orders filled
    pub filled_orders: usize,
    
    /// Number of orders partially filled
    pub partially_filled_orders: usize,
    
    /// Number of orders canceled
    pub canceled_orders: usize,
    
    /// Number of winning trades
    pub winning_trades: usize,
    
    /// Number of losing trades
    pub losing_trades: usize,
    
    /// Win rate (percentage)
    pub win_rate: f64,
    
    /// Average profit of winning trades
    pub avg_win_profit: f64,
    
    /// Average loss of losing trades
    pub avg_loss: f64,
    
    /// Profit factor (gross profit / gross loss)
    pub profit_factor: f64,
    
    /// Average holding time for trades
    pub avg_holding_time_seconds: f64,
}

/// Risk metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskMetrics {
    /// Maximum drawdown percentage
    pub max_drawdown_percent: f64,
    
    /// Maximum drawdown amount in USDT
    pub max_drawdown_usdt: f64,
    
    /// Peak account value
    pub peak_value: f64,
    
    /// Trough account value
    pub trough_value: f64,
    
    /// Sharpe ratio (if enough data points are available)
    pub sharpe_ratio: Option<f64>,
    
    /// Calmar ratio (annualized return / max drawdown)
    pub calmar_ratio: Option<f64>,
    
    /// Return volatility (standard deviation of returns)
    pub volatility: Option<f64>,
}

/// Symbol-specific metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolMetrics {
    /// Symbol name
    pub symbol: String,
    
    /// Number of trades for this symbol
    pub trade_count: usize,
    
    /// Total profit/loss for this symbol in USDT
    pub total_pnl: f64,
    
    /// Win rate for this symbol
    pub win_rate: f64,
    
    /// Average position size
    pub avg_position_size: f64,
    
    /// Average holding time in seconds
    pub avg_holding_time_seconds: f64,
    
    /// Maximum profit in a single trade
    pub max_profit: f64,
    
    /// Maximum loss in a single trade
    pub max_loss: f64,
}

/// Trade record for a single completed trade
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeRecord {
    /// Trade ID
    pub id: String,
    
    /// Symbol traded
    pub symbol: String,
    
    /// Trade direction
    pub direction: String,
    
    /// Entry timestamp
    pub entry_time: DateTime<Utc>,
    
    /// Exit timestamp
    pub exit_time: Option<DateTime<Utc>>,
    
    /// Entry price
    pub entry_price: f64,
    
    /// Exit price
    pub exit_price: Option<f64>,
    
    /// Trade quantity
    pub quantity: f64,
    
    /// Realized profit/loss
    pub realized_pnl: Option<f64>,
    
    /// Fees paid
    pub fees: f64,
    
    /// Signal source that triggered the trade
    pub signal_source: Option<String>,
    
    /// Order IDs associated with this trade
    pub order_ids: Vec<String>,
}

impl PerformanceMetrics {
    /// Create a new PerformanceMetrics instance
    pub fn new(scenario_name: String) -> Self {
        Self {
            timestamp: Utc::now(),
            scenario_name,
            account: AccountMetrics {
                initial_value_usdt: 0.0,
                final_value_usdt: 0.0,
                total_pnl_usdt: 0.0,
                total_pnl_percent: 0.0,
                balances: HashMap::new(),
            },
            trading: TradingMetrics {
                total_orders: 0,
                filled_orders: 0,
                partially_filled_orders: 0,
                canceled_orders: 0,
                winning_trades: 0,
                losing_trades: 0,
                win_rate: 0.0,
                avg_win_profit: 0.0,
                avg_loss: 0.0,
                profit_factor: 0.0,
                avg_holding_time_seconds: 0.0,
            },
            risk: RiskMetrics {
                max_drawdown_percent: 0.0,
                max_drawdown_usdt: 0.0,
                peak_value: 0.0,
                trough_value: 0.0,
                sharpe_ratio: None,
                calmar_ratio: None,
                volatility: None,
            },
            symbols: HashMap::new(),
        }
    }
    
    /// Calculate key metrics from trade records
    pub fn calculate_from_trades(
        &mut self,
        trades: &[TradeRecord],
        initial_value: f64,
        final_value: f64,
        balances: HashMap<String, f64>,
    ) {
        // Account metrics
        self.account.initial_value_usdt = initial_value;
        self.account.final_value_usdt = final_value;
        self.account.total_pnl_usdt = final_value - initial_value;
        self.account.total_pnl_percent = (final_value - initial_value) / initial_value * 100.0;
        self.account.balances = balances;
        
        // Skip if no trades
        if trades.is_empty() {
            return;
        }
        
        // Trading metrics
        let mut winning_trades = Vec::new();
        let mut losing_trades = Vec::new();
        let mut symbol_trades: HashMap<String, Vec<&TradeRecord>> = HashMap::new();
        let mut total_holding_time = 0.0;
        
        for trade in trades {
            // Count by symbol
            symbol_trades.entry(trade.symbol.clone())
                .or_insert_with(Vec::new)
                .push(trade);
            
            // Skip trades without exit info
            if let (Some(exit_time), Some(pnl)) = (trade.exit_time, trade.realized_pnl) {
                // Calculate holding time
                let holding_time = exit_time.signed_duration_since(trade.entry_time).num_seconds() as f64;
                total_holding_time += holding_time;
                
                // Classify winning and losing trades
                if pnl > 0.0 {
                    winning_trades.push((pnl, holding_time));
                } else if pnl < 0.0 {
                    losing_trades.push((pnl, holding_time));
                }
            }
        }
        
        // Calculate trading metrics
        self.trading.total_orders = trades.len(); // Approximation, may count both sides of a trade
        self.trading.winning_trades = winning_trades.len();
        self.trading.losing_trades = losing_trades.len();
        
        let completed_trades = self.trading.winning_trades + self.trading.losing_trades;
        if completed_trades > 0 {
            self.trading.win_rate = self.trading.winning_trades as f64 / completed_trades as f64 * 100.0;
            self.trading.avg_holding_time_seconds = total_holding_time / completed_trades as f64;
            
            // Calculate average win and loss
            if !winning_trades.is_empty() {
                let total_win = winning_trades.iter().map(|(pnl, _)| pnl).sum::<f64>();
                self.trading.avg_win_profit = total_win / winning_trades.len() as f64;
            }
            
            if !losing_trades.is_empty() {
                let total_loss = losing_trades.iter().map(|(pnl, _)| pnl.abs()).sum::<f64>();
                self.trading.avg_loss = total_loss / losing_trades.len() as f64;
            }
            
            // Calculate profit factor
            let gross_profit = winning_trades.iter().map(|(pnl, _)| pnl).sum::<f64>();
            let gross_loss = losing_trades.iter().map(|(pnl, _)| pnl.abs()).sum::<f64>();
            if gross_loss > 0.0 {
                self.trading.profit_factor = gross_profit / gross_loss;
            }
        }
        
        // Calculate symbol-specific metrics
        for (symbol, trades) in symbol_trades {
            let mut symbol_metric = SymbolMetrics {
                symbol: symbol.clone(),
                trade_count: trades.len(),
                total_pnl: 0.0,
                win_rate: 0.0,
                avg_position_size: 0.0,
                avg_holding_time_seconds: 0.0,
                max_profit: 0.0,
                max_loss: 0.0,
            };
            
            let mut winning_count = 0;
            let mut total_position_size = 0.0;
            let mut total_holding_time = 0.0;
            let mut completed_count = 0;
            
            for trade in trades {
                // Add to position size total
                total_position_size += trade.quantity;
                
                // Skip trades without exit info
                if let (Some(exit_time), Some(pnl)) = (trade.exit_time, trade.realized_pnl) {
                    // Calculate holding time
                    let holding_time = exit_time.signed_duration_since(trade.entry_time).num_seconds() as f64;
                    total_holding_time += holding_time;
                    completed_count += 1;
                    
                    // Update total PnL
                    symbol_metric.total_pnl += pnl;
                    
                    // Update win count
                    if pnl > 0.0 {
                        winning_count += 1;
                        symbol_metric.max_profit = symbol_metric.max_profit.max(pnl);
                    } else if pnl < 0.0 {
                        symbol_metric.max_loss = symbol_metric.max_loss.max(pnl.abs());
                    }
                }
            }
            
            // Calculate averages
            if trades.len() > 0 {
                symbol_metric.avg_position_size = total_position_size / trades.len() as f64;
            }
            
            if completed_count > 0 {
                symbol_metric.win_rate = winning_count as f64 / completed_count as f64 * 100.0;
                symbol_metric.avg_holding_time_seconds = total_holding_time / completed_count as f64;
            }
            
            // Add to metrics
            self.symbols.insert(symbol, symbol_metric);
        }
    }
    
    /// Update risk metrics
    pub fn update_risk_metrics(
        &mut self,
        max_drawdown_percent: f64,
        max_drawdown_usdt: f64,
        peak_value: f64,
        trough_value: f64,
        returns: &[f64],
    ) {
        self.risk.max_drawdown_percent = max_drawdown_percent;
        self.risk.max_drawdown_usdt = max_drawdown_usdt;
        self.risk.peak_value = peak_value;
        self.risk.trough_value = trough_value;
        
        // Calculate Sharpe ratio if enough returns are available
        if returns.len() >= 10 {
            let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
            let sum_squared_diff = returns.iter()
                .map(|r| (r - mean_return).powi(2))
                .sum::<f64>();
            let variance = sum_squared_diff / returns.len() as f64;
            let std_dev = variance.sqrt();
            
            if std_dev > 0.0 {
                self.risk.sharpe_ratio = Some(mean_return / std_dev);
                self.risk.volatility = Some(std_dev);
                
                // Calculate Calmar ratio if max drawdown is non-zero
                if max_drawdown_percent > 0.0 {
                    // Annualize the return (approximation)
                    let annualized_return = mean_return * 252.0; // Assuming daily returns
                    self.risk.calmar_ratio = Some(annualized_return / max_drawdown_percent);
                }
            }
        }
    }
}

/// Test result reporter
pub struct MetricsReporter {
    metrics: Vec<PerformanceMetrics>,
}

impl MetricsReporter {
    /// Create a new metrics reporter
    pub fn new() -> Self {
        Self {
            metrics: Vec::new(),
        }
    }
    
    /// Add metrics to the reporter
    pub fn add_metrics(&mut self, metrics: PerformanceMetrics) {
        self.metrics.push(metrics);
    }
    
    /// Generate a summary report
    pub fn generate_summary(&self) -> String {
        let mut summary = String::new();
        
        summary.push_str("# Paper Trading Test Summary\n\n");
        summary.push_str(&format!("Test Run: {}\n", chrono::Utc::now().format("%Y-%m-%d %H:%M:%S")));
        summary.push_str(&format!("Total Scenarios: {}\n\n", self.metrics.len()));
        
        // Overall metrics
        let total_pnl: f64 = self.metrics.iter().map(|m| m.account.total_pnl_usdt).sum();
        let avg_win_rate: f64 = if !self.metrics.is_empty() {
            self.metrics.iter().map(|m| m.trading.win_rate).sum::<f64>() / self.metrics.len() as f64
        } else {
            0.0
        };
        
        summary.push_str("## Overall Performance\n\n");
        summary.push_str(&format!("Total P&L: ${:.2}\n", total_pnl));
        summary.push_str(&format!("Average Win Rate: {:.2}%\n\n", avg_win_rate));
        
        // Scenario metrics
        summary.push_str("## Scenario Results\n\n");
        
        for (i, metrics) in self.metrics.iter().enumerate() {
            summary.push_str(&format!("### Scenario {}: {}\n\n", i + 1, metrics.scenario_name));
            
            // Account metrics
            summary.push_str("#### Account Metrics\n\n");
            summary.push_str(&format!("- Initial Value: ${:.2}\n", metrics.account.initial_value_usdt));
            summary.push_str(&format!("- Final Value: ${:.2}\n", metrics.account.final_value_usdt));
            summary.push_str(&format!("- Total P&L: ${:.2} ({:.2}%)\n\n", 
                metrics.account.total_pnl_usdt, metrics.account.total_pnl_percent));
            
            // Trading metrics
            summary.push_str("#### Trading Metrics\n\n");
            summary.push_str(&format!("- Total Orders: {}\n", metrics.trading.total_orders));
            summary.push_str(&format!("- Win Rate: {:.2}%\n", metrics.trading.win_rate));
            summary.push_str(&format!("- Avg. Win: ${:.2}\n", metrics.trading.avg_win_profit));
            summary.push_str(&format!("- Avg. Loss: ${:.2}\n", metrics.trading.avg_loss));
            summary.push_str(&format!("- Profit Factor: {:.2}\n\n", metrics.trading.profit_factor));
            
            // Risk metrics
            summary.push_str("#### Risk Metrics\n\n");
            summary.push_str(&format!("- Max Drawdown: {:.2}% (${:.2})\n", 
                metrics.risk.max_drawdown_percent, metrics.risk.max_drawdown_usdt));
            
            if let Some(sharpe) = metrics.risk.sharpe_ratio {
                summary.push_str(&format!("- Sharpe Ratio: {:.2}\n", sharpe));
            }
            
            if let Some(calmar) = metrics.risk.calmar_ratio {
                summary.push_str(&format!("- Calmar Ratio: {:.2}\n", calmar));
            }
            
            summary.push_str("\n");
            
            // Symbol metrics
            if !metrics.symbols.is_empty() {
                summary.push_str("#### Symbol Performance\n\n");
                summary.push_str("| Symbol | Trades | P&L | Win Rate | Avg. Holding Time |\n");
                summary.push_str("|--------|--------|-----|----------|------------------|\n");
                
                for (_, symbol) in metrics.symbols.iter() {
                    let holding_time_minutes = symbol.avg_holding_time_seconds / 60.0;
                    summary.push_str(&format!("| {} | {} | ${:.2} | {:.2}% | {:.2} min |\n",
                        symbol.symbol, symbol.trade_count, symbol.total_pnl, 
                        symbol.win_rate, holding_time_minutes));
                }
                
                summary.push_str("\n");
            }
        }
        
        summary
    }
    
    /// Save metrics to a JSON file
    pub fn save_to_file(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(&self.metrics)?;
        std::fs::write(path, json)
    }
}
