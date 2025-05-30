use crate::models::order::{Order, OrderSide, OrderStatus, OrderType, TimeInForce};
use crate::models::order_book::OrderBook;
use crate::services::optimized_rest_client::OptimizedRestClient;
use crate::services::paper_trading::virtual_account::{Trade, VirtualAccount, VirtualAccountError};
use rand::Rng;
use std::sync::Arc;
use thiserror::Error;
use tracing::{debug, error, info, warn};

/// MEXC fee structure (taker fee)
const MEXC_TAKER_FEE: f64 = 0.001; // 0.1%

/// MEXC fee structure (maker fee)
const MEXC_MAKER_FEE: f64 = 0.0008; // 0.08%

/// Matching engine error type
#[derive(Debug, Error)]
pub enum MatchingError {
    #[error("Market data error: {0}")]
    MarketDataError(String),
    
    #[error("Virtual account error: {0}")]
    VirtualAccountError(#[from] VirtualAccountError),
    
    #[error("Execution error: {0}")]
    ExecutionError(String),
}

/// Matching engine for paper trading
pub struct MatchingEngine {
    market_data_client: Arc<OptimizedRestClient>,
    max_slippage: f64,
}

impl MatchingEngine {
    /// Create a new matching engine
    pub fn new(market_data_client: Arc<OptimizedRestClient>, max_slippage: f64) -> Self {
        MatchingEngine {
            market_data_client,
            max_slippage,
        }
    }
    
    /// Process an order against current market data
    pub async fn process_order(&self, order: &Order, account: &mut VirtualAccount) -> Result<Vec<Trade>, MatchingError> {
        // Get current market data
        let order_book = self.market_data_client.get_order_book(&order.symbol, 100).await
            .map_err(|e| MatchingError::MarketDataError(e))?;
        
        // Process based on order type
        match order.order_type {
            OrderType::Market => {
                self.execute_market_order(order, &order_book, account).await
            },
            OrderType::Limit => {
                self.execute_limit_order(order, &order_book, account).await
            },
            OrderType::StopLoss => {
                self.execute_stop_order(order, &order_book, account).await
            },
            OrderType::TakeProfit => {
                self.execute_stop_order(order, &order_book, account).await
            },
            _ => {
                // Other order types not implemented yet
                Err(MatchingError::ExecutionError(format!(
                    "Order type {:?} not implemented", order.order_type
                )))
            }
        }
    }
    
    /// Execute a market order
    async fn execute_market_order(&self, order: &Order, order_book: &OrderBook, account: &mut VirtualAccount) -> Result<Vec<Trade>, MatchingError> {
        // Remaining quantity to execute
        let mut remaining_qty = order.remaining_qty();
        
        // Get the relevant side of the order book
        let levels = match order.side {
            OrderSide::Buy => &order_book.asks.levels,
            OrderSide::Sell => &order_book.bids.levels,
        };
        
        // Check if there's enough liquidity
        if levels.is_empty() {
            return Err(MatchingError::ExecutionError(
                "No liquidity available".to_string()
            ));
        }
        
        // Calculate realistic slippage
        let slippage_factor = self.calculate_slippage(order, order_book);
        
        // Execute against each level until filled or no more liquidity
        let mut trades = Vec::new();
        
        for &(price, available_qty) in levels {
            // Apply slippage
            let execution_price = if order.side == OrderSide::Buy {
                price * (1.0 + slippage_factor)
            } else {
                price * (1.0 - slippage_factor)
            };
            
            // Calculate fill quantity
            let fill_qty = remaining_qty.min(available_qty);
            
            if fill_qty > 0.0 {
                // Calculate fill value and commission
                let fill_value = fill_qty * execution_price;
                let commission = fill_value * MEXC_TAKER_FEE;
                
                // Process the trade
                let trade = account.process_trade(
                    &order.id,
                    execution_price,
                    fill_qty,
                    commission,
                    "USDC"
                )?;
                
                trades.push(trade);
                
                // Update remaining quantity
                remaining_qty -= fill_qty;
                
                if remaining_qty <= 0.0 {
                    break;
                }
            }
        }
        
        // Update order status based on executed quantity
        if remaining_qty <= 0.0 {
            account.update_order_status(order.id.clone(), OrderStatus::Filled)?;
        } else if trades.is_empty() {
            account.update_order_status(order.id.clone(), OrderStatus::Rejected)?;
            return Err(MatchingError::ExecutionError(
                "No liquidity available to execute order".to_string()
            ));
        } else {
            account.update_order_status(order.id.clone(), OrderStatus::PartiallyFilled)?;
        }
        
        Ok(trades)
    }
    
    /// Execute a limit order
    async fn execute_limit_order(&self, order: &Order, order_book: &OrderBook, account: &mut VirtualAccount) -> Result<Vec<Trade>, MatchingError> {
        // Check if limit price is provided
        let limit_price = match order.price {
            Some(price) => price,
            None => return Err(MatchingError::ExecutionError(
                "Limit order requires a price".to_string()
            )),
        };
        
        // Check if order can be executed based on limit price
        let can_execute = match order.side {
            OrderSide::Buy => {
                !order_book.asks.levels.is_empty() && order_book.asks.levels[0].0 <= limit_price
            },
            OrderSide::Sell => {
                !order_book.bids.levels.is_empty() && order_book.bids.levels[0].0 >= limit_price
            },
        };
        
        // If order can be executed, process it
        if can_execute {
            // Remaining quantity to execute
            let mut remaining_qty = order.remaining_qty();
            
            // Get the relevant side of the order book
            let levels = match order.side {
                OrderSide::Buy => &order_book.asks.levels,
                OrderSide::Sell => &order_book.bids.levels,
            };
            
            // Execute against each level until filled, no more liquidity, or price exceeds limit
            let mut trades = Vec::new();
            
            for &(price, available_qty) in levels {
                // Check if price is within limit
                let price_ok = match order.side {
                    OrderSide::Buy => price <= limit_price,
                    OrderSide::Sell => price >= limit_price,
                };
                
                if !price_ok {
                    break;
                }
                
                // Calculate fill quantity
                let fill_qty = remaining_qty.min(available_qty);
                
                if fill_qty > 0.0 {
                    // Calculate fill value and commission (maker fee for limit orders)
                    let fill_value = fill_qty * price;
                    let commission = fill_value * MEXC_MAKER_FEE;
                    
                    // Process the trade
                    let trade = account.process_trade(
                        &order.id,
                        price,
                        fill_qty,
                        commission,
                        "USDC"
                    )?;
                    
                    trades.push(trade);
                    
                    // Update remaining quantity
                    remaining_qty -= fill_qty;
                    
                    if remaining_qty <= 0.0 {
                        break;
                    }
                }
            }
            
            // Update order status based on executed quantity
            if remaining_qty <= 0.0 {
                account.update_order_status(order.id.clone(), OrderStatus::Filled)?;
            } else if !trades.is_empty() {
                account.update_order_status(order.id.clone(), OrderStatus::PartiallyFilled)?;
            } else {
                // No fills but still within limit - keep as New
                account.update_order_status(order.id.clone(), OrderStatus::New)?;
            }
            
            Ok(trades)
        } else {
            // Price not yet reached, set as New
            account.update_order_status(order.id.clone(), OrderStatus::New)?;
            Ok(Vec::new())
        }
    }
    
    /// Execute a stop order (stop loss or take profit)
    async fn execute_stop_order(&self, order: &Order, order_book: &OrderBook, account: &mut VirtualAccount) -> Result<Vec<Trade>, MatchingError> {
        // Check if stop price is provided
        let stop_price = match order.stop_price {
            Some(price) => price,
            None => return Err(MatchingError::ExecutionError(
                "Stop order requires a stop price".to_string()
            )),
        };
        
        // Check if stop price is triggered
        let is_triggered = match order.side {
            OrderSide::Buy => {
                !order_book.asks.levels.is_empty() && order_book.asks.levels[0].0 >= stop_price
            },
            OrderSide::Sell => {
                !order_book.bids.levels.is_empty() && order_book.bids.levels[0].0 <= stop_price
            },
        };
        
        // If triggered, convert to market order and execute
        if is_triggered {
            // Create a market order with the same parameters
            let mut market_order = order.clone();
            market_order.order_type = OrderType::Market;
            
            // Execute as market order
            self.execute_market_order(&market_order, order_book, account).await
        } else {
            // Not triggered, keep as New
            account.update_order_status(order.id.clone(), OrderStatus::New)?;
            Ok(Vec::new())
        }
    }
    
    /// Calculate realistic slippage based on order size and book depth
    fn calculate_slippage(&self, order: &Order, order_book: &OrderBook) -> f64 {
        // Use order's slippage factor if available
        if order.slippage_factor > 0.0 {
            return order.slippage_factor;
        }
        
        // Get the relevant side of the order book
        let levels = match order.side {
            OrderSide::Buy => &order_book.asks.levels,
            OrderSide::Sell => &order_book.bids.levels,
        };
        
        if levels.is_empty() {
            return self.max_slippage;
        }
        
        // Calculate total available liquidity
        let total_liquidity: f64 = levels.iter().map(|&(_, qty)| qty).sum();
        
        // Calculate order size as percentage of available liquidity
        let order_size_percentage = if total_liquidity > 0.0 {
            order.quantity / total_liquidity
        } else {
            1.0
        };
        
        // Calculate slippage based on order size
        let base_slippage = if order_size_percentage < 0.01 {
            // Small order: 0.01% - 0.05%
            0.0001 + (0.0004 * order_size_percentage * 100.0)
        } else if order_size_percentage < 0.05 {
            // Medium order: 0.05% - 0.2%
            0.0005 + (0.0015 * order_size_percentage * 20.0)
        } else if order_size_percentage < 0.2 {
            // Large order: 0.2% - 0.5%
            0.002 + (0.003 * order_size_percentage * 5.0)
        } else {
            // Very large order: 0.5% - max
            0.005 + (self.max_slippage - 0.005) * (order_size_percentage.min(1.0))
        };
        
        // Add some randomness
        let mut rng = rand::thread_rng();
        let randomness = rng.gen_range(0.8..1.2);
        
        // Return slippage capped at max
        (base_slippage * randomness).min(self.max_slippage)
    }
}
