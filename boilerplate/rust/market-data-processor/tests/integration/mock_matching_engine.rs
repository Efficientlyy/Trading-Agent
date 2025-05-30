use std::sync::Arc;
use tokio::sync::Mutex;

use market_data_processor::models::order::{Order, OrderSide, OrderStatus, OrderType};
use market_data_processor::services::paper_trading::MatchingEngine;

use crate::test_framework::MockMarketDataService;

/// Mock matching engine implementation for testing
pub struct MockMatchingEngine {
    market_data: Arc<Mutex<MockMarketDataService>>,
    max_slippage: f64,
}

impl MockMatchingEngine {
    /// Create a new mock matching engine
    pub fn new(market_data: Arc<Mutex<MockMarketDataService>>, max_slippage: f64) -> Self {
        Self {
            market_data,
            max_slippage,
        }
    }
    
    /// Match an order against current market data
    pub async fn match_order(&self, order: &mut Order) -> Result<(), String> {
        // Skip if order is already filled or canceled
        if order.status == OrderStatus::Filled || order.status == OrderStatus::Canceled {
            return Ok(());
        }
        
        // Get current market data
        let market_data = self.market_data.lock().await;
        
        match order.order_type {
            OrderType::Market => {
                // Execute market order
                self.execute_market_order(order, &market_data).await?;
            },
            OrderType::Limit => {
                // Check if limit order can be executed
                if let Some(limit_price) = order.price {
                    self.execute_limit_order(order, limit_price, &market_data).await?;
                } else {
                    return Err("Limit order missing price".to_string());
                }
            },
        }
        
        Ok(())
    }
    
    /// Execute a market order
    async fn execute_market_order(
        &self,
        order: &mut Order,
        market_data: &MockMarketDataService,
    ) -> Result<(), String> {
        let symbol = &order.symbol;
        
        // Get current price
        let current_price = market_data.get_price(symbol)
            .ok_or_else(|| format!("No price data for {}", symbol))?;
        
        // Get order book
        let (bids, asks) = market_data.get_order_book(symbol)
            .ok_or_else(|| format!("No order book data for {}", symbol))?;
        
        // Calculate execution details
        let (executed_qty, avg_price) = match order.side {
            OrderSide::Buy => {
                // For buy orders, match against asks
                let mut remaining_qty = order.quantity - order.executed_qty;
                let mut total_cost = 0.0;
                let mut executed_qty_this_match = 0.0;
                
                for (price, qty) in asks.iter() {
                    // Apply slippage for large orders
                    let slippage = if remaining_qty > 1.0 {
                        remaining_qty * 0.001 * self.max_slippage
                    } else {
                        0.0
                    };
                    
                    let adjusted_price = price * (1.0 + slippage);
                    let matched_qty = remaining_qty.min(*qty);
                    
                    total_cost += matched_qty * adjusted_price;
                    executed_qty_this_match += matched_qty;
                    remaining_qty -= matched_qty;
                    
                    if remaining_qty <= 0.0 {
                        break;
                    }
                }
                
                if executed_qty_this_match > 0.0 {
                    let avg_price = total_cost / executed_qty_this_match;
                    (executed_qty_this_match, avg_price)
                } else {
                    (0.0, current_price)
                }
            },
            OrderSide::Sell => {
                // For sell orders, match against bids
                let mut remaining_qty = order.quantity - order.executed_qty;
                let mut total_value = 0.0;
                let mut executed_qty_this_match = 0.0;
                
                for (price, qty) in bids.iter() {
                    // Apply slippage for large orders
                    let slippage = if remaining_qty > 1.0 {
                        remaining_qty * 0.001 * self.max_slippage
                    } else {
                        0.0
                    };
                    
                    let adjusted_price = price * (1.0 - slippage);
                    let matched_qty = remaining_qty.min(*qty);
                    
                    total_value += matched_qty * adjusted_price;
                    executed_qty_this_match += matched_qty;
                    remaining_qty -= matched_qty;
                    
                    if remaining_qty <= 0.0 {
                        break;
                    }
                }
                
                if executed_qty_this_match > 0.0 {
                    let avg_price = total_value / executed_qty_this_match;
                    (executed_qty_this_match, avg_price)
                } else {
                    (0.0, current_price)
                }
            },
        };
        
        // Update order with execution details
        order.executed_qty += executed_qty;
        
        // Update average execution price
        if order.avg_execution_price.is_none() {
            order.avg_execution_price = Some(avg_price);
        } else if executed_qty > 0.0 {
            let prev_avg = order.avg_execution_price.unwrap();
            let prev_qty = order.executed_qty - executed_qty;
            let new_avg = (prev_avg * prev_qty + avg_price * executed_qty) / order.executed_qty;
            order.avg_execution_price = Some(new_avg);
        }
        
        // Update order status
        if (order.executed_qty - order.quantity).abs() < 0.000001 {
            order.status = OrderStatus::Filled;
        } else if order.executed_qty > 0.0 {
            order.status = OrderStatus::PartiallyFilled;
        }
        
        Ok(())
    }
    
    /// Execute a limit order
    async fn execute_limit_order(
        &self,
        order: &mut Order,
        limit_price: f64,
        market_data: &MockMarketDataService,
    ) -> Result<(), String> {
        let symbol = &order.symbol;
        
        // Get current price
        let current_price = market_data.get_price(symbol)
            .ok_or_else(|| format!("No price data for {}", symbol))?;
        
        // Check if limit price is valid for execution
        let can_execute = match order.side {
            OrderSide::Buy => current_price <= limit_price,
            OrderSide::Sell => current_price >= limit_price,
        };
        
        if can_execute {
            // Execute like a market order but with price limits
            let (bids, asks) = market_data.get_order_book(symbol)
                .ok_or_else(|| format!("No order book data for {}", symbol))?;
            
            // Calculate execution details
            let (executed_qty, avg_price) = match order.side {
                OrderSide::Buy => {
                    // For buy limit orders, match against asks that are <= limit_price
                    let mut remaining_qty = order.quantity - order.executed_qty;
                    let mut total_cost = 0.0;
                    let mut executed_qty_this_match = 0.0;
                    
                    for (price, qty) in asks.iter().filter(|(p, _)| *p <= limit_price) {
                        let matched_qty = remaining_qty.min(*qty);
                        total_cost += matched_qty * price;
                        executed_qty_this_match += matched_qty;
                        remaining_qty -= matched_qty;
                        
                        if remaining_qty <= 0.0 {
                            break;
                        }
                    }
                    
                    if executed_qty_this_match > 0.0 {
                        let avg_price = total_cost / executed_qty_this_match;
                        (executed_qty_this_match, avg_price)
                    } else {
                        (0.0, current_price)
                    }
                },
                OrderSide::Sell => {
                    // For sell limit orders, match against bids that are >= limit_price
                    let mut remaining_qty = order.quantity - order.executed_qty;
                    let mut total_value = 0.0;
                    let mut executed_qty_this_match = 0.0;
                    
                    for (price, qty) in bids.iter().filter(|(p, _)| *p >= limit_price) {
                        let matched_qty = remaining_qty.min(*qty);
                        total_value += matched_qty * price;
                        executed_qty_this_match += matched_qty;
                        remaining_qty -= matched_qty;
                        
                        if remaining_qty <= 0.0 {
                            break;
                        }
                    }
                    
                    if executed_qty_this_match > 0.0 {
                        let avg_price = total_value / executed_qty_this_match;
                        (executed_qty_this_match, avg_price)
                    } else {
                        (0.0, current_price)
                    }
                },
            };
            
            // Update order with execution details
            order.executed_qty += executed_qty;
            
            // Update average execution price
            if order.avg_execution_price.is_none() {
                order.avg_execution_price = Some(avg_price);
            } else if executed_qty > 0.0 {
                let prev_avg = order.avg_execution_price.unwrap();
                let prev_qty = order.executed_qty - executed_qty;
                let new_avg = (prev_avg * prev_qty + avg_price * executed_qty) / order.executed_qty;
                order.avg_execution_price = Some(new_avg);
            }
            
            // Update order status
            if (order.executed_qty - order.quantity).abs() < 0.000001 {
                order.status = OrderStatus::Filled;
            } else if order.executed_qty > 0.0 {
                order.status = OrderStatus::PartiallyFilled;
            }
        }
        
        Ok(())
    }
}
