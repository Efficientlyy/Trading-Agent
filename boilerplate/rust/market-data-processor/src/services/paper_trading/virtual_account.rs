use crate::models::order::{Order, OrderFill, OrderSide, OrderStatus};
use crate::models::position::{Position, PositionDirection, PositionTrade};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// Virtual account error type
#[derive(Debug, Error)]
pub enum VirtualAccountError {
    #[error("Insufficient balance: {0}")]
    InsufficientBalance(String),
    
    #[error("Invalid order: {0}")]
    InvalidOrder(String),
    
    #[error("Order not found: {0}")]
    OrderNotFound(String),
    
    #[error("Position not found: {0}")]
    PositionNotFound(String),
}

/// Trade record structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub id: String,
    pub order_id: String,
    pub symbol: String,
    pub side: OrderSide,
    pub quantity: f64,
    pub price: f64,
    pub fee: f64,
    pub fee_currency: String,
    pub timestamp: DateTime<Utc>,
}

/// Virtual account for paper trading
pub struct VirtualAccount {
    balances: HashMap<String, f64>,
    positions: HashMap<String, Position>,
    orders: HashMap<String, Order>,
    trades: Vec<Trade>,
    last_updated: DateTime<Utc>,
}

impl VirtualAccount {
    /// Create a new virtual account with initial balances
    pub fn new(initial_balances: HashMap<String, f64>) -> Self {
        VirtualAccount {
            balances: initial_balances,
            positions: HashMap::new(),
            orders: HashMap::new(),
            trades: Vec::new(),
            last_updated: Utc::now(),
        }
    }
    
    /// Validate an order before placing it
    pub fn validate_order(&self, order: &Order) -> Result<(), VirtualAccountError> {
        // Extract symbol components (e.g., BTCUSDC -> BTC, USDC)
        let parts: Vec<&str> = order.symbol.splitn(2, |c| c == 'U').collect();
        if parts.len() != 2 {
            return Err(VirtualAccountError::InvalidOrder(
                format!("Invalid symbol format: {}", order.symbol)
            ));
        }
        
        let base_asset = parts[0];
        let quote_asset = format!("U{}", parts[1]);
        
        // Check if we have enough balance for the order
        match order.side {
            OrderSide::Buy => {
                // For buy orders, check if we have enough quote currency (e.g., USDC)
                let quote_balance = self.balances.get(&quote_asset).copied().unwrap_or(0.0);
                
                // Calculate required amount (with some buffer for fees)
                let required_amount = if let Some(price) = order.price {
                    order.quantity * price * 1.01 // 1% buffer for fees
                } else {
                    // For market orders without price, require more buffer
                    order.quantity * self.estimate_market_price(&order.symbol, order.side) * 1.05
                };
                
                if quote_balance < required_amount {
                    return Err(VirtualAccountError::InsufficientBalance(
                        format!("Insufficient {} balance. Required: {}, Available: {}", 
                            quote_asset, required_amount, quote_balance)
                    ));
                }
            },
            OrderSide::Sell => {
                // For sell orders, check if we have enough base currency (e.g., BTC)
                let base_balance = self.balances.get(base_asset).copied().unwrap_or(0.0);
                
                if base_balance < order.quantity {
                    return Err(VirtualAccountError::InsufficientBalance(
                        format!("Insufficient {} balance. Required: {}, Available: {}", 
                            base_asset, order.quantity, base_balance)
                    ));
                }
            },
        }
        
        Ok(())
    }
    
    /// Place an order in the virtual account
    pub fn place_order(&mut self, order: Order) -> Result<String, VirtualAccountError> {
        // Validate order
        self.validate_order(&order)?;
        
        // Generate order ID if not provided
        let order_id = order.id.clone();
        
        // Store order
        self.orders.insert(order_id.clone(), order);
        self.last_updated = Utc::now();
        
        Ok(order_id)
    }
    
    /// Process a trade (execution of an order)
    pub fn process_trade(
        &mut self,
        order_id: &str,
        price: f64,
        quantity: f64,
        fee: f64,
        fee_currency: &str,
    ) -> Result<Trade, VirtualAccountError> {
        // Find the order
        let order = self.orders.get_mut(order_id)
            .ok_or_else(|| VirtualAccountError::OrderNotFound(order_id.to_string()))?;
        
        // Extract symbol components (e.g., BTCUSDC -> BTC, USDC)
        let parts: Vec<&str> = order.symbol.splitn(2, |c| c == 'U').collect();
        if parts.len() != 2 {
            return Err(VirtualAccountError::InvalidOrder(
                format!("Invalid symbol format: {}", order.symbol)
            ));
        }
        
        let base_asset = parts[0].to_string();
        let quote_asset = format!("U{}", parts[1]);
        
        // Create trade record
        let trade = Trade {
            id: Uuid::new_v4().to_string(),
            order_id: order_id.to_string(),
            symbol: order.symbol.clone(),
            side: order.side,
            quantity,
            price,
            fee,
            fee_currency: fee_currency.to_string(),
            timestamp: Utc::now(),
        };
        
        // Update order
        order.add_fill(price, quantity, fee, fee_currency.to_string());
        
        // Update balances
        match order.side {
            OrderSide::Buy => {
                // Increase base asset (e.g., BTC)
                *self.balances.entry(base_asset.clone()).or_insert(0.0) += quantity;
                
                // Decrease quote asset (e.g., USDC)
                let cost = price * quantity;
                *self.balances.entry(quote_asset.clone()).or_insert(0.0) -= (cost + fee);
            },
            OrderSide::Sell => {
                // Decrease base asset (e.g., BTC)
                *self.balances.entry(base_asset.clone()).or_insert(0.0) -= quantity;
                
                // Increase quote asset (e.g., USDC)
                let revenue = price * quantity;
                *self.balances.entry(quote_asset.clone()).or_insert(0.0) += (revenue - fee);
            },
        }
        
        // Update position
        let position = self.positions.entry(order.symbol.clone())
            .or_insert_with(|| Position::new(&order.symbol));
        
        let position_trade = PositionTrade::from_order(order, quantity, price, fee, fee_currency);
        position.add_trade(position_trade);
        
        // Add trade to history
        self.trades.push(trade.clone());
        self.last_updated = Utc::now();
        
        Ok(trade)
    }
    
    /// Update order status
    pub fn update_order_status(&mut self, order_id: String, status: OrderStatus) -> Result<(), VirtualAccountError> {
        let order = self.orders.get_mut(&order_id)
            .ok_or_else(|| VirtualAccountError::OrderNotFound(order_id))?;
        
        // Update status if allowed
        match (order.status, status) {
            // Allow transitions to Canceled only if not already filled
            (current, OrderStatus::Canceled) if current != OrderStatus::Filled => {
                order.status = OrderStatus::Canceled;
            },
            // Allow transitions to Rejected only if not already filled or canceled
            (current, OrderStatus::Rejected) if current != OrderStatus::Filled && current != OrderStatus::Canceled => {
                order.status = OrderStatus::Rejected;
            },
            // Don't allow other transitions if already in a terminal state
            (current, _) if current == OrderStatus::Filled || 
                            current == OrderStatus::Canceled || 
                            current == OrderStatus::Rejected => {
                return Err(VirtualAccountError::InvalidOrder(
                    format!("Cannot change order status from {:?} to {:?}", current, status)
                ));
            },
            // Allow all other transitions
            (_, _) => {
                order.status = status;
            }
        }
        
        self.last_updated = Utc::now();
        Ok(())
    }
    
    /// Get order by ID
    pub fn get_order(&self, order_id: &str) -> Option<Order> {
        self.orders.get(order_id).cloned()
    }
    
    /// Get all orders
    pub fn get_orders(&self) -> Vec<Order> {
        self.orders.values().cloned().collect()
    }
    
    /// Get active orders
    pub fn get_active_orders(&self) -> Vec<Order> {
        self.orders.values()
            .filter(|order| order.is_active())
            .cloned()
            .collect()
    }
    
    /// Get completed orders
    pub fn get_completed_orders(&self) -> Vec<Order> {
        self.orders.values()
            .filter(|order| !order.is_active())
            .cloned()
            .collect()
    }
    
    /// Get position by symbol
    pub fn get_position(&self, symbol: &str) -> Option<Position> {
        self.positions.get(symbol).cloned()
    }
    
    /// Get all positions
    pub fn get_positions(&self) -> Vec<Position> {
        self.positions.values().cloned().collect()
    }
    
    /// Get open positions
    pub fn get_open_positions(&self) -> Vec<Position> {
        self.positions.values()
            .filter(|pos| pos.is_open())
            .cloned()
            .collect()
    }
    
    /// Get trades
    pub fn get_trades(&self) -> Vec<Trade> {
        self.trades.clone()
    }
    
    /// Get balance for specific asset
    pub fn get_balance(&self, asset: &str) -> f64 {
        *self.balances.get(asset).unwrap_or(&0.0)
    }
    
    /// Get all balances
    pub fn get_balances(&self) -> Vec<(String, f64)> {
        self.balances.iter()
            .map(|(asset, balance)| (asset.clone(), *balance))
            .collect()
    }
    
    /// Update position with current market price
    pub fn update_position_price(&mut self, symbol: &str, price: f64) -> Result<(), VirtualAccountError> {
        let position = self.positions.get_mut(symbol)
            .ok_or_else(|| VirtualAccountError::PositionNotFound(symbol.to_string()))?;
        
        position.update_price(price);
        self.last_updated = Utc::now();
        
        Ok(())
    }
    
    /// Estimate market price for a symbol and side
    fn estimate_market_price(&self, symbol: &str, side: OrderSide) -> f64 {
        // In a real implementation, this would look up the current market price
        // For now, just return a placeholder value
        100.0
    }
}
