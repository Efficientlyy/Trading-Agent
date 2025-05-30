use crate::models::{Order, OrderSide, OrderStatus, OrderType, Position};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

/// Paper trading engine for simulating trades with real market data
pub struct PaperTradingEngine {
    /// USDC balance
    usdc_balance: f64,
    
    /// BTC balance
    btc_balance: f64,
    
    /// Current BTC/USDC price
    current_price: f64,
    
    /// Open orders
    open_orders: HashMap<String, Order>,
    
    /// Order history
    order_history: Vec<Order>,
    
    /// Current position
    position: Option<Position>,
    
    /// Total realized profit/loss
    total_realized_pnl: f64,
}

impl PaperTradingEngine {
    /// Create a new paper trading engine
    pub fn new(initial_usdc: f64, initial_btc: f64) -> Self {
        Self {
            usdc_balance: initial_usdc,
            btc_balance: initial_btc,
            current_price: 0.0,
            open_orders: HashMap::new(),
            order_history: Vec::new(),
            position: None,
            total_realized_pnl: 0.0,
        }
    }
    
    /// Update the current market price
    pub fn update_price(&mut self, price: f64) {
        self.current_price = price;
        
        // Update position with new price
        if let Some(position) = &mut self.position {
            let timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64;
                
            position.update_price(price, timestamp);
        }
        
        // Process limit orders that might be triggered by the new price
        self.process_limit_orders();
    }
    
    /// Process limit orders that might be triggered by the current price
    fn process_limit_orders(&mut self) {
        // Collect orders that should be executed
        let mut orders_to_execute = Vec::new();
        
        for (id, order) in &self.open_orders {
            match order.order_type {
                OrderType::Limit => {
                    match order.side {
                        OrderSide::Buy => {
                            // Execute buy limit order if price falls below limit
                            if let Some(limit_price) = order.price {
                                if self.current_price <= limit_price {
                                    orders_to_execute.push(id.clone());
                                }
                            }
                        }
                        OrderSide::Sell => {
                            // Execute sell limit order if price rises above limit
                            if let Some(limit_price) = order.price {
                                if self.current_price >= limit_price {
                                    orders_to_execute.push(id.clone());
                                }
                            }
                        }
                    }
                }
                // Add other order types as needed
                _ => {}
            }
        }
        
        // Execute the collected orders
        for id in orders_to_execute {
            if let Some(order) = self.open_orders.remove(&id) {
                self.execute_order(order);
            }
        }
    }
    
    /// Place a new order
    pub fn place_order(
        &mut self,
        symbol: &str,
        side: OrderSide,
        order_type: OrderType,
        price: Option<f64>,
        quantity: f64,
    ) -> Result<String, String> {
        // Validate order
        if quantity <= 0.0 {
            return Err("Quantity must be positive".to_string());
        }
        
        if order_type == OrderType::Limit && price.is_none() {
            return Err("Limit orders require a price".to_string());
        }
        
        // Check if we have enough balance
        match side {
            OrderSide::Buy => {
                let required_balance = match order_type {
                    OrderType::Market => quantity * self.current_price,
                    OrderType::Limit => {
                        if let Some(limit_price) = price {
                            quantity * limit_price
                        } else {
                            return Err("Limit price is required".to_string());
                        }
                    }
                    _ => return Err("Unsupported order type".to_string()),
                };
                
                if required_balance > self.usdc_balance {
                    return Err("Insufficient USDC balance".to_string());
                }
            }
            OrderSide::Sell => {
                if quantity > self.btc_balance {
                    return Err("Insufficient BTC balance".to_string());
                }
            }
        }
        
        // Create order
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
            
        let id = format!("order-{}", timestamp);
        
        let order = Order {
            id: id.clone(),
            symbol: symbol.to_string(),
            side,
            order_type,
            price,
            quantity,
            status: OrderStatus::New,
            timestamp,
            filled_quantity: 0.0,
            avg_fill_price: None,
            fee: None,
            fee_asset: None,
        };
        
        // For market orders, execute immediately
        if order_type == OrderType::Market {
            self.execute_order(order.clone());
        } else {
            // For limit orders, add to open orders
            self.open_orders.insert(id.clone(), order);
        }
        
        Ok(id)
    }
    
    /// Execute an order
    fn execute_order(&mut self, mut order: Order) {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
            
        // Determine execution price
        let execution_price = match order.order_type {
            OrderType::Market => self.current_price,
            OrderType::Limit => order.price.unwrap_or(self.current_price),
            _ => self.current_price, // Default for other order types
        };
        
        // Apply a small slippage for market orders (0.1%)
        let adjusted_price = if order.order_type == OrderType::Market {
            match order.side {
                OrderSide::Buy => execution_price * 1.001, // Pay slightly more when buying
                OrderSide::Sell => execution_price * 0.999, // Receive slightly less when selling
            }
        } else {
            execution_price
        };
        
        // Calculate fee (0.1% of transaction value)
        let fee = adjusted_price * order.quantity * 0.001;
        
        // Update order
        order.status = OrderStatus::Filled;
        order.filled_quantity = order.quantity;
        order.avg_fill_price = Some(adjusted_price);
        order.fee = Some(fee);
        order.fee_asset = Some("USDC".to_string());
        
        // Update balances
        match order.side {
            OrderSide::Buy => {
                let cost = adjusted_price * order.quantity + fee;
                self.usdc_balance -= cost;
                self.btc_balance += order.quantity;
                
                // Update position
                if let Some(position) = &mut self.position {
                    position.add(order.quantity, adjusted_price, timestamp);
                } else {
                    self.position = Some(Position::new(
                        order.symbol.clone(),
                        order.quantity,
                        adjusted_price,
                        self.current_price,
                        timestamp,
                    ));
                }
            }
            OrderSide::Sell => {
                let proceeds = adjusted_price * order.quantity - fee;
                self.usdc_balance += proceeds;
                self.btc_balance -= order.quantity;
                
                // Update position
                if let Some(position) = &mut self.position {
                    position.add(-order.quantity, adjusted_price, timestamp);
                    
                    // If position is closed, update total realized PnL
                    if position.size == 0.0 {
                        self.total_realized_pnl += position.realized_pnl;
                        self.position = None;
                    }
                }
            }
        }
        
        // Add to order history
        self.order_history.push(order);
    }
    
    /// Cancel an open order
    pub fn cancel_order(&mut self, order_id: &str) -> Result<(), String> {
        if let Some(mut order) = self.open_orders.remove(order_id) {
            order.status = OrderStatus::Canceled;
            self.order_history.push(order);
            Ok(())
        } else {
            Err("Order not found".to_string())
        }
    }
    
    /// Get current USDC balance
    pub fn usdc_balance(&self) -> f64 {
        self.usdc_balance
    }
    
    /// Get current BTC balance
    pub fn btc_balance(&self) -> f64 {
        self.btc_balance
    }
    
    /// Get current position
    pub fn position(&self) -> Option<&Position> {
        self.position.as_ref()
    }
    
    /// Get open orders
    pub fn open_orders(&self) -> &HashMap<String, Order> {
        &self.open_orders
    }
    
    /// Get order history
    pub fn order_history(&self) -> &[Order] {
        &self.order_history
    }
    
    /// Get total portfolio value in USDC
    pub fn portfolio_value(&self) -> f64 {
        self.usdc_balance + (self.btc_balance * self.current_price)
    }
    
    /// Get unrealized profit/loss
    pub fn unrealized_pnl(&self) -> f64 {
        self.position
            .as_ref()
            .map(|p| p.unrealized_pnl)
            .unwrap_or(0.0)
    }
    
    /// Get total profit/loss (realized + unrealized)
    pub fn total_pnl(&self) -> f64 {
        self.total_realized_pnl + self.unrealized_pnl()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_place_market_buy_order() {
        let mut engine = PaperTradingEngine::new(10000.0, 0.0);
        engine.update_price(35000.0);
        
        let result = engine.place_order(
            "BTCUSDC",
            OrderSide::Buy,
            OrderType::Market,
            None,
            0.1,
        );
        
        assert!(result.is_ok());
        assert_eq!(engine.btc_balance(), 0.1);
        assert!(engine.usdc_balance() < 10000.0); // Balance reduced by cost + fee
    }
    
    #[test]
    fn test_place_limit_buy_order() {
        let mut engine = PaperTradingEngine::new(10000.0, 0.0);
        engine.update_price(35000.0);
        
        let result = engine.place_order(
            "BTCUSDC",
            OrderSide::Buy,
            OrderType::Limit,
            Some(34000.0),
            0.1,
        );
        
        assert!(result.is_ok());
        assert_eq!(engine.btc_balance(), 0.0); // Order not executed yet
        assert_eq!(engine.usdc_balance(), 10000.0); // Balance not reduced yet
        assert_eq!(engine.open_orders().len(), 1);
        
        // Update price to trigger limit order
        engine.update_price(34000.0);
        
        assert_eq!(engine.btc_balance(), 0.1); // Order executed
        assert!(engine.usdc_balance() < 10000.0); // Balance reduced by cost + fee
        assert_eq!(engine.open_orders().len(), 0);
    }
    
    #[test]
    fn test_insufficient_balance() {
        let mut engine = PaperTradingEngine::new(10000.0, 0.0);
        engine.update_price(35000.0);
        
        let result = engine.place_order(
            "BTCUSDC",
            OrderSide::Buy,
            OrderType::Market,
            None,
            1.0, // Too much BTC to buy with available balance
        );
        
        assert!(result.is_err());
        assert_eq!(engine.btc_balance(), 0.0);
        assert_eq!(engine.usdc_balance(), 10000.0);
    }
}
