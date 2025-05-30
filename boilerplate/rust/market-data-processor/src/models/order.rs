use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fmt;
use uuid::Uuid;

/// Order side (Buy or Sell)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderSide {
    Buy,
    Sell,
}

impl fmt::Display for OrderSide {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            OrderSide::Buy => write!(f, "BUY"),
            OrderSide::Sell => write!(f, "SELL"),
        }
    }
}

/// Order type (Market, Limit, etc.)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderType {
    Market,
    Limit,
    StopLoss,
    TakeProfit,
    StopLossLimit,
    TakeProfitLimit,
}

impl fmt::Display for OrderType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            OrderType::Market => write!(f, "MARKET"),
            OrderType::Limit => write!(f, "LIMIT"),
            OrderType::StopLoss => write!(f, "STOP_LOSS"),
            OrderType::TakeProfit => write!(f, "TAKE_PROFIT"),
            OrderType::StopLossLimit => write!(f, "STOP_LOSS_LIMIT"),
            OrderType::TakeProfitLimit => write!(f, "TAKE_PROFIT_LIMIT"),
        }
    }
}

/// Time in force options for orders
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TimeInForce {
    GoodTillCancel,  // GTC
    ImmediateOrCancel, // IOC
    FillOrKill,      // FOK
    GoodTillDate,    // GTD
}

impl fmt::Display for TimeInForce {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TimeInForce::GoodTillCancel => write!(f, "GTC"),
            TimeInForce::ImmediateOrCancel => write!(f, "IOC"),
            TimeInForce::FillOrKill => write!(f, "FOK"),
            TimeInForce::GoodTillDate => write!(f, "GTD"),
        }
    }
}

/// Order status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderStatus {
    Created,
    New,
    PartiallyFilled,
    Filled,
    Canceled,
    Rejected,
    Expired,
}

impl fmt::Display for OrderStatus {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            OrderStatus::Created => write!(f, "CREATED"),
            OrderStatus::New => write!(f, "NEW"),
            OrderStatus::PartiallyFilled => write!(f, "PARTIALLY_FILLED"),
            OrderStatus::Filled => write!(f, "FILLED"),
            OrderStatus::Canceled => write!(f, "CANCELED"),
            OrderStatus::Rejected => write!(f, "REJECTED"),
            OrderStatus::Expired => write!(f, "EXPIRED"),
        }
    }
}

/// Fill details for an order
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderFill {
    pub price: f64,
    pub quantity: f64,
    pub commission: f64,
    pub commission_asset: String,
    pub trade_id: String,
    pub timestamp: DateTime<Utc>,
}

/// Order structure for paper trading
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    // Order identification
    pub id: String,
    pub client_order_id: Option<String>,
    pub symbol: String,
    
    // Order details
    pub side: OrderSide,
    pub order_type: OrderType,
    pub quantity: f64,
    pub price: Option<f64>,
    pub stop_price: Option<f64>,
    
    // Order execution parameters
    pub time_in_force: TimeInForce,
    pub iceberg_qty: Option<f64>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    
    // Order status and execution details
    pub status: OrderStatus,
    pub executed_qty: f64,
    pub avg_execution_price: Option<f64>,
    pub fills: Vec<OrderFill>,
    pub cumulative_commission: f64,
    
    // Simulation specific fields
    pub expected_latency_ms: u64,   // Expected execution latency in milliseconds
    pub slippage_factor: f64,       // Slippage as a factor (0.001 = 0.1%)
    pub scheduled_execution_time: Option<DateTime<Utc>>, // Time when the order will be executed
}

impl Order {
    /// Create a new market order
    pub fn new_market_order(
        symbol: String,
        side: OrderSide,
        quantity: f64,
        expected_latency_ms: u64,
        slippage_factor: f64,
    ) -> Self {
        let now = Utc::now();
        
        Self {
            id: Uuid::new_v4().to_string(),
            client_order_id: None,
            symbol,
            side,
            order_type: OrderType::Market,
            quantity,
            price: None,
            stop_price: None,
            time_in_force: TimeInForce::GoodTillCancel,
            iceberg_qty: None,
            created_at: now,
            updated_at: now,
            status: OrderStatus::Created,
            executed_qty: 0.0,
            avg_execution_price: None,
            fills: Vec::new(),
            cumulative_commission: 0.0,
            expected_latency_ms,
            slippage_factor,
            scheduled_execution_time: Some(now + chrono::Duration::milliseconds(expected_latency_ms as i64)),
        }
    }
    
    /// Create a new limit order
    pub fn new_limit_order(
        symbol: String,
        side: OrderSide,
        quantity: f64,
        price: f64,
        time_in_force: TimeInForce,
        expected_latency_ms: u64,
        slippage_factor: f64,
    ) -> Self {
        let now = Utc::now();
        
        Self {
            id: Uuid::new_v4().to_string(),
            client_order_id: None,
            symbol,
            side,
            order_type: OrderType::Limit,
            quantity,
            price: Some(price),
            stop_price: None,
            time_in_force,
            iceberg_qty: None,
            created_at: now,
            updated_at: now,
            status: OrderStatus::Created,
            executed_qty: 0.0,
            avg_execution_price: None,
            fills: Vec::new(),
            cumulative_commission: 0.0,
            expected_latency_ms,
            slippage_factor,
            scheduled_execution_time: Some(now + chrono::Duration::milliseconds(expected_latency_ms as i64)),
        }
    }
    
    /// Create a new stop loss order
    pub fn new_stop_loss_order(
        symbol: String,
        side: OrderSide,
        quantity: f64,
        stop_price: f64,
        expected_latency_ms: u64,
        slippage_factor: f64,
    ) -> Self {
        let now = Utc::now();
        
        Self {
            id: Uuid::new_v4().to_string(),
            client_order_id: None,
            symbol,
            side,
            order_type: OrderType::StopLoss,
            quantity,
            price: None,
            stop_price: Some(stop_price),
            time_in_force: TimeInForce::GoodTillCancel,
            iceberg_qty: None,
            created_at: now,
            updated_at: now,
            status: OrderStatus::Created,
            executed_qty: 0.0,
            avg_execution_price: None,
            fills: Vec::new(),
            cumulative_commission: 0.0,
            expected_latency_ms,
            slippage_factor,
            scheduled_execution_time: None, // Will be set when stop price is triggered
        }
    }
    
    /// Calculate the total cost of the order (quantity * price)
    pub fn total_cost(&self) -> Option<f64> {
        match self.avg_execution_price {
            Some(price) => Some(self.executed_qty * price),
            None => None,
        }
    }
    
    /// Calculate the remaining quantity to be executed
    pub fn remaining_qty(&self) -> f64 {
        self.quantity - self.executed_qty
    }
    
    /// Check if the order is active (can be executed)
    pub fn is_active(&self) -> bool {
        matches!(
            self.status,
            OrderStatus::Created | OrderStatus::New | OrderStatus::PartiallyFilled
        )
    }
    
    /// Check if the order is fully executed
    pub fn is_filled(&self) -> bool {
        self.status == OrderStatus::Filled
    }
    
    /// Check if the order execution is due based on latency
    pub fn is_execution_due(&self) -> bool {
        if let Some(execution_time) = self.scheduled_execution_time {
            Utc::now() >= execution_time
        } else {
            false
        }
    }
    
    /// Add a fill to the order
    pub fn add_fill(
        &mut self,
        price: f64,
        quantity: f64,
        commission: f64,
        commission_asset: String,
    ) {
        let fill = OrderFill {
            price,
            quantity,
            commission,
            commission_asset,
            trade_id: Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
        };
        
        // Update order execution details
        self.executed_qty += quantity;
        self.cumulative_commission += commission;
        
        // Update average execution price
        let total_value = self.fills.iter().map(|f| f.price * f.quantity).sum::<f64>() + (price * quantity);
        self.avg_execution_price = Some(total_value / self.executed_qty);
        
        // Update order status
        if (self.executed_qty - self.quantity).abs() < f64::EPSILON {
            self.status = OrderStatus::Filled;
        } else if self.executed_qty > 0.0 {
            self.status = OrderStatus::PartiallyFilled;
        }
        
        // Add fill to the list
        self.fills.push(fill);
        
        // Update timestamp
        self.updated_at = Utc::now();
    }
    
    /// Cancel the order
    pub fn cancel(&mut self) -> bool {
        if self.is_active() {
            self.status = OrderStatus::Canceled;
            self.updated_at = Utc::now();
            true
        } else {
            false
        }
    }
    
    /// Reject the order
    pub fn reject(&mut self, reason: &str) {
        self.status = OrderStatus::Rejected;
        self.updated_at = Utc::now();
        // In a real implementation, we would store the rejection reason
    }
    
    /// Set the order as new (accepted by the exchange)
    pub fn set_new(&mut self) {
        if self.status == OrderStatus::Created {
            self.status = OrderStatus::New;
            self.updated_at = Utc::now();
        }
    }
}
