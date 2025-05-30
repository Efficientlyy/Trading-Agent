use serde::{Deserialize, Serialize};

/// Order side (buy or sell)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderSide {
    #[serde(rename = "BUY")]
    Buy,
    #[serde(rename = "SELL")]
    Sell,
}

/// Order type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderType {
    #[serde(rename = "LIMIT")]
    Limit,
    #[serde(rename = "MARKET")]
    Market,
    #[serde(rename = "STOP_LOSS")]
    StopLoss,
    #[serde(rename = "STOP_LOSS_LIMIT")]
    StopLossLimit,
    #[serde(rename = "TAKE_PROFIT")]
    TakeProfit,
    #[serde(rename = "TAKE_PROFIT_LIMIT")]
    TakeProfitLimit,
}

/// Order status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderStatus {
    #[serde(rename = "NEW")]
    New,
    #[serde(rename = "PARTIALLY_FILLED")]
    PartiallyFilled,
    #[serde(rename = "FILLED")]
    Filled,
    #[serde(rename = "CANCELED")]
    Canceled,
    #[serde(rename = "REJECTED")]
    Rejected,
    #[serde(rename = "EXPIRED")]
    Expired,
}

/// Represents an order
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    /// Order ID
    pub id: String,
    
    /// Trading pair symbol (e.g., "BTCUSDC")
    pub symbol: String,
    
    /// Order side (buy or sell)
    pub side: OrderSide,
    
    /// Order type
    pub order_type: OrderType,
    
    /// Order price (null for market orders)
    pub price: Option<f64>,
    
    /// Order quantity
    pub quantity: f64,
    
    /// Order status
    pub status: OrderStatus,
    
    /// Timestamp when the order was created
    pub timestamp: u64,
    
    /// Filled quantity
    pub filled_quantity: f64,
    
    /// Average fill price
    pub avg_fill_price: Option<f64>,
    
    /// Fee amount
    pub fee: Option<f64>,
    
    /// Fee asset
    pub fee_asset: Option<String>,
}

impl Order {
    /// Create a new order instance
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: String,
        symbol: String,
        side: OrderSide,
        order_type: OrderType,
        price: Option<f64>,
        quantity: f64,
        status: OrderStatus,
        timestamp: u64,
    ) -> Self {
        Self {
            id,
            symbol,
            side,
            order_type,
            price,
            quantity,
            status,
            timestamp,
            filled_quantity: 0.0,
            avg_fill_price: None,
            fee: None,
            fee_asset: None,
        }
    }
    
    /// Check if the order is active
    pub fn is_active(&self) -> bool {
        matches!(self.status, OrderStatus::New | OrderStatus::PartiallyFilled)
    }
    
    /// Check if the order is fully filled
    pub fn is_filled(&self) -> bool {
        self.status == OrderStatus::Filled
    }
    
    /// Get the remaining quantity to be filled
    pub fn remaining_quantity(&self) -> f64 {
        self.quantity - self.filled_quantity
    }
    
    /// Get the fill percentage
    pub fn fill_percentage(&self) -> f64 {
        if self.quantity == 0.0 {
            0.0
        } else {
            (self.filled_quantity / self.quantity) * 100.0
        }
    }
    
    /// Get the total value of the order
    pub fn total_value(&self) -> Option<f64> {
        self.price.map(|p| p * self.quantity)
    }
    
    /// Get the filled value of the order
    pub fn filled_value(&self) -> Option<f64> {
        self.avg_fill_price.map(|p| p * self.filled_quantity)
    }
    
    /// Convert order to JSON string
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
    
    /// Create order from JSON string
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_order_active_status() {
        let new_order = Order::new(
            "123".to_string(),
            "BTCUSDC".to_string(),
            OrderSide::Buy,
            OrderType::Limit,
            Some(35000.0),
            0.1,
            OrderStatus::New,
            1621500000000,
        );
        
        let filled_order = Order::new(
            "456".to_string(),
            "BTCUSDC".to_string(),
            OrderSide::Sell,
            OrderType::Market,
            None,
            0.1,
            OrderStatus::Filled,
            1621500000000,
        );
        
        assert!(new_order.is_active());
        assert!(!filled_order.is_active());
    }
    
    #[test]
    fn test_order_remaining_quantity() {
        let mut order = Order::new(
            "123".to_string(),
            "BTCUSDC".to_string(),
            OrderSide::Buy,
            OrderType::Limit,
            Some(35000.0),
            0.1,
            OrderStatus::PartiallyFilled,
            1621500000000,
        );
        
        order.filled_quantity = 0.04;
        
        assert_eq!(order.remaining_quantity(), 0.06);
        assert_eq!(order.fill_percentage(), 40.0);
    }
}
