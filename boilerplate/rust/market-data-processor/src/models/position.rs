use crate::models::order::{Order, OrderSide};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Position direction (Long or Short)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PositionDirection {
    Long,
    Short,
    Flat,
}

impl std::fmt::Display for PositionDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            PositionDirection::Long => write!(f, "LONG"),
            PositionDirection::Short => write!(f, "SHORT"),
            PositionDirection::Flat => write!(f, "FLAT"),
        }
    }
}

/// Position trade record - represents a single trade within a position
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionTrade {
    pub id: String,
    pub order_id: String,
    pub symbol: String,
    pub direction: OrderSide,
    pub quantity: f64,
    pub price: f64,
    pub commission: f64,
    pub commission_asset: String,
    pub timestamp: DateTime<Utc>,
}

impl PositionTrade {
    /// Create a new position trade from an order fill
    pub fn from_order(order: &Order, quantity: f64, price: f64, commission: f64, commission_asset: &str) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            order_id: order.id.clone(),
            symbol: order.symbol.clone(),
            direction: order.side,
            quantity,
            price,
            commission,
            commission_asset: commission_asset.to_string(),
            timestamp: Utc::now(),
        }
    }
    
    /// Calculate the trade value (quantity * price)
    pub fn value(&self) -> f64 {
        self.quantity * self.price
    }
}

/// Stop loss or take profit configuration for a position
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionExit {
    pub price: f64,
    pub quantity: Option<f64>, // None means entire position
    pub order_id: Option<String>,
    pub triggered: bool,
}

/// Position structure for paper trading
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub id: String,
    pub symbol: String,
    pub direction: PositionDirection,
    pub quantity: f64,
    pub entry_price: f64,
    pub avg_entry_price: f64,
    pub liquidation_price: Option<f64>,
    
    // Trade history
    pub trades: Vec<PositionTrade>,
    pub realized_pnl: f64,
    pub fees_paid: f64,
    
    // Position risk management
    pub stop_loss: Option<PositionExit>,
    pub take_profit: Option<PositionExit>,
    
    // Position timestamps
    pub opened_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub closed_at: Option<DateTime<Utc>>,
    
    // Current market data
    pub last_price: Option<f64>,
    pub mark_price: Option<f64>,
}

impl Position {
    /// Create a new position
    pub fn new(symbol: &str) -> Self {
        let now = Utc::now();
        
        Self {
            id: Uuid::new_v4().to_string(),
            symbol: symbol.to_string(),
            direction: PositionDirection::Flat,
            quantity: 0.0,
            entry_price: 0.0,
            avg_entry_price: 0.0,
            liquidation_price: None,
            trades: Vec::new(),
            realized_pnl: 0.0,
            fees_paid: 0.0,
            stop_loss: None,
            take_profit: None,
            opened_at: now,
            updated_at: now,
            closed_at: None,
            last_price: None,
            mark_price: None,
        }
    }
    
    /// Add a trade to the position
    pub fn add_trade(&mut self, trade: PositionTrade) {
        let trade_direction = if trade.direction == OrderSide::Buy {
            PositionDirection::Long
        } else {
            PositionDirection::Short
        };
        
        // Current position is flat
        if self.direction == PositionDirection::Flat {
            self.direction = trade_direction;
            self.quantity = trade.quantity;
            self.entry_price = trade.price;
            self.avg_entry_price = trade.price;
            self.opened_at = trade.timestamp;
        }
        // Adding to the same direction
        else if (self.direction == PositionDirection::Long && trade_direction == PositionDirection::Long)
            || (self.direction == PositionDirection::Short && trade_direction == PositionDirection::Short)
        {
            // Recalculate average entry price
            let total_value = self.avg_entry_price * self.quantity + trade.price * trade.quantity;
            self.quantity += trade.quantity;
            self.avg_entry_price = total_value / self.quantity;
        }
        // Reducing or closing position
        else {
            // Calculate realized PnL
            let close_quantity = trade.quantity.min(self.quantity);
            let price_diff = if self.direction == PositionDirection::Long {
                trade.price - self.avg_entry_price
            } else {
                self.avg_entry_price - trade.price
            };
            
            let trade_pnl = close_quantity * price_diff;
            self.realized_pnl += trade_pnl;
            
            // Update position
            self.quantity -= close_quantity;
            
            // Position is now flat
            if self.quantity < 0.000001 { // Small epsilon to handle floating point errors
                self.direction = PositionDirection::Flat;
                self.quantity = 0.0;
                self.closed_at = Some(trade.timestamp);
            }
            // Position flipped to opposite direction
            else if trade.quantity > self.quantity {
                let remaining_qty = trade.quantity - self.quantity;
                self.direction = trade_direction;
                self.quantity = remaining_qty;
                self.avg_entry_price = trade.price;
                self.closed_at = None;
            }
        }
        
        // Update fees paid
        self.fees_paid += trade.commission;
        
        // Add trade to history and update timestamp
        self.trades.push(trade);
        self.updated_at = Utc::now();
    }
    
    /// Update the position with the latest market price
    pub fn update_price(&mut self, price: f64) {
        self.last_price = Some(price);
        self.mark_price = Some(price); // In a real system, mark price might be different
        self.updated_at = Utc::now();
        
        // Check if stop loss or take profit is triggered
        self.check_exits();
    }
    
    /// Check if stop loss or take profit is triggered
    fn check_exits(&mut self) {
        if self.quantity <= 0.0 || self.mark_price.is_none() {
            return;
        }
        
        let current_price = self.mark_price.unwrap();
        
        // Check stop loss
        if let Some(ref mut stop_loss) = self.stop_loss {
            if !stop_loss.triggered {
                let is_triggered = match self.direction {
                    PositionDirection::Long => current_price <= stop_loss.price,
                    PositionDirection::Short => current_price >= stop_loss.price,
                    PositionDirection::Flat => false,
                };
                
                if is_triggered {
                    stop_loss.triggered = true;
                    // In a real implementation, this would create a market order to close the position
                }
            }
        }
        
        // Check take profit
        if let Some(ref mut take_profit) = self.take_profit {
            if !take_profit.triggered {
                let is_triggered = match self.direction {
                    PositionDirection::Long => current_price >= take_profit.price,
                    PositionDirection::Short => current_price <= take_profit.price,
                    PositionDirection::Flat => false,
                };
                
                if is_triggered {
                    take_profit.triggered = true;
                    // In a real implementation, this would create a market order to close the position
                }
            }
        }
    }
    
    /// Set a stop loss for the position
    pub fn set_stop_loss(&mut self, price: f64, quantity: Option<f64>) -> Result<(), String> {
        if self.quantity <= 0.0 {
            return Err("Cannot set stop loss on flat position".to_string());
        }
        
        // Validate stop loss price
        match self.direction {
            PositionDirection::Long => {
                if price >= self.avg_entry_price {
                    return Err("Stop loss price must be below entry price for long positions".to_string());
                }
            },
            PositionDirection::Short => {
                if price <= self.avg_entry_price {
                    return Err("Stop loss price must be above entry price for short positions".to_string());
                }
            },
            PositionDirection::Flat => {
                return Err("Cannot set stop loss on flat position".to_string());
            },
        }
        
        // Validate quantity
        if let Some(qty) = quantity {
            if qty > self.quantity {
                return Err(format!("Stop loss quantity ({}) exceeds position size ({})", qty, self.quantity));
            }
        }
        
        self.stop_loss = Some(PositionExit {
            price,
            quantity,
            order_id: None,
            triggered: false,
        });
        
        self.updated_at = Utc::now();
        Ok(())
    }
    
    /// Set a take profit for the position
    pub fn set_take_profit(&mut self, price: f64, quantity: Option<f64>) -> Result<(), String> {
        if self.quantity <= 0.0 {
            return Err("Cannot set take profit on flat position".to_string());
        }
        
        // Validate take profit price
        match self.direction {
            PositionDirection::Long => {
                if price <= self.avg_entry_price {
                    return Err("Take profit price must be above entry price for long positions".to_string());
                }
            },
            PositionDirection::Short => {
                if price >= self.avg_entry_price {
                    return Err("Take profit price must be below entry price for short positions".to_string());
                }
            },
            PositionDirection::Flat => {
                return Err("Cannot set take profit on flat position".to_string());
            },
        }
        
        // Validate quantity
        if let Some(qty) = quantity {
            if qty > self.quantity {
                return Err(format!("Take profit quantity ({}) exceeds position size ({})", qty, self.quantity));
            }
        }
        
        self.take_profit = Some(PositionExit {
            price,
            quantity,
            order_id: None,
            triggered: false,
        });
        
        self.updated_at = Utc::now();
        Ok(())
    }
    
    /// Calculate unrealized PnL
    pub fn unrealized_pnl(&self) -> Option<f64> {
        if self.quantity <= 0.0 || self.mark_price.is_none() {
            return None;
        }
        
        let current_price = self.mark_price.unwrap();
        let price_diff = if self.direction == PositionDirection::Long {
            current_price - self.avg_entry_price
        } else {
            self.avg_entry_price - current_price
        };
        
        Some(self.quantity * price_diff)
    }
    
    /// Calculate total PnL (realized + unrealized)
    pub fn total_pnl(&self) -> Option<f64> {
        self.unrealized_pnl().map(|upnl| upnl + self.realized_pnl)
    }
    
    /// Calculate return on investment (ROI)
    pub fn roi(&self) -> Option<f64> {
        self.total_pnl().map(|pnl| {
            let investment = self.quantity * self.avg_entry_price;
            if investment > 0.0 {
                pnl / investment
            } else {
                0.0
            }
        })
    }
    
    /// Calculate position notional value
    pub fn notional_value(&self) -> Option<f64> {
        if self.quantity <= 0.0 || self.mark_price.is_none() {
            return None;
        }
        
        Some(self.quantity * self.mark_price.unwrap())
    }
    
    /// Calculate average holding time
    pub fn avg_holding_time(&self) -> Option<chrono::Duration> {
        if self.trades.is_empty() {
            return None;
        }
        
        let first_trade_time = self.trades.first().unwrap().timestamp;
        let last_time = self.closed_at.unwrap_or(Utc::now());
        
        Some(last_time - first_trade_time)
    }
    
    /// Check if position is currently open
    pub fn is_open(&self) -> bool {
        self.quantity > 0.0 && self.direction != PositionDirection::Flat
    }
    
    /// Get position risk to reward ratio
    pub fn risk_reward_ratio(&self) -> Option<f64> {
        match (self.stop_loss, self.take_profit) {
            (Some(sl), Some(tp)) => {
                let risk = if self.direction == PositionDirection::Long {
                    self.avg_entry_price - sl.price
                } else {
                    sl.price - self.avg_entry_price
                };
                
                let reward = if self.direction == PositionDirection::Long {
                    tp.price - self.avg_entry_price
                } else {
                    self.avg_entry_price - tp.price
                };
                
                if risk > 0.0 {
                    Some(reward / risk)
                } else {
                    None
                }
            },
            _ => None,
        }
    }
}
