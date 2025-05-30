use serde::{Deserialize, Serialize};

/// Represents a trading position
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    /// Trading pair symbol (e.g., "BTCUSDC")
    pub symbol: String,
    
    /// Position size (positive for long, negative for short)
    pub size: f64,
    
    /// Average entry price
    pub entry_price: f64,
    
    /// Current market price
    pub current_price: f64,
    
    /// Unrealized profit/loss
    pub unrealized_pnl: f64,
    
    /// Realized profit/loss
    pub realized_pnl: f64,
    
    /// Timestamp when the position was opened
    pub open_timestamp: u64,
    
    /// Timestamp of the last update
    pub update_timestamp: u64,
}

impl Position {
    /// Create a new position instance
    pub fn new(
        symbol: String,
        size: f64,
        entry_price: f64,
        current_price: f64,
        open_timestamp: u64,
    ) -> Self {
        let unrealized_pnl = if size > 0.0 {
            // Long position
            size * (current_price - entry_price)
        } else {
            // Short position
            size * (entry_price - current_price)
        };
        
        Self {
            symbol,
            size,
            entry_price,
            current_price,
            unrealized_pnl,
            realized_pnl: 0.0,
            open_timestamp,
            update_timestamp: open_timestamp,
        }
    }
    
    /// Update the position with a new market price
    pub fn update_price(&mut self, current_price: f64, timestamp: u64) {
        self.current_price = current_price;
        self.update_timestamp = timestamp;
        
        // Recalculate unrealized P&L
        self.unrealized_pnl = if self.size > 0.0 {
            // Long position
            self.size * (current_price - self.entry_price)
        } else {
            // Short position
            self.size * (self.entry_price - current_price)
        };
    }
    
    /// Add to the position
    pub fn add(&mut self, size: f64, price: f64, timestamp: u64) {
        if (self.size > 0.0 && size > 0.0) || (self.size < 0.0 && size < 0.0) {
            // Adding to existing position (same direction)
            let new_size = self.size + size;
            let new_entry_price = ((self.size * self.entry_price) + (size * price)) / new_size;
            
            self.size = new_size;
            self.entry_price = new_entry_price;
        } else {
            // Reducing or flipping position (opposite direction)
            if size.abs() < self.size.abs() {
                // Partial reduction
                let realized_pnl = if self.size > 0.0 {
                    // Long position being reduced
                    size.abs() * (price - self.entry_price)
                } else {
                    // Short position being reduced
                    size.abs() * (self.entry_price - price)
                };
                
                self.realized_pnl += realized_pnl;
                self.size += size; // size is negative when reducing
            } else {
                // Full reduction or flip
                let realized_pnl = if self.size > 0.0 {
                    // Long position being reduced/flipped
                    self.size * (price - self.entry_price)
                } else {
                    // Short position being reduced/flipped
                    self.size.abs() * (self.entry_price - price)
                };
                
                self.realized_pnl += realized_pnl;
                
                if size.abs() > self.size.abs() {
                    // Position flipped
                    let remaining_size = size + self.size; // self.size is opposite sign
                    self.size = remaining_size;
                    self.entry_price = price;
                    self.open_timestamp = timestamp;
                } else {
                    // Position fully closed
                    self.size = 0.0;
                }
            }
        }
        
        self.update_timestamp = timestamp;
        
        // Recalculate unrealized P&L
        if self.size != 0.0 {
            self.unrealized_pnl = if self.size > 0.0 {
                // Long position
                self.size * (self.current_price - self.entry_price)
            } else {
                // Short position
                self.size * (self.entry_price - self.current_price)
            };
        } else {
            self.unrealized_pnl = 0.0;
        }
    }
    
    /// Close the position
    pub fn close(&mut self, price: f64, timestamp: u64) -> f64 {
        let realized_pnl = if self.size > 0.0 {
            // Long position
            self.size * (price - self.entry_price)
        } else {
            // Short position
            self.size.abs() * (self.entry_price - price)
        };
        
        self.realized_pnl += realized_pnl;
        self.size = 0.0;
        self.unrealized_pnl = 0.0;
        self.update_timestamp = timestamp;
        
        realized_pnl
    }
    
    /// Get the position side (long, short, or flat)
    pub fn side(&self) -> &str {
        if self.size > 0.0 {
            "long"
        } else if self.size < 0.0 {
            "short"
        } else {
            "flat"
        }
    }
    
    /// Get the position value (size * current_price)
    pub fn value(&self) -> f64 {
        self.size.abs() * self.current_price
    }
    
    /// Get the position duration in milliseconds
    pub fn duration_ms(&self) -> u64 {
        self.update_timestamp - self.open_timestamp
    }
    
    /// Get the position P&L percentage
    pub fn pnl_percentage(&self) -> f64 {
        if self.entry_price == 0.0 {
            0.0
        } else {
            (self.unrealized_pnl / (self.size.abs() * self.entry_price)) * 100.0
        }
    }
    
    /// Convert position to JSON string
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
    
    /// Create position from JSON string
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_position_long() {
        let mut position = Position::new(
            "BTCUSDC".to_string(),
            0.1,
            35000.0,
            35000.0,
            1621500000000,
        );
        
        // Update price
        position.update_price(36000.0, 1621500100000);
        assert_eq!(position.unrealized_pnl, 100.0); // 0.1 * (36000 - 35000)
        
        // Add to position
        position.add(0.1, 36000.0, 1621500200000);
        assert_eq!(position.size, 0.2);
        assert_eq!(position.entry_price, 35500.0); // (0.1 * 35000 + 0.1 * 36000) / 0.2
        
        // Close position
        let realized_pnl = position.close(37000.0, 1621500300000);
        assert_eq!(realized_pnl, 300.0); // 0.2 * (37000 - 35500)
        assert_eq!(position.size, 0.0);
        assert_eq!(position.unrealized_pnl, 0.0);
        assert_eq!(position.realized_pnl, 300.0);
    }
    
    #[test]
    fn test_position_short() {
        let mut position = Position::new(
            "BTCUSDC".to_string(),
            -0.1,
            35000.0,
            35000.0,
            1621500000000,
        );
        
        // Update price
        position.update_price(34000.0, 1621500100000);
        assert_eq!(position.unrealized_pnl, 100.0); // 0.1 * (35000 - 34000)
        
        // Add to position
        position.add(-0.1, 34000.0, 1621500200000);
        assert_eq!(position.size, -0.2);
        assert_eq!(position.entry_price, 34500.0); // (0.1 * 35000 + 0.1 * 34000) / 0.2
        
        // Close position
        let realized_pnl = position.close(33000.0, 1621500300000);
        assert_eq!(realized_pnl, 300.0); // 0.2 * (34500 - 33000)
        assert_eq!(position.size, 0.0);
        assert_eq!(position.unrealized_pnl, 0.0);
        assert_eq!(position.realized_pnl, 300.0);
    }
}
