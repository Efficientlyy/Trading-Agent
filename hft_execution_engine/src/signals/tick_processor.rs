//! Tick-by-tick data processing for high-frequency trading
//! 
//! This module provides efficient processing of tick-level market data
//! for ultra-fast signal generation.

use std::collections::VecDeque;
use std::time::{SystemTime, UNIX_EPOCH};
use serde::{Serialize, Deserialize};
use crate::orderbook::types::{OrderBook, OrderBookUpdate};
use super::DEFAULT_LOOKBACK_WINDOW;

/// Tick-level market data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickData {
    /// Symbol this tick is for
    pub symbol: String,
    /// Current price
    pub price: f64,
    /// Volume at this price
    pub volume: f64,
    /// Bid price at time of tick
    pub bid: f64,
    /// Ask price at time of tick
    pub ask: f64,
    /// Microsecond-precision timestamp
    pub timestamp_us: u64,
    /// Trade direction if available (true = buy, false = sell, None = unknown)
    pub is_buy: Option<bool>,
}

impl TickData {
    /// Create a new tick data point
    pub fn new(symbol: &str, price: f64, volume: f64, bid: f64, ask: f64) -> Self {
        let timestamp_us = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;
            
        Self {
            symbol: symbol.to_string(),
            price,
            volume,
            bid,
            ask,
            timestamp_us,
            is_buy: None,
        }
    }
    
    /// Create a new tick data with trade direction
    pub fn with_direction(symbol: &str, price: f64, volume: f64, bid: f64, ask: f64, is_buy: bool) -> Self {
        let mut tick = Self::new(symbol, price, volume, bid, ask);
        tick.is_buy = Some(is_buy);
        tick
    }
    
    /// Calculate the mid price
    pub fn mid_price(&self) -> f64 {
        (self.bid + self.ask) / 2.0
    }
    
    /// Calculate the spread
    pub fn spread(&self) -> f64 {
        self.ask - self.bid
    }
    
    /// Calculate the spread as a percentage of the mid price
    pub fn spread_pct(&self) -> f64 {
        let mid = self.mid_price();
        if mid > 0.0 {
            (self.ask - self.bid) / mid * 100.0
        } else {
            0.0
        }
    }
    
    /// Determine if this tick is likely aggressive (crossed the spread)
    pub fn is_aggressive(&self) -> bool {
        match self.is_buy {
            Some(true) => self.price >= self.ask,
            Some(false) => self.price <= self.bid,
            None => false,
        }
    }
}

/// Processor for tick-by-tick data analysis
#[derive(Debug)]
pub struct TickProcessor {
    /// Symbol this processor is for
    symbol: String,
    /// Window of recent ticks
    ticks: VecDeque<TickData>,
    /// Maximum window size
    window_size: usize,
    /// Last processed timestamp
    last_timestamp_us: u64,
    /// Current price
    current_price: f64,
    /// Current bid
    current_bid: f64,
    /// Current ask
    current_ask: f64,
    /// Cumulative volume in current window
    cumulative_volume: f64,
    /// Buy volume in current window
    buy_volume: f64,
    /// Sell volume in current window
    sell_volume: f64,
}

impl TickProcessor {
    /// Create a new tick processor for the given symbol
    pub fn new(symbol: &str) -> Self {
        Self {
            symbol: symbol.to_string(),
            ticks: VecDeque::with_capacity(DEFAULT_LOOKBACK_WINDOW),
            window_size: DEFAULT_LOOKBACK_WINDOW,
            last_timestamp_us: 0,
            current_price: 0.0,
            current_bid: 0.0,
            current_ask: 0.0,
            cumulative_volume: 0.0,
            buy_volume: 0.0,
            sell_volume: 0.0,
        }
    }
    
    /// Create a new tick processor with custom window size
    pub fn with_window_size(symbol: &str, window_size: usize) -> Self {
        let mut processor = Self::new(symbol);
        processor.window_size = window_size;
        processor.ticks = VecDeque::with_capacity(window_size);
        processor
    }
    
    /// Process a new tick
    pub fn process_tick(&mut self, tick: TickData) {
        // Update current state
        self.current_price = tick.price;
        self.current_bid = tick.bid;
        self.current_ask = tick.ask;
        self.last_timestamp_us = tick.timestamp_us;
        
        // Update volume statistics
        self.cumulative_volume += tick.volume;
        
        if let Some(is_buy) = tick.is_buy {
            if is_buy {
                self.buy_volume += tick.volume;
            } else {
                self.sell_volume += tick.volume;
            }
        }
        
        // Add to window and maintain size
        self.ticks.push_back(tick);
        
        if self.ticks.len() > self.window_size {
            if let Some(removed_tick) = self.ticks.pop_front() {
                self.cumulative_volume -= removed_tick.volume;
                
                if let Some(is_buy) = removed_tick.is_buy {
                    if is_buy {
                        self.buy_volume -= removed_tick.volume;
                    } else {
                        self.sell_volume -= removed_tick.volume;
                    }
                }
            }
        }
    }
    
    /// Get the current price
    pub fn price(&self) -> f64 {
        self.current_price
    }
    
    /// Get the current bid price
    pub fn bid(&self) -> f64 {
        self.current_bid
    }
    
    /// Get the current ask price
    pub fn ask(&self) -> f64 {
        self.current_ask
    }
    
    /// Get the current mid price
    pub fn mid_price(&self) -> f64 {
        (self.current_bid + self.current_ask) / 2.0
    }
    
    /// Get the current spread
    pub fn spread(&self) -> f64 {
        self.current_ask - self.current_bid
    }
    
    /// Get the current spread as a percentage of the mid price
    pub fn spread_pct(&self) -> f64 {
        let mid = self.mid_price();
        if mid > 0.0 {
            (self.current_ask - self.current_bid) / mid * 100.0
        } else {
            0.0
        }
    }
    
    /// Get the buy/sell volume ratio
    pub fn buy_sell_ratio(&self) -> f64 {
        if self.sell_volume > 0.0 {
            self.buy_volume / self.sell_volume
        } else {
            1.0
        }
    }
    
    /// Get the price change over the window
    pub fn price_change(&self) -> f64 {
        if let (Some(first), Some(last)) = (self.ticks.front(), self.ticks.back()) {
            last.price - first.price
        } else {
            0.0
        }
    }
    
    /// Get the price change as a percentage
    pub fn price_change_pct(&self) -> f64 {
        if let (Some(first), Some(last)) = (self.ticks.front(), self.ticks.back()) {
            if first.price > 0.0 {
                (last.price - first.price) / first.price * 100.0
            } else {
                0.0
            }
        } else {
            0.0
        }
    }
    
    /// Get the price velocity (change per microsecond)
    pub fn price_velocity(&self) -> f64 {
        if let (Some(first), Some(last)) = (self.ticks.front(), self.ticks.back()) {
            let time_diff = last.timestamp_us - first.timestamp_us;
            if time_diff > 0 {
                (last.price - first.price) / (time_diff as f64)
            } else {
                0.0
            }
        } else {
            0.0
        }
    }
    
    /// Get the tick window
    pub fn ticks(&self) -> &VecDeque<TickData> {
        &self.ticks
    }
    
    /// Get the window size
    pub fn window_size(&self) -> usize {
        self.window_size
    }
    
    /// Get the current window fill percentage
    pub fn window_fill_pct(&self) -> f64 {
        self.ticks.len() as f64 / self.window_size as f64 * 100.0
    }
    
    /// Check if the window is fully populated
    pub fn is_window_full(&self) -> bool {
        self.ticks.len() >= self.window_size
    }
    
    /// Clear the tick window
    pub fn clear(&mut self) {
        self.ticks.clear();
        self.cumulative_volume = 0.0;
        self.buy_volume = 0.0;
        self.sell_volume = 0.0;
    }
}
