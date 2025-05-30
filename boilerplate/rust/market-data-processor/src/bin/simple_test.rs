use std::collections::BTreeMap;

// A simple representation of an order book to test basic functionality
#[derive(Debug, Clone)]
struct SimpleOrderBook {
    symbol: String,
    bids: BTreeMap<f64, f64>,  // price -> quantity
    asks: BTreeMap<f64, f64>,  // price -> quantity
}

impl SimpleOrderBook {
    fn new(symbol: &str) -> Self {
        Self {
            symbol: symbol.to_string(),
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
        }
    }

    fn update_bid(&mut self, price: f64, quantity: f64) {
        if quantity > 0.0 {
            self.bids.insert(price, quantity);
        } else {
            self.bids.remove(&price);
        }
    }

    fn update_ask(&mut self, price: f64, quantity: f64) {
        if quantity > 0.0 {
            self.asks.insert(price, quantity);
        } else {
            self.asks.remove(&price);
        }
    }

    fn best_bid(&self) -> Option<(f64, f64)> {
        self.bids.iter().next_back().map(|(k, v)| (*k, *v))
    }

    fn best_ask(&self) -> Option<(f64, f64)> {
        self.asks.iter().next().map(|(k, v)| (*k, *v))
    }

    fn spread(&self) -> Option<f64> {
        match (self.best_ask(), self.best_bid()) {
            (Some((ask, _)), Some((bid, _))) => Some(ask - bid),
            _ => None,
        }
    }
}

fn main() {
    println!("Running simple order book test");

    // Create a simple order book
    let mut book = SimpleOrderBook::new("BTCUSDT");

    // Add some bids and asks
    book.update_bid(39000.0, 1.5);
    book.update_bid(38900.0, 2.0);
    book.update_ask(39100.0, 1.0);
    book.update_ask(39200.0, 2.5);

    // Display the order book
    println!("Order book for {}", book.symbol);
    println!("Bids:");
    for (price, qty) in book.bids.iter().rev() {
        println!("  {} @ {}", qty, price);
    }

    println!("Asks:");
    for (price, qty) in book.asks.iter() {
        println!("  {} @ {}", qty, price);
    }

    // Display best bid/ask and spread
    if let Some((price, qty)) = book.best_bid() {
        println!("Best bid: {} @ {}", qty, price);
    }

    if let Some((price, qty)) = book.best_ask() {
        println!("Best ask: {} @ {}", qty, price);
    }

    if let Some(spread) = book.spread() {
        println!("Spread: {}", spread);
    }

    println!("Simple test completed successfully");
}
