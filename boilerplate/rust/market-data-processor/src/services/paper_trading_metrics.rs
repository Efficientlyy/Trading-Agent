use crate::metrics::{
    self,
    update_market_price,
    update_paper_trading_balance,
    update_pnl,
    record_order,
    OrderHistoryEntry,
    OPEN_ORDERS_COUNT,
    ORDERS_EXECUTED_TOTAL,
    SIGNALS_GENERATED_TOTAL,
    MARKET_PRICE,
    MARKET_VOLUME_24H,
};
use crate::models::order::{Order, OrderSide, OrderStatus, OrderType};
use chrono::Utc;
use std::collections::HashMap;

/// Update all paper trading metrics to ensure dashboard functionality
pub fn update_all_paper_trading_metrics(
    account_balances: &HashMap<String, f64>,
    open_orders: &HashMap<String, Order>,
    positions: &HashMap<String, crate::models::position::Position>,
    market_prices: &HashMap<String, f64>,
    completed_orders: &[Order],
) {
    // Update account balances
    let usdt_balance = *account_balances.get("USDT").unwrap_or(&0.0);
    let btc_balance = *account_balances.get("BTC").unwrap_or(&0.0);
    let eth_balance = *account_balances.get("ETH").unwrap_or(&0.0);
    
    update_paper_trading_balance(Some(usdt_balance), Some(btc_balance), Some(eth_balance));
    
    // Update open orders count by trading pair
    let mut orders_by_pair: HashMap<String, i32> = HashMap::new();
    for order in open_orders.values() {
        *orders_by_pair.entry(order.symbol.clone()).or_insert(0) += 1;
    }
    
    // Reset all counts first to ensure pairs with zero orders show correctly
    for pair in ["BTCUSDT", "ETHUSDT"].iter() {
        OPEN_ORDERS_COUNT.with_label_values(&[pair]).set(0.0);
    }
    
    // Then set the actual counts
    for (pair, count) in orders_by_pair.iter() {
        OPEN_ORDERS_COUNT.with_label_values(&[pair]).set(*count as f64);
    }
    
    // Update market prices
    for (symbol, price) in market_prices.iter() {
        update_market_price(symbol, *price);
    }
    
    // Calculate P&L for each position
    let mut realized_pnl = 0.0;
    let mut unrealized_pnl = 0.0;
    
    for position in positions.values() {
        realized_pnl += position.realized_pnl;
        
        if position.is_open() {
            if let Some(market_price) = market_prices.get(&position.symbol) {
                let direction_multiplier = match position.direction {
                    crate::models::position::PositionDirection::Long => 1.0,
                    crate::models::position::PositionDirection::Short => -1.0,
                };
                
                let entry_value = position.entry_price * position.quantity;
                let current_value = market_price * position.quantity;
                unrealized_pnl += (current_value - entry_value) * direction_multiplier;
            }
        }
    }
    
    // Update P&L metrics
    update_pnl(Some(realized_pnl), Some(unrealized_pnl));
    
    // Record the most recent orders for the order history table
    if !completed_orders.is_empty() {
        let latest_orders = if completed_orders.len() > 10 {
            &completed_orders[completed_orders.len() - 10..]
        } else {
            completed_orders
        };
        
        for order in latest_orders {
            if order.status == OrderStatus::Filled || order.status == OrderStatus::PartiallyFilled {
                let side_str = match order.side {
                    OrderSide::Buy => "buy",
                    OrderSide::Sell => "sell",
                };
                
                let order_entry = OrderHistoryEntry {
                    id: order.id.clone(),
                    time: Utc::now().timestamp(),
                    trading_pair: order.symbol.clone(),
                    side: side_str.to_string(),
                    price: order.avg_fill_price.unwrap_or(order.price.unwrap_or(0.0)),
                    quantity: order.filled_quantity,
                    status: order.status.to_string(),
                    value: order.avg_fill_price.unwrap_or(order.price.unwrap_or(0.0)) * order.filled_quantity,
                };
                
                record_order(order_entry);
            }
        }
    }
}

/// Record a trading signal in metrics for dashboard display
pub fn record_trading_signal(strategy: &str, trading_pair: &str, signal_type: &str) {
    SIGNALS_GENERATED_TOTAL
        .with_label_values(&[strategy, trading_pair, signal_type])
        .inc();
}

/// Record an executed order in metrics for dashboard display
pub fn record_executed_order(trading_pair: &str, order_type: &str, side: &str, status: &str) {
    ORDERS_EXECUTED_TOTAL
        .with_label_values(&[trading_pair, order_type, side, status])
        .inc();
}

/// Update market volume metrics
pub fn update_market_volume(trading_pair: &str, volume_24h: f64) {
    MARKET_VOLUME_24H.with_label_values(&[trading_pair]).set(volume_24h);
}

/// Initialize all metrics with zero values to ensure dashboard displays properly
pub fn initialize_metrics(trading_pairs: &[String], initial_balances: &HashMap<String, f64>) {
    // Initialize paper trading balances
    let usdt_balance = *initial_balances.get("USDT").unwrap_or(&10000.0);
    let btc_balance = *initial_balances.get("BTC").unwrap_or(&1.0);
    let eth_balance = *initial_balances.get("ETH").unwrap_or(&10.0);
    
    metrics::init_paper_trading_metrics(usdt_balance, btc_balance, eth_balance);
    
    // Initialize metrics for all trading pairs to ensure dashboard displays properly
    for pair in trading_pairs {
        // Zero values for all metrics to start
        OPEN_ORDERS_COUNT.with_label_values(&[pair]).set(0.0);
        MARKET_PRICE.with_label_values(&[pair]).set(0.0);
        MARKET_VOLUME_24H.with_label_values(&[pair]).set(0.0);
    }
    
    // Initialize P&L metrics
    update_pnl(Some(0.0), Some(0.0));
}
