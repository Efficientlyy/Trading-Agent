use prometheus::{
    register_counter, register_gauge, register_histogram,
    Counter, Gauge, Histogram, HistogramOpts, Opts,
};
use lazy_static::lazy_static;

lazy_static! {
    // Market data metrics
    pub static ref MARKET_DATA_UPDATES: Counter = register_counter!(
        Opts::new(
            "market_data_updates_total",
            "Total number of market data updates received"
        )
        .namespace("mexc_trading")
    )
    .unwrap();
    
    pub static ref TICKER_UPDATES: Counter = register_counter!(
        Opts::new(
            "ticker_updates_total",
            "Total number of ticker updates received"
        )
        .namespace("mexc_trading")
    )
    .unwrap();
    
    pub static ref ORDERBOOK_UPDATES: Counter = register_counter!(
        Opts::new(
            "orderbook_updates_total",
            "Total number of order book updates received"
        )
        .namespace("mexc_trading")
    )
    .unwrap();
    
    pub static ref TRADE_UPDATES: Counter = register_counter!(
        Opts::new(
            "trade_updates_total",
            "Total number of trade updates received"
        )
        .namespace("mexc_trading")
    )
    .unwrap();
    
    // WebSocket metrics
    pub static ref WEBSOCKET_RECONNECTS: Counter = register_counter!(
        Opts::new(
            "websocket_reconnects_total",
            "Total number of WebSocket reconnection attempts"
        )
        .namespace("mexc_trading")
    )
    .unwrap();
    
    pub static ref WEBSOCKET_MESSAGES: Counter = register_counter!(
        Opts::new(
            "websocket_messages_total",
            "Total number of WebSocket messages received"
        )
        .namespace("mexc_trading")
    )
    .unwrap();
    
    pub static ref WEBSOCKET_ERRORS: Counter = register_counter!(
        Opts::new(
            "websocket_errors_total",
            "Total number of WebSocket errors"
        )
        .namespace("mexc_trading")
    )
    .unwrap();
    
    // API metrics
    pub static ref API_REQUESTS: Counter = register_counter!(
        Opts::new(
            "api_requests_total",
            "Total number of API requests made"
        )
        .namespace("mexc_trading")
    )
    .unwrap();
    
    pub static ref API_ERRORS: Counter = register_counter!(
        Opts::new(
            "api_errors_total",
            "Total number of API errors"
        )
        .namespace("mexc_trading")
    )
    .unwrap();
    
    pub static ref API_LATENCY: Histogram = register_histogram!(
        HistogramOpts::new(
            "api_request_duration_seconds",
            "API request latency in seconds"
        )
        .namespace("mexc_trading")
        .buckets(vec![0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0])
    )
    .unwrap();
    
    // Market data values
    pub static ref LATEST_PRICE: Gauge = register_gauge!(
        Opts::new(
            "latest_price",
            "Latest price for the trading pair"
        )
        .namespace("mexc_trading")
    )
    .unwrap();
    
    pub static ref BID_ASK_SPREAD: Gauge = register_gauge!(
        Opts::new(
            "bid_ask_spread",
            "Current bid-ask spread for the trading pair"
        )
        .namespace("mexc_trading")
    )
    .unwrap();
    
    pub static ref ORDER_BOOK_DEPTH: Gauge = register_gauge!(
        Opts::new(
            "order_book_depth",
            "Current order book depth (number of price levels)"
        )
        .namespace("mexc_trading")
    )
    .unwrap();
    
    // Paper trading metrics
    pub static ref PAPER_TRADING_BALANCE: Gauge = register_gauge!(
        Opts::new(
            "paper_trading_balance",
            "Current paper trading account balance in USDC"
        )
        .namespace("mexc_trading")
    )
    .unwrap();
    
    pub static ref PAPER_TRADING_BTC_BALANCE: Gauge = register_gauge!(
        Opts::new(
            "paper_trading_btc_balance",
            "Current paper trading account BTC balance"
        )
        .namespace("mexc_trading")
    )
    .unwrap();
    
    pub static ref PAPER_TRADING_PNL: Gauge = register_gauge!(
        Opts::new(
            "paper_trading_pnl",
            "Current paper trading profit/loss in USDC"
        )
        .namespace("mexc_trading")
    )
    .unwrap();
    
    pub static ref PAPER_TRADING_TRADES: Counter = register_counter!(
        Opts::new(
            "paper_trading_trades_total",
            "Total number of paper trades executed"
        )
        .namespace("mexc_trading")
    )
    .unwrap();
}

/// Update ticker metrics
pub fn update_ticker_metrics(symbol: &str, price: f64) {
    TICKER_UPDATES.inc();
    MARKET_DATA_UPDATES.inc();
    LATEST_PRICE.set(price);
}

/// Update order book metrics
pub fn update_orderbook_metrics(symbol: &str, bids: &[(f64, f64)], asks: &[(f64, f64)]) {
    ORDERBOOK_UPDATES.inc();
    MARKET_DATA_UPDATES.inc();
    ORDER_BOOK_DEPTH.set((bids.len() + asks.len()) as f64);
    
    if let (Some((bid_price, _)), Some((ask_price, _))) = (bids.first(), asks.first()) {
        BID_ASK_SPREAD.set(ask_price - bid_price);
    }
}

/// Update trade metrics
pub fn update_trade_metrics(symbol: &str) {
    TRADE_UPDATES.inc();
    MARKET_DATA_UPDATES.inc();
}

/// Update paper trading metrics
pub fn update_paper_trading_metrics(usdc_balance: f64, btc_balance: f64, pnl: f64) {
    PAPER_TRADING_BALANCE.set(usdc_balance);
    PAPER_TRADING_BTC_BALANCE.set(btc_balance);
    PAPER_TRADING_PNL.set(pnl);
}

/// Record a paper trade
pub fn record_paper_trade() {
    PAPER_TRADING_TRADES.inc();
}

/// Record API request with latency
pub fn record_api_request(duration_secs: f64) {
    API_REQUESTS.inc();
    API_LATENCY.observe(duration_secs);
}

/// Record API error
pub fn record_api_error() {
    API_ERRORS.inc();
}

/// Record WebSocket message
pub fn record_websocket_message() {
    WEBSOCKET_MESSAGES.inc();
}

/// Record WebSocket reconnect
pub fn record_websocket_reconnect() {
    WEBSOCKET_RECONNECTS.inc();
}

/// Record WebSocket error
pub fn record_websocket_error() {
    WEBSOCKET_ERRORS.inc();
}
