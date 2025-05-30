use prometheus::{register_counter_vec, register_gauge_vec, register_histogram_vec};
use prometheus::{CounterVec, GaugeVec, HistogramVec};
use lazy_static::lazy_static;
use std::thread;
use std::time::Duration;
use rand::Rng;

lazy_static! {
    pub static ref ORDER_EXECUTION_LATENCY: HistogramVec = register_histogram_vec!(
        "order_execution_latency_seconds",
        "Time taken from strategy signal to order submission",
        &["trading_pair", "order_type", "side"],
        vec![0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    ).unwrap();

    pub static ref MARKET_DATA_THROUGHPUT: CounterVec = register_counter_vec!(
        "market_data_updates_total",
        "Number of market data updates processed",
        &["trading_pair", "update_type"]
    ).unwrap();

    pub static ref TRADING_BALANCE: GaugeVec = register_gauge_vec!(
        "trading_balance",
        "Current balance in paper trading account",
        &["currency"]
    ).unwrap();

    pub static ref SIGNAL_GENERATION_TIME: HistogramVec = register_histogram_vec!(
        "signal_generation_time_seconds",
        "Time taken to generate trading signals",
        &["strategy", "trading_pair"],
        vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
    ).unwrap();

    pub static ref API_RESPONSE_TIME: HistogramVec = register_histogram_vec!(
        "api_response_time_seconds",
        "Response time for API requests",
        &["endpoint", "method"],
        vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
    ).unwrap();

    pub static ref CPU_USAGE: GaugeVec = register_gauge_vec!(
        "cpu_usage_percent",
        "CPU usage percentage",
        &["process"]
    ).unwrap();

    pub static ref MEMORY_USAGE: GaugeVec = register_gauge_vec!(
        "memory_usage_bytes",
        "Memory usage in bytes",
        &["process"]
    ).unwrap();
}

pub fn start_metrics_simulation() {
    thread::spawn(|| {
        let mut rng = rand::thread_rng();
        let trading_pairs = vec!["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT"];
        let update_types = vec!["trade", "orderbook", "ticker"];
        let order_types = vec!["market", "limit"];
        let sides = vec!["buy", "sell"];
        let strategies = vec!["momentum", "mean_reversion", "breakout"];
        let endpoints = vec!["/api/market", "/api/orders", "/api/account"];
        let methods = vec!["GET", "POST"];
        
        let mut btc_balance = 1.0;
        let mut usdt_balance = 10000.0;
        
        loop {
            // Update market data throughput
            for &pair in trading_pairs.iter() {
                for &update_type in update_types.iter() {
                    let count = rng.gen_range(1..10);
                    MARKET_DATA_THROUGHPUT
                        .with_label_values(&[pair, update_type])
                        .inc_by(count as f64);
                }
            }
            
            // Simulate order execution latency
            for &pair in trading_pairs.iter() {
                for &order_type in order_types.iter() {
                    for &side in sides.iter() {
                        let latency = rng.gen_range(0.01..0.5);
                        ORDER_EXECUTION_LATENCY
                            .with_label_values(&[pair, order_type, side])
                            .observe(latency);
                    }
                }
            }
            
            // Update trading balances
            let btc_change = rng.gen_range(-0.01..0.01);
            let usdt_change = rng.gen_range(-50.0..50.0);
            btc_balance += btc_change;
            usdt_balance += usdt_change;
            
            TRADING_BALANCE
                .with_label_values(&["BTC"])
                .set(btc_balance);
            TRADING_BALANCE
                .with_label_values(&["USDT"])
                .set(usdt_balance);
            
            // Signal generation time
            for &strategy in strategies.iter() {
                for &pair in trading_pairs.iter() {
                    let time = rng.gen_range(0.005..0.2);
                    SIGNAL_GENERATION_TIME
                        .with_label_values(&[strategy, pair])
                        .observe(time);
                }
            }
            
            // API response time
            for &endpoint in endpoints.iter() {
                for &method in methods.iter() {
                    let time = rng.gen_range(0.002..0.1);
                    API_RESPONSE_TIME
                        .with_label_values(&[endpoint, method])
                        .observe(time);
                }
            }
            
            // System metrics
            CPU_USAGE
                .with_label_values(&["market-data-processor"])
                .set(rng.gen_range(10.0..40.0));
            
            MEMORY_USAGE
                .with_label_values(&["market-data-processor"])
                .set(rng.gen_range(100_000_000.0..500_000_000.0));
            
            thread::sleep(Duration::from_secs(1));
        }
    });
}
