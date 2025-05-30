use std::collections::HashMap;
use chrono::{DateTime, TimeZone, Utc};
use serde::{Deserialize, Serialize};

use market_data_processor::models::signal::SignalType;
use market_data_processor::services::decision_module::DecisionAction;

use crate::test_framework::{MarketCondition, MarketDataSnapshot, TestScenario, TestSignal, ExpectedOutcomes};

/// Load test scenarios for integration testing
pub fn load_test_scenarios() -> Vec<TestScenario> {
    vec![
        trending_market_scenario(),
        ranging_market_scenario(),
        volatile_market_scenario(),
        low_liquidity_scenario(),
        extreme_volatility_scenario(),
    ]
}

/// Generate a scenario for trending market conditions
fn trending_market_scenario() -> TestScenario {
    // Initial balances
    let mut initial_balances = HashMap::new();
    initial_balances.insert("USDT".to_string(), 10000.0);
    initial_balances.insert("BTC".to_string(), 0.0);

    // Signals for a trending market (strong uptrend)
    let signals = vec![
        TestSignal {
            timestamp: Utc.with_ymd_and_hms(2025, 5, 1, 10, 0, 0).unwrap(),
            symbol: "BTCUSDT".to_string(),
            signal_type: SignalType::Buy,
            strength: 0.8,
            source: "RSI".to_string(),
            expected_action: Some(DecisionAction::Buy),
        },
        TestSignal {
            timestamp: Utc.with_ymd_and_hms(2025, 5, 1, 12, 0, 0).unwrap(),
            symbol: "BTCUSDT".to_string(),
            signal_type: SignalType::Buy,
            strength: 0.85,
            source: "MACD".to_string(),
            expected_action: Some(DecisionAction::Buy),
        },
        TestSignal {
            timestamp: Utc.with_ymd_and_hms(2025, 5, 1, 14, 0, 0).unwrap(),
            symbol: "BTCUSDT".to_string(),
            signal_type: SignalType::Hold,
            strength: 0.5,
            source: "Bollinger".to_string(),
            expected_action: Some(DecisionAction::Hold),
        },
        TestSignal {
            timestamp: Utc.with_ymd_and_hms(2025, 5, 1, 16, 0, 0).unwrap(),
            symbol: "BTCUSDT".to_string(),
            signal_type: SignalType::Sell,
            strength: 0.75,
            source: "RSI".to_string(),
            expected_action: Some(DecisionAction::Sell),
        },
    ];

    // Expected outcomes
    let mut expected_balance_ranges = HashMap::new();
    expected_balance_ranges.insert("USDT".to_string(), (9000.0, 11000.0)); // +/- 10% range
    expected_balance_ranges.insert("BTC".to_string(), (0.0, 0.2));         // Should end with no BTC

    let expected_outcomes = ExpectedOutcomes {
        final_balances: expected_balance_ranges,
        order_count: (2, 4),         // Should have at least 2 orders (buy and sell)
        filled_order_count: (2, 4),  // All orders should be filled
        pnl_range: (100.0, 1000.0),  // Should be profitable
        max_drawdown: 5.0,           // Max 5% drawdown allowed
    };

    TestScenario {
        name: "Trending Market Scenario".to_string(),
        description: "Tests system behavior in a strong uptrend market".to_string(),
        market_condition: MarketCondition::Trending,
        initial_balances,
        signals,
        expected_outcomes,
        max_test_duration_seconds: 60,
    }
}

/// Generate a scenario for ranging market conditions
fn ranging_market_scenario() -> TestScenario {
    // Initial balances
    let mut initial_balances = HashMap::new();
    initial_balances.insert("USDT".to_string(), 10000.0);
    initial_balances.insert("BTC".to_string(), 0.1);

    // Signals for a ranging market (oscillating)
    let signals = vec![
        TestSignal {
            timestamp: Utc.with_ymd_and_hms(2025, 5, 1, 10, 0, 0).unwrap(),
            symbol: "BTCUSDT".to_string(),
            signal_type: SignalType::Sell,
            strength: 0.7,
            source: "RSI".to_string(),
            expected_action: Some(DecisionAction::Sell),
        },
        TestSignal {
            timestamp: Utc.with_ymd_and_hms(2025, 5, 1, 12, 0, 0).unwrap(),
            symbol: "BTCUSDT".to_string(),
            signal_type: SignalType::Buy,
            strength: 0.65,
            source: "MACD".to_string(),
            expected_action: Some(DecisionAction::Buy),
        },
        TestSignal {
            timestamp: Utc.with_ymd_and_hms(2025, 5, 1, 14, 0, 0).unwrap(),
            symbol: "BTCUSDT".to_string(),
            signal_type: SignalType::Sell,
            strength: 0.72,
            source: "RSI".to_string(),
            expected_action: Some(DecisionAction::Sell),
        },
        TestSignal {
            timestamp: Utc.with_ymd_and_hms(2025, 5, 1, 16, 0, 0).unwrap(),
            symbol: "BTCUSDT".to_string(),
            signal_type: SignalType::Buy,
            strength: 0.68,
            source: "MACD".to_string(),
            expected_action: Some(DecisionAction::Buy),
        },
    ];

    // Expected outcomes
    let mut expected_balance_ranges = HashMap::new();
    expected_balance_ranges.insert("USDT".to_string(), (9000.0, 11000.0)); // +/- 10% range
    expected_balance_ranges.insert("BTC".to_string(), (0.05, 0.15));       // +/- 50% range

    let expected_outcomes = ExpectedOutcomes {
        final_balances: expected_balance_ranges,
        order_count: (4, 6),         // Should have all 4 orders executed
        filled_order_count: (4, 6),  // All orders should be filled
        pnl_range: (-200.0, 200.0),  // May be slightly profitable or unprofitable due to fees
        max_drawdown: 3.0,           // Max 3% drawdown allowed in ranging market
    };

    TestScenario {
        name: "Ranging Market Scenario".to_string(),
        description: "Tests system behavior in a sideways (ranging) market".to_string(),
        market_condition: MarketCondition::Ranging,
        initial_balances,
        signals,
        expected_outcomes,
        max_test_duration_seconds: 60,
    }
}

/// Generate a scenario for volatile market conditions
fn volatile_market_scenario() -> TestScenario {
    // Initial balances
    let mut initial_balances = HashMap::new();
    initial_balances.insert("USDT".to_string(), 10000.0);
    initial_balances.insert("BTC".to_string(), 0.05);

    // Signals for a volatile market (rapid price changes)
    let signals = vec![
        TestSignal {
            timestamp: Utc.with_ymd_and_hms(2025, 5, 1, 10, 0, 0).unwrap(),
            symbol: "BTCUSDT".to_string(),
            signal_type: SignalType::Sell,
            strength: 0.9,
            source: "RSI".to_string(),
            expected_action: Some(DecisionAction::Sell),
        },
        TestSignal {
            timestamp: Utc.with_ymd_and_hms(2025, 5, 1, 10, 15, 0).unwrap(), // Short time frame
            symbol: "BTCUSDT".to_string(),
            signal_type: SignalType::Buy,
            strength: 0.75,
            source: "MACD".to_string(),
            expected_action: Some(DecisionAction::Buy),
        },
        TestSignal {
            timestamp: Utc.with_ymd_and_hms(2025, 5, 1, 10, 30, 0).unwrap(), // Short time frame
            symbol: "BTCUSDT".to_string(),
            signal_type: SignalType::Sell,
            strength: 0.8,
            source: "RSI".to_string(),
            expected_action: Some(DecisionAction::Sell),
        },
        TestSignal {
            timestamp: Utc.with_ymd_and_hms(2025, 5, 1, 10, 45, 0).unwrap(), // Short time frame
            symbol: "BTCUSDT".to_string(),
            signal_type: SignalType::Buy,
            strength: 0.82,
            source: "MACD".to_string(),
            expected_action: Some(DecisionAction::Buy),
        },
        TestSignal {
            timestamp: Utc.with_ymd_and_hms(2025, 5, 1, 11, 0, 0).unwrap(),
            symbol: "BTCUSDT".to_string(),
            signal_type: SignalType::Sell,
            strength: 0.85,
            source: "RSI".to_string(),
            expected_action: Some(DecisionAction::Sell),
        },
    ];

    // Expected outcomes
    let mut expected_balance_ranges = HashMap::new();
    expected_balance_ranges.insert("USDT".to_string(), (9000.0, 11000.0)); // +/- 10% range
    expected_balance_ranges.insert("BTC".to_string(), (0.0, 0.1));         // May end with some BTC

    let expected_outcomes = ExpectedOutcomes {
        final_balances: expected_balance_ranges,
        order_count: (4, 6),         // Should have multiple orders
        filled_order_count: (4, 6),  // Most orders should be filled
        pnl_range: (-500.0, 500.0),  // Could be profitable or unprofitable due to volatility
        max_drawdown: 7.0,           // Higher drawdown allowed in volatile market
    };

    TestScenario {
        name: "Volatile Market Scenario".to_string(),
        description: "Tests system behavior in a highly volatile market with rapid price changes".to_string(),
        market_condition: MarketCondition::Volatile,
        initial_balances,
        signals,
        expected_outcomes,
        max_test_duration_seconds: 60,
    }
}

/// Generate a scenario for low liquidity market conditions
fn low_liquidity_scenario() -> TestScenario {
    // Initial balances
    let mut initial_balances = HashMap::new();
    initial_balances.insert("USDT".to_string(), 10000.0);
    initial_balances.insert("BTC".to_string(), 0.0);

    // Signals for a low liquidity market
    let signals = vec![
        TestSignal {
            timestamp: Utc.with_ymd_and_hms(2025, 5, 1, 10, 0, 0).unwrap(),
            symbol: "BTCUSDT".to_string(),
            signal_type: SignalType::Buy,
            strength: 0.75,
            source: "RSI".to_string(),
            expected_action: Some(DecisionAction::Buy),
        },
        TestSignal {
            timestamp: Utc.with_ymd_and_hms(2025, 5, 1, 12, 0, 0).unwrap(),
            symbol: "BTCUSDT".to_string(),
            signal_type: SignalType::Sell,
            strength: 0.8,
            source: "MACD".to_string(),
            expected_action: Some(DecisionAction::Sell),
        },
    ];

    // Expected outcomes
    let mut expected_balance_ranges = HashMap::new();
    expected_balance_ranges.insert("USDT".to_string(), (9000.0, 11000.0)); // +/- 10% range
    expected_balance_ranges.insert("BTC".to_string(), (0.0, 0.05));        // May end with small BTC amount due to partial fills

    let expected_outcomes = ExpectedOutcomes {
        final_balances: expected_balance_ranges,
        order_count: (2, 3),         // Should have at least 2 orders
        filled_order_count: (1, 3),  // May have partial fills due to low liquidity
        pnl_range: (-300.0, 300.0),  // Higher slippage expected
        max_drawdown: 6.0,           // Higher drawdown allowed in low liquidity market
    };

    TestScenario {
        name: "Low Liquidity Scenario".to_string(),
        description: "Tests system behavior in a market with low liquidity and higher slippage".to_string(),
        market_condition: MarketCondition::LowLiquidity,
        initial_balances,
        signals,
        expected_outcomes,
        max_test_duration_seconds: 60,
    }
}

/// Generate a scenario for extreme market volatility (stress test)
fn extreme_volatility_scenario() -> TestScenario {
    // Initial balances
    let mut initial_balances = HashMap::new();
    initial_balances.insert("USDT".to_string(), 10000.0);
    initial_balances.insert("BTC".to_string(), 0.1);

    // Signals for extreme volatility (rapid, large price movements)
    let signals = vec![
        TestSignal {
            timestamp: Utc.with_ymd_and_hms(2025, 5, 1, 10, 0, 0).unwrap(),
            symbol: "BTCUSDT".to_string(),
            signal_type: SignalType::Sell,
            strength: 0.95,
            source: "RSI".to_string(),
            expected_action: Some(DecisionAction::Sell),
        },
        TestSignal {
            timestamp: Utc.with_ymd_and_hms(2025, 5, 1, 10, 5, 0).unwrap(), // Very short time frame
            symbol: "BTCUSDT".to_string(),
            signal_type: SignalType::Buy,
            strength: 0.90,
            source: "MACD".to_string(),
            expected_action: Some(DecisionAction::Buy),
        },
        TestSignal {
            timestamp: Utc.with_ymd_and_hms(2025, 5, 1, 10, 10, 0).unwrap(), // Very short time frame
            symbol: "BTCUSDT".to_string(),
            signal_type: SignalType::Sell,
            strength: 0.88,
            source: "RSI".to_string(),
            expected_action: Some(DecisionAction::Sell),
        },
        TestSignal {
            timestamp: Utc.with_ymd_and_hms(2025, 5, 1, 10, 15, 0).unwrap(), // Very short time frame
            symbol: "BTCUSDT".to_string(),
            signal_type: SignalType::Buy,
            strength: 0.92,
            source: "MACD".to_string(),
            expected_action: Some(DecisionAction::Buy),
        },
        TestSignal {
            timestamp: Utc.with_ymd_and_hms(2025, 5, 1, 10, 20, 0).unwrap(), // Very short time frame
            symbol: "BTCUSDT".to_string(),
            signal_type: SignalType::Sell,
            strength: 0.85,
            source: "RSI".to_string(),
            expected_action: Some(DecisionAction::Sell),
        },
        TestSignal {
            timestamp: Utc.with_ymd_and_hms(2025, 5, 1, 10, 25, 0).unwrap(), // Very short time frame
            symbol: "BTCUSDT".to_string(),
            signal_type: SignalType::Buy,
            strength: 0.87,
            source: "MACD".to_string(),
            expected_action: Some(DecisionAction::Buy),
        },
    ];

    // Expected outcomes
    let mut expected_balance_ranges = HashMap::new();
    expected_balance_ranges.insert("USDT".to_string(), (8500.0, 11500.0)); // +/- 15% range (wider due to volatility)
    expected_balance_ranges.insert("BTC".to_string(), (0.0, 0.2));         // May end with variable BTC amount

    let expected_outcomes = ExpectedOutcomes {
        final_balances: expected_balance_ranges,
        order_count: (5, 8),         // Should have multiple orders in rapid succession
        filled_order_count: (5, 8),  // Most orders should be filled
        pnl_range: (-800.0, 800.0),  // Wide range due to extreme volatility
        max_drawdown: 15.0,          // High drawdown expected in extreme conditions
    };

    TestScenario {
        name: "Extreme Volatility Scenario".to_string(),
        description: "Stress test of system behavior in extreme market volatility with rapid, large price changes".to_string(),
        market_condition: MarketCondition::Volatile,
        initial_balances,
        signals,
        expected_outcomes,
        max_test_duration_seconds: 60,
    }
}

/// Generate historical market data for testing based on scenario
pub fn generate_market_data(scenario: &TestScenario) -> Vec<MarketDataSnapshot> {
    match scenario.market_condition {
        MarketCondition::Trending => generate_trending_market_data(),
        MarketCondition::Ranging => generate_ranging_market_data(),
        MarketCondition::Volatile => generate_volatile_market_data(),
        MarketCondition::LowLiquidity => generate_low_liquidity_market_data(),
        MarketCondition::Normal => generate_normal_market_data(),
    }
}

/// Generate market data for a trending market
fn generate_trending_market_data() -> Vec<MarketDataSnapshot> {
    let mut snapshots = Vec::new();
    let start_price = 50000.0;
    let base_time = Utc.with_ymd_and_hms(2025, 5, 1, 9, 0, 0).unwrap();
    
    // Generate 8 hours of market data at 15-minute intervals
    for i in 0..32 {
        let timestamp = base_time + chrono::Duration::minutes(15 * i);
        
        // Uptrending price (increasing ~5% over 8 hours)
        let price_factor = 1.0 + (i as f64 * 0.0015);
        let current_price = start_price * price_factor;
        
        // Generate order book with realistic spreads
        let spread = current_price * 0.0005; // 0.05% spread
        let bid_price = current_price - spread / 2.0;
        let ask_price = current_price + spread / 2.0;
        
        // Generate bids (descending prices)
        let mut bids = Vec::new();
        for j in 0..10 {
            let price = bid_price * (1.0 - j as f64 * 0.0005);
            let quantity = 0.5 + (10.0 - j as f64) * 0.05;
            bids.push((price, quantity));
        }
        
        // Generate asks (ascending prices)
        let mut asks = Vec::new();
        for j in 0..10 {
            let price = ask_price * (1.0 + j as f64 * 0.0005);
            let quantity = 0.5 + (10.0 - j as f64) * 0.05;
            asks.push((price, quantity));
        }
        
        // Create snapshot
        snapshots.push(MarketDataSnapshot {
            timestamp,
            symbol: "BTCUSDT".to_string(),
            price: current_price,
            bids,
            asks,
            volume_24h: 5000.0 + (i as f64 * 100.0),
        });
    }
    
    snapshots
}

/// Generate market data for a ranging market
fn generate_ranging_market_data() -> Vec<MarketDataSnapshot> {
    let mut snapshots = Vec::new();
    let base_price = 50000.0;
    let base_time = Utc.with_ymd_and_hms(2025, 5, 1, 9, 0, 0).unwrap();
    
    // Generate 8 hours of market data at 15-minute intervals
    for i in 0..32 {
        let timestamp = base_time + chrono::Duration::minutes(15 * i);
        
        // Oscillating price (sine wave pattern)
        let oscillation = (i as f64 * std::f64::consts::PI / 8.0).sin() * 500.0;
        let current_price = base_price + oscillation;
        
        // Generate order book with realistic spreads
        let spread = current_price * 0.0008; // 0.08% spread (slightly wider in ranging market)
        let bid_price = current_price - spread / 2.0;
        let ask_price = current_price + spread / 2.0;
        
        // Generate bids (descending prices)
        let mut bids = Vec::new();
        for j in 0..10 {
            let price = bid_price * (1.0 - j as f64 * 0.0006);
            let quantity = 0.4 + (10.0 - j as f64) * 0.06;
            bids.push((price, quantity));
        }
        
        // Generate asks (ascending prices)
        let mut asks = Vec::new();
        for j in 0..10 {
            let price = ask_price * (1.0 + j as f64 * 0.0006);
            let quantity = 0.4 + (10.0 - j as f64) * 0.06;
            asks.push((price, quantity));
        }
        
        // Create snapshot
        snapshots.push(MarketDataSnapshot {
            timestamp,
            symbol: "BTCUSDT".to_string(),
            price: current_price,
            bids,
            asks,
            volume_24h: 3000.0 + oscillation.abs() * 2.0,
        });
    }
    
    snapshots
}

/// Generate market data for a volatile market
fn generate_volatile_market_data() -> Vec<MarketDataSnapshot> {
    let mut snapshots = Vec::new();
    let base_price = 50000.0;
    let base_time = Utc.with_ymd_and_hms(2025, 5, 1, 9, 0, 0).unwrap();
    
    // Generate 2 hours of market data at 5-minute intervals (volatile markets need higher frequency)
    for i in 0..24 {
        let timestamp = base_time + chrono::Duration::minutes(5 * i);
        
        // Volatile price (random walk with higher amplitude)
        let volatility = (i as f64 * 0.5).sin() * 2000.0 + (i as f64 * 0.3).cos() * 1500.0;
        let current_price = base_price + volatility;
        
        // Generate order book with wider spreads (typical in volatile markets)
        let spread = current_price * 0.002; // 0.2% spread
        let bid_price = current_price - spread / 2.0;
        let ask_price = current_price + spread / 2.0;
        
        // Generate bids (descending prices, thinner book in volatile market)
        let mut bids = Vec::new();
        for j in 0..10 {
            let price = bid_price * (1.0 - j as f64 * 0.001);
            let quantity = 0.3 + (10.0 - j as f64) * 0.04 + (j as f64 * 0.01).sin();
            bids.push((price, quantity));
        }
        
        // Generate asks (ascending prices, thinner book in volatile market)
        let mut asks = Vec::new();
        for j in 0..10 {
            let price = ask_price * (1.0 + j as f64 * 0.001);
            let quantity = 0.3 + (10.0 - j as f64) * 0.04 + (j as f64 * 0.01).cos();
            asks.push((price, quantity));
        }
        
        // Create snapshot
        snapshots.push(MarketDataSnapshot {
            timestamp,
            symbol: "BTCUSDT".to_string(),
            price: current_price,
            bids,
            asks,
            volume_24h: 8000.0 + volatility.abs() * 3.0,
        });
    }
    
    snapshots
}

/// Generate market data for a low liquidity market
fn generate_low_liquidity_market_data() -> Vec<MarketDataSnapshot> {
    let mut snapshots = Vec::new();
    let base_price = 50000.0;
    let base_time = Utc.with_ymd_and_hms(2025, 5, 1, 9, 0, 0).unwrap();
    
    // Generate 4 hours of market data at 15-minute intervals
    for i in 0..16 {
        let timestamp = base_time + chrono::Duration::minutes(15 * i);
        
        // Slightly fluctuating price with occasional jumps
        let random_factor = (i as f64 * 0.7).sin() * 200.0;
        let jump_factor = if i % 5 == 0 { 500.0 } else { 0.0 };
        let current_price = base_price + random_factor + jump_factor;
        
        // Generate order book with wide spreads (typical in low liquidity markets)
        let spread = current_price * 0.003; // 0.3% spread
        let bid_price = current_price - spread / 2.0;
        let ask_price = current_price + spread / 2.0;
        
        // Generate bids (descending prices, very thin book)
        let mut bids = Vec::new();
        for j in 0..10 {
            let price = bid_price * (1.0 - j as f64 * 0.002);
            let quantity = 0.1 + (10.0 - j as f64) * 0.02; // Low quantities
            bids.push((price, quantity));
        }
        
        // Generate asks (ascending prices, very thin book)
        let mut asks = Vec::new();
        for j in 0..10 {
            let price = ask_price * (1.0 + j as f64 * 0.002);
            let quantity = 0.1 + (10.0 - j as f64) * 0.02; // Low quantities
            asks.push((price, quantity));
        }
        
        // Create snapshot
        snapshots.push(MarketDataSnapshot {
            timestamp,
            symbol: "BTCUSDT".to_string(),
            price: current_price,
            bids,
            asks,
            volume_24h: 1000.0 + (i as f64 * 30.0),
        });
    }
    
    snapshots
}

/// Generate market data for normal market conditions
fn generate_normal_market_data() -> Vec<MarketDataSnapshot> {
    let mut snapshots = Vec::new();
    let base_price = 50000.0;
    let base_time = Utc.with_ymd_and_hms(2025, 5, 1, 9, 0, 0).unwrap();
    
    // Generate 8 hours of market data at 15-minute intervals
    for i in 0..32 {
        let timestamp = base_time + chrono::Duration::minutes(15 * i);
        
        // Normal price movements
        let random_factor = (i as f64 * 0.2).sin() * 100.0 + (i as f64 * 0.1).cos() * 50.0;
        let current_price = base_price + random_factor;
        
        // Generate order book with normal spreads
        let spread = current_price * 0.0006; // 0.06% spread
        let bid_price = current_price - spread / 2.0;
        let ask_price = current_price + spread / 2.0;
        
        // Generate bids (descending prices)
        let mut bids = Vec::new();
        for j in 0..10 {
            let price = bid_price * (1.0 - j as f64 * 0.0004);
            let quantity = 0.5 + (10.0 - j as f64) * 0.08;
            bids.push((price, quantity));
        }
        
        // Generate asks (ascending prices)
        let mut asks = Vec::new();
        for j in 0..10 {
            let price = ask_price * (1.0 + j as f64 * 0.0004);
            let quantity = 0.5 + (10.0 - j as f64) * 0.08;
            asks.push((price, quantity));
        }
        
        // Create snapshot
        snapshots.push(MarketDataSnapshot {
            timestamp,
            symbol: "BTCUSDT".to_string(),
            price: current_price,
            bids,
            asks,
            volume_24h: 4000.0 + (i as f64 * 50.0),
        });
    }
    
    snapshots
}
