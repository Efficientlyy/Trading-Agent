use std::sync::Arc;
use std::collections::HashMap;

use market_data_processor::models::order::{OrderSide, OrderType};
use market_data_processor::services::order_execution::OrderRequest;
use market_data_processor::services::paper_trading::PaperTradingService;
use market_data_processor::utils::enhanced_config::EnhancedConfig;

use crate::test_framework::{MockMarketDataService, MarketDataSnapshot};
use crate::mock_matching_engine::MockMatchingEngine;

/// A simple integration test for paper trading module
#[tokio::test]
async fn test_paper_trading_basic_functionality() {
    // Setup test logging
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .try_init();
    
    // Create mock market data
    let market_data = vec![
        MarketDataSnapshot {
            timestamp: chrono::Utc::now(),
            symbol: "BTCUSDC".to_string(),
            price: 50000.0,
            bids: vec![(49990.0, 1.0), (49980.0, 2.0), (49970.0, 3.0)],
            asks: vec![(50010.0, 1.0), (50020.0, 2.0), (50030.0, 3.0)],
            volume_24h: 1000.0,
        },
    ];
    
    // Create mock market data service
    let mock_market_data = Arc::new(tokio::sync::Mutex::new(MockMarketDataService::new(market_data)));
    
    // Create configuration
    let mut config = EnhancedConfig {
        paper_trading: true,
        paper_trading_initial_balance_usdt: 10000.0,
        paper_trading_initial_balance_btc: 1.0,
        max_position_size: 1.0,
        default_order_size: 0.1,
        max_drawdown_percent: 10.0,
        trading_pairs: vec!["BTCUSDC".to_string()],
        ..Default::default()
    };
    
    // Create mock matching engine
    let matching_engine = Arc::new(MockMatchingEngine::new(mock_market_data.clone(), 0.005));
    
    // Create initial balances
    let mut initial_balances = HashMap::new();
    initial_balances.insert("USDC".to_string(), 10000.0);
    initial_balances.insert("BTC".to_string(), 1.0);
    
    // Create virtual account
    let virtual_account = market_data_processor::services::paper_trading::virtual_account::VirtualAccount::new(initial_balances);
    
    // Create paper trading service with mocked components
    let paper_trading_service = PaperTradingService::new_with_components(
        Arc::new(config),
        matching_engine,
        Arc::new(tokio::sync::Mutex::new(virtual_account)),
    );
    
    // Test 1: Place a buy order
    let buy_order_request = OrderRequest {
        symbol: "BTCUSDC".to_string(),
        side: OrderSide::Buy,
        order_type: OrderType::Market,
        quantity: 0.1,
        price: None,
        time_in_force: None,
        client_order_id: None,
    };
    
    let buy_result = paper_trading_service.place_order(buy_order_request).await;
    assert!(buy_result.is_ok(), "Buy order should succeed");
    
    if let Ok(response) = buy_result {
        println!("Buy order placed: {}", response.order_id);
        assert!(response.executed_qty.unwrap_or_default() > 0.0, "Buy order should execute partially or fully");
    }
    
    // Check balances after buy
    let balances = paper_trading_service.get_balances().await;
    assert!(balances.is_ok(), "Should be able to get balances");
    
    if let Ok(balances) = balances {
        let usdc_balance = balances.iter().find(|(asset, _)| asset == "USDC").map(|(_, balance)| *balance).unwrap_or_default();
        let btc_balance = balances.iter().find(|(asset, _)| asset == "BTC").map(|(_, balance)| *balance).unwrap_or_default();
        
        println!("After buy - USDC: {}, BTC: {}", usdc_balance, btc_balance);
        assert!(btc_balance > 1.0, "BTC balance should increase after buy");
        assert!(usdc_balance < 10000.0, "USDC balance should decrease after buy");
    }
    
    // Test 2: Place a sell order
    let sell_order_request = OrderRequest {
        symbol: "BTCUSDC".to_string(),
        side: OrderSide::Sell,
        order_type: OrderType::Market,
        quantity: 0.05,
        price: None,
        time_in_force: None,
        client_order_id: None,
    };
    
    let sell_result = paper_trading_service.place_order(sell_order_request).await;
    assert!(sell_result.is_ok(), "Sell order should succeed");
    
    if let Ok(response) = sell_result {
        println!("Sell order placed: {}", response.order_id);
        assert!(response.executed_qty.unwrap_or_default() > 0.0, "Sell order should execute partially or fully");
    }
    
    // Check balances after sell
    let balances = paper_trading_service.get_balances().await;
    assert!(balances.is_ok(), "Should be able to get balances");
    
    if let Ok(balances) = balances {
        let btc_balance = balances.iter().find(|(asset, _)| asset == "BTC").map(|(_, balance)| *balance).unwrap_or_default();
        
        println!("After sell - BTC: {}", btc_balance);
        assert!(btc_balance < 1.1, "BTC balance should decrease after sell");
    }
    
    // Test 3: Check positions
    let positions = paper_trading_service.get_positions().await;
    assert!(positions.is_ok(), "Should be able to get positions");
    
    if let Ok(positions) = positions {
        println!("Positions: {:?}", positions);
        assert!(!positions.is_empty(), "Should have at least one position");
    }
    
    // Test 4: Update positions with current prices
    let update_result = paper_trading_service.update_positions().await;
    assert!(update_result.is_ok(), "Should be able to update positions");
    
    // Test 5: Get order history
    let order_history = paper_trading_service.get_order_history().await;
    assert!(order_history.is_ok(), "Should be able to get order history");
    
    if let Ok(orders) = order_history {
        println!("Order history count: {}", orders.len());
        assert_eq!(orders.len(), 2, "Should have 2 orders in history");
    }
    
    // Test 6: Get trade history
    let trade_history = paper_trading_service.get_trade_history().await;
    assert!(trade_history.is_ok(), "Should be able to get trade history");
    
    if let Ok(trades) = trade_history {
        println!("Trade history count: {}", trades.len());
        assert!(trades.len() >= 2, "Should have at least 2 trades in history");
        
        // Check trade details
        for trade in trades {
            println!("Trade: {} {} {} at {}", 
                trade.symbol, 
                trade.side, 
                trade.quantity, 
                trade.price);
        }
    }
    
    println!("All paper trading tests passed!");
}
