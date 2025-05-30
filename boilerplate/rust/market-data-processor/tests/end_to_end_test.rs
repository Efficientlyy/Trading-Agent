#[cfg(test)]
mod end_to_end_tests {
    use std::sync::Arc;
    use std::time::Duration;
    use std::collections::HashMap;
    use tokio::time::sleep;
    use anyhow::{Result, anyhow};
    use reqwest::Client;
    use serde_json::{json, Value};
    use std::env;
    use tokio::sync::oneshot;
    use tracing::{info, error, debug, warn};
    use tracing_subscriber::{fmt, EnvFilter};
    
    use market_data_processor::config::enhanced_config::EnhancedConfig;
    use market_data_processor::services::market_data_service::MarketDataService;
    use market_data_processor::services::paper_trading::service::PaperTradingService;
    use market_data_processor::api::server::start_api_server;
    use market_data_processor::models::order::{Order, OrderSide, OrderType, OrderStatus};
    
    // Initialize tracing for tests
    fn init_tracing() {
        let filter = EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new("info,market_data_processor=debug"));
            
        fmt()
            .with_env_filter(filter)
            .with_test_writer()
            .init();
    }
    
    // Create test config
    fn create_test_config() -> EnhancedConfig {
        let mut initial_balances = HashMap::new();
        initial_balances.insert("USDT".to_string(), 10000.0);
        initial_balances.insert("BTC".to_string(), 1.0);
        
        EnhancedConfig {
            grpc_server_addr: "127.0.0.1:50051".to_string(),
            http_server_addr: "127.0.0.1:8081".to_string(), // Use different port for tests
            paper_trading_enabled: true,
            paper_trading_initial_balances: initial_balances,
            trading_pairs: vec!["BTCUSDT".to_string(), "ETHUSDT".to_string()],
            max_position_size: 1.0,
            default_order_size: 0.1,
            max_drawdown_percent: 10.0,
            serve_dashboard: false, // Don't serve dashboard in tests
            dashboard_path: None,
            paper_trading_slippage_model: "REALISTIC".to_string(),
            paper_trading_latency_model: "NORMAL".to_string(),
            paper_trading_fee_rate: 0.001,
        }
    }
    
    // Helper function to validate a full paper trading workflow
    async fn validate_trading_workflow(client: &Client, base_url: &str) -> Result<()> {
        // Step 1: Check account balances
        info!("Step 1: Checking initial account balances");
        let response = client.get(&format!("{}/api/paper-trading/account", base_url))
            .send()
            .await?;
            
        assert!(response.status().is_success(), "Failed to get account data");
        let account_data: Value = response.json().await?;
        
        // Verify we have balances
        assert!(account_data["balances"].is_object(), "Account data doesn't contain balances");
        assert!(account_data["balances"]["USDT"].as_f64().unwrap() > 0.0, "USDT balance should be positive");
        
        // Step 2: Check market data
        info!("Step 2: Checking market data availability");
        let response = client.get(&format!("{}/api/paper-trading/market-data/BTCUSDT", base_url))
            .send()
            .await?;
            
        assert!(response.status().is_success(), "Failed to get market data");
        let market_data: Value = response.json().await?;
        
        // Verify we have price data
        assert!(market_data["price"].is_number(), "Market data doesn't contain price");
        let btc_price = market_data["price"].as_f64().unwrap();
        info!("Current BTC price: {}", btc_price);
        
        // Step 3: Place a market buy order
        info!("Step 3: Placing market buy order");
        let order_request = json!({
            "symbol": "BTCUSDT",
            "side": "BUY",
            "type": "MARKET",
            "quantity": 0.01
        });
        
        let response = client.post(&format!("{}/api/paper-trading/orders", base_url))
            .json(&order_request)
            .send()
            .await?;
            
        assert!(response.status().is_success(), "Failed to place order");
        let order_response: Value = response.json().await?;
        
        // Verify order was placed
        assert!(order_response["orderId"].is_string(), "Order response doesn't contain orderId");
        let order_id = order_response["orderId"].as_str().unwrap();
        info!("Order placed with ID: {}", order_id);
        
        // Step 4: Wait a moment for the order to be processed
        info!("Step 4: Waiting for order to be processed");
        sleep(Duration::from_millis(500)).await;
        
        // Step 5: Check order status
        info!("Step 5: Checking order status");
        let response = client.get(&format!("{}/api/paper-trading/orders/{}", base_url, order_id))
            .send()
            .await?;
            
        assert!(response.status().is_success(), "Failed to get order status");
        let order_status: Value = response.json().await?;
        
        // Verify order was filled
        assert_eq!(
            order_status["status"].as_str().unwrap(), 
            "FILLED", 
            "Order should be filled, but got status: {}", 
            order_status["status"]
        );
        
        // Step 6: Check updated account balances
        info!("Step 6: Checking updated account balances after trade");
        let response = client.get(&format!("{}/api/paper-trading/account", base_url))
            .send()
            .await?;
            
        assert!(response.status().is_success(), "Failed to get updated account data");
        let updated_account: Value = response.json().await?;
        
        // Verify BTC balance increased
        let btc_balance = updated_account["balances"]["BTC"].as_f64().unwrap();
        info!("Updated BTC balance: {}", btc_balance);
        assert!(btc_balance > 1.0, "BTC balance should have increased");
        
        // Step 7: Place a limit sell order
        info!("Step 7: Placing limit sell order");
        let limit_price = btc_price * 1.05; // 5% above current price
        let order_request = json!({
            "symbol": "BTCUSDT",
            "side": "SELL",
            "type": "LIMIT",
            "quantity": 0.01,
            "price": limit_price
        });
        
        let response = client.post(&format!("{}/api/paper-trading/orders", base_url))
            .json(&order_request)
            .send()
            .await?;
            
        assert!(response.status().is_success(), "Failed to place limit order");
        let limit_order: Value = response.json().await?;
        let limit_order_id = limit_order["orderId"].as_str().unwrap();
        
        // Step 8: Check open orders
        info!("Step 8: Checking open orders");
        let response = client.get(&format!("{}/api/paper-trading/open-orders", base_url))
            .send()
            .await?;
            
        assert!(response.status().is_success(), "Failed to get open orders");
        let open_orders: Value = response.json().await?;
        
        // Verify our limit order is in the open orders
        let found_order = open_orders.as_array().unwrap().iter().any(|order| {
            order["orderId"].as_str().unwrap() == limit_order_id
        });
        
        assert!(found_order, "Limit order should be in open orders");
        
        // Step 9: Cancel the limit order
        info!("Step 9: Cancelling limit order");
        let response = client.delete(&format!("{}/api/paper-trading/orders/{}", base_url, limit_order_id))
            .send()
            .await?;
            
        assert!(response.status().is_success(), "Failed to cancel order");
        
        // Step 10: Check system status
        info!("Step 10: Checking system status");
        let response = client.get(&format!("{}/status", base_url))
            .send()
            .await?;
            
        assert!(response.status().is_success(), "Failed to get system status");
        let status: Value = response.json().await?;
        
        // Verify system is healthy
        assert_eq!(
            status["overall_status"].as_str().unwrap(),
            "OK",
            "System status should be OK"
        );
        
        // Verify trading stats are being tracked
        assert!(status["trading_stats"]["total_trades"].as_u64().unwrap() > 0, 
            "Should have recorded at least one trade");
        
        info!("All validation steps completed successfully!");
        Ok(())
    }
    
    #[tokio::test]
    async fn test_full_trading_pipeline() -> Result<()> {
        // Initialize tracing
        init_tracing();
        info!("Starting end-to-end test of the full trading pipeline");
        
        // Create test configuration
        let config = Arc::new(create_test_config());
        let http_addr = config.http_server_addr.parse().unwrap();
        let grpc_addr = config.grpc_server_addr.parse().unwrap();
        
        // Create services
        let market_data_service = Arc::new(MarketDataService::new_for_testing());
        
        // Add test market data
        market_data_service.add_test_market_data("BTCUSDT", 50000.0);
        market_data_service.add_test_market_data("ETHUSDT", 3000.0);
        
        // Create paper trading service
        let paper_trading_service = Arc::new(
            PaperTradingService::new(
                market_data_service.clone(),
                config.clone(),
            )
        );
        
        // Start API server
        info!("Starting test API server on {}", http_addr);
        start_api_server(
            market_data_service.clone(),
            config.clone(),
            grpc_addr,
            http_addr,
            Some(paper_trading_service.clone()),
            None,
        ).await?;
        
        // Wait for server to start
        sleep(Duration::from_secs(1)).await;
        
        // Create HTTP client
        let client = Client::new();
        let base_url = format!("http://{}", config.http_server_addr);
        
        // Run the validation workflow
        match validate_trading_workflow(&client, &base_url).await {
            Ok(_) => {
                info!("✅ End-to-end test passed successfully!");
                Ok(())
            },
            Err(e) => {
                error!("❌ End-to-end test failed: {}", e);
                Err(anyhow!("End-to-end test failed: {}", e))
            }
        }
    }
    
    #[tokio::test]
    async fn test_market_scenarios() -> Result<()> {
        // Initialize tracing
        init_tracing();
        info!("Starting market scenario tests");
        
        // Create test configuration
        let config = Arc::new(create_test_config());
        let http_addr = "127.0.0.1:8082".parse().unwrap(); // Different port to avoid conflicts
        let grpc_addr = "127.0.0.1:50052".parse().unwrap();
        
        // Create market data service with various scenarios
        let market_data_service = Arc::new(MarketDataService::new_for_testing());
        
        // Scenario 1: Trending market (gradually increasing price)
        info!("Setting up trending market scenario");
        let mut trending_prices = Vec::new();
        let base_price = 50000.0;
        for i in 0..20 {
            trending_prices.push(base_price * (1.0 + (i as f64 * 0.01)));
        }
        market_data_service.add_test_price_sequence("BTCUSDT", trending_prices);
        
        // Create paper trading service
        let paper_trading_service = Arc::new(
            PaperTradingService::new(
                market_data_service.clone(),
                config.clone(),
            )
        );
        
        // Start API server
        info!("Starting test API server for scenario testing on {}", http_addr);
        start_api_server(
            market_data_service.clone(),
            config.clone(),
            grpc_addr,
            http_addr,
            Some(paper_trading_service.clone()),
            None,
        ).await?;
        
        // Wait for server to start
        sleep(Duration::from_secs(1)).await;
        
        // Create HTTP client
        let client = Client::new();
        let base_url = format!("http://127.0.0.1:8082");
        
        // Test scenario 1: Trending market strategy
        info!("Testing trending market strategy");
        
        // Place a buy order at the beginning of the trend
        let order_request = json!({
            "symbol": "BTCUSDT",
            "side": "BUY",
            "type": "MARKET",
            "quantity": 0.1
        });
        
        let response = client.post(&format!("{}/api/paper-trading/orders", base_url))
            .json(&order_request)
            .send()
            .await?;
            
        assert!(response.status().is_success(), "Failed to place buy order");
        
        // Advance the market (simulate time passing)
        for _ in 0..10 {
            market_data_service.advance_test_price("BTCUSDT");
            sleep(Duration::from_millis(100)).await;
        }
        
        // Place a sell order after the price has increased
        let order_request = json!({
            "symbol": "BTCUSDT",
            "side": "SELL",
            "type": "MARKET",
            "quantity": 0.1
        });
        
        let response = client.post(&format!("{}/api/paper-trading/orders", base_url))
            .json(&order_request)
            .send()
            .await?;
            
        assert!(response.status().is_success(), "Failed to place sell order");
        
        // Check account to verify profit
        let response = client.get(&format!("{}/api/paper-trading/account", base_url))
            .send()
            .await?;
            
        let account: Value = response.json().await?;
        let profit = account["unrealized_pnl"].as_f64().unwrap_or(0.0);
        
        info!("Profit from trending market strategy: {}", profit);
        assert!(profit >= 0.0, "Should have made profit in trending market");
        
        // Check trading history
        let response = client.get(&format!("{}/api/paper-trading/order-history", base_url))
            .send()
            .await?;
            
        assert!(response.status().is_success(), "Failed to get order history");
        let history: Value = response.json().await?;
        
        assert!(history.as_array().unwrap().len() >= 2, 
            "Should have at least 2 orders in history");
            
        info!("✅ Market scenarios test passed successfully!");
        Ok(())
    }
}
