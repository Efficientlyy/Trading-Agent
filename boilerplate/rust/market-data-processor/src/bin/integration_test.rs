use std::env;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use chrono::Utc;
use market_data_processor::utils::enhanced_config::EnhancedConfig;
use tokio::sync::Mutex;
use tracing::{info, warn, error, Level};
use tracing_subscriber::FmtSubscriber;

// Import our test modules
mod tests {
    pub use market_data_processor_tests::integration::*;
}

use tests::{
    load_test_scenarios, generate_market_data, 
    IntegrationTestRunner, MockMarketDataService,
    PerformanceMetrics, MetricsReporter, TradeRecord
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging with timestamps
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .with_target(false)
        .with_ansi(false) // Disable colors for better log file readability
        .with_timer(tracing_subscriber::fmt::time::uptime())
        .finish();
    
    tracing::subscriber::set_global_default(subscriber)?;
    
    info!("========== Paper Trading Integration Tests ==========");
    info!("Starting integration tests at {}", Utc::now());
    
    // Load configuration
    let config = load_test_configuration()?;
    info!("Test configuration loaded successfully");
    
    // Load test scenarios
    let scenarios = load_test_scenarios();
    info!("Loaded {} test scenarios", scenarios.len());
    
    // Create metrics reporter
    let mut metrics_reporter = MetricsReporter::new();
    
    // Track overall test statistics
    let mut total_tests = 0;
    let mut passed_tests = 0;
    let start_time = Instant::now();
    
    // Run each scenario as an independent test
    for scenario in &scenarios {
        info!("======= Test Scenario: {} =======", scenario.name);
        info!("Description: {}", scenario.description);
        info!("Market Condition: {:?}", scenario.market_condition);
        
        // Generate market data for this scenario
        let market_data = generate_market_data(scenario);
        info!("Generated {} market data points", market_data.len());
        
        // Create mock market data service
        let mock_market_data = Arc::new(Mutex::new(MockMarketDataService::new(market_data)));
        
        // Create test runner for this scenario
        let test_runner = IntegrationTestRunner::new(
            config.clone(),
            vec![scenario.clone()],
            mock_market_data,
        ).await;
        
        // Run the test
        info!("Executing test...");
        let result = test_runner.run_scenario(scenario).await;
        
        // Record test outcome
        total_tests += 1;
        if result.passed {
            passed_tests += 1;
            info!("✅ Test PASSED: {}", scenario.name);
        } else {
            error!("❌ Test FAILED: {}", scenario.name);
            for failure in &result.failures {
                error!("  - {}", failure);
            }
        }
        
        // Create performance metrics
        let mut performance_metrics = PerformanceMetrics::new(scenario.name.clone());
        
        // Convert test results to metrics
        let trade_records = test_runner.get_trade_records().await;
        
        performance_metrics.calculate_from_trades(
            &trade_records,
            result.initial_balance,
            result.final_balance,
            result.final_balances.clone(),
        );
        
        // Update risk metrics
        performance_metrics.update_risk_metrics(
            result.max_drawdown,
            result.max_drawdown_amount,
            result.peak_balance,
            result.trough_balance,
            &result.returns,
        );
        
        // Add metrics to reporter
        metrics_reporter.add_metrics(performance_metrics);
        
        // Output key metrics
        info!("Test Execution Time: {}ms", result.execution_time_ms);
        info!("Total P&L: ${:.2}", result.total_pnl);
        info!("Win Rate: {:.2}%", result.win_rate);
        info!("Max Drawdown: {:.2}%", result.max_drawdown);
        if let Some(sharpe) = result.sharpe_ratio {
            info!("Sharpe Ratio: {:.2}", sharpe);
        }
        
        info!(""); // Empty line for better readability
    }
    
    // Output overall results
    let elapsed = start_time.elapsed();
    info!("========== Test Results Summary ==========");
    info!("Total Tests: {}", total_tests);
    info!("Passed: {} ({}%)", passed_tests, (passed_tests as f64 / total_tests as f64) * 100.0);
    info!("Failed: {}", total_tests - passed_tests);
    info!("Total Execution Time: {:.2}s", elapsed.as_secs_f64());
    
    // Generate summary report
    let summary = metrics_reporter.generate_summary();
    
    // Save metrics to file
    let results_dir = Path::new("tests").join("results");
    std::fs::create_dir_all(&results_dir)?;
    
    let timestamp = Utc::now().format("%Y%m%d_%H%M%S");
    let metrics_file = results_dir.join(format!("metrics_{}.json", timestamp));
    metrics_reporter.save_to_file(metrics_file.to_str().unwrap())?;
    
    // Also save as latest_metrics.json for the report generator
    metrics_reporter.save_to_file(results_dir.join("latest_metrics.json").to_str().unwrap())?;
    
    // Save summary to file
    let summary_file = results_dir.join(format!("summary_{}.md", timestamp));
    std::fs::write(&summary_file, summary)?;
    
    info!("Metrics saved to: {}", metrics_file.display());
    info!("Summary saved to: {}", summary_file.display());
    
    // Return error if any test failed
    if passed_tests < total_tests {
        let error_message = format!("{} out of {} tests failed", total_tests - passed_tests, total_tests);
        return Err(error_message.into());
    }
    
    info!("All tests passed successfully!");
    Ok(())
}

/// Load test configuration from environment variables
fn load_test_configuration() -> Result<EnhancedConfig, Box<dyn std::error::Error>> {
    // Set default environment variables if not already set
    if env::var("PAPER_TRADING").is_err() {
        env::set_var("PAPER_TRADING", "true");
    }
    
    if env::var("PAPER_TRADING_INITIAL_BALANCE_USDT").is_err() {
        env::set_var("PAPER_TRADING_INITIAL_BALANCE_USDT", "10000");
    }
    
    if env::var("PAPER_TRADING_INITIAL_BALANCE_BTC").is_err() {
        env::set_var("PAPER_TRADING_INITIAL_BALANCE_BTC", "1");
    }
    
    if env::var("MAX_POSITION_SIZE").is_err() {
        env::set_var("MAX_POSITION_SIZE", "1.0");
    }
    
    if env::var("DEFAULT_ORDER_SIZE").is_err() {
        env::set_var("DEFAULT_ORDER_SIZE", "0.1");
    }
    
    if env::var("MAX_DRAWDOWN_PERCENT").is_err() {
        env::set_var("MAX_DRAWDOWN_PERCENT", "10");
    }
    
    if env::var("TRADING_PAIRS").is_err() {
        env::set_var("TRADING_PAIRS", "BTCUSDT,ETHUSDT");
    }
    
    // Log environment configuration
    info!("Test Environment Configuration:");
    info!("  PAPER_TRADING={}", env::var("PAPER_TRADING").unwrap_or_default());
    info!("  PAPER_TRADING_INITIAL_BALANCE_USDT={}", env::var("PAPER_TRADING_INITIAL_BALANCE_USDT").unwrap_or_default());
    info!("  PAPER_TRADING_INITIAL_BALANCE_BTC={}", env::var("PAPER_TRADING_INITIAL_BALANCE_BTC").unwrap_or_default());
    info!("  MAX_POSITION_SIZE={}", env::var("MAX_POSITION_SIZE").unwrap_or_default());
    info!("  DEFAULT_ORDER_SIZE={}", env::var("DEFAULT_ORDER_SIZE").unwrap_or_default());
    info!("  MAX_DRAWDOWN_PERCENT={}", env::var("MAX_DRAWDOWN_PERCENT").unwrap_or_default());
    info!("  TRADING_PAIRS={}", env::var("TRADING_PAIRS").unwrap_or_default());
    
    // Load enhanced configuration
    let config = EnhancedConfig::load()?;
    
    Ok(config)
}
