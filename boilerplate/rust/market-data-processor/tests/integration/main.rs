mod test_framework;
mod test_scenarios;

use std::env;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use chrono::Utc;
use test_framework::{IntegrationTestRunner, TestResults};
use test_scenarios::{load_test_scenarios, generate_market_data};
use market_data_processor::utils::enhanced_config::EnhancedConfig;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;
    
    info!("Starting Paper Trading Integration Tests");
    
    // Load test configuration (using environment variables for Docker compatibility)
    let config = load_test_configuration()?;
    
    // Load test scenarios
    let synthetic_scenarios = load_test_scenarios();
    info!("Loaded {} synthetic test scenarios", synthetic_scenarios.len());
    
    // Create historical data test scenarios
    let data_dir = Path::new("data/historical");
    let historical_scenarios = create_historical_test_scenarios(data_dir).await;
    info!("Loaded {} historical test scenarios", historical_scenarios.len());
    
    // Combine all scenarios
    let mut all_scenarios = synthetic_scenarios;
    all_scenarios.extend(historical_scenarios);
    
    info!("Running {} test scenarios ({} synthetic, {} historical)", 
          all_scenarios.len(), 
          synthetic_scenarios.len(),
          historical_scenarios.len());
    
    // Generate market data for each scenario
    let mut all_market_data = Vec::new();
    for scenario in &all_scenarios {
        let market_data = generate_market_data(scenario);
        info!("Generated {} market data points for scenario: {}", market_data.len(), scenario.name);
        all_market_data.extend(market_data);
    }
    
    // Create test runner
    let test_runner = IntegrationTestRunner::new(
        config,
        scenarios,
        all_market_data,
    ).await;
    
    // Run all tests
    let results = test_runner.run_all_tests().await;
    
    // Output results
    output_test_results(&results)?;
    
    // Return error if any test failed
    if results.iter().any(|r| !r.passed) {
        let failed_count = results.iter().filter(|r| !r.passed).count();
        let error_message = format!("{} out of {} tests failed", failed_count, results.len());
        info!("{}", error_message);
        return Err(error_message.into());
    }
    
    info!("All tests passed successfully!");
    Ok(())
}

/// Load test configuration from environment variables
fn load_test_configuration() -> Result<EnhancedConfig, Box<dyn std::error::Error>> {
    // Set required environment variables for paper trading testing
    env::set_var("PAPER_TRADING", "true");
    env::set_var("PAPER_TRADING_INITIAL_BALANCE_USDT", "10000");
    env::set_var("PAPER_TRADING_INITIAL_BALANCE_BTC", "1");
    env::set_var("MAX_POSITION_SIZE", "1.0");
    env::set_var("DEFAULT_ORDER_SIZE", "0.1");
    env::set_var("MAX_DRAWDOWN_PERCENT", "10");
    env::set_var("TRADING_PAIRS", "BTCUSDT,ETHUSDT");
    
    // Load configuration
    let config = EnhancedConfig::load()?;
    
    Ok(config)
}

/// Output test results to console and JSON file
fn output_test_results(results: &[TestResults]) -> Result<(), Box<dyn std::error::Error>> {
    // Output summary to console
    info!("Test Results Summary:");
    info!("=====================");
    
    let total_tests = results.len();
    let passed_tests = results.iter().filter(|r| r.passed).count();
    
    info!("Total Tests: {}", total_tests);
    info!("Passed: {} ({}%)", passed_tests, (passed_tests as f64 / total_tests as f64) * 100.0);
    info!("Failed: {}", total_tests - passed_tests);
    
    // Output detailed results
    for result in results {
        let status = if result.passed { "PASSED" } else { "FAILED" };
        info!("{}: {} - Execution time: {}ms", status, result.scenario_name, result.execution_time_ms);
        
        if !result.passed {
            for failure in &result.failures {
                info!("  - Failure: {}", failure);
            }
        }
        
        info!("  - Total P&L: {:.2}", result.total_pnl);
        info!("  - Win Rate: {:.2}%", result.win_rate);
        info!("  - Max Drawdown: {:.2}%", result.max_drawdown);
        if let Some(sharpe) = result.sharpe_ratio {
            info!("  - Sharpe Ratio: {:.2}", sharpe);
        }
    }
    
    // Output results to JSON file
    let json = serde_json::to_string_pretty(results)?;
    let timestamp = Utc::now().format("%Y%m%d_%H%M%S");
    let filename = format!("test_results_{}.json", timestamp);
    let path = Path::new("tests").join("results").join(&filename);
    
    // Create directory if it doesn't exist
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    
    // Write results to file
    let mut file = File::create(&path)?;
    file.write_all(json.as_bytes())?;
    
    info!("Test results written to: {}", path.display());
    
    Ok(())
}

/// Format currency value with 2 decimal places
fn format_currency(value: f64) -> String {
    format!("${:.2}", value)
}
