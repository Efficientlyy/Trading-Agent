use std::env;
use std::process::Command;
use std::path::Path;

/// Docker test runner
/// This script helps run tests inside a Docker container to ensure consistent environment
/// matching the production deployment of the Market Data Processor.
fn main() {
    println!("=== Docker Integration Test Runner ===");
    println!("Running tests in a containerized environment");
    
    // Get project root directory
    let project_dir = env::current_dir()
        .expect("Failed to get current directory")
        .parent()
        .expect("Failed to get parent directory")
        .to_path_buf();
    
    println!("Project directory: {}", project_dir.display());
    
    // Check if Docker is available
    match Command::new("docker").arg("--version").output() {
        Ok(output) => {
            if output.status.success() {
                println!("Docker available: {}", String::from_utf8_lossy(&output.stdout).trim());
            } else {
                eprintln!("Docker check failed: {}", String::from_utf8_lossy(&output.stderr).trim());
                std::process::exit(1);
            }
        }
        Err(e) => {
            eprintln!("Failed to execute Docker: {}. Is Docker installed?", e);
            std::process::exit(1);
        }
    }
    
    // Build the test Docker image
    println!("Building test Docker image...");
    let build_status = Command::new("docker")
        .args(&[
            "build",
            "-t", "mexc-trading-agent-test",
            "-f", "Dockerfile.test",
            "."
        ])
        .current_dir(&project_dir)
        .status()
        .expect("Failed to build Docker image");
    
    if !build_status.success() {
        eprintln!("Failed to build Docker image");
        std::process::exit(1);
    }
    
    // Run tests in Docker container
    println!("Running tests in Docker container...");
    let run_status = Command::new("docker")
        .args(&[
            "run",
            "--rm",
            "-v", &format!("{}:/app", project_dir.display()),
            "-e", "RUST_BACKTRACE=1",
            "-e", "PAPER_TRADING=true",
            "-e", "PAPER_TRADING_INITIAL_BALANCE_USDT=10000",
            "-e", "PAPER_TRADING_INITIAL_BALANCE_BTC=1",
            "-e", "MAX_POSITION_SIZE=1.0",
            "-e", "DEFAULT_ORDER_SIZE=0.1",
            "-e", "MAX_DRAWDOWN_PERCENT=10",
            "-e", "TRADING_PAIRS=BTCUSDT,ETHUSDT",
            "mexc-trading-agent-test",
            "/bin/bash", "-c", "cd /app && cargo test --test market_data_processor_tests"
        ])
        .current_dir(&project_dir)
        .status()
        .expect("Failed to run tests in Docker");
    
    if run_status.success() {
        println!("✅ Tests passed in Docker container!");
    } else {
        eprintln!("❌ Tests failed in Docker container");
        std::process::exit(1);
    }
}
