// Main entry point for the Market Data Processor
use std::net::SocketAddr;
use std::sync::Arc;
use std::path::Path;
use tokio::sync::{mpsc, watch};
use tracing::{info, debug, error};

use market_data_processor::api::server::{ApiServer, start_api_server};
use market_data_processor::services::market_data_service::MarketDataService;
use market_data_processor::services::paper_trading::service::PaperTradingService;
use market_data_processor::config::EnhancedConfig;
use market_data_processor::utils::config::Config;
use market_data_processor::utils::logging;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load configuration
    let config = Config::load()?;
    
    // Create enhanced config for services
    let enhanced_config = Arc::new(EnhancedConfig::from_config(&config));
    
    // Initialize logging
    logging::init(&config)?;
    
    info!("Starting Market Data Processor");
    debug!("Paper trading mode: {}", enhanced_config.paper_trading_enabled);
    
    // Create channels
    let (message_tx, message_rx) = mpsc::channel(10000);
    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    
    // Create market data service
    let market_data_service = Arc::new(MarketDataService::new(
        message_rx, 
        shutdown_rx.clone(),
        enhanced_config.clone(),
    ));
    
    // Create paper trading service if enabled
    let paper_trading_service = if enhanced_config.paper_trading_enabled {
        info!("Initializing paper trading service");
        let service = PaperTradingService::new(
            enhanced_config.clone(),
            market_data_service.clone(),
        );
        Some(Arc::new(service))
    } else {
        None
    };
    
    // Determine the dashboard path if we're serving the dashboard
    let dashboard_path = if enhanced_config.serve_dashboard {
        let path = enhanced_config.dashboard_path.clone()
            .unwrap_or_else(|| "./dashboard/build".to_string());
            
        info!("Will serve dashboard from: {}", path);
        
        if !Path::new(&path).exists() {
            error!("Dashboard path does not exist: {}", path);
            error!("Make sure to build the dashboard with 'npm run build' before starting the server");
            None
        } else {
            Some(path)
        }
    } else {
        None
    };
    
    // Parse socket addresses
    let grpc_addr = config.grpc_server_addr.parse::<SocketAddr>()?;
    let http_addr = enhanced_config.http_server_addr.parse::<SocketAddr>()?;
    
    // Create API server
    let api_server = ApiServer::new(market_data_service.clone(), grpc_addr);
    
    // Start API server (combines gRPC and HTTP)
    info!("Starting API servers (gRPC on {}, HTTP on {})", grpc_addr, http_addr);
    start_api_server(
        market_data_service.clone(),
        enhanced_config.clone(),
        grpc_addr,
        http_addr,
        paper_trading_service,
        dashboard_path,
    ).await?;
    
    // Run market data service
    let market_data_handle = tokio::spawn(async move {
        market_data_service.run().await;
    });
    
    // Wait for shutdown signal
    tokio::signal::ctrl_c().await?;
    
    info!("Shutdown signal received, initiating graceful shutdown");
    
    // Initiate graceful shutdown
    shutdown_tx.send(true)?;
    
    // Wait for services to shut down
    let _ = tokio::join!(market_data_handle);
    
    info!("Market Data Processor shutdown complete");
    
    Ok(())
}
