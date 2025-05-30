use market_data_processor::{
    ApiServer, MarketDataService, OrderExecutionService, PaperTradingService, 
    SignalGenerator, EnhancedConfig
};
use market_data_processor::api::visualization::dashboard::{TradingDashboard, DashboardConfig};
use market_data_processor::services::optimized_rest_client::OptimizedRestClient;
use market_data_processor::services::decision_module::DecisionModule;
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{info, error, Level};
use tracing_subscriber::FmtSubscriber;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;
    
    info!("Starting MEXC Trading Agent - Market Data Processor");
    
    // Load enhanced configuration
    let config = match EnhancedConfig::load() {
        Ok(cfg) => Arc::new(cfg),
        Err(e) => {
            error!("Failed to load configuration: {}", e);
            return Err(Box::new(e));
        }
    };
    
    // Initialize REST client for market data
    let rest_client = Arc::new(OptimizedRestClient::new(&config));
    
    // Set up signal channel for communication between components
    let (signal_sender, signal_receiver) = mpsc::channel(100);
    
    // Initialize signal generator
    let signal_generator = Arc::new(SignalGenerator::new(
        config.clone(),
        rest_client.clone(),
        signal_sender,
    ));
    
    // Initialize paper trading service
    let paper_trading_service = Arc::new(PaperTradingService::new(
        config.clone(),
        rest_client.clone(),
    ));
    
    // Start paper trading service
    paper_trading_service.start().await?;
    
    // Initialize order execution service (with paper trading only for now)
    let order_execution_service = Arc::new(OrderExecutionService::new(
        config.clone(),
        paper_trading_service.clone(),
        None, // No live trading service for now
    ));
    
    // Initialize decision module
    let (mut decision_module, decision_receiver) = DecisionModule::new(
        config.clone(),
        signal_receiver,
        order_execution_service.clone(),
    );
    
    // Initialize dashboard
    let dashboard = TradingDashboard::new(
        config.clone(),
        order_execution_service.clone(),
        paper_trading_service.clone(),
        signal_generator.clone(),
    );
    
    // Configure dashboard
    let dashboard_config = DashboardConfig {
        port: 8080,
        host: "0.0.0.0".to_string(), // Bind to all interfaces for Docker compatibility
        static_files_path: "./static".to_string(),
        enable_cors: true,
    };
    
    // Start signal generator
    signal_generator.start().await;
    
    // Start components in separate tasks
    let decision_task = tokio::spawn(async move {
        decision_module.start().await;
    });
    
    let dashboard_task = tokio::spawn(async move {
        if let Err(e) = dashboard.start(dashboard_config).await {
            error!("Dashboard error: {}", e);
        }
    });
    
    // Wait for tasks to complete (they should run indefinitely)
    tokio::select! {
        _ = decision_task => error!("Decision module task ended unexpectedly"),
        _ = dashboard_task => error!("Dashboard task ended unexpectedly"),
    }
    
    Ok(())
}
