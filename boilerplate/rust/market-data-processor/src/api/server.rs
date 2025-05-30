use std::sync::Arc;
use std::net::SocketAddr;
use std::path::Path;
use tonic::transport::Server;
use warp::{Filter, fs, path};
use tracing::{info, error, debug};

use crate::api::grpc::GrpcService;
use crate::api::paper_trading_routes;
use crate::api::status_api::StatusService;
use crate::services::market_data_service::MarketDataService;
use crate::services::paper_trading::service::PaperTradingService;
use crate::config::EnhancedConfig;

pub struct ApiServer {
    market_data_service: Arc<MarketDataService>,
    paper_trading_service: Option<Arc<PaperTradingService>>,
    status_service: StatusService,
    config: Arc<EnhancedConfig>,
    grpc_addr: SocketAddr,
    http_addr: SocketAddr,
    dashboard_path: Option<String>,
}

impl ApiServer {
    pub fn new(
        market_data_service: Arc<MarketDataService>, 
        config: Arc<EnhancedConfig>,
        grpc_addr: SocketAddr,
        http_addr: SocketAddr,
    ) -> Self {
        // Initialize status service
        let status_service = StatusService::new(config.clone());
        
        Self {
            market_data_service,
            paper_trading_service: None,
            status_service,
            config,
            grpc_addr,
            http_addr,
            dashboard_path: None,
        }
    }
    
    pub fn with_paper_trading(
        mut self, 
        paper_trading_service: Arc<PaperTradingService>
    ) -> Self {
        self.paper_trading_service = Some(paper_trading_service);
        self
    }
    
    pub fn with_dashboard(
        mut self,
        dashboard_path: String
    ) -> Self {
        self.dashboard_path = Some(dashboard_path);
        self
    }
    
    pub async fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        // Start gRPC server in a background task
        info!("Starting gRPC API server on {}", self.grpc_addr);
        let grpc_service = GrpcService::new(self.market_data_service.clone());
        let grpc_addr = self.grpc_addr;
        
        tokio::spawn(async move {
            match Server::builder()
                .add_service(grpc_service.into_server())
                .serve(grpc_addr)
                .await {
                Ok(_) => info!("gRPC server shut down gracefully"),
                Err(e) => error!("gRPC server error: {}", e),
            }
        });
        
        // Set up HTTP routes for the REST API and dashboard
        info!("Starting HTTP server on {}", self.http_addr);
        
        // Add status API routes
        debug!("Adding status and monitoring API routes");
        let status_routes = self.status_service.routes();
        
        // Create base API routes with CORS enabled
        let cors = warp::cors()
            .allow_any_origin()
            .allow_methods(vec!["GET", "POST", "PUT", "DELETE"])
            .allow_headers(vec!["Content-Type", "Authorization"]);
            
        let mut routes = status_routes.with(cors.clone());
        
        // Add paper trading API routes if paper trading service is available
        if let Some(paper_trading_service) = &self.paper_trading_service {
            debug!("Adding paper trading API routes");
            let paper_trading_routes = paper_trading_routes(
                paper_trading_service.clone(),
                self.config.clone(),
            ).with(cors.clone());
            routes = routes.or(paper_trading_routes);
        }
        
        // Add dashboard static files if dashboard path is provided
        if let Some(dashboard_path) = &self.dashboard_path {
            debug!("Serving dashboard from {}", dashboard_path);
            
            // Serve index.html for all routes that don't match API or static files
            // This supports client-side routing in React
            let index_fallback = warp::get()
                .and(warp::path::end().or(warp::path::tail()).unify())
                .and(warp::fs::file(format!("{}/index.html", dashboard_path)));
            
            // Serve static files from the dashboard build directory
            let static_files = warp::path("static")
                .and(warp::fs::dir(format!("{}/static", dashboard_path)));
                
            // Serve assets directory
            let assets_files = warp::path("assets")
                .and(warp::fs::dir(format!("{}/assets", dashboard_path)));
            
            // Add favicon.ico
            let favicon = warp::path("favicon.ico")
                .and(warp::fs::file(format!("{}/favicon.ico", dashboard_path)));
            
            // Add static file serving to routes
            routes = routes
                .or(static_files)
                .or(assets_files)
                .or(favicon)
                .or(index_fallback);
        }
        
        // Start HTTP server
        let http_addr = self.http_addr;
        warp::serve(routes)
            .run(http_addr)
            .await;
            
        Ok(())
    }
}

// Helper function to start the API server in a background task
pub async fn start_api_server(
    market_data_service: Arc<MarketDataService>,
    config: Arc<EnhancedConfig>,
    grpc_addr: SocketAddr,
    http_addr: SocketAddr,
    paper_trading_service: Option<Arc<PaperTradingService>>,
    dashboard_path: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Create base server
    let mut server = ApiServer::new(market_data_service, config, grpc_addr, http_addr);
    
    // Add paper trading if enabled
    if let Some(paper_trading_service) = paper_trading_service {
        server = server.with_paper_trading(paper_trading_service);
    }
    
    // Add dashboard if path is provided
    if let Some(dashboard_path) = dashboard_path {
        server = server.with_dashboard(dashboard_path);
    }
    
    // Start server in background task
    tokio::spawn(async move {
        match server.run().await {
            Ok(_) => info!("API server shut down gracefully"),
            Err(e) => error!("API server error: {}", e),
        }
    });
    
    // Give the server a moment to start
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    
    Ok(())
}
