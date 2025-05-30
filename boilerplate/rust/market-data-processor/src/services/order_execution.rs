use crate::models::order::{Order, OrderSide, OrderStatus, OrderType, TimeInForce};
use crate::services::live_trading::LiveTradingService;
use crate::services::paper_trading::PaperTradingService;
use crate::utils::enhanced_config::EnhancedConfig;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// Order request from the decision module
#[derive(Debug, Clone)]
pub struct OrderRequest {
    pub symbol: String,
    pub side: OrderSide,
    pub order_type: OrderType,
    pub quantity: f64,
    pub price: Option<f64>,
    pub time_in_force: Option<TimeInForce>,
    pub client_order_id: Option<String>,
}

/// Order response to the decision module
#[derive(Debug, Clone)]
pub struct OrderResponse {
    pub order_id: String,
    pub status: String,
    pub message: Option<String>,
    pub executed_qty: Option<f64>,
    pub executed_price: Option<f64>,
}

/// Service error type
#[derive(Debug, thiserror::Error)]
pub enum ServiceError {
    #[error("Order validation error: {0}")]
    ValidationError(String),
    
    #[error("Order execution error: {0}")]
    ExecutionError(String),
    
    #[error("Service unavailable: {0}")]
    ServiceUnavailable(String),
}

/// Order Execution Facade that routes requests to either live or paper trading
pub struct OrderExecutionService {
    config: Arc<EnhancedConfig>,
    paper_trading_service: Arc<PaperTradingService>,
    live_trading_service: Option<Arc<LiveTradingService>>,
    is_paper_trading: Arc<RwLock<bool>>,
}

impl OrderExecutionService {
    /// Create a new order execution service
    pub fn new(
        config: Arc<EnhancedConfig>,
        paper_trading_service: Arc<PaperTradingService>,
        live_trading_service: Option<Arc<LiveTradingService>>,
    ) -> Self {
        let is_paper_trading = Arc::new(RwLock::new(config.is_paper_trading));
        
        OrderExecutionService {
            config,
            paper_trading_service,
            live_trading_service,
            is_paper_trading,
        }
    }
    
    /// Place an order through the appropriate service
    pub async fn place_order(&self, order_request: OrderRequest) -> Result<OrderResponse, ServiceError> {
        let is_paper = *self.is_paper_trading.read().await;
        
        if is_paper {
            // Route to paper trading
            debug!("Routing order to paper trading: {:?}", order_request);
            self.paper_trading_service.place_order(order_request).await
        } else if let Some(live_service) = &self.live_trading_service {
            // Route to live trading
            info!("Routing order to live trading: {:?}", order_request);
            live_service.place_order(order_request).await
        } else {
            // Live trading service not available
            error!("Live trading service not available");
            Err(ServiceError::ServiceUnavailable("Live trading service not available".to_string()))
        }
    }
    
    /// Cancel an order
    pub async fn cancel_order(&self, order_id: &str) -> Result<OrderResponse, ServiceError> {
        let is_paper = *self.is_paper_trading.read().await;
        
        if is_paper {
            // Route to paper trading
            debug!("Routing cancel request to paper trading for order: {}", order_id);
            self.paper_trading_service.cancel_order(order_id).await
        } else if let Some(live_service) = &self.live_trading_service {
            // Route to live trading
            info!("Routing cancel request to live trading for order: {}", order_id);
            live_service.cancel_order(order_id).await
        } else {
            // Live trading service not available
            error!("Live trading service not available");
            Err(ServiceError::ServiceUnavailable("Live trading service not available".to_string()))
        }
    }
    
    /// Get order details
    pub async fn get_order(&self, order_id: &str) -> Result<Order, ServiceError> {
        let is_paper = *self.is_paper_trading.read().await;
        
        if is_paper {
            // Route to paper trading
            debug!("Routing get order request to paper trading for order: {}", order_id);
            self.paper_trading_service.get_order(order_id).await
        } else if let Some(live_service) = &self.live_trading_service {
            // Route to live trading
            debug!("Routing get order request to live trading for order: {}", order_id);
            live_service.get_order(order_id).await
        } else {
            // Live trading service not available
            error!("Live trading service not available");
            Err(ServiceError::ServiceUnavailable("Live trading service not available".to_string()))
        }
    }
    
    /// Get account balances
    pub async fn get_balances(&self) -> Result<Vec<(String, f64)>, ServiceError> {
        let is_paper = *self.is_paper_trading.read().await;
        
        if is_paper {
            // Route to paper trading
            debug!("Routing get balances request to paper trading");
            self.paper_trading_service.get_balances().await
        } else if let Some(live_service) = &self.live_trading_service {
            // Route to live trading
            debug!("Routing get balances request to live trading");
            live_service.get_balances().await
        } else {
            // Live trading service not available
            error!("Live trading service not available");
            Err(ServiceError::ServiceUnavailable("Live trading service not available".to_string()))
        }
    }
    
    /// Get current positions
    pub async fn get_positions(&self) -> Result<Vec<(String, f64, f64)>, ServiceError> {
        let is_paper = *self.is_paper_trading.read().await;
        
        if is_paper {
            // Route to paper trading
            debug!("Routing get positions request to paper trading");
            self.paper_trading_service.get_positions().await
        } else if let Some(live_service) = &self.live_trading_service {
            // Route to live trading
            debug!("Routing get positions request to live trading");
            live_service.get_positions().await
        } else {
            // Live trading service not available
            error!("Live trading service not available");
            Err(ServiceError::ServiceUnavailable("Live trading service not available".to_string()))
        }
    }
    
    /// Toggle between paper trading and live trading
    pub async fn set_paper_trading(&self, enabled: bool) -> Result<(), ServiceError> {
        let mut mode = self.is_paper_trading.write().await;
        
        // Don't allow switching to live trading if live service is not available
        if !enabled && self.live_trading_service.is_none() {
            return Err(ServiceError::ServiceUnavailable("Live trading service not available".to_string()));
        }
        
        *mode = enabled;
        
        if enabled {
            info!("Switched to PAPER TRADING mode");
        } else {
            warn!("Switched to LIVE TRADING mode - REAL FUNDS WILL BE USED");
        }
        
        Ok(())
    }
    
    /// Get current trading mode
    pub async fn is_paper_trading(&self) -> bool {
        *self.is_paper_trading.read().await
    }
}
