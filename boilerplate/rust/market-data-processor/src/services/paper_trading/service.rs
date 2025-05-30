use crate::models::order::{Order, OrderSide, OrderStatus, OrderType, TimeInForce};
use crate::services::optimized_rest_client::OptimizedRestClient;
use crate::services::order_execution::{OrderRequest, OrderResponse, ServiceError};
use crate::services::paper_trading::matching_engine::{MatchingEngine, MatchingError};
use crate::services::paper_trading::virtual_account::{Trade, VirtualAccount, VirtualAccountError};
use crate::utils::enhanced_config::EnhancedConfig;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, Mutex};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// Default maximum slippage for paper trading
const DEFAULT_MAX_SLIPPAGE: f64 = 0.005; // 0.5%

/// Paper trading service
pub struct PaperTradingService {
    config: Arc<EnhancedConfig>,
    virtual_account: Arc<Mutex<VirtualAccount>>,
    matching_engine: Arc<MatchingEngine>,
    market_data_client: Arc<OptimizedRestClient>,
}

impl PaperTradingService {
    /// Create a new paper trading service
    pub fn new(
        config: Arc<EnhancedConfig>,
        market_data_client: Arc<OptimizedRestClient>,
    ) -> Self {
        // Initialize virtual account with balances
        let mut initial_balances = HashMap::new();
        
        // Add USDC balance
        initial_balances.insert("USDC".to_string(), 100000.0); // 100,000 USDC
        
        // Add balances for each trading pair
        for pair in &config.trading_pairs {
            let base_asset = pair.split("USDC").next().unwrap_or("BTC");
            initial_balances.insert(base_asset.to_string(), 1.0); // 1 unit of each base asset
        }
        
        let virtual_account = VirtualAccount::new(initial_balances);
        let matching_engine = MatchingEngine::new(market_data_client.clone(), DEFAULT_MAX_SLIPPAGE);
        
        Self {
            config,
            virtual_account: Arc::new(Mutex::new(virtual_account)),
            matching_engine: Arc::new(matching_engine),
            market_data_client,
        }
    }
    
    /// Create a new paper trading service with custom components (for testing)
    pub fn new_with_components(
        config: Arc<EnhancedConfig>,
        matching_engine: Arc<MatchingEngine>,
        virtual_account: Arc<Mutex<VirtualAccount>>,
    ) -> Self {
        // Create mock market data client if none provided
        let market_data_client = Arc::new(OptimizedRestClient::new(&config));
        
        Self {
            config,
            virtual_account,
            matching_engine,
            market_data_client,
        }
    }
    
    /// Place an order
    pub async fn place_order(&self, order_request: OrderRequest) -> Result<OrderResponse, ServiceError> {
        debug!("Processing paper trading order: {:?}", order_request);
        
        // Create internal order model
        let order = Self::create_order_from_request(order_request);
        
        // Lock virtual account
        let mut account = self.virtual_account.lock().await;
        
        // Place order in virtual account
        let order_id = match account.place_order(order.clone()) {
            Ok(id) => id,
            Err(e) => {
                error!("Failed to place order: {}", e);
                return Err(ServiceError::ValidationError(e.to_string()));
            }
        };
        
        // Process order with matching engine
        let trades = match self.matching_engine.process_order(&order, &mut account).await {
            Ok(trades) => trades,
            Err(e) => {
                error!("Failed to process order: {}", e);
                
                // Update order status to rejected if processing failed
                if let Err(status_err) = account.update_order_status(order_id.clone(), OrderStatus::Rejected) {
                    warn!("Failed to update order status: {}", status_err);
                }
                
                return Err(ServiceError::ExecutionError(e.to_string()));
            }
        };
        
        // Get updated order
        let updated_order = account.get_order(&order_id)
            .ok_or_else(|| ServiceError::ExecutionError("Order not found after processing".to_string()))?;
        
        // Log order execution
        if !trades.is_empty() {
            info!("Paper trading order executed: {} {:?} {} @ {:.2} (Status: {:?})",
                updated_order.symbol, updated_order.side, updated_order.executed_qty,
                updated_order.avg_execution_price.unwrap_or_default(), updated_order.status);
        } else {
            info!("Paper trading order placed but not executed: {} {:?} {} @ {:.2} (Status: {:?})",
                updated_order.symbol, updated_order.side, updated_order.quantity,
                updated_order.price.unwrap_or_default(), updated_order.status);
        }
        
        // Create response
        let response = OrderResponse {
            order_id: updated_order.id.clone(),
            status: updated_order.status.to_string(),
            message: None,
            executed_qty: Some(updated_order.executed_qty),
            executed_price: updated_order.avg_execution_price,
        };
        
        Ok(response)
    }
    
    /// Cancel an order
    pub async fn cancel_order(&self, order_id: &str) -> Result<OrderResponse, ServiceError> {
        debug!("Cancelling paper trading order: {}", order_id);
        
        // Lock virtual account
        let mut account = self.virtual_account.lock().await;
        
        // Get order
        let order = account.get_order(order_id)
            .ok_or_else(|| ServiceError::ValidationError(format!("Order not found: {}", order_id)))?;
        
        // Check if order can be canceled
        if !order.is_active() {
            return Err(ServiceError::ValidationError(
                format!("Order cannot be canceled: status is {:?}", order.status)
            ));
        }
        
        // Update order status
        match account.update_order_status(order_id.to_string(), OrderStatus::Canceled) {
            Ok(_) => {
                info!("Paper trading order canceled: {}", order_id);
                
                // Get updated order
                let updated_order = account.get_order(order_id)
                    .ok_or_else(|| ServiceError::ExecutionError("Order not found after cancellation".to_string()))?;
                
                // Create response
                let response = OrderResponse {
                    order_id: updated_order.id.clone(),
                    status: updated_order.status.to_string(),
                    message: Some("Order canceled".to_string()),
                    executed_qty: Some(updated_order.executed_qty),
                    executed_price: updated_order.avg_execution_price,
                };
                
                Ok(response)
            },
            Err(e) => {
                error!("Failed to cancel order: {}", e);
                Err(ServiceError::ExecutionError(e.to_string()))
            }
        }
    }
    
    /// Get order details
    pub async fn get_order(&self, order_id: &str) -> Result<Order, ServiceError> {
        debug!("Getting paper trading order: {}", order_id);
        
        // Lock virtual account
        let account = self.virtual_account.lock().await;
        
        // Get order
        account.get_order(order_id)
            .ok_or_else(|| ServiceError::ValidationError(format!("Order not found: {}", order_id)))
    }
    
    /// Get account balances
    pub async fn get_balances(&self) -> Result<Vec<(String, f64)>, ServiceError> {
        debug!("Getting paper trading balances");
        
        // Lock virtual account
        let account = self.virtual_account.lock().await;
        
        // Get balances
        Ok(account.get_balances())
    }
    
    /// Get positions
    pub async fn get_positions(&self) -> Result<Vec<(String, f64, f64)>, ServiceError> {
        debug!("Getting paper trading positions");
        
        // Lock virtual account
        let account = self.virtual_account.lock().await;
        
        // Get positions and format as (symbol, quantity, entry_price)
        let positions = account.get_positions();
        let formatted_positions = positions.into_iter()
            .filter(|p| p.is_open())
            .map(|p| (p.symbol.clone(), p.quantity, p.avg_entry_price))
            .collect();
        
        Ok(formatted_positions)
    }
    
    /// Get order history
    pub async fn get_order_history(&self) -> Result<Vec<Order>, ServiceError> {
        debug!("Getting paper trading order history");
        
        // Lock virtual account
        let account = self.virtual_account.lock().await;
        
        // Get completed orders
        Ok(account.get_completed_orders())
    }
    
    /// Get trade history
    pub async fn get_trade_history(&self) -> Result<Vec<Trade>, ServiceError> {
        debug!("Getting paper trading trade history");
        
        // Lock virtual account
        let account = self.virtual_account.lock().await;
        
        // Get trades
        Ok(account.get_trades())
    }
    
    /// Update positions with current market prices
    pub async fn update_positions(&self) -> Result<(), ServiceError> {
        debug!("Updating paper trading positions");
        
        // Lock virtual account
        let mut account = self.virtual_account.lock().await;
        
        // Get positions
        let positions = account.get_positions();
        
        // Update each position with current market price
        for position in positions {
            if !position.is_open() {
                continue;
            }
            
            // Get current price from ticker
            match self.market_data_client.get_ticker(&position.symbol).await {
                Ok(ticker) => {
                    let price = ticker["lastPrice"]
                        .as_str()
                        .and_then(|p| p.parse::<f64>().ok())
                        .unwrap_or_default();
                        
                    if price > 0.0 {
                        if let Err(e) = account.update_position_price(&position.symbol, price) {
                            warn!("Failed to update position price for {}: {}", position.symbol, e);
                        }
                    }
                },
                Err(e) => {
                    warn!("Failed to get ticker for {}: {}", position.symbol, e);
                }
            }
        }
        
        Ok(())
    }
    
    /// Create an internal order model from a request
    fn create_order_from_request(request: OrderRequest) -> Order {
        let now = chrono::Utc::now();
        
        // Generate a random latency between 50ms and 200ms
        let latency_ms = rand::random::<u64>() % 150 + 50;
        
        // Generate a random slippage between 0.01% and 0.1%
        let slippage_factor = rand::random::<f64>() * 0.0009 + 0.0001;
        
        Order {
            id: Uuid::new_v4().to_string(),
            client_order_id: request.client_order_id,
            symbol: request.symbol,
            side: request.side,
            order_type: request.order_type,
            quantity: request.quantity,
            price: request.price,
            stop_price: None,
            time_in_force: request.time_in_force.unwrap_or(TimeInForce::GoodTillCancel),
            iceberg_qty: None,
            created_at: now,
            updated_at: now,
            status: OrderStatus::Created,
            executed_qty: 0.0,
            avg_execution_price: None,
            fills: Vec::new(),
            cumulative_commission: 0.0,
            expected_latency_ms: latency_ms,
            slippage_factor,
            scheduled_execution_time: Some(now + chrono::Duration::milliseconds(latency_ms as i64)),
        }
    }
    
    /// Start the paper trading service
    pub async fn start(&self) -> Result<(), ServiceError> {
        info!("Starting paper trading service");
        
        // Start position update loop
        let service_clone = self.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(5));
            
            loop {
                interval.tick().await;
                
                // Update positions with current market prices
                if let Err(e) = service_clone.update_positions().await {
                    warn!("Failed to update positions: {}", e);
                }
            }
        });
        
        Ok(())
    }
}

impl Clone for PaperTradingService {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            virtual_account: self.virtual_account.clone(),
            matching_engine: self.matching_engine.clone(),
            market_data_client: self.market_data_client.clone(),
        }
    }
}
