use crate::models::order::Order;
use crate::models::position::Position;
use crate::services::decision_module::DecisionOutput;
use crate::services::order_execution::OrderExecutionService;
use crate::services::paper_trading::PaperTradingService;
use crate::services::signal_generator::SignalGenerator;
use crate::utils::enhanced_config::EnhancedConfig;
use actix_web::{get, post, web, HttpResponse, Responder};
use actix_web::middleware::Logger;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::collections::HashMap;
use tracing::{debug, error, info, warn};

/// Dashboard configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardConfig {
    pub port: u16,
    pub host: String,
    pub static_files_path: String,
    pub enable_cors: bool,
}

/// Trading dashboard API
#[derive(Debug, Clone)]
pub struct TradingDashboard {
    config: Arc<EnhancedConfig>,
    order_execution: Arc<OrderExecutionService>,
    paper_trading: Arc<PaperTradingService>,
    signal_generator: Arc<SignalGenerator>,
    // In a real implementation, these would be channels or databases for storing history
    decisions_history: Arc<web::Data<Vec<DecisionOutput>>>,
}

impl TradingDashboard {
    /// Create a new trading dashboard
    pub fn new(
        config: Arc<EnhancedConfig>,
        order_execution: Arc<OrderExecutionService>,
        paper_trading: Arc<PaperTradingService>,
        signal_generator: Arc<SignalGenerator>,
    ) -> Self {
        Self {
            config,
            order_execution,
            paper_trading,
            signal_generator,
            decisions_history: Arc::new(web::Data::new(Vec::new())),
        }
    }
    
    /// Start the dashboard server
    pub async fn start(&self, dashboard_config: DashboardConfig) -> std::io::Result<()> {
        let address = format!("{}:{}", dashboard_config.host, dashboard_config.port);
        info!("Starting trading dashboard on {}", address);
        
        // Clone references for the actix server
        let order_execution = self.order_execution.clone();
        let paper_trading = self.paper_trading.clone();
        let signal_generator = self.signal_generator.clone();
        let decisions_history = self.decisions_history.clone();
        let config = self.config.clone();
        
        // Create the actix server
        let server = actix_web::HttpServer::new(move || {
            let cors = if dashboard_config.enable_cors {
                actix_cors::Cors::default()
                    .allow_any_origin()
                    .allow_any_method()
                    .allow_any_header()
                    .max_age(3600)
            } else {
                actix_cors::Cors::default()
            };
            
            actix_web::App::new()
                .wrap(Logger::default())
                .wrap(cors)
                .app_data(web::Data::new(order_execution.clone()))
                .app_data(web::Data::new(paper_trading.clone()))
                .app_data(web::Data::new(signal_generator.clone()))
                .app_data(decisions_history.clone())
                .app_data(web::Data::new(config.clone()))
                .service(get_trading_mode)
                .service(set_trading_mode)
                .service(get_positions)
                .service(get_orders)
                .service(get_balances)
                .service(get_trading_pairs)
                .service(get_order_history)
                .service(get_performance)
                .service(place_order)
                .service(cancel_order)
                .service(actix_files::Files::new("/", dashboard_config.static_files_path)
                    .index_file("index.html"))
        });
        
        // Start the server
        server.bind(address)?.run().await
    }
    
    /// Register a new decision
    pub async fn register_decision(&self, decision: DecisionOutput) {
        // In a real implementation, this would store the decision in a database
        // For now, we'll just log it
        info!("Dashboard registered decision: {:?}", decision);
    }
}

/// Response type for trading mode
#[derive(Debug, Serialize, Deserialize)]
struct TradingModeResponse {
    mode: String,
    is_paper: bool,
}

/// Response type for positions
#[derive(Debug, Serialize, Deserialize)]
struct PositionsResponse {
    positions: Vec<PositionInfo>,
}

/// Position information
#[derive(Debug, Serialize, Deserialize)]
struct PositionInfo {
    symbol: String,
    direction: String,
    quantity: f64,
    entry_price: f64,
    current_price: Option<f64>,
    unrealized_pnl: Option<f64>,
    realized_pnl: f64,
    timestamp: u64,
}

/// Response type for orders
#[derive(Debug, Serialize, Deserialize)]
struct OrdersResponse {
    orders: Vec<OrderInfo>,
}

/// Order information
#[derive(Debug, Serialize, Deserialize)]
struct OrderInfo {
    id: String,
    symbol: String,
    side: String,
    order_type: String,
    quantity: f64,
    price: Option<f64>,
    status: String,
    executed_qty: f64,
    avg_execution_price: Option<f64>,
    timestamp: u64,
}

/// Response type for balances
#[derive(Debug, Serialize, Deserialize)]
struct BalancesResponse {
    balances: HashMap<String, f64>,
}

/// Response type for trading pairs
#[derive(Debug, Serialize, Deserialize)]
struct TradingPairsResponse {
    pairs: Vec<String>,
}

/// Request type for setting trading mode
#[derive(Debug, Serialize, Deserialize)]
struct SetTradingModeRequest {
    paper_trading: bool,
}

/// Request type for placing an order
#[derive(Debug, Serialize, Deserialize)]
struct PlaceOrderRequest {
    symbol: String,
    side: String,
    order_type: String,
    quantity: f64,
    price: Option<f64>,
}

/// Response type for performance metrics
#[derive(Debug, Serialize, Deserialize)]
struct PerformanceResponse {
    total_pnl: f64,
    win_rate: f64,
    average_win: f64,
    average_loss: f64,
    largest_win: f64,
    largest_loss: f64,
    sharpe_ratio: Option<f64>,
    drawdown: f64,
    metrics_by_pair: HashMap<String, PairPerformance>,
}

/// Performance metrics for a trading pair
#[derive(Debug, Serialize, Deserialize)]
struct PairPerformance {
    symbol: String,
    total_pnl: f64,
    win_rate: f64,
    trade_count: usize,
    average_holding_time: String,
}

/// Get current trading mode (paper or live)
#[get("/api/trading/mode")]
async fn get_trading_mode(
    order_execution: web::Data<Arc<OrderExecutionService>>,
) -> impl Responder {
    let is_paper = order_execution.is_paper_trading().await;
    
    HttpResponse::Ok().json(TradingModeResponse {
        mode: if is_paper { "paper".to_string() } else { "live".to_string() },
        is_paper,
    })
}

/// Set trading mode (paper or live)
#[post("/api/trading/mode")]
async fn set_trading_mode(
    order_execution: web::Data<Arc<OrderExecutionService>>,
    request: web::Json<SetTradingModeRequest>,
) -> impl Responder {
    match order_execution.set_paper_trading(request.paper_trading).await {
        Ok(_) => {
            let mode = if request.paper_trading { "paper" } else { "live" };
            info!("Trading mode set to: {}", mode);
            HttpResponse::Ok().json(TradingModeResponse {
                mode: mode.to_string(),
                is_paper: request.paper_trading,
            })
        },
        Err(e) => {
            error!("Failed to set trading mode: {}", e);
            HttpResponse::InternalServerError().body(format!("Failed to set trading mode: {}", e))
        }
    }
}

/// Get current positions
#[get("/api/trading/positions")]
async fn get_positions(
    paper_trading: web::Data<Arc<PaperTradingService>>,
) -> impl Responder {
    match paper_trading.get_open_positions().await {
        Ok(positions) => {
            let position_infos = positions.into_iter().map(|p| {
                PositionInfo {
                    symbol: p.symbol,
                    direction: p.direction.to_string(),
                    quantity: p.quantity,
                    entry_price: p.avg_entry_price,
                    current_price: p.last_price,
                    unrealized_pnl: p.unrealized_pnl(),
                    realized_pnl: p.realized_pnl,
                    timestamp: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_millis() as u64,
                }
            }).collect();
            
            HttpResponse::Ok().json(PositionsResponse {
                positions: position_infos,
            })
        },
        Err(e) => {
            error!("Failed to get positions: {}", e);
            HttpResponse::InternalServerError().body(format!("Failed to get positions: {}", e))
        }
    }
}

/// Get current orders
#[get("/api/trading/orders")]
async fn get_orders(
    paper_trading: web::Data<Arc<PaperTradingService>>,
) -> impl Responder {
    match paper_trading.get_open_orders().await {
        Ok(orders) => {
            let order_infos = orders.into_iter().map(|o| {
                OrderInfo {
                    id: o.id,
                    symbol: o.symbol,
                    side: o.side.to_string(),
                    order_type: o.order_type.to_string(),
                    quantity: o.quantity,
                    price: o.price,
                    status: o.status.to_string(),
                    executed_qty: o.executed_qty,
                    avg_execution_price: o.avg_execution_price,
                    timestamp: o.created_at.timestamp_millis() as u64,
                }
            }).collect();
            
            HttpResponse::Ok().json(OrdersResponse {
                orders: order_infos,
            })
        },
        Err(e) => {
            error!("Failed to get orders: {}", e);
            HttpResponse::InternalServerError().body(format!("Failed to get orders: {}", e))
        }
    }
}

/// Get account balances
#[get("/api/trading/balances")]
async fn get_balances(
    paper_trading: web::Data<Arc<PaperTradingService>>,
) -> impl Responder {
    match paper_trading.get_balances().await {
        Ok(balances) => {
            let balances_map: HashMap<String, f64> = balances.into_iter().collect();
            
            HttpResponse::Ok().json(BalancesResponse {
                balances: balances_map,
            })
        },
        Err(e) => {
            error!("Failed to get balances: {}", e);
            HttpResponse::InternalServerError().body(format!("Failed to get balances: {}", e))
        }
    }
}

/// Get available trading pairs
#[get("/api/trading/pairs")]
async fn get_trading_pairs(
    config: web::Data<Arc<EnhancedConfig>>,
) -> impl Responder {
    HttpResponse::Ok().json(TradingPairsResponse {
        pairs: config.trading_pairs.clone(),
    })
}

/// Get order history
#[get("/api/trading/history/orders")]
async fn get_order_history(
    paper_trading: web::Data<Arc<PaperTradingService>>,
) -> impl Responder {
    match paper_trading.get_order_history().await {
        Ok(orders) => {
            let order_infos = orders.into_iter().map(|o| {
                OrderInfo {
                    id: o.id,
                    symbol: o.symbol,
                    side: o.side.to_string(),
                    order_type: o.order_type.to_string(),
                    quantity: o.quantity,
                    price: o.price,
                    status: o.status.to_string(),
                    executed_qty: o.executed_qty,
                    avg_execution_price: o.avg_execution_price,
                    timestamp: o.created_at.timestamp_millis() as u64,
                }
            }).collect();
            
            HttpResponse::Ok().json(OrdersResponse {
                orders: order_infos,
            })
        },
        Err(e) => {
            error!("Failed to get order history: {}", e);
            HttpResponse::InternalServerError().body(format!("Failed to get order history: {}", e))
        }
    }
}

/// Get performance metrics
#[get("/api/trading/performance")]
async fn get_performance(
    paper_trading: web::Data<Arc<PaperTradingService>>,
    config: web::Data<Arc<EnhancedConfig>>,
) -> impl Responder {
    // In a real implementation, this would query the performance metrics
    // For now, we'll return placeholder data
    let mut metrics_by_pair = HashMap::new();
    
    for pair in &config.trading_pairs {
        metrics_by_pair.insert(pair.clone(), PairPerformance {
            symbol: pair.clone(),
            total_pnl: 0.0, // Placeholder
            win_rate: 0.0,  // Placeholder
            trade_count: 0, // Placeholder
            average_holding_time: "0h 0m".to_string(), // Placeholder
        });
    }
    
    HttpResponse::Ok().json(PerformanceResponse {
        total_pnl: 0.0, // Placeholder
        win_rate: 0.0,  // Placeholder
        average_win: 0.0, // Placeholder
        average_loss: 0.0, // Placeholder
        largest_win: 0.0, // Placeholder
        largest_loss: 0.0, // Placeholder
        sharpe_ratio: None, // Placeholder
        drawdown: 0.0, // Placeholder
        metrics_by_pair,
    })
}

/// Place a new order
#[post("/api/trading/orders")]
async fn place_order(
    order_execution: web::Data<Arc<OrderExecutionService>>,
    request: web::Json<PlaceOrderRequest>,
) -> impl Responder {
    // Convert request to internal format
    let side = match request.side.to_lowercase().as_str() {
        "buy" => crate::models::order::OrderSide::Buy,
        "sell" => crate::models::order::OrderSide::Sell,
        _ => {
            return HttpResponse::BadRequest().body(format!("Invalid order side: {}", request.side));
        }
    };
    
    let order_type = match request.order_type.to_lowercase().as_str() {
        "market" => crate::models::order::OrderType::Market,
        "limit" => crate::models::order::OrderType::Limit,
        _ => {
            return HttpResponse::BadRequest().body(format!("Invalid order type: {}", request.order_type));
        }
    };
    
    // Validate price for limit orders
    if order_type == crate::models::order::OrderType::Limit && request.price.is_none() {
        return HttpResponse::BadRequest().body("Limit orders require a price");
    }
    
    let order_request = crate::services::order_execution::OrderRequest {
        symbol: request.symbol.clone(),
        side,
        order_type,
        quantity: request.quantity,
        price: request.price,
        time_in_force: None,
        client_order_id: None,
    };
    
    // Place order
    match order_execution.place_order(order_request).await {
        Ok(response) => {
            HttpResponse::Ok().json(response)
        },
        Err(e) => {
            error!("Failed to place order: {}", e);
            HttpResponse::InternalServerError().body(format!("Failed to place order: {}", e))
        }
    }
}

/// Cancel an order
#[post("/api/trading/orders/{order_id}/cancel")]
async fn cancel_order(
    order_execution: web::Data<Arc<OrderExecutionService>>,
    path: web::Path<String>,
) -> impl Responder {
    let order_id = path.into_inner();
    
    match order_execution.cancel_order(&order_id).await {
        Ok(response) => {
            HttpResponse::Ok().json(response)
        },
        Err(e) => {
            error!("Failed to cancel order: {}", e);
            HttpResponse::InternalServerError().body(format!("Failed to cancel order: {}", e))
        }
    }
}
