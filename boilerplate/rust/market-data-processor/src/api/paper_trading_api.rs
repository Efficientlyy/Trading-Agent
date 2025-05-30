use std::convert::Infallible;
use std::sync::Arc;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use warp::{Filter, Rejection, Reply};
use warp::reply::json;
use tracing::{debug, error, info};

use crate::models::order::{Order, OrderRequest, OrderResponse, OrderType, Side};
use crate::services::paper_trading::service::PaperTradingService;
use crate::config::EnhancedConfig;

// API response models
#[derive(Debug, Serialize, Deserialize)]
pub struct AccountDataResponse {
    pub balances: Vec<BalanceInfo>,
    pub total_value: f64,
    pub pnl: f64,
    pub pnl_percentage: f64,
    pub performance: PerformanceData,
    pub active_orders: Vec<ActiveOrderInfo>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BalanceInfo {
    pub asset: String,
    pub free: f64,
    pub locked: f64,
    pub usd_value: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PerformanceData {
    pub profit_loss: f64,
    pub win_rate: f64,
    pub max_drawdown: f64,
    pub sharpe_ratio: f64,
    pub total_trades: u32,
    pub successful_trades: u32,
    pub average_profit_per_trade: f64,
    pub average_trade_time: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ActiveOrderInfo {
    pub id: String,
    pub symbol: String,
    pub side: String,
    pub type_field: String,
    pub price: Option<f64>,
    pub quantity: f64,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MarketDataResponse {
    pub prices: Vec<PriceInfo>,
    pub last_updated: DateTime<Utc>,
    pub available_symbols: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PriceInfo {
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    pub price: f64,
    pub volume: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TradeInfo {
    pub id: String,
    pub symbol: String,
    pub side: String,
    pub price: f64,
    pub quantity: f64,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OrderHistoryResponse {
    pub orders: Vec<OrderHistoryInfo>,
    pub total: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OrderHistoryInfo {
    pub id: String,
    pub symbol: String,
    pub side: String,
    pub type_field: String,
    pub price: Option<f64>,
    pub quantity: f64,
    pub status: String,
    pub timestamp: DateTime<Utc>,
    pub fill_price: Option<f64>,
    pub fill_quantity: f64,
    pub fee: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PaperTradingSettingsResponse {
    pub initial_balances: std::collections::HashMap<String, f64>,
    pub trading_pairs: Vec<String>,
    pub max_position_size: f64,
    pub default_order_size: f64,
    pub max_drawdown_percent: f64,
    pub slippage_model: String,
    pub latency_model: String,
    pub trading_fees: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: String,
}

// API request models
#[derive(Debug, Serialize, Deserialize)]
pub struct PlaceOrderRequest {
    pub symbol: String,
    pub side: String,
    pub type_field: String,
    pub quantity: f64,
    pub price: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OrderHistoryRequest {
    pub symbol: Option<String>,
    pub status: Option<String>,
    pub start_date: Option<i64>,
    pub end_date: Option<i64>,
    pub page: Option<usize>,
    pub limit: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct UpdateSettingsRequest {
    pub initial_balances: std::collections::HashMap<String, f64>,
    pub trading_pairs: Vec<String>,
    pub max_position_size: f64,
    pub default_order_size: f64,
    pub max_drawdown_percent: f64,
    pub slippage_model: String,
    pub latency_model: String,
    pub trading_fees: f64,
}

/// Create paper trading API routes
pub fn paper_trading_routes(
    paper_trading_service: Arc<PaperTradingService>,
    config: Arc<EnhancedConfig>,
) -> impl Filter<Extract = impl Reply, Error = Rejection> + Clone {
    let service_clone = paper_trading_service.clone();
    let config_clone = config.clone();
    
    // GET /api/account - Get account data
    let account_route = warp::path!("api" / "account")
        .and(warp::get())
        .and(with_paper_trading_service(paper_trading_service.clone()))
        .and_then(handle_get_account);
    
    // GET /api/market/data - Get market data
    let market_data_route = warp::path!("api" / "market" / "data")
        .and(warp::get())
        .and(with_paper_trading_service(paper_trading_service.clone()))
        .and_then(handle_get_market_data);
    
    // GET /api/trades/recent - Get recent trades
    let recent_trades_route = warp::path!("api" / "trades" / "recent")
        .and(warp::get())
        .and(with_paper_trading_service(paper_trading_service.clone()))
        .and_then(handle_get_recent_trades);
    
    // GET /api/orders/history - Get order history
    let order_history_route = warp::path!("api" / "orders" / "history")
        .and(warp::get())
        .and(warp::query::<OrderHistoryRequest>())
        .and(with_paper_trading_service(paper_trading_service.clone()))
        .and_then(handle_get_order_history);
    
    // POST /api/orders - Place order
    let place_order_route = warp::path!("api" / "orders")
        .and(warp::post())
        .and(warp::body::json())
        .and(with_paper_trading_service(paper_trading_service.clone()))
        .and_then(handle_place_order);
    
    // DELETE /api/orders/:id - Cancel order
    let cancel_order_route = warp::path!("api" / "orders" / String)
        .and(warp::delete())
        .and(with_paper_trading_service(paper_trading_service.clone()))
        .and_then(handle_cancel_order);
    
    // GET /api/settings/paper-trading - Get paper trading settings
    let get_settings_route = warp::path!("api" / "settings" / "paper-trading")
        .and(warp::get())
        .and(with_config(config_clone.clone()))
        .and_then(handle_get_settings);
    
    // PUT /api/settings/paper-trading - Update paper trading settings
    let update_settings_route = warp::path!("api" / "settings" / "paper-trading")
        .and(warp::put())
        .and(warp::body::json())
        .and(with_config(config_clone))
        .and(with_paper_trading_service(service_clone))
        .and_then(handle_update_settings);
    
    // POST /api/account/reset - Reset paper trading account
    let reset_account_route = warp::path!("api" / "account" / "reset")
        .and(warp::post())
        .and(with_paper_trading_service(paper_trading_service))
        .and_then(handle_reset_account);
    
    // Combine all routes
    account_route
        .or(market_data_route)
        .or(recent_trades_route)
        .or(order_history_route)
        .or(place_order_route)
        .or(cancel_order_route)
        .or(get_settings_route)
        .or(update_settings_route)
        .or(reset_account_route)
}

// Helper functions to inject dependencies into route handlers
fn with_paper_trading_service(service: Arc<PaperTradingService>) -> impl Filter<Extract = (Arc<PaperTradingService>,), Error = Infallible> + Clone {
    warp::any().map(move || service.clone())
}

fn with_config(config: Arc<EnhancedConfig>) -> impl Filter<Extract = (Arc<EnhancedConfig>,), Error = Infallible> + Clone {
    warp::any().map(move || config.clone())
}

// Handler implementations
async fn handle_get_account(service: Arc<PaperTradingService>) -> Result<impl Reply, Rejection> {
    debug!("Handling GET /api/account request");
    
    // Get account data from service
    match service.get_account_data().await {
        Ok(account_data) => {
            // Transform to API response format
            let response = AccountDataResponse {
                balances: account_data.balances.iter().map(|(asset, balance)| {
                    BalanceInfo {
                        asset: asset.clone(),
                        free: balance.free,
                        locked: balance.locked,
                        usd_value: balance.usd_value,
                    }
                }).collect(),
                total_value: account_data.total_value,
                pnl: account_data.pnl,
                pnl_percentage: account_data.pnl_percentage,
                performance: PerformanceData {
                    profit_loss: account_data.performance.profit_loss,
                    win_rate: account_data.performance.win_rate,
                    max_drawdown: account_data.performance.max_drawdown,
                    sharpe_ratio: account_data.performance.sharpe_ratio,
                    total_trades: account_data.performance.total_trades,
                    successful_trades: account_data.performance.successful_trades,
                    average_profit_per_trade: account_data.performance.average_profit_per_trade,
                    average_trade_time: account_data.performance.average_trade_time,
                },
                active_orders: account_data.active_orders.iter().map(|order| {
                    ActiveOrderInfo {
                        id: order.id.clone(),
                        symbol: order.symbol.clone(),
                        side: format!("{:?}", order.side),
                        type_field: format!("{:?}", order.order_type),
                        price: order.price,
                        quantity: order.quantity,
                        timestamp: order.timestamp,
                    }
                }).collect(),
            };
            
            Ok(json(&response))
        },
        Err(err) => {
            error!("Error getting account data: {}", err);
            let error_response = ErrorResponse {
                error: err.to_string(),
            };
            Ok(warp::reply::with_status(
                json(&error_response),
                warp::http::StatusCode::INTERNAL_SERVER_ERROR,
            ))
        }
    }
}

async fn handle_get_market_data(service: Arc<PaperTradingService>) -> Result<impl Reply, Rejection> {
    debug!("Handling GET /api/market/data request");
    
    // Get market data from service
    match service.get_market_data().await {
        Ok(market_data) => {
            // Transform to API response format
            let response = MarketDataResponse {
                prices: market_data.prices.iter().map(|price_point| {
                    PriceInfo {
                        symbol: price_point.symbol.clone(),
                        timestamp: price_point.timestamp,
                        price: price_point.price,
                        volume: price_point.volume,
                    }
                }).collect(),
                last_updated: market_data.last_updated,
                available_symbols: market_data.available_symbols,
            };
            
            Ok(json(&response))
        },
        Err(err) => {
            error!("Error getting market data: {}", err);
            let error_response = ErrorResponse {
                error: err.to_string(),
            };
            Ok(warp::reply::with_status(
                json(&error_response),
                warp::http::StatusCode::INTERNAL_SERVER_ERROR,
            ))
        }
    }
}

async fn handle_get_recent_trades(service: Arc<PaperTradingService>) -> Result<impl Reply, Rejection> {
    debug!("Handling GET /api/trades/recent request");
    
    // Get recent trades from service
    match service.get_recent_trades().await {
        Ok(trades) => {
            // Transform to API response format
            let response: Vec<TradeInfo> = trades.iter().map(|trade| {
                TradeInfo {
                    id: trade.id.clone(),
                    symbol: trade.symbol.clone(),
                    side: format!("{:?}", trade.side),
                    price: trade.price,
                    quantity: trade.quantity,
                    timestamp: trade.timestamp,
                }
            }).collect();
            
            Ok(json(&response))
        },
        Err(err) => {
            error!("Error getting recent trades: {}", err);
            let error_response = ErrorResponse {
                error: err.to_string(),
            };
            Ok(warp::reply::with_status(
                json(&error_response),
                warp::http::StatusCode::INTERNAL_SERVER_ERROR,
            ))
        }
    }
}

async fn handle_get_order_history(
    query: OrderHistoryRequest,
    service: Arc<PaperTradingService>
) -> Result<impl Reply, Rejection> {
    debug!("Handling GET /api/orders/history request with query: {:?}", query);
    
    // Get order history from service
    match service.get_order_history(
        query.symbol.as_deref(),
        query.status.as_deref(),
        query.start_date.map(|ts| DateTime::<Utc>::from_timestamp(ts / 1000, 0).unwrap_or_default()),
        query.end_date.map(|ts| DateTime::<Utc>::from_timestamp(ts / 1000, 0).unwrap_or_default()),
        query.page.unwrap_or(1),
        query.limit.unwrap_or(10),
    ).await {
        Ok((orders, total)) => {
            // Transform to API response format
            let response = OrderHistoryResponse {
                orders: orders.iter().map(|order| {
                    OrderHistoryInfo {
                        id: order.id.clone(),
                        symbol: order.symbol.clone(),
                        side: format!("{:?}", order.side),
                        type_field: format!("{:?}", order.order_type),
                        price: order.price,
                        quantity: order.quantity,
                        status: format!("{:?}", order.status),
                        timestamp: order.timestamp,
                        fill_price: order.fill_price,
                        fill_quantity: order.fill_quantity,
                        fee: order.fee,
                    }
                }).collect(),
                total,
            };
            
            Ok(json(&response))
        },
        Err(err) => {
            error!("Error getting order history: {}", err);
            let error_response = ErrorResponse {
                error: err.to_string(),
            };
            Ok(warp::reply::with_status(
                json(&error_response),
                warp::http::StatusCode::INTERNAL_SERVER_ERROR,
            ))
        }
    }
}

async fn handle_place_order(
    order_req: PlaceOrderRequest,
    service: Arc<PaperTradingService>
) -> Result<impl Reply, Rejection> {
    debug!("Handling POST /api/orders request with body: {:?}", order_req);
    
    // Convert API request to internal order request
    let side = match order_req.side.to_uppercase().as_str() {
        "BUY" => Side::Buy,
        "SELL" => Side::Sell,
        _ => {
            let error_response = ErrorResponse {
                error: format!("Invalid order side: {}", order_req.side),
            };
            return Ok(warp::reply::with_status(
                json(&error_response),
                warp::http::StatusCode::BAD_REQUEST,
            ));
        }
    };
    
    let order_type = match order_req.type_field.to_uppercase().as_str() {
        "LIMIT" => OrderType::Limit,
        "MARKET" => OrderType::Market,
        "STOP_LOSS" => OrderType::StopLoss,
        "STOP_LIMIT" => OrderType::StopLimit,
        _ => {
            let error_response = ErrorResponse {
                error: format!("Invalid order type: {}", order_req.type_field),
            };
            return Ok(warp::reply::with_status(
                json(&error_response),
                warp::http::StatusCode::BAD_REQUEST,
            ));
        }
    };
    
    // Validate that limit orders have a price
    if order_type == OrderType::Limit && order_req.price.is_none() {
        let error_response = ErrorResponse {
            error: "Limit orders must specify a price".to_string(),
        };
        return Ok(warp::reply::with_status(
            json(&error_response),
            warp::http::StatusCode::BAD_REQUEST,
        ));
    }
    
    let order_request = OrderRequest {
        symbol: order_req.symbol,
        side,
        order_type,
        quantity: order_req.quantity,
        price: order_req.price,
    };
    
    // Place order via service
    match service.place_order(order_request).await {
        Ok(response) => {
            Ok(json(&response))
        },
        Err(err) => {
            error!("Error placing order: {}", err);
            let error_response = ErrorResponse {
                error: err.to_string(),
            };
            Ok(warp::reply::with_status(
                json(&error_response),
                warp::http::StatusCode::INTERNAL_SERVER_ERROR,
            ))
        }
    }
}

async fn handle_cancel_order(
    order_id: String,
    service: Arc<PaperTradingService>
) -> Result<impl Reply, Rejection> {
    debug!("Handling DELETE /api/orders/{} request", order_id);
    
    // Cancel order via service
    match service.cancel_order(&order_id).await {
        Ok(_) => {
            Ok(json(&serde_json::json!({
                "success": true,
                "message": format!("Order {} canceled successfully", order_id)
            })))
        },
        Err(err) => {
            error!("Error canceling order: {}", err);
            let error_response = ErrorResponse {
                error: err.to_string(),
            };
            Ok(warp::reply::with_status(
                json(&error_response),
                warp::http::StatusCode::INTERNAL_SERVER_ERROR,
            ))
        }
    }
}

async fn handle_get_settings(config: Arc<EnhancedConfig>) -> Result<impl Reply, Rejection> {
    debug!("Handling GET /api/settings/paper-trading request");
    
    // Extract paper trading settings from config
    let settings = PaperTradingSettingsResponse {
        initial_balances: config.paper_trading_initial_balances.clone(),
        trading_pairs: config.trading_pairs.clone(),
        max_position_size: config.max_position_size,
        default_order_size: config.default_order_size,
        max_drawdown_percent: config.max_drawdown_percent,
        slippage_model: config.paper_trading_slippage_model.clone(),
        latency_model: config.paper_trading_latency_model.clone(),
        trading_fees: config.paper_trading_fee_rate,
    };
    
    Ok(json(&settings))
}

async fn handle_update_settings(
    settings: UpdateSettingsRequest,
    config: Arc<EnhancedConfig>,
    service: Arc<PaperTradingService>
) -> Result<impl Reply, Rejection> {
    debug!("Handling PUT /api/settings/paper-trading request with body: {:?}", settings);
    
    // Update config with new settings
    // Note: In a real implementation, you would update the config and persist it
    // Here we're just simulating the update
    
    // Update the service with new settings
    match service.update_settings(
        settings.initial_balances,
        settings.trading_pairs,
        settings.max_position_size,
        settings.default_order_size,
        settings.max_drawdown_percent,
        settings.slippage_model,
        settings.latency_model,
        settings.trading_fees,
    ).await {
        Ok(_) => {
            Ok(json(&serde_json::json!({
                "success": true,
                "message": "Paper trading settings updated successfully"
            })))
        },
        Err(err) => {
            error!("Error updating settings: {}", err);
            let error_response = ErrorResponse {
                error: err.to_string(),
            };
            Ok(warp::reply::with_status(
                json(&error_response),
                warp::http::StatusCode::INTERNAL_SERVER_ERROR,
            ))
        }
    }
}

async fn handle_reset_account(service: Arc<PaperTradingService>) -> Result<impl Reply, Rejection> {
    debug!("Handling POST /api/account/reset request");
    
    // Reset account via service
    match service.reset_account().await {
        Ok(_) => {
            Ok(json(&serde_json::json!({
                "success": true,
                "message": "Paper trading account reset successfully"
            })))
        },
        Err(err) => {
            error!("Error resetting account: {}", err);
            let error_response = ErrorResponse {
                error: err.to_string(),
            };
            Ok(warp::reply::with_status(
                json(&error_response),
                warp::http::StatusCode::INTERNAL_SERVER_ERROR,
            ))
        }
    }
}
