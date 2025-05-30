use std::sync::Arc;
use warp::{Filter, Rejection, Reply};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use crate::config::enhanced_config::EnhancedConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentStatus {
    pub status: String,
    pub last_updated: DateTime<Utc>,
    pub details: Option<String>,
    pub metrics: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStatus {
    pub overall_status: String,
    pub timestamp: DateTime<Utc>,
    pub components: HashMap<String, ComponentStatus>,
    pub trading_stats: TradingStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingStats {
    pub total_trades: u64,
    pub successful_trades: u64,
    pub failed_trades: u64,
    pub total_volume: f64,
    pub current_balance: HashMap<String, f64>,
    pub profit_loss: f64,
    pub profit_loss_percent: f64,
    pub drawdown: f64,
    pub drawdown_percent: f64,
}

pub struct StatusService {
    config: Arc<EnhancedConfig>,
    system_status: SystemStatus,
}

impl StatusService {
    pub fn new(config: Arc<EnhancedConfig>) -> Self {
        let mut components = HashMap::new();
        
        // Initialize with basic components
        components.insert("market_data".to_string(), ComponentStatus {
            status: "OK".to_string(),
            last_updated: Utc::now(),
            details: None,
            metrics: HashMap::new(),
        });
        
        components.insert("paper_trading".to_string(), ComponentStatus {
            status: "OK".to_string(),
            last_updated: Utc::now(),
            details: None,
            metrics: HashMap::new(),
        });
        
        components.insert("order_execution".to_string(), ComponentStatus {
            status: "OK".to_string(),
            last_updated: Utc::now(),
            details: None,
            metrics: HashMap::new(),
        });
        
        // Initial trading stats
        let trading_stats = TradingStats {
            total_trades: 0,
            successful_trades: 0,
            failed_trades: 0,
            total_volume: 0.0,
            current_balance: config.paper_trading_initial_balances.clone(),
            profit_loss: 0.0,
            profit_loss_percent: 0.0,
            drawdown: 0.0,
            drawdown_percent: 0.0,
        };
        
        // Initial system status
        let system_status = SystemStatus {
            overall_status: "OK".to_string(),
            timestamp: Utc::now(),
            components,
            trading_stats,
        };
        
        Self {
            config,
            system_status,
        }
    }
    
    pub fn update_component_status(&mut self, component: &str, status: &str, details: Option<String>, metrics: HashMap<String, f64>) {
        if let Some(component_status) = self.system_status.components.get_mut(component) {
            component_status.status = status.to_string();
            component_status.last_updated = Utc::now();
            component_status.details = details;
            component_status.metrics = metrics;
            
            // Update overall status if any component is not OK
            if status != "OK" && self.system_status.overall_status == "OK" {
                self.system_status.overall_status = "DEGRADED".to_string();
            }
            
            // Update timestamp
            self.system_status.timestamp = Utc::now();
        }
    }
    
    pub fn update_trading_stats(&mut self, stats: TradingStats) {
        self.system_status.trading_stats = stats;
        self.system_status.timestamp = Utc::now();
    }
    
    pub fn get_system_status(&self) -> SystemStatus {
        self.system_status.clone()
    }
    
    pub fn health_check(&self) -> bool {
        self.system_status.overall_status == "OK"
    }
    
    // Status API routes
    pub fn routes(&self) -> impl Filter<Extract = impl Reply, Error = Rejection> + Clone {
        let status_service = Arc::new(std::sync::RwLock::new(self.clone()));
        
        let status_route = warp::path("status")
            .and(warp::get())
            .and(with_status_service(status_service.clone()))
            .and_then(get_system_status);
            
        let health_route = warp::path("health")
            .and(warp::get())
            .and(with_status_service(status_service.clone()))
            .and_then(health_check);
            
        let metrics_route = warp::path("metrics")
            .and(warp::get())
            .and(with_status_service(status_service.clone()))
            .and_then(get_prometheus_metrics);
            
        status_route.or(health_route).or(metrics_route)
    }
}

fn with_status_service(
    status_service: Arc<std::sync::RwLock<StatusService>>,
) -> impl Filter<Extract = (Arc<std::sync::RwLock<StatusService>>,), Error = std::convert::Infallible> + Clone {
    warp::any().map(move || status_service.clone())
}

async fn get_system_status(
    status_service: Arc<std::sync::RwLock<StatusService>>,
) -> Result<impl Reply, Rejection> {
    let status = status_service.read().unwrap().get_system_status();
    Ok(warp::reply::json(&status))
}

async fn health_check(
    status_service: Arc<std::sync::RwLock<StatusService>>,
) -> Result<impl Reply, Rejection> {
    let healthy = status_service.read().unwrap().health_check();
    
    if healthy {
        Ok(warp::reply::with_status(
            "OK",
            warp::http::StatusCode::OK,
        ))
    } else {
        Ok(warp::reply::with_status(
            "Service Degraded",
            warp::http::StatusCode::SERVICE_UNAVAILABLE,
        ))
    }
}

async fn get_prometheus_metrics(
    status_service: Arc<std::sync::RwLock<StatusService>>,
) -> Result<impl Reply, Rejection> {
    let status = status_service.read().unwrap().get_system_status();
    
    // Generate Prometheus metrics format
    let mut metrics = String::new();
    
    // Overall system status
    metrics.push_str(&format!("# HELP system_status Overall system status (0=OK, 1=DEGRADED, 2=ERROR)\n"));
    metrics.push_str(&format!("# TYPE system_status gauge\n"));
    let status_value = match status.overall_status.as_str() {
        "OK" => 0,
        "DEGRADED" => 1,
        _ => 2,
    };
    metrics.push_str(&format!("system_status {}\n", status_value));
    
    // Trading stats
    metrics.push_str(&format!("# HELP trading_total_trades Total number of trades executed\n"));
    metrics.push_str(&format!("# TYPE trading_total_trades counter\n"));
    metrics.push_str(&format!("trading_total_trades {}\n", status.trading_stats.total_trades));
    
    metrics.push_str(&format!("# HELP trading_successful_trades Number of successful trades\n"));
    metrics.push_str(&format!("# TYPE trading_successful_trades counter\n"));
    metrics.push_str(&format!("trading_successful_trades {}\n", status.trading_stats.successful_trades));
    
    metrics.push_str(&format!("# HELP trading_failed_trades Number of failed trades\n"));
    metrics.push_str(&format!("# TYPE trading_failed_trades counter\n"));
    metrics.push_str(&format!("trading_failed_trades {}\n", status.trading_stats.failed_trades));
    
    metrics.push_str(&format!("# HELP trading_total_volume Total trading volume\n"));
    metrics.push_str(&format!("# TYPE trading_total_volume counter\n"));
    metrics.push_str(&format!("trading_total_volume {}\n", status.trading_stats.total_volume));
    
    metrics.push_str(&format!("# HELP trading_profit_loss_percent Profit/loss percentage\n"));
    metrics.push_str(&format!("# TYPE trading_profit_loss_percent gauge\n"));
    metrics.push_str(&format!("trading_profit_loss_percent {}\n", status.trading_stats.profit_loss_percent));
    
    metrics.push_str(&format!("# HELP trading_drawdown_percent Current drawdown percentage\n"));
    metrics.push_str(&format!("# TYPE trading_drawdown_percent gauge\n"));
    metrics.push_str(&format!("trading_drawdown_percent {}\n", status.trading_stats.drawdown_percent));
    
    // Component-specific metrics
    for (component_name, component) in status.components.iter() {
        for (metric_name, metric_value) in component.metrics.iter() {
            metrics.push_str(&format!("# HELP {}_{}_{} {} metric for {} component\n", 
                component_name, metric_name, "value", metric_name, component_name));
            metrics.push_str(&format!("# TYPE {}_{}_{} gauge\n", 
                component_name, metric_name, "value"));
            metrics.push_str(&format!("{}_{}_{} {}\n", 
                component_name, metric_name, "value", metric_value));
        }
    }
    
    Ok(warp::reply::with_header(
        metrics,
        "content-type",
        "text/plain; version=0.0.4",
    ))
}

impl Clone for StatusService {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            system_status: self.system_status.clone(),
        }
    }
}
