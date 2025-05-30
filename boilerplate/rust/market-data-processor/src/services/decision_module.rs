use crate::models::signal::Signal;
use crate::services::order_execution::{OrderExecutionService, OrderRequest, OrderResponse, ServiceError};
use crate::utils::enhanced_config::EnhancedConfig;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::mpsc::{self, Receiver, Sender};
use tokio::time::{self, Duration};
use tracing::{debug, error, info, warn};

/// Decision output from the LLM decision engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionOutput {
    pub symbol: String,
    pub action: DecisionAction,
    pub confidence: f64,
    pub reasoning: String,
    pub timestamp: u64,
    pub risk_score: f64,
    pub position_size: f64,
}

/// Decision action type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DecisionAction {
    Buy,
    Sell,
    Hold,
}

impl std::fmt::Display for DecisionAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecisionAction::Buy => write!(f, "BUY"),
            DecisionAction::Sell => write!(f, "SELL"),
            DecisionAction::Hold => write!(f, "HOLD"),
        }
    }
}

/// LLM-based decision module
pub struct DecisionModule {
    config: Arc<EnhancedConfig>,
    signal_receiver: Receiver<Signal>,
    decision_sender: Sender<DecisionOutput>,
    order_execution: Arc<OrderExecutionService>,
}

impl DecisionModule {
    /// Create a new decision module
    pub fn new(
        config: Arc<EnhancedConfig>,
        signal_receiver: Receiver<Signal>,
        order_execution: Arc<OrderExecutionService>,
    ) -> (Self, Receiver<DecisionOutput>) {
        let (decision_sender, decision_receiver) = mpsc::channel(100);
        
        (
            Self {
                config,
                signal_receiver,
                decision_sender,
                order_execution,
            },
            decision_receiver
        )
    }
    
    /// Start the decision module
    pub async fn start(&mut self) {
        info!("Starting LLM-based decision module");
        
        // Process signals and generate decisions
        while let Some(signal) = self.signal_receiver.recv().await {
            debug!("Received signal: {:?}", signal);
            
            // In a real implementation, this would send the signal to the LLM
            // and receive a decision. For now, we'll simulate a decision.
            let decision = self.simulate_decision(signal).await;
            
            // Send decision to subscribers
            if let Err(e) = self.decision_sender.send(decision.clone()).await {
                error!("Failed to send decision: {}", e);
            }
            
            // Execute decision if confidence is high enough
            if decision.confidence >= 0.7 && decision.action != DecisionAction::Hold {
                self.execute_decision(decision).await;
            }
        }
    }
    
    /// Simulate a decision from the LLM (placeholder for real LLM integration)
    async fn simulate_decision(&self, signal: Signal) -> DecisionOutput {
        // In a real implementation, this would send the signal to the LLM
        // and receive a decision. For now, we'll create a simple decision
        // based on the signal.
        
        // Convert signal type to decision action
        let action = match signal.signal_type {
            crate::models::signal::SignalType::Buy => DecisionAction::Buy,
            crate::models::signal::SignalType::Sell => DecisionAction::Sell,
            _ => DecisionAction::Hold,
        };
        
        // Calculate confidence based on signal strength
        let confidence = signal.strength.as_f64();
        
        // Generate reasoning
        let reasoning = match action {
            DecisionAction::Buy => format!(
                "Buy signal detected from {} with strength {}. Technical indicators suggest upward momentum.",
                signal.source, confidence
            ),
            DecisionAction::Sell => format!(
                "Sell signal detected from {} with strength {}. Technical indicators suggest downward momentum.",
                signal.source, confidence
            ),
            DecisionAction::Hold => format!(
                "No clear signal detected from {}. Strength {} is insufficient for action.",
                signal.source, confidence
            ),
        };
        
        // Calculate risk score based on signal volatility
        let risk_score = 0.5; // Mid-level risk (0.0 to 1.0)
        
        // Calculate position size based on risk score and config
        let position_size = self.calculate_position_size(signal.symbol.as_str(), risk_score);
        
        DecisionOutput {
            symbol: signal.symbol,
            action,
            confidence,
            reasoning,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            risk_score,
            position_size,
        }
    }
    
    /// Calculate position size based on risk parameters
    fn calculate_position_size(&self, symbol: &str, risk_score: f64) -> f64 {
        // Base size from config
        let base_size = self.config.default_order_size;
        
        // Adjust based on risk score (lower risk = larger position)
        let risk_factor = 1.0 - (risk_score * 0.5); // 0.5 to 1.0
        
        // Ensure position size doesn't exceed max
        (base_size * risk_factor).min(self.config.max_position_size)
    }
    
    /// Execute a decision by placing an order
    async fn execute_decision(&self, decision: DecisionOutput) {
        info!("Executing decision: {} {} with confidence {:.2}", 
            decision.action, decision.symbol, decision.confidence);
        
        // Skip if action is Hold
        if decision.action == DecisionAction::Hold {
            return;
        }
        
        // Convert decision to order request
        let side = match decision.action {
            DecisionAction::Buy => crate::models::order::OrderSide::Buy,
            DecisionAction::Sell => crate::models::order::OrderSide::Sell,
            DecisionAction::Hold => return,
        };
        
        let order_request = OrderRequest {
            symbol: decision.symbol.clone(),
            side,
            order_type: crate::models::order::OrderType::Market,
            quantity: decision.position_size,
            price: None,
            time_in_force: None,
            client_order_id: None,
        };
        
        // Place order
        match self.order_execution.place_order(order_request).await {
            Ok(response) => {
                info!("Order placed successfully: {} (executed: {})", 
                    response.order_id, response.executed_qty.unwrap_or_default());
            },
            Err(e) => {
                error!("Failed to place order: {}", e);
            }
        }
    }
}
