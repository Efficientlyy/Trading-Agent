use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub id: u64,
    pub symbol: String,
    pub price: f64,
    pub quantity: f64,
    pub buyer_order_id: u64,
    pub seller_order_id: u64,
    pub timestamp: u64,
    pub is_buyer_maker: bool,
}
