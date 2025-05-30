use crate::models::{OrderBook, Ticker, Trade};
use crate::utils::{config::Config, metrics, Result};
use hmac::{Hmac, Mac};
use reqwest::{Client, RequestBuilder};
use sha2::Sha256;
use std::time::{SystemTime, UNIX_EPOCH};
use std::sync::Arc;

type HmacSha256 = Hmac<Sha256>;

/// MEXC REST API client
pub struct MexcClient {
    /// HTTP client
    client: Client,
    
    /// Configuration
    config: Arc<Config>,
}

impl MexcClient {
    /// Create a new MEXC API client
    pub fn new(config: &Arc<Config>) -> Self {
        Self {
            client: Client::new(),
            config: Arc::clone(config),
        }
    }
    
    /// Get ticker for a symbol
    pub async fn get_ticker(&self, symbol: &str) -> Result<Ticker> {
        let start_time = SystemTime::now();
        let url = format!("{}/api/v3/ticker/24hr", self.config.rest_api_url);
        
        let response = self.client
            .get(&url)
            .query(&[("symbol", symbol)])
            .send()
            .await?;
            
        let duration = SystemTime::now().duration_since(start_time).unwrap();
        metrics::record_api_request(duration.as_secs_f64());
        
        if !response.status().is_success() {
            metrics::record_api_error();
            return Err(format!("API error: {}", response.status()).into());
        }
        
        let data: serde_json::Value = response.json().await?;
        
        let ticker = Ticker::new(
            symbol.to_string(),
            data["lastPrice"].as_str().unwrap_or("0").parse().unwrap_or(0.0),
            data["volume"].as_str().unwrap_or("0").parse().unwrap_or(0.0),
            data["highPrice"].as_str().unwrap_or("0").parse().unwrap_or(0.0),
            data["lowPrice"].as_str().unwrap_or("0").parse().unwrap_or(0.0),
            data["closeTime"].as_u64().unwrap_or_else(|| {
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as u64
            }),
        );
        
        Ok(ticker)
    }
    
    /// Get order book for a symbol
    pub async fn get_order_book(&self, symbol: &str, limit: Option<usize>) -> Result<OrderBook> {
        let start_time = SystemTime::now();
        let url = format!("{}/api/v3/depth", self.config.rest_api_url);
        
        let limit = limit.unwrap_or(self.config.orderbook_depth);
        
        let response = self.client
            .get(&url)
            .query(&[("symbol", symbol), ("limit", &limit.to_string())])
            .send()
            .await?;
            
        let duration = SystemTime::now().duration_since(start_time).unwrap();
        metrics::record_api_request(duration.as_secs_f64());
        
        if !response.status().is_success() {
            metrics::record_api_error();
            return Err(format!("API error: {}", response.status()).into());
        }
        
        let data: serde_json::Value = response.json().await?;
        
        let timestamp = data["lastUpdateId"].as_u64().unwrap_or_else(|| {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64
        });
        
        let bids = data["bids"]
            .as_array()
            .unwrap_or(&Vec::new())
            .iter()
            .map(|bid| {
                let price = bid[0].as_str().unwrap_or("0").parse().unwrap_or(0.0);
                let quantity = bid[1].as_str().unwrap_or("0").parse().unwrap_or(0.0);
                (price, quantity)
            })
            .collect();
            
        let asks = data["asks"]
            .as_array()
            .unwrap_or(&Vec::new())
            .iter()
            .map(|ask| {
                let price = ask[0].as_str().unwrap_or("0").parse().unwrap_or(0.0);
                let quantity = ask[1].as_str().unwrap_or("0").parse().unwrap_or(0.0);
                (price, quantity)
            })
            .collect();
            
        let order_book = OrderBook::new(
            symbol.to_string(),
            bids,
            asks,
            timestamp,
        );
        
        Ok(order_book)
    }
    
    /// Get recent trades for a symbol
    pub async fn get_trades(&self, symbol: &str, limit: Option<usize>) -> Result<Vec<Trade>> {
        let start_time = SystemTime::now();
        let url = format!("{}/api/v3/trades", self.config.rest_api_url);
        
        let limit = limit.unwrap_or(20);
        
        let response = self.client
            .get(&url)
            .query(&[("symbol", symbol), ("limit", &limit.to_string())])
            .send()
            .await?;
            
        let duration = SystemTime::now().duration_since(start_time).unwrap();
        metrics::record_api_request(duration.as_secs_f64());
        
        if !response.status().is_success() {
            metrics::record_api_error();
            return Err(format!("API error: {}", response.status()).into());
        }
        
        let data: Vec<serde_json::Value> = response.json().await?;
        
        let trades = data
            .iter()
            .map(|trade| {
                Trade::new(
                    trade["id"].as_u64().unwrap_or(0).to_string(),
                    symbol.to_string(),
                    trade["price"].as_str().unwrap_or("0").parse().unwrap_or(0.0),
                    trade["qty"].as_str().unwrap_or("0").parse().unwrap_or(0.0),
                    trade["isBuyerMaker"].as_bool().unwrap_or(false),
                    trade["time"].as_u64().unwrap_or_else(|| {
                        SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap()
                            .as_millis() as u64
                    }),
                )
            })
            .collect();
            
        Ok(trades)
    }
    
    /// Sign a request with API key and secret
    fn sign_request(&self, mut request: RequestBuilder, params: &[(&str, &str)]) -> RequestBuilder {
        if let (Some(api_key), Some(api_secret)) = (&self.config.api_key, &self.config.api_secret) {
            // Add API key to headers
            request = request.header("X-MEXC-APIKEY", api_key);
            
            // Build query string
            let mut query = params
                .iter()
                .map(|(k, v)| format!("{}={}", k, v))
                .collect::<Vec<String>>()
                .join("&");
                
            // Add timestamp
            let timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis()
                .to_string();
                
            if !query.is_empty() {
                query.push_str("&");
            }
            
            query.push_str(&format!("timestamp={}", timestamp));
            
            // Sign the query
            let mut mac = HmacSha256::new_from_slice(api_secret.as_bytes())
                .expect("HMAC can take key of any size");
                
            mac.update(query.as_bytes());
            let signature = hex::encode(mac.finalize().into_bytes());
            
            // Add signature to query
            query.push_str(&format!("&signature={}", signature));
            
            // Add query to request
            request = request.header("Content-Type", "application/x-www-form-urlencoded")
                .body(query);
        }
        
        request
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mockito::{mock, server_url};
    
    #[tokio::test]
    async fn test_get_ticker() {
        let mock_response = r#"{
            "symbol": "BTCUSDC",
            "priceChange": "100.00",
            "priceChangePercent": "0.29",
            "lastPrice": "35000.00",
            "volume": "1000.00",
            "highPrice": "36000.00",
            "lowPrice": "34000.00",
            "closeTime": 1621500000000
        }"#;
        
        let _m = mock("GET", "/api/v3/ticker/24hr?symbol=BTCUSDC")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(mock_response)
            .create();
            
        let mut config = Config::default();
        config.rest_api_url = server_url();
        let config = Arc::new(config);
        
        let client = MexcClient::new(&config);
        let ticker = client.get_ticker("BTCUSDC").await.unwrap();
        
        assert_eq!(ticker.symbol, "BTCUSDC");
        assert_eq!(ticker.price, 35000.0);
        assert_eq!(ticker.volume, 1000.0);
        assert_eq!(ticker.high, 36000.0);
        assert_eq!(ticker.low, 34000.0);
        assert_eq!(ticker.timestamp, 1621500000000);
    }
    
    #[tokio::test]
    async fn test_get_order_book() {
        let mock_response = r#"{
            "lastUpdateId": 1621500000000,
            "bids": [
                ["34900.00", "1.00"],
                ["34800.00", "2.00"]
            ],
            "asks": [
                ["35000.00", "1.00"],
                ["35100.00", "2.00"]
            ]
        }"#;
        
        let _m = mock("GET", "/api/v3/depth?symbol=BTCUSDC&limit=20")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(mock_response)
            .create();
            
        let mut config = Config::default();
        config.rest_api_url = server_url();
        let config = Arc::new(config);
        
        let client = MexcClient::new(&config);
        let order_book = client.get_order_book("BTCUSDC", None).await.unwrap();
        
        assert_eq!(order_book.symbol, "BTCUSDC");
        assert_eq!(order_book.bids.len(), 2);
        assert_eq!(order_book.asks.len(), 2);
        assert_eq!(order_book.bids[0].0, 34900.0);
        assert_eq!(order_book.bids[0].1, 1.0);
        assert_eq!(order_book.asks[0].0, 35000.0);
        assert_eq!(order_book.asks[0].1, 1.0);
        assert_eq!(order_book.timestamp, 1621500000000);
    }
}
