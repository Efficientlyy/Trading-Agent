use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use chrono::{DateTime, NaiveDateTime, TimeZone, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tracing::{debug, error, info, warn};

use crate::test_framework::{MarketDataSnapshot, MarketCondition};

/// MEXC historical kline data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MexcKline {
    pub open_time: i64,
    pub open: String,
    pub high: String,
    pub low: String,
    pub close: String,
    pub volume: String,
    pub close_time: i64,
    pub quote_asset_volume: String,
    pub number_of_trades: i64,
    pub taker_buy_base_asset_volume: String,
    pub taker_buy_quote_asset_volume: String,
    pub ignore: String,
}

/// MEXC historical order book data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MexcOrderBook {
    pub last_update_id: i64,
    pub bids: Vec<Vec<String>>,
    pub asks: Vec<Vec<String>>,
}

/// Historical data timeframe
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Timeframe {
    OneMinute,
    FiveMinutes,
    FifteenMinutes,
    OneHour,
    FourHours,
    OneDay,
}

impl ToString for Timeframe {
    fn to_string(&self) -> String {
        match self {
            Timeframe::OneMinute => "1m".to_string(),
            Timeframe::FiveMinutes => "5m".to_string(),
            Timeframe::FifteenMinutes => "15m".to_string(),
            Timeframe::OneHour => "1h".to_string(),
            Timeframe::FourHours => "4h".to_string(),
            Timeframe::OneDay => "1d".to_string(),
        }
    }
}

/// Historical data period
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalPeriod {
    pub symbol: String,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub market_condition: MarketCondition,
    pub description: String,
}

/// Historical data manager
pub struct HistoricalDataManager {
    data_dir: PathBuf,
    periods: HashMap<String, HistoricalPeriod>,
    cached_data: HashMap<String, Vec<MarketDataSnapshot>>,
    http_client: reqwest::Client,
}

impl HistoricalDataManager {
    /// Create a new historical data manager
    pub fn new<P: AsRef<Path>>(data_dir: P) -> Self {
        let data_dir = data_dir.as_ref().to_path_buf();
        
        // Create data directory if it doesn't exist
        if !data_dir.exists() {
            fs::create_dir_all(&data_dir).expect("Failed to create data directory");
        }
        
        // Load predefined periods
        let periods = Self::load_predefined_periods();
        
        Self {
            data_dir,
            periods,
            cached_data: HashMap::new(),
            http_client: reqwest::Client::new(),
        }
    }
    
    /// Load predefined historical periods
    fn load_predefined_periods() -> HashMap<String, HistoricalPeriod> {
        let mut periods = HashMap::new();
        
        // Bitcoin bull run March 2021
        periods.insert(
            "btc_bull_run_march_2021".to_string(),
            HistoricalPeriod {
                symbol: "BTCUSDT".to_string(),
                start_time: Utc.with_ymd_and_hms(2021, 3, 1, 0, 0, 0).unwrap(),
                end_time: Utc.with_ymd_and_hms(2021, 3, 13, 0, 0, 0).unwrap(),
                market_condition: MarketCondition::Trending,
                description: "Bitcoin bull run in March 2021, strong uptrend with price rising from ~$45,000 to ~$61,000".to_string(),
            },
        );
        
        // Bitcoin crash May 2021
        periods.insert(
            "btc_crash_may_2021".to_string(),
            HistoricalPeriod {
                symbol: "BTCUSDT".to_string(),
                start_time: Utc.with_ymd_and_hms(2021, 5, 12, 0, 0, 0).unwrap(),
                end_time: Utc.with_ymd_and_hms(2021, 5, 19, 0, 0, 0).unwrap(),
                market_condition: MarketCondition::Volatile,
                description: "Bitcoin crash in May 2021, high volatility with price dropping from ~$56,000 to ~$30,000".to_string(),
            },
        );
        
        // Bitcoin ranging period July 2021
        periods.insert(
            "btc_ranging_july_2021".to_string(),
            HistoricalPeriod {
                symbol: "BTCUSDT".to_string(),
                start_time: Utc.with_ymd_and_hms(2021, 7, 1, 0, 0, 0).unwrap(),
                end_time: Utc.with_ymd_and_hms(2021, 7, 21, 0, 0, 0).unwrap(),
                market_condition: MarketCondition::Ranging,
                description: "Bitcoin ranging in July 2021, sideways movement between ~$32,000 and ~$34,000".to_string(),
            },
        );
        
        // Ethereum London fork August 2021
        periods.insert(
            "eth_london_fork_august_2021".to_string(),
            HistoricalPeriod {
                symbol: "ETHUSDT".to_string(),
                start_time: Utc.with_ymd_and_hms(2021, 8, 4, 0, 0, 0).unwrap(),
                end_time: Utc.with_ymd_and_hms(2021, 8, 6, 0, 0, 0).unwrap(),
                market_condition: MarketCondition::Volatile,
                description: "Ethereum London fork in August 2021, increased volatility around EIP-1559 implementation".to_string(),
            },
        );
        
        // Low liquidity period for smaller coin
        periods.insert(
            "low_liquidity_altcoin_2021".to_string(),
            HistoricalPeriod {
                symbol: "DOGEUSDT".to_string(),
                start_time: Utc.with_ymd_and_hms(2021, 6, 15, 0, 0, 0).unwrap(),
                end_time: Utc.with_ymd_and_hms(2021, 6, 22, 0, 0, 0).unwrap(),
                market_condition: MarketCondition::LowLiquidity,
                description: "Low liquidity period for DOGE in June 2021 after initial hype died down".to_string(),
            },
        );
        
        // Add more periods as needed...
        
        periods
    }
    
    /// Get a list of available historical periods
    pub fn get_available_periods(&self) -> Vec<&HistoricalPeriod> {
        self.periods.values().collect()
    }
    
    /// Get historical data for a specific period
    pub async fn get_period_data(&mut self, period_id: &str) -> Result<Vec<MarketDataSnapshot>, String> {
        // Check if data is already cached
        if let Some(data) = self.cached_data.get(period_id) {
            return Ok(data.clone());
        }
        
        // Get period info
        let period = self.periods.get(period_id)
            .ok_or_else(|| format!("Historical period not found: {}", period_id))?;
        
        // Check if data exists on disk
        let data_path = self.data_dir.join(format!("{}.json", period_id));
        
        if data_path.exists() {
            // Load data from disk
            let data = self.load_data_from_file(&data_path)?;
            self.cached_data.insert(period_id.to_string(), data.clone());
            Ok(data)
        } else {
            // Fetch data from API
            let data = self.fetch_historical_data(period).await?;
            
            // Save data to disk
            self.save_data_to_file(&data_path, &data)?;
            
            // Cache data
            self.cached_data.insert(period_id.to_string(), data.clone());
            
            Ok(data)
        }
    }
    
    /// Fetch historical data from MEXC API
    async fn fetch_historical_data(&self, period: &HistoricalPeriod) -> Result<Vec<MarketDataSnapshot>, String> {
        info!("Fetching historical data for {} from {} to {}", 
            period.symbol, period.start_time, period.end_time);
        
        let mut snapshots = Vec::new();
        
        // Calculate timeframe based on period duration
        let duration = period.end_time.signed_duration_since(period.start_time);
        let days = duration.num_days();
        
        let timeframe = if days <= 2 {
            Timeframe::FiveMinutes
        } else if days <= 7 {
            Timeframe::FifteenMinutes
        } else {
            Timeframe::OneHour
        };
        
        // Fetch klines first
        let klines = self.fetch_klines(
            &period.symbol, 
            timeframe,
            period.start_time.timestamp_millis(),
            period.end_time.timestamp_millis(),
        ).await?;
        
        // Process each kline
        for kline in klines {
            let timestamp = Utc.timestamp_millis_opt(kline.open_time).unwrap();
            let price = kline.close.parse::<f64>().unwrap_or_default();
            let volume = kline.volume.parse::<f64>().unwrap_or_default();
            
            // Fetch order book for this timestamp (may not be available for all historical data)
            let (bids, asks) = match self.fetch_order_book(&period.symbol, kline.open_time).await {
                Ok(book) => book,
                Err(_) => {
                    // If order book is not available, generate synthetic one based on the price
                    Self::generate_synthetic_order_book(price, period.market_condition)
                }
            };
            
            // Create snapshot
            let snapshot = MarketDataSnapshot {
                timestamp,
                symbol: period.symbol.clone(),
                price,
                bids,
                asks,
                volume_24h: volume,
            };
            
            snapshots.push(snapshot);
        }
        
        info!("Fetched {} historical data points for {}", snapshots.len(), period.symbol);
        
        Ok(snapshots)
    }
    
    /// Fetch klines from MEXC API
    async fn fetch_klines(
        &self,
        symbol: &str,
        timeframe: Timeframe,
        start_time: i64,
        end_time: i64,
    ) -> Result<Vec<MexcKline>, String> {
        let url = format!(
            "https://api.mexc.com/api/v3/klines?symbol={}&interval={}&startTime={}&endTime={}&limit=1000",
            symbol,
            timeframe.to_string(),
            start_time,
            end_time
        );
        
        let response = self.http_client.get(&url)
            .send()
            .await
            .map_err(|e| format!("Failed to fetch klines: {}", e))?;
        
        if !response.status().is_success() {
            return Err(format!("Failed to fetch klines: HTTP {}", response.status()));
        }
        
        // MEXC returns klines as a JSON array of arrays
        let klines_raw: Vec<Vec<serde_json::Value>> = response.json()
            .await
            .map_err(|e| format!("Failed to parse klines response: {}", e))?;
        
        // Convert to our struct format
        let klines = klines_raw.into_iter()
            .map(|raw_kline| {
                MexcKline {
                    open_time: raw_kline[0].as_i64().unwrap_or_default(),
                    open: raw_kline[1].as_str().unwrap_or_default().to_string(),
                    high: raw_kline[2].as_str().unwrap_or_default().to_string(),
                    low: raw_kline[3].as_str().unwrap_or_default().to_string(),
                    close: raw_kline[4].as_str().unwrap_or_default().to_string(),
                    volume: raw_kline[5].as_str().unwrap_or_default().to_string(),
                    close_time: raw_kline[6].as_i64().unwrap_or_default(),
                    quote_asset_volume: raw_kline[7].as_str().unwrap_or_default().to_string(),
                    number_of_trades: raw_kline[8].as_i64().unwrap_or_default(),
                    taker_buy_base_asset_volume: raw_kline[9].as_str().unwrap_or_default().to_string(),
                    taker_buy_quote_asset_volume: raw_kline[10].as_str().unwrap_or_default().to_string(),
                    ignore: raw_kline[11].as_str().unwrap_or_default().to_string(),
                }
            })
            .collect();
        
        Ok(klines)
    }
    
    /// Fetch order book from MEXC API
    async fn fetch_order_book(
        &self,
        symbol: &str,
        timestamp: i64,
    ) -> Result<(Vec<(f64, f64)>, Vec<(f64, f64)>), String> {
        // Historical order books are not readily available via public API
        // We'll use the current order book API and acknowledge the limitation
        let url = format!(
            "https://api.mexc.com/api/v3/depth?symbol={}&limit=20",
            symbol
        );
        
        let response = self.http_client.get(&url)
            .send()
            .await
            .map_err(|e| format!("Failed to fetch order book: {}", e))?;
        
        if !response.status().is_success() {
            return Err(format!("Failed to fetch order book: HTTP {}", response.status()));
        }
        
        let order_book: MexcOrderBook = response.json()
            .await
            .map_err(|e| format!("Failed to parse order book response: {}", e))?;
        
        // Convert to our format
        let bids = order_book.bids.into_iter()
            .map(|bid| {
                let price = bid[0].parse::<f64>().unwrap_or_default();
                let quantity = bid[1].parse::<f64>().unwrap_or_default();
                (price, quantity)
            })
            .collect();
        
        let asks = order_book.asks.into_iter()
            .map(|ask| {
                let price = ask[0].parse::<f64>().unwrap_or_default();
                let quantity = ask[1].parse::<f64>().unwrap_or_default();
                (price, quantity)
            })
            .collect();
        
        Ok((bids, asks))
    }
    
    /// Generate a synthetic order book based on price and market condition
    fn generate_synthetic_order_book(price: f64, market_condition: MarketCondition) -> (Vec<(f64, f64)>, Vec<(f64, f64)>) {
        let (spread_factor, depth_factor) = match market_condition {
            MarketCondition::Trending => (0.0005, 1.0), // Tight spreads, good depth
            MarketCondition::Ranging => (0.0008, 0.8),  // Moderate spreads, moderate depth
            MarketCondition::Volatile => (0.002, 0.6),  // Wide spreads, shallow depth
            MarketCondition::LowLiquidity => (0.005, 0.3), // Very wide spreads, very shallow depth
            MarketCondition::Normal => (0.0006, 0.9),   // Normal spreads and depth
        };
        
        // Generate bids (descending prices)
        let mut bids = Vec::new();
        for i in 0..10 {
            let price_factor = 1.0 - spread_factor/2.0 - (i as f64 * 0.0005);
            let bid_price = price * price_factor;
            let quantity = depth_factor * (0.5 + (10.0 - i as f64) * 0.05);
            bids.push((bid_price, quantity));
        }
        
        // Generate asks (ascending prices)
        let mut asks = Vec::new();
        for i in 0..10 {
            let price_factor = 1.0 + spread_factor/2.0 + (i as f64 * 0.0005);
            let ask_price = price * price_factor;
            let quantity = depth_factor * (0.5 + (10.0 - i as f64) * 0.05);
            asks.push((ask_price, quantity));
        }
        
        (bids, asks)
    }
    
    /// Load data from file
    fn load_data_from_file(&self, path: &Path) -> Result<Vec<MarketDataSnapshot>, String> {
        let file = File::open(path)
            .map_err(|e| format!("Failed to open data file: {}", e))?;
        
        let reader = BufReader::new(file);
        let data: Vec<MarketDataSnapshot> = serde_json::from_reader(reader)
            .map_err(|e| format!("Failed to parse data file: {}", e))?;
        
        Ok(data)
    }
    
    /// Save data to file
    fn save_data_to_file(&self, path: &Path, data: &Vec<MarketDataSnapshot>) -> Result<(), String> {
        let file = File::create(path)
            .map_err(|e| format!("Failed to create data file: {}", e))?;
        
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, data)
            .map_err(|e| format!("Failed to write data file: {}", e))?;
        
        Ok(())
    }
}

/// Historical data loader for testing
pub struct HistoricalDataLoader {
    data_manager: Arc<Mutex<HistoricalDataManager>>,
}

impl HistoricalDataLoader {
    /// Create a new historical data loader
    pub fn new<P: AsRef<Path>>(data_dir: P) -> Self {
        let data_manager = Arc::new(Mutex::new(HistoricalDataManager::new(data_dir)));
        
        Self {
            data_manager,
        }
    }
    
    /// Load historical data for a specific period
    pub async fn load_period(&self, period_id: &str) -> Result<Vec<MarketDataSnapshot>, String> {
        let mut data_manager = self.data_manager.lock().await;
        data_manager.get_period_data(period_id).await
    }
    
    /// Get list of available periods
    pub async fn get_available_periods(&self) -> Vec<HistoricalPeriod> {
        let data_manager = self.data_manager.lock().await;
        data_manager.get_available_periods().iter()
            .map(|p| (*p).clone())
            .collect()
    }
    
    /// Load all available periods
    pub async fn load_all_periods(&self) -> HashMap<String, Vec<MarketDataSnapshot>> {
        let mut data_manager = self.data_manager.lock().await;
        let period_ids: Vec<String> = data_manager.periods.keys().cloned().collect();
        
        let mut all_data = HashMap::new();
        
        for period_id in period_ids {
            match data_manager.get_period_data(&period_id).await {
                Ok(data) => {
                    all_data.insert(period_id, data);
                },
                Err(e) => {
                    error!("Failed to load period {}: {}", period_id, e);
                }
            }
        }
        
        all_data
    }
}

/// Create a historical data test that uses real market data
pub async fn create_historical_data_test(
    period_id: &str,
    data_dir: &Path,
) -> Result<Vec<MarketDataSnapshot>, String> {
    // Create data loader
    let loader = HistoricalDataLoader::new(data_dir);
    
    // Load historical data
    let data = loader.load_period(period_id).await?;
    
    Ok(data)
}
