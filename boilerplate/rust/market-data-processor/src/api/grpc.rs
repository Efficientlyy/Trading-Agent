use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::mpsc;
use tokio_stream::{Stream, StreamExt};
use tonic::{Request, Response, Status};
use tracing::{debug, error};

use crate::services::market_data_service::MarketDataService;

// Import the generated code
pub mod market_data {
    tonic::include_proto!("market_data");
}

use market_data::{
    market_data_service_server::{MarketDataService as GrpcMarketDataService, MarketDataServiceServer},
    OrderBookRequest, OrderBookResponse, OrderBookEntry, OrderBookUpdate,
    TickerRequest, TickerResponse, TickerUpdate,
    RecentTradesRequest, RecentTradesResponse, Trade, TradeUpdate,
    OrderBookSubscriptionRequest, TradeSubscriptionRequest, TickerSubscriptionRequest,
};

pub struct GrpcService {
    market_data_service: Arc<MarketDataService>,
}

impl GrpcService {
    pub fn new(market_data_service: Arc<MarketDataService>) -> Self {
        Self { market_data_service }
    }
    
    pub fn into_server(self) -> MarketDataServiceServer<Self> {
        MarketDataServiceServer::new(self)
    }
}

#[tonic::async_trait]
impl GrpcMarketDataService for GrpcService {
    async fn get_order_book(
        &self,
        request: Request<OrderBookRequest>,
    ) -> Result<Response<OrderBookResponse>, Status> {
        let req = request.into_inner();
        let symbol = req.symbol;
        let depth = req.depth as usize;
        
        debug!("GetOrderBook request for symbol: {}, depth: {}", symbol, depth);
        
        let order_book_manager = self.market_data_service.get_order_book_manager();
        
        match order_book_manager.get_market_depth(&symbol, depth) {
            Some((bids, asks)) => {
                let order_book = OrderBookResponse {
                    symbol: symbol.clone(),
                    last_update_id: 0, // Not exposed in this API
                    bids: bids.into_iter().map(|(price, qty)| OrderBookEntry { price, quantity: qty }).collect(),
                    asks: asks.into_iter().map(|(price, qty)| OrderBookEntry { price, quantity: qty }).collect(),
                    timestamp: 0, // Not exposed in this API
                };
                
                Ok(Response::new(order_book))
            }
            None => Err(Status::not_found(format!("Order book for symbol {} not found", symbol))),
        }
    }
    
    async fn get_ticker(
        &self,
        request: Request<TickerRequest>,
    ) -> Result<Response<TickerResponse>, Status> {
        let req = request.into_inner();
        let symbol = req.symbol;
        
        debug!("GetTicker request for symbol: {}", symbol);
        
        match self.market_data_service.get_ticker(&symbol) {
            Some(ticker) => {
                let response = TickerResponse {
                    symbol: ticker.symbol,
                    price_change: ticker.price_change,
                    price_change_percent: ticker.price_change_percent,
                    weighted_avg_price: ticker.weighted_avg_price,
                    last_price: ticker.last_price,
                    last_quantity: ticker.last_quantity,
                    bid_price: ticker.bid_price,
                    bid_quantity: ticker.bid_quantity,
                    ask_price: ticker.ask_price,
                    ask_quantity: ticker.ask_quantity,
                    open_price: ticker.open_price,
                    high_price: ticker.high_price,
                    low_price: ticker.low_price,
                    volume: ticker.volume,
                    quote_volume: ticker.quote_volume,
                    open_time: ticker.open_time,
                    close_time: ticker.close_time,
                    first_id: ticker.first_id,
                    last_id: ticker.last_id,
                    count: ticker.count,
                };
                
                Ok(Response::new(response))
            }
            None => Err(Status::not_found(format!("Ticker for symbol {} not found", symbol))),
        }
    }
    
    async fn get_recent_trades(
        &self,
        request: Request<RecentTradesRequest>,
    ) -> Result<Response<RecentTradesResponse>, Status> {
        let req = request.into_inner();
        let symbol = req.symbol;
        let limit = req.limit as usize;
        
        debug!("GetRecentTrades request for symbol: {}, limit: {}", symbol, limit);
        
        let trades = self.market_data_service.get_recent_trades(&symbol, limit);
        
        let response = RecentTradesResponse {
            trades: trades.into_iter().map(|t| Trade {
                id: t.id,
                symbol: t.symbol,
                price: t.price,
                quantity: t.quantity,
                buyer_order_id: t.buyer_order_id,
                seller_order_id: t.seller_order_id,
                timestamp: t.timestamp,
                is_buyer_maker: t.is_buyer_maker,
            }).collect(),
        };
        
        Ok(Response::new(response))
    }
    
    type SubscribeToOrderBookUpdatesStream = Pin<Box<dyn Stream<Item = Result<OrderBookUpdate, Status>> + Send + 'static>>;
    
    async fn subscribe_to_order_book_updates(
        &self,
        request: Request<OrderBookSubscriptionRequest>,
    ) -> Result<Response<Self::SubscribeToOrderBookUpdatesStream>, Status> {
        let req = request.into_inner();
        let symbol = req.symbol;
        let depth = req.depth as usize;
        let update_speed = req.update_speed as u64;
        
        debug!("SubscribeToOrderBookUpdates for symbol: {}, depth: {}, update speed: {}ms", 
            symbol, depth, update_speed);
        
        let order_book_manager = self.market_data_service.get_order_book_manager();
        let mut receiver = order_book_manager.get_update_receiver();
        
        let (tx, rx) = mpsc::channel(100);
        
        tokio::spawn(async move {
            let mut last_update_time = std::time::Instant::now();
            
            while let Ok(update) = receiver.recv().await {
                if update.symbol != symbol {
                    continue;
                }
                
                // Rate limit updates
                let now = std::time::Instant::now();
                if !update.is_snapshot && now.duration_since(last_update_time).as_millis() < update_speed as u128 {
                    continue;
                }
                
                last_update_time = now;
                
                // Convert to gRPC response type
                let order_book = update.order_book;
                
                let bids: Vec<OrderBookEntry> = order_book.bids.iter()
                    .rev()
                    .take(depth)
                    .map(|(price, qty)| OrderBookEntry { price: *price, quantity: *qty })
                    .collect();
                    
                let asks: Vec<OrderBookEntry> = order_book.asks.iter()
                    .take(depth)
                    .map(|(price, qty)| OrderBookEntry { price: *price, quantity: *qty })
                    .collect();
                
                let response = OrderBookUpdate {
                    symbol: symbol.clone(),
                    is_snapshot: update.is_snapshot,
                    order_book: Some(OrderBookResponse {
                        symbol: symbol.clone(),
                        last_update_id: order_book.last_update_id,
                        bids,
                        asks,
                        timestamp: order_book.timestamp,
                    }),
                };
                
                if tx.send(Ok(response)).await.is_err() {
                    break;
                }
            }
        });
        
        let output_stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        Ok(Response::new(Box::pin(output_stream) as Self::SubscribeToOrderBookUpdatesStream))
    }
    
    type SubscribeToTradesStream = Pin<Box<dyn Stream<Item = Result<TradeUpdate, Status>> + Send + 'static>>;
    
    async fn subscribe_to_trades(
        &self,
        request: Request<TradeSubscriptionRequest>,
    ) -> Result<Response<Self::SubscribeToTradesStream>, Status> {
        let req = request.into_inner();
        let symbol = req.symbol;
        
        debug!("SubscribeToTrades for symbol: {}", symbol);
        
        // For simplicity, we'll simulate trade updates using a timer
        // In a real implementation, you would use a proper event system
        let (tx, rx) = mpsc::channel(100);
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(1));
            
            loop {
                interval.tick().await;
                
                // Get latest trades
                // This is simplified - in a real system you'd subscribe to a trade event stream
                let trade = Trade {
                    id: 0,
                    symbol: symbol.clone(),
                    price: 0.0,
                    quantity: 0.0,
                    buyer_order_id: 0,
                    seller_order_id: 0,
                    timestamp: 0,
                    is_buyer_maker: false,
                };
                
                let update = TradeUpdate {
                    trade: Some(trade),
                };
                
                if tx.send(Ok(update)).await.is_err() {
                    break;
                }
            }
        });
        
        let output_stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        Ok(Response::new(Box::pin(output_stream) as Self::SubscribeToTradesStream))
    }
    
    type SubscribeToTickersStream = Pin<Box<dyn Stream<Item = Result<TickerUpdate, Status>> + Send + 'static>>;
    
    async fn subscribe_to_tickers(
        &self,
        request: Request<TickerSubscriptionRequest>,
    ) -> Result<Response<Self::SubscribeToTickersStream>, Status> {
        let req = request.into_inner();
        let symbol = req.symbol;
        
        debug!("SubscribeToTickers for symbol: {}", symbol);
        
        // For simplicity, we'll simulate ticker updates using a timer
        // In a real implementation, you would use a proper event system
        let (tx, rx) = mpsc::channel(100);
        let market_data_service = self.market_data_service.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(1));
            
            loop {
                interval.tick().await;
                
                // Get latest ticker
                if let Some(ticker) = market_data_service.get_ticker(&symbol) {
                    let ticker_response = TickerResponse {
                        symbol: ticker.symbol,
                        price_change: ticker.price_change,
                        price_change_percent: ticker.price_change_percent,
                        weighted_avg_price: ticker.weighted_avg_price,
                        last_price: ticker.last_price,
                        last_quantity: ticker.last_quantity,
                        bid_price: ticker.bid_price,
                        bid_quantity: ticker.bid_quantity,
                        ask_price: ticker.ask_price,
                        ask_quantity: ticker.ask_quantity,
                        open_price: ticker.open_price,
                        high_price: ticker.high_price,
                        low_price: ticker.low_price,
                        volume: ticker.volume,
                        quote_volume: ticker.quote_volume,
                        open_time: ticker.open_time,
                        close_time: ticker.close_time,
                        first_id: ticker.first_id,
                        last_id: ticker.last_id,
                        count: ticker.count,
                    };
                    
                    let update = TickerUpdate {
                        ticker: Some(ticker_response),
                    };
                    
                    if tx.send(Ok(update)).await.is_err() {
                        break;
                    }
                }
            }
        });
        
        let output_stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        Ok(Response::new(Box::pin(output_stream) as Self::SubscribeToTickersStream))
    }
}
