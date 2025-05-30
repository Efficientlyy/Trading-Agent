mod api;
mod models;
mod utils;
mod paper_trading;

use actix_cors::Cors;
use actix_web::{web, App, HttpServer, Responder, HttpResponse};
use actix_web_actors::ws;
use api::{MexcClient, MexcWebsocketClient};
use models::{OrderBook, Ticker, Trade};
use paper_trading::PaperTradingEngine;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;
use tokio::time::{sleep, Duration};
use utils::config::Config;

// WebSocket session actor
struct WebSocketSession {
    ticker_rx: mpsc::Receiver<Ticker>,
    orderbook_rx: mpsc::Receiver<OrderBook>,
    trades_rx: mpsc::Receiver<Trade>,
}

// HTTP handlers
async fn get_ticker(
    symbol: web::Path<String>,
    client: web::Data<Arc<MexcClient>>,
) -> impl Responder {
    match client.get_ticker(&symbol).await {
        Ok(ticker) => HttpResponse::Ok().json(ticker),
        Err(e) => {
            log::error!("Error getting ticker: {}", e);
            HttpResponse::InternalServerError().body(format!("Error: {}", e))
        }
    }
}

async fn get_order_book(
    symbol: web::Path<String>,
    client: web::Data<Arc<MexcClient>>,
) -> impl Responder {
    match client.get_order_book(&symbol, None).await {
        Ok(order_book) => HttpResponse::Ok().json(order_book),
        Err(e) => {
            log::error!("Error getting order book: {}", e);
            HttpResponse::InternalServerError().body(format!("Error: {}", e))
        }
    }
}

async fn get_trades(
    symbol: web::Path<String>,
    client: web::Data<Arc<MexcClient>>,
) -> impl Responder {
    match client.get_trades(&symbol, None).await {
        Ok(trades) => HttpResponse::Ok().json(trades),
        Err(e) => {
            log::error!("Error getting trades: {}", e);
            HttpResponse::InternalServerError().body(format!("Error: {}", e))
        }
    }
}

async fn get_account(
    paper_trading: web::Data<Arc<Mutex<PaperTradingEngine>>>,
) -> impl Responder {
    match paper_trading.lock() {
        Ok(engine) => {
            let account = engine.get_account();
            HttpResponse::Ok().json(account)
        }
        Err(e) => {
            log::error!("Error getting account: {}", e);
            HttpResponse::InternalServerError().body(format!("Error: {}", e))
        }
    }
}

async fn place_order(
    order_data: web::Json<serde_json::Value>,
    paper_trading: web::Data<Arc<Mutex<PaperTradingEngine>>>,
) -> impl Responder {
    match paper_trading.lock() {
        Ok(mut engine) => {
            // Parse order data
            let symbol = order_data.get("symbol").and_then(|v| v.as_str()).unwrap_or("BTCUSDC");
            let side = order_data.get("side").and_then(|v| v.as_str()).unwrap_or("buy");
            let order_type = order_data.get("type").and_then(|v| v.as_str()).unwrap_or("market");
            let quantity = order_data.get("quantity").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let price = order_data.get("price").and_then(|v| v.as_f64());
            
            // Create and place order
            let result = engine.place_order(symbol, side, order_type, quantity, price);
            
            match result {
                Ok(order) => HttpResponse::Ok().json(order),
                Err(e) => {
                    log::error!("Error placing order: {}", e);
                    HttpResponse::BadRequest().body(format!("Error: {}", e))
                }
            }
        }
        Err(e) => {
            log::error!("Error accessing paper trading engine: {}", e);
            HttpResponse::InternalServerError().body(format!("Error: {}", e))
        }
    }
}

async fn cancel_order(
    order_id: web::Path<String>,
    paper_trading: web::Data<Arc<Mutex<PaperTradingEngine>>>,
) -> impl Responder {
    match paper_trading.lock() {
        Ok(mut engine) => {
            let result = engine.cancel_order(&order_id);
            
            match result {
                Ok(_) => HttpResponse::Ok().json(serde_json::json!({ "success": true })),
                Err(e) => {
                    log::error!("Error canceling order: {}", e);
                    HttpResponse::BadRequest().body(format!("Error: {}", e))
                }
            }
        }
        Err(e) => {
            log::error!("Error accessing paper trading engine: {}", e);
            HttpResponse::InternalServerError().body(format!("Error: {}", e))
        }
    }
}

// WebSocket handler
async fn websocket_handler(
    req: actix_web::HttpRequest,
    stream: web::Payload,
    data: web::Data<WebSocketSessionData>,
) -> Result<HttpResponse, actix_web::Error> {
    // Create new channel receivers for this session
    let (ticker_tx, ticker_rx) = mpsc::channel(100);
    let (orderbook_tx, orderbook_rx) = mpsc::channel(100);
    let (trades_tx, trades_rx) = mpsc::channel(100);
    
    // Clone the shared transmitters to forward data to this session
    let mut ticker_tx_clone = data.ticker_tx.lock().unwrap().clone();
    let mut orderbook_tx_clone = data.orderbook_tx.lock().unwrap().clone();
    let mut trades_tx_clone = data.trades_tx.lock().unwrap().clone();
    
    // Forward data from shared transmitters to this session's receivers
    tokio::spawn(async move {
        while let Some(ticker) = ticker_rx.recv().await {
            if let Err(e) = ticker_tx_clone.send(ticker).await {
                log::error!("Error forwarding ticker: {}", e);
                break;
            }
        }
    });
    
    tokio::spawn(async move {
        while let Some(orderbook) = orderbook_rx.recv().await {
            if let Err(e) = orderbook_tx_clone.send(orderbook).await {
                log::error!("Error forwarding orderbook: {}", e);
                break;
            }
        }
    });
    
    tokio::spawn(async move {
        while let Some(trade) = trades_rx.recv().await {
            if let Err(e) = trades_tx_clone.send(trade).await {
                log::error!("Error forwarding trade: {}", e);
                break;
            }
        }
    });
    
    // Create WebSocket session
    let session = WebSocketSession {
        ticker_rx,
        orderbook_rx,
        trades_rx,
    };
    
    // Start WebSocket connection
    ws::start(session, &req, stream)
}

// Shared WebSocket session data
struct WebSocketSessionData {
    ticker_tx: Mutex<mpsc::Sender<Ticker>>,
    orderbook_tx: Mutex<mpsc::Sender<OrderBook>>,
    trades_tx: Mutex<mpsc::Sender<Trade>>,
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // Initialize logger
    env_logger::init_from_env(env_logger::Env::default().default_filter_or("info"));
    
    // Load configuration
    let config = Arc::new(Config::load(None));
    log::info!("Configuration loaded");
    
    // Create MEXC API client
    let mexc_client = Arc::new(MexcClient::new(&config));
    log::info!("MEXC API client created");
    
    // Create paper trading engine
    let paper_trading_engine = Arc::new(Mutex::new(PaperTradingEngine::new(
        config.initial_balance_usdc,
        config.initial_balance_btc,
    )));
    log::info!("Paper trading engine created");
    
    // Create channels for data distribution
    let ((ticker_tx, ticker_rx), (orderbook_tx, orderbook_rx), (trades_tx, trades_rx)) =
        MexcWebsocketClient::create_channels();
    
    // Create WebSocket session data
    let websocket_data = web::Data::new(WebSocketSessionData {
        ticker_tx: Mutex::new(ticker_tx),
        orderbook_tx: Mutex::new(orderbook_tx),
        trades_tx: Mutex::new(trades_tx),
    });
    
    // Start WebSocket client in a separate task
    let config_clone = Arc::clone(&config);
    tokio::spawn(async move {
        loop {
            let mut ws_client = MexcWebsocketClient::new(Arc::clone(&config_clone));
            ws_client.set_ticker_channel(ticker_tx.clone());
            ws_client.set_orderbook_channel(orderbook_tx.clone());
            ws_client.set_trades_channel(trades_tx.clone());
            
            if let Err(e) = ws_client.connect(&config_clone.default_pair).await {
                log::error!("WebSocket connection error: {}", e);
            }
            
            // Wait before reconnecting
            sleep(Duration::from_secs(config_clone.websocket_reconnect_interval)).await;
        }
    });
    
    // Start HTTP server
    log::info!("Starting HTTP server at http://0.0.0.0:8080");
    HttpServer::new(move || {
        // Configure CORS
        let cors = Cors::default()
            .allow_any_origin()
            .allow_any_method()
            .allow_any_header();
        
        App::new()
            .wrap(cors)
            .app_data(web::Data::new(Arc::clone(&mexc_client)))
            .app_data(web::Data::new(Arc::clone(&paper_trading_engine)))
            .app_data(websocket_data.clone())
            .service(
                web::scope("/api/v1")
                    .route("/ticker/{symbol}", web::get().to(get_ticker))
                    .route("/orderbook/{symbol}", web::get().to(get_order_book))
                    .route("/trades/{symbol}", web::get().to(get_trades))
                    .route("/account", web::get().to(get_account))
                    .route("/order", web::post().to(place_order))
                    .route("/order/{id}", web::delete().to(cancel_order))
            )
            .route("/ws", web::get().to(websocket_handler))
            .service(actix_files::Files::new("/", "./dashboard/build").index_file("index.html"))
    })
    .bind("0.0.0.0:8080")?
    .run()
    .await
}
