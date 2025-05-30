use std::time::Duration;
use tokio::sync::mpsc;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};
use futures::{SinkExt, StreamExt};
use url::Url;
use serde_json::{json, Value};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;
    
    info!("Starting Minimal MEXC WebSocket Demo");
    
    // MEXC WebSocket URL
    let url_str = "wss://wbs.mexc.com/ws";
    let url = Url::parse(url_str)?;
    
    info!("Connecting to MEXC WebSocket at {}", url_str);
    let (ws_stream, _) = connect_async(url).await?;
    info!("WebSocket connection established");
    
    let (mut write, mut read) = ws_stream.split();
    
    // Subscribe to BTC/USDT ticker
    let symbol = "BTCUSDT";
    let channel = "spot@public.ticker.v3.api";
    let subscription = format!("{}@{}", channel, symbol);
    
    info!("Subscribing to {} for {}", channel, symbol);
    let subscription_request = json!({
        "method": "SUBSCRIPTION",
        "params": [subscription]
    });
    
    write.send(Message::Text(subscription_request.to_string())).await?;
    
    // Create a channel to terminate after a few messages
    let (terminate_tx, mut terminate_rx) = mpsc::channel::<bool>(1);
    let terminate_tx_clone = terminate_tx.clone();
    
    // Spawn a task to terminate after 10 seconds
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_secs(10)).await;
        let _ = terminate_tx_clone.send(true).await;
    });
    
    // Process messages
    let mut message_count = 0;
    
    loop {
        tokio::select! {
            _ = terminate_rx.recv() => {
                info!("Terminating after timeout");
                break;
            }
            Some(msg) = read.next() => {
                match msg {
                    Ok(Message::Text(text)) => {
                        info!("Received message: {}", text);
                        message_count += 1;
                        
                        // After receiving 5 messages, terminate
                        if message_count >= 5 {
                            let _ = terminate_tx.send(true).await;
                        }
                    }
                    Ok(Message::Ping(data)) => {
                        info!("Received ping, sending pong");
                        if let Err(e) = write.send(Message::Pong(data)).await {
                            info!("Failed to send pong: {}", e);
                        }
                    }
                    Ok(Message::Close(_)) => {
                        info!("WebSocket connection closed by server");
                        break;
                    }
                    Err(e) => {
                        info!("WebSocket error: {}", e);
                        break;
                    }
                    _ => {} // Ignore other message types
                }
            }
        }
    }
    
    info!("Demo completed, received {} messages", message_count);
    Ok(())
}
