use futures::{SinkExt, StreamExt};
use serde_json::{json, Value};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};
use tracing::{info, error, warn, debug, Level};
use tracing_subscriber::FmtSubscriber;
use url::Url;
use std::time::SystemTime;
use hmac::{Hmac, Mac};
use sha2::Sha256;

/// Test script for BTCUSDC order book subscription verification
/// This test will attempt to connect to MEXC WebSocket API and subscribe to BTCUSDC order book
/// It will log detailed information about the connection status and responses
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging with detailed format
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::DEBUG)
        .with_file(true)
        .with_line_number(true)
        .with_thread_ids(true)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;
    
    info!("Starting BTCUSDC Order Book Subscription Test");
    
    // New whitelisted API credentials
    let api_key = std::env::var("MEXC_API_KEY").unwrap_or_else(|_| "mx0vglTbKSqTso4bzf".to_string());
    let api_secret = std::env::var("MEXC_API_SECRET").unwrap_or_else(|_| "63c248e899524b4499f13f428ad01e24".to_string());
    
    // WebSocket connection parameters
    let ws_url = "wss://wbs.mexc.com/ws";
    let symbol = "BTCUSDC";  // Specifically testing with BTCUSDC as requested
    let channel = "spot@public.depth.v3.api";
    let subscription = format!("{}@{}", channel, symbol);
    
    // Get the public IP for verification
    info!("Determining public IP address for verification");
    match get_public_ip().await {
        Ok(ip) => info!("Public IP address: {} - This IP must be whitelisted with MEXC", ip),
        Err(e) => warn!("Unable to determine public IP: {}", e),
    }
    
    // Connect to WebSocket
    info!("Connecting to MEXC WebSocket at {}", ws_url);
    let url = Url::parse(ws_url)?;
    let (ws_stream, _) = connect_async(url).await?;
    info!("WebSocket connection established");
    
    let (mut write, mut read) = ws_stream.split();
    
    // Generate timestamp for authentication
    let timestamp = format!("{}", SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)?
        .as_millis());
    
    // Generate signature
    let signature = generate_signature(&api_secret, &timestamp)?;
    
    // Send authentication request
    let auth_msg = json!({
        "method": "api_key",
        "api_key": api_key,
        "sign": signature,
        "reqTime": timestamp
    });
    
    info!("Sending authentication request");
    write.send(Message::Text(auth_msg.to_string())).await?;
    
    // Wait for auth response
    let auth_response = read.next().await.ok_or("No response received")??;
    match auth_response {
        Message::Text(text) => {
            info!("Authentication response: {}", text);
            match serde_json::from_str::<Value>(&text) {
                Ok(value) => {
                    if let Some(code) = value.get("code") {
                        if code.as_i64() != Some(0) {
                            warn!("Authentication error: {}", value);
                        } else {
                            info!("Authentication successful");
                        }
                    }
                },
                Err(e) => warn!("Failed to parse authentication response: {}", e),
            }
        },
        _ => warn!("Unexpected authentication response format"),
    }
    
    // Subscribe to order book
    let subscription_msg = json!({
        "method": "SUBSCRIPTION",
        "params": [subscription]
    });
    
    info!("Sending subscription request for {}", subscription);
    write.send(Message::Text(subscription_msg.to_string())).await?;
    
    // Variables to track received messages
    let mut snapshot_received = false;
    let mut update_count = 0;
    let start_time = Instant::now();
    let test_duration = Duration::from_secs(30);
    
    // Monitor for responses
    info!("Waiting for order book messages (test will run for {} seconds)...", test_duration.as_secs());
    while start_time.elapsed() < test_duration {
        tokio::select! {
            Some(msg) = read.next() => {
                match msg {
                    Ok(Message::Text(text)) => {
                        match serde_json::from_str::<Value>(&text) {
                            Ok(value) => {
                                // Check if this is a subscription response
                                if let Some(code) = value.get("code") {
                                    let msg = value.get("msg").and_then(|m| m.as_str()).unwrap_or("");
                                    info!("Subscription response: {} (code: {})", msg, code);
                                    
                                    // Check for "Blocked" error
                                    if msg.contains("Blocked") {
                                        error!("CRITICAL: Subscription blocked. IP whitelisting is required.");
                                        error!("Contact MEXC support to whitelist IP: {}", get_public_ip().await.unwrap_or_else(|_| "unknown".to_string()));
                                        break;
                                    }
                                } 
                                // Check if this is a data message
                                else if let Some(channel_type) = value.get("c") {
                                    if channel_type.as_str().unwrap_or("").contains("depth") {
                                        let symbol = value.get("s").and_then(|s| s.as_str()).unwrap_or("");
                                        
                                        if !snapshot_received {
                                            info!("Received order book snapshot for {}", symbol);
                                            snapshot_received = true;
                                            
                                            // Print detailed info about the snapshot
                                            if let Some(data) = value.get("data") {
                                                let bid_count = data.get("bids").and_then(|b| b.as_array()).map_or(0, |a| a.len());
                                                let ask_count = data.get("asks").and_then(|a| a.as_array()).map_or(0, |a| a.len());
                                                info!("Order book contains {} bids and {} asks", bid_count, ask_count);
                                                
                                                // Print top 3 bids and asks
                                                if bid_count > 0 {
                                                    info!("Top 3 bids:");
                                                    if let Some(bids) = data.get("bids").and_then(|b| b.as_array()) {
                                                        for (i, bid) in bids.iter().take(3).enumerate() {
                                                            if let (Some(price), Some(qty)) = (bid.get(0), bid.get(1)) {
                                                                info!("  {}: {} @ {}", i+1, qty, price);
                                                            }
                                                        }
                                                    }
                                                }
                                                
                                                if ask_count > 0 {
                                                    info!("Top 3 asks:");
                                                    if let Some(asks) = data.get("asks").and_then(|a| a.as_array()) {
                                                        for (i, ask) in asks.iter().take(3).enumerate() {
                                                            if let (Some(price), Some(qty)) = (ask.get(0), ask.get(1)) {
                                                                info!("  {}: {} @ {}", i+1, qty, price);
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        } else {
                                            update_count += 1;
                                            if update_count % 5 == 0 {  // Log every 5th update to avoid verbose output
                                                info!("Received order book update #{} for {}", update_count, symbol);
                                            }
                                        }
                                    }
                                }
                            },
                            Err(e) => warn!("Failed to parse message: {} - Text: {}", e, text),
                        }
                    },
                    Ok(Message::Ping(data)) => {
                        debug!("Received ping, sending pong");
                        write.send(Message::Pong(data)).await?;
                    },
                    Ok(Message::Close(frame)) => {
                        warn!("WebSocket closed by server: {:?}", frame);
                        break;
                    },
                    Err(e) => {
                        error!("WebSocket error: {}", e);
                        break;
                    },
                    _ => {} // Ignore other message types
                }
            },
            _ = tokio::time::sleep(Duration::from_secs(5)) => {
                if !snapshot_received && update_count == 0 {
                    warn!("No messages received in 5 seconds, sending ping...");
                    write.send(Message::Ping(vec![])).await?;
                }
            }
        }
    }
    
    // Final status report
    info!("\n--- TEST RESULTS ---");
    info!("Test duration: {:.2} seconds", start_time.elapsed().as_secs_f64());
    info!("Snapshot received: {}", snapshot_received);
    info!("Updates received: {}", update_count);
    
    if snapshot_received || update_count > 0 {
        info!("✅ TEST PASSED: Successfully received BTCUSDC order book data");
        info!("The IP {} is properly whitelisted with MEXC", 
            get_public_ip().await.unwrap_or_else(|_| "unknown".to_string()));
    } else {
        error!("❌ TEST FAILED: No order book data received");
        error!("Possible reasons:");
        error!("1. IP not whitelisted with MEXC");
        error!("2. API credentials invalid or expired");
        error!("3. Network connectivity issues");
        error!("4. MEXC service issues");
    }
    
    // Close WebSocket connection
    write.send(Message::Close(None)).await?;
    
    Ok(())
}

/// Get the public IP address
async fn get_public_ip() -> Result<String, Box<dyn std::error::Error>> {
    let resp = reqwest::get("https://api.ipify.org").await?;
    let ip = resp.text().await?;
    Ok(ip)
}

/// Generate HMAC-SHA256 signature for MEXC API authentication
fn generate_signature(secret_key: &str, timestamp: &str) -> Result<String, Box<dyn std::error::Error>> {
    type HmacSha256 = Hmac<Sha256>;
    
    let mut mac = HmacSha256::new_from_slice(secret_key.as_bytes())?;
    mac.update(timestamp.as_bytes());
    
    let result = mac.finalize();
    let signature = hex::encode(result.into_bytes());
    
    Ok(signature)
}
