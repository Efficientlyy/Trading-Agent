// WebSocket implementation for the backend
use actix::{Actor, StreamHandler};
use actix_web_actors::ws;
use crate::models::{Ticker, OrderBook, Trade};
use serde_json::json;
use tokio::sync::mpsc;

// WebSocket session actor
pub struct WebSocketSession {
    pub ticker_rx: mpsc::Receiver<Ticker>,
    pub orderbook_rx: mpsc::Receiver<OrderBook>,
    pub trades_rx: mpsc::Receiver<Trade>,
}

impl Actor for WebSocketSession {
    type Context = ws::WebsocketContext<Self>;

    fn started(&self, ctx: &mut Self::Context) {
        // Start ticker receiver
        let mut ticker_rx = self.ticker_rx.clone();
        ctx.spawn(
            async move {
                while let Some(ticker) = ticker_rx.recv().await {
                    let msg = json!({
                        "type": "ticker",
                        "data": ticker
                    });
                    return ws::Message::Text(msg.to_string());
                }
                ws::Message::Close(None)
            }.into_actor(self)
        );

        // Start orderbook receiver
        let mut orderbook_rx = self.orderbook_rx.clone();
        ctx.spawn(
            async move {
                while let Some(orderbook) = orderbook_rx.recv().await {
                    let msg = json!({
                        "type": "orderbook",
                        "data": orderbook
                    });
                    return ws::Message::Text(msg.to_string());
                }
                ws::Message::Close(None)
            }.into_actor(self)
        );

        // Start trades receiver
        let mut trades_rx = self.trades_rx.clone();
        ctx.spawn(
            async move {
                while let Some(trade) = trades_rx.recv().await {
                    let msg = json!({
                        "type": "trades",
                        "data": trade
                    });
                    return ws::Message::Text(msg.to_string());
                }
                ws::Message::Close(None)
            }.into_actor(self)
        );
    }
}

impl StreamHandler<Result<ws::Message, ws::ProtocolError>> for WebSocketSession {
    fn handle(&mut self, msg: Result<ws::Message, ws::ProtocolError>, ctx: &mut Self::Context) {
        match msg {
            Ok(ws::Message::Ping(msg)) => ctx.pong(&msg),
            Ok(ws::Message::Text(text)) => {
                // Handle client messages (e.g., subscription requests)
                if let Ok(data) = serde_json::from_str::<serde_json::Value>(&text) {
                    if let Some(msg_type) = data.get("type").and_then(|v| v.as_str()) {
                        match msg_type {
                            "subscribe" => {
                                // Handle subscription request
                                if let Some(channel) = data.get("channel").and_then(|v| v.as_str()) {
                                    log::info!("Client subscribed to channel: {}", channel);
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
            Ok(ws::Message::Close(reason)) => {
                ctx.close(reason);
                ctx.stop();
            }
            _ => {}
        }
    }
}
