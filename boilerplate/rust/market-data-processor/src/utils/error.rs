use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Configuration error: {0}")]
    ConfigError(#[from] config::ConfigError),
    
    #[error("WebSocket error: {0}")]
    WebSocketError(#[from] tokio_tungstenite::tungstenite::Error),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),
    
    #[error("HTTP error: {0}")]
    HttpError(#[from] reqwest::Error),
    
    #[error("URL parse error: {0}")]
    UrlParseError(#[from] url::ParseError),
    
    #[error("Message parsing error: {0}")]
    MessageParseError(String),
    
    #[error("Exchange API error: {0}")]
    ExchangeApiError(String),
    
    #[error("Validation error: {0}")]
    ValidationError(String),
    
    #[error("{0}")]
    Other(String),
}

// Result type alias for convenience
pub type Result<T> = std::result::Result<T, Error>;
