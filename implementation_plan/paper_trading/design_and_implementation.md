# Paper Trading Mode Implementation

This document outlines the design and implementation of the paper trading mode for the MEXC trading system. Paper trading allows developers to test the system with real market data without executing actual trades or risking real funds.

## Architecture

The paper trading mode is implemented as a configurable layer in the Order Execution module, allowing the system to operate in two modes:
1. **Live Trading Mode**: Executes real trades on the MEXC exchange
2. **Paper Trading Mode**: Simulates order execution using real market data

## Key Components

### 1. Order Execution Service with Paper Trading Support

The Order Execution Service is modified to support paper trading through a configuration flag:

```rust
// In order_execution/src/config.rs
pub struct Config {
    // Other configuration fields...
    pub paper_trading: bool,
}

impl Config {
    pub fn load() -> Result<Self, ConfigError> {
        // Load configuration from environment variables
        let paper_trading = env::var("PAPER_TRADING")
            .unwrap_or_else(|_| "true".to_string())
            .parse::<bool>()
            .unwrap_or(true);
        
        // Return config
        Ok(Config {
            // Other fields...
            paper_trading,
        })
    }
}
```

### 2. Virtual Account Manager

A new component to track virtual balances, positions, and order history:

```rust
// In order_execution/src/services/virtual_account.rs
pub struct VirtualAccount {
    balances: HashMap<String, Decimal>,
    positions: HashMap<String, Position>,
    orders: HashMap<String, Order>,
    trades: Vec<Trade>,
}

impl VirtualAccount {
    pub fn new(initial_balances: HashMap<String, Decimal>) -> Self {
        VirtualAccount {
            balances: initial_balances,
            positions: HashMap::new(),
            orders: HashMap::new(),
            trades: Vec::new(),
        }
    }
    
    pub fn place_order(&mut self, order: Order) -> Result<String, VirtualAccountError> {
        // Validate order
        self.validate_order(&order)?;
        
        // Generate order ID
        let order_id = Uuid::new_v4().to_string();
        
        // Store order
        self.orders.insert(order_id.clone(), order);
        
        Ok(order_id)
    }
    
    // Other methods for order management, position tracking, etc.
}
```

### 3. Order Matching Engine

A simplified matching engine to simulate order execution against real market data:

```rust
// In order_execution/src/services/paper_trading/matching_engine.rs
pub struct MatchingEngine {
    market_data_client: Arc<dyn MarketDataClient>,
}

impl MatchingEngine {
    pub fn new(market_data_client: Arc<dyn MarketDataClient>) -> Self {
        MatchingEngine {
            market_data_client,
        }
    }
    
    pub async fn process_order(&self, order: &Order, account: &mut VirtualAccount) -> Result<Vec<Trade>, MatchingError> {
        // Get current market data
        let order_book = self.market_data_client.get_order_book(order.symbol.clone(), 20).await?;
        
        // Simulate order execution based on order type
        match order.order_type {
            OrderType::Market => self.execute_market_order(order, order_book, account).await,
            OrderType::Limit => self.execute_limit_order(order, order_book, account).await,
            // Other order types...
        }
    }
    
    // Methods for different order types
}
```

### 4. Paper Trading Service

The main service that coordinates paper trading operations:

```rust
// In order_execution/src/services/paper_trading/service.rs
pub struct PaperTradingService {
    virtual_account: Mutex<VirtualAccount>,
    matching_engine: Arc<MatchingEngine>,
    market_data_client: Arc<dyn MarketDataClient>,
}

impl PaperTradingService {
    pub fn new(
        initial_balances: HashMap<String, Decimal>,
        market_data_client: Arc<dyn MarketDataClient>,
    ) -> Self {
        let virtual_account = VirtualAccount::new(initial_balances);
        let matching_engine = Arc::new(MatchingEngine::new(market_data_client.clone()));
        
        PaperTradingService {
            virtual_account: Mutex::new(virtual_account),
            matching_engine,
            market_data_client,
        }
    }
    
    pub async fn place_order(&self, order_request: OrderRequest) -> Result<OrderResponse, ServiceError> {
        // Convert request to internal order model
        let order = Order::from(order_request);
        
        // Lock virtual account
        let mut account = self.virtual_account.lock().await;
        
        // Place order in virtual account
        let order_id = account.place_order(order.clone())?;
        
        // Process order with matching engine
        let trades = self.matching_engine.process_order(&order, &mut account).await?;
        
        // Update order status
        account.update_order_status(order_id.clone(), OrderStatus::Filled)?;
        
        // Return response
        Ok(OrderResponse {
            order_id,
            status: "success".to_string(),
            // Other fields...
        })
    }
    
    // Other methods for order management, account queries, etc.
}
```

### 5. Order Execution Facade

A facade that routes requests to either live trading or paper trading based on configuration:

```rust
// In order_execution/src/services/order_execution.rs
pub struct OrderExecutionService {
    config: Arc<Config>,
    paper_trading_service: Arc<PaperTradingService>,
    live_trading_service: Arc<LiveTradingService>,
}

impl OrderExecutionService {
    pub fn new(
        config: Arc<Config>,
        paper_trading_service: Arc<PaperTradingService>,
        live_trading_service: Arc<LiveTradingService>,
    ) -> Self {
        OrderExecutionService {
            config,
            paper_trading_service,
            live_trading_service,
        }
    }
    
    pub async fn place_order(&self, order_request: OrderRequest) -> Result<OrderResponse, ServiceError> {
        if self.config.paper_trading {
            // Route to paper trading
            self.paper_trading_service.place_order(order_request).await
        } else {
            // Route to live trading
            self.live_trading_service.place_order(order_request).await
        }
    }
    
    // Other methods with similar routing logic
}
```

## Configuration

Paper trading mode is controlled through environment variables:

```
# In .env file or environment variables
PAPER_TRADING=true  # Enable paper trading mode
PAPER_TRADING_INITIAL_BALANCE_USDT=10000  # Initial USDT balance for paper trading
PAPER_TRADING_INITIAL_BALANCE_BTC=1  # Initial BTC balance for paper trading
```

## API Endpoints

The Order Execution Service exposes the same API endpoints regardless of whether paper trading is enabled:

```
POST /api/orders - Place a new order
GET /api/orders/{id} - Get order details
DELETE /api/orders/{id} - Cancel an order
GET /api/account/balances - Get account balances
GET /api/account/positions - Get current positions
```

When paper trading is enabled, these endpoints operate on the virtual account instead of the real MEXC account.

## Dashboard Integration

The dashboard includes visual indicators to clearly show when paper trading mode is active:

1. A prominent "PAPER TRADING" banner at the top of the trading interface
2. Different color schemes for paper trading vs. live trading
3. A toggle switch in the settings to enable/disable paper trading mode

## Implementation Steps

1. Add paper trading configuration to the Order Execution Service
2. Implement the Virtual Account Manager to track balances and positions
3. Create the Order Matching Engine to simulate order execution
4. Develop the Paper Trading Service to coordinate operations
5. Modify the Order Execution Facade to route requests based on configuration
6. Update the dashboard to indicate paper trading mode
7. Add comprehensive logging for paper trading activities
8. Implement persistence for paper trading state (to survive restarts)

## Testing

The paper trading implementation includes comprehensive tests:

1. Unit tests for the Virtual Account Manager and Order Matching Engine
2. Integration tests comparing paper trading results with expected outcomes
3. System tests verifying the entire paper trading workflow
4. Performance tests ensuring paper trading doesn't introduce significant latency

## Windows Development Considerations

When developing the paper trading mode on Windows:

1. Ensure all file paths use forward slashes or `Path::new()` for cross-platform compatibility
2. Use the `chrono` crate for time-related operations to handle cross-platform time differences
3. Test database persistence with both Windows and Linux paths
4. Verify that Docker volume mounting works correctly for state persistence

## Security Considerations

Even though paper trading doesn't involve real funds, security is still important:

1. API keys are still required but only used for market data access
2. All security measures remain in place to prevent accidental switching to live trading
3. Clear logging and audit trails distinguish between paper and live trading activities
