# Trading Session Awareness

This document describes the session awareness functionality in the Flash Trading System, which allows the system to adapt its trading behavior based on the current global trading session (Asia, Europe, US).

## Overview

Cryptocurrency markets operate 24/7, but exhibit different characteristics during different global trading sessions. The session awareness functionality allows the Flash Trading System to adapt to these changing market conditions by:

1. Automatically detecting the current global trading session
2. Applying session-specific parameters to trading decisions
3. Tracking performance metrics by session
4. Allowing custom session definitions and overlaps

## Trading Sessions

The system defines three default trading sessions:

| Session | UTC Hours | Characteristics |
|---------|-----------|----------------|
| ASIA    | 00:00-08:00 | Higher volatility, lower volume |
| EUROPE  | 08:00-16:00 | Moderate volatility, moderate volume |
| US      | 16:00-24:00 | Lower volatility, higher volume |

## Session-Specific Parameters

Each session has its own set of parameters that affect trading behavior:

### Asia Session
- Lower order book imbalance threshold (0.15) - More sensitive to imbalances due to lower liquidity
- Higher volatility threshold (0.12) - Accounts for typically higher volatility
- Lower momentum threshold (0.04) - More sensitive to price movements
- Smaller position sizes (0.8x) - More conservative due to higher volatility
- Higher take profit targets (25 bps) - Accounts for wider price swings
- Higher stop loss levels (15 bps) - Prevents premature stops during volatile periods

### Europe Session
- Standard order book imbalance threshold (0.2)
- Standard volatility threshold (0.1)
- Standard momentum threshold (0.05)
- Standard position sizes (1.0x)
- Standard take profit targets (20 bps)
- Standard stop loss levels (10 bps)

### US Session
- Higher order book imbalance threshold (0.25) - Less sensitive due to higher liquidity
- Lower volatility threshold (0.08) - Accounts for typically lower volatility
- Higher momentum threshold (0.06) - Requires stronger price movements
- Larger position sizes (1.2x) - More aggressive due to higher liquidity
- Lower take profit targets (15 bps) - Accounts for tighter price action
- Lower stop loss levels (8 bps) - Tighter risk management in more efficient markets

## Implementation

### TradingSessionManager

The `TradingSessionManager` class is responsible for:

1. Detecting the current trading session based on UTC time
2. Managing session-specific parameters
3. Handling session transitions
4. Tracking performance metrics by session

```python
# Example usage
session_manager = TradingSessionManager()
current_session = session_manager.get_current_session_name()
imbalance_threshold = session_manager.get_session_parameter("imbalance_threshold", 0.2)
```

### Session-Aware Decision Making

The `SignalGenerator` class integrates with the `TradingSessionManager` to make session-aware trading decisions:

1. Retrieves the current session when generating signals
2. Applies session-specific thresholds to signal generation
3. Uses session-specific parameters for position sizing
4. Tracks performance metrics by session

```python
# Example of session-aware decision making
def make_trading_decision(self, symbol, signals=None):
    # Get current trading session
    current_session = self.session_manager.get_current_session_name()
    
    # Get session-specific parameters
    session_params = self.session_manager.get_all_session_parameters()
    position_size_factor = session_params.get("position_size_factor", 1.0)
    
    # Apply session-specific position sizing
    size = base_size * position_size_factor
```

## Custom Sessions

The system supports defining custom trading sessions:

```python
# Add a custom session
session_manager.add_session(
    "CUSTOM_SESSION",
    start_hour_utc=12,
    end_hour_utc=14,
    description="Custom Trading Session"
)

# Update parameters for custom session
session_manager.update_session_parameter(
    "CUSTOM_SESSION",
    "position_size_factor",
    1.5
)
```

Custom sessions have higher priority than default sessions when they overlap.

## Configuration

Session configuration can be saved to and loaded from a JSON file:

```python
# Save configuration
session_manager.save_config("trading_session_config.json")

# Load configuration
session_manager = TradingSessionManager("trading_session_config.json")
```

## Testing

The system includes comprehensive tests for session awareness:

1. Session detection tests
2. Parameter loading tests
3. Session transition tests
4. Signal generation tests with session awareness
5. Decision making tests with session awareness

To run the tests:

```bash
python test_session_awareness.py --duration 20
```

The test harness can simulate different trading sessions to validate session-specific behavior:

```python
# Force a specific session for testing
test._force_session("ASIA")
```

## Performance Tracking

The system tracks performance metrics by session:

```python
# Get metrics for current session
metrics = session_manager.get_session_metrics()

# Get metrics for all sessions
all_metrics = session_manager.get_all_sessions_metrics()
```

## Best Practices

1. **Monitor Session Performance**: Regularly review performance metrics by session to identify which sessions perform best for your strategy.

2. **Optimize Session Parameters**: Fine-tune parameters for each session based on historical performance.

3. **Consider Custom Sessions**: Define custom sessions for specific market events or overlapping regional trading hours.

4. **Test Session Transitions**: Ensure your strategy handles session transitions smoothly, especially for positions held across sessions.

5. **Validate with Historical Data**: Backtest your strategy with session awareness across different market conditions.

## Future Enhancements

1. **Dynamic Parameter Adjustment**: Automatically adjust session parameters based on recent performance.

2. **Machine Learning Integration**: Use ML to identify optimal parameters for each session.

3. **Market Regime Detection**: Combine session awareness with market regime detection for more adaptive trading.

4. **Calendar Integration**: Incorporate economic calendar events to adjust session behavior during high-impact announcements.

5. **Multi-Asset Correlation**: Consider cross-asset correlations that vary by trading session.
