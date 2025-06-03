# Capital Allocation Strategy

## Overview

The Capital Allocation Strategy module implements a configurable approach to allocate a specific percentage of available USDC for trading operations. This module replaces the previous profit-based compounding strategy with a direct capital allocation approach.

## Key Features

- **Configurable Allocation Percentage**: Set what percentage of available USDC to use for trading (default: 80%)
- **Minimum Reserve Protection**: Ensures a minimum reserve is always maintained
- **Historical Tracking**: Records all allocation events for analysis
- **Simple API**: Easy-to-use methods for getting trading capital

## Configuration

The capital allocation strategy can be configured through the `settings.json` file:

```json
{
  "trading": {
    "allocation": {
      "enabled": true,
      "percentage": 0.8,
      "min_reserve": 100
    }
  }
}
```

## Usage

### Basic Usage

```python
from llm_overseer.strategy.compounding import CapitalAllocationStrategy

# Initialize with configuration
strategy = CapitalAllocationStrategy(config)

# Get trading capital amount
available_usdc = 1000.0
trading_capital = strategy.get_trading_capital(available_usdc)
# Result: 800.0 (80% of available USDC)

# Execute full allocation (updates history and tracking)
result = strategy.execute_allocation(available_usdc)
```

### Adjusting Allocation Percentage

```python
# Change allocation percentage to 70%
strategy.set_allocation_percentage(0.7)

# Get new trading capital amount
trading_capital = strategy.get_trading_capital(1000.0)
# Result: 700.0 (70% of available USDC)
```

### Minimum Reserve

The strategy ensures a minimum reserve is always maintained:

```python
# With minimum reserve of $100
strategy.set_min_reserve(100.0)

# When available USDC is only $120
trading_capital = strategy.get_trading_capital(120.0)
# Result: $20.0 (to maintain $100 reserve)

# When available USDC is only $100
trading_capital = strategy.get_trading_capital(100.0)
# Result: $0.0 (all funds kept in reserve)
```

## Statistics and Monitoring

```python
# Get allocation statistics
stats = strategy.get_statistics()
```

Example statistics:
```json
{
  "enabled": true,
  "allocation_percentage": 0.8,
  "min_reserve": 100.0,
  "total_capital": 1000.0,
  "allocated_capital": 800.0,
  "reserve_capital": 200.0,
  "last_allocation_date": "2025-06-03T18:25:30.123456",
  "allocation_events": 5
}
```

## Backward Compatibility

For backward compatibility, the previous `CompoundingStrategy` class name is maintained as an alias for `CapitalAllocationStrategy`.
