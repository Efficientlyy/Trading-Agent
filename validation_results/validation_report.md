# RL Framework Validation Report

## Summary

Overall Status: **Failed**

| Test Suite | Status |
|-----------|--------|
| Environment Tests | Passed |
| Agent Tests | Failed |
| Integration Tests | Failed |
| Performance Tests | Failed |
| Edge Case Tests | Failed |

## Detailed Results

### Environment Tests

Status: **Passed**

#### Initialization

Status: **Passed**

Details:

- Env Type: TradingRLEnvironment

#### Reset

Status: **Passed**

Details:

- State Type: dict
- State Keys: ['price_1m', 'price_5m', 'price_15m', 'price_1h', 'rsi_1m', 'rsi_5m', 'rsi_15m', 'rsi_1h', 'macd_1m', 'macd_5m', 'macd_15m', 'macd_1h', 'bb_1m', 'bb_5m', 'bb_15m', 'bb_1h', 'order_imbalance', 'spread', 'volatility', 'session', 'price_change_1m', 'price_change_5m', 'price_change_15m', 'param_imbalance_threshold', 'param_momentum_threshold', 'param_volatility_threshold', 'param_rsi_threshold', 'param_macd_threshold', 'param_bb_threshold', 'metric_pnl', 'metric_sharpe_ratio', 'metric_win_rate', 'metric_trade_count', 'metric_max_drawdown', 'metric_volatility', 'metric_profit_factor', 'metric_recovery_factor']
- State Size: 37

#### Step

Status: **Passed**

Details:

- Next State Type: dict
- Reward Type: float64
- Done Type: bool
- Info Type: dict

#### Reward

Status: **Passed**

Details:

- Reward Range: [np.float64(-3.1940016882461906e+39), np.float64(-40.90130343497389)]
- Reward Mean: -6.388003376492588e+38
- Reward Std: 1.277600675298466e+39

#### State

Status: **Passed**

Details:

- State Keys: ['price_1m', 'price_5m', 'price_15m', 'price_1h', 'rsi_1m', 'rsi_5m', 'rsi_15m', 'rsi_1h', 'macd_1m', 'macd_5m', 'macd_15m', 'macd_1h', 'bb_1m', 'bb_5m', 'bb_15m', 'bb_1h', 'order_imbalance', 'spread', 'volatility', 'session', 'price_change_1m', 'price_change_5m', 'price_change_15m', 'param_imbalance_threshold', 'param_momentum_threshold', 'param_volatility_threshold', 'param_rsi_threshold', 'param_macd_threshold', 'param_bb_threshold', 'metric_pnl', 'metric_sharpe_ratio', 'metric_win_rate', 'metric_trade_count', 'metric_max_drawdown', 'metric_volatility', 'metric_profit_factor', 'metric_recovery_factor', 'metric_pnl_percent']
- State Size: 38

### Agent Tests

Status: **Failed**

#### Initialization

Status: **Passed**

Details:

- Agent Type: PPOAgent

#### Action Selection

Status: **Passed**

Details:

- Action Type: ndarray
- Action Shape: (4,)
- Log Prob Type: ndarray
- Value Type: ndarray

#### Update

Status: **Failed**

Details:

- Update Result: None

#### Save Load

Status: **Failed**

Details:

- Actions Match: False
- Save Path: validation_results/test_agent.pt

### Integration Tests

Status: **Failed**

Error: new(): data must be a sequence (got dict)

#### Initialization

Status: **Passed**

Details:

- Integration Type: RLIntegration

#### Training

Status: **Not Run**

Details:


#### Evaluation

Status: **Not Run**

Details:


#### Parameter Extraction

Status: **Not Run**

Details:


### Performance Tests

Status: **Failed**

Error: new(): data must be a sequence (got dict)

#### Environment Performance

Status: **Passed**

Details:

- Metrics Type: dict
- Avg Step Time: 0.0007786393165588379
- Max Step Time: 0.0009388923645019531

#### Agent Performance

Status: **Passed**

Details:

- Metrics Type: dict
- Avg Select Action Time: 0.0007047867774963379
- Avg Update Time: 1.3470649719238281e-05

#### Integration Performance

Status: **Not Run**

Details:


#### Optimization

Status: **Not Run**

Details:


### Edge Case Tests

Status: **Failed**

#### Empty Data

Status: **Passed**

Details:

- Empty Data Handled: False

#### Invalid Action

Status: **Failed**

Details:

- Invalid Action Results: [True, True, True, False, True]

#### Extreme Values

Status: **Passed**

Details:

- Extreme Value Results: [True, True, True, True]

#### Error Handling

Status: **Passed**

Details:

- Error Handled: True

