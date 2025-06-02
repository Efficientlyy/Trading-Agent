# RL Framework Validation Report

## Summary

Overall Status: **Failed**

| Test Suite | Status |
|-----------|--------|
| Environment Tests | Passed |
| Agent Tests | Failed |
| Integration Tests | Passed |
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

- Reward Range: [np.float64(-3.163671232114502e+40), np.float64(-152.90458612541866)]
- Reward Mean: -6.327342501297385e+39
- Reward Std: 1.2654684909923818e+40

#### State

Status: **Passed**

Details:

- State Keys: ['price_1m', 'price_5m', 'price_15m', 'price_1h', 'rsi_1m', 'rsi_5m', 'rsi_15m', 'rsi_1h', 'macd_1m', 'macd_5m', 'macd_15m', 'macd_1h', 'bb_1m', 'bb_5m', 'bb_15m', 'bb_1h', 'order_imbalance', 'spread', 'volatility', 'session', 'price_change_1m', 'price_change_5m', 'price_change_15m', 'param_imbalance_threshold', 'param_momentum_threshold', 'param_volatility_threshold', 'param_rsi_threshold', 'param_macd_threshold', 'param_bb_threshold', 'metric_pnl', 'metric_sharpe_ratio', 'metric_win_rate', 'metric_trade_count', 'metric_max_drawdown', 'metric_volatility', 'metric_profit_factor', 'metric_recovery_factor', 'metric_pnl_percent']
- State Size: 38

### Agent Tests

Status: **Failed**

Error: index 100 is out of bounds for dimension 0 with size 100

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

Status: **Not Run**

Details:


#### Save Load

Status: **Not Run**

Details:


### Integration Tests

Status: **Passed**

#### Initialization

Status: **Passed**

Details:

- Integration Type: RLIntegration

#### Training

Status: **Passed**

Details:

- Metrics Type: dict
- Metrics Keys: ['episode_rewards', 'episode_lengths', 'parameter_history', 'performance_metrics']

#### Evaluation

Status: **Passed**

Details:

- Metrics Type: dict
- Metrics Keys: ['avg_reward', 'avg_steps', 'avg_pnl', 'avg_win_rate', 'rewards', 'steps', 'pnls', 'win_rates']

#### Parameter Extraction

Status: **Passed**

Details:

- Params Type: dict
- Params Keys: ['imbalance_threshold', 'momentum_threshold', 'volatility_threshold', 'rsi_threshold', '_metadata']

### Performance Tests

Status: **Failed**

Error: index 138 is out of bounds for dimension 0 with size 100

#### Environment Performance

Status: **Passed**

Details:

- Avg Step Time: 0.000579071044921875
- Max Step Time: 0.0006375312805175781
- Min Step Time: 0.0005357265472412109

#### Agent Performance

Status: **Not Run**

Details:


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

