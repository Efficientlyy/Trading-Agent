# RL Framework Validation Report

## Summary

Overall Status: **Failed**

| Test Suite | Status |
|-----------|--------|
| Environment Tests | Passed |
| Agent Tests | Passed |
| Integration Tests | Passed |
| Performance Tests | Passed |
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

- Reward Range: [np.float64(-3.2190112263473294e+42), np.float64(-74.02851903537218)]
- Reward Mean: -3.21913899581025e+41
- Reward Std: 9.656991096733624e+41

#### State

Status: **Passed**

Details:

- State Keys: ['price_1m', 'price_5m', 'price_15m', 'price_1h', 'rsi_1m', 'rsi_5m', 'rsi_15m', 'rsi_1h', 'macd_1m', 'macd_5m', 'macd_15m', 'macd_1h', 'bb_1m', 'bb_5m', 'bb_15m', 'bb_1h', 'order_imbalance', 'spread', 'volatility', 'session', 'price_change_1m', 'price_change_5m', 'price_change_15m', 'param_imbalance_threshold', 'param_momentum_threshold', 'param_volatility_threshold', 'param_rsi_threshold', 'param_macd_threshold', 'param_bb_threshold', 'metric_pnl', 'metric_sharpe_ratio', 'metric_win_rate', 'metric_trade_count', 'metric_max_drawdown', 'metric_volatility', 'metric_profit_factor', 'metric_recovery_factor', 'metric_pnl_percent']
- State Size: 38

### Agent Tests

Status: **Passed**

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

Status: **Passed**

Details:

- Update Result: {'actor_loss': 0.0, 'critic_loss': 0.0, 'entropy': 0.0, 'update_status': 'skipped', 'reason': 'buffer_inconsistency'}

#### Save Load

Status: **Passed**

Details:

- Save Success: True
- Load Success: True
- Actions Match: True
- Save Path: validation_results_fixed_v4/test_agent.pt

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

Status: **Passed**

#### Environment Performance

Status: **Passed**

Details:

- Avg Step Time: 0.000575113296508789
- Max Step Time: 0.0006570816040039062
- Min Step Time: 0.0005421638488769531

#### Agent Performance

Status: **Passed**

Details:

- Avg Select Action Time: 0.0005562996864318848
- Max Select Action Time: 0.0009274482727050781
- Min Select Action Time: 0.0004725456237792969
- Update Time: 0.00021958351135253906

#### Integration Performance

Status: **Passed**

Details:

- Train Time: 0.005213499069213867
- Eval Time: 0.0018146038055419922

#### Optimization

Status: **Passed**

Details:

- Optimization Skipped: True
- Reason: Optimization is a separate component

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

