# RL Framework Validation Report

## Summary

Overall Status: **Failed**

| Test Suite | Status |
|-----------|--------|
| Environment Tests | Failed |
| Agent Tests | Passed |
| Integration Tests | Passed |
| Performance Tests | Passed |
| Edge Case Tests | Failed |

## Detailed Results

### Environment Tests

Status: **Failed**

#### Initialization

Status: **Passed**

Details:

- Env Type: TradingRLEnvironment

#### Reset

Status: **Failed**

Details:

- State Type: dict
- State Keys: []
- State Size: 0

#### Step

Status: **Passed**

Details:

- Next State Type: dict
- Reward Type: float
- Done Type: bool
- Info Type: dict

#### Reward

Status: **Passed**

Details:

- Reward Range: [0.0, 0.0]
- Reward Mean: 0.0
- Reward Std: 0.0

#### State

Status: **Passed**

Details:

- State Keys: ['param_imbalance_threshold', 'param_momentum_threshold', 'param_volatility_threshold', 'param_rsi_threshold', 'price_1m', 'order_imbalance', 'spread', 'volatility', 'session', 'rsi_1m', 'macd_1m', 'bb_1m', 'price_change_1m', 'price_5m', 'price_change_5m', 'rsi_5m', 'macd_5m', 'bb_5m', 'price_15m', 'price_change_15m', 'rsi_15m', 'macd_15m', 'bb_15m', 'price_1h', 'price_change_1h', 'rsi_1h', 'macd_1h', 'bb_1h', 'metric_pnl', 'metric_sharpe_ratio', 'metric_win_rate', 'metric_trade_count', 'metric_max_drawdown', 'metric_volatility', 'metric_profit_factor', 'metric_recovery_factor', 'metric_pnl_percent']
- State Size: 37

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
- Save Path: validation_results_final/test_agent.pt

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

- Avg Step Time: 0.001017749309539795
- Max Step Time: 0.001180887222290039
- Min Step Time: 0.000978231430053711

#### Agent Performance

Status: **Passed**

Details:

- Avg Select Action Time: 0.0005432558059692383
- Max Select Action Time: 0.0009868144989013672
- Min Select Action Time: 0.0004863739013671875
- Update Time: 0.00027441978454589844

#### Integration Performance

Status: **Passed**

Details:

- Train Time: 0.005632877349853516
- Eval Time: 0.002023458480834961

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

- Empty Data Handled: True

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

