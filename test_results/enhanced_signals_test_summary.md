# Enhanced Flash Trading Signals Test Summary

Test run at: 2025-06-01 09:38:50

## Overall Result: PASS

## Component Results

| Component | Result | Details |
|-----------|--------|--------|
| technical_indicators | PASS | rsi: PASS, macd: PASS, bollinger_bands: PASS, all_indicators: PASS, empty_data: PASS, insufficient_data: PASS |
| multi_timeframe | PASS | 1m: PASS, 5m: PASS, 15m: PASS, 1h: PASS, 1m_indicators: PASS, 5m_indicators: PASS, 15m_indicators: PASS, candle_closing: PASS |
| dynamic_thresholds | PASS | initial: PASS, adaptation: PASS, bounds: PASS |
| liquidity_slippage | PASS | liquidity: PASS, slippage: PASS, order_size_impact: PASS, unfillable_order: PASS |
| integration | PASS | initialization: PASS, signal_generation: PASS, decision_making: PASS, multi_timeframe_analysis: PASS |

## Detailed Results

### technical_indicators

#### rsi

Result: PASS

- value: 41.67855706635279
- expected_range: 0-100

#### macd

Result: PASS

- value: {'macd_line': 0.02000691699707602, 'signal_line': 0.45009996484840764, 'histogram': -0.4300930478513316}

#### bollinger_bands

Result: PASS

- value: {'upper_band': 96.96343947492362, 'middle_band': 94.32664419605214, 'lower_band': 91.68984891718067, 'bandwidth': 0.05590775122649455, 'percent_b': 0.029573408609308953}

#### all_indicators

Result: PASS

- indicators_count: 11

#### empty_data

Result: PASS

- value: None

#### insufficient_data

Result: PASS

- value: None


### multi_timeframe

#### 1m

Result: PASS

- count: 300

#### 5m

Result: PASS

- count: 60

#### 15m

Result: PASS

- count: 20

#### 1h

Result: PASS

- count: 5

#### 1m_indicators

Result: PASS

- count: 11

#### 5m_indicators

Result: PASS

- count: 11

#### 15m_indicators

Result: PASS

- count: 11

#### candle_closing

Result: PASS

- 1m_last_close: 1622523540000
- 5m_last_close: 1622523300000
- 15m_last_close: 1622522700000
- 1h_last_close: 1622520000000


### dynamic_thresholds

#### initial

Result: PASS

- sessions: ['ASIA', 'EUROPE', 'US']
- signal_types: ['order_imbalance', 'momentum', 'volatility']

#### adaptation

Result: PASS

- initial_threshold: 0.15
- updated_threshold: 0.22499999999999998

#### bounds

Result: PASS

- updated_threshold: 0.3
- max_threshold: 0.3


### liquidity_slippage

#### liquidity

Result: PASS

- bid_liquidity: 15.0
- ask_liquidity: 15.0

#### slippage

Result: PASS

- slippage_estimate: 0.0

#### order_size_impact

Result: PASS

- small_order_slippage: 0.0
- large_order_slippage: 0.0006664445184938354

#### unfillable_order

Result: PASS

- unfillable_slippage: 0.0012662445851382871


### integration

#### initialization

Result: PASS

- market_states: ['BTCUSDC', 'ETHUSDC']

#### signal_generation

Result: PASS

- signals_count: 0

#### decision_making

Result: PASS

- decision: False

#### multi_timeframe_analysis

Result: PASS

- timeframes: ['1m', '5m', '15m', '1h']


