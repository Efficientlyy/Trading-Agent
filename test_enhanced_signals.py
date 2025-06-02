#!/usr/bin/env python
"""
Test script for validating the enhanced flash trading signals module.

This script performs comprehensive testing of all new features:
1. Technical indicators (RSI, MACD, Bollinger Bands)
2. Multi-timeframe analysis
3. Dynamic thresholding
4. Liquidity and slippage awareness

The tests include both unit tests for individual components and
integration tests for the complete signal generation pipeline.
"""

import os
import sys
import time
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from threading import Thread, Event
from indicators import TechnicalIndicators
from enhanced_flash_trading_signals import EnhancedMarketState, EnhancedFlashTradingSignals

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("enhanced_signals_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("test_enhanced_signals")

class TestEnhancedSignals:
    """Test suite for enhanced flash trading signals"""
    
    def __init__(self):
        """Initialize test suite"""
        self.test_results = {
            "technical_indicators": {},
            "multi_timeframe": {},
            "dynamic_thresholds": {},
            "liquidity_slippage": {},
            "integration": {}
        }
        
        # Create test output directory
        os.makedirs("test_results", exist_ok=True)
    
    def run_all_tests(self):
        """Run all tests and save results"""
        logger.info("Starting enhanced signals test suite")
        
        # Run tests
        self.test_technical_indicators()
        self.test_multi_timeframe_analysis()
        self.test_dynamic_thresholds()
        self.test_liquidity_slippage()
        self.test_integration()
        
        # Save results
        self.save_results()
        
        logger.info("Enhanced signals test suite completed")
    
    def test_technical_indicators(self):
        """Test technical indicators implementation"""
        logger.info("Testing technical indicators")
        
        # Generate sample data
        np.random.seed(42)
        n = 200
        close_prices = np.cumsum(np.random.normal(0, 1, n)) + 100
        high_prices = close_prices + np.random.uniform(0, 2, n)
        low_prices = close_prices - np.random.uniform(0, 2, n)
        volumes = np.random.uniform(1000, 5000, n)
        timestamps = np.array([1622505600000 + i * 60000 for i in range(n)])  # 1-minute intervals
        
        # Create market data dictionary
        market_data = {
            'close': close_prices,
            'high': high_prices,
            'low': low_prices,
            'volume': volumes,
            'timestamp': timestamps
        }
        
        # Test individual indicators
        try:
            # Test RSI
            rsi = TechnicalIndicators.calculate_rsi(close_prices)
            self.test_results["technical_indicators"]["rsi"] = {
                "success": rsi is not None,
                "value": float(rsi) if rsi is not None else None,
                "expected_range": "0-100"
            }
            logger.info(f"RSI test: {'PASS' if rsi is not None else 'FAIL'} - Value: {rsi}")
            
            # Test MACD
            macd = TechnicalIndicators.calculate_macd(close_prices)
            self.test_results["technical_indicators"]["macd"] = {
                "success": macd is not None,
                "value": {k: float(v) for k, v in macd.items()} if macd is not None else None
            }
            logger.info(f"MACD test: {'PASS' if macd is not None else 'FAIL'}")
            
            # Test Bollinger Bands
            bb = TechnicalIndicators.calculate_bollinger_bands(close_prices)
            self.test_results["technical_indicators"]["bollinger_bands"] = {
                "success": bb is not None,
                "value": {k: float(v) for k, v in bb.items()} if bb is not None else None
            }
            logger.info(f"Bollinger Bands test: {'PASS' if bb is not None else 'FAIL'}")
            
            # Test all indicators
            all_indicators = TechnicalIndicators.calculate_all_indicators(market_data)
            self.test_results["technical_indicators"]["all_indicators"] = {
                "success": len(all_indicators) > 0,
                "indicators_count": len(all_indicators)
            }
            logger.info(f"All indicators test: {'PASS' if len(all_indicators) > 0 else 'FAIL'} - Count: {len(all_indicators)}")
            
            # Test edge cases
            # Empty data
            empty_rsi = TechnicalIndicators.calculate_rsi(np.array([]))
            self.test_results["technical_indicators"]["empty_data"] = {
                "success": empty_rsi is None,
                "value": empty_rsi
            }
            logger.info(f"Empty data test: {'PASS' if empty_rsi is None else 'FAIL'}")
            
            # Insufficient data
            short_data = TechnicalIndicators.calculate_rsi(np.array([100, 101, 102]))
            self.test_results["technical_indicators"]["insufficient_data"] = {
                "success": short_data is None,
                "value": short_data
            }
            logger.info(f"Insufficient data test: {'PASS' if short_data is None else 'FAIL'}")
            
            # Overall success
            self.test_results["technical_indicators"]["overall"] = {
                "success": all([
                    self.test_results["technical_indicators"]["rsi"]["success"],
                    self.test_results["technical_indicators"]["macd"]["success"],
                    self.test_results["technical_indicators"]["bollinger_bands"]["success"],
                    self.test_results["technical_indicators"]["all_indicators"]["success"],
                    self.test_results["technical_indicators"]["empty_data"]["success"],
                    self.test_results["technical_indicators"]["insufficient_data"]["success"]
                ])
            }
            logger.info(f"Technical indicators overall test: {'PASS' if self.test_results['technical_indicators']['overall']['success'] else 'FAIL'}")
            
        except Exception as e:
            logger.error(f"Error in technical indicators test: {str(e)}")
            self.test_results["technical_indicators"]["error"] = str(e)
    
    def test_multi_timeframe_analysis(self):
        """Test multi-timeframe analysis framework"""
        logger.info("Testing multi-timeframe analysis")
        
        try:
            # Create market state
            market_state = EnhancedMarketState("BTCUSDC")
            
            # Generate sample data for multiple timeframes
            np.random.seed(42)
            
            # 1-minute data
            for i in range(300):
                timestamp = 1622505600000 + i * 60000  # 1-minute intervals
                bid_price = 30000 + np.cumsum(np.random.normal(0, 10, i+1))[-1]
                ask_price = bid_price + 1
                
                # Update price history
                market_state.timestamp = timestamp
                market_state.bid_price = bid_price
                market_state.ask_price = ask_price
                market_state.mid_price = (bid_price + ask_price) / 2
                
                # Update timeframe data
                market_state._update_timeframe_data('1m', timestamp)
                
                # Check if candles closed for other timeframes
                if market_state._is_candle_closed('5m', timestamp):
                    market_state._update_timeframe_data('5m', timestamp)
                
                if market_state._is_candle_closed('15m', timestamp):
                    market_state._update_timeframe_data('15m', timestamp)
                
                if market_state._is_candle_closed('1h', timestamp):
                    market_state._update_timeframe_data('1h', timestamp)
            
            # Verify data in each timeframe
            self.test_results["multi_timeframe"]["1m"] = {
                "success": len(market_state.price_history['1m']) > 0,
                "count": len(market_state.price_history['1m'])
            }
            logger.info(f"1m timeframe test: {'PASS' if len(market_state.price_history['1m']) > 0 else 'FAIL'} - Count: {len(market_state.price_history['1m'])}")
            
            self.test_results["multi_timeframe"]["5m"] = {
                "success": len(market_state.price_history['5m']) > 0,
                "count": len(market_state.price_history['5m'])
            }
            logger.info(f"5m timeframe test: {'PASS' if len(market_state.price_history['5m']) > 0 else 'FAIL'} - Count: {len(market_state.price_history['5m'])}")
            
            self.test_results["multi_timeframe"]["15m"] = {
                "success": len(market_state.price_history['15m']) > 0,
                "count": len(market_state.price_history['15m'])
            }
            logger.info(f"15m timeframe test: {'PASS' if len(market_state.price_history['15m']) > 0 else 'FAIL'} - Count: {len(market_state.price_history['15m'])}")
            
            self.test_results["multi_timeframe"]["1h"] = {
                "success": len(market_state.price_history['1h']) > 0,
                "count": len(market_state.price_history['1h'])
            }
            logger.info(f"1h timeframe test: {'PASS' if len(market_state.price_history['1h']) > 0 else 'FAIL'} - Count: {len(market_state.price_history['1h'])}")
            
            # Calculate technical indicators for all timeframes
            market_state._calculate_technical_indicators()
            
            # Verify indicators in each timeframe
            for timeframe in ['1m', '5m', '15m', '1h']:
                if len(market_state.price_history[timeframe]) >= 20:
                    self.test_results["multi_timeframe"][f"{timeframe}_indicators"] = {
                        "success": len(market_state.indicators[timeframe]) > 0,
                        "count": len(market_state.indicators[timeframe])
                    }
                    logger.info(f"{timeframe} indicators test: {'PASS' if len(market_state.indicators[timeframe]) > 0 else 'FAIL'} - Count: {len(market_state.indicators[timeframe])}")
            
            # Test candle closing logic
            self.test_results["multi_timeframe"]["candle_closing"] = {
                "success": True,
                "1m_last_close": market_state.last_candle_close['1m'],
                "5m_last_close": market_state.last_candle_close['5m'],
                "15m_last_close": market_state.last_candle_close['15m'],
                "1h_last_close": market_state.last_candle_close['1h']
            }
            logger.info(f"Candle closing test: PASS")
            
            # Overall success
            self.test_results["multi_timeframe"]["overall"] = {
                "success": all([
                    self.test_results["multi_timeframe"]["1m"]["success"],
                    self.test_results["multi_timeframe"]["5m"]["success"],
                    self.test_results["multi_timeframe"]["15m"]["success"],
                    self.test_results["multi_timeframe"]["1h"]["success"]
                ])
            }
            logger.info(f"Multi-timeframe analysis overall test: {'PASS' if self.test_results['multi_timeframe']['overall']['success'] else 'FAIL'}")
            
        except Exception as e:
            logger.error(f"Error in multi-timeframe analysis test: {str(e)}")
            self.test_results["multi_timeframe"]["error"] = str(e)
    
    def test_dynamic_thresholds(self):
        """Test dynamic thresholding system"""
        logger.info("Testing dynamic thresholds")
        
        try:
            # Create signal generator
            signal_generator = EnhancedFlashTradingSignals()
            
            # Verify initial thresholds
            initial_thresholds = signal_generator.dynamic_thresholds
            
            self.test_results["dynamic_thresholds"]["initial"] = {
                "success": len(initial_thresholds) > 0,
                "sessions": list(initial_thresholds.keys()),
                "signal_types": list(initial_thresholds["ASIA"].keys()) if "ASIA" in initial_thresholds else []
            }
            logger.info(f"Initial thresholds test: {'PASS' if len(initial_thresholds) > 0 else 'FAIL'}")
            
            # Create market state with high volatility
            market_state = EnhancedMarketState("BTCUSDC")
            market_state.volatility = 0.05  # High volatility
            market_state.trend = 0.02  # Strong uptrend
            
            # Add to signal generator
            signal_generator.market_states["BTCUSDC"] = market_state
            
            # Update dynamic thresholds
            signal_generator._update_dynamic_thresholds()
            
            # Verify threshold adaptation
            updated_thresholds = signal_generator.dynamic_thresholds
            
            # Check if order imbalance threshold was increased due to high volatility
            order_imbalance_adapted = (
                updated_thresholds["ASIA"]["order_imbalance"]["current"] > 
                initial_thresholds["ASIA"]["order_imbalance"]["current"]
            )
            
            self.test_results["dynamic_thresholds"]["adaptation"] = {
                "success": order_imbalance_adapted,
                "initial_threshold": initial_thresholds["ASIA"]["order_imbalance"]["current"],
                "updated_threshold": updated_thresholds["ASIA"]["order_imbalance"]["current"]
            }
            logger.info(f"Threshold adaptation test: {'PASS' if order_imbalance_adapted else 'FAIL'}")
            
            # Test threshold bounds
            # Create market state with extreme volatility
            market_state.volatility = 1.0  # Extreme volatility
            
            # Update dynamic thresholds
            signal_generator._update_dynamic_thresholds()
            
            # Verify threshold bounds
            extreme_thresholds = signal_generator.dynamic_thresholds
            
            # Check if order imbalance threshold was capped at maximum
            order_imbalance_bounded = (
                extreme_thresholds["ASIA"]["order_imbalance"]["current"] <= 
                initial_thresholds["ASIA"]["order_imbalance"]["max"]
            )
            
            self.test_results["dynamic_thresholds"]["bounds"] = {
                "success": order_imbalance_bounded,
                "updated_threshold": extreme_thresholds["ASIA"]["order_imbalance"]["current"],
                "max_threshold": initial_thresholds["ASIA"]["order_imbalance"]["max"]
            }
            logger.info(f"Threshold bounds test: {'PASS' if order_imbalance_bounded else 'FAIL'}")
            
            # Overall success
            self.test_results["dynamic_thresholds"]["overall"] = {
                "success": all([
                    self.test_results["dynamic_thresholds"]["initial"]["success"],
                    self.test_results["dynamic_thresholds"]["adaptation"]["success"],
                    self.test_results["dynamic_thresholds"]["bounds"]["success"]
                ])
            }
            logger.info(f"Dynamic thresholds overall test: {'PASS' if self.test_results['dynamic_thresholds']['overall']['success'] else 'FAIL'}")
            
        except Exception as e:
            logger.error(f"Error in dynamic thresholds test: {str(e)}")
            self.test_results["dynamic_thresholds"]["error"] = str(e)
    
    def test_liquidity_slippage(self):
        """Test liquidity and slippage awareness"""
        logger.info("Testing liquidity and slippage awareness")
        
        try:
            # Create market state
            market_state = EnhancedMarketState("BTCUSDC")
            
            # Generate sample order book
            bids = [
                [29990.0, 1.0],
                [29980.0, 2.0],
                [29970.0, 3.0],
                [29960.0, 4.0],
                [29950.0, 5.0]
            ]
            
            asks = [
                [30010.0, 1.0],
                [30020.0, 2.0],
                [30030.0, 3.0],
                [30040.0, 4.0],
                [30050.0, 5.0]
            ]
            
            # Update order book
            market_state.update_order_book(bids, asks)
            
            # Verify liquidity metrics
            self.test_results["liquidity_slippage"]["liquidity"] = {
                "success": market_state.bid_liquidity > 0 and market_state.ask_liquidity > 0,
                "bid_liquidity": market_state.bid_liquidity,
                "ask_liquidity": market_state.ask_liquidity
            }
            logger.info(f"Liquidity metrics test: {'PASS' if market_state.bid_liquidity > 0 and market_state.ask_liquidity > 0 else 'FAIL'}")
            
            # Verify slippage estimate
            self.test_results["liquidity_slippage"]["slippage"] = {
                "success": market_state.slippage_estimate >= 0,
                "slippage_estimate": market_state.slippage_estimate
            }
            logger.info(f"Slippage estimate test: {'PASS' if market_state.slippage_estimate >= 0 else 'FAIL'}")
            
            # Test slippage calculation with different order sizes
            small_slippage = market_state._estimate_slippage(asks, 0.5, "buy")
            large_slippage = market_state._estimate_slippage(asks, 10.0, "buy")
            
            self.test_results["liquidity_slippage"]["order_size_impact"] = {
                "success": large_slippage > small_slippage,
                "small_order_slippage": small_slippage,
                "large_order_slippage": large_slippage
            }
            logger.info(f"Order size impact test: {'PASS' if large_slippage > small_slippage else 'FAIL'}")
            
            # Test unfillable order
            unfillable_slippage = market_state._estimate_slippage(asks, 100.0, "buy")
            
            self.test_results["liquidity_slippage"]["unfillable_order"] = {
                "success": unfillable_slippage > 0,
                "unfillable_slippage": unfillable_slippage
            }
            logger.info(f"Unfillable order test: {'PASS' if unfillable_slippage > 0 else 'FAIL'}")
            
            # Create signal generator
            signal_generator = EnhancedFlashTradingSignals()
            
            # Add market state
            signal_generator.market_states["BTCUSDC"] = market_state
            
            # Generate signals
            signals = signal_generator.generate_signals("BTCUSDC")
            
            # Verify slippage in signals
            if signals:
                slippage_in_signals = all("slippage_estimate" in signal for signal in signals)
                
                self.test_results["liquidity_slippage"]["signals"] = {
                    "success": slippage_in_signals,
                    "signals_count": len(signals)
                }
                logger.info(f"Slippage in signals test: {'PASS' if slippage_in_signals else 'FAIL'}")
            
            # Overall success
            self.test_results["liquidity_slippage"]["overall"] = {
                "success": all([
                    self.test_results["liquidity_slippage"]["liquidity"]["success"],
                    self.test_results["liquidity_slippage"]["slippage"]["success"],
                    self.test_results["liquidity_slippage"]["order_size_impact"]["success"],
                    self.test_results["liquidity_slippage"]["unfillable_order"]["success"]
                ])
            }
            logger.info(f"Liquidity and slippage awareness overall test: {'PASS' if self.test_results['liquidity_slippage']['overall']['success'] else 'FAIL'}")
            
        except Exception as e:
            logger.error(f"Error in liquidity and slippage awareness test: {str(e)}")
            self.test_results["liquidity_slippage"]["error"] = str(e)
    
    def test_integration(self):
        """Test integration of all components"""
        logger.info("Testing integration of all components")
        
        try:
            # Create signal generator
            signal_generator = EnhancedFlashTradingSignals()
            
            # Start signal generation
            signal_generator.start(["BTCUSDC", "ETHUSDC"])
            
            # Wait for initialization
            time.sleep(2)
            
            # Verify market states
            market_states_initialized = "BTCUSDC" in signal_generator.market_states
            
            self.test_results["integration"]["initialization"] = {
                "success": market_states_initialized,
                "market_states": list(signal_generator.market_states.keys())
            }
            logger.info(f"Initialization test: {'PASS' if market_states_initialized else 'FAIL'}")
            
            # Generate signals
            signals = signal_generator.generate_signals("BTCUSDC")
            
            self.test_results["integration"]["signal_generation"] = {
                "success": True,  # Even if no signals, the process should complete without errors
                "signals_count": len(signals)
            }
            logger.info(f"Signal generation test: PASS - Signals count: {len(signals)}")
            
            # Make trading decision
            decision = signal_generator.make_trading_decision("BTCUSDC")
            
            self.test_results["integration"]["decision_making"] = {
                "success": True,  # Even if no decision, the process should complete without errors
                "decision": decision is not None
            }
            logger.info(f"Decision making test: PASS - Decision: {decision is not None}")
            
            # Get multi-timeframe analysis
            analysis = signal_generator.get_multi_timeframe_analysis("BTCUSDC")
            
            self.test_results["integration"]["multi_timeframe_analysis"] = {
                "success": analysis is not None,
                "timeframes": list(analysis["timeframes"].keys()) if analysis and "timeframes" in analysis else []
            }
            logger.info(f"Multi-timeframe analysis test: {'PASS' if analysis is not None else 'FAIL'}")
            
            # Stop signal generation
            signal_generator.stop()
            
            # Verify stopped
            self.test_results["integration"]["shutdown"] = {
                "success": not signal_generator.running
            }
            logger.info(f"Shutdown test: {'PASS' if not signal_generator.running else 'FAIL'}")
            
            # Overall success
            self.test_results["integration"]["overall"] = {
                "success": all([
                    self.test_results["integration"]["initialization"]["success"],
                    self.test_results["integration"]["signal_generation"]["success"],
                    self.test_results["integration"]["decision_making"]["success"],
                    self.test_results["integration"]["multi_timeframe_analysis"]["success"],
                    self.test_results["integration"]["shutdown"]["success"]
                ])
            }
            logger.info(f"Integration overall test: {'PASS' if self.test_results['integration']['overall']['success'] else 'FAIL'}")
            
        except Exception as e:
            logger.error(f"Error in integration test: {str(e)}")
            self.test_results["integration"]["error"] = str(e)
    
    def save_results(self):
        """Save test results to file"""
        try:
            # Add overall success
            self.test_results["overall"] = {
                "success": all([
                    self.test_results["technical_indicators"]["overall"]["success"],
                    self.test_results["multi_timeframe"]["overall"]["success"],
                    self.test_results["dynamic_thresholds"]["overall"]["success"],
                    self.test_results["liquidity_slippage"]["overall"]["success"],
                    self.test_results["integration"]["overall"]["success"]
                ]),
                "timestamp": int(time.time() * 1000)
            }
            
            # Save to file
            with open("test_results/enhanced_signals_test_results.json", "w") as f:
                json.dump(self.test_results, f, indent=2)
            
            # Generate summary report
            summary = self._generate_summary()
            
            # Save summary to file
            with open("test_results/enhanced_signals_test_summary.md", "w") as f:
                f.write(summary)
            
            logger.info("Test results saved")
            
        except Exception as e:
            logger.error(f"Error saving test results: {str(e)}")
    
    def _generate_summary(self):
        """Generate summary report"""
        summary = "# Enhanced Flash Trading Signals Test Summary\n\n"
        summary += f"Test run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Overall result
        overall_success = self.test_results["overall"]["success"]
        summary += f"## Overall Result: {'PASS' if overall_success else 'FAIL'}\n\n"
        
        # Component results
        summary += "## Component Results\n\n"
        summary += "| Component | Result | Details |\n"
        summary += "|-----------|--------|--------|\n"
        
        for component, results in self.test_results.items():
            if component == "overall":
                continue
                
            if "overall" in results and "success" in results["overall"]:
                success = results["overall"]["success"]
                result = "PASS" if success else "FAIL"
                
                # Get details
                details = []
                for test, test_results in results.items():
                    if test != "overall" and test != "error":
                        if "success" in test_results:
                            test_success = test_results["success"]
                            details.append(f"{test}: {'PASS' if test_success else 'FAIL'}")
                
                summary += f"| {component} | {result} | {', '.join(details)} |\n"
        
        # Detailed results
        summary += "\n## Detailed Results\n\n"
        
        for component, results in self.test_results.items():
            if component == "overall":
                continue
                
            summary += f"### {component}\n\n"
            
            for test, test_results in results.items():
                if test == "error":
                    summary += f"**Error:** {test_results}\n\n"
                elif test != "overall":
                    summary += f"#### {test}\n\n"
                    summary += f"Result: {'PASS' if test_results.get('success', False) else 'FAIL'}\n\n"
                    
                    # Add details
                    for key, value in test_results.items():
                        if key != "success":
                            summary += f"- {key}: {value}\n"
                    
                    summary += "\n"
            
            summary += "\n"
        
        return summary

if __name__ == "__main__":
    # Run tests
    test_suite = TestEnhancedSignals()
    test_suite.run_all_tests()
    
    # Print overall result
    overall_success = test_suite.test_results["overall"]["success"]
    print(f"\nOverall test result: {'PASS' if overall_success else 'FAIL'}")
    print(f"See test_results/enhanced_signals_test_summary.md for details")
