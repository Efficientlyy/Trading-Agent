#!/usr/bin/env python
"""
Integration and Paper Trading Test Runner

This script runs integration tests and paper trading simulations for the flash trading system,
with comprehensive performance monitoring and result analysis.
"""

import time
import logging
import json
import os
import argparse
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from threading import Thread, Event
from flash_trading_signals import SignalGenerator
from trading_session_manager import TradingSessionManager
from optimized_mexc_client import OptimizedMexcClient
from paper_trading import PaperTradingSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_results.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("integration_test")

class IntegrationTestRunner:
    """Runs integration tests and paper trading simulations"""
    
    def __init__(self, env_path=None, config_path=None):
        """Initialize test runner with configuration"""
        self.env_path = env_path or ".env-secure/.env"
        self.config_path = config_path
        
        # Initialize components
        self.client = OptimizedMexcClient(env_path=self.env_path)
        self.session_manager = TradingSessionManager()
        self.signal_generator = None
        self.paper_trading = None
        
        # Test configuration
        self.config = {
            "test_duration": 3600,  # 1 hour by default
            "symbols": ["BTCUSDC", "ETHUSDC"],
            "initial_balance": {
                "USDC": 10000.0,
                "BTC": 0.0,
                "ETH": 0.0
            },
            "metrics_interval": 60,  # seconds
            "report_interval": 300,  # seconds
            "save_results": True,
            "plot_results": True
        }
        
        # Load configuration if provided
        if config_path and os.path.exists(config_path):
            self._load_config(config_path)
        
        # Test results
        self.results = {
            "start_time": None,
            "end_time": None,
            "duration": 0,
            "signals": [],
            "decisions": [],
            "trades": [],
            "balances": [],
            "metrics": {
                "signal_count": 0,
                "decision_count": 0,
                "trade_count": 0,
                "profitable_trades": 0,
                "losing_trades": 0,
                "total_profit_loss": 0.0,
                "win_rate": 0.0,
                "avg_latency_ms": 0.0,
                "max_latency_ms": 0.0
            },
            "session_metrics": {}
        }
        
        # Control flags
        self.running = False
        self.stop_event = Event()
    
    def _load_config(self, config_path):
        """Load test configuration from file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Update configuration
            self.config.update(config)
            logger.info(f"Configuration loaded from {config_path}")
            
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
    
    def run_test(self):
        """Run integration test and paper trading simulation"""
        if self.running:
            logger.warning("Test already running")
            return False
        
        try:
            # Set running flag
            self.running = True
            self.stop_event.clear()
            
            # Initialize results
            self.results["start_time"] = int(time.time() * 1000)
            self.results["signals"] = []
            self.results["decisions"] = []
            self.results["trades"] = []
            self.results["balances"] = []
            
            # Initialize components
            self._initialize_components()
            
            # Start monitoring thread
            monitor_thread = Thread(target=self._monitoring_loop)
            monitor_thread.daemon = True
            monitor_thread.start()
            
            # Run test for specified duration
            logger.info(f"Starting integration test for {self.config['test_duration']} seconds")
            logger.info(f"Testing symbols: {self.config['symbols']}")
            logger.info(f"Initial balance: {self.config['initial_balance']}")
            
            # Start signal generator
            self.signal_generator.start(self.config["symbols"])
            
            # Wait for test duration
            start_time = time.time()
            end_time = start_time + self.config["test_duration"]
            
            last_report_time = start_time
            
            while time.time() < end_time and not self.stop_event.is_set():
                # Sleep for a bit
                time.sleep(1)
                
                # Generate periodic reports
                current_time = time.time()
                if current_time - last_report_time >= self.config["report_interval"]:
                    self._generate_report()
                    last_report_time = current_time
            
            # Stop components
            self._stop_components()
            
            # Finalize results
            self.results["end_time"] = int(time.time() * 1000)
            self.results["duration"] = self.results["end_time"] - self.results["start_time"]
            
            # Generate final report
            self._generate_final_report()
            
            # Save results if requested
            if self.config["save_results"]:
                self._save_results()
            
            # Plot results if requested
            if self.config["plot_results"]:
                self._plot_results()
            
            # Reset running flag
            self.running = False
            
            return True
            
        except Exception as e:
            logger.error(f"Error running test: {str(e)}")
            self.running = False
            return False
    
    def stop_test(self):
        """Stop running test"""
        if not self.running:
            logger.warning("No test running")
            return False
        
        # Set stop event
        self.stop_event.set()
        
        # Wait for components to stop
        time.sleep(2)
        
        # Stop components if still running
        self._stop_components()
        
        # Reset running flag
        self.running = False
        
        return True
    
    def _initialize_components(self):
        """Initialize test components"""
        # Initialize signal generator
        self.signal_generator = SignalGenerator(
            client=self.client,
            env_path=self.env_path
        )
        
        # Initialize paper trading engine
        self.paper_trading = PaperTradingSystem(
            client=self.client
        )
        
        # Connect signal generator to paper trading engine
        self._connect_components()
    
    def _connect_components(self):
        """Connect signal generator to paper trading engine"""
        # Create decision handler
        def handle_decision(symbol, decision):
            if decision:
                # Record decision
                self.results["decisions"].append(decision)
                self.results["metrics"]["decision_count"] += 1
                
                # Execute paper trade
                trade_result = self._execute_paper_trade(decision)
                
                if trade_result:
                    # Record trade
                    self.results["trades"].append(trade_result)
                    self.results["metrics"]["trade_count"] += 1
                    
                    # Update metrics
                    profit_loss = trade_result.get("profit_loss", 0.0)
                    self.results["metrics"]["total_profit_loss"] += profit_loss
                    
                    if profit_loss > 0:
                        self.results["metrics"]["profitable_trades"] += 1
                    elif profit_loss < 0:
                        self.results["metrics"]["losing_trades"] += 1
                    
                    # Update win rate
                    if self.results["metrics"]["trade_count"] > 0:
                        self.results["metrics"]["win_rate"] = (
                            self.results["metrics"]["profitable_trades"] / 
                            self.results["metrics"]["trade_count"]
                        ) * 100
                    
                    # Record latency
                    latency = trade_result.get("latency_ms", 0.0)
                    if latency > 0:
                        # Update average latency
                        self.results["metrics"]["avg_latency_ms"] = (
                            (self.results["metrics"]["avg_latency_ms"] * (self.results["metrics"]["trade_count"] - 1)) +
                            latency
                        ) / self.results["metrics"]["trade_count"]
                        
                        # Update max latency
                        if latency > self.results["metrics"]["max_latency_ms"]:
                            self.results["metrics"]["max_latency_ms"] = latency
                    
                    # Record balance
                    self.results["balances"].append({
                        "timestamp": int(time.time() * 1000),
                        "balances": self.paper_trading.get_all_balances(),
                        "equity": self._calculate_total_equity()
                    })
                    
                    # Record session metrics
                    session = decision.get("session")
                    if session:
                        if session not in self.results["session_metrics"]:
                            self.results["session_metrics"][session] = {
                                "trade_count": 0,
                                "profitable_trades": 0,
                                "losing_trades": 0,
                                "total_profit_loss": 0.0,
                                "win_rate": 0.0
                            }
                        
                        # Update session metrics
                        self.results["session_metrics"][session]["trade_count"] += 1
                        self.results["session_metrics"][session]["total_profit_loss"] += profit_loss
                        
                        if profit_loss > 0:
                            self.results["session_metrics"][session]["profitable_trades"] += 1
                        elif profit_loss < 0:
                            self.results["session_metrics"][session]["losing_trades"] += 1
                        
                        # Update session win rate
                        if self.results["session_metrics"][session]["trade_count"] > 0:
                            self.results["session_metrics"][session]["win_rate"] = (
                                self.results["session_metrics"][session]["profitable_trades"] / 
                                self.results["session_metrics"][session]["trade_count"]
                            ) * 100
        
        # Create signal handler
        def handle_signals(signals):
            # Record signals
            for signal in signals:
                self.results["signals"].append(signal)
                self.results["metrics"]["signal_count"] += 1
            
            # Process signals for each symbol
            for symbol in self.config["symbols"]:
                symbol_signals = [s for s in signals if s["symbol"] == symbol]
                if symbol_signals:
                    # Make trading decision
                    decision = self.signal_generator.make_trading_decision(symbol, symbol_signals)
                    
                    # Handle decision
                    if decision:
                        handle_decision(symbol, decision)
        
        # Set up signal processing thread
        def signal_processing_thread():
            while not self.stop_event.is_set():
                try:
                    # Get recent signals
                    signals = []
                    for symbol in self.config["symbols"]:
                        symbol_signals = self.signal_generator.get_recent_signals(10, symbol)
                        signals.extend(symbol_signals)
                    
                    # Handle signals
                    if signals:
                        handle_signals(signals)
                    
                    # Sleep for a bit
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error in signal processing thread: {str(e)}")
                    time.sleep(1)
        
        # Start signal processing thread
        signal_thread = Thread(target=signal_processing_thread)
        signal_thread.daemon = True
        signal_thread.start()
    
    def _execute_paper_trade(self, decision):
        """Execute a paper trade based on a decision"""
        try:
            start_time = time.time()
            
            # Extract decision parameters - using 'side' consistently
            symbol = decision["symbol"]
            side = decision["side"]  # Using 'side' key consistently
            order_type = decision.get("order_type", "LIMIT")
            quantity = decision["size"]
            price = decision.get("price")
            
            # Log the decision for debugging
            logger.debug(f"Executing paper trade: {side} {quantity} {symbol} @ {price}")
            
            # Place order
            order = self.paper_trading.place_order(
                symbol=symbol,
                side=side,  # Using 'side' key consistently
                order_type=order_type,
                quantity=quantity,
                price=price
            )
            
            if not order:
                logger.warning(f"Failed to place paper order: {side} {quantity} {symbol}")
                return None
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Create trade result
            trade_result = {
                "timestamp": int(time.time() * 1000),
                "symbol": symbol,
                "side": side,  # Using 'side' key consistently
                "quantity": quantity,
                "price": price,
                "order_id": order["orderId"],
                "latency_ms": latency_ms,
                "profit_loss": 0.0,  # Will be updated later
                "session": decision.get("session")
            }
            
            logger.info(f"Paper trade executed: {side} {quantity} {symbol} @ {price}")
            return trade_result
            
        except Exception as e:
            logger.error(f"Error executing paper trade: {str(e)}")
            return None
    
    def _calculate_total_equity(self):
        """Calculate total equity based on current balances and prices"""
        try:
            total_equity = 0.0
            
            # Get balances
            balances = self.paper_trading.get_all_balances()
            
            # Add USDC directly
            if "USDC" in balances:
                total_equity += balances["USDC"]
            
            # Convert other assets to USDC
            for asset, amount in balances.items():
                if asset != "USDC" and amount > 0:
                    # Try to get price
                    symbol = f"{asset}USDC"
                    try:
                        price = self.client.get_ticker_price(symbol)
                        if price:
                            total_equity += amount * float(price["price"])
                    except:
                        pass
            
            return total_equity
            
        except Exception as e:
            logger.error(f"Error calculating total equity: {str(e)}")
            return 0.0
    
    def _monitoring_loop(self):
        """Background thread for monitoring test progress"""
        last_metrics_time = time.time()
        
        while not self.stop_event.is_set() and self.running:
            try:
                # Sleep for a bit
                time.sleep(1)
                
                # Process open orders
                self.paper_trading.process_open_orders()
                
                # Collect metrics periodically
                current_time = time.time()
                if current_time - last_metrics_time >= self.config["metrics_interval"]:
                    self._collect_metrics()
                    last_metrics_time = current_time
                    
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(5)
    
    def _collect_metrics(self):
        """Collect performance metrics"""
        try:
            # Get current balances
            balances = self.paper_trading.get_all_balances()
            
            # Calculate total equity
            equity = self._calculate_total_equity()
            
            # Record balance snapshot
            self.results["balances"].append({
                "timestamp": int(time.time() * 1000),
                "balances": balances,
                "equity": equity
            })
            
            # Log metrics
            logger.info(f"Current equity: {equity:.2f} USDC")
            logger.info(f"Signals: {self.results['metrics']['signal_count']}, "
                       f"Decisions: {self.results['metrics']['decision_count']}, "
                       f"Trades: {self.results['metrics']['trade_count']}")
            
            if self.results["metrics"]["trade_count"] > 0:
                logger.info(f"Win rate: {self.results['metrics']['win_rate']:.2f}%, "
                           f"P&L: {self.results['metrics']['total_profit_loss']:.2f} USDC")
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {str(e)}")
    
    def _generate_report(self):
        """Generate periodic report"""
        try:
            # Calculate test duration
            duration = int(time.time() * 1000) - self.results["start_time"]
            duration_minutes = duration / (1000 * 60)
            
            # Log report
            logger.info(f"=== Test Report ({duration_minutes:.1f} minutes) ===")
            logger.info(f"Signals generated: {self.results['metrics']['signal_count']}")
            logger.info(f"Trading decisions: {self.results['metrics']['decision_count']}")
            logger.info(f"Trades executed: {self.results['metrics']['trade_count']}")
            
            if self.results["metrics"]["trade_count"] > 0:
                logger.info(f"Win rate: {self.results['metrics']['win_rate']:.2f}%")
                logger.info(f"Total P&L: {self.results['metrics']['total_profit_loss']:.2f} USDC")
                logger.info(f"Average latency: {self.results['metrics']['avg_latency_ms']:.2f} ms")
            
            # Log session metrics
            for session, metrics in self.results["session_metrics"].items():
                if metrics["trade_count"] > 0:
                    logger.info(f"Session {session}: {metrics['trade_count']} trades, "
                               f"Win rate: {metrics['win_rate']:.2f}%, "
                               f"P&L: {metrics['total_profit_loss']:.2f} USDC")
            
            # Log current equity
            if self.results["balances"]:
                latest_balance = self.results["balances"][-1]
                logger.info(f"Current equity: {latest_balance['equity']:.2f} USDC")
            
            logger.info("=" * 40)
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
    
    def _generate_final_report(self):
        """Generate final test report"""
        try:
            # Calculate test duration
            duration = self.results["duration"]
            duration_minutes = duration / (1000 * 60)
            
            # Log report
            logger.info(f"=== Final Test Report ({duration_minutes:.1f} minutes) ===")
            logger.info(f"Signals generated: {self.results['metrics']['signal_count']}")
            logger.info(f"Trading decisions: {self.results['metrics']['decision_count']}")
            logger.info(f"Trades executed: {self.results['metrics']['trade_count']}")
            
            if self.results["metrics"]["trade_count"] > 0:
                logger.info(f"Win rate: {self.results['metrics']['win_rate']:.2f}%")
                logger.info(f"Total P&L: {self.results['metrics']['total_profit_loss']:.2f} USDC")
                logger.info(f"Average latency: {self.results['metrics']['avg_latency_ms']:.2f} ms")
                logger.info(f"Maximum latency: {self.results['metrics']['max_latency_ms']:.2f} ms")
            
            # Log session metrics
            for session, metrics in self.results["session_metrics"].items():
                if metrics["trade_count"] > 0:
                    logger.info(f"Session {session}: {metrics['trade_count']} trades, "
                               f"Win rate: {metrics['win_rate']:.2f}%, "
                               f"P&L: {metrics['total_profit_loss']:.2f} USDC")
            
            # Log final equity
            if self.results["balances"]:
                initial_balance = self.config["initial_balance"].get("USDC", 0)
                final_balance = self.results["balances"][-1]["equity"]
                total_return = final_balance - initial_balance
                percent_return = (total_return / initial_balance) * 100 if initial_balance > 0 else 0
                
                logger.info(f"Initial equity: {initial_balance:.2f} USDC")
                logger.info(f"Final equity: {final_balance:.2f} USDC")
                logger.info(f"Total return: {total_return:.2f} USDC ({percent_return:.2f}%)")
                
                # Annualized return
                if duration_minutes > 0:
                    minutes_per_year = 365 * 24 * 60
                    annualized_return = ((1 + percent_return / 100) ** (minutes_per_year / duration_minutes) - 1) * 100
                    logger.info(f"Annualized return: {annualized_return:.2f}%")
            
            logger.info("=" * 40)
            
        except Exception as e:
            logger.error(f"Error generating final report: {str(e)}")
    
    def _save_results(self):
        """Save test results to file"""
        try:
            # Create results directory if it doesn't exist
            os.makedirs("results", exist_ok=True)
            
            # Generate timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save results to JSON file
            results_file = f"results/test_results_{timestamp}.json"
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2)
            
            logger.info(f"Test results saved to {results_file}")
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
    
    def _plot_results(self):
        """Plot test results"""
        try:
            # Create plots directory if it doesn't exist
            os.makedirs("plots", exist_ok=True)
            
            # Generate timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Plot equity curve
            if self.results["balances"]:
                # Extract data
                timestamps = [b["timestamp"] for b in self.results["balances"]]
                equity = [b["equity"] for b in self.results["balances"]]
                
                # Convert timestamps to datetime
                dates = [datetime.fromtimestamp(ts / 1000) for ts in timestamps]
                
                # Create figure
                plt.figure(figsize=(12, 6))
                plt.plot(dates, equity)
                plt.title("Equity Curve")
                plt.xlabel("Time")
                plt.ylabel("Equity (USDC)")
                plt.grid(True)
                
                # Save figure
                equity_plot_file = f"plots/equity_curve_{timestamp}.png"
                plt.savefig(equity_plot_file)
                plt.close()
                
                logger.info(f"Equity curve saved to {equity_plot_file}")
            
            # Plot session performance
            if self.results["session_metrics"]:
                # Extract data
                sessions = list(self.results["session_metrics"].keys())
                win_rates = [self.results["session_metrics"][s]["win_rate"] for s in sessions]
                profits = [self.results["session_metrics"][s]["total_profit_loss"] for s in sessions]
                trade_counts = [self.results["session_metrics"][s]["trade_count"] for s in sessions]
                
                # Create figure
                plt.figure(figsize=(12, 8))
                
                # Plot win rates
                plt.subplot(2, 1, 1)
                plt.bar(sessions, win_rates)
                plt.title("Win Rate by Session")
                plt.ylabel("Win Rate (%)")
                plt.grid(True)
                
                # Plot profits
                plt.subplot(2, 1, 2)
                plt.bar(sessions, profits)
                plt.title("Profit/Loss by Session")
                plt.ylabel("P&L (USDC)")
                plt.grid(True)
                
                # Adjust layout
                plt.tight_layout()
                
                # Save figure
                session_plot_file = f"plots/session_performance_{timestamp}.png"
                plt.savefig(session_plot_file)
                plt.close()
                
                logger.info(f"Session performance plot saved to {session_plot_file}")
            
        except Exception as e:
            logger.error(f"Error plotting results: {str(e)}")
    
    def _stop_components(self):
        """Stop all components"""
        try:
            # Stop signal generator
            if self.signal_generator:
                self.signal_generator.stop()
            
            logger.info("All components stopped")
            
        except Exception as e:
            logger.error(f"Error stopping components: {str(e)}")


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Integration Test Runner')
    parser.add_argument('--env', default=".env-secure/.env", help='Path to .env file')
    parser.add_argument('--config', default=None, help='Path to config file')
    parser.add_argument('--duration', type=int, default=300, help='Test duration in seconds')
    parser.add_argument('--symbols', default="BTCUSDC,ETHUSDC", help='Comma-separated list of symbols')
    
    args = parser.parse_args()
    
    # Create test runner
    test_runner = IntegrationTestRunner(env_path=args.env, config_path=args.config)
    
    # Update configuration
    test_runner.config["test_duration"] = args.duration
    test_runner.config["symbols"] = args.symbols.split(",")
    
    # Run test
    try:
        test_runner.run_test()
    except KeyboardInterrupt:
        print("Test interrupted by user")
        test_runner.stop_test()
