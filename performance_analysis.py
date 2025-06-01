#!/usr/bin/env python
"""
Performance Analysis for Flash Trading System

This script analyzes the performance of the flash trading system based on test results,
providing detailed metrics, visualizations, and recommendations for optimization.
"""

import os
import json
import logging
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("performance_analysis")

class PerformanceAnalyzer:
    """Analyzes flash trading system performance"""
    
    def __init__(self, results_file=None):
        """Initialize analyzer with results file"""
        self.results_file = results_file
        self.results = None
        self.metrics = {}
        self.session_metrics = {}
        
        # Load results if provided
        if results_file and os.path.exists(results_file):
            self._load_results(results_file)
    
    def _load_results(self, results_file):
        """Load test results from file"""
        try:
            with open(results_file, 'r') as f:
                self.results = json.load(f)
            
            logger.info(f"Results loaded from {results_file}")
            
            # Extract metrics
            if "metrics" in self.results:
                self.metrics = self.results["metrics"]
            
            # Extract session metrics
            if "session_metrics" in self.results:
                self.session_metrics = self.results["session_metrics"]
            
        except Exception as e:
            logger.error(f"Error loading results: {str(e)}")
    
    def analyze_signals(self):
        """Analyze signal generation performance"""
        if not self.results or "signals" not in self.results:
            logger.warning("No signals found in results")
            return {}
        
        signals = self.results["signals"]
        
        # Count signals by type and symbol
        signal_counts = defaultdict(lambda: defaultdict(int))
        for signal in signals:
            signal_type = signal.get("type", "UNKNOWN")
            symbol = signal.get("symbol", "UNKNOWN")
            signal_counts[symbol][signal_type] += 1
        
        # Calculate signal metrics
        signal_metrics = {
            "total_signals": len(signals),
            "signals_per_minute": 0,
            "signal_counts_by_symbol": dict(signal_counts),
            "signal_types": {}
        }
        
        # Calculate signals per minute
        if self.results.get("duration", 0) > 0:
            duration_minutes = self.results["duration"] / (1000 * 60)
            if duration_minutes > 0:
                signal_metrics["signals_per_minute"] = len(signals) / duration_minutes
        
        # Calculate signal type distribution
        signal_types = defaultdict(int)
        for signal in signals:
            signal_type = signal.get("type", "UNKNOWN")
            signal_types[signal_type] += 1
        
        signal_metrics["signal_types"] = dict(signal_types)
        
        return signal_metrics
    
    def analyze_decisions(self):
        """Analyze trading decision performance"""
        if not self.results or "decisions" not in self.results:
            logger.warning("No decisions found in results")
            return {}
        
        decisions = self.results["decisions"]
        
        # Count decisions by side and symbol
        decision_counts = defaultdict(lambda: defaultdict(int))
        for decision in decisions:
            side = decision.get("side", "UNKNOWN")
            symbol = decision.get("symbol", "UNKNOWN")
            decision_counts[symbol][side] += 1
        
        # Calculate decision metrics
        decision_metrics = {
            "total_decisions": len(decisions),
            "decisions_per_minute": 0,
            "decision_counts_by_symbol": dict(decision_counts),
            "decision_sides": {}
        }
        
        # Calculate decisions per minute
        if self.results.get("duration", 0) > 0:
            duration_minutes = self.results["duration"] / (1000 * 60)
            if duration_minutes > 0:
                decision_metrics["decisions_per_minute"] = len(decisions) / duration_minutes
        
        # Calculate decision side distribution
        decision_sides = defaultdict(int)
        for decision in decisions:
            side = decision.get("side", "UNKNOWN")
            decision_sides[side] += 1
        
        decision_metrics["decision_sides"] = dict(decision_sides)
        
        # Calculate signal to decision ratio
        if "signals" in self.results and len(self.results["signals"]) > 0:
            decision_metrics["signal_to_decision_ratio"] = len(decisions) / len(self.results["signals"])
        
        return decision_metrics
    
    def analyze_trades(self):
        """Analyze trade execution performance"""
        if not self.results or "trades" not in self.results:
            logger.warning("No trades found in results")
            return {}
        
        trades = self.results["trades"]
        
        # Count trades by side and symbol
        trade_counts = defaultdict(lambda: defaultdict(int))
        for trade in trades:
            side = trade.get("side", "UNKNOWN")
            symbol = trade.get("symbol", "UNKNOWN")
            trade_counts[symbol][side] += 1
        
        # Calculate trade metrics
        trade_metrics = {
            "total_trades": len(trades),
            "trades_per_minute": 0,
            "trade_counts_by_symbol": dict(trade_counts),
            "trade_sides": {},
            "avg_latency_ms": 0,
            "max_latency_ms": 0,
            "min_latency_ms": float('inf'),
            "profit_loss": 0.0,
            "win_rate": 0.0
        }
        
        # Calculate trades per minute
        if self.results.get("duration", 0) > 0:
            duration_minutes = self.results["duration"] / (1000 * 60)
            if duration_minutes > 0:
                trade_metrics["trades_per_minute"] = len(trades) / duration_minutes
        
        # Calculate trade side distribution
        trade_sides = defaultdict(int)
        for trade in trades:
            side = trade.get("side", "UNKNOWN")
            trade_sides[side] += 1
        
        trade_metrics["trade_sides"] = dict(trade_sides)
        
        # Calculate latency metrics
        latencies = [trade.get("latency_ms", 0) for trade in trades if "latency_ms" in trade]
        if latencies:
            trade_metrics["avg_latency_ms"] = sum(latencies) / len(latencies)
            trade_metrics["max_latency_ms"] = max(latencies)
            trade_metrics["min_latency_ms"] = min(latencies)
        else:
            trade_metrics["min_latency_ms"] = 0
        
        # Calculate profit/loss metrics
        profits = [trade.get("profit_loss", 0) for trade in trades if "profit_loss" in trade]
        if profits:
            trade_metrics["profit_loss"] = sum(profits)
            trade_metrics["avg_profit_per_trade"] = sum(profits) / len(profits)
            trade_metrics["profitable_trades"] = sum(1 for p in profits if p > 0)
            trade_metrics["losing_trades"] = sum(1 for p in profits if p < 0)
            
            if len(profits) > 0:
                trade_metrics["win_rate"] = (trade_metrics["profitable_trades"] / len(profits)) * 100
        
        # Calculate decision to trade ratio
        if "decisions" in self.results and len(self.results["decisions"]) > 0:
            trade_metrics["decision_to_trade_ratio"] = len(trades) / len(self.results["decisions"])
        
        return trade_metrics
    
    def analyze_session_performance(self):
        """Analyze performance by trading session"""
        if not self.session_metrics:
            logger.warning("No session metrics found in results")
            return {}
        
        # Calculate session performance metrics
        session_performance = {}
        for session, metrics in self.session_metrics.items():
            session_performance[session] = {
                "trade_count": metrics.get("trade_count", 0),
                "win_rate": metrics.get("win_rate", 0.0),
                "total_profit_loss": metrics.get("total_profit_loss", 0.0)
            }
            
            # Calculate average profit per trade
            if metrics.get("trade_count", 0) > 0:
                session_performance[session]["avg_profit_per_trade"] = (
                    metrics.get("total_profit_loss", 0.0) / metrics.get("trade_count", 1)
                )
            else:
                session_performance[session]["avg_profit_per_trade"] = 0.0
        
        return session_performance
    
    def analyze_equity_curve(self):
        """Analyze equity curve performance"""
        if not self.results or "balances" not in self.results:
            logger.warning("No balance data found in results")
            return {}
        
        balances = self.results["balances"]
        
        # Calculate equity curve metrics
        equity_metrics = {
            "initial_equity": 0.0,
            "final_equity": 0.0,
            "max_equity": 0.0,
            "min_equity": float('inf'),
            "total_return": 0.0,
            "percent_return": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_percent": 0.0
        }
        
        # Extract equity values
        equity_values = [b.get("equity", 0.0) for b in balances if "equity" in b]
        timestamps = [b.get("timestamp", 0) for b in balances if "timestamp" in b]
        
        if not equity_values:
            logger.warning("No equity values found in balance data")
            return equity_metrics
        
        # Calculate basic metrics
        equity_metrics["initial_equity"] = equity_values[0]
        equity_metrics["final_equity"] = equity_values[-1]
        equity_metrics["max_equity"] = max(equity_values)
        equity_metrics["min_equity"] = min(equity_values)
        equity_metrics["total_return"] = equity_values[-1] - equity_values[0]
        
        if equity_values[0] > 0:
            equity_metrics["percent_return"] = (equity_metrics["total_return"] / equity_values[0]) * 100
        
        # Calculate drawdown
        max_value = equity_values[0]
        max_drawdown = 0.0
        max_drawdown_percent = 0.0
        
        for value in equity_values:
            if value > max_value:
                max_value = value
            
            drawdown = max_value - value
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                max_drawdown_percent = (drawdown / max_value) * 100 if max_value > 0 else 0.0
        
        equity_metrics["max_drawdown"] = max_drawdown
        equity_metrics["max_drawdown_percent"] = max_drawdown_percent
        
        # Calculate annualized return
        if len(timestamps) >= 2:
            duration_ms = timestamps[-1] - timestamps[0]
            duration_years = duration_ms / (1000 * 60 * 60 * 24 * 365)
            
            if duration_years > 0 and equity_values[0] > 0:
                annualized_return = ((equity_values[-1] / equity_values[0]) ** (1 / duration_years)) - 1
                equity_metrics["annualized_return"] = annualized_return * 100
        
        return equity_metrics
    
    def analyze_performance(self):
        """Analyze overall system performance"""
        # Analyze signals
        signal_metrics = self.analyze_signals()
        
        # Analyze decisions
        decision_metrics = self.analyze_decisions()
        
        # Analyze trades
        trade_metrics = self.analyze_trades()
        
        # Analyze session performance
        session_performance = self.analyze_session_performance()
        
        # Analyze equity curve
        equity_metrics = self.analyze_equity_curve()
        
        # Combine all metrics
        performance_metrics = {
            "signal_metrics": signal_metrics,
            "decision_metrics": decision_metrics,
            "trade_metrics": trade_metrics,
            "session_performance": session_performance,
            "equity_metrics": equity_metrics
        }
        
        return performance_metrics
    
    def generate_report(self, output_file=None):
        """Generate performance report"""
        # Analyze performance
        performance_metrics = self.analyze_performance()
        
        # Create report
        report = {
            "timestamp": int(datetime.now().timestamp() * 1000),
            "results_file": self.results_file,
            "performance_metrics": performance_metrics
        }
        
        # Save report if output file provided
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    json.dump(report, f, indent=2)
                
                logger.info(f"Performance report saved to {output_file}")
            except Exception as e:
                logger.error(f"Error saving performance report: {str(e)}")
        
        return report
    
    def plot_performance(self, output_dir="plots"):
        """Generate performance plots"""
        if not self.results:
            logger.warning("No results to plot")
            return
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Plot equity curve
        self._plot_equity_curve(output_dir, timestamp)
        
        # Plot session performance
        self._plot_session_performance(output_dir, timestamp)
        
        # Plot signal and decision distribution
        self._plot_signal_decision_distribution(output_dir, timestamp)
        
        # Plot latency distribution
        self._plot_latency_distribution(output_dir, timestamp)
    
    def _plot_equity_curve(self, output_dir, timestamp):
        """Plot equity curve"""
        if not self.results or "balances" not in self.results:
            return
        
        balances = self.results["balances"]
        
        # Extract equity values
        equity_values = [b.get("equity", 0.0) for b in balances if "equity" in b]
        timestamps = [b.get("timestamp", 0) for b in balances if "timestamp" in b]
        
        if not equity_values or not timestamps:
            return
        
        # Convert timestamps to datetime
        dates = [datetime.fromtimestamp(ts / 1000) for ts in timestamps]
        
        # Create figure
        plt.figure(figsize=(12, 6))
        plt.plot(dates, equity_values)
        plt.title("Equity Curve")
        plt.xlabel("Time")
        plt.ylabel("Equity (USDC)")
        plt.grid(True)
        
        # Save figure
        equity_plot_file = os.path.join(output_dir, f"equity_curve_{timestamp}.png")
        plt.savefig(equity_plot_file)
        plt.close()
        
        logger.info(f"Equity curve saved to {equity_plot_file}")
    
    def _plot_session_performance(self, output_dir, timestamp):
        """Plot session performance"""
        if not self.session_metrics:
            return
        
        # Extract data
        sessions = list(self.session_metrics.keys())
        win_rates = [self.session_metrics[s].get("win_rate", 0.0) for s in sessions]
        profits = [self.session_metrics[s].get("total_profit_loss", 0.0) for s in sessions]
        trade_counts = [self.session_metrics[s].get("trade_count", 0) for s in sessions]
        
        if not sessions:
            return
        
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
        session_plot_file = os.path.join(output_dir, f"session_performance_{timestamp}.png")
        plt.savefig(session_plot_file)
        plt.close()
        
        logger.info(f"Session performance plot saved to {session_plot_file}")
    
    def _plot_signal_decision_distribution(self, output_dir, timestamp):
        """Plot signal and decision distribution"""
        if not self.results:
            return
        
        # Extract signals
        signals = self.results.get("signals", [])
        signal_types = defaultdict(int)
        for signal in signals:
            signal_type = signal.get("type", "UNKNOWN")
            signal_types[signal_type] += 1
        
        # Extract decisions
        decisions = self.results.get("decisions", [])
        decision_sides = defaultdict(int)
        for decision in decisions:
            side = decision.get("side", "UNKNOWN")
            decision_sides[side] += 1
        
        if not signal_types and not decision_sides:
            return
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot signal types
        if signal_types:
            plt.subplot(2, 1, 1)
            plt.bar(signal_types.keys(), signal_types.values())
            plt.title("Signal Type Distribution")
            plt.ylabel("Count")
            plt.grid(True)
        
        # Plot decision sides
        if decision_sides:
            plt.subplot(2, 1, 2)
            plt.bar(decision_sides.keys(), decision_sides.values())
            plt.title("Decision Side Distribution")
            plt.ylabel("Count")
            plt.grid(True)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        distribution_plot_file = os.path.join(output_dir, f"signal_decision_distribution_{timestamp}.png")
        plt.savefig(distribution_plot_file)
        plt.close()
        
        logger.info(f"Signal and decision distribution plot saved to {distribution_plot_file}")
    
    def _plot_latency_distribution(self, output_dir, timestamp):
        """Plot latency distribution"""
        if not self.results or "trades" not in self.results:
            return
        
        trades = self.results["trades"]
        
        # Extract latencies
        latencies = [trade.get("latency_ms", 0) for trade in trades if "latency_ms" in trade]
        
        if not latencies:
            return
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot latency histogram
        plt.hist(latencies, bins=20)
        plt.title("Latency Distribution")
        plt.xlabel("Latency (ms)")
        plt.ylabel("Count")
        plt.grid(True)
        
        # Save figure
        latency_plot_file = os.path.join(output_dir, f"latency_distribution_{timestamp}.png")
        plt.savefig(latency_plot_file)
        plt.close()
        
        logger.info(f"Latency distribution plot saved to {latency_plot_file}")


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Performance Analysis')
    parser.add_argument('--results', required=True, help='Path to results file')
    parser.add_argument('--output', default=None, help='Path to output report file')
    parser.add_argument('--plots', default="plots", help='Directory for output plots')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = PerformanceAnalyzer(results_file=args.results)
    
    # Generate report
    report = analyzer.generate_report(output_file=args.output)
    
    # Generate plots
    analyzer.plot_performance(output_dir=args.plots)
    
    # Print summary
    print("Performance Analysis Summary:")
    
    if "performance_metrics" in report and "equity_metrics" in report["performance_metrics"]:
        equity_metrics = report["performance_metrics"]["equity_metrics"]
        print(f"Initial Equity: {equity_metrics.get('initial_equity', 0.0):.2f} USDC")
        print(f"Final Equity: {equity_metrics.get('final_equity', 0.0):.2f} USDC")
        print(f"Total Return: {equity_metrics.get('total_return', 0.0):.2f} USDC ({equity_metrics.get('percent_return', 0.0):.2f}%)")
        print(f"Max Drawdown: {equity_metrics.get('max_drawdown', 0.0):.2f} USDC ({equity_metrics.get('max_drawdown_percent', 0.0):.2f}%)")
    
    if "performance_metrics" in report and "trade_metrics" in report["performance_metrics"]:
        trade_metrics = report["performance_metrics"]["trade_metrics"]
        print(f"Total Trades: {trade_metrics.get('total_trades', 0)}")
        print(f"Win Rate: {trade_metrics.get('win_rate', 0.0):.2f}%")
        print(f"Average Latency: {trade_metrics.get('avg_latency_ms', 0.0):.2f} ms")
