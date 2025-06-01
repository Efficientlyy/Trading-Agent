#!/usr/bin/env python
"""
Extended Signal Analysis Script

This script runs the Trading-Agent system for an extended period,
captures all signals and decisions, and generates a comprehensive report.
"""

import os
import sys
import time
import json
import logging
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("signal_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("signal_analysis")

class SignalAnalyzer:
    """Analyzes trading signals and decisions from the Trading-Agent system"""
    
    def __init__(self, output_dir="./signal_analysis"):
        """Initialize the signal analyzer"""
        self.output_dir = output_dir
        self.signals = []
        self.decisions = []
        self.market_states = {}
        self.start_time = None
        self.end_time = None
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")
    
    def run_system(self, duration_minutes=60, env_path=".env-secure/.env"):
        """Run the Trading-Agent system for a specified duration
        
        Args:
            duration_minutes: Duration to run the system in minutes
            env_path: Path to the environment file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Import required modules
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from flash_trading import FlashTradingSystem
            
            # Load environment variables
            load_dotenv(env_path)
            
            # Record start time
            self.start_time = datetime.now()
            logger.info(f"Starting system run at {self.start_time}")
            
            # Create flash trading system with signal capture
            flash_trading = FlashTradingSystem(env_path=env_path)
            
            # Monkey patch signal generation to capture signals
            original_generate_signals = flash_trading.signal_generator.generate_signals
            
            def patched_generate_signals(symbol):
                signals = original_generate_signals(symbol)
                if signals:
                    for signal in signals:
                        signal['capture_time'] = datetime.now().isoformat()
                        self.signals.append(signal)
                        logger.info(f"Captured signal: {signal}")
                return signals
            
            flash_trading.signal_generator.generate_signals = patched_generate_signals
            
            # Monkey patch decision making to capture decisions
            original_execute_paper_trading_decision = flash_trading._execute_paper_trading_decision
            
            def patched_execute_paper_trading_decision(decision):
                # Capture the decision before execution
                if decision:
                    decision_copy = decision.copy() if isinstance(decision, dict) else {"raw": str(decision)}
                    decision_copy['capture_time'] = datetime.now().isoformat()
                    self.decisions.append(decision_copy)
                    logger.info(f"Captured decision: {decision_copy}")
                
                # Call the original method
                return original_execute_paper_trading_decision(decision)
            
            flash_trading._execute_paper_trading_decision = patched_execute_paper_trading_decision
            
            # Run for specified duration
            logger.info(f"Running system for {duration_minutes} minutes")
            flash_trading.run_for_duration(duration_minutes * 60)
            
            # Record end time
            self.end_time = datetime.now()
            logger.info(f"Finished system run at {self.end_time}")
            
            # Capture final market states
            for symbol, state in flash_trading.signal_generator.market_states.items():
                self.market_states[symbol] = {
                    'bid_price': state.bid_price,
                    'ask_price': state.ask_price,
                    'mid_price': state.mid_price,
                    'spread': state.spread,
                    'spread_bps': state.spread_bps,
                    'order_imbalance': state.order_imbalance,
                    'momentum': state.momentum,
                    'volatility': state.volatility,
                    'trend': state.trend
                }
            
            # Save captured data
            self._save_data()
            
            return True
            
        except Exception as e:
            logger.error(f"Error running system: {str(e)}")
            return False
    
    def _save_data(self):
        """Save captured data to files"""
        try:
            # Save signals
            signals_file = os.path.join(self.output_dir, "signals.json")
            with open(signals_file, 'w') as f:
                json.dump(self.signals, f, indent=2)
            logger.info(f"Saved {len(self.signals)} signals to {signals_file}")
            
            # Save decisions
            decisions_file = os.path.join(self.output_dir, "decisions.json")
            with open(decisions_file, 'w') as f:
                json.dump(self.decisions, f, indent=2)
            logger.info(f"Saved {len(self.decisions)} decisions to {decisions_file}")
            
            # Save market states
            market_states_file = os.path.join(self.output_dir, "market_states.json")
            with open(market_states_file, 'w') as f:
                json.dump(self.market_states, f, indent=2)
            logger.info(f"Saved market states to {market_states_file}")
            
            # Save run info
            run_info = {
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'end_time': self.end_time.isoformat() if self.end_time else None,
                'duration_seconds': (self.end_time - self.start_time).total_seconds() if self.start_time and self.end_time else None,
                'signal_count': len(self.signals),
                'decision_count': len(self.decisions),
                'symbols': list(self.market_states.keys())
            }
            
            run_info_file = os.path.join(self.output_dir, "run_info.json")
            with open(run_info_file, 'w') as f:
                json.dump(run_info, f, indent=2)
            logger.info(f"Saved run info to {run_info_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            return False
    
    def analyze_signals(self):
        """Analyze captured signals and generate visualizations
        
        Returns:
            dict: Analysis results
        """
        try:
            if not self.signals:
                logger.warning("No signals to analyze")
                return {'signal_count': 0}
            
            # Convert signals to DataFrame for analysis
            signals_df = pd.DataFrame(self.signals)
            
            # Add datetime column
            if 'capture_time' in signals_df.columns:
                signals_df['datetime'] = pd.to_datetime(signals_df['capture_time'])
            elif 'timestamp' in signals_df.columns:
                signals_df['datetime'] = pd.to_datetime(signals_df['timestamp'], unit='ms')
            
            # Group signals by type and symbol
            signal_counts_by_type = signals_df.groupby('type').size().to_dict()
            signal_counts_by_symbol = signals_df.groupby('symbol').size().to_dict()
            signal_counts_by_source = signals_df.groupby('source').size().to_dict()
            
            # Calculate signal strength statistics
            strength_stats = {}
            if 'strength' in signals_df.columns:
                strength_stats = {
                    'mean': signals_df['strength'].mean(),
                    'median': signals_df['strength'].median(),
                    'min': signals_df['strength'].min(),
                    'max': signals_df['strength'].max(),
                    'std': signals_df['strength'].std()
                }
            
            # Generate visualizations
            self._generate_signal_visualizations(signals_df)
            
            # Compile analysis results
            analysis_results = {
                'signal_count': len(self.signals),
                'signal_counts_by_type': signal_counts_by_type,
                'signal_counts_by_symbol': signal_counts_by_symbol,
                'signal_counts_by_source': signal_counts_by_source,
                'strength_stats': strength_stats
            }
            
            # Save analysis results
            analysis_file = os.path.join(self.output_dir, "signal_analysis.json")
            with open(analysis_file, 'w') as f:
                json.dump(analysis_results, f, indent=2)
            logger.info(f"Saved signal analysis to {analysis_file}")
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error analyzing signals: {str(e)}")
            return {'signal_count': len(self.signals), 'error': str(e)}
    
    def _generate_signal_visualizations(self, signals_df):
        """Generate visualizations for signals
        
        Args:
            signals_df: DataFrame containing signals
        """
        try:
            if signals_df.empty:
                return
            
            # Create visualizations directory
            viz_dir = os.path.join(self.output_dir, "visualizations")
            if not os.path.exists(viz_dir):
                os.makedirs(viz_dir)
            
            # Signal count by type
            plt.figure(figsize=(10, 6))
            signals_df['type'].value_counts().plot(kind='bar')
            plt.title('Signal Count by Type')
            plt.xlabel('Signal Type')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'signal_count_by_type.png'))
            plt.close()
            
            # Signal count by source
            if 'source' in signals_df.columns:
                plt.figure(figsize=(10, 6))
                signals_df['source'].value_counts().plot(kind='bar')
                plt.title('Signal Count by Source')
                plt.xlabel('Signal Source')
                plt.ylabel('Count')
                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, 'signal_count_by_source.png'))
                plt.close()
            
            # Signal strength distribution
            if 'strength' in signals_df.columns:
                plt.figure(figsize=(10, 6))
                signals_df['strength'].hist(bins=20)
                plt.title('Signal Strength Distribution')
                plt.xlabel('Strength')
                plt.ylabel('Frequency')
                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, 'signal_strength_distribution.png'))
                plt.close()
            
            # Signal timeline
            if 'datetime' in signals_df.columns:
                plt.figure(figsize=(12, 6))
                signals_df.groupby([signals_df['datetime'].dt.floor('5min'), 'type']).size().unstack().plot()
                plt.title('Signal Timeline')
                plt.xlabel('Time')
                plt.ylabel('Count')
                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, 'signal_timeline.png'))
                plt.close()
            
            logger.info(f"Generated signal visualizations in {viz_dir}")
            
        except Exception as e:
            logger.error(f"Error generating signal visualizations: {str(e)}")
    
    def analyze_decisions(self):
        """Analyze captured trading decisions
        
        Returns:
            dict: Analysis results
        """
        try:
            if not self.decisions:
                logger.warning("No decisions to analyze")
                return {'decision_count': 0}
            
            # Convert decisions to DataFrame for analysis
            decisions_df = pd.DataFrame(self.decisions)
            
            # Add datetime column
            if 'capture_time' in decisions_df.columns:
                decisions_df['datetime'] = pd.to_datetime(decisions_df['capture_time'])
            elif 'timestamp' in decisions_df.columns:
                decisions_df['datetime'] = pd.to_datetime(decisions_df['timestamp'], unit='ms')
            
            # Group decisions by type and symbol
            decision_counts_by_side = decisions_df.groupby('side').size().to_dict() if 'side' in decisions_df.columns else {}
            decision_counts_by_symbol = decisions_df.groupby('symbol').size().to_dict() if 'symbol' in decisions_df.columns else {}
            decision_counts_by_order_type = decisions_df.groupby('order_type').size().to_dict() if 'order_type' in decisions_df.columns else {}
            
            # Calculate size statistics
            size_stats = {}
            if 'size' in decisions_df.columns:
                decisions_df['size'] = pd.to_numeric(decisions_df['size'], errors='coerce')
                size_stats = {
                    'mean': decisions_df['size'].mean(),
                    'median': decisions_df['size'].median(),
                    'min': decisions_df['size'].min(),
                    'max': decisions_df['size'].max(),
                    'std': decisions_df['size'].std()
                }
            
            # Generate visualizations
            self._generate_decision_visualizations(decisions_df)
            
            # Compile analysis results
            analysis_results = {
                'decision_count': len(self.decisions),
                'decision_counts_by_side': decision_counts_by_side,
                'decision_counts_by_symbol': decision_counts_by_symbol,
                'decision_counts_by_order_type': decision_counts_by_order_type,
                'size_stats': size_stats
            }
            
            # Save analysis results
            analysis_file = os.path.join(self.output_dir, "decision_analysis.json")
            with open(analysis_file, 'w') as f:
                json.dump(analysis_results, f, indent=2)
            logger.info(f"Saved decision analysis to {analysis_file}")
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error analyzing decisions: {str(e)}")
            return {'decision_count': len(self.decisions), 'error': str(e)}
    
    def _generate_decision_visualizations(self, decisions_df):
        """Generate visualizations for trading decisions
        
        Args:
            decisions_df: DataFrame containing decisions
        """
        try:
            if decisions_df.empty:
                return
            
            # Create visualizations directory
            viz_dir = os.path.join(self.output_dir, "visualizations")
            if not os.path.exists(viz_dir):
                os.makedirs(viz_dir)
            
            # Decision count by side
            if 'side' in decisions_df.columns:
                plt.figure(figsize=(10, 6))
                decisions_df['side'].value_counts().plot(kind='bar')
                plt.title('Decision Count by Side')
                plt.xlabel('Side')
                plt.ylabel('Count')
                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, 'decision_count_by_side.png'))
                plt.close()
            
            # Decision count by order type
            if 'order_type' in decisions_df.columns:
                plt.figure(figsize=(10, 6))
                decisions_df['order_type'].value_counts().plot(kind='bar')
                plt.title('Decision Count by Order Type')
                plt.xlabel('Order Type')
                plt.ylabel('Count')
                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, 'decision_count_by_order_type.png'))
                plt.close()
            
            # Decision size distribution
            if 'size' in decisions_df.columns:
                plt.figure(figsize=(10, 6))
                decisions_df['size'] = pd.to_numeric(decisions_df['size'], errors='coerce')
                decisions_df['size'].hist(bins=20)
                plt.title('Decision Size Distribution')
                plt.xlabel('Size')
                plt.ylabel('Frequency')
                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, 'decision_size_distribution.png'))
                plt.close()
            
            # Decision timeline
            if 'datetime' in decisions_df.columns and 'side' in decisions_df.columns:
                plt.figure(figsize=(12, 6))
                decisions_df.groupby([decisions_df['datetime'].dt.floor('5min'), 'side']).size().unstack().plot()
                plt.title('Decision Timeline')
                plt.xlabel('Time')
                plt.ylabel('Count')
                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, 'decision_timeline.png'))
                plt.close()
            
            logger.info(f"Generated decision visualizations in {viz_dir}")
            
        except Exception as e:
            logger.error(f"Error generating decision visualizations: {str(e)}")
    
    def generate_report(self):
        """Generate a comprehensive report on signals and decisions
        
        Returns:
            str: Path to the generated report
        """
        try:
            # Analyze signals and decisions
            signal_analysis = self.analyze_signals()
            decision_analysis = self.analyze_decisions()
            
            # Create report file
            report_file = os.path.join(self.output_dir, "signal_decision_report.md")
            
            with open(report_file, 'w') as f:
                # Report header
                f.write("# Trading-Agent System: Signal and Decision Analysis Report\n\n")
                f.write(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Run information
                f.write("## Run Information\n\n")
                if self.start_time and self.end_time:
                    duration = self.end_time - self.start_time
                    f.write(f"- **Start Time:** {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"- **End Time:** {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"- **Duration:** {duration.total_seconds()/60:.2f} minutes\n")
                
                f.write(f"- **Symbols Analyzed:** {', '.join(self.market_states.keys())}\n")
                f.write(f"- **Total Signals Generated:** {len(self.signals)}\n")
                f.write(f"- **Total Trading Decisions:** {len(self.decisions)}\n\n")
                
                # Executive summary
                f.write("## Executive Summary\n\n")
                
                if len(self.signals) == 0:
                    f.write("During the test period, no trading signals were generated. This could be due to:\n\n")
                    f.write("1. Current market conditions not meeting the configured thresholds\n")
                    f.write("2. Conservative signal generation parameters\n")
                    f.write("3. Insufficient test duration for proper market analysis\n\n")
                    f.write("Consider adjusting signal thresholds or running tests during more volatile market periods.\n\n")
                else:
                    f.write(f"During the test period, {len(self.signals)} trading signals were generated, ")
                    f.write(f"resulting in {len(self.decisions)} trading decisions. ")
                    
                    # Signal type distribution
                    if 'signal_counts_by_type' in signal_analysis and signal_analysis['signal_counts_by_type']:
                        f.write("The most common signal type was ")
                        most_common_type = max(signal_analysis['signal_counts_by_type'].items(), key=lambda x: x[1])
                        f.write(f"**{most_common_type[0]}** ({most_common_type[1]} signals, ")
                        f.write(f"{most_common_type[1]/len(self.signals)*100:.1f}% of total). ")
                    
                    # Signal source distribution
                    if 'signal_counts_by_source' in signal_analysis and signal_analysis['signal_counts_by_source']:
                        f.write("The primary signal source was ")
                        most_common_source = max(signal_analysis['signal_counts_by_source'].items(), key=lambda x: x[1])
                        f.write(f"**{most_common_source[0]}** ({most_common_source[1]} signals, ")
                        f.write(f"{most_common_source[1]/len(self.signals)*100:.1f}% of total).\n\n")
                    else:
                        f.write("\n\n")
                    
                    # Decision distribution
                    if len(self.decisions) > 0 and 'decision_counts_by_side' in decision_analysis and decision_analysis['decision_counts_by_side']:
                        buy_count = decision_analysis['decision_counts_by_side'].get('BUY', 0)
                        sell_count = decision_analysis['decision_counts_by_side'].get('SELL', 0)
                        buy_pct = buy_count / len(self.decisions) * 100 if len(self.decisions) > 0 else 0
                        sell_pct = sell_count / len(self.decisions) * 100 if len(self.decisions) > 0 else 0
                        
                        f.write(f"Trading decisions were distributed as **{buy_count} BUY** ({buy_pct:.1f}%) and ")
                        f.write(f"**{sell_count} SELL** ({sell_pct:.1f}%) orders.\n\n")
                
                # Market conditions
                f.write("## Market Conditions\n\n")
                
                if self.market_states:
                    f.write("### Market State Summary\n\n")
                    f.write("| Symbol | Bid Price | Ask Price | Spread (bps) | Order Imbalance | Momentum | Volatility |\n")
                    f.write("|--------|-----------|-----------|--------------|-----------------|----------|------------|\n")
                    
                    for symbol, state in self.market_states.items():
                        bid_price = state.get('bid_price', 'N/A')
                        ask_price = state.get('ask_price', 'N/A')
                        spread_bps = state.get('spread_bps', 'N/A')
                        order_imbalance = state.get('order_imbalance', 'N/A')
                        momentum = state.get('momentum', 'N/A')
                        volatility = state.get('volatility', 'N/A')
                        
                        f.write(f"| {symbol} | {bid_price} | {ask_price} | {spread_bps} | {order_imbalance:.4f} | {momentum:.4f} | {volatility:.4f} |\n")
                    
                    f.write("\n")
                else:
                    f.write("No market state information available.\n\n")
                
                # Signal analysis
                f.write("## Signal Analysis\n\n")
                
                if len(self.signals) == 0:
                    f.write("No signals were generated during the test period.\n\n")
                else:
                    # Signal counts by type
                    if 'signal_counts_by_type' in signal_analysis and signal_analysis['signal_counts_by_type']:
                        f.write("### Signal Distribution by Type\n\n")
                        f.write("| Signal Type | Count | Percentage |\n")
                        f.write("|------------|-------|------------|\n")
                        
                        for signal_type, count in signal_analysis['signal_counts_by_type'].items():
                            percentage = count / len(self.signals) * 100
                            f.write(f"| {signal_type} | {count} | {percentage:.1f}% |\n")
                        
                        f.write("\n")
                    
                    # Signal counts by source
                    if 'signal_counts_by_source' in signal_analysis and signal_analysis['signal_counts_by_source']:
                        f.write("### Signal Distribution by Source\n\n")
                        f.write("| Signal Source | Count | Percentage |\n")
                        f.write("|--------------|-------|------------|\n")
                        
                        for source, count in signal_analysis['signal_counts_by_source'].items():
                            percentage = count / len(self.signals) * 100
                            f.write(f"| {source} | {count} | {percentage:.1f}% |\n")
                        
                        f.write("\n")
                    
                    # Signal strength statistics
                    if 'strength_stats' in signal_analysis and signal_analysis['strength_stats']:
                        f.write("### Signal Strength Statistics\n\n")
                        f.write("| Statistic | Value |\n")
                        f.write("|-----------|-------|\n")
                        
                        for stat, value in signal_analysis['strength_stats'].items():
                            f.write(f"| {stat.capitalize()} | {value:.4f} |\n")
                        
                        f.write("\n")
                    
                    # Signal visualizations
                    viz_dir = os.path.join(self.output_dir, "visualizations")
                    if os.path.exists(viz_dir):
                        signal_viz_files = [
                            "signal_count_by_type.png",
                            "signal_count_by_source.png",
                            "signal_strength_distribution.png",
                            "signal_timeline.png"
                        ]
                        
                        existing_viz = [f for f in signal_viz_files if os.path.exists(os.path.join(viz_dir, f))]
                        
                        if existing_viz:
                            f.write("### Signal Visualizations\n\n")
                            f.write("Please refer to the following visualization files in the 'visualizations' directory:\n\n")
                            
                            for viz_file in existing_viz:
                                f.write(f"- {viz_file}\n")
                            
                            f.write("\n")
                
                # Decision analysis
                f.write("## Trading Decision Analysis\n\n")
                
                if len(self.decisions) == 0:
                    f.write("No trading decisions were made during the test period.\n\n")
                else:
                    # Decision counts by side
                    if 'decision_counts_by_side' in decision_analysis and decision_analysis['decision_counts_by_side']:
                        f.write("### Decision Distribution by Side\n\n")
                        f.write("| Side | Count | Percentage |\n")
                        f.write("|------|-------|------------|\n")
                        
                        for side, count in decision_analysis['decision_counts_by_side'].items():
                            percentage = count / len(self.decisions) * 100
                            f.write(f"| {side} | {count} | {percentage:.1f}% |\n")
                        
                        f.write("\n")
                    
                    # Decision counts by order type
                    if 'decision_counts_by_order_type' in decision_analysis and decision_analysis['decision_counts_by_order_type']:
                        f.write("### Decision Distribution by Order Type\n\n")
                        f.write("| Order Type | Count | Percentage |\n")
                        f.write("|------------|-------|------------|\n")
                        
                        for order_type, count in decision_analysis['decision_counts_by_order_type'].items():
                            percentage = count / len(self.decisions) * 100
                            f.write(f"| {order_type} | {count} | {percentage:.1f}% |\n")
                        
                        f.write("\n")
                    
                    # Size statistics
                    if 'size_stats' in decision_analysis and decision_analysis['size_stats']:
                        f.write("### Order Size Statistics\n\n")
                        f.write("| Statistic | Value |\n")
                        f.write("|-----------|-------|\n")
                        
                        for stat, value in decision_analysis['size_stats'].items():
                            f.write(f"| {stat.capitalize()} | {value:.6f} |\n")
                        
                        f.write("\n")
                    
                    # Decision visualizations
                    viz_dir = os.path.join(self.output_dir, "visualizations")
                    if os.path.exists(viz_dir):
                        decision_viz_files = [
                            "decision_count_by_side.png",
                            "decision_count_by_order_type.png",
                            "decision_size_distribution.png",
                            "decision_timeline.png"
                        ]
                        
                        existing_viz = [f for f in decision_viz_files if os.path.exists(os.path.join(viz_dir, f))]
                        
                        if existing_viz:
                            f.write("### Decision Visualizations\n\n")
                            f.write("Please refer to the following visualization files in the 'visualizations' directory:\n\n")
                            
                            for viz_file in existing_viz:
                                f.write(f"- {viz_file}\n")
                            
                            f.write("\n")
                
                # Recommendations
                f.write("## Recommendations\n\n")
                
                if len(self.signals) == 0:
                    f.write("Based on the lack of signals during the test period, we recommend:\n\n")
                    f.write("1. **Adjust Signal Thresholds**: Consider lowering the thresholds for signal generation to increase sensitivity.\n")
                    f.write("   - Current imbalance thresholds (0.08-0.12) could be reduced to 0.05-0.08\n")
                    f.write("   - Current volatility thresholds (0.03-0.05) could be reduced to 0.02-0.03\n")
                    f.write("   - Current momentum thresholds (0.02-0.03) could be reduced to 0.01-0.02\n\n")
                    f.write("2. **Extended Testing**: Run tests during more volatile market periods or for longer durations.\n\n")
                    f.write("3. **Additional Signal Sources**: Implement additional signal sources such as technical indicators or sentiment analysis.\n\n")
                else:
                    f.write("Based on the analysis of signals and decisions, we recommend:\n\n")
                    
                    # Most common signal source
                    if 'signal_counts_by_source' in signal_analysis and signal_analysis['signal_counts_by_source']:
                        most_common_source = max(signal_analysis['signal_counts_by_source'].items(), key=lambda x: x[1])
                        other_sources = [s for s in signal_analysis['signal_counts_by_source'].keys() if s != most_common_source[0]]
                        
                        f.write(f"1. **Optimize {most_common_source[0]} Signals**: Since this is the primary signal source, ")
                        f.write("fine-tune its parameters for better performance.\n\n")
                        
                        if other_sources:
                            f.write(f"2. **Enhance Other Signal Sources**: Improve the sensitivity of ")
                            f.write(f"{', '.join(other_sources)} signals to achieve a more balanced signal mix.\n\n")
                    
                    # Buy/Sell balance
                    if 'decision_counts_by_side' in decision_analysis and decision_analysis['decision_counts_by_side']:
                        buy_count = decision_analysis['decision_counts_by_side'].get('BUY', 0)
                        sell_count = decision_analysis['decision_counts_by_side'].get('SELL', 0)
                        
                        if buy_count > sell_count * 2:
                            f.write("3. **Balance Buy/Sell Decisions**: The system shows a strong bias toward BUY decisions. ")
                            f.write("Consider adjusting sell signal thresholds to achieve better balance.\n\n")
                        elif sell_count > buy_count * 2:
                            f.write("3. **Balance Buy/Sell Decisions**: The system shows a strong bias toward SELL decisions. ")
                            f.write("Consider adjusting buy signal thresholds to achieve better balance.\n\n")
                    
                    f.write("4. **Implement Performance Metrics**: Add profit/loss tracking to evaluate the effectiveness of trading decisions.\n\n")
                    f.write("5. **Backtesting**: Conduct comprehensive backtesting with historical data to validate signal generation logic.\n\n")
                
                # Conclusion
                f.write("## Conclusion\n\n")
                
                if len(self.signals) == 0:
                    f.write("The Trading-Agent system is technically sound but requires threshold adjustments to generate signals in the current market conditions. ")
                    f.write("The system successfully connects to the exchange API, retrieves market data, and processes it correctly, but the current ")
                    f.write("signal generation thresholds may be too conservative for the observed market volatility.\n\n")
                    f.write("With the recommended threshold adjustments and extended testing, the system should begin generating trading signals ")
                    f.write("and provide valuable insights into its decision-making capabilities.\n")
                else:
                    f.write(f"The Trading-Agent system successfully generated {len(self.signals)} signals and {len(self.decisions)} trading decisions ")
                    f.write("during the test period. The system demonstrates the ability to identify trading opportunities based on ")
                    
                    if 'signal_counts_by_source' in signal_analysis and signal_analysis['signal_counts_by_source']:
                        sources = list(signal_analysis['signal_counts_by_source'].keys())
                        if len(sources) > 1:
                            f.write(f"{', '.join(sources[:-1])} and {sources[-1]}. ")
                        else:
                            f.write(f"{sources[0]}. ")
                    
                    f.write("With the recommended optimizations, the system can be further enhanced to improve signal quality and decision-making accuracy.\n")
            
            logger.info(f"Generated comprehensive report at {report_file}")
            return report_file
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            
            # Create a minimal report in case of error
            error_report_file = os.path.join(self.output_dir, "error_report.md")
            
            with open(error_report_file, 'w') as f:
                f.write("# Trading-Agent System: Signal and Decision Analysis Report\n\n")
                f.write(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("## Error During Report Generation\n\n")
                f.write(f"An error occurred while generating the full report: {str(e)}\n\n")
                f.write("### Basic Statistics\n\n")
                f.write(f"- **Total Signals Generated:** {len(self.signals)}\n")
                f.write(f"- **Total Trading Decisions:** {len(self.decisions)}\n")
            
            return error_report_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extended Signal Analysis for Trading-Agent")
    parser.add_argument("--duration", type=int, default=60, help="Duration to run the system in minutes")
    parser.add_argument("--env", default=".env-secure/.env", help="Path to environment file")
    parser.add_argument("--output", default="./signal_analysis", help="Output directory for analysis results")
    args = parser.parse_args()
    
    # Create analyzer and run system
    analyzer = SignalAnalyzer(output_dir=args.output)
    
    print(f"Running Trading-Agent system for {args.duration} minutes...")
    success = analyzer.run_system(duration_minutes=args.duration, env_path=args.env)
    
    if success:
        print("System run completed successfully.")
        
        # Generate report
        print("Generating comprehensive report...")
        report_file = analyzer.generate_report()
        
        print(f"Analysis complete. Report available at: {report_file}")
    else:
        print("System run failed. Check logs for details.")
