#!/usr/bin/env python
"""
Enhanced LLM Strategic Overseer Integration

This module provides an improved integration of the LLM strategic overseer
with the trading system, including proper logging, error handling, and
decision-making capabilities.
"""

import os
import sys
import json
import time
import logging
import threading
from queue import Queue
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required modules
from enhanced_logging_fixed import EnhancedLogger
from enhanced_signal_processor import EnhancedSignalProcessor, SignalOrderIntegration
from llm_overseer.main import LLMOverseer
from llm_overseer.analysis.pattern_recognition import PatternRecognition

# Initialize enhanced logger
logger = EnhancedLogger("llm_strategic_overseer")

class EnhancedLLMOverseer:
    """Enhanced LLM Strategic Overseer with improved integration"""
    
    def __init__(self, config=None):
        """Initialize enhanced LLM overseer
        
        Args:
            config: Configuration dictionary (optional)
        """
        self.config = config or {}
        self.logger = logger
        
        # Initialize components
        self.llm_overseer = None
        self.pattern_recognition = None
        self.signal_processor = None
        self.integration = None
        
        # Decision queue
        self.decision_queue = Queue()
        
        # Running flag
        self.running = False
        
        # Initialize components
        self.initialize_components()
        
        self.logger.system.info("Enhanced LLM Strategic Overseer initialized")
    
    def initialize_components(self):
        """Initialize all required components"""
        try:
            # Initialize LLM Overseer
            self.logger.system.info("Initializing LLM Overseer")
            self.llm_overseer = LLMOverseer()
            
            # Initialize Pattern Recognition
            self.logger.system.info("Initializing Pattern Recognition")
            self.pattern_recognition = PatternRecognition()
            
            # Initialize Signal Processor
            self.logger.system.info("Initializing Signal Processor")
            self.signal_processor = EnhancedSignalProcessor(self.config)
            
            # Initialize Integration
            self.logger.system.info("Initializing Signal-Order Integration")
            self.integration = SignalOrderIntegration(self.config)
            
            self.logger.system.info("All components initialized successfully")
        except Exception as e:
            self.logger.log_error("Error initializing components", component="initialization")
            raise
    
    def start(self):
        """Start the LLM strategic overseer"""
        self.logger.system.info("Starting LLM Strategic Overseer")
        
        try:
            # Start signal-order integration
            self.integration.start()
            
            # Start decision processing thread
            self.running = True
            self.decision_thread = threading.Thread(target=self.process_decisions)
            self.decision_thread.daemon = True
            self.decision_thread.start()
            
            self.logger.system.info("LLM Strategic Overseer started")
        except Exception as e:
            self.logger.log_error("Error starting LLM Strategic Overseer", component="startup")
            raise
    
    def stop(self):
        """Stop the LLM strategic overseer"""
        self.logger.system.info("Stopping LLM Strategic Overseer")
        
        try:
            # Stop decision processing
            self.running = False
            
            # Wait for thread to terminate
            if hasattr(self, 'decision_thread') and self.decision_thread.is_alive():
                self.decision_thread.join(timeout=5.0)
            
            # Stop signal-order integration
            self.integration.stop()
            
            self.logger.system.info("LLM Strategic Overseer stopped")
        except Exception as e:
            self.logger.log_error("Error stopping LLM Strategic Overseer", component="shutdown")
            raise
    
    def process_decisions(self):
        """Process decisions from the decision queue"""
        self.logger.system.info("Decision processing thread started")
        last_heartbeat = time.time()
        
        while self.running:
            try:
                # Send heartbeat log periodically
                current_time = time.time()
                if current_time - last_heartbeat > 60:
                    self.logger.system.debug("Decision processing thread heartbeat")
                    last_heartbeat = current_time
                
                # Process decisions from queue
                if not self.decision_queue.empty():
                    decision = self.decision_queue.get(timeout=0.1)
                    self.execute_decision(decision)
                else:
                    time.sleep(0.01)
            except Exception as e:
                self.logger.log_error("Error in decision processing thread", component="decision_processing")
                time.sleep(1)  # Prevent tight loop on persistent errors
        
        self.logger.system.info("Decision processing thread stopped")
    
    def execute_decision(self, decision):
        """Execute a trading decision
        
        Args:
            decision: Trading decision dictionary
        """
        try:
            # Log decision execution
            decision_id = decision.get('id', f"DECISION-{int(time.time())}")
            self.logger.set_context(decision_id=decision_id)
            self.logger.system.info(f"Executing decision: {decision_id}")
            
            # Extract signals from decision
            signals = decision.get('signals', [])
            
            # Add signals to integration
            for signal in signals:
                self.logger.log_signal(
                    signal.get('id', f"SIG-{int(time.time())}"),
                    f"Signal from decision {decision_id}: {signal['type']} {signal['symbol']}",
                    source=signal.get('source', 'llm_overseer'),
                    strength=signal.get('strength', 0.0)
                )
                self.integration.add_signal(signal)
            
            self.logger.system.info(f"Decision executed: {len(signals)} signals added to queue")
        except Exception as e:
            self.logger.log_error(f"Error executing decision: {str(e)}", decision_id=decision_id)
        finally:
            # Clear decision context
            self.logger.clear_context('decision_id')
    
    def analyze_market_state(self, market_state):
        """Analyze market state and generate decisions
        
        Args:
            market_state: Market state dictionary
            
        Returns:
            dict: Decision dictionary
        """
        try:
            # Log analysis start
            self.logger.system.info(f"Analyzing market state for {market_state.get('symbol', 'unknown')}")
            
            # Recognize patterns
            patterns = self.recognize_patterns(market_state)
            
            # Generate LLM context
            context = self.generate_llm_context(market_state, patterns)
            
            # Get LLM decision
            decision = self.get_llm_decision(context)
            
            # Add decision to queue
            if decision:
                self.decision_queue.put(decision)
                self.logger.system.info(f"Decision added to queue: {decision.get('id', 'unknown')}")
            
            return decision
        except Exception as e:
            self.logger.log_error(f"Error analyzing market state: {str(e)}")
            return None
    
    def recognize_patterns(self, market_state):
        """Recognize patterns in market state
        
        Args:
            market_state: Market state dictionary
            
        Returns:
            list: Recognized patterns
        """
        try:
            # Extract price history
            price_history = market_state.get('price_history', [])
            
            # Recognize patterns
            patterns = self.pattern_recognition.analyze(price_history)
            
            self.logger.system.info(f"Recognized {len(patterns)} patterns")
            return patterns
        except Exception as e:
            self.logger.log_error(f"Error recognizing patterns: {str(e)}")
            return []
    
    def generate_llm_context(self, market_state, patterns):
        """Generate context for LLM decision
        
        Args:
            market_state: Market state dictionary
            patterns: Recognized patterns
            
        Returns:
            dict: LLM context
        """
        try:
            # Create context
            context = {
                'timestamp': int(time.time() * 1000),
                'symbol': market_state.get('symbol', 'unknown'),
                'current_price': market_state.get('last_trade_price', 0.0),
                'bid_price': market_state.get('bid_price', 0.0),
                'ask_price': market_state.get('ask_price', 0.0),
                'spread_bps': market_state.get('spread_bps', 0.0),
                'order_imbalance': market_state.get('order_imbalance', 0.0),
                'momentum': market_state.get('momentum', 0.0),
                'volatility': market_state.get('volatility', 0.0),
                'trend': market_state.get('trend', 0.0),
                'patterns': patterns,
                'recent_signals': self.get_recent_signals(market_state.get('symbol', 'unknown')),
                'recent_orders': self.get_recent_orders(market_state.get('symbol', 'unknown'))
            }
            
            self.logger.system.info(f"Generated LLM context for {context['symbol']}")
            return context
        except Exception as e:
            self.logger.log_error(f"Error generating LLM context: {str(e)}")
            return {}
    
    def get_recent_signals(self, symbol, limit=10):
        """Get recent signals for a symbol
        
        Args:
            symbol: Trading pair symbol
            limit: Maximum number of signals to return
            
        Returns:
            list: Recent signals
        """
        # This is a placeholder - in a real implementation, this would retrieve
        # recent signals from a database or in-memory store
        return []
    
    def get_recent_orders(self, symbol, limit=10):
        """Get recent orders for a symbol
        
        Args:
            symbol: Trading pair symbol
            limit: Maximum number of orders to return
            
        Returns:
            list: Recent orders
        """
        # This is a placeholder - in a real implementation, this would retrieve
        # recent orders from a database or in-memory store
        return []
    
    def get_llm_decision(self, context):
        """Get LLM decision based on context
        
        Args:
            context: LLM context
            
        Returns:
            dict: Decision dictionary
        """
        try:
            # Generate decision ID
            decision_id = f"DECISION-{int(time.time())}"
            
            # Set context for logging
            self.logger.set_context(decision_id=decision_id)
            
            # Log decision request
            self.logger.system.info(f"Requesting LLM decision for {context.get('symbol', 'unknown')}")
            
            # Get LLM decision
            start_time = time.time()
            decision = self.llm_overseer.get_trading_decision(context)
            end_time = time.time()
            
            # Log performance
            self.logger.log_performance(
                "llm_decision_time",
                (end_time - start_time) * 1000,
                decision_id=decision_id,
                symbol=context.get('symbol', 'unknown')
            )
            
            # Add decision ID
            if decision:
                decision['id'] = decision_id
                self.logger.system.info(f"LLM decision received: {decision.get('action', 'unknown')}")
            else:
                self.logger.system.warning(f"LLM decision returned None")
            
            return decision
        except Exception as e:
            self.logger.log_error(f"Error getting LLM decision: {str(e)}")
            return None
        finally:
            # Clear decision context
            self.logger.clear_context('decision_id')
    
    def process_market_update(self, market_update):
        """Process market update and generate decisions
        
        Args:
            market_update: Market update dictionary
            
        Returns:
            dict: Decision dictionary
        """
        try:
            # Extract market state
            market_state = market_update.get('market_state', {})
            
            # Analyze market state
            decision = self.analyze_market_state(market_state)
            
            return decision
        except Exception as e:
            self.logger.log_error(f"Error processing market update: {str(e)}")
            return None


# Example usage
if __name__ == "__main__":
    # Create configuration
    config = {
        'min_signal_strength': 0.5,
        'max_signal_age_ms': 5000,
        'order_creation_retries': 3,
        'order_creation_retry_delay_ms': 500,
        'risk_per_trade_pct': 1.0,
        'max_position_pct': 5.0,
        'min_position_size': 0.001,
        'position_precision': 3,
        'default_position_size': 0.001,
        'buy_price_factor': 1.001,
        'sell_price_factor': 0.999,
        'max_spread_pct': 1.0,
        'max_price_change_pct': 0.5
    }
    
    # Create enhanced LLM overseer
    overseer = EnhancedLLMOverseer(config)
    
    # Start overseer
    overseer.start()
    
    # Create test market state
    market_state = {
        'symbol': 'BTCUSDC',
        'timestamp': int(time.time() * 1000),
        'bid_price': 105000.0,
        'ask_price': 105100.0,
        'mid_price': 105050.0,
        'spread': 100.0,
        'spread_bps': 9.52,
        'order_imbalance': 0.2,
        'price_history': [105000.0, 104900.0, 104950.0, 105050.0, 105100.0],
        'timestamp_history': [int(time.time() * 1000) - i * 60000 for i in range(5)],
        'volume_history': [10.0, 15.0, 12.0, 8.0, 11.0],
        'last_trade_price': 105050.0,
        'last_trade_size': 0.1,
        'last_trade_side': 'BUY',
        'last_trade_time': int(time.time() * 1000),
        'momentum': 0.15,
        'volatility': 0.02,
        'trend': 0.05
    }
    
    # Create market update
    market_update = {
        'market_state': market_state
    }
    
    # Process market update
    decision = overseer.process_market_update(market_update)
    
    # Run for a while
    try:
        print("Running LLM Strategic Overseer for 30 seconds...")
        time.sleep(30)
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Stop overseer
        overseer.stop()
        print("LLM Strategic Overseer stopped")
