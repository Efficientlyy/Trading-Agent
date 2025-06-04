#!/usr/bin/env python
"""
Fixed LLM Overseer Integration

This module integrates the fixed OpenRouter client with the LLM Overseer
component, ensuring proper API access and error handling.
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
from fixed_openrouter_client import OpenRouterClient
from enhanced_signal_processor import EnhancedSignalProcessor, SignalOrderIntegration

# Initialize enhanced logger
logger = EnhancedLogger("fixed_llm_overseer")

class FixedLLMOverseer:
    """Fixed LLM Strategic Overseer with proper OpenRouter integration"""
    
    def __init__(self, config=None):
        """Initialize fixed LLM overseer
        
        Args:
            config: Configuration dictionary (optional)
        """
        self.config = config or {}
        self.logger = logger
        
        # Initialize components
        self.openrouter_client = None
        self.pattern_recognition = None
        self.signal_processor = None
        self.integration = None
        
        # Decision queue
        self.decision_queue = Queue()
        
        # Running flag
        self.running = False
        
        # Initialize components
        self.initialize_components()
        
        self.logger.system.info("Fixed LLM Strategic Overseer initialized")
    
    def initialize_components(self):
        """Initialize all required components"""
        try:
            # Initialize OpenRouter client
            self.logger.system.info("Initializing OpenRouter client")
            api_key = self.config.get("openrouter_api_key") or os.environ.get("OPENROUTER_API_KEY")
            self.openrouter_client = OpenRouterClient(api_key)
            
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
            
            # Prepare messages for LLM
            messages = [
                {
                    "role": "system",
                    "content": "You are a trading assistant that analyzes market data and provides trading decisions. Your responses should be structured as JSON with 'action', 'reason', 'confidence', and 'signals' fields."
                },
                {
                    "role": "user",
                    "content": f"Please analyze the following market data and provide a trading decision:\n{json.dumps(context, indent=2)}"
                }
            ]
            
            # Get LLM decision
            start_time = time.time()
            response = self.openrouter_client.chat_completion(messages, model="balanced")
            end_time = time.time()
            
            # Log performance
            self.logger.log_performance(
                "llm_decision_time",
                (end_time - start_time) * 1000,
                decision_id=decision_id,
                symbol=context.get('symbol', 'unknown')
            )
            
            # Parse response
            if "error" in response:
                self.logger.system.error(f"Error from LLM: {response['error']}")
                return None
            
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # Try to extract JSON from content
            try:
                # Find JSON in content
                json_start = content.find("{")
                json_end = content.rfind("}") + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = content[json_start:json_end]
                    decision_data = json.loads(json_str)
                else:
                    # No JSON found, create structured decision from text
                    decision_data = self.parse_decision_from_text(content, context)
            except json.JSONDecodeError:
                # JSON parsing failed, create structured decision from text
                decision_data = self.parse_decision_from_text(content, context)
            
            # Add decision ID and timestamp
            decision = {
                "id": decision_id,
                "timestamp": int(time.time() * 1000),
                "action": decision_data.get("action", "HOLD"),
                "reason": decision_data.get("reason", "No specific reason provided"),
                "confidence": decision_data.get("confidence", 0.5),
                "signals": []
            }
            
            # Create signals from decision
            if decision["action"] in ["BUY", "SELL"]:
                signal = {
                    "id": f"SIG-{int(time.time())}",
                    "type": decision["action"],
                    "source": "llm_overseer",
                    "strength": decision["confidence"],
                    "timestamp": int(time.time() * 1000),
                    "price": context.get("current_price", 0.0),
                    "symbol": context.get("symbol", "unknown"),
                    "session": "LLM"
                }
                decision["signals"].append(signal)
            
            self.logger.system.info(f"LLM decision received: {decision['action']} with confidence {decision['confidence']}")
            
            return decision
        except Exception as e:
            self.logger.log_error(f"Error getting LLM decision: {str(e)}")
            return None
        finally:
            # Clear decision context
            self.logger.clear_context('decision_id')
    
    def parse_decision_from_text(self, text, context):
        """Parse decision from text when JSON parsing fails
        
        Args:
            text: Decision text
            context: LLM context
            
        Returns:
            dict: Decision dictionary
        """
        # Default decision
        decision = {
            "action": "HOLD",
            "reason": "Unable to parse clear decision from LLM response",
            "confidence": 0.5
        }
        
        # Try to extract action
        text_lower = text.lower()
        if "buy" in text_lower or "long" in text_lower or "bullish" in text_lower:
            decision["action"] = "BUY"
            decision["confidence"] = 0.6
        elif "sell" in text_lower or "short" in text_lower or "bearish" in text_lower:
            decision["action"] = "SELL"
            decision["confidence"] = 0.6
        
        # Try to extract reason
        if "because" in text_lower:
            parts = text.split("because", 1)
            if len(parts) > 1:
                decision["reason"] = "Because" + parts[1].split(".")[0] + "."
        
        # Try to extract confidence
        confidence_indicators = [
            "confident", "strong", "clear", "definite", "certain",
            "likely", "probable", "possible", "uncertain", "weak"
        ]
        for indicator in confidence_indicators:
            if indicator in text_lower:
                if indicator in ["confident", "strong", "clear", "definite", "certain"]:
                    decision["confidence"] = 0.8
                elif indicator in ["likely", "probable"]:
                    decision["confidence"] = 0.7
                elif indicator in ["possible"]:
                    decision["confidence"] = 0.6
                elif indicator in ["uncertain", "weak"]:
                    decision["confidence"] = 0.4
                break
        
        return decision
    
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


class PatternRecognition:
    """Pattern recognition for market data"""
    
    def __init__(self):
        """Initialize pattern recognition"""
        self.logger = logger
        self.logger.system.info("Pattern recognition initialized")
    
    def analyze(self, price_history):
        """Analyze price history for patterns
        
        Args:
            price_history: List of price points
            
        Returns:
            list: Recognized patterns
        """
        if not price_history or len(price_history) < 5:
            return []
        
        patterns = []
        
        # Simple trend detection
        if price_history[-1] > price_history[0]:
            patterns.append({
                "type": "UPTREND",
                "strength": min(1.0, (price_history[-1] - price_history[0]) / price_history[0] * 10),
                "description": "Price is in an uptrend"
            })
        elif price_history[-1] < price_history[0]:
            patterns.append({
                "type": "DOWNTREND",
                "strength": min(1.0, (price_history[0] - price_history[-1]) / price_history[0] * 10),
                "description": "Price is in a downtrend"
            })
        
        # Simple reversal detection
        if len(price_history) >= 3:
            if price_history[-3] > price_history[-2] and price_history[-1] > price_history[-2]:
                patterns.append({
                    "type": "REVERSAL_BULLISH",
                    "strength": min(1.0, (price_history[-1] - price_history[-2]) / price_history[-2] * 20),
                    "description": "Potential bullish reversal"
                })
            elif price_history[-3] < price_history[-2] and price_history[-1] < price_history[-2]:
                patterns.append({
                    "type": "REVERSAL_BEARISH",
                    "strength": min(1.0, (price_history[-2] - price_history[-1]) / price_history[-2] * 20),
                    "description": "Potential bearish reversal"
                })
        
        return patterns


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
    
    # Create fixed LLM overseer
    overseer = FixedLLMOverseer(config)
    
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
    
    # Print decision
    if decision:
        print(f"Decision: {decision['action']} with confidence {decision['confidence']}")
        print(f"Reason: {decision['reason']}")
        print(f"Signals: {len(decision['signals'])}")
    
    # Run for a while
    try:
        print("Running Fixed LLM Strategic Overseer for 10 seconds...")
        time.sleep(10)
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Stop overseer
        overseer.stop()
        print("Fixed LLM Strategic Overseer stopped")
