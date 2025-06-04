#!/usr/bin/env python
"""
Extended Flash Trading Signals Module

This module extends the FlashTradingSignals class with additional methods
for testing with mock data and direct state-based signal generation.
"""

import os
import sys
import json
import time
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

# Import original signals module
from flash_trading_signals import FlashTradingSignals

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("flash_signals_extension")

class ExtendedFlashTradingSignals(FlashTradingSignals):
    """Extended Flash Trading Signals with mock data support"""
    
    def __init__(self):
        """Initialize extended flash trading signals"""
        super().__init__()
        logger.info("Initialized ExtendedFlashTradingSignals")
    
    def generate_signals_from_state(self, market_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate signals directly from a provided market state
        
        This method allows testing signal generation with mock data by
        bypassing the real-time data collection and using a pre-populated
        market state dictionary.
        
        Args:
            market_state: Dictionary with market state data
            
        Returns:
            List of generated signals
        """
        logger.info(f"Generating signals from provided market state for {market_state.get('symbol', 'unknown')}")
        
        # Validate required fields
        required_fields = [
            "symbol", "timestamp", "bid_price", "ask_price", "mid_price",
            "spread", "spread_bps", "order_imbalance"
        ]
        
        for field in required_fields:
            if field not in market_state:
                logger.error(f"Missing required field in market state: {field}")
                return []
        
        # Extract symbol
        symbol = market_state["symbol"]
        
        # Generate signals
        signals = []
        
        # Check for order book imbalance signal
        if abs(market_state["order_imbalance"]) > self.imbalance_threshold:
            signal_type = "BUY" if market_state["order_imbalance"] > 0 else "SELL"
            signal_strength = abs(market_state["order_imbalance"]) * 1.25  # Scale for better visibility
            
            signals.append({
                "type": signal_type,
                "source": "order_imbalance",
                "strength": signal_strength,
                "timestamp": market_state["timestamp"],
                "price": market_state["bid_price"] if signal_type == "SELL" else market_state["ask_price"],
                "symbol": symbol,
                "session": self.session_manager.get_current_session_name()
            })
            
            logger.info(f"Generated {signal_type} signal from order imbalance with strength {signal_strength:.4f}")
        
        # Check for volatility signal if price history is available
        if "price_history" in market_state and len(market_state["price_history"]) > 1:
            # Calculate volatility if not provided
            if "volatility" not in market_state or market_state["volatility"] is None:
                prices = np.array(market_state["price_history"])
                pct_changes = np.diff(prices) / prices[:-1]
                volatility = np.std(pct_changes) * 100
            else:
                volatility = market_state["volatility"]
            
            if volatility > self.volatility_threshold:
                # Determine direction based on recent price movement
                recent_prices = market_state["price_history"][-5:]
                signal_type = "BUY" if recent_prices[-1] > recent_prices[0] else "SELL"
                signal_strength = min(volatility / self.volatility_threshold, 1.0)
                
                signals.append({
                    "type": signal_type,
                    "source": "volatility",
                    "strength": signal_strength,
                    "timestamp": market_state["timestamp"],
                    "price": market_state["mid_price"],
                    "symbol": symbol,
                    "session": self.session_manager.get_current_session_name()
                })
                
                logger.info(f"Generated {signal_type} signal from volatility with strength {signal_strength:.4f}")
        
        # Check for momentum signal if provided
        if "momentum" in market_state and market_state["momentum"] is not None:
            momentum = market_state["momentum"]
            
            if abs(momentum) > self.momentum_threshold:
                signal_type = "BUY" if momentum > 0 else "SELL"
                signal_strength = min(abs(momentum) / self.momentum_threshold, 1.0)
                
                signals.append({
                    "type": signal_type,
                    "source": "momentum",
                    "strength": signal_strength,
                    "timestamp": market_state["timestamp"],
                    "price": market_state["mid_price"],
                    "symbol": symbol,
                    "session": self.session_manager.get_current_session_name()
                })
                
                logger.info(f"Generated {signal_type} signal from momentum with strength {signal_strength:.4f}")
        
        return signals

# Example usage
if __name__ == "__main__":
    signals = ExtendedFlashTradingSignals()
    
    # Create mock market state
    market_state = {
        "symbol": "BTCUSDC",
        "timestamp": int(time.time() * 1000),
        "bid_price": 105000.0,
        "ask_price": 105100.0,
        "mid_price": 105050.0,
        "spread": 100.0,
        "spread_bps": 9.52,
        "order_imbalance": 0.75,
        "price_history": [105000.0, 105010.0, 105020.0, 105030.0, 105050.0],
        "timestamp_history": [int(time.time() * 1000) - i * 60000 for i in range(5, 0, -1)],
        "volume_history": [1.5, 2.1, 1.8, 2.5, 3.0],
        "momentum": 0.08,
        "volatility": 0.12
    }
    
    # Generate signals
    generated_signals = signals.generate_signals_from_state(market_state)
    
    # Print results
    print(f"Generated {len(generated_signals)} signals:")
    for signal in generated_signals:
        print(f"  {signal['type']} signal from {signal['source']} with strength {signal['strength']:.4f}")
