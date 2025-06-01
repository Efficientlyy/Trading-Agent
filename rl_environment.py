#!/usr/bin/env python
"""
Reinforcement Learning Environment for Trading Parameter Optimization

This module implements the RL environment that interfaces with the trading system
to optimize parameters using reinforcement learning techniques.
"""

import os
import sys
import time
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from collections import deque
from typing import Dict, List, Tuple, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rl_environment.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("rl_environment")

class TradingRLEnvironment:
    """Reinforcement Learning Environment for Trading Parameter Optimization"""
    
    def __init__(self, 
                 trading_system=None, 
                 config_path=None, 
                 historical_data_path=None,
                 mode="simulation"):
        """Initialize the RL environment
        
        Args:
            trading_system: Reference to the trading system (if None, will use simulation)
            config_path: Path to configuration file
            historical_data_path: Path to historical data for simulation
            mode: Operation mode ("simulation", "shadow", "assisted", "autonomous")
        """
        self.trading_system = trading_system
        self.config_path = config_path
        self.historical_data_path = historical_data_path
        self.mode = mode
        
        # Load configuration
        self.config = self._load_config()
        
        # State tracking
        self.current_state = {}
        self.previous_state = {}
        self.action_history = deque(maxlen=100)
        self.reward_history = deque(maxlen=100)
        
        # Performance tracking
        self.initial_balance = 10000.0  # Default starting balance
        self.current_balance = self.initial_balance
        self.positions = {}
        self.trades = []
        self.performance_metrics = {
            "pnl": 0.0,
            "sharpe_ratio": 0.0,
            "win_rate": 0.0,
            "trade_count": 0,
            "max_drawdown": 0.0
        }
        
        # Episode tracking
        self.episode_step = 0
        self.episode_reward = 0.0
        self.episode_trades = 0
        self.episode_pnl = 0.0
        
        # Historical data for simulation mode
        self.historical_data = None
        self.current_data_idx = 0
        
        # Initialize environment
        self._initialize_environment()
    
    def _load_config(self) -> Dict:
        """Load configuration from file or use defaults"""
        default_config = {
            "state_space": {
                "market_features": [
                    "price_1m", "price_5m", "price_15m", "price_1h",
                    "rsi_1m", "rsi_5m", "rsi_15m", "rsi_1h",
                    "macd_1m", "macd_5m", "macd_15m", "macd_1h",
                    "bb_1m", "bb_5m", "bb_15m", "bb_1h",
                    "order_imbalance", "spread", "volatility", "session"
                ],
                "agent_state": [
                    "imbalance_threshold", "momentum_threshold", "volatility_threshold",
                    "rsi_threshold", "macd_threshold", "bb_threshold",
                    "win_rate", "profit_factor", "current_drawdown"
                ]
            },
            "action_space": {
                "continuous": {
                    "imbalance_threshold": {"min": 0.05, "max": 0.30, "step": 0.01},
                    "momentum_threshold": {"min": 0.01, "max": 0.10, "step": 0.005},
                    "volatility_threshold": {"min": 0.02, "max": 0.20, "step": 0.005},
                    "rsi_threshold": {"min": 60.0, "max": 80.0, "step": 1.0},
                    "macd_threshold": {"min": 0.0001, "max": 0.001, "step": 0.0001},
                    "bb_threshold": {"min": 0.5, "max": 1.0, "step": 0.05}
                },
                "discrete": {
                    "signal_types": ["order_imbalance", "momentum", "volatility", "rsi", "macd", "bb"],
                    "trading_mode": ["aggressive", "balanced", "conservative"],
                    "timeframe_priority": ["1m", "5m", "15m", "1h"]
                }
            },
            "reward": {
                "weights": {
                    "pnl": 1.0,
                    "sharpe": 0.5,
                    "win_rate": 0.3,
                    "trading_frequency": -0.2,
                    "drawdown": -0.5,
                    "slippage": -0.1
                }
            },
            "simulation": {
                "episode_length": 1440,  # 1 day in minutes
                "warmup_period": 100
            }
        }
        
        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    loaded_config = json.load(f)
                    # Merge with defaults
                    for key, value in loaded_config.items():
                        if key in default_config:
                            if isinstance(value, dict) and isinstance(default_config[key], dict):
                                default_config[key].update(value)
                            else:
                                default_config[key] = value
                        else:
                            default_config[key] = value
                logger.info(f"Loaded configuration from {self.config_path}")
            except Exception as e:
                logger.error(f"Error loading configuration: {str(e)}")
        
        return default_config
    
    def _initialize_environment(self):
        """Initialize the environment based on mode"""
        if self.mode == "simulation":
            self._load_historical_data()
        elif self.mode in ["shadow", "assisted", "autonomous"]:
            if not self.trading_system:
                raise ValueError(f"Trading system required for {self.mode} mode")
            logger.info(f"Initialized in {self.mode} mode with trading system")
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
    
    def _load_historical_data(self):
        """Load historical data for simulation mode"""
        if not self.historical_data_path or not os.path.exists(self.historical_data_path):
            logger.warning("Historical data path not provided or does not exist")
            # Generate synthetic data for testing
            self._generate_synthetic_data()
            return
        
        try:
            # Load data based on file extension
            if self.historical_data_path.endswith('.csv'):
                self.historical_data = pd.read_csv(self.historical_data_path)
            elif self.historical_data_path.endswith('.json'):
                with open(self.historical_data_path, 'r') as f:
                    self.historical_data = pd.DataFrame(json.load(f))
            else:
                logger.error(f"Unsupported file format: {self.historical_data_path}")
                self._generate_synthetic_data()
                return
            
            logger.info(f"Loaded historical data from {self.historical_data_path}")
            logger.info(f"Data shape: {self.historical_data.shape}")
        except Exception as e:
            logger.error(f"Error loading historical data: {str(e)}")
            self._generate_synthetic_data()
    
    def _generate_synthetic_data(self):
        """Generate synthetic data for testing"""
        logger.info("Generating synthetic market data for simulation")
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Generate timestamps (1-minute intervals)
        n_samples = 10000  # Approximately 1 week of 1-minute data
        base_timestamp = int(datetime(2025, 1, 1).timestamp() * 1000)
        timestamps = [base_timestamp + i * 60000 for i in range(n_samples)]
        
        # Generate price data (random walk with drift)
        base_price = 30000.0  # Starting price (e.g., BTC)
        price_changes = np.random.normal(0.0001, 0.001, n_samples)  # Small drift, moderate volatility
        prices = [base_price]
        for change in price_changes:
            prices.append(prices[-1] * (1 + change))
        prices = prices[1:]  # Remove initial base price
        
        # Generate volume data
        volumes = np.random.lognormal(10, 1, n_samples)
        
        # Generate order book data
        bid_ask_spreads = np.random.lognormal(-6, 0.5, n_samples)  # Log-normal for positive spreads
        order_imbalances = np.random.normal(0, 0.3, n_samples)  # Normal distribution centered at 0
        order_imbalances = np.clip(order_imbalances, -0.99, 0.99)  # Clip to valid range
        
        # Create DataFrame
        self.historical_data = pd.DataFrame({
            'timestamp': timestamps,
            'price': prices,
            'volume': volumes,
            'bid_ask_spread': bid_ask_spreads,
            'order_imbalance': order_imbalances
        })
        
        # Add technical indicators
        self._add_synthetic_indicators()
        
        logger.info(f"Generated synthetic data with shape: {self.historical_data.shape}")
    
    def _add_synthetic_indicators(self):
        """Add synthetic technical indicators to historical data"""
        df = self.historical_data
        
        # RSI (roughly correlated with price changes but with mean-reversion)
        df['rsi'] = 50 + np.random.normal(0, 15, len(df))
        df['rsi'] = df['rsi'].rolling(14).mean().fillna(50)
        df['rsi'] = np.clip(df['rsi'], 0, 100)
        
        # MACD (correlated with medium-term trend)
        df['macd'] = np.random.normal(0, 0.001, len(df))
        df['macd'] = df['macd'].rolling(12).mean().fillna(0)
        
        # Bollinger Bands (percent b - position within bands)
        df['bb_percent_b'] = np.random.normal(0.5, 0.3, len(df))
        df['bb_percent_b'] = df['bb_percent_b'].rolling(20).mean().fillna(0.5)
        df['bb_percent_b'] = np.clip(df['bb_percent_b'], 0, 1)
        
        # Volatility (based on rolling standard deviation of returns)
        returns = df['price'].pct_change().fillna(0)
        df['volatility'] = returns.rolling(20).std().fillna(0) * np.sqrt(20)
        
        # Trading session (0: ASIA, 1: EUROPE, 2: US)
        hours = [(timestamp // (60 * 60 * 1000)) % 24 for timestamp in df['timestamp']]
        sessions = []
        for hour in hours:
            if 0 <= hour < 8:
                sessions.append(0)  # ASIA
            elif 8 <= hour < 16:
                sessions.append(1)  # EUROPE
            else:
                sessions.append(2)  # US
        df['session'] = sessions
    
    def reset(self) -> Dict:
        """Reset the environment and return the initial state
        
        Returns:
            dict: Initial state
        """
        # Reset episode tracking
        self.episode_step = 0
        self.episode_reward = 0.0
        self.episode_trades = 0
        self.episode_pnl = 0.0
        
        # Reset performance metrics
        self.current_balance = self.initial_balance
        self.positions = {}
        self.trades = []
        self.performance_metrics = {
            "pnl": 0.0,
            "sharpe_ratio": 0.0,
            "win_rate": 0.0,
            "trade_count": 0,
            "max_drawdown": 0.0
        }
        
        # Reset history
        self.action_history.clear()
        self.reward_history.clear()
        
        # Reset data index for simulation mode
        if self.mode == "simulation":
            warmup = self.config["simulation"]["warmup_period"]
            self.current_data_idx = warmup
        
        # Get initial state
        self.current_state = self._get_state()
        self.previous_state = self.current_state.copy()
        
        return self.current_state
    
    def step(self, action: Dict) -> Tuple[Dict, float, bool, Dict]:
        """Take a step in the environment
        
        Args:
            action: Action to take (parameter adjustments)
            
        Returns:
            tuple: (next_state, reward, done, info)
        """
        # Store previous state
        self.previous_state = self.current_state.copy()
        
        # Apply action
        self._apply_action(action)
        
        # Update environment
        if self.mode == "simulation":
            self._simulation_step()
        else:
            self._live_step()
        
        # Get new state
        self.current_state = self._get_state()
        
        # Calculate reward
        reward = self._calculate_reward()
        self.episode_reward += reward
        self.reward_history.append(reward)
        
        # Check if episode is done
        done = self._is_episode_done()
        
        # Prepare info dict
        info = {
            "episode_step": self.episode_step,
            "episode_reward": self.episode_reward,
            "episode_trades": self.episode_trades,
            "episode_pnl": self.episode_pnl,
            "performance_metrics": self.performance_metrics
        }
        
        # Increment step counter
        self.episode_step += 1
        
        return self.current_state, reward, done, info
    
    def _apply_action(self, action: Dict):
        """Apply action to adjust parameters
        
        Args:
            action: Action dictionary with parameter adjustments
        """
        # Store action for history
        self.action_history.append(action)
        
        # Apply continuous actions (parameter adjustments)
        if "continuous" in action:
            for param, value in action["continuous"].items():
                if param in self.config["action_space"]["continuous"]:
                    # Ensure value is within bounds
                    param_config = self.config["action_space"]["continuous"][param]
                    min_val = param_config["min"]
                    max_val = param_config["max"]
                    value = max(min_val, min(max_val, value))
                    
                    # Update parameter in trading system or simulation
                    if self.mode == "simulation":
                        # Just store for simulation
                        self.current_state[f"param_{param}"] = value
                    else:
                        # Update in actual trading system
                        if self.trading_system and hasattr(self.trading_system, "set_parameter"):
                            self.trading_system.set_parameter(param, value)
        
        # Apply discrete actions
        if "discrete" in action:
            for action_type, choice in action["discrete"].items():
                if action_type in self.config["action_space"]["discrete"]:
                    valid_choices = self.config["action_space"]["discrete"][action_type]
                    if choice in valid_choices:
                        # Update in trading system or simulation
                        if self.mode == "simulation":
                            # Just store for simulation
                            self.current_state[f"discrete_{action_type}"] = choice
                        else:
                            # Update in actual trading system
                            if self.trading_system and hasattr(self.trading_system, "set_discrete_parameter"):
                                self.trading_system.set_discrete_parameter(action_type, choice)
    
    def _simulation_step(self):
        """Take a step in simulation mode"""
        if self.historical_data is None or self.current_data_idx >= len(self.historical_data):
            logger.warning("No more historical data available")
            return
        
        # Get current market data
        current_data = self.historical_data.iloc[self.current_data_idx]
        
        # Simulate trading based on current parameters and market data
        self._simulate_trading(current_data)
        
        # Move to next data point
        self.current_data_idx += 1
    
    def _live_step(self):
        """Take a step in live mode (shadow, assisted, autonomous)"""
        # In live mode, we don't need to simulate trading
        # The trading system will handle actual trading
        # We just need to update our state and performance metrics
        
        if not self.trading_system:
            logger.error("Trading system not available for live step")
            return
        
        # Get performance metrics from trading system
        if hasattr(self.trading_system, "get_performance_metrics"):
            metrics = self.trading_system.get_performance_metrics()
            if metrics:
                self.performance_metrics.update(metrics)
        
        # Get recent trades
        if hasattr(self.trading_system, "get_recent_trades"):
            recent_trades = self.trading_system.get_recent_trades()
            if recent_trades:
                self.trades.extend(recent_trades)
                self.episode_trades += len(recent_trades)
                
                # Calculate episode PnL from recent trades
                for trade in recent_trades:
                    if "pnl" in trade:
                        self.episode_pnl += trade["pnl"]
    
    def _simulate_trading(self, market_data):
        """Simulate trading based on current parameters and market data
        
        Args:
            market_data: Current market data row
        """
        # Extract current parameters
        imbalance_threshold = self.current_state.get("param_imbalance_threshold", 0.2)
        momentum_threshold = self.current_state.get("param_momentum_threshold", 0.005)
        volatility_threshold = self.current_state.get("param_volatility_threshold", 0.01)
        rsi_threshold = self.current_state.get("param_rsi_threshold", 70.0)
        
        # Extract market features
        price = market_data["price"]
        order_imbalance = market_data["order_imbalance"]
        rsi = market_data["rsi"]
        volatility = market_data["volatility"]
        
        # Simple trading logic for simulation
        signal = None
        signal_strength = 0.0
        
        # Order imbalance signal
        if abs(order_imbalance) > imbalance_threshold:
            signal = "BUY" if order_imbalance > 0 else "SELL"
            signal_strength = abs(order_imbalance) / 2  # Scale to 0-0.5 range
        
        # RSI signal
        if rsi > rsi_threshold:
            signal = "SELL"
            signal_strength = max(signal_strength, (rsi - rsi_threshold) / 30)  # Scale to 0-1 range
        elif rsi < (100 - rsi_threshold):
            signal = "BUY"
            signal_strength = max(signal_strength, ((100 - rsi_threshold) - rsi) / 30)  # Scale to 0-1 range
        
        # Volatility breakout signal
        if volatility > volatility_threshold:
            # Use momentum to determine direction
            momentum = (price / self.historical_data.iloc[self.current_data_idx - 5]["price"]) - 1
            if abs(momentum) > momentum_threshold:
                signal = "BUY" if momentum > 0 else "SELL"
                signal_strength = max(signal_strength, abs(momentum) / 0.01)  # Scale to 0-1 range
        
        # Execute trade if signal is strong enough
        if signal and signal_strength > 0.3:  # Minimum signal strength threshold
            self._execute_simulated_trade(signal, price, signal_strength)
    
    def _execute_simulated_trade(self, signal, price, strength):
        """Execute a simulated trade
        
        Args:
            signal: Trade direction ("BUY" or "SELL")
            price: Current price
            strength: Signal strength (0-1)
        """
        # Determine position size based on signal strength and balance
        position_size = 0.1 * strength * self.current_balance / price
        position_value = position_size * price
        
        # Apply simple slippage model
        slippage = 0.001 * (1 + strength)  # Higher strength = higher slippage
        execution_price = price * (1 + slippage) if signal == "BUY" else price * (1 - slippage)
        
        # Execute trade
        trade = {
            "timestamp": self.historical_data.iloc[self.current_data_idx]["timestamp"],
            "signal": signal,
            "price": price,
            "execution_price": execution_price,
            "size": position_size,
            "value": position_value,
            "slippage": slippage
        }
        
        # Update positions
        symbol = "BTC-USD"  # Default symbol for simulation
        if signal == "BUY":
            # Add to position
            if symbol not in self.positions:
                self.positions[symbol] = {"size": 0, "value": 0, "avg_price": 0}
            
            # Update position
            current_size = self.positions[symbol]["size"]
            current_value = self.positions[symbol]["value"]
            
            # Calculate new average price
            if current_size + position_size > 0:
                self.positions[symbol]["avg_price"] = (current_value + position_value) / (current_size + position_size)
            
            # Update size and value
            self.positions[symbol]["size"] += position_size
            self.positions[symbol]["value"] += position_value
            
            # Update balance
            self.current_balance -= position_value
            
        elif signal == "SELL":
            # Reduce position
            if symbol in self.positions and self.positions[symbol]["size"] > 0:
                # Calculate PnL
                avg_price = self.positions[symbol]["avg_price"]
                pnl = position_size * (execution_price - avg_price)
                
                # Update position
                self.positions[symbol]["size"] -= position_size
                self.positions[symbol]["value"] -= position_size * avg_price
                
                # Update balance
                self.current_balance += position_value + pnl
                
                # Add PnL to trade
                trade["pnl"] = pnl
                self.episode_pnl += pnl
            else:
                # Short selling (simplified)
                if symbol not in self.positions:
                    self.positions[symbol] = {"size": 0, "value": 0, "avg_price": 0}
                
                # Update position
                self.positions[symbol]["size"] -= position_size
                self.positions[symbol]["value"] -= position_value
                self.positions[symbol]["avg_price"] = execution_price
                
                # Update balance
                self.current_balance += position_value
        
        # Add trade to history
        self.trades.append(trade)
        self.episode_trades += 1
        
        # Update performance metrics
        self._update_performance_metrics()
    
    def _update_performance_metrics(self):
        """Update performance metrics based on current state"""
        # Calculate total portfolio value
        portfolio_value = self.current_balance
        for symbol, position in self.positions.items():
            # For simplicity, use last known price
            last_price = self.historical_data.iloc[self.current_data_idx]["price"]
            portfolio_value += position["size"] * last_price
        
        # Calculate PnL
        pnl = portfolio_value - self.initial_balance
        pnl_percent = pnl / self.initial_balance
        
        # Calculate win rate
        if len(self.trades) > 0:
            winning_trades = sum(1 for trade in self.trades if trade.get("pnl", 0) > 0)
            win_rate = winning_trades / len(self.trades)
        else:
            win_rate = 0.0
        
        # Calculate max drawdown
        # For simplicity, just track the worst portfolio value
        if self.performance_metrics["max_drawdown"] == 0:
            self.performance_metrics["max_drawdown"] = max(0, -pnl_percent)
        else:
            self.performance_metrics["max_drawdown"] = max(
                self.performance_metrics["max_drawdown"], 
                max(0, -pnl_percent)
            )
        
        # Update metrics
        self.performance_metrics["pnl"] = pnl
        self.performance_metrics["pnl_percent"] = pnl_percent
        self.performance_metrics["win_rate"] = win_rate
        self.performance_metrics["trade_count"] = len(self.trades)
        
        # Simplified Sharpe ratio calculation
        if len(self.trades) > 1:
            returns = [trade.get("pnl", 0) / self.initial_balance for trade in self.trades]
            if np.std(returns) > 0:
                self.performance_metrics["sharpe_ratio"] = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
            else:
                self.performance_metrics["sharpe_ratio"] = 0.0
    
    def _get_state(self) -> Dict:
        """Get current state representation
        
        Returns:
            dict: Current state
        """
        state = {}
        
        if self.mode == "simulation":
            if self.historical_data is not None and self.current_data_idx < len(self.historical_data):
                # Get current and recent market data
                current_data = self.historical_data.iloc[self.current_data_idx]
                
                # Add market features
                for feature in self.config["state_space"]["market_features"]:
                    if feature in current_data:
                        state[feature] = current_data[feature]
                
                # Add derived features
                state["price_change_1m"] = 0.0
                if self.current_data_idx > 0:
                    prev_price = self.historical_data.iloc[self.current_data_idx - 1]["price"]
                    state["price_change_1m"] = (current_data["price"] / prev_price) - 1
                
                state["price_change_5m"] = 0.0
                if self.current_data_idx > 5:
                    prev_price = self.historical_data.iloc[self.current_data_idx - 5]["price"]
                    state["price_change_5m"] = (current_data["price"] / prev_price) - 1
                
                state["price_change_15m"] = 0.0
                if self.current_data_idx > 15:
                    prev_price = self.historical_data.iloc[self.current_data_idx - 15]["price"]
                    state["price_change_15m"] = (current_data["price"] / prev_price) - 1
            
            # Add agent state
            for param in self.config["action_space"]["continuous"]:
                state[f"param_{param}"] = self.current_state.get(f"param_{param}", 
                                                               self.config["action_space"]["continuous"][param]["min"])
            
            # Add performance metrics
            for metric, value in self.performance_metrics.items():
                state[f"metric_{metric}"] = value
            
        else:
            # Live mode - get state from trading system
            if self.trading_system:
                if hasattr(self.trading_system, "get_market_features"):
                    market_features = self.trading_system.get_market_features()
                    if market_features:
                        state.update(market_features)
                
                if hasattr(self.trading_system, "get_current_parameters"):
                    parameters = self.trading_system.get_current_parameters()
                    if parameters:
                        for param, value in parameters.items():
                            state[f"param_{param}"] = value
                
                if hasattr(self.trading_system, "get_performance_metrics"):
                    metrics = self.trading_system.get_performance_metrics()
                    if metrics:
                        for metric, value in metrics.items():
                            state[f"metric_{metric}"] = value
        
        return state
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on performance and actions
        
        Returns:
            float: Reward value
        """
        reward = 0.0
        weights = self.config["reward"]["weights"]
        
        # PnL component (most important)
        pnl_change = self.performance_metrics["pnl"] - self.previous_state.get("metric_pnl", 0)
        normalized_pnl = np.tanh(pnl_change / 100)  # Normalize to [-1, 1] range
        reward += weights["pnl"] * normalized_pnl
        
        # Sharpe ratio component
        sharpe = self.performance_metrics["sharpe_ratio"]
        normalized_sharpe = np.tanh(sharpe / 2)  # Normalize to [-1, 1] range
        reward += weights["sharpe"] * normalized_sharpe
        
        # Win rate component
        win_rate = self.performance_metrics["win_rate"]
        reward += weights["win_rate"] * (win_rate - 0.5) * 2  # Scale to [-1, 1] range
        
        # Trading frequency penalty
        trade_count_change = self.performance_metrics["trade_count"] - self.previous_state.get("metric_trade_count", 0)
        if trade_count_change > 3:  # Penalize excessive trading
            reward += weights["trading_frequency"] * (trade_count_change - 3) / 10
        
        # Drawdown penalty
        max_drawdown = self.performance_metrics["max_drawdown"]
        reward += weights["drawdown"] * max_drawdown  # Already negative weight
        
        # Slippage penalty
        recent_trades = self.trades[-trade_count_change:] if trade_count_change > 0 else []
        avg_slippage = np.mean([trade.get("slippage", 0) for trade in recent_trades]) if recent_trades else 0
        reward += weights["slippage"] * avg_slippage * 100  # Scale up and apply negative weight
        
        return reward
    
    def _is_episode_done(self) -> bool:
        """Check if current episode is done
        
        Returns:
            bool: True if episode is done, False otherwise
        """
        # Check if we've reached the end of the episode
        if self.episode_step >= self.config["simulation"]["episode_length"]:
            return True
        
        # Check if we've run out of historical data
        if self.mode == "simulation" and self.current_data_idx >= len(self.historical_data) - 1:
            return True
        
        # Check for bankruptcy (simplified)
        if self.current_balance <= 0:
            return True
        
        return False
    
    def render(self, mode="human"):
        """Render the current state of the environment
        
        Args:
            mode: Rendering mode
        """
        if mode == "human":
            # Print current state
            print(f"Episode Step: {self.episode_step}")
            print(f"Balance: ${self.current_balance:.2f}")
            print(f"PnL: ${self.performance_metrics['pnl']:.2f} ({self.performance_metrics['pnl_percent']:.2%})")
            print(f"Win Rate: {self.performance_metrics['win_rate']:.2%}")
            print(f"Trades: {self.performance_metrics['trade_count']}")
            print(f"Max Drawdown: {self.performance_metrics['max_drawdown']:.2%}")
            print(f"Sharpe Ratio: {self.performance_metrics['sharpe_ratio']:.2f}")
            print(f"Episode Reward: {self.episode_reward:.4f}")
            
            # Print positions
            print("\nPositions:")
            for symbol, position in self.positions.items():
                print(f"  {symbol}: {position['size']:.4f} @ ${position['avg_price']:.2f}")
            
            # Print recent trades
            print("\nRecent Trades:")
            for trade in self.trades[-5:]:
                print(f"  {trade['signal']} {trade['size']:.4f} @ ${trade['execution_price']:.2f} (PnL: ${trade.get('pnl', 0):.2f})")
            
            print("\n" + "-" * 50 + "\n")
    
    def close(self):
        """Clean up resources"""
        pass


# Example usage
if __name__ == "__main__":
    # Create environment
    env = TradingRLEnvironment(mode="simulation")
    
    # Reset environment
    state = env.reset()
    
    # Run a few steps
    for i in range(100):
        # Random action
        action = {
            "continuous": {
                "imbalance_threshold": np.random.uniform(0.05, 0.3),
                "momentum_threshold": np.random.uniform(0.01, 0.1),
                "volatility_threshold": np.random.uniform(0.02, 0.2),
                "rsi_threshold": np.random.uniform(60, 80)
            },
            "discrete": {
                "trading_mode": np.random.choice(["aggressive", "balanced", "conservative"])
            }
        }
        
        # Take step
        next_state, reward, done, info = env.step(action)
        
        # Render
        env.render()
        
        if done:
            break
    
    # Close environment
    env.close()
