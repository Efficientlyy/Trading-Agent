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

# Configure logging with a more robust setup
def setup_logger(name, log_file, level=logging.INFO):
    """Set up a logger with file and console handlers"""
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers if any
    if logger.handlers:
        logger.handlers = []
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Set up logger
logger = setup_logger("rl_environment", "rl_environment.log")

class TradingRLEnvironment:
    """Reinforcement Learning Environment for Trading Parameter Optimization"""
    
    def __init__(self, 
                 trading_system=None, 
                 config_path=None, 
                 historical_data_path=None,
                 mode="simulation",
                 random_seed=None,
                 initial_balance=10000.0,
                 max_episode_steps=None):
        """Initialize the RL environment
        
        Args:
            trading_system: Reference to the trading system (if None, will use simulation)
            config_path: Path to configuration file
            historical_data_path: Path to historical data for simulation
            mode: Operation mode ("simulation", "shadow", "assisted", "autonomous")
            random_seed: Seed for random number generators (for reproducibility)
            initial_balance: Initial balance for simulation
            max_episode_steps: Maximum steps per episode (if None, use config value)
        """
        self.trading_system = trading_system
        self.config_path = config_path
        self.historical_data_path = historical_data_path
        self.mode = mode
        self.random_seed = random_seed
        self.initial_balance = initial_balance
        
        # Set random seed for reproducibility if provided
        if random_seed is not None:
            np.random.seed(random_seed)
            logger.info(f"Random seed set to {random_seed}")
        
        # Load configuration
        self.config = self._load_config()
        
        # Set max episode steps
        self.max_episode_steps = max_episode_steps or self.config["simulation"]["episode_length"]
        
        # State tracking
        self.current_state = {}
        self.previous_state = {}
        self.action_history = deque(maxlen=100)
        self.reward_history = deque(maxlen=100)
        
        # Performance tracking
        self.current_balance = self.initial_balance
        self.positions = {}
        self.trades = []
        self.performance_metrics = {
            "pnl": 0.0,
            "sharpe_ratio": 0.0,
            "win_rate": 0.0,
            "trade_count": 0,
            "max_drawdown": 0.0,
            "volatility": 0.0,
            "profit_factor": 0.0,
            "recovery_factor": 0.0
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
        try:
            self._initialize_environment()
            logger.info(f"Environment initialized successfully in {mode} mode")
        except Exception as e:
            logger.error(f"Failed to initialize environment: {str(e)}")
            raise
    
    def _load_config(self) -> Dict:
        """Load configuration from file or use defaults
        
        Returns:
            dict: Configuration dictionary
        
        Raises:
            ValueError: If configuration is invalid
        """
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
                "warmup_period": 100,
                "data_chunk_size": 5000,  # Number of data points to load at once
                "max_position_size": 1.0,  # Maximum position size as fraction of balance
                "transaction_cost": 0.001  # 10 basis points per trade
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
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in configuration file: {str(e)}")
                raise ValueError(f"Invalid JSON in configuration file: {str(e)}")
            except Exception as e:
                logger.error(f"Error loading configuration: {str(e)}")
                raise
        
        # Validate configuration
        self._validate_config(default_config)
        
        return default_config
    
    def _validate_config(self, config):
        """Validate configuration parameters
        
        Args:
            config: Configuration dictionary to validate
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Check required sections
        required_sections = ["state_space", "action_space", "reward", "simulation"]
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate action space
        if "continuous" not in config["action_space"]:
            raise ValueError("Missing continuous action space configuration")
        
        # Validate reward weights
        reward_weights = config["reward"].get("weights", {})
        if not reward_weights:
            raise ValueError("Missing reward weights configuration")
        
        # Validate simulation parameters
        sim_config = config["simulation"]
        if "episode_length" not in sim_config or not isinstance(sim_config["episode_length"], int):
            raise ValueError("Invalid episode_length in simulation configuration")
        if "warmup_period" not in sim_config or not isinstance(sim_config["warmup_period"], int):
            raise ValueError("Invalid warmup_period in simulation configuration")
        
        logger.info("Configuration validated successfully")
    
    def _initialize_environment(self):
        """Initialize the environment based on mode
        
        Raises:
            ValueError: If mode is invalid or trading system is missing
        """
        if self.mode == "simulation":
            self._load_historical_data()
        elif self.mode in ["shadow", "assisted", "autonomous"]:
            if not self.trading_system:
                raise ValueError(f"Trading system required for {self.mode} mode")
            logger.info(f"Initialized in {self.mode} mode with trading system")
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
    
    def _load_historical_data(self):
        """Load historical data for simulation mode
        
        Raises:
            ValueError: If data format is invalid
            IOError: If data file cannot be read
        """
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
            elif self.historical_data_path.endswith('.parquet'):
                self.historical_data = pd.read_parquet(self.historical_data_path)
            else:
                logger.error(f"Unsupported file format: {self.historical_data_path}")
                raise ValueError(f"Unsupported file format: {self.historical_data_path}")
            
            # Validate data
            required_columns = ['timestamp', 'price']
            missing_columns = [col for col in required_columns if col not in self.historical_data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns in historical data: {missing_columns}")
            
            # Sort by timestamp if not already sorted
            if not self.historical_data['timestamp'].is_monotonic_increasing:
                logger.warning("Historical data not sorted by timestamp, sorting now")
                self.historical_data = self.historical_data.sort_values('timestamp').reset_index(drop=True)
            
            logger.info(f"Loaded historical data from {self.historical_data_path}")
            logger.info(f"Data shape: {self.historical_data.shape}")
            
            # Add missing indicators if needed
            self._ensure_indicators()
            
        except pd.errors.EmptyDataError:
            logger.error("Historical data file is empty")
            raise ValueError("Historical data file is empty")
        except pd.errors.ParserError as e:
            logger.error(f"Error parsing historical data: {str(e)}")
            raise ValueError(f"Error parsing historical data: {str(e)}")
        except Exception as e:
            logger.error(f"Error loading historical data: {str(e)}")
            self._generate_synthetic_data()
    
    def _ensure_indicators(self):
        """Ensure all required indicators are present in the data"""
        df = self.historical_data
        
        # Check for required indicators and calculate if missing
        if 'rsi' not in df.columns:
            logger.info("Calculating RSI indicator")
            self._calculate_rsi(df)
        
        if 'macd' not in df.columns:
            logger.info("Calculating MACD indicator")
            self._calculate_macd(df)
        
        if 'bb_percent_b' not in df.columns:
            logger.info("Calculating Bollinger Bands indicator")
            self._calculate_bollinger_bands(df)
        
        if 'volatility' not in df.columns:
            logger.info("Calculating volatility")
            self._calculate_volatility(df)
        
        if 'session' not in df.columns:
            logger.info("Adding trading session information")
            self._add_session_info(df)
    
    def _calculate_rsi(self, df, period=14):
        """Calculate RSI indicator
        
        Args:
            df: DataFrame with price data
            period: RSI period
        """
        delta = df['price'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].fillna(50)
    
    def _calculate_macd(self, df, fast=12, slow=26, signal=9):
        """Calculate MACD indicator
        
        Args:
            df: DataFrame with price data
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal EMA period
        """
        ema_fast = df['price'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['price'].ewm(span=slow, adjust=False).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
    
    def _calculate_bollinger_bands(self, df, period=20, std_dev=2):
        """Calculate Bollinger Bands indicator
        
        Args:
            df: DataFrame with price data
            period: Bollinger Bands period
            std_dev: Number of standard deviations
        """
        df['bb_middle'] = df['price'].rolling(window=period).mean()
        df['bb_std'] = df['price'].rolling(window=period).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * std_dev)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * std_dev)
        
        # Calculate %B
        df['bb_percent_b'] = (df['price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_percent_b'] = df['bb_percent_b'].clip(0, 1)
    
    def _calculate_volatility(self, df, period=20):
        """Calculate volatility
        
        Args:
            df: DataFrame with price data
            period: Volatility period
        """
        returns = df['price'].pct_change().fillna(0)
        df['volatility'] = returns.rolling(period).std().fillna(0) * np.sqrt(period)
    
    def _add_session_info(self, df):
        """Add trading session information
        
        Args:
            df: DataFrame with timestamp data
        """
        # Convert timestamp to hour (assuming millisecond timestamps)
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
    
    def _generate_synthetic_data(self):
        """Generate synthetic data for testing
        
        This method creates realistic synthetic market data for simulation
        when no historical data is available.
        """
        logger.info("Generating synthetic market data for simulation")
        
        # Set random seed for reproducibility
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        
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
        df['macd_signal'] = df['macd'].rolling(9).mean().fillna(0)
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
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
            
        Raises:
            RuntimeError: If environment cannot be reset
        """
        try:
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
                "max_drawdown": 0.0,
                "volatility": 0.0,
                "profit_factor": 0.0,
                "recovery_factor": 0.0
            }
            
            # Reset history
            self.action_history.clear()
            self.reward_history.clear()
            
            # Reset data index for simulation mode
            if self.mode == "simulation":
                if self.historical_data is None or len(self.historical_data) == 0:
                    raise RuntimeError("No historical data available for simulation")
                
                warmup = self.config["simulation"]["warmup_period"]
                if warmup >= len(self.historical_data):
                    logger.warning(f"Warmup period ({warmup}) exceeds data length ({len(self.historical_data)})")
                    warmup = max(0, len(self.historical_data) - 1)
                
                self.current_data_idx = warmup
            
            # Get initial state
            self.current_state = self._get_state()
            self.previous_state = self.current_state.copy()
            
            logger.info(f"Environment reset, initial balance: {self.initial_balance}")
            return self.current_state
            
        except Exception as e:
            logger.error(f"Error resetting environment: {str(e)}")
            raise RuntimeError(f"Failed to reset environment: {str(e)}")
    
    def step(self, action: Dict) -> Tuple[Dict, float, bool, Dict]:
        """Take a step in the environment
        
        Args:
            action: Action to take (parameter adjustments)
            
        Returns:
            tuple: (next_state, reward, done, info)
            
        Raises:
            RuntimeError: If step fails
            ValueError: If action is invalid
        """
        try:
            # Validate action
            if not isinstance(action, dict):
                raise ValueError(f"Action must be a dictionary, got {type(action)}")
            
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
            
        except Exception as e:
            logger.error(f"Error in environment step: {str(e)}")
            raise RuntimeError(f"Failed to execute step: {str(e)}")
    
    def _apply_action(self, action: Dict):
        """Apply action to adjust parameters
        
        Args:
            action: Action dictionary with parameter adjustments
            
        Raises:
            ValueError: If action is invalid
        """
        # Validate action structure
        if "continuous" not in action and "discrete" not in action:
            raise ValueError("Action must contain 'continuous' or 'discrete' key")
        
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
                    
                    # Check for NaN or infinity
                    if not np.isfinite(value):
                        logger.warning(f"Non-finite value for parameter {param}: {value}, using default")
                        value = (min_val + max_val) / 2
                    
                    value = max(min_val, min(max_val, value))
                    
                    # Update parameter in trading system or simulation
                    if self.mode == "simulation":
                        # Just store for simulation
                        self.current_state[f"param_{param}"] = value
                    else:
                        # Update in actual trading system
                        if self.trading_system and hasattr(self.trading_system, "set_parameter"):
                            self.trading_system.set_parameter(param, value)
                else:
                    logger.warning(f"Unknown continuous parameter: {param}")
        
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
                    else:
                        logger.warning(f"Invalid choice for {action_type}: {choice}")
                else:
                    logger.warning(f"Unknown discrete action type: {action_type}")
    
    def _simulation_step(self):
        """Take a step in simulation mode
        
        Raises:
            RuntimeError: If simulation step fails
        """
        if self.historical_data is None:
            raise RuntimeError("No historical data available for simulation")
        
        if self.current_data_idx >= len(self.historical_data):
            logger.warning("No more historical data available")
            return
        
        try:
            # Get current market data
            current_data = self.historical_data.iloc[self.current_data_idx]
            
            # Simulate trading based on current parameters and market data
            self._simulate_trading(current_data)
            
            # Move to next data point
            self.current_data_idx += 1
            
        except Exception as e:
            logger.error(f"Error in simulation step: {str(e)}")
            raise RuntimeError(f"Simulation step failed: {str(e)}")
    
    def _live_step(self):
        """Take a step in live mode (shadow, assisted, autonomous)
        
        Raises:
            RuntimeError: If live step fails
        """
        # In live mode, we don't need to simulate trading
        # The trading system will handle actual trading
        # We just need to update our state and performance metrics
        
        if not self.trading_system:
            raise RuntimeError("Trading system not available for live step")
        
        try:
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
                            
        except Exception as e:
            logger.error(f"Error in live step: {str(e)}")
            raise RuntimeError(f"Live step failed: {str(e)}")
    
    def _simulate_trading(self, market_data):
        """Simulate trading based on current parameters and market data
        
        Args:
            market_data: Current market data row
            
        Raises:
            ValueError: If market data is invalid
        """
        # Validate market data
        required_fields = ["price", "timestamp"]
        for field in required_fields:
            if field not in market_data:
                raise ValueError(f"Missing required field in market data: {field}")
        
        try:
            # Extract current parameters
            imbalance_threshold = self.current_state.get("param_imbalance_threshold", 0.2)
            momentum_threshold = self.current_state.get("param_momentum_threshold", 0.005)
            volatility_threshold = self.current_state.get("param_volatility_threshold", 0.01)
            rsi_threshold = self.current_state.get("param_rsi_threshold", 70.0)
            macd_threshold = self.current_state.get("param_macd_threshold", 0.0005)
            bb_threshold = self.current_state.get("param_bb_threshold", 0.8)
            
            # Extract market features
            price = market_data["price"]
            
            # Default values for optional features
            order_imbalance = market_data.get("order_imbalance", 0.0)
            rsi = market_data.get("rsi", 50.0)
            macd = market_data.get("macd", 0.0)
            bb_percent_b = market_data.get("bb_percent_b", 0.5)
            volatility = market_data.get("volatility", 0.01)
            
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
            
            # MACD signal
            if abs(macd) > macd_threshold:
                signal = "BUY" if macd > 0 else "SELL"
                signal_strength = max(signal_strength, abs(macd) / (macd_threshold * 5))  # Scale to 0-1 range
            
            # Bollinger Bands signal
            if bb_percent_b > bb_threshold:
                signal = "SELL"
                signal_strength = max(signal_strength, (bb_percent_b - bb_threshold) / (1 - bb_threshold))
            elif bb_percent_b < (1 - bb_threshold):
                signal = "BUY"
                signal_strength = max(signal_strength, ((1 - bb_threshold) - bb_percent_b) / (1 - bb_threshold))
            
            # Volatility breakout signal
            if volatility > volatility_threshold:
                # Use momentum to determine direction
                momentum = 0.0
                if self.current_data_idx >= 5:
                    prev_price = self.historical_data.iloc[self.current_data_idx - 5]["price"]
                    momentum = (price / prev_price) - 1
                
                if abs(momentum) > momentum_threshold:
                    signal = "BUY" if momentum > 0 else "SELL"
                    signal_strength = max(signal_strength, abs(momentum) / 0.01)  # Scale to 0-1 range
            
            # Execute trade if signal is strong enough
            if signal and signal_strength > 0.3:  # Minimum signal strength threshold
                self._execute_simulated_trade(signal, price, signal_strength)
                
        except Exception as e:
            logger.error(f"Error in trading simulation: {str(e)}")
            raise RuntimeError(f"Trading simulation failed: {str(e)}")
    
    def _execute_simulated_trade(self, signal, price, strength):
        """Execute a simulated trade
        
        Args:
            signal: Trade direction ("BUY" or "SELL")
            price: Current price
            strength: Signal strength (0-1)
            
        Raises:
            ValueError: If signal is invalid
        """
        if signal not in ["BUY", "SELL"]:
            raise ValueError(f"Invalid signal: {signal}")
        
        try:
            # Get max position size from config
            max_position_size = self.config["simulation"].get("max_position_size", 1.0)
            
            # Determine position size based on signal strength and balance
            position_size = max_position_size * strength * self.current_balance / price
            position_value = position_size * price
            
            # Apply transaction cost
            transaction_cost = self.config["simulation"].get("transaction_cost", 0.001)
            
            # Apply simple slippage model
            slippage = 0.001 * (1 + strength)  # Higher strength = higher slippage
            execution_price = price * (1 + slippage) if signal == "BUY" else price * (1 - slippage)
            
            # Calculate transaction fee
            fee = position_value * transaction_cost
            
            # Execute trade
            trade = {
                "timestamp": self.historical_data.iloc[self.current_data_idx]["timestamp"],
                "signal": signal,
                "price": price,
                "execution_price": execution_price,
                "size": position_size,
                "value": position_value,
                "slippage": slippage,
                "fee": fee
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
                
                # Update balance (subtract value and fee)
                self.current_balance -= (position_value + fee)
                
            elif signal == "SELL":
                # Reduce position
                if symbol in self.positions and self.positions[symbol]["size"] > 0:
                    # Calculate PnL
                    avg_price = self.positions[symbol]["avg_price"]
                    pnl = position_size * (execution_price - avg_price) - fee
                    
                    # Update position
                    self.positions[symbol]["size"] -= position_size
                    if self.positions[symbol]["size"] > 0:
                        self.positions[symbol]["value"] -= position_size * avg_price
                    else:
                        # Position closed
                        self.positions[symbol]["value"] = 0
                        self.positions[symbol]["avg_price"] = 0
                    
                    # Update balance
                    self.current_balance += position_value + pnl - fee
                    
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
                    self.current_balance += position_value - fee
            
            # Add trade to history
            self.trades.append(trade)
            self.episode_trades += 1
            
            # Update performance metrics
            self._update_performance_metrics()
            
            logger.debug(f"Executed {signal} trade: {position_size:.4f} @ ${execution_price:.2f}")
            
        except Exception as e:
            logger.error(f"Error executing simulated trade: {str(e)}")
            raise RuntimeError(f"Trade execution failed: {str(e)}")
    
    def _update_performance_metrics(self):
        """Update performance metrics based on current state
        
        This method calculates various performance metrics including PnL,
        Sharpe ratio, win rate, drawdown, etc.
        """
        try:
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
            
            # Calculate profit factor
            if len(self.trades) > 0:
                gross_profit = sum(trade.get("pnl", 0) for trade in self.trades if trade.get("pnl", 0) > 0)
                gross_loss = sum(abs(trade.get("pnl", 0)) for trade in self.trades if trade.get("pnl", 0) < 0)
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else 1.0
            else:
                profit_factor = 0.0
            
            # Calculate max drawdown
            # For simplicity, just track the worst portfolio value
            if self.performance_metrics["max_drawdown"] == 0:
                self.performance_metrics["max_drawdown"] = max(0, -pnl_percent)
            else:
                self.performance_metrics["max_drawdown"] = max(
                    self.performance_metrics["max_drawdown"], 
                    max(0, -pnl_percent)
                )
            
            # Calculate recovery factor
            max_drawdown = self.performance_metrics["max_drawdown"]
            recovery_factor = pnl_percent / max_drawdown if max_drawdown > 0 else 0.0
            
            # Update metrics
            self.performance_metrics["pnl"] = pnl
            self.performance_metrics["pnl_percent"] = pnl_percent
            self.performance_metrics["win_rate"] = win_rate
            self.performance_metrics["trade_count"] = len(self.trades)
            self.performance_metrics["profit_factor"] = profit_factor
            self.performance_metrics["recovery_factor"] = recovery_factor
            
            # Simplified Sharpe ratio calculation
            if len(self.trades) > 1:
                returns = [trade.get("pnl", 0) / self.initial_balance for trade in self.trades]
                if np.std(returns) > 0:
                    self.performance_metrics["sharpe_ratio"] = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
                    self.performance_metrics["volatility"] = np.std(returns) * np.sqrt(252)  # Annualized
                else:
                    self.performance_metrics["sharpe_ratio"] = 0.0
                    self.performance_metrics["volatility"] = 0.0
                    
        except Exception as e:
            logger.error(f"Error updating performance metrics: {str(e)}")
            # Continue without updating metrics
    
    def _get_state(self) -> Dict:
        """Get current state representation
        
        Returns:
            dict: Current state
            
        Raises:
            RuntimeError: If state cannot be retrieved
        """
        try:
            state = {}
            
            if self.mode == "simulation":
                if self.historical_data is not None and self.current_data_idx < len(self.historical_data):
                    # Get current and recent market data
                    current_data = self.historical_data.iloc[self.current_data_idx]
                    
                    # Add market features
                    for feature in self.config["state_space"]["market_features"]:
                        if feature in current_data:
                            state[feature] = current_data[feature]
                        else:
                            # Use default value for missing features
                            state[feature] = 0.0
                    
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
                    key = f"param_{param}"
                    if key in self.current_state:
                        state[key] = self.current_state[key]
                    else:
                        # Use default value (middle of range)
                        param_config = self.config["action_space"]["continuous"][param]
                        state[key] = (param_config["min"] + param_config["max"]) / 2
                
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
            
        except Exception as e:
            logger.error(f"Error getting state: {str(e)}")
            raise RuntimeError(f"Failed to get state: {str(e)}")
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on performance and actions
        
        Returns:
            float: Reward value
        """
        try:
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
            
            # Ensure reward is finite
            if not np.isfinite(reward):
                logger.warning(f"Non-finite reward calculated: {reward}, using zero")
                reward = 0.0
            
            return reward
            
        except Exception as e:
            logger.error(f"Error calculating reward: {str(e)}")
            return 0.0  # Default to zero reward on error
    
    def _is_episode_done(self) -> bool:
        """Check if current episode is done
        
        Returns:
            bool: True if episode is done, False otherwise
        """
        # Check if we've reached the end of the episode
        if self.episode_step >= self.max_episode_steps:
            logger.info(f"Episode done: reached max steps ({self.max_episode_steps})")
            return True
        
        # Check if we've run out of historical data
        if self.mode == "simulation" and self.current_data_idx >= len(self.historical_data) - 1:
            logger.info("Episode done: end of historical data")
            return True
        
        # Check for bankruptcy (simplified)
        if self.current_balance <= 0:
            logger.info("Episode done: bankruptcy (balance <= 0)")
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
        logger.info("Closing environment and releasing resources")
        # Close any open resources
        pass
    
    def seed(self, seed=None):
        """Set random seed
        
        Args:
            seed: Random seed
            
        Returns:
            list: List containing the seed
        """
        self.random_seed = seed
        if seed is not None:
            np.random.seed(seed)
            logger.info(f"Random seed set to {seed}")
        return [seed]
    
    def get_performance_summary(self) -> Dict:
        """Get a summary of performance metrics
        
        Returns:
            dict: Performance summary
        """
        return {
            "pnl": self.performance_metrics["pnl"],
            "pnl_percent": self.performance_metrics["pnl_percent"],
            "win_rate": self.performance_metrics["win_rate"],
            "trade_count": self.performance_metrics["trade_count"],
            "sharpe_ratio": self.performance_metrics["sharpe_ratio"],
            "max_drawdown": self.performance_metrics["max_drawdown"],
            "profit_factor": self.performance_metrics["profit_factor"],
            "recovery_factor": self.performance_metrics["recovery_factor"],
            "volatility": self.performance_metrics["volatility"],
            "episode_reward": self.episode_reward,
            "episode_trades": self.episode_trades
        }


# Example usage
if __name__ == "__main__":
    # Create environment
    env = TradingRLEnvironment(mode="simulation", random_seed=42)
    
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
    
    # Print performance summary
    print("Performance Summary:")
    for metric, value in env.get_performance_summary().items():
        print(f"  {metric}: {value}")
    
    # Close environment
    env.close()
