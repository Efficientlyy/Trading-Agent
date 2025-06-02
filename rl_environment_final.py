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
        # Convert timestamp to hour of day
        if isinstance(df['timestamp'].iloc[0], str):
            # Parse string timestamp
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        else:
            # Convert millisecond timestamp to hour
            df['hour'] = pd.to_datetime(df['timestamp'], unit='ms').dt.hour
        
        # Assign session (0: ASIA, 1: EUROPE, 2: US)
        df['session'] = df['hour'].apply(lambda h: 0 if 0 <= h < 8 else (1 if 8 <= h < 16 else 2))
        
        # Drop temporary hour column
        df.drop('hour', axis=1, inplace=True)
    
    def _generate_synthetic_data(self):
        """Generate synthetic data for testing"""
        logger.info("Generating synthetic data for testing")
        
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
        df = pd.DataFrame({
            'timestamp': timestamps,
            'price': prices,
            'volume': volumes,
            'bid_ask_spread': bid_ask_spreads,
            'order_imbalance': order_imbalances
        })
        
        # Add indicators
        self._calculate_rsi(df)
        self._calculate_macd(df)
        self._calculate_bollinger_bands(df)
        self._calculate_volatility(df)
        self._add_session_info(df)
        
        self.historical_data = df
        logger.info(f"Generated synthetic data with {len(df)} samples")
    
    def reset(self):
        """Reset the environment to initial state
        
        Returns:
            dict: Initial state
        """
        # Reset episode tracking
        self.episode_step = 0
        self.episode_reward = 0.0
        self.episode_trades = 0
        self.episode_pnl = 0.0
        
        # Reset performance tracking
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
        
        # Reset state tracking
        self.current_state = {}
        self.previous_state = {}
        self.action_history.clear()
        self.reward_history.clear()
        
        # Reset data index for simulation mode
        if self.mode == "simulation":
            self.current_data_idx = self.config["simulation"]["warmup_period"]
            
            # Initialize default parameters
            for param, config in self.config["action_space"]["continuous"].items():
                default_value = (config["min"] + config["max"]) / 2
                self.current_state[f"param_{param}"] = default_value
            
            # Update market state from historical data
            self._update_market_state()
        
        # Get initial state
        self.current_state = self._get_state()
        
        logger.info(f"Environment reset, initial balance: {self.current_balance}")
        
        return self.current_state
    
    def step(self, action):
        """Take a step in the environment
        
        Args:
            action: Action to take (dict with continuous and/or discrete parameters)
            
        Returns:
            tuple: (next_state, reward, done, info)
        """
        try:
            # Validate action
            self._validate_action(action)
            
            # Store previous state
            self.previous_state = self.current_state.copy()
            
            # Apply action
            self._apply_action(action)
            
            # Update environment
            self._update_environment()
            
            # Get new state
            self.current_state = self._get_state()
            
            # Calculate reward
            reward = self._calculate_reward()
            
            # Check if episode is done
            done = self._is_done()
            
            # Get info
            info = self._get_info()
            
            # Update tracking
            self.episode_step += 1
            self.episode_reward += reward
            self.action_history.append(action)
            self.reward_history.append(reward)
            
            return self.current_state, reward, done, info
            
        except Exception as e:
            logger.error(f"Error in environment step: {str(e)}")
            raise ValueError(f"Failed to execute step: {str(e)}")
    
    def _validate_action(self, action):
        """Validate action format and values
        
        Args:
            action: Action to validate
            
        Raises:
            ValueError: If action is invalid
        """
        # Check if action is None
        if action is None:
            raise ValueError("Action must be a dictionary, got None")
        
        # Check if action is a dictionary
        if not isinstance(action, dict):
            raise ValueError(f"Action must be a dictionary, got {type(action)}")
        
        # Check if action contains continuous or discrete key
        if "continuous" not in action and "discrete" not in action:
            raise ValueError("Action must contain 'continuous' or 'discrete' key")
        
        # Validate continuous actions
        if "continuous" in action:
            continuous_action = action["continuous"]
            
            # Check if continuous action is a dictionary
            if not isinstance(continuous_action, dict):
                raise ValueError(f"Continuous action must be a dictionary, got {type(continuous_action)}")
            
            # Check continuous action parameters
            for param, value in continuous_action.items():
                # Check if parameter exists in config
                if param not in self.config["action_space"]["continuous"]:
                    logger.warning(f"Unknown continuous parameter: {param}")
                    continue
                
                # Check if value is numeric
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    raise ValueError(f"Continuous action value must be numeric, got {type(value)} for {param}")
                
                # Check if value is finite
                if not np.isfinite(value):
                    logger.warning(f"Non-finite value for parameter {param}: {value}, using default")
                    continue
                
                # Check if value is within bounds
                param_config = self.config["action_space"]["continuous"][param]
                min_val = param_config["min"]
                max_val = param_config["max"]
                
                if value < min_val or value > max_val:
                    logger.warning(f"Value for {param} out of bounds: {value}, clipping to [{min_val}, {max_val}]")
        
        # Validate discrete actions
        if "discrete" in action:
            discrete_action = action["discrete"]
            
            # Check if discrete action is a dictionary
            if not isinstance(discrete_action, dict):
                raise ValueError(f"Discrete action must be a dictionary, got {type(discrete_action)}")
            
            # Check discrete action parameters
            for param, value in discrete_action.items():
                # Check if parameter exists in config
                if param not in self.config["action_space"]["discrete"]:
                    logger.warning(f"Unknown discrete parameter: {param}")
                    continue
                
                # Check if value is in allowed values
                allowed_values = self.config["action_space"]["discrete"][param]
                if value not in allowed_values:
                    logger.warning(f"Invalid value for {param}: {value}, must be one of {allowed_values}")
    
    def _apply_action(self, action):
        """Apply action to the environment
        
        Args:
            action: Action to apply
        """
        # Apply continuous actions
        if "continuous" in action:
            continuous_action = action["continuous"]
            
            for param, value in continuous_action.items():
                if param in self.config["action_space"]["continuous"]:
                    # Convert to float and clip to bounds
                    try:
                        value = float(value)
                        if not np.isfinite(value):
                            # Use default value for non-finite values
                            param_config = self.config["action_space"]["continuous"][param]
                            value = (param_config["min"] + param_config["max"]) / 2
                        else:
                            # Clip to bounds
                            param_config = self.config["action_space"]["continuous"][param]
                            min_val = param_config["min"]
                            max_val = param_config["max"]
                            value = np.clip(value, min_val, max_val)
                    except (ValueError, TypeError):
                        # Use default value for non-numeric values
                        param_config = self.config["action_space"]["continuous"][param]
                        value = (param_config["min"] + param_config["max"]) / 2
                    
                    # Store parameter in state
                    self.current_state[f"param_{param}"] = value
        
        # Apply discrete actions
        if "discrete" in action:
            discrete_action = action["discrete"]
            
            for param, value in discrete_action.items():
                if param in self.config["action_space"]["discrete"]:
                    allowed_values = self.config["action_space"]["discrete"][param]
                    
                    # Validate and store parameter
                    if value in allowed_values:
                        self.current_state[f"param_{param}"] = value
                    else:
                        # Use first value as default
                        self.current_state[f"param_{param}"] = allowed_values[0]
    
    def _update_environment(self):
        """Update environment state based on mode"""
        if self.mode == "simulation":
            self._update_simulation()
        else:
            self._update_live()
    
    def _update_simulation(self):
        """Update environment in simulation mode"""
        # Move to next data point
        self.current_data_idx += 1
        
        # Check if we've reached the end of the data
        if self.current_data_idx >= len(self.historical_data):
            logger.info("Reached end of historical data")
            return
        
        # Update current market state
        self._update_market_state()
        
        # Simulate trading based on current parameters
        self._simulate_trading()
    
    def _update_live(self):
        """Update environment in live mode"""
        if not self.trading_system:
            logger.error("Trading system not available for live update")
            return
        
        # Get current market state from trading system
        market_state = self.trading_system.get_market_state()
        
        # Update state with market data
        for key, value in market_state.items():
            self.current_state[key] = value
        
        # Apply current parameters to trading system
        parameters = {k.replace("param_", ""): v for k, v in self.current_state.items() if k.startswith("param_")}
        self.trading_system.update_parameters(parameters)
        
        # Get performance metrics from trading system
        metrics = self.trading_system.get_performance_metrics()
        self.performance_metrics.update(metrics)
    
    def _update_market_state(self):
        """Update market state from historical data"""
        # Get current data point
        current_data = self.historical_data.iloc[self.current_data_idx]
        
        # Update basic market features
        self.current_state["price_1m"] = current_data["price"]
        self.current_state["order_imbalance"] = current_data["order_imbalance"]
        self.current_state["spread"] = current_data.get("bid_ask_spread", 0.001)  # Default if not available
        self.current_state["volatility"] = current_data.get("volatility", 0.01)  # Default if not available
        self.current_state["session"] = current_data.get("session", 0)  # Default if not available
        
        # Update technical indicators
        self.current_state["rsi_1m"] = current_data.get("rsi", 50)  # Default if not available
        self.current_state["macd_1m"] = current_data.get("macd", 0)  # Default if not available
        self.current_state["bb_1m"] = current_data.get("bb_percent_b", 0.5)  # Default if not available
        
        # Calculate price changes
        if self.current_data_idx > 0:
            prev_price = self.historical_data.iloc[self.current_data_idx - 1]["price"]
            self.current_state["price_change_1m"] = (current_data["price"] - prev_price) / prev_price
        else:
            self.current_state["price_change_1m"] = 0.0
        
        # Update multi-timeframe data
        self._update_multi_timeframe_data()
    
    def _update_multi_timeframe_data(self):
        """Update data for multiple timeframes"""
        # Define timeframes in minutes
        timeframes = {
            "5m": 5,
            "15m": 15,
            "1h": 60
        }
        
        for tf_name, tf_minutes in timeframes.items():
            # Calculate start index for this timeframe
            start_idx = max(0, self.current_data_idx - tf_minutes)
            
            # Get data for this timeframe
            tf_data = self.historical_data.iloc[start_idx:self.current_data_idx + 1]
            
            if len(tf_data) > 0:
                # Price for this timeframe (last value)
                self.current_state[f"price_{tf_name}"] = tf_data.iloc[-1]["price"]
                
                # Price change for this timeframe
                if len(tf_data) > 1:
                    first_price = tf_data.iloc[0]["price"]
                    last_price = tf_data.iloc[-1]["price"]
                    self.current_state[f"price_change_{tf_name}"] = (last_price - first_price) / first_price
                else:
                    self.current_state[f"price_change_{tf_name}"] = 0.0
                
                # Technical indicators for this timeframe
                if "rsi" in tf_data.columns:
                    self.current_state[f"rsi_{tf_name}"] = tf_data.iloc[-1]["rsi"]
                else:
                    self.current_state[f"rsi_{tf_name}"] = 50.0
                
                if "macd" in tf_data.columns:
                    self.current_state[f"macd_{tf_name}"] = tf_data.iloc[-1]["macd"]
                else:
                    self.current_state[f"macd_{tf_name}"] = 0.0
                
                if "bb_percent_b" in tf_data.columns:
                    self.current_state[f"bb_{tf_name}"] = tf_data.iloc[-1]["bb_percent_b"]
                else:
                    self.current_state[f"bb_{tf_name}"] = 0.5
            else:
                # Default values if not enough data
                self.current_state[f"price_{tf_name}"] = self.current_state["price_1m"]
                self.current_state[f"price_change_{tf_name}"] = 0.0
                self.current_state[f"rsi_{tf_name}"] = 50.0
                self.current_state[f"macd_{tf_name}"] = 0.0
                self.current_state[f"bb_{tf_name}"] = 0.5
    
    def _simulate_trading(self):
        """Simulate trading based on current parameters"""
        # Get current price
        current_price = self.current_state["price_1m"]
        
        # Get current parameters
        imbalance_threshold = self.current_state.get("param_imbalance_threshold", 0.2)
        momentum_threshold = self.current_state.get("param_momentum_threshold", 0.05)
        volatility_threshold = self.current_state.get("param_volatility_threshold", 0.1)
        rsi_threshold = self.current_state.get("param_rsi_threshold", 70.0)
        
        # Get current market features
        order_imbalance = self.current_state["order_imbalance"]
        price_change_1m = self.current_state["price_change_1m"]
        volatility = self.current_state["volatility"]
        rsi_1m = self.current_state["rsi_1m"]
        
        # Generate trading signals
        buy_signal = (
            order_imbalance > imbalance_threshold and
            price_change_1m > momentum_threshold and
            volatility < volatility_threshold and
            rsi_1m < 100 - rsi_threshold
        )
        
        sell_signal = (
            order_imbalance < -imbalance_threshold and
            price_change_1m < -momentum_threshold and
            volatility < volatility_threshold and
            rsi_1m > rsi_threshold
        )
        
        # Execute trades based on signals
        if buy_signal and "BTC" not in self.positions:
            # Calculate position size
            position_size = self.current_balance * self.config["simulation"]["max_position_size"]
            
            # Calculate transaction cost
            transaction_cost = position_size * self.config["simulation"]["transaction_cost"]
            
            # Execute buy
            self.positions["BTC"] = {
                "size": position_size - transaction_cost,
                "entry_price": current_price,
                "entry_time": self.current_data_idx
            }
            
            # Update balance
            self.current_balance -= position_size
            
            # Record trade
            self.trades.append({
                "type": "buy",
                "time": self.current_data_idx,
                "price": current_price,
                "size": position_size - transaction_cost,
                "cost": transaction_cost
            })
            
            self.episode_trades += 1
            
            logger.debug(f"BUY signal at price {current_price}, position size: {position_size - transaction_cost}")
        
        elif sell_signal and "BTC" in self.positions:
            # Get position details
            position = self.positions["BTC"]
            position_value = position["size"] * current_price
            
            # Calculate transaction cost
            transaction_cost = position_value * self.config["simulation"]["transaction_cost"]
            
            # Calculate PnL
            entry_value = position["size"] * position["entry_price"]
            exit_value = position_value - transaction_cost
            pnl = exit_value - entry_value
            
            # Execute sell
            del self.positions["BTC"]
            
            # Update balance
            self.current_balance += position_value - transaction_cost
            
            # Update PnL
            self.performance_metrics["pnl"] += pnl
            self.episode_pnl += pnl
            
            # Record trade
            self.trades.append({
                "type": "sell",
                "time": self.current_data_idx,
                "price": current_price,
                "size": position["size"],
                "cost": transaction_cost,
                "pnl": pnl
            })
            
            self.episode_trades += 1
            
            logger.debug(f"SELL signal at price {current_price}, PnL: {pnl}")
        
        # Update performance metrics
        self._update_performance_metrics()
    
    def _update_performance_metrics(self):
        """Update performance metrics based on current state"""
        # Calculate unrealized PnL
        unrealized_pnl = 0.0
        current_price = self.current_state["price_1m"]
        
        for symbol, position in self.positions.items():
            position_value = position["size"] * current_price
            entry_value = position["size"] * position["entry_price"]
            unrealized_pnl += position_value - entry_value
        
        # Calculate total equity
        total_equity = self.current_balance
        for symbol, position in self.positions.items():
            total_equity += position["size"] * current_price
        
        # Calculate drawdown
        if len(self.trades) > 0:
            # Calculate peak equity
            peak_equity = total_equity
            for i in range(len(self.reward_history)):
                if i > 0:
                    peak_equity = max(peak_equity, peak_equity + self.reward_history[i])
            
            # Calculate drawdown
            drawdown = (peak_equity - total_equity) / peak_equity if peak_equity > 0 else 0.0
            self.performance_metrics["max_drawdown"] = max(self.performance_metrics["max_drawdown"], drawdown)
        
        # Calculate win rate
        if len(self.trades) > 0:
            winning_trades = sum(1 for trade in self.trades if trade.get("pnl", 0) > 0)
            self.performance_metrics["win_rate"] = winning_trades / len(self.trades)
        
        # Calculate trade count
        self.performance_metrics["trade_count"] = len(self.trades)
        
        # Calculate profit factor
        if len(self.trades) > 0:
            gross_profit = sum(trade.get("pnl", 0) for trade in self.trades if trade.get("pnl", 0) > 0)
            gross_loss = sum(abs(trade.get("pnl", 0)) for trade in self.trades if trade.get("pnl", 0) < 0)
            self.performance_metrics["profit_factor"] = gross_profit / gross_loss if gross_loss > 0 else 1.0
        
        # Calculate Sharpe ratio (simplified)
        if len(self.reward_history) > 0:
            returns = np.array(self.reward_history)
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            self.performance_metrics["sharpe_ratio"] = mean_return / std_return if std_return > 0 else 0.0
        
        # Calculate volatility
        if len(self.reward_history) > 0:
            self.performance_metrics["volatility"] = np.std(self.reward_history)
        
        # Calculate recovery factor
        if self.performance_metrics["max_drawdown"] > 0:
            self.performance_metrics["recovery_factor"] = self.performance_metrics["pnl"] / self.performance_metrics["max_drawdown"]
        
        # Update state with metrics
        for metric, value in self.performance_metrics.items():
            self.current_state[f"metric_{metric}"] = value
        
        # Add PnL percentage
        self.current_state["metric_pnl_percent"] = (self.performance_metrics["pnl"] / self.initial_balance) * 100 if self.initial_balance > 0 else 0.0
    
    def _calculate_reward(self):
        """Calculate reward based on current state
        
        Returns:
            float: Reward value
        """
        # Get reward weights
        weights = self.config["reward"]["weights"]
        
        # Calculate PnL component
        pnl_change = self.performance_metrics["pnl"] - (self.previous_state.get("metric_pnl", 0.0) if self.previous_state else 0.0)
        pnl_reward = pnl_change * weights["pnl"]
        
        # Calculate Sharpe ratio component
        sharpe_reward = self.performance_metrics["sharpe_ratio"] * weights["sharpe"]
        
        # Calculate win rate component
        win_rate_reward = self.performance_metrics["win_rate"] * weights["win_rate"]
        
        # Calculate trading frequency penalty
        trade_count_change = self.performance_metrics["trade_count"] - (self.previous_state.get("metric_trade_count", 0) if self.previous_state else 0)
        trading_frequency_penalty = trade_count_change * weights["trading_frequency"]
        
        # Calculate drawdown penalty
        drawdown_penalty = self.performance_metrics["max_drawdown"] * weights["drawdown"]
        
        # Calculate slippage penalty
        slippage_penalty = 0.0
        if len(self.trades) > 0 and self.trades[-1]["time"] == self.current_data_idx:
            # New trade executed in this step
            slippage_penalty = self.trades[-1].get("cost", 0.0) * weights["slippage"]
        
        # Calculate total reward
        reward = pnl_reward + sharpe_reward + win_rate_reward + trading_frequency_penalty + drawdown_penalty + slippage_penalty
        
        return reward
    
    def _is_done(self):
        """Check if episode is done
        
        Returns:
            bool: Whether episode is done
        """
        # Check if maximum steps reached
        if self.episode_step >= self.max_episode_steps:
            logger.info(f"Episode done: reached max steps ({self.max_episode_steps})")
            return True
        
        # Check if end of data reached in simulation mode
        if self.mode == "simulation" and self.current_data_idx >= len(self.historical_data) - 1:
            logger.info("Episode done: reached end of historical data")
            return True
        
        # Check if bankruptcy (balance <= 0)
        total_equity = self.current_balance
        current_price = self.current_state["price_1m"]
        
        for symbol, position in self.positions.items():
            total_equity += position["size"] * current_price
        
        if total_equity <= 0:
            logger.info("Episode done: bankruptcy (balance <= 0)")
            return True
        
        # Check if drawdown exceeds threshold
        max_drawdown = self.performance_metrics["max_drawdown"]
        if max_drawdown > 0.5:  # 50% drawdown threshold
            logger.info(f"Episode done: max drawdown ({max_drawdown:.2f}) exceeded threshold")
            return True
        
        return False
    
    def _get_state(self):
        """Get current state
        
        Returns:
            dict: Current state
        """
        return self.current_state.copy()
    
    def _get_info(self):
        """Get additional info
        
        Returns:
            dict: Additional info
        """
        return {
            "performance_metrics": self.performance_metrics.copy(),
            "positions": self.positions.copy(),
            "balance": self.current_balance,
            "episode_step": self.episode_step,
            "episode_reward": self.episode_reward,
            "episode_trades": self.episode_trades,
            "episode_pnl": self.episode_pnl
        }
    
    def render(self, mode="human"):
        """Render the environment
        
        Args:
            mode: Rendering mode
            
        Returns:
            None or rendered frame
        """
        if mode == "human":
            # Print current state
            print(f"Step: {self.episode_step}")
            print(f"Balance: {self.current_balance:.2f}")
            print(f"PnL: {self.performance_metrics['pnl']:.2f}")
            print(f"Positions: {self.positions}")
            print(f"Reward: {self.reward_history[-1] if len(self.reward_history) > 0 else 0.0:.4f}")
            print("-" * 50)
        elif mode == "rgb_array":
            # TODO: Implement visualization
            return None
    
    def close(self):
        """Close the environment"""
        logger.info("Environment closed")
