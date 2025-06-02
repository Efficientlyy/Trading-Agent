#!/usr/bin/env python
"""
Reinforcement Learning Integration Module for Trading-Agent System

This module integrates the RL agent with the trading signal engine,
providing a unified interface for parameter optimization.
"""

import os
import sys
import time
import json
import logging
import numpy as np
import torch
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union

# Import RL components
from rl_environment_improved import TradingRLEnvironment
from rl_agent_fixed import PPOAgent

# Import trading components
from enhanced_flash_trading_signals import EnhancedFlashTradingSignals
from trading_session_manager import TradingSessionManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rl_integration.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("rl_integration")

class RLIntegration:
    """Integration between RL agent and trading signal engine"""
    
    def __init__(self, 
                 config_path=None, 
                 historical_data_path=None,
                 model_save_path="models/rl_agent.pt",
                 mode="simulation",
                 device="cpu"):
        """Initialize RL integration
        
        Args:
            config_path: Path to configuration file
            historical_data_path: Path to historical data for simulation
            model_save_path: Path to save/load model
            mode: Operation mode ("simulation", "shadow", "assisted", "autonomous")
            device: Device to run the agent on
        """
        self.config_path = config_path
        self.historical_data_path = historical_data_path
        self.model_save_path = model_save_path
        self.mode = mode
        self.device = device
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize trading system (if not in simulation mode)
        self.trading_system = None
        if mode != "simulation":
            self._initialize_trading_system()
        
        # Initialize RL environment
        self.environment = TradingRLEnvironment(
            trading_system=self.trading_system,
            config_path=config_path,
            historical_data_path=historical_data_path,
            mode=mode
        )
        
        # Define state and action dimensions
        state = self.environment.reset()
        self.state_dim = len(state)
        self.action_dim = len(self.config["action_space"]["continuous"])
        self.action_bounds = [
            (self.config["action_space"]["continuous"][param]["min"],
             self.config["action_space"]["continuous"][param]["max"])
            for param in self.config["action_space"]["continuous"]
        ]
        
        # Initialize RL agent
        self.agent = PPOAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            action_bounds=self.action_bounds,
            device=self.device,
            **self.config["agent_hyperparams"]
        )
        
        # Try to load pre-trained model
        if os.path.exists(self.model_save_path):
            self.agent.load(self.model_save_path)
        
        # Training metrics
        self.training_metrics = {
            "episode_rewards": [],
            "episode_lengths": [],
            "parameter_history": [],
            "performance_metrics": []
        }
        
        logger.info(f"RL integration initialized in {mode} mode")
        logger.info(f"State dimension: {self.state_dim}, Action dimension: {self.action_dim}")
    
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
                    "rsi_threshold": {"min": 60.0, "max": 80.0, "step": 1.0}
                },
                "discrete": {
                    "signal_types": ["order_imbalance", "momentum", "volatility", "rsi", "macd", "bb"],
                    "trading_mode": ["aggressive", "balanced", "conservative"],
                    "timeframe_priority": ["1m", "5m", "15m", "1h"]
                }
            },
            "agent_hyperparams": {
                "lr_actor": 3e-4,
                "lr_critic": 3e-4,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_param": 0.2,
                "value_coef": 0.5,
                "entropy_coef": 0.01,
                "max_grad_norm": 0.5,
                "ppo_epochs": 10,
                "mini_batch_size": 64,
                "update_timestep": 2048
            },
            "training": {
                "num_episodes": 100,
                "max_steps_per_episode": 1440,  # 1 day in minutes
                "save_frequency": 10,  # Save every 10 episodes
                "eval_frequency": 5,   # Evaluate every 5 episodes
                "log_frequency": 1     # Log every episode
            },
            "evaluation": {
                "num_episodes": 5,
                "deterministic": True
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
    
    def _initialize_trading_system(self):
        """Initialize trading system for live modes"""
        try:
            # Initialize trading session manager
            session_manager = TradingSessionManager()
            
            # Initialize enhanced flash trading signals
            self.trading_system = EnhancedFlashTradingSignals(env_path=".env-secure/.env")
            
            logger.info("Trading system initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing trading system: {str(e)}")
            raise
    
    def train(self, num_episodes=None, max_steps=None):
        """Train the RL agent
        
        Args:
            num_episodes: Number of episodes to train for (overrides config)
            max_steps: Maximum steps per episode (overrides config)
            
        Returns:
            dict: Training metrics
        """
        # Use config values if not specified
        if num_episodes is None:
            num_episodes = self.config["training"]["num_episodes"]
        if max_steps is None:
            max_steps = self.config["training"]["max_steps_per_episode"]
        
        logger.info(f"Starting training for {num_episodes} episodes, max {max_steps} steps per episode")
        
        for episode in range(1, num_episodes + 1):
            # Reset environment
            state = self.environment.reset()
            
            episode_reward = 0
            episode_steps = 0
            episode_start_time = time.time()
            
            # Store initial parameters
            self.training_metrics["parameter_history"].append({
                "episode": episode,
                "step": 0,
                "parameters": self._extract_parameters(state)
            })
            
            # Episode loop
            done = False
            while not done and episode_steps < max_steps:
                # Select action
                action_dict = self._state_to_action(state)
                
                # Take step in environment
                next_state, reward, done, info = self.environment.step(action_dict)
                
                # Store transition
                update_info = self.agent.store_transition(reward, done)
                
                # Update state
                state = next_state
                
                # Update metrics
                episode_reward += reward
                episode_steps += 1
                
                # Store parameters periodically
                if episode_steps % 100 == 0:
                    self.training_metrics["parameter_history"].append({
                        "episode": episode,
                        "step": episode_steps,
                        "parameters": self._extract_parameters(state)
                    })
                
                # Render environment (for debugging)
                if episode % self.config["training"]["log_frequency"] == 0 and episode_steps % 100 == 0:
                    self.environment.render()
            
            # End of episode
            episode_end_time = time.time()
            episode_duration = episode_end_time - episode_start_time
            
            # Update agent metrics
            self.agent.end_episode(episode_reward, episode_steps)
            
            # Update training metrics
            self.training_metrics["episode_rewards"].append(float(episode_reward))
            self.training_metrics["episode_lengths"].append(int(episode_steps))
            
            # Make a deep copy of performance metrics to avoid reference issues
            if hasattr(self.environment, 'performance_metrics'):
                perf_metrics = {}
                for k, v in self.environment.performance_metrics.items():
                    if isinstance(v, (int, float, bool, str)):
                        perf_metrics[k] = v
                    elif isinstance(v, np.ndarray):
                        perf_metrics[k] = v.tolist()
                    else:
                        perf_metrics[k] = str(v)
                self.training_metrics["performance_metrics"].append(perf_metrics)
            
            # Log progress
            if episode % self.config["training"]["log_frequency"] == 0:
                logger.info(f"Episode {episode}/{num_episodes} - Reward: {episode_reward:.4f}, Steps: {episode_steps}, Duration: {episode_duration:.2f}s")
                if hasattr(self.environment, 'performance_metrics'):
                    logger.info(f"Performance: PnL: ${self.environment.performance_metrics.get('pnl', 0):.2f}, Win Rate: {self.environment.performance_metrics.get('win_rate', 0):.2%}")
            
            # Save model
            if episode % self.config["training"]["save_frequency"] == 0:
                self.save_model()
            
            # Evaluate model
            if episode % self.config["training"]["eval_frequency"] == 0:
                eval_metrics = self.evaluate()
                logger.info(f"Evaluation after episode {episode} - Avg Reward: {eval_metrics['avg_reward']:.4f}, Avg PnL: ${eval_metrics['avg_pnl']:.2f}")
        
        # Final save
        self.save_model()
        
        logger.info("Training completed")
        return self.training_metrics
    
    def evaluate(self, num_episodes=None, deterministic=None):
        """Evaluate the RL agent
        
        Args:
            num_episodes: Number of episodes to evaluate for (overrides config)
            deterministic: Whether to use deterministic policy (overrides config)
            
        Returns:
            dict: Evaluation metrics
        """
        # Use config values if not specified
        if num_episodes is None:
            num_episodes = self.config["evaluation"]["num_episodes"]
        if deterministic is None:
            deterministic = self.config["evaluation"]["deterministic"]
        
        logger.info(f"Starting evaluation for {num_episodes} episodes (deterministic: {deterministic})")
        
        eval_rewards = []
        eval_steps = []
        eval_pnls = []
        eval_win_rates = []
        
        for episode in range(1, num_episodes + 1):
            # Reset environment
            state = self.environment.reset()
            
            episode_reward = 0
            episode_steps = 0
            
            # Episode loop
            done = False
            while not done and episode_steps < self.config["training"]["max_steps_per_episode"]:
                # Select action (deterministic)
                action, _, _ = self.agent.select_action(state, deterministic=deterministic)
                
                # Convert to action dict
                action_dict = self._action_to_dict(action)
                
                # Take step in environment
                next_state, reward, done, info = self.environment.step(action_dict)
                
                # Update state
                state = next_state
                
                # Update metrics
                episode_reward += reward
                episode_steps += 1
            
            # Store metrics
            eval_rewards.append(float(episode_reward))
            eval_steps.append(int(episode_steps))
            
            if hasattr(self.environment, 'performance_metrics'):
                eval_pnls.append(float(self.environment.performance_metrics.get('pnl', 0)))
                eval_win_rates.append(float(self.environment.performance_metrics.get('win_rate', 0)))
            
            logger.info(f"Eval Episode {episode}/{num_episodes} - Reward: {episode_reward:.4f}, Steps: {episode_steps}")
        
        # Calculate averages
        avg_reward = float(np.mean(eval_rewards))
        avg_steps = float(np.mean(eval_steps))
        avg_pnl = float(np.mean(eval_pnls)) if eval_pnls else 0.0
        avg_win_rate = float(np.mean(eval_win_rates)) if eval_win_rates else 0.0
        
        # Create metrics dict
        eval_metrics = {
            "avg_reward": avg_reward,
            "avg_steps": avg_steps,
            "avg_pnl": avg_pnl,
            "avg_win_rate": avg_win_rate,
            "rewards": eval_rewards,
            "steps": eval_steps,
            "pnls": eval_pnls,
            "win_rates": eval_win_rates
        }
        
        logger.info(f"Evaluation completed - Avg Reward: {avg_reward:.4f}, Avg PnL: ${avg_pnl:.2f}, Avg Win Rate: {avg_win_rate:.2%}")
        
        return eval_metrics
    
    def _state_to_action(self, state):
        """Convert state to action using agent
        
        Args:
            state: Current state
            
        Returns:
            dict: Action dictionary
        """
        # Get action from agent
        action, _, _ = self.agent.select_action(state)
        
        # Convert to action dict
        return self._action_to_dict(action)
    
    def _action_to_dict(self, action):
        """Convert action array to action dictionary
        
        Args:
            action: Action array
            
        Returns:
            dict: Action dictionary
        """
        # Create action dict
        action_dict = {"continuous": {}, "discrete": {}}
        
        # Fill continuous actions
        param_names = list(self.config["action_space"]["continuous"].keys())
        for i, param in enumerate(param_names):
            if i < len(action):
                action_dict["continuous"][param] = float(action[i])
        
        return action_dict
    
    def _extract_parameters(self, state):
        """Extract parameters from state
        
        Args:
            state: Current state
            
        Returns:
            dict: Parameters
        """
        # Extract parameters from state
        parameters = {}
        
        # Add continuous parameters
        for param in self.config["action_space"]["continuous"]:
            if f"param_{param}" in state:
                parameters[param] = float(state[f"param_{param}"])
        
        return parameters
    
    def save_model(self):
        """Save agent model"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        
        # Save agent
        success = self.agent.save(self.model_save_path)
        
        if success:
            logger.info(f"Model saved to {self.model_save_path}")
        else:
            logger.error(f"Failed to save model to {self.model_save_path}")
    
    def load_model(self):
        """Load agent model"""
        if os.path.exists(self.model_save_path):
            success = self.agent.load(self.model_save_path)
            if success:
                logger.info(f"Model loaded from {self.model_save_path}")
            else:
                logger.error(f"Failed to load model from {self.model_save_path}")
            return success
        else:
            logger.warning(f"Model file not found: {self.model_save_path}")
            return False
    
    def get_optimal_parameters(self):
        """Get optimal parameters from agent
        
        Returns:
            dict: Optimal parameters
        """
        # Get agent metrics
        metrics = self.agent.get_metrics()
        
        # Get best episode
        if len(self.training_metrics["episode_rewards"]) > 0:
            best_episode_idx = np.argmax(self.training_metrics["episode_rewards"])
            best_episode = best_episode_idx + 1
            
            # Get parameters from best episode
            for param_history in reversed(self.training_metrics["parameter_history"]):
                if param_history["episode"] == best_episode:
                    optimal_params = param_history["parameters"]
                    break
            else:
                # Fallback to current parameters
                state = self.environment.reset()
                optimal_params = self._extract_parameters(state)
        else:
            # No training data, use current parameters
            state = self.environment.reset()
            optimal_params = self._extract_parameters(state)
        
        # Add metadata
        optimal_params["_metadata"] = {
            "training_steps": metrics["training_step"],
            "episodes": metrics["episode_count"],
            "best_reward": metrics["best_reward"],
            "timestamp": datetime.now().isoformat()
        }
        
        return optimal_params
