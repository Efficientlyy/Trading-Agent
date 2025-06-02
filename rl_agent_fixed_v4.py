#!/usr/bin/env python
"""
Reinforcement Learning Agent for Trading Parameter Optimization

This module implements the RL agent that learns optimal parameter settings
for the trading system using Proximal Policy Optimization (PPO).
"""

import os
import sys
import time
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from collections import deque
from typing import Dict, List, Tuple, Any, Optional, Union
import uuid
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rl_agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("rl_agent")

class ActorNetwork(nn.Module):
    """Actor network for PPO algorithm"""
    
    def __init__(self, state_dim, action_dim, hidden_dims=(256, 128, 64)):
        """Initialize actor network
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: Dimensions of hidden layers
        """
        super(ActorNetwork, self).__init__()
        
        # Build network layers
        layers = []
        prev_dim = state_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim
        
        # Output layers for mean and log_std
        self.feature_extractor = nn.Sequential(*layers)
        self.mean_layer = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_layer = nn.Linear(hidden_dims[-1], action_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
        
        # Special initialization for output layers
        nn.init.orthogonal_(self.mean_layer.weight, gain=0.01)
        nn.init.zeros_(self.mean_layer.bias)
        nn.init.orthogonal_(self.log_std_layer.weight, gain=0.01)
        nn.init.zeros_(self.log_std_layer.bias)
    
    def forward(self, state):
        """Forward pass
        
        Args:
            state: State tensor
            
        Returns:
            tuple: (action_mean, action_log_std)
        """
        features = self.feature_extractor(state)
        action_mean = self.mean_layer(features)
        action_log_std = self.log_std_layer(features)
        
        # Clamp log_std for numerical stability
        action_log_std = torch.clamp(action_log_std, -20, 2)
        
        return action_mean, action_log_std
    
    def get_action(self, state, deterministic=False):
        """Get action from policy
        
        Args:
            state: State tensor
            deterministic: Whether to use deterministic policy
            
        Returns:
            tuple: (action, log_prob, entropy)
        """
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        if deterministic:
            action = mean
            log_prob = None
            entropy = None
        else:
            # Create normal distribution
            normal = Normal(mean, std)
            
            # Sample action
            action = normal.sample()
            
            # Calculate log probability and entropy
            log_prob = normal.log_prob(action).sum(dim=-1, keepdim=True)
            entropy = normal.entropy().sum(dim=-1, keepdim=True)
        
        return action, log_prob, entropy

class CriticNetwork(nn.Module):
    """Critic network for PPO algorithm"""
    
    def __init__(self, state_dim, hidden_dims=(256, 128, 64)):
        """Initialize critic network
        
        Args:
            state_dim: Dimension of state space
            hidden_dims: Dimensions of hidden layers
        """
        super(CriticNetwork, self).__init__()
        
        # Build network layers
        layers = []
        prev_dim = state_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim
        
        # Output layer for value
        layers.append(nn.Linear(hidden_dims[-1], 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
    
    def forward(self, state):
        """Forward pass
        
        Args:
            state: State tensor
            
        Returns:
            tensor: State value
        """
        return self.network(state)

class PPOAgent:
    """Proximal Policy Optimization agent for trading parameter optimization"""
    
    def __init__(self, 
                 state_dim, 
                 action_dim, 
                 action_bounds,
                 device="cpu",
                 lr_actor=3e-4,
                 lr_critic=3e-4,
                 gamma=0.99,
                 gae_lambda=0.95,
                 clip_param=0.2,
                 value_coef=0.5,
                 entropy_coef=0.01,
                 max_grad_norm=0.5,
                 ppo_epochs=10,
                 mini_batch_size=64,
                 update_timestep=2048,
                 buffer_size=10000):
        """Initialize PPO agent
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            action_bounds: Bounds for each action dimension (min, max)
            device: Device to run the agent on
            lr_actor: Learning rate for actor network
            lr_critic: Learning rate for critic network
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_param: PPO clipping parameter
            value_coef: Value function coefficient
            entropy_coef: Entropy coefficient
            max_grad_norm: Maximum gradient norm
            ppo_epochs: Number of PPO epochs
            mini_batch_size: Mini batch size
            update_timestep: Number of timesteps between updates
            buffer_size: Maximum buffer size for experience replay
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bounds = action_bounds
        self.device = torch.device(device)
        self.buffer_size = buffer_size
        
        # PPO hyperparameters
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.update_timestep = update_timestep
        
        # Initialize networks
        self.actor = ActorNetwork(state_dim, action_dim).to(self.device)
        self.critic = CriticNetwork(state_dim).to(self.device)
        
        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Initialize memory buffers with fixed size
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
        # Training metrics
        self.training_step = 0
        self.episode_count = 0
        self.best_reward = -float('inf')
        self.metrics = {
            "episode_rewards": [],
            "episode_lengths": [],
            "actor_losses": [],
            "critic_losses": [],
            "entropy": []
        }
    
    def _dict_to_array(self, state_dict):
        """Convert state dictionary to numpy array
        
        Args:
            state_dict: State dictionary
            
        Returns:
            numpy.ndarray: State array
        """
        if not isinstance(state_dict, dict):
            if isinstance(state_dict, np.ndarray):
                return state_dict
            elif isinstance(state_dict, torch.Tensor):
                return state_dict.cpu().numpy()
            else:
                try:
                    # Try to convert to numpy array
                    return np.array(state_dict, dtype=np.float32)
                except:
                    raise ValueError(f"Unsupported state type: {type(state_dict)}")
        
        # Extract values from dictionary and flatten
        state_values = []
        for key, value in sorted(state_dict.items()):
            if isinstance(value, (int, float, bool)):
                state_values.append(float(value))
            elif isinstance(value, np.ndarray):
                state_values.extend(value.flatten())
            elif isinstance(value, torch.Tensor):
                state_values.extend(value.cpu().numpy().flatten())
            else:
                try:
                    # Try to convert to float
                    state_values.append(float(value))
                except:
                    # If conversion fails, use a default value
                    logger.warning(f"Could not convert value for key {key} to float, using 0.0")
                    state_values.append(0.0)
        
        return np.array(state_values, dtype=np.float32)
    
    def select_action(self, state, deterministic=False):
        """Select action based on current policy
        
        Args:
            state: Current state (can be dict, numpy array, or tensor)
            deterministic: Whether to use deterministic policy
            
        Returns:
            tuple: (action, log_prob, value)
        """
        try:
            # Convert state to numpy array if it's a dictionary
            if isinstance(state, dict):
                state_array = self._dict_to_array(state)
            elif isinstance(state, np.ndarray):
                state_array = state.astype(np.float32)
            elif isinstance(state, torch.Tensor):
                state_array = state.cpu().numpy().astype(np.float32)
            else:
                try:
                    # Try to convert to numpy array
                    state_array = np.array(state, dtype=np.float32)
                except:
                    raise ValueError(f"Unsupported state type: {type(state)}")
            
            # Ensure state has correct dimension
            if state_array.shape[0] != self.state_dim:
                logger.warning(f"State dimension mismatch: expected {self.state_dim}, got {state_array.shape[0]}")
                # Pad or truncate state to match expected dimension
                if state_array.shape[0] < self.state_dim:
                    state_array = np.pad(state_array, (0, self.state_dim - state_array.shape[0]), 'constant')
                else:
                    state_array = state_array[:self.state_dim]
            
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state_array).unsqueeze(0).to(self.device)
            
            # Get action from policy
            with torch.no_grad():
                action, log_prob, entropy = self.actor.get_action(state_tensor, deterministic)
                value = self.critic(state_tensor)
            
            # Convert to numpy and scale to action bounds
            action_np = action.cpu().numpy().flatten()
            
            # Scale action to bounds
            scaled_action = self._scale_action(action_np)
            
            # Store in memory if not deterministic
            if not deterministic:
                # Manage buffer size - remove oldest entries if buffer is full
                if len(self.states) >= self.buffer_size:
                    self.states.pop(0)
                    if len(self.actions) > 0:
                        self.actions.pop(0)
                    if len(self.log_probs) > 0:
                        self.log_probs.pop(0)
                    if len(self.values) > 0:
                        self.values.pop(0)
                
                self.states.append(state_array)
                self.actions.append(action_np)
                if log_prob is not None:
                    self.log_probs.append(log_prob.cpu().numpy())
                if value is not None:
                    self.values.append(value.cpu().numpy())
            
            return scaled_action, log_prob.cpu().numpy() if log_prob is not None else None, value.cpu().numpy() if value is not None else None
        
        except Exception as e:
            logger.error(f"Error in select_action: {str(e)}")
            # Return safe default values
            default_action = np.zeros(self.action_dim)
            for i in range(self.action_dim):
                low, high = self.action_bounds[i]
                default_action[i] = (low + high) / 2.0
            return default_action, None, None
    
    def _scale_action(self, action):
        """Scale action from [-1, 1] to actual bounds
        
        Args:
            action: Action in [-1, 1] range
            
        Returns:
            numpy.ndarray: Scaled action
        """
        # Clip action to [-1, 1]
        action = np.clip(action, -1.0, 1.0)
        
        # Scale to actual bounds
        scaled_action = np.zeros_like(action)
        for i in range(len(action)):
            if i < len(self.action_bounds):
                low, high = self.action_bounds[i]
                scaled_action[i] = low + (action[i] + 1.0) * 0.5 * (high - low)
            else:
                # Default to middle of [-1, 1] if bounds not provided
                scaled_action[i] = action[i]
        
        return scaled_action

# Add TradingRLAgent class that was missing
class TradingRLAgent:
    """Trading Reinforcement Learning Agent for end-to-end testing"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        """Initialize Trading RL Agent
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Dimension of hidden layers
        """
        # Ensure state_dim is at least 20 to accommodate all features
        self.state_dim = max(state_dim, 20)
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Define action bounds for each action dimension
        self.action_bounds = [(-1.0, 1.0) for _ in range(action_dim)]
        
        # Initialize PPO agent
        self.agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            action_bounds=self.action_bounds,
            device="cpu",
            lr_actor=3e-4,
            lr_critic=3e-4,
            gamma=0.99,
            buffer_size=10000
        )
        
        logger.info(f"Initialized TradingRLAgent with state_dim={state_dim}, action_dim={action_dim}")
    
    def act(self, state):
        """Select action based on current state
        
        Args:
            state: Current state
            
        Returns:
            numpy.ndarray: Selected action
        """
        action, _, _ = self.agent.select_action(state, deterministic=False)
        return action
    
    def signal_to_state(self, signal):
        """Convert trading signal to state representation
        
        Args:
            signal: Trading signal dictionary
            
        Returns:
            numpy.ndarray: State representation
        """
        # Extract relevant features from signal
        state = np.zeros(self.state_dim)
        
        try:
            # Fill state vector with signal features
            idx = 0
            
            # Basic signal properties
            if 'value' in signal:
                state[idx] = float(signal['value'])
                idx += 1
            
            if 'confidence' in signal:
                state[idx] = float(signal['confidence'])
                idx += 1
            
            # Direction encoding (one-hot)
            if 'direction' in signal:
                direction = signal['direction'].lower()
                if direction == 'buy':
                    state[idx] = 1.0
                    state[idx+1] = 0.0
                    state[idx+2] = 0.0
                elif direction == 'sell':
                    state[idx] = 0.0
                    state[idx+1] = 1.0
                    state[idx+2] = 0.0
                else:  # neutral
                    state[idx] = 0.0
                    state[idx+1] = 0.0
                    state[idx+2] = 1.0
                idx += 3
            
            # Signal type encoding
            if 'type' in signal:
                signal_type = signal['type'].lower()
                if signal_type == 'pattern':
                    state[idx] = 1.0
                    state[idx+1] = 0.0
                elif signal_type == 'order_book':
                    state[idx] = 0.0
                    state[idx+1] = 1.0
                else:
                    state[idx] = 0.0
                    state[idx+1] = 0.0
                idx += 2
            
            # Extract metadata if available
            if 'metadata' in signal and isinstance(signal['metadata'], dict):
                metadata = signal['metadata']
                
                # Price information
                if 'bid_price' in metadata and 'ask_price' in metadata:
                    mid_price = (float(metadata['bid_price']) + float(metadata['ask_price'])) / 2.0
                    state[idx] = mid_price / 100000.0  # Normalize large price values
                    idx += 1
                
                # Spread information
                if 'spread_bps' in metadata:
                    state[idx] = float(metadata['spread_bps'])
                    idx += 1
                
                # Liquidity information
                if 'bid_liquidity' in metadata and 'ask_liquidity' in metadata:
                    bid_liq = float(metadata['bid_liquidity'])
                    ask_liq = float(metadata['ask_liquidity'])
                    state[idx] = bid_liq
                    state[idx+1] = ask_liq
                    state[idx+2] = bid_liq / (bid_liq + ask_liq) if (bid_liq + ask_liq) > 0 else 0.5
                    idx += 3
            
            # Ensure we don't exceed state dimension
            if idx < self.state_dim:
                # Fill remaining state with zeros
                pass
            elif idx > self.state_dim:
                # Truncate state if too large
                state = state[:self.state_dim]
                
            logger.debug(f"Converted signal to state with {idx} features")
            
        except Exception as e:
            logger.error(f"Error converting signal to state: {e}")
            # Return zero state as fallback
            state = np.zeros(self.state_dim)
        
        return state
    
    def action_to_decision(self, action, signal):
        """Convert action to trading decision
        
        Args:
            action: Action from RL agent
            signal: Original trading signal
            
        Returns:
            dict: Trading decision
        """
        try:
            # Extract action components
            if len(action) >= 3:
                action_type = np.argmax(action[:3])  # 0: buy, 1: sell, 2: hold
                size_factor = np.clip(action[2], 0.0, 1.0) if len(action) > 2 else 0.5
            else:
                # Default values if action doesn't have enough components
                action_type = 2  # hold
                size_factor = 0.5
            
            # Determine action direction
            if action_type == 0:
                direction = "buy"
            elif action_type == 1:
                direction = "sell"
            else:
                direction = "hold"
            
            # Skip decision if action is hold
            if direction == "hold":
                return None
            
            # Extract price information from signal
            price = None
            if 'metadata' in signal and isinstance(signal['metadata'], dict):
                metadata = signal['metadata']
                if 'bid_price' in metadata and 'ask_price' in metadata:
                    if direction == "buy":
                        price = float(metadata['ask_price'])  # Buy at ask
                    else:
                        price = float(metadata['bid_price'])  # Sell at bid
            
            # Use default price if not available
            if price is None:
                price = 50000.0  # Default price for testing
            
            # Calculate quantity based on size factor (0.001 to 0.01 BTC)
            base_quantity = 0.001
            max_quantity = 0.01
            quantity = base_quantity + size_factor * (max_quantity - base_quantity)
            
            # Create decision object
            decision = {
                "id": str(uuid.uuid4()),
                "timestamp": signal.get("timestamp", str(datetime.now().isoformat())),
                "symbol": signal.get("symbol", "BTC/USDC"),
                "action": direction,
                "quantity": quantity,
                "price": price,
                "signal_id": signal.get("id", "unknown"),
                "confidence": float(action[2]) if len(action) > 2 else 0.5,
                "metadata": {
                    "source": "rl_agent",
                    "original_signal_type": signal.get("type", "unknown"),
                    "size_factor": float(size_factor)
                }
            }
            
            logger.debug(f"Generated decision: {direction} {quantity} {signal.get('symbol', 'BTC/USDC')} at {price}")
            return decision
            
        except Exception as e:
            logger.error(f"Error converting action to decision: {e}")
            return None
    
    def get_action(self, state):
        """Get action from policy
        
        Args:
            state: State representation
            
        Returns:
            numpy.ndarray: Action
        """
        return self.act(state)
    
    def generate_decisions(self, signals):
        """Generate trading decisions based on signals
        
        Args:
            signals: Trading signals
            
        Returns:
            list: Trading decisions
        """
        decisions = []
        
        for signal in signals:
            # Convert signal to state
            state = self.signal_to_state(signal)
            
            # Get action
            action = self.act(state)
            
            # Convert action to decision
            decision = self._action_to_decision(action, signal)
            
            # Add to decisions
            decisions.append(decision)
        
        logger.info(f"Generated {len(decisions)} decisions from {len(signals)} signals")
        return decisions
    
    def _signal_to_state(self, signal):
        """Convert signal to state
        
        Args:
            signal: Trading signal
            
        Returns:
            numpy.ndarray: State
        """
        # Extract features from signal
        features = []
        
        # Add basic features
        if isinstance(signal, dict):
            # Extract numerical values
            for key in ["confidence", "strength", "volatility"]:
                if key in signal:
                    features.append(float(signal[key]))
                else:
                    features.append(0.0)
            
            # Add price features if available
            if "price" in signal:
                features.append(float(signal["price"]))
            else:
                features.append(0.0)
            
            # Add signal type as one-hot encoding
            signal_type = signal.get("type", "unknown")
            if signal_type == "buy":
                features.extend([1.0, 0.0, 0.0])
            elif signal_type == "sell":
                features.extend([0.0, 1.0, 0.0])
            else:
                features.extend([0.0, 0.0, 1.0])
        else:
            # Default features if signal is not a dictionary
            features = [0.0] * self.state_dim
        
        # Ensure state has correct dimension
        if len(features) < self.state_dim:
            features.extend([0.0] * (self.state_dim - len(features)))
        elif len(features) > self.state_dim:
            features = features[:self.state_dim]
        
        return np.array(features, dtype=np.float32)
    
    def _action_to_decision(self, action, signal):
        """Convert action to decision
        
        Args:
            action: Action from agent
            signal: Original signal
            
        Returns:
            dict: Trading decision
        """
        # Default decision
        decision = {
            "timestamp": signal.get("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S")),
            "symbol": signal.get("symbol", "BTC/USDC"),
            "timeframe": signal.get("timeframe", "5m"),
            "confidence": float(action[0]) if len(action) > 0 else 0.0,
            "quantity": abs(float(action[1])) if len(action) > 1 else 0.1,
            "price": signal.get("price", 0.0),
            "signal_id": signal.get("id", "unknown")
        }
        
        # Determine action type based on first action value
        if len(action) > 0 and action[0] > 0.2:
            decision["action"] = "buy"
        elif len(action) > 0 and action[0] < -0.2:
            decision["action"] = "sell"
        else:
            decision["action"] = "hold"
        
        return decision
    
    def save(self, path):
        """Save agent to file
        
        Args:
            path: Path to save file
            
        Returns:
            bool: Success
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save model weights
            torch.save({
                'actor_state_dict': self.agent.actor.state_dict(),
                'critic_state_dict': self.agent.critic.state_dict(),
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'hidden_dim': self.hidden_dim
            }, path)
            
            logger.info(f"Saved agent to {path}")
            return True
        except Exception as e:
            logger.error(f"Error saving agent: {str(e)}")
            return False
    
    def load(self, path):
        """Load agent from file
        
        Args:
            path: Path to load file
            
        Returns:
            bool: Success
        """
        try:
            # Load model weights
            checkpoint = torch.load(path, map_location=torch.device('cpu'))
            
            # Update parameters
            self.state_dim = checkpoint.get('state_dim', self.state_dim)
            self.action_dim = checkpoint.get('action_dim', self.action_dim)
            self.hidden_dim = checkpoint.get('hidden_dim', self.hidden_dim)
            
            # Reinitialize agent if dimensions changed
            if self.state_dim != self.agent.state_dim or self.action_dim != self.agent.action_dim:
                self.agent = PPOAgent(
                    state_dim=self.state_dim,
                    action_dim=self.action_dim,
                    action_bounds=[(0.0, 1.0) for _ in range(self.action_dim)],
                    device="cpu"
                )
            
            # Load state dictionaries
            self.agent.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.agent.critic.load_state_dict(checkpoint['critic_state_dict'])
            
            logger.info(f"Loaded agent from {path}")
            return True
        except Exception as e:
            logger.error(f"Error loading agent: {str(e)}")
            return False
