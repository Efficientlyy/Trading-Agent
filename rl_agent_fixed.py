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
                 update_timestep=2048):
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
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bounds = action_bounds
        self.device = torch.device(device)
        
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
        
        # Initialize memory buffers
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
    
    def select_action(self, state, deterministic=False):
        """Select action based on current policy
        
        Args:
            state: Current state
            deterministic: Whether to use deterministic policy
            
        Returns:
            tuple: (action, log_prob, value)
        """
        # Convert state to tensor
        if isinstance(state, np.ndarray):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        elif isinstance(state, torch.Tensor):
            state_tensor = state.unsqueeze(0).to(self.device) if state.dim() == 1 else state.to(self.device)
        else:
            raise ValueError(f"Unsupported state type: {type(state)}")
        
        # Get action from policy
        with torch.no_grad():
            action, log_prob, _ = self.actor.get_action(state_tensor, deterministic)
            value = self.critic(state_tensor)
        
        # Convert to numpy and scale to action bounds
        action_np = action.cpu().numpy().flatten()
        
        # Scale action to bounds
        scaled_action = self._scale_action(action_np)
        
        # Store in memory if not deterministic
        if not deterministic:
            if isinstance(state, np.ndarray):
                self.states.append(state)
            else:
                self.states.append(state.cpu().numpy())
            self.actions.append(action_np)
            self.log_probs.append(log_prob.cpu().numpy())
            self.values.append(value.cpu().numpy())
        
        return scaled_action, log_prob.cpu().numpy() if log_prob is not None else None, value.cpu().numpy()
    
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
            low, high = self.action_bounds[i]
            scaled_action[i] = low + (action[i] + 1.0) * 0.5 * (high - low)
        
        return scaled_action
    
    def _unscale_action(self, scaled_action):
        """Unscale action from actual bounds to [-1, 1]
        
        Args:
            scaled_action: Scaled action
            
        Returns:
            numpy.ndarray: Unscaled action in [-1, 1] range
        """
        unscaled_action = np.zeros_like(scaled_action)
        for i in range(len(scaled_action)):
            low, high = self.action_bounds[i]
            unscaled_action[i] = 2.0 * (scaled_action[i] - low) / (high - low) - 1.0
        
        return unscaled_action
    
    def update(self):
        """Update policy using PPO algorithm
        
        Returns:
            dict: Update metrics
        """
        # Check if enough data is collected
        if len(self.states) < self.mini_batch_size:
            logger.warning(f"Not enough data for update: {len(self.states)} < {self.mini_batch_size}")
            return {
                "actor_loss": 0.0,
                "critic_loss": 0.0,
                "entropy": 0.0,
                "update_status": "skipped",
                "reason": "insufficient_data"
            }
        
        # Calculate advantages and returns
        returns, advantages = self._compute_gae()
        
        # Convert to tensors
        try:
            states = torch.FloatTensor(np.array(self.states)).to(self.device)
            actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
            old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(self.device)
            returns = torch.FloatTensor(returns).to(self.device)
            advantages = torch.FloatTensor(advantages).to(self.device)
        except Exception as e:
            logger.error(f"Error converting data to tensors: {str(e)}")
            return {
                "actor_loss": 0.0,
                "critic_loss": 0.0,
                "entropy": 0.0,
                "update_status": "failed",
                "reason": f"tensor_conversion_error: {str(e)}"
            }
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        actor_losses = []
        critic_losses = []
        entropies = []
        
        # Mini-batch update
        batch_size = len(self.states)
        mini_batch_size = min(self.mini_batch_size, batch_size)
        
        for _ in range(self.ppo_epochs):
            # Generate random indices
            indices = np.random.permutation(batch_size)
            
            # Update in mini-batches
            for start_idx in range(0, batch_size, mini_batch_size):
                end_idx = min(start_idx + mini_batch_size, batch_size)
                mb_indices = indices[start_idx:end_idx]
                
                # Get mini-batch data
                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_returns = returns[mb_indices]
                mb_advantages = advantages[mb_indices]
                
                # Get current policy outputs
                _, log_std = self.actor(mb_states)
                std = torch.exp(log_std)
                dist = Normal(self.actor.mean_layer(self.actor.feature_extractor(mb_states)), std)
                
                # Get log probabilities
                mb_new_log_probs = dist.log_prob(mb_actions).sum(dim=-1, keepdim=True)
                
                # Get entropy
                mb_entropy = dist.entropy().sum(dim=-1, keepdim=True).mean()
                
                # Get state values
                mb_values = self.critic(mb_states)
                
                # Calculate ratios
                ratios = torch.exp(mb_new_log_probs - mb_old_log_probs)
                
                # Calculate surrogate losses
                surr1 = ratios * mb_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.clip_param, 1.0 + self.clip_param) * mb_advantages
                
                # Calculate actor loss
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Calculate critic loss
                critic_loss = nn.MSELoss()(mb_values, mb_returns)
                
                # Calculate total loss
                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * mb_entropy
                
                # Update actor
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                
                # Clip gradients
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                
                # Update networks
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                
                # Store metrics
                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropies.append(mb_entropy.item())
        
        # Update metrics
        mean_actor_loss = np.mean(actor_losses)
        mean_critic_loss = np.mean(critic_losses)
        mean_entropy = np.mean(entropies)
        
        self.metrics["actor_losses"].append(mean_actor_loss)
        self.metrics["critic_losses"].append(mean_critic_loss)
        self.metrics["entropy"].append(mean_entropy)
        
        # Clear memory
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
        # Increment training step
        self.training_step += 1
        
        logger.info(f"Policy updated at step {self.training_step}")
        logger.info(f"Actor Loss: {mean_actor_loss:.4f}, Critic Loss: {mean_critic_loss:.4f}, Entropy: {mean_entropy:.4f}")
        
        # Return update metrics
        return {
            "actor_loss": float(mean_actor_loss),
            "critic_loss": float(mean_critic_loss),
            "entropy": float(mean_entropy),
            "update_status": "success",
            "training_step": self.training_step
        }
    
    def _compute_gae(self):
        """Compute Generalized Advantage Estimation
        
        Returns:
            tuple: (returns, advantages)
        """
        # Convert to numpy arrays
        rewards = np.array(self.rewards)
        values = np.array(self.values).reshape(-1)
        dones = np.array(self.dones)
        
        # Get next value (bootstrap)
        if len(self.states) > 0:
            state = self.states[-1]
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                next_value = self.critic(state_tensor).cpu().numpy()[0, 0]
        else:
            next_value = 0.0
        
        # Initialize arrays
        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)
        
        # Initialize for GAE calculation
        gae = 0
        
        # Compute returns and advantages (backwards)
        for t in reversed(range(len(rewards))):
            # Calculate next value
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            
            # Calculate next non-terminal
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[-1]
            else:
                next_non_terminal = 1.0 - dones[t]
            
            # Calculate delta
            delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
            
            # Calculate GAE
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            
            # Store advantage and return
            advantages[t] = gae
            returns[t] = gae + values[t]
        
        return returns, advantages
    
    def store_transition(self, reward, done):
        """Store transition in memory
        
        Args:
            reward: Reward received
            done: Whether episode is done
        """
        self.rewards.append(reward)
        self.dones.append(done)
        
        # Update training step
        self.training_step += 1
        
        # Check if update is needed
        if self.training_step % self.update_timestep == 0:
            return self.update()
        return None
    
    def end_episode(self, episode_reward, episode_length):
        """End episode and update metrics
        
        Args:
            episode_reward: Total reward for the episode
            episode_length: Length of the episode
        """
        self.episode_count += 1
        self.metrics["episode_rewards"].append(episode_reward)
        self.metrics["episode_lengths"].append(episode_length)
        
        # Update best reward
        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
            logger.info(f"New best reward: {self.best_reward:.4f}")
        
        logger.info(f"Episode {self.episode_count} ended with reward {episode_reward:.4f} and length {episode_length}")
    
    def save(self, path):
        """Save agent to file
        
        Args:
            path: Path to save agent
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Convert action bounds to list for JSON serialization
        action_bounds_list = [(float(low), float(high)) for low, high in self.action_bounds]
        
        # Save model
        save_dict = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'best_reward': float(self.best_reward),
            'metrics': {
                'episode_rewards': [float(r) for r in self.metrics['episode_rewards']],
                'episode_lengths': [int(l) for l in self.metrics['episode_lengths']],
                'actor_losses': [float(l) for l in self.metrics['actor_losses']],
                'critic_losses': [float(l) for l in self.metrics['critic_losses']],
                'entropy': [float(e) for e in self.metrics['entropy']]
            },
            'hyperparams': {
                'state_dim': int(self.state_dim),
                'action_dim': int(self.action_dim),
                'action_bounds': action_bounds_list,
                'lr_actor': float(self.lr_actor),
                'lr_critic': float(self.lr_critic),
                'gamma': float(self.gamma),
                'gae_lambda': float(self.gae_lambda),
                'clip_param': float(self.clip_param),
                'value_coef': float(self.value_coef),
                'entropy_coef': float(self.entropy_coef),
                'max_grad_norm': float(self.max_grad_norm),
                'ppo_epochs': int(self.ppo_epochs),
                'mini_batch_size': int(self.mini_batch_size),
                'update_timestep': int(self.update_timestep)
            }
        }
        
        try:
            torch.save(save_dict, path)
            logger.info(f"Agent saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Error saving agent to {path}: {str(e)}")
            return False
    
    def load(self, path):
        """Load agent from file
        
        Args:
            path: Path to load agent from
            
        Returns:
            bool: Whether loading was successful
        """
        if not os.path.exists(path):
            logger.error(f"File not found: {path}")
            return False
        
        try:
            # Load model
            checkpoint = torch.load(path, map_location=self.device)
            
            # Load hyperparameters
            hyperparams = checkpoint['hyperparams']
            self.state_dim = hyperparams['state_dim']
            self.action_dim = hyperparams['action_dim']
            self.action_bounds = hyperparams['action_bounds']
            self.lr_actor = hyperparams['lr_actor']
            self.lr_critic = hyperparams['lr_critic']
            self.gamma = hyperparams['gamma']
            self.gae_lambda = hyperparams['gae_lambda']
            self.clip_param = hyperparams['clip_param']
            self.value_coef = hyperparams['value_coef']
            self.entropy_coef = hyperparams['entropy_coef']
            self.max_grad_norm = hyperparams['max_grad_norm']
            self.ppo_epochs = hyperparams['ppo_epochs']
            self.mini_batch_size = hyperparams['mini_batch_size']
            self.update_timestep = hyperparams['update_timestep']
            
            # Reinitialize networks with correct dimensions
            self.actor = ActorNetwork(self.state_dim, self.action_dim).to(self.device)
            self.critic = CriticNetwork(self.state_dim).to(self.device)
            
            # Load state dicts
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            
            # Reinitialize optimizers
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)
            
            # Load optimizer state dicts
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            
            # Load training state
            self.training_step = checkpoint['training_step']
            self.episode_count = checkpoint['episode_count']
            self.best_reward = checkpoint['best_reward']
            self.metrics = checkpoint['metrics']
            
            logger.info(f"Agent loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Error loading agent from {path}: {str(e)}")
            return False
    
    def get_metrics(self):
        """Get agent metrics
        
        Returns:
            dict: Agent metrics
        """
        return {
            "training_step": self.training_step,
            "episode_count": self.episode_count,
            "best_reward": float(self.best_reward),
            "recent_rewards": self.metrics["episode_rewards"][-10:] if len(self.metrics["episode_rewards"]) > 0 else [],
            "recent_actor_losses": self.metrics["actor_losses"][-10:] if len(self.metrics["actor_losses"]) > 0 else [],
            "recent_critic_losses": self.metrics["critic_losses"][-10:] if len(self.metrics["critic_losses"]) > 0 else [],
            "recent_entropy": self.metrics["entropy"][-10:] if len(self.metrics["entropy"]) > 0 else []
        }
