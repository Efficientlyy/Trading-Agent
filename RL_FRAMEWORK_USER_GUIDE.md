# Reinforcement Learning Framework User Guide

## Overview

This user guide provides comprehensive documentation for the Reinforcement Learning (RL) framework implemented in the Trading-Agent system. The framework enables automated optimization of trading parameters through reinforcement learning techniques.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Components](#components)
4. [Configuration](#configuration)
5. [Usage Examples](#usage-examples)
6. [Performance Optimization](#performance-optimization)
7. [Troubleshooting](#troubleshooting)
8. [API Reference](#api-reference)
9. [Advanced Topics](#advanced-topics)

## Installation

The RL framework is part of the Trading-Agent system and requires the following dependencies:

```bash
# Core dependencies
pip install numpy pandas torch

# Optional dependencies for performance optimization
pip install psutil memory_profiler

# Visualization dependencies
pip install matplotlib seaborn
```

## Quick Start

Here's a minimal example to get started with the RL framework:

```python
from rl_environment_improved import TradingRLEnvironment
from rl_agent import PPOAgent
from rl_integration import RLIntegration

# Initialize integration with simulation mode
integration = RLIntegration(
    config_path="config/rl_config.json",
    historical_data_path="data/historical_data.csv",
    model_save_path="models/rl_agent.pt",
    mode="simulation"
)

# Train the agent
integration.train(num_episodes=50, max_steps=1000)

# Evaluate the agent
eval_metrics = integration.evaluate(num_episodes=10)
print(f"Average reward: {eval_metrics['avg_reward']}")
print(f"Average PnL: ${eval_metrics['avg_pnl']}")

# Get optimal parameters
optimal_params = integration.get_optimal_parameters()
print(f"Optimal parameters: {optimal_params}")
```

## Components

The RL framework consists of four main components:

### 1. RL Environment (`rl_environment_improved.py`)

The environment simulates the trading system and market conditions, providing a standardized interface for the agent to interact with. It implements the following key features:

- **State Space**: Represents market features and agent state
- **Action Space**: Defines parameter adjustments (continuous and discrete)
- **Reward Function**: Evaluates trading performance
- **Simulation**: Simulates trading based on historical data

```python
# Create environment
env = TradingRLEnvironment(
    mode="simulation",
    config_path="config/rl_config.json",
    historical_data_path="data/historical_data.csv",
    random_seed=42,
    initial_balance=10000.0
)

# Reset environment
state = env.reset()

# Take a step
action = {
    "continuous": {
        "imbalance_threshold": 0.2,
        "momentum_threshold": 0.05
    }
}
next_state, reward, done, info = env.step(action)
```

### 2. RL Agent (`rl_agent.py`)

The agent implements the Proximal Policy Optimization (PPO) algorithm for learning optimal parameter settings. It includes:

- **Actor Network**: Determines which actions to take
- **Critic Network**: Evaluates the value of states
- **Training Process**: Updates policy based on experience
- **Action Selection**: Chooses actions based on current policy

```python
# Create agent
agent = PPOAgent(
    state_dim=20,
    action_dim=4,
    action_bounds=[
        (0.05, 0.30),  # imbalance_threshold
        (0.01, 0.10),  # momentum_threshold
        (0.02, 0.20),  # volatility_threshold
        (60.0, 80.0)   # rsi_threshold
    ],
    device="cpu"
)

# Select action
state = np.random.rand(20)
action, log_prob, value = agent.select_action(state)

# Save/load model
agent.save("models/agent.pt")
agent.load("models/agent.pt")
```

### 3. RL Integration (`rl_integration.py`)

The integration module connects the RL components with the trading system, providing a unified interface for parameter optimization:

- **Training Interface**: Methods for training the agent
- **Evaluation Interface**: Methods for evaluating performance
- **Parameter Management**: Handles parameter extraction and application

```python
# Create integration
integration = RLIntegration(
    config_path="config/rl_config.json",
    historical_data_path="data/historical_data.csv",
    model_save_path="models/rl_agent.pt",
    mode="simulation"
)

# Train agent
integration.train(num_episodes=50, max_steps=1000)

# Evaluate agent
eval_metrics = integration.evaluate(num_episodes=10)

# Get optimal parameters
optimal_params = integration.get_optimal_parameters()
```

### 4. RL Performance Optimization (`rl_performance_optimization.py`)

The performance optimization module provides tools for profiling and optimizing the RL framework:

- **Profiling**: Measures execution time and memory usage
- **Optimization**: Applies performance improvements
- **Reporting**: Generates optimization reports

```python
# Create optimizer
optimizer = PerformanceOptimizer()

# Profile components
optimizer.profile_environment(num_steps=50)
optimizer.profile_agent(batch_size=32, num_updates=5)
optimizer.profile_integration(num_episodes=1, max_steps=50)
optimizer.profile_memory_usage()

# Apply optimizations
optimizer.optimize_environment()
optimizer.optimize_agent()
optimizer.optimize_integration()

# Generate report
optimizer.generate_optimization_report()
```

## Configuration

The RL framework is configured through a JSON configuration file with the following structure:

```json
{
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
    "episode_length": 1440,
    "warmup_period": 100,
    "data_chunk_size": 5000,
    "max_position_size": 1.0,
    "transaction_cost": 0.001
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
    "max_steps_per_episode": 1440,
    "save_frequency": 10,
    "eval_frequency": 5,
    "log_frequency": 1
  },
  "evaluation": {
    "num_episodes": 5,
    "deterministic": true
  }
}
```

### Configuration Sections

#### State Space

Defines the features included in the state representation:

- **market_features**: Market data features (prices, indicators, etc.)
- **agent_state**: Agent-specific features (parameters, performance metrics)

#### Action Space

Defines the parameters that can be adjusted by the agent:

- **continuous**: Continuous parameters with min/max/step values
- **discrete**: Discrete parameters with valid choices

#### Reward

Defines the reward function components:

- **weights**: Weights for different reward components (PnL, Sharpe, win rate, etc.)

#### Simulation

Defines simulation parameters:

- **episode_length**: Maximum steps per episode
- **warmup_period**: Number of data points to skip at the beginning
- **data_chunk_size**: Number of data points to load at once
- **max_position_size**: Maximum position size as fraction of balance
- **transaction_cost**: Transaction cost per trade

#### Agent Hyperparameters

Defines PPO algorithm hyperparameters:

- **lr_actor**: Learning rate for actor network
- **lr_critic**: Learning rate for critic network
- **gamma**: Discount factor
- **gae_lambda**: GAE lambda parameter
- **clip_param**: PPO clipping parameter
- **value_coef**: Value function coefficient
- **entropy_coef**: Entropy coefficient
- **max_grad_norm**: Maximum gradient norm
- **ppo_epochs**: Number of PPO epochs
- **mini_batch_size**: Mini batch size
- **update_timestep**: Number of timesteps between updates

#### Training

Defines training parameters:

- **num_episodes**: Number of episodes for training
- **max_steps_per_episode**: Maximum steps per episode
- **save_frequency**: Save frequency (in episodes)
- **eval_frequency**: Evaluation frequency (in episodes)
- **log_frequency**: Logging frequency (in episodes)

#### Evaluation

Defines evaluation parameters:

- **num_episodes**: Number of episodes for evaluation
- **deterministic**: Whether to use deterministic policy for evaluation

## Usage Examples

### Basic Training and Evaluation

```python
from rl_integration import RLIntegration

# Initialize integration
integration = RLIntegration(
    config_path="config/rl_config.json",
    historical_data_path="data/historical_data.csv",
    model_save_path="models/rl_agent.pt",
    mode="simulation"
)

# Train agent
integration.train(num_episodes=50, max_steps=1000)

# Evaluate agent
eval_metrics = integration.evaluate(num_episodes=10)
print(f"Average reward: {eval_metrics['avg_reward']}")
print(f"Average PnL: ${eval_metrics['avg_pnl']}")
print(f"Average win rate: {eval_metrics['avg_win_rate']}")

# Get optimal parameters
optimal_params = integration.get_optimal_parameters()
print(f"Optimal parameters: {optimal_params}")
```

### Custom Environment Setup

```python
from rl_environment_improved import TradingRLEnvironment

# Create custom environment
env = TradingRLEnvironment(
    mode="simulation",
    config_path="config/custom_config.json",
    historical_data_path="data/custom_data.csv",
    random_seed=42,
    initial_balance=100000.0,
    max_episode_steps=2000
)

# Reset environment
state = env.reset()

# Run simulation
total_reward = 0
for i in range(1000):
    # Custom action
    action = {
        "continuous": {
            "imbalance_threshold": 0.2,
            "momentum_threshold": 0.05,
            "volatility_threshold": 0.1,
            "rsi_threshold": 70.0
        },
        "discrete": {
            "trading_mode": "balanced"
        }
    }
    
    # Take step
    next_state, reward, done, info = env.step(action)
    total_reward += reward
    
    if done:
        break

# Get performance summary
performance = env.get_performance_summary()
print(f"Total reward: {total_reward}")
print(f"PnL: ${performance['pnl']}")
print(f"Win rate: {performance['win_rate']}")
```

### Custom Agent Configuration

```python
from rl_agent import PPOAgent
import torch

# Create custom agent
agent = PPOAgent(
    state_dim=20,
    action_dim=4,
    action_bounds=[
        (0.05, 0.30),  # imbalance_threshold
        (0.01, 0.10),  # momentum_threshold
        (0.02, 0.20),  # volatility_threshold
        (60.0, 80.0)   # rsi_threshold
    ],
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    lr_actor=1e-4,
    lr_critic=1e-4,
    gamma=0.95,
    gae_lambda=0.9,
    clip_param=0.1,
    value_coef=0.5,
    entropy_coef=0.01,
    max_grad_norm=0.5,
    ppo_epochs=5,
    mini_batch_size=32,
    update_timestep=1024
)

# Save agent
agent.save("models/custom_agent.pt")
```

### Performance Optimization

```python
from rl_performance_optimization import PerformanceOptimizer
from rl_environment_improved import TradingRLEnvironment
from rl_agent import PPOAgent
from rl_integration import RLIntegration

# Create components
env = TradingRLEnvironment(mode="simulation")
agent = PPOAgent(state_dim=20, action_dim=4, action_bounds=[(0, 1)] * 4)
integration = RLIntegration(mode="simulation")

# Create optimizer
optimizer = PerformanceOptimizer(
    environment=env,
    agent=agent,
    integration=integration,
    profile_output_dir="performance_profiles"
)

# Profile components
optimizer.profile_environment(num_steps=50)
optimizer.profile_agent(batch_size=32, num_updates=5)
optimizer.profile_integration(num_episodes=1, max_steps=50)
optimizer.profile_memory_usage()

# Apply optimizations
optimized_env = optimizer.optimize_environment()
optimized_agent = optimizer.optimize_agent()
optimized_integration = optimizer.optimize_integration()

# Generate report
optimizer.generate_optimization_report()
```

### Integration with Live Trading System

```python
from rl_integration import RLIntegration
from enhanced_flash_trading_signals import EnhancedFlashTradingSignals

# Create trading system
trading_system = EnhancedFlashTradingSignals(env_path=".env-secure/.env")

# Initialize integration with live trading system
integration = RLIntegration(
    config_path="config/rl_config.json",
    model_save_path="models/rl_agent.pt",
    mode="shadow",  # Use shadow mode for testing
    trading_system=trading_system
)

# Load pre-trained model
integration.load_model()

# Get optimal parameters
optimal_params = integration.get_optimal_parameters()

# Apply parameters to trading system
integration.apply_parameters_to_trading_system(optimal_params)
```

## Performance Optimization

The RL framework includes a performance optimization module that can significantly improve execution speed and memory efficiency. Here are some key optimization techniques:

### Environment Optimizations

1. **State Caching**: Memoization of the `_get_state` method to avoid redundant calculations
2. **Reward Calculation**: Optimized reward calculation to only recompute when performance metrics change
3. **Data Access**: Converted DataFrame access to NumPy arrays for faster data retrieval
4. **Indicator Pre-computation**: Pre-compute and cache technical indicators
5. **NumPy Cache**: Cache for NumPy arrays to avoid repeated conversions

### Agent Optimizations

1. **Mixed Precision Training**: Mixed precision training for GPU acceleration
2. **Memory-Efficient GAE**: Optimized Generalized Advantage Estimation for large datasets
3. **Batch Processing**: Batch processing for action selection to reduce overhead
4. **Tensor Operations**: Optimized tensor operations for better performance

### Integration Optimizations

1. **Vectorized Operations**: Replaced loops with vectorized operations where possible
2. **Batch Processing**: Batch processing for environment steps
3. **Pre-computation**: Pre-computation of indicators before training
4. **Efficient Data Handling**: Improved data handling to reduce memory allocations

### Profiling

The performance optimization module includes profiling tools to identify bottlenecks:

```python
from rl_performance_optimization import PerformanceOptimizer

# Create optimizer
optimizer = PerformanceOptimizer()

# Profile environment
env_metrics = optimizer.profile_environment(num_steps=100)
print(f"Average step time: {env_metrics['avg_step_time']:.6f} seconds")

# Profile agent
agent_metrics = optimizer.profile_agent(batch_size=64, num_updates=10)
print(f"Average update time: {agent_metrics['avg_update_time']:.6f} seconds")

# Profile memory usage
memory_metrics = optimizer.profile_memory_usage()
print(f"Total memory usage: {memory_metrics['total_memory_mb']:.2f} MB")
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Out of Memory Errors

**Symptoms**: RuntimeError: CUDA out of memory, or system memory errors

**Solutions**:
- Reduce batch size in agent configuration
- Use data streaming for large historical datasets
- Enable memory-efficient GAE calculation
- Use CPU instead of GPU for smaller models

```python
# Reduce batch size
agent = PPOAgent(
    # ...
    mini_batch_size=32,  # Reduced from 64
    # ...
)

# Use CPU for smaller models
agent = PPOAgent(
    # ...
    device="cpu",
    # ...
)
```

#### 2. Slow Training Performance

**Symptoms**: Training takes too long, especially with large datasets

**Solutions**:
- Apply performance optimizations
- Use GPU acceleration if available
- Reduce episode length or number of episodes
- Use smaller historical datasets for initial testing

```python
# Apply performance optimizations
from rl_performance_optimization import PerformanceOptimizer
optimizer = PerformanceOptimizer(environment=env, agent=agent, integration=integration)
optimizer.optimize_environment()
optimizer.optimize_agent()
optimizer.optimize_integration()

# Use GPU acceleration
agent = PPOAgent(
    # ...
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    # ...
)
```

#### 3. Poor Learning Performance

**Symptoms**: Agent doesn't learn effective policies, rewards don't improve

**Solutions**:
- Adjust reward function weights
- Tune hyperparameters (learning rates, discount factor, etc.)
- Increase training duration
- Check state and action space definitions

```python
# Adjust reward weights in config
config["reward"]["weights"] = {
    "pnl": 1.0,
    "sharpe": 0.5,
    "win_rate": 0.3,
    "trading_frequency": -0.2,
    "drawdown": -0.5,
    "slippage": -0.1
}

# Tune hyperparameters
agent = PPOAgent(
    # ...
    lr_actor=5e-5,  # Reduced learning rate
    gamma=0.99,     # Higher discount factor
    # ...
)
```

#### 4. Configuration Errors

**Symptoms**: ValueError or KeyError when loading configuration

**Solutions**:
- Check JSON syntax in configuration file
- Ensure all required sections are present
- Validate parameter values

```python
# Validate configuration manually
import json
with open("config/rl_config.json", 'r') as f:
    config = json.load(f)

# Check required sections
required_sections = ["state_space", "action_space", "reward", "simulation"]
for section in required_sections:
    if section not in config:
        print(f"Missing required section: {section}")
```

#### 5. Data Format Issues

**Symptoms**: ValueError when loading historical data

**Solutions**:
- Check data file format (CSV, JSON, etc.)
- Ensure required columns are present
- Verify data types and values

```python
# Check data format
import pandas as pd
data = pd.read_csv("data/historical_data.csv")
print(data.columns)
print(data.dtypes)
print(data.head())

# Check for missing values
print(data.isnull().sum())
```

## API Reference

### TradingRLEnvironment

```python
class TradingRLEnvironment:
    def __init__(self, 
                 trading_system=None, 
                 config_path=None, 
                 historical_data_path=None,
                 mode="simulation",
                 random_seed=None,
                 initial_balance=10000.0,
                 max_episode_steps=None):
        """Initialize the RL environment"""
        
    def reset(self):
        """Reset the environment and return the initial state"""
        
    def step(self, action):
        """Take a step in the environment"""
        
    def render(self, mode="human"):
        """Render the current state of the environment"""
        
    def close(self):
        """Clean up resources"""
        
    def seed(self, seed=None):
        """Set random seed"""
        
    def get_performance_summary(self):
        """Get a summary of performance metrics"""
```

### PPOAgent

```python
class PPOAgent:
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
        """Initialize PPO agent"""
        
    def select_action(self, state, deterministic=False):
        """Select action based on current policy"""
        
    def update(self):
        """Update policy using PPO algorithm"""
        
    def store_transition(self, reward, done):
        """Store transition in memory"""
        
    def end_episode(self, episode_reward, episode_length):
        """End episode and update metrics"""
        
    def save(self, path):
        """Save agent to file"""
        
    def load(self, path):
        """Load agent from file"""
```

### RLIntegration

```python
class RLIntegration:
    def __init__(self, 
                 config_path=None, 
                 historical_data_path=None,
                 model_save_path="models/rl_agent.pt",
                 mode="simulation",
                 device="cpu"):
        """Initialize RL integration"""
        
    def train(self, num_episodes=None, max_steps=None):
        """Train the RL agent"""
        
    def evaluate(self, num_episodes=None, deterministic=None):
        """Evaluate the RL agent"""
        
    def save_model(self):
        """Save agent model"""
        
    def load_model(self):
        """Load agent model"""
        
    def get_optimal_parameters(self):
        """Get optimal parameters from trained agent"""
        
    def apply_parameters_to_trading_system(self, parameters):
        """Apply parameters to trading system"""
```

### PerformanceOptimizer

```python
class PerformanceOptimizer:
    def __init__(self, 
                 environment=None, 
                 agent=None, 
                 integration=None,
                 profile_output_dir="performance_profiles"):
        """Initialize performance optimizer"""
        
    def profile_environment(self, num_steps=100):
        """Profile RL environment performance"""
        
    def profile_agent(self, state_dim=20, action_dim=4, batch_size=64, num_updates=10):
        """Profile RL agent performance"""
        
    def profile_integration(self, num_episodes=2, max_steps=100):
        """Profile RL integration performance"""
        
    def profile_memory_usage(self):
        """Profile memory usage of RL components"""
        
    def optimize_environment(self):
        """Apply optimizations to the RL environment"""
        
    def optimize_agent(self):
        """Apply optimizations to the RL agent"""
        
    def optimize_integration(self):
        """Apply optimizations to the RL integration"""
        
    def generate_optimization_report(self):
        """Generate optimization report"""
```

## Advanced Topics

### Custom Reward Functions

You can customize the reward function by modifying the weights in the configuration:

```json
"reward": {
  "weights": {
    "pnl": 1.0,
    "sharpe": 0.5,
    "win_rate": 0.3,
    "trading_frequency": -0.2,
    "drawdown": -0.5,
    "slippage": -0.1
  }
}
```

For more advanced customization, you can override the `_calculate_reward` method in the environment:

```python
class CustomEnvironment(TradingRLEnvironment):
    def _calculate_reward(self):
        """Custom reward calculation"""
        # Get base reward
        base_reward = super()._calculate_reward()
        
        # Add custom components
        custom_component = self._calculate_custom_component()
        
        # Combine rewards
        total_reward = base_reward + custom_component
        
        return total_reward
    
    def _calculate_custom_component(self):
        """Calculate custom reward component"""
        # Custom logic here
        return custom_value
```

### Multi-Agent Training

For more complex scenarios, you can implement multi-agent training:

```python
# Create multiple agents
agents = []
for i in range(3):
    agent = PPOAgent(
        state_dim=20,
        action_dim=4,
        action_bounds=[(0, 1)] * 4,
        device="cpu"
    )
    agents.append(agent)

# Create environment
env = TradingRLEnvironment(mode="simulation")

# Training loop
for episode in range(100):
    state = env.reset()
    done = False
    
    while not done:
        # Select agent based on market conditions
        agent_idx = determine_agent_idx(state)
        agent = agents[agent_idx]
        
        # Select action
        action, _, _ = agent.select_action(state)
        
        # Convert to action dict
        action_dict = convert_to_action_dict(action)
        
        # Take step
        next_state, reward, done, info = env.step(action_dict)
        
        # Store transition for the selected agent
        agent.store_transition(reward, done)
        
        # Update state
        state = next_state
    
    # Update all agents
    for agent in agents:
        agent.update()
```

### Distributed Training

For large-scale training, you can implement distributed training:

```python
import torch.distributed as dist
import torch.multiprocessing as mp

def train_worker(rank, world_size):
    # Initialize process group
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://localhost:12345',
        world_size=world_size,
        rank=rank
    )
    
    # Create agent with distributed data parallel
    agent = PPOAgent(
        state_dim=20,
        action_dim=4,
        action_bounds=[(0, 1)] * 4,
        device=f"cuda:{rank}"
    )
    agent.actor = torch.nn.parallel.DistributedDataParallel(
        agent.actor,
        device_ids=[rank]
    )
    agent.critic = torch.nn.parallel.DistributedDataParallel(
        agent.critic,
        device_ids=[rank]
    )
    
    # Create environment
    env = TradingRLEnvironment(mode="simulation")
    
    # Training loop
    for episode in range(100):
        # Training logic here
        pass

# Launch distributed training
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train_worker, args=(world_size,), nprocs=world_size)
```

### Custom Neural Network Architectures

You can customize the neural network architectures for the actor and critic:

```python
class CustomActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CustomActorNetwork, self).__init__()
        
        # Custom architecture
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # LSTM layer for sequential data
        self.lstm = nn.LSTM(64, 64, batch_first=True)
        
        # Output layers
        self.mean_layer = nn.Linear(64, action_dim)
        self.log_std_layer = nn.Linear(64, action_dim)
    
    def forward(self, state):
        # Extract features
        features = self.feature_extractor(state)
        
        # Reshape for LSTM
        features = features.unsqueeze(0)  # Add batch dimension
        
        # LSTM processing
        lstm_out, _ = self.lstm(features)
        lstm_out = lstm_out.squeeze(0)  # Remove batch dimension
        
        # Output
        action_mean = self.mean_layer(lstm_out)
        action_log_std = self.log_std_layer(lstm_out)
        action_log_std = torch.clamp(action_log_std, -20, 2)
        
        return action_mean, action_log_std

# Use custom network in agent
agent = PPOAgent(
    state_dim=20,
    action_dim=4,
    action_bounds=[(0, 1)] * 4
)
agent.actor = CustomActorNetwork(state_dim=20, action_dim=4).to(agent.device)
```

### Transfer Learning

You can use transfer learning to leverage pre-trained models:

```python
# Load pre-trained agent
pretrained_agent = PPOAgent(
    state_dim=20,
    action_dim=4,
    action_bounds=[(0, 1)] * 4
)
pretrained_agent.load("models/pretrained_agent.pt")

# Create new agent with same architecture
new_agent = PPOAgent(
    state_dim=20,
    action_dim=4,
    action_bounds=[(0, 1)] * 4
)

# Transfer actor weights
new_agent.actor.load_state_dict(pretrained_agent.actor.state_dict())

# Freeze feature extractor
for param in new_agent.actor.feature_extractor.parameters():
    param.requires_grad = False

# Train only output layers
# Training logic here
```

### Hyperparameter Tuning

You can use hyperparameter tuning to find optimal hyperparameters:

```python
import optuna

def objective(trial):
    # Define hyperparameters to tune
    lr_actor = trial.suggest_float("lr_actor", 1e-5, 1e-3, log=True)
    lr_critic = trial.suggest_float("lr_critic", 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.999)
    clip_param = trial.suggest_float("clip_param", 0.1, 0.3)
    
    # Create agent with trial hyperparameters
    agent = PPOAgent(
        state_dim=20,
        action_dim=4,
        action_bounds=[(0, 1)] * 4,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_param=clip_param
    )
    
    # Create integration
    integration = RLIntegration(
        mode="simulation",
        agent=agent
    )
    
    # Train agent
    integration.train(num_episodes=20, max_steps=500)
    
    # Evaluate agent
    eval_metrics = integration.evaluate(num_episodes=5)
    
    # Return metric to optimize
    return eval_metrics["avg_reward"]

# Create study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# Print best hyperparameters
print("Best hyperparameters:", study.best_params)
```
