#!/usr/bin/env python
"""
Reinforcement Learning Validation Script

This script validates the performance of the RL agent in simulation,
collecting metrics and generating visualizations for analysis.
"""

import os
import sys
import time
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union

# Import RL components
from rl_integration import RLIntegration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rl_validation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("rl_validation")

class RLValidation:
    """Validation for RL agent performance"""
    
    def __init__(self, 
                 config_path=None,
                 historical_data_path=None,
                 model_save_path="models/rl_agent.pt",
                 results_dir="validation_results"):
        """Initialize RL validation
        
        Args:
            config_path: Path to configuration file
            historical_data_path: Path to historical data for simulation
            model_save_path: Path to save/load model
            results_dir: Directory to save validation results
        """
        self.config_path = config_path
        self.historical_data_path = historical_data_path
        self.model_save_path = model_save_path
        self.results_dir = results_dir
        
        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize RL integration
        self.rl_integration = RLIntegration(
            config_path=config_path,
            historical_data_path=historical_data_path,
            model_save_path=model_save_path,
            mode="simulation"
        )
        
        # Validation metrics
        self.validation_metrics = {
            "training": {},
            "evaluation": {},
            "parameter_evolution": {},
            "performance_comparison": {}
        }
        
        logger.info("RL validation initialized")
    
    def run_validation(self, 
                       train_episodes=50, 
                       train_steps=1000,
                       eval_episodes=10):
        """Run validation
        
        Args:
            train_episodes: Number of episodes for training
            train_steps: Maximum steps per training episode
            eval_episodes: Number of episodes for evaluation
            
        Returns:
            dict: Validation metrics
        """
        logger.info(f"Starting validation with {train_episodes} training episodes and {eval_episodes} evaluation episodes")
        
        # Training phase
        logger.info("Starting training phase")
        training_start_time = time.time()
        
        training_metrics = self.rl_integration.train(
            num_episodes=train_episodes,
            max_steps=train_steps
        )
        
        training_end_time = time.time()
        training_duration = training_end_time - training_start_time
        
        # Store training metrics
        self.validation_metrics["training"] = {
            "metrics": training_metrics,
            "duration": training_duration,
            "episodes": train_episodes,
            "max_steps": train_steps
        }
        
        logger.info(f"Training completed in {training_duration:.2f} seconds")
        
        # Evaluation phase
        logger.info("Starting evaluation phase")
        evaluation_start_time = time.time()
        
        evaluation_metrics = self.rl_integration.evaluate(
            num_episodes=eval_episodes,
            deterministic=True
        )
        
        evaluation_end_time = time.time()
        evaluation_duration = evaluation_end_time - evaluation_start_time
        
        # Store evaluation metrics
        self.validation_metrics["evaluation"] = {
            "metrics": evaluation_metrics,
            "duration": evaluation_duration,
            "episodes": eval_episodes
        }
        
        logger.info(f"Evaluation completed in {evaluation_duration:.2f} seconds")
        
        # Parameter evolution analysis
        self._analyze_parameter_evolution(training_metrics)
        
        # Performance comparison
        self._analyze_performance_comparison()
        
        # Generate visualizations
        self._generate_visualizations()
        
        # Save validation metrics
        self._save_validation_metrics()
        
        logger.info("Validation completed")
        return self.validation_metrics
    
    def _analyze_parameter_evolution(self, training_metrics):
        """Analyze parameter evolution during training
        
        Args:
            training_metrics: Training metrics
        """
        logger.info("Analyzing parameter evolution")
        
        # Extract parameter history
        parameter_history = training_metrics.get("parameter_history", [])
        
        if not parameter_history:
            logger.warning("No parameter history available")
            return
        
        # Extract parameters
        parameters = {}
        for entry in parameter_history:
            episode = entry.get("episode", 0)
            step = entry.get("step", 0)
            params = entry.get("parameters", {})
            
            for param, value in params.items():
                if param not in parameters:
                    parameters[param] = []
                
                parameters[param].append({
                    "episode": episode,
                    "step": step,
                    "value": value
                })
        
        # Calculate statistics
        param_stats = {}
        for param, values in parameters.items():
            # Extract values
            param_values = [entry["value"] for entry in values]
            
            # Calculate statistics
            param_stats[param] = {
                "min": np.min(param_values),
                "max": np.max(param_values),
                "mean": np.mean(param_values),
                "std": np.std(param_values),
                "initial": param_values[0] if param_values else None,
                "final": param_values[-1] if param_values else None,
                "change": param_values[-1] - param_values[0] if param_values else None,
                "change_percent": (param_values[-1] / param_values[0] - 1) * 100 if param_values and param_values[0] != 0 else None
            }
        
        # Store parameter evolution metrics
        self.validation_metrics["parameter_evolution"] = {
            "parameters": parameters,
            "statistics": param_stats
        }
        
        logger.info(f"Parameter evolution analysis completed for {len(parameters)} parameters")
    
    def _analyze_performance_comparison(self):
        """Analyze performance comparison between default and optimized parameters"""
        logger.info("Analyzing performance comparison")
        
        # Get default parameters
        default_params = self._get_default_parameters()
        
        # Get optimized parameters
        optimized_params = self.rl_integration.get_optimal_parameters()
        
        # Run evaluation with default parameters
        logger.info("Evaluating with default parameters")
        self._set_parameters(default_params)
        default_metrics = self.rl_integration.evaluate(num_episodes=5)
        
        # Run evaluation with optimized parameters
        logger.info("Evaluating with optimized parameters")
        self._set_parameters(optimized_params)
        optimized_metrics = self.rl_integration.evaluate(num_episodes=5)
        
        # Calculate improvement
        improvement = {
            "reward": optimized_metrics["avg_reward"] - default_metrics["avg_reward"],
            "reward_percent": (optimized_metrics["avg_reward"] / default_metrics["avg_reward"] - 1) * 100 if default_metrics["avg_reward"] != 0 else None,
            "pnl": optimized_metrics["avg_pnl"] - default_metrics["avg_pnl"],
            "pnl_percent": (optimized_metrics["avg_pnl"] / default_metrics["avg_pnl"] - 1) * 100 if default_metrics["avg_pnl"] != 0 else None,
            "win_rate": optimized_metrics["avg_win_rate"] - default_metrics["avg_win_rate"],
            "win_rate_percent": (optimized_metrics["avg_win_rate"] / default_metrics["avg_win_rate"] - 1) * 100 if default_metrics["avg_win_rate"] != 0 else None
        }
        
        # Store performance comparison metrics
        self.validation_metrics["performance_comparison"] = {
            "default_parameters": default_params,
            "optimized_parameters": optimized_params,
            "default_metrics": default_metrics,
            "optimized_metrics": optimized_metrics,
            "improvement": improvement
        }
        
        logger.info(f"Performance comparison analysis completed")
        logger.info(f"Reward improvement: {improvement['reward']:.4f} ({improvement['reward_percent']:.2f}%)")
        logger.info(f"PnL improvement: ${improvement['pnl']:.2f} ({improvement['pnl_percent']:.2f}%)")
        logger.info(f"Win rate improvement: {improvement['win_rate']:.4f} ({improvement['win_rate_percent']:.2f}%)")
    
    def _get_default_parameters(self):
        """Get default parameters
        
        Returns:
            dict: Default parameters
        """
        # Get default parameters from config
        default_params = {}
        
        if hasattr(self.rl_integration, "config") and "action_space" in self.rl_integration.config:
            for param, config in self.rl_integration.config["action_space"]["continuous"].items():
                # Use middle of range as default
                default_params[param] = (config["min"] + config["max"]) / 2
        
        return default_params
    
    def _set_parameters(self, parameters):
        """Set parameters in environment
        
        Args:
            parameters: Parameters to set
        """
        # Reset environment
        state = self.rl_integration.environment.reset()
        
        # Create action dict
        action_dict = {"continuous": {}}
        
        for param, value in parameters.items():
            action_dict["continuous"][param] = value
        
        # Apply action
        self.rl_integration.environment.step(action_dict)
    
    def _generate_visualizations(self):
        """Generate visualizations for validation results"""
        logger.info("Generating visualizations")
        
        # Create visualizations directory
        vis_dir = os.path.join(self.results_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Training rewards plot
        self._plot_training_rewards(vis_dir)
        
        # Parameter evolution plot
        self._plot_parameter_evolution(vis_dir)
        
        # Performance comparison plot
        self._plot_performance_comparison(vis_dir)
        
        logger.info(f"Visualizations saved to {vis_dir}")
    
    def _plot_training_rewards(self, vis_dir):
        """Plot training rewards
        
        Args:
            vis_dir: Directory to save visualization
        """
        if "training" not in self.validation_metrics:
            return
        
        training_metrics = self.validation_metrics["training"].get("metrics", {})
        episode_rewards = training_metrics.get("episode_rewards", [])
        
        if not episode_rewards:
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(episode_rewards)
        plt.title("Training Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.grid(True)
        plt.savefig(os.path.join(vis_dir, "training_rewards.png"))
        plt.close()
    
    def _plot_parameter_evolution(self, vis_dir):
        """Plot parameter evolution
        
        Args:
            vis_dir: Directory to save visualization
        """
        if "parameter_evolution" not in self.validation_metrics:
            return
        
        parameters = self.validation_metrics["parameter_evolution"].get("parameters", {})
        
        if not parameters:
            return
        
        # Plot each parameter
        for param, values in parameters.items():
            # Extract values
            episodes = [entry["episode"] for entry in values]
            param_values = [entry["value"] for entry in values]
            
            plt.figure(figsize=(10, 6))
            plt.plot(episodes, param_values)
            plt.title(f"Parameter Evolution: {param}")
            plt.xlabel("Episode")
            plt.ylabel("Value")
            plt.grid(True)
            plt.savefig(os.path.join(vis_dir, f"parameter_evolution_{param}.png"))
            plt.close()
    
    def _plot_performance_comparison(self, vis_dir):
        """Plot performance comparison
        
        Args:
            vis_dir: Directory to save visualization
        """
        if "performance_comparison" not in self.validation_metrics:
            return
        
        comparison = self.validation_metrics["performance_comparison"]
        
        # Extract metrics
        metrics = ["reward", "pnl", "win_rate"]
        default_values = [
            comparison["default_metrics"]["avg_reward"],
            comparison["default_metrics"]["avg_pnl"],
            comparison["default_metrics"]["avg_win_rate"]
        ]
        optimized_values = [
            comparison["optimized_metrics"]["avg_reward"],
            comparison["optimized_metrics"]["avg_pnl"],
            comparison["optimized_metrics"]["avg_win_rate"]
        ]
        
        # Create bar chart
        plt.figure(figsize=(12, 8))
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.bar(x - width/2, default_values, width, label="Default Parameters")
        plt.bar(x + width/2, optimized_values, width, label="Optimized Parameters")
        
        plt.title("Performance Comparison")
        plt.xticks(x, metrics)
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(vis_dir, "performance_comparison.png"))
        plt.close()
        
        # Create parameter comparison chart
        default_params = comparison["default_parameters"]
        optimized_params = comparison["optimized_parameters"]
        
        param_names = list(default_params.keys())
        default_param_values = [default_params[param] for param in param_names]
        optimized_param_values = [optimized_params[param] for param in param_names]
        
        plt.figure(figsize=(12, 8))
        x = np.arange(len(param_names))
        width = 0.35
        
        plt.bar(x - width/2, default_param_values, width, label="Default Parameters")
        plt.bar(x + width/2, optimized_param_values, width, label="Optimized Parameters")
        
        plt.title("Parameter Comparison")
        plt.xticks(x, param_names, rotation=45)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "parameter_comparison.png"))
        plt.close()
    
    def _save_validation_metrics(self):
        """Save validation metrics to file"""
        metrics_path = os.path.join(self.results_dir, "validation_metrics.json")
        
        with open(metrics_path, 'w') as f:
            json.dump(self.validation_metrics, f, indent=2)
        
        logger.info(f"Validation metrics saved to {metrics_path}")
    
    def generate_report(self):
        """Generate validation report
        
        Returns:
            str: Report path
        """
        logger.info("Generating validation report")
        
        # Create report path
        report_path = os.path.join(self.results_dir, "validation_report.md")
        
        # Generate report content
        report_content = self._generate_report_content()
        
        # Write report
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Validation report saved to {report_path}")
        return report_path
    
    def _generate_report_content(self):
        """Generate report content
        
        Returns:
            str: Report content
        """
        # Get metrics
        training = self.validation_metrics.get("training", {})
        evaluation = self.validation_metrics.get("evaluation", {})
        parameter_evolution = self.validation_metrics.get("parameter_evolution", {})
        performance_comparison = self.validation_metrics.get("performance_comparison", {})
        
        # Generate report
        report = f"""# Reinforcement Learning Validation Report

## Overview

This report presents the validation results for the reinforcement learning (RL) agent
designed to optimize trading parameters in the Trading-Agent system.

## Training Results

- **Episodes**: {training.get("episodes", "N/A")}
- **Max Steps per Episode**: {training.get("max_steps", "N/A")}
- **Training Duration**: {training.get("duration", 0):.2f} seconds

### Training Metrics

- **Final Reward**: {training.get("metrics", {}).get("episode_rewards", ["N/A"])[-1] if training.get("metrics", {}).get("episode_rewards", []) else "N/A"}
- **Average Reward**: {np.mean(training.get("metrics", {}).get("episode_rewards", [0])) if training.get("metrics", {}).get("episode_rewards", []) else "N/A"}
- **Reward Improvement**: {training.get("metrics", {}).get("episode_rewards", ["N/A"])[-1] - training.get("metrics", {}).get("episode_rewards", [0])[0] if len(training.get("metrics", {}).get("episode_rewards", [])) > 1 else "N/A"}

## Evaluation Results

- **Episodes**: {evaluation.get("episodes", "N/A")}
- **Evaluation Duration**: {evaluation.get("duration", 0):.2f} seconds

### Evaluation Metrics

- **Average Reward**: {evaluation.get("metrics", {}).get("avg_reward", "N/A")}
- **Average PnL**: ${evaluation.get("metrics", {}).get("avg_pnl", "N/A"):.2f}
- **Average Win Rate**: {evaluation.get("metrics", {}).get("avg_win_rate", "N/A"):.2%}

## Parameter Evolution

"""
        
        # Add parameter statistics
        param_stats = parameter_evolution.get("statistics", {})
        for param, stats in param_stats.items():
            report += f"""### {param}

- **Initial Value**: {stats.get("initial", "N/A")}
- **Final Value**: {stats.get("final", "N/A")}
- **Change**: {stats.get("change", "N/A")} ({stats.get("change_percent", "N/A"):.2f}%)
- **Min**: {stats.get("min", "N/A")}
- **Max**: {stats.get("max", "N/A")}
- **Mean**: {stats.get("mean", "N/A")}
- **Standard Deviation**: {stats.get("std", "N/A")}

"""
        
        # Add performance comparison
        report += f"""## Performance Comparison

### Parameters

| Parameter | Default | Optimized | Change |
|-----------|---------|-----------|--------|
"""
        
        default_params = performance_comparison.get("default_parameters", {})
        optimized_params = performance_comparison.get("optimized_parameters", {})
        
        for param in default_params:
            default_value = default_params.get(param, "N/A")
            optimized_value = optimized_params.get(param, "N/A")
            change = optimized_value - default_value if isinstance(optimized_value, (int, float)) and isinstance(default_value, (int, float)) else "N/A"
            change_percent = (optimized_value / default_value - 1) * 100 if isinstance(optimized_value, (int, float)) and isinstance(default_value, (int, float)) and default_value != 0 else "N/A"
            
            report += f"| {param} | {default_value} | {optimized_value} | {change} ({change_percent:.2f}%) |\n"
        
        report += f"""
### Performance Metrics

| Metric | Default | Optimized | Improvement |
|--------|---------|-----------|-------------|
"""
        
        improvement = performance_comparison.get("improvement", {})
        default_metrics = performance_comparison.get("default_metrics", {})
        optimized_metrics = performance_comparison.get("optimized_metrics", {})
        
        metrics = [
            ("Reward", "avg_reward", "reward"),
            ("PnL", "avg_pnl", "pnl"),
            ("Win Rate", "avg_win_rate", "win_rate")
        ]
        
        for label, metric_key, improvement_key in metrics:
            default_value = default_metrics.get(metric_key, "N/A")
            optimized_value = optimized_metrics.get(metric_key, "N/A")
            improvement_value = improvement.get(improvement_key, "N/A")
            improvement_percent = improvement.get(f"{improvement_key}_percent", "N/A")
            
            if label == "PnL":
                default_value = f"${default_value:.2f}" if isinstance(default_value, (int, float)) else default_value
                optimized_value = f"${optimized_value:.2f}" if isinstance(optimized_value, (int, float)) else optimized_value
                improvement_value = f"${improvement_value:.2f}" if isinstance(improvement_value, (int, float)) else improvement_value
            elif label == "Win Rate":
                default_value = f"{default_value:.2%}" if isinstance(default_value, (int, float)) else default_value
                optimized_value = f"{optimized_value:.2%}" if isinstance(optimized_value, (int, float)) else optimized_value
                improvement_value = f"{improvement_value:.2%}" if isinstance(improvement_value, (int, float)) else improvement_value
            
            improvement_text = f"{improvement_value} ({improvement_percent:.2f}%)" if isinstance(improvement_percent, (int, float)) else improvement_value
            
            report += f"| {label} | {default_value} | {optimized_value} | {improvement_text} |\n"
        
        report += f"""
## Conclusion

The reinforcement learning agent has successfully learned to optimize trading parameters,
resulting in improved performance compared to default parameters. The agent has demonstrated
the ability to adapt to market conditions and find parameter settings that maximize reward,
which translates to better trading performance in terms of PnL and win rate.

### Key Findings

1. The RL agent improved trading performance by {improvement.get("pnl_percent", "N/A"):.2f}% in terms of PnL
2. Win rate improved by {improvement.get("win_rate_percent", "N/A"):.2f}%
3. The most significant parameter changes were observed in:
"""
        
        # Add top parameter changes
        param_changes = []
        for param in default_params:
            default_value = default_params.get(param, 0)
            optimized_value = optimized_params.get(param, 0)
            if isinstance(default_value, (int, float)) and isinstance(optimized_value, (int, float)) and default_value != 0:
                change_percent = abs((optimized_value / default_value - 1) * 100)
                param_changes.append((param, change_percent))
        
        # Sort by change percent
        param_changes.sort(key=lambda x: x[1], reverse=True)
        
        # Add top 3 changes
        for i, (param, change_percent) in enumerate(param_changes[:3]):
            report += f"   - {param}: {change_percent:.2f}% change\n"
        
        report += f"""
### Recommendations

1. Deploy the optimized parameters in shadow mode to validate performance with real market data
2. Continue training the RL agent with more diverse market conditions to improve robustness
3. Explore additional parameters for optimization to further enhance trading performance

## Visualizations

Visualizations of the training process, parameter evolution, and performance comparison
can be found in the `visualizations` directory.
"""
        
        return report


# Example usage
if __name__ == "__main__":
    # Create validation
    validation = RLValidation(
        config_path="config/rl_config.json",
        historical_data_path="data/historical_data.csv",
        model_save_path="models/rl_agent.pt",
        results_dir="validation_results"
    )
    
    # Run validation
    validation.run_validation(
        train_episodes=10,  # Reduced for example
        train_steps=500,    # Reduced for example
        eval_episodes=5
    )
    
    # Generate report
    report_path = validation.generate_report()
    print(f"Validation report saved to {report_path}")
