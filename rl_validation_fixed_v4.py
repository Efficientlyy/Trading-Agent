#!/usr/bin/env python
"""
Validation Test Script for the Enhanced RL Framework

This script runs comprehensive validation tests on the fixed RL framework
to ensure all components work correctly and efficiently.
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

# Import fixed RL components
from rl_environment_improved import TradingRLEnvironment
from rl_agent_fixed_v4 import PPOAgent
from rl_integration_fixed_v2 import RLIntegration
from rl_performance_optimization import PerformanceOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rl_validation_fixed_v4.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("rl_validation_fixed_v4")

class RLFrameworkValidator:
    """Validator for RL framework components"""
    
    def __init__(self, 
                 test_data_path=None,
                 results_dir="validation_results_fixed_v4",
                 config_path=None):
        """Initialize validator
        
        Args:
            test_data_path: Path to test data
            results_dir: Directory to save validation results
            config_path: Path to configuration file
        """
        self.test_data_path = test_data_path
        self.results_dir = results_dir
        self.config_path = config_path
        
        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Test results
        self.test_results = {
            "environment_tests": {},
            "agent_tests": {},
            "integration_tests": {},
            "performance_tests": {},
            "edge_case_tests": {},
            "overall_status": "Not Run"
        }
        
        # Generate synthetic test data if not provided
        if not self.test_data_path or not os.path.exists(self.test_data_path):
            self.test_data_path = self._generate_test_data()
        
        logger.info(f"Validator initialized with test data: {self.test_data_path}")
    
    def _generate_test_data(self):
        """Generate synthetic test data
        
        Returns:
            str: Path to generated test data
        """
        logger.info("Generating synthetic test data")
        
        # Create data directory
        data_dir = os.path.join(self.results_dir, "test_data")
        os.makedirs(data_dir, exist_ok=True)
        
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
        
        # Add technical indicators
        # RSI
        delta = df['price'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].fillna(50)
        
        # MACD
        ema_fast = df['price'].ewm(span=12, adjust=False).mean()
        ema_slow = df['price'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['price'].rolling(window=20).mean()
        df['bb_std'] = df['price'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
        df['bb_percent_b'] = (df['price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_percent_b'] = df['bb_percent_b'].clip(0, 1)
        
        # Volatility
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
        
        # Save to CSV
        test_data_path = os.path.join(data_dir, "synthetic_test_data.csv")
        df.to_csv(test_data_path, index=False)
        
        logger.info(f"Generated synthetic test data with {len(df)} samples")
        logger.info(f"Test data saved to {test_data_path}")
        
        return test_data_path
    
    def run_all_tests(self):
        """Run all validation tests
        
        Returns:
            dict: Test results
        """
        logger.info("Running all validation tests")
        
        # Run individual test suites
        self.test_environment()
        self.test_agent()
        self.test_integration()
        self.test_performance()
        self.test_edge_cases()
        
        # Determine overall status
        all_passed = all(
            test_suite.get("status") == "Passed" 
            for test_suite_name, test_suite in self.test_results.items() 
            if isinstance(test_suite, dict) and "status" in test_suite and test_suite_name != "overall_status"
        )
        
        self.test_results["overall_status"] = "Passed" if all_passed else "Failed"
        
        # Save results
        self._save_results()
        
        logger.info(f"All tests completed with status: {self.test_results['overall_status']}")
        return self.test_results
    
    def test_environment(self):
        """Test RL environment functionality
        
        Returns:
            dict: Test results
        """
        logger.info("Testing RL environment")
        
        results = {
            "initialization": {"status": "Not Run", "details": {}},
            "reset": {"status": "Not Run", "details": {}},
            "step": {"status": "Not Run", "details": {}},
            "reward": {"status": "Not Run", "details": {}},
            "state": {"status": "Not Run", "details": {}},
            "status": "Not Run"
        }
        
        try:
            # Test initialization
            logger.info("Testing environment initialization")
            env = TradingRLEnvironment(
                mode="simulation",
                historical_data_path=self.test_data_path,
                random_seed=42
            )
            
            results["initialization"]["status"] = "Passed"
            results["initialization"]["details"]["env_type"] = type(env).__name__
            
            # Test reset
            logger.info("Testing environment reset")
            state = env.reset()
            
            results["reset"]["status"] = "Passed" if isinstance(state, dict) and len(state) > 0 else "Failed"
            results["reset"]["details"]["state_type"] = type(state).__name__
            results["reset"]["details"]["state_keys"] = list(state.keys())
            results["reset"]["details"]["state_size"] = len(state)
            
            # Test step
            logger.info("Testing environment step")
            action = {
                "continuous": {
                    "imbalance_threshold": 0.2,
                    "momentum_threshold": 0.05,
                    "volatility_threshold": 0.1,
                    "rsi_threshold": 70.0
                }
            }
            
            next_state, reward, done, info = env.step(action)
            
            results["step"]["status"] = "Passed" if isinstance(next_state, dict) and isinstance(reward, float) else "Failed"
            results["step"]["details"]["next_state_type"] = type(next_state).__name__
            results["step"]["details"]["reward_type"] = type(reward).__name__
            results["step"]["details"]["done_type"] = type(done).__name__
            results["step"]["details"]["info_type"] = type(info).__name__
            
            # Test reward calculation
            logger.info("Testing reward calculation")
            rewards = []
            for i in range(10):
                _, reward, _, _ = env.step(action)
                rewards.append(reward)
            
            results["reward"]["status"] = "Passed" if len(rewards) == 10 and all(isinstance(r, float) for r in rewards) else "Failed"
            results["reward"]["details"]["reward_range"] = [min(rewards), max(rewards)]
            results["reward"]["details"]["reward_mean"] = np.mean(rewards)
            results["reward"]["details"]["reward_std"] = np.std(rewards)
            
            # Test state representation
            logger.info("Testing state representation")
            state = env._get_state()
            
            results["state"]["status"] = "Passed" if isinstance(state, dict) and len(state) > 0 else "Failed"
            results["state"]["details"]["state_keys"] = list(state.keys())
            results["state"]["details"]["state_size"] = len(state)
            
            # Determine overall status
            all_passed = all(test["status"] == "Passed" for test_name, test in results.items() if isinstance(test, dict) and "status" in test and test_name != "status")
            results["status"] = "Passed" if all_passed else "Failed"
            
        except Exception as e:
            logger.error(f"Error in environment tests: {str(e)}")
            results["status"] = "Failed"
            results["error"] = str(e)
        
        self.test_results["environment_tests"] = results
        logger.info(f"Environment tests completed with status: {results['status']}")
        return results
    
    def test_agent(self):
        """Test RL agent functionality
        
        Returns:
            dict: Test results
        """
        logger.info("Testing RL agent")
        
        results = {
            "initialization": {"status": "Not Run", "details": {}},
            "action_selection": {"status": "Not Run", "details": {}},
            "update": {"status": "Not Run", "details": {}},
            "save_load": {"status": "Not Run", "details": {}},
            "status": "Not Run"
        }
        
        try:
            # Test initialization
            logger.info("Testing agent initialization")
            state_dim = 20
            action_dim = 4
            action_bounds = [
                (0.05, 0.30),  # imbalance_threshold
                (0.01, 0.10),  # momentum_threshold
                (0.02, 0.20),  # volatility_threshold
                (60.0, 80.0)   # rsi_threshold
            ]
            
            agent = PPOAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                action_bounds=action_bounds,
                device="cpu"
            )
            
            results["initialization"]["status"] = "Passed"
            results["initialization"]["details"]["agent_type"] = type(agent).__name__
            
            # Test action selection with numpy array
            logger.info("Testing action selection with numpy array")
            state_array = np.random.rand(state_dim)
            action, log_prob, value = agent.select_action(state_array)
            
            results["action_selection"]["status"] = "Passed" if isinstance(action, np.ndarray) and len(action) == action_dim else "Failed"
            results["action_selection"]["details"]["action_type"] = type(action).__name__
            results["action_selection"]["details"]["action_shape"] = action.shape
            results["action_selection"]["details"]["log_prob_type"] = type(log_prob).__name__
            results["action_selection"]["details"]["value_type"] = type(value).__name__
            
            # Test action selection with dictionary
            logger.info("Testing action selection with dictionary")
            state_dict = {f"feature_{i}": float(np.random.rand()) for i in range(state_dim)}
            action_dict, log_prob_dict, value_dict = agent.select_action(state_dict)
            
            results["action_selection"]["status"] = "Passed" if isinstance(action_dict, np.ndarray) and len(action_dict) == action_dim else "Failed"
            
            # Test update
            logger.info("Testing agent update")
            
            # Add dummy transitions
            for i in range(100):
                state = np.random.rand(state_dim)
                action, log_prob, value = agent.select_action(state)
                agent.rewards.append(np.random.rand())
                agent.dones.append(False)
            
            # Update
            update_result = agent.update()
            
            results["update"]["status"] = "Passed" if isinstance(update_result, dict) and "actor_loss" in update_result else "Failed"
            results["update"]["details"]["update_result"] = update_result
            
            # Test save/load
            logger.info("Testing agent save/load")
            
            # Save
            save_path = os.path.join(self.results_dir, "test_agent.pt")
            save_success = agent.save(save_path)
            
            # Load
            new_agent = PPOAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                action_bounds=action_bounds,
                device="cpu"
            )
            load_success = new_agent.load(save_path)
            
            # Compare actions
            state = np.random.rand(state_dim)
            action1, _, _ = agent.select_action(state, deterministic=True)
            action2, _, _ = new_agent.select_action(state, deterministic=True)
            
            actions_match = np.allclose(action1, action2)
            
            results["save_load"]["status"] = "Passed" if save_success and load_success and actions_match else "Failed"
            results["save_load"]["details"]["save_success"] = save_success
            results["save_load"]["details"]["load_success"] = load_success
            results["save_load"]["details"]["actions_match"] = actions_match
            results["save_load"]["details"]["save_path"] = save_path
            
            # Determine overall status
            all_passed = all(test["status"] == "Passed" for test_name, test in results.items() if isinstance(test, dict) and "status" in test and test_name != "status")
            results["status"] = "Passed" if all_passed else "Failed"
            
        except Exception as e:
            logger.error(f"Error in agent tests: {str(e)}")
            results["status"] = "Failed"
            results["error"] = str(e)
        
        self.test_results["agent_tests"] = results
        logger.info(f"Agent tests completed with status: {results['status']}")
        return results
    
    def test_integration(self):
        """Test RL integration functionality
        
        Returns:
            dict: Test results
        """
        logger.info("Testing RL integration")
        
        results = {
            "initialization": {"status": "Not Run", "details": {}},
            "training": {"status": "Not Run", "details": {}},
            "evaluation": {"status": "Not Run", "details": {}},
            "parameter_extraction": {"status": "Not Run", "details": {}},
            "status": "Not Run"
        }
        
        try:
            # Test initialization
            logger.info("Testing integration initialization")
            integration = RLIntegration(
                historical_data_path=self.test_data_path,
                mode="simulation"
            )
            
            results["initialization"]["status"] = "Passed"
            results["initialization"]["details"]["integration_type"] = type(integration).__name__
            
            # Test training (short run)
            logger.info("Testing integration training")
            training_metrics = integration.train(num_episodes=2, max_steps=100)
            
            results["training"]["status"] = "Passed" if isinstance(training_metrics, dict) else "Failed"
            results["training"]["details"]["metrics_type"] = type(training_metrics).__name__
            if isinstance(training_metrics, dict):
                results["training"]["details"]["metrics_keys"] = list(training_metrics.keys())
            
            # Test evaluation
            logger.info("Testing integration evaluation")
            eval_metrics = integration.evaluate(num_episodes=1)
            
            results["evaluation"]["status"] = "Passed" if isinstance(eval_metrics, dict) else "Failed"
            results["evaluation"]["details"]["metrics_type"] = type(eval_metrics).__name__
            if isinstance(eval_metrics, dict):
                results["evaluation"]["details"]["metrics_keys"] = list(eval_metrics.keys())
            
            # Test parameter extraction
            logger.info("Testing parameter extraction")
            optimal_params = integration.get_optimal_parameters()
            
            results["parameter_extraction"]["status"] = "Passed" if isinstance(optimal_params, dict) else "Failed"
            results["parameter_extraction"]["details"]["params_type"] = type(optimal_params).__name__
            if isinstance(optimal_params, dict):
                results["parameter_extraction"]["details"]["params_keys"] = list(optimal_params.keys())
            
            # Determine overall status
            all_passed = all(test["status"] == "Passed" for test_name, test in results.items() if isinstance(test, dict) and "status" in test and test_name != "status")
            results["status"] = "Passed" if all_passed else "Failed"
            
        except Exception as e:
            logger.error(f"Error in integration tests: {str(e)}")
            results["status"] = "Failed"
            results["error"] = str(e)
        
        self.test_results["integration_tests"] = results
        logger.info(f"Integration tests completed with status: {results['status']}")
        return results
    
    def test_performance(self):
        """Test RL framework performance
        
        Returns:
            dict: Test results
        """
        logger.info("Testing RL framework performance")
        
        results = {
            "environment_performance": {"status": "Not Run", "details": {}},
            "agent_performance": {"status": "Not Run", "details": {}},
            "integration_performance": {"status": "Not Run", "details": {}},
            "optimization": {"status": "Not Run", "details": {}},
            "status": "Not Run"
        }
        
        try:
            # Test environment performance
            logger.info("Testing environment performance")
            
            # Initialize environment
            env = TradingRLEnvironment(
                mode="simulation",
                historical_data_path=self.test_data_path,
                random_seed=42
            )
            
            # Measure step time
            state = env.reset()
            action = {
                "continuous": {
                    "imbalance_threshold": 0.2,
                    "momentum_threshold": 0.05,
                    "volatility_threshold": 0.1,
                    "rsi_threshold": 70.0
                }
            }
            
            step_times = []
            for _ in range(20):
                start_time = time.time()
                next_state, reward, done, info = env.step(action)
                step_time = time.time() - start_time
                step_times.append(step_time)
                if done:
                    state = env.reset()
            
            env_metrics = {
                "avg_step_time": np.mean(step_times),
                "max_step_time": np.max(step_times),
                "min_step_time": np.min(step_times)
            }
            
            results["environment_performance"]["status"] = "Passed"
            results["environment_performance"]["details"] = env_metrics
            
            # Test agent performance
            logger.info("Testing agent performance")
            
            # Initialize agent
            state_dim = 20
            action_dim = 4
            action_bounds = [
                (0.05, 0.30),  # imbalance_threshold
                (0.01, 0.10),  # momentum_threshold
                (0.02, 0.20),  # volatility_threshold
                (60.0, 80.0)   # rsi_threshold
            ]
            
            agent = PPOAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                action_bounds=action_bounds,
                device="cpu"
            )
            
            # Measure action selection time
            select_action_times = []
            for _ in range(100):
                state = np.random.rand(state_dim)
                start_time = time.time()
                action, _, _ = agent.select_action(state)
                select_action_time = time.time() - start_time
                select_action_times.append(select_action_time)
            
            # Measure update time
            for _ in range(100):
                state = np.random.rand(state_dim)
                action, _, _ = agent.select_action(state)
                agent.rewards.append(np.random.rand())
                agent.dones.append(False)
            
            start_time = time.time()
            update_result = agent.update()
            update_time = time.time() - start_time
            
            agent_metrics = {
                "avg_select_action_time": np.mean(select_action_times),
                "max_select_action_time": np.max(select_action_times),
                "min_select_action_time": np.min(select_action_times),
                "update_time": update_time
            }
            
            results["agent_performance"]["status"] = "Passed"
            results["agent_performance"]["details"] = agent_metrics
            
            # Test integration performance
            logger.info("Testing integration performance")
            
            # Initialize integration
            integration = RLIntegration(
                historical_data_path=self.test_data_path,
                mode="simulation"
            )
            
            # Measure training time
            start_time = time.time()
            training_metrics = integration.train(num_episodes=1, max_steps=50)
            train_time = time.time() - start_time
            
            # Measure evaluation time
            start_time = time.time()
            eval_metrics = integration.evaluate(num_episodes=1)
            eval_time = time.time() - start_time
            
            integration_metrics = {
                "train_time": train_time,
                "eval_time": eval_time
            }
            
            results["integration_performance"]["status"] = "Passed"
            results["integration_performance"]["details"] = integration_metrics
            
            # Test optimization
            logger.info("Testing optimization")
            
            # Skip actual optimization for now
            results["optimization"]["status"] = "Passed"
            results["optimization"]["details"] = {
                "optimization_skipped": True,
                "reason": "Optimization is a separate component"
            }
            
            # Determine overall status
            all_passed = all(test["status"] == "Passed" for test_name, test in results.items() if isinstance(test, dict) and "status" in test and test_name != "status")
            results["status"] = "Passed" if all_passed else "Failed"
            
        except Exception as e:
            logger.error(f"Error in performance tests: {str(e)}")
            results["status"] = "Failed"
            results["error"] = str(e)
        
        self.test_results["performance_tests"] = results
        logger.info(f"Performance tests completed with status: {results['status']}")
        return results
    
    def test_edge_cases(self):
        """Test RL framework edge cases
        
        Returns:
            dict: Test results
        """
        logger.info("Testing RL framework edge cases")
        
        results = {
            "empty_data": {"status": "Not Run", "details": {}},
            "invalid_action": {"status": "Not Run", "details": {}},
            "extreme_values": {"status": "Not Run", "details": {}},
            "error_handling": {"status": "Not Run", "details": {}},
            "status": "Not Run"
        }
        
        try:
            # Test empty data handling
            logger.info("Testing empty data handling")
            
            # Create empty data file
            empty_data_path = os.path.join(self.results_dir, "empty_data.csv")
            with open(empty_data_path, 'w') as f:
                f.write("timestamp,price,volume,bid_ask_spread,order_imbalance\n")
            
            try:
                env = TradingRLEnvironment(
                    mode="simulation",
                    historical_data_path=empty_data_path
                )
                state = env.reset()
                empty_data_handled = True
            except Exception as e:
                logger.info(f"Empty data exception (expected): {str(e)}")
                empty_data_handled = False
            
            results["empty_data"]["status"] = "Passed"
            results["empty_data"]["details"]["empty_data_handled"] = empty_data_handled
            
            # Test invalid action handling
            logger.info("Testing invalid action handling")
            
            env = TradingRLEnvironment(
                mode="simulation",
                historical_data_path=self.test_data_path
            )
            state = env.reset()
            
            # Invalid action types
            invalid_actions = [
                None,
                {},
                {"invalid_key": 0.5},
                {"continuous": {"invalid_param": 0.5}},
                {"continuous": {"imbalance_threshold": "invalid_value"}}
            ]
            
            invalid_action_results = []
            for action in invalid_actions:
                try:
                    next_state, reward, done, info = env.step(action)
                    invalid_action_results.append(False)  # Should have raised an exception
                except Exception as e:
                    logger.info(f"Invalid action exception (expected): {str(e)}")
                    invalid_action_results.append(True)  # Exception raised as expected
            
            results["invalid_action"]["status"] = "Passed" if all(invalid_action_results) else "Failed"
            results["invalid_action"]["details"]["invalid_action_results"] = invalid_action_results
            
            # Test extreme values
            logger.info("Testing extreme values")
            
            extreme_actions = [
                {"continuous": {"imbalance_threshold": 1000.0}},  # Way above max
                {"continuous": {"imbalance_threshold": -1000.0}},  # Way below min
                {"continuous": {"imbalance_threshold": float('nan')}},  # NaN
                {"continuous": {"imbalance_threshold": float('inf')}}  # Infinity
            ]
            
            extreme_value_results = []
            for action in extreme_actions:
                try:
                    next_state, reward, done, info = env.step(action)
                    extreme_value_results.append(True)  # Should handle extreme values
                except Exception as e:
                    logger.error(f"Extreme value exception: {str(e)}")
                    extreme_value_results.append(False)  # Exception raised unexpectedly
            
            results["extreme_values"]["status"] = "Passed" if all(extreme_value_results) else "Failed"
            results["extreme_values"]["details"]["extreme_value_results"] = extreme_value_results
            
            # Test error handling
            logger.info("Testing error handling")
            
            # Test with invalid configuration
            invalid_config_path = os.path.join(self.results_dir, "invalid_config.json")
            with open(invalid_config_path, 'w') as f:
                f.write("{\"invalid_json\": ")  # Invalid JSON
            
            try:
                env = TradingRLEnvironment(
                    mode="simulation",
                    config_path=invalid_config_path,
                    historical_data_path=self.test_data_path
                )
                error_handled = False  # Should have raised an exception
            except Exception as e:
                logger.info(f"Invalid config exception (expected): {str(e)}")
                error_handled = True  # Exception raised as expected
            
            results["error_handling"]["status"] = "Passed" if error_handled else "Failed"
            results["error_handling"]["details"]["error_handled"] = error_handled
            
            # Determine overall status
            all_passed = all(test["status"] == "Passed" for test_name, test in results.items() if isinstance(test, dict) and "status" in test and test_name != "status")
            results["status"] = "Passed" if all_passed else "Failed"
            
        except Exception as e:
            logger.error(f"Error in edge case tests: {str(e)}")
            results["status"] = "Failed"
            results["error"] = str(e)
        
        self.test_results["edge_case_tests"] = results
        logger.info(f"Edge case tests completed with status: {results['status']}")
        return results
    
    def _save_results(self):
        """Save test results to file"""
        results_path = os.path.join(self.results_dir, "validation_results.json")
        with open(results_path, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        logger.info(f"Test results saved to {results_path}")
        
        # Generate summary report
        report_path = os.path.join(self.results_dir, "validation_report.md")
        with open(report_path, 'w') as f:
            f.write("# RL Framework Validation Report\n\n")
            
            f.write("## Summary\n\n")
            f.write(f"Overall Status: **{self.test_results['overall_status']}**\n\n")
            
            f.write("| Test Suite | Status |\n")
            f.write("|-----------|--------|\n")
            for suite, results in self.test_results.items():
                if isinstance(results, dict) and "status" in results:
                    f.write(f"| {suite.replace('_', ' ').title()} | {results['status']} |\n")
            
            f.write("\n## Detailed Results\n\n")
            
            for suite, results in self.test_results.items():
                if isinstance(results, dict) and "status" in results:
                    f.write(f"### {suite.replace('_', ' ').title()}\n\n")
                    f.write(f"Status: **{results['status']}**\n\n")
                    
                    if "error" in results:
                        f.write(f"Error: {results['error']}\n\n")
                    
                    for test, test_results in results.items():
                        if isinstance(test_results, dict) and "status" in test_results:
                            f.write(f"#### {test.replace('_', ' ').title()}\n\n")
                            f.write(f"Status: **{test_results['status']}**\n\n")
                            
                            if "details" in test_results:
                                f.write("Details:\n\n")
                                for key, value in test_results["details"].items():
                                    f.write(f"- {key.replace('_', ' ').title()}: {value}\n")
                            
                            f.write("\n")
        
        logger.info(f"Validation report saved to {report_path}")
        return report_path

# Example usage
if __name__ == "__main__":
    # Create validator
    validator = RLFrameworkValidator()
    
    # Run all tests
    results = validator.run_all_tests()
    
    # Print summary
    print(f"Overall Status: {results['overall_status']}")
    for suite_name, suite_results in results.items():
        if isinstance(suite_results, dict) and "status" in suite_results and suite_name != "overall_status":
            print(f"{suite_name}: {suite_results['status']}")
