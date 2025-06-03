#!/usr/bin/env python
"""
Test module for capital allocation strategy.

This module tests the capital allocation strategy functionality,
ensuring that the configurable percentage allocation works correctly.
"""

import os
import sys
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import components
from strategy.compounding import CapitalAllocationStrategy

class TestCapitalAllocation(unittest.TestCase):
    """Test capital allocation strategy functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a mock config class that returns real values
        self.mock_config = MagicMock()
        def mock_config_get(key, default=None):
            if key == "trading.allocation.enabled":
                return True
            elif key == "trading.allocation.percentage":
                return 0.8  # 80% by default
            elif key == "trading.allocation.min_reserve":
                return 100  # $100 minimum reserve
            return default
        self.mock_config.get.side_effect = mock_config_get
        
        # Initialize capital allocation strategy
        self.strategy = CapitalAllocationStrategy(self.mock_config)
        
        # Set up test data directory
        self.test_data_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "test_data"
        )
        os.makedirs(self.test_data_dir, exist_ok=True)
        
        # Override data file path
        self.strategy.data_file = os.path.join(
            self.test_data_dir,
            "test_allocation_history.json"
        )
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove test data file if it exists
        if os.path.exists(self.strategy.data_file):
            os.remove(self.strategy.data_file)
    
    def test_initialization(self):
        """Test initialization of capital allocation strategy."""
        self.assertTrue(self.strategy.enabled)
        self.assertEqual(self.strategy.allocation_percentage, 0.8)
        self.assertEqual(self.strategy.min_reserve, 100)
        self.assertEqual(self.strategy.total_capital, 0.0)
        self.assertEqual(self.strategy.allocated_capital, 0.0)
        self.assertEqual(self.strategy.reserve_capital, 0.0)
        self.assertIsNone(self.strategy.last_allocation_date)
    
    def test_update_capital(self):
        """Test updating capital."""
        self.strategy.update_capital(1000.0)
        self.assertEqual(self.strategy.total_capital, 1000.0)
    
    def test_calculate_allocation(self):
        """Test calculating capital allocation."""
        # Test with 1000 USDC
        result = self.strategy.calculate_allocation(1000.0)
        self.assertEqual(result["total_capital"], 1000.0)
        self.assertEqual(result["allocation_amount"], 800.0)  # 80% of 1000
        self.assertEqual(result["reserve_amount"], 200.0)  # 20% of 1000
        self.assertEqual(result["target_percentage"], 0.8)
        self.assertEqual(result["actual_percentage"], 0.8)
        
        # Test with minimum reserve enforcement
        result = self.strategy.calculate_allocation(100.0)
        self.assertEqual(result["total_capital"], 100.0)
        self.assertEqual(result["allocation_amount"], 0.0)  # All goes to reserve
        self.assertEqual(result["reserve_amount"], 100.0)  # Minimum reserve
        self.assertEqual(result["target_percentage"], 0.8)
        self.assertEqual(result["actual_percentage"], 0.0)
        
        # Test with small amount
        result = self.strategy.calculate_allocation(120.0)
        self.assertEqual(result["total_capital"], 120.0)
        self.assertEqual(result["allocation_amount"], 20.0)  # 120 - 100 (min reserve)
        self.assertEqual(result["reserve_amount"], 100.0)  # Minimum reserve
        self.assertEqual(result["target_percentage"], 0.8)
        self.assertAlmostEqual(result["actual_percentage"], 20.0/120.0)
    
    def test_execute_allocation(self):
        """Test executing capital allocation."""
        result = self.strategy.execute_allocation(1000.0)
        self.assertTrue(result["allocated"])
        self.assertEqual(result["total_capital"], 1000.0)
        self.assertEqual(result["allocated_amount"], 800.0)
        self.assertEqual(result["reserve_amount"], 200.0)
        self.assertEqual(result["allocation_percentage"], 0.8)
        
        # Check that tracking variables were updated
        self.assertEqual(self.strategy.total_capital, 1000.0)
        self.assertEqual(self.strategy.allocated_capital, 800.0)
        self.assertEqual(self.strategy.reserve_capital, 200.0)
        self.assertIsNotNone(self.strategy.last_allocation_date)
        
        # Check that history was updated
        self.assertEqual(len(self.strategy.history), 1)
        self.assertEqual(self.strategy.history[0]["total_capital"], 1000.0)
        self.assertEqual(self.strategy.history[0]["allocated"], 800.0)
        self.assertEqual(self.strategy.history[0]["reserve"], 200.0)
    
    def test_get_trading_capital(self):
        """Test getting trading capital."""
        # Test with 1000 USDC
        trading_capital = self.strategy.get_trading_capital(1000.0)
        self.assertEqual(trading_capital, 800.0)  # 80% of 1000
        
        # Test with minimum reserve enforcement
        trading_capital = self.strategy.get_trading_capital(100.0)
        self.assertEqual(trading_capital, 0.0)  # All goes to reserve
        
        # Test with small amount
        trading_capital = self.strategy.get_trading_capital(120.0)
        self.assertEqual(trading_capital, 20.0)  # 120 - 100 (min reserve)
    
    def test_set_allocation_percentage(self):
        """Test setting allocation percentage."""
        # Test valid percentage
        result = self.strategy.set_allocation_percentage(0.7)
        self.assertTrue(result["success"])
        self.assertEqual(result["old_value"], 0.8)
        self.assertEqual(result["new_value"], 0.7)
        self.assertEqual(self.strategy.allocation_percentage, 0.7)
        
        # Test invalid percentage (too low)
        result = self.strategy.set_allocation_percentage(-0.1)
        self.assertFalse(result["success"])
        self.assertEqual(self.strategy.allocation_percentage, 0.7)  # Unchanged
        
        # Test invalid percentage (too high)
        result = self.strategy.set_allocation_percentage(1.1)
        self.assertFalse(result["success"])
        self.assertEqual(self.strategy.allocation_percentage, 0.7)  # Unchanged
    
    def test_set_min_reserve(self):
        """Test setting minimum reserve."""
        # Test valid minimum reserve
        result = self.strategy.set_min_reserve(200.0)
        self.assertTrue(result["success"])
        self.assertEqual(result["old_value"], 100.0)
        self.assertEqual(result["new_value"], 200.0)
        self.assertEqual(self.strategy.min_reserve, 200.0)
        
        # Test invalid minimum reserve (negative)
        result = self.strategy.set_min_reserve(-50.0)
        self.assertFalse(result["success"])
        self.assertEqual(self.strategy.min_reserve, 200.0)  # Unchanged
    
    def test_enable_disable(self):
        """Test enabling and disabling capital allocation strategy."""
        # Test disabling
        result = self.strategy.disable()
        self.assertTrue(result["success"])
        self.assertEqual(result["old_value"], True)
        self.assertEqual(result["new_value"], False)
        self.assertFalse(self.strategy.enabled)
        
        # Test enabling
        result = self.strategy.enable()
        self.assertTrue(result["success"])
        self.assertEqual(result["old_value"], False)
        self.assertEqual(result["new_value"], True)
        self.assertTrue(self.strategy.enabled)
    
    def test_get_statistics(self):
        """Test getting allocation statistics."""
        # Initialize with some data
        self.strategy.update_capital(1000.0)
        self.strategy.execute_allocation(1000.0)
        
        # Get statistics
        stats = self.strategy.get_statistics()
        self.assertTrue(stats["enabled"])
        self.assertEqual(stats["allocation_percentage"], 0.8)
        self.assertEqual(stats["min_reserve"], 100.0)
        self.assertEqual(stats["total_capital"], 1000.0)
        self.assertEqual(stats["allocated_capital"], 800.0)
        self.assertEqual(stats["reserve_capital"], 200.0)
        self.assertIsNotNone(stats["last_allocation_date"])
        self.assertEqual(stats["allocation_events"], 1)
    
    def test_backward_compatibility(self):
        """Test backward compatibility with CompoundingStrategy name."""
        from strategy.compounding import CompoundingStrategy
        
        # Create a CompoundingStrategy instance
        strategy = CompoundingStrategy(self.mock_config)
        
        # Verify it's actually a CapitalAllocationStrategy
        self.assertIsInstance(strategy, CapitalAllocationStrategy)
        
        # Test basic functionality
        result = strategy.calculate_allocation(1000.0)
        self.assertEqual(result["allocation_amount"], 800.0)

if __name__ == "__main__":
    unittest.main()
