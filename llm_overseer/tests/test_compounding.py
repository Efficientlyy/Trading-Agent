#!/usr/bin/env python
"""
Test module for compounding strategy logic.

This module tests the compounding strategy implementation,
focusing on profit reinvestment and capital growth.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use absolute imports
from llm_overseer.strategy.compounding import CompoundingStrategy
from llm_overseer.config.config import Config

class TestCompoundingStrategy(unittest.TestCase):
    """Test compounding strategy logic."""
    
    def setUp(self):
        """Set up test environment."""
        # Create mock config
        self.config = MagicMock(spec=Config)
        
        # Configure mock to return specific values
        def config_get_side_effect(key, default=None):
            config_values = {
                "trading.compounding.enabled": True,
                "trading.compounding.reinvestment_rate": 0.8,
                "trading.compounding.min_profit_threshold": 100,
                "trading.compounding.frequency": "monthly"
            }
            return config_values.get(key, default)
        
        self.config.get.side_effect = config_get_side_effect
        
        # Create temporary data file path
        self.test_data_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "test_compounding_history.json"
        )
        
        # Patch the data file path
        with patch('llm_overseer.strategy.compounding.os.path.join', return_value=self.test_data_file):
            # Create compounding strategy
            self.compounding = CompoundingStrategy(self.config)
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove test data file if it exists
        if os.path.exists(self.test_data_file):
            os.remove(self.test_data_file)
    
    def test_initialization(self):
        """Test initialization of compounding strategy."""
        self.assertTrue(self.compounding.enabled)
        self.assertEqual(self.compounding.reinvestment_rate, 0.8)
        self.assertEqual(self.compounding.min_profit_threshold, 100)
        self.assertEqual(self.compounding.frequency, "monthly")
        self.assertEqual(self.compounding.initial_capital, 0.0)
        self.assertEqual(self.compounding.current_capital, 0.0)
    
    def test_initialize_capital(self):
        """Test initializing capital."""
        self.compounding.initialize_capital(10000.0)
        self.assertEqual(self.compounding.initial_capital, 10000.0)
        self.assertEqual(self.compounding.current_capital, 10000.0)
    
    def test_update_capital(self):
        """Test updating capital."""
        self.compounding.initialize_capital(10000.0)
        self.compounding.update_capital(10500.0)
        self.assertEqual(self.compounding.current_capital, 10500.0)
        self.assertEqual(self.compounding.total_profit, 500.0)
    
    def test_should_compound_first_time(self):
        """Test should_compound for first time."""
        current_date = datetime(2025, 6, 1)
        self.assertTrue(self.compounding.should_compound(current_date))
    
    def test_should_compound_monthly(self):
        """Test should_compound for monthly frequency."""
        # Set last compounding date to May 15, 2025
        self.compounding.last_compounding_date = datetime(2025, 5, 15)
        
        # Same month - should not compound
        self.assertFalse(self.compounding.should_compound(datetime(2025, 5, 30)))
        
        # Next month - should compound
        self.assertTrue(self.compounding.should_compound(datetime(2025, 6, 1)))
    
    def test_should_compound_weekly(self):
        """Test should_compound for weekly frequency."""
        # Set frequency to weekly
        self.compounding.frequency = "weekly"
        
        # Set last compounding date to June 1, 2025
        self.compounding.last_compounding_date = datetime(2025, 6, 1)
        
        # Less than 7 days - should not compound
        self.assertFalse(self.compounding.should_compound(datetime(2025, 6, 7)))
        
        # 7 days or more - should compound
        self.assertTrue(self.compounding.should_compound(datetime(2025, 6, 8)))
    
    def test_should_compound_daily(self):
        """Test should_compound for daily frequency."""
        # Set frequency to daily
        self.compounding.frequency = "daily"
        
        # Set last compounding date to June 1, 2025
        self.compounding.last_compounding_date = datetime(2025, 6, 1)
        
        # Same day - should not compound
        self.assertFalse(self.compounding.should_compound(datetime(2025, 6, 1)))
        
        # Next day - should compound
        self.assertTrue(self.compounding.should_compound(datetime(2025, 6, 2)))
    
    def test_calculate_compounding_below_threshold(self):
        """Test calculate_compounding below threshold."""
        self.compounding.initialize_capital(10000.0)
        result = self.compounding.calculate_compounding(10050.0)
        
        self.assertFalse(result["can_compound"])
        self.assertEqual(result["profit"], 50.0)
        self.assertEqual(result["reinvest_amount"], 0.0)
        self.assertEqual(result["withdraw_amount"], 0.0)
    
    def test_calculate_compounding_above_threshold(self):
        """Test calculate_compounding above threshold."""
        self.compounding.initialize_capital(10000.0)
        result = self.compounding.calculate_compounding(10200.0)
        
        self.assertTrue(result["can_compound"])
        self.assertEqual(result["profit"], 200.0)
        self.assertEqual(result["reinvest_amount"], 160.0)  # 80% of 200
        self.assertEqual(result["withdraw_amount"], 40.0)   # 20% of 200
        self.assertEqual(result["new_capital"], 10160.0)    # 10000 + 160
    
    def test_execute_compounding_not_scheduled(self):
        """Test execute_compounding when not scheduled."""
        self.compounding.initialize_capital(10000.0)
        self.compounding.last_compounding_date = datetime(2025, 6, 1)
        
        result = self.compounding.execute_compounding(10200.0, datetime(2025, 6, 15))
        
        self.assertFalse(result["compounded"])
        self.assertEqual(result["reason"], "Compounding not scheduled for this period")
    
    def test_execute_compounding_below_threshold(self):
        """Test execute_compounding below threshold."""
        self.compounding.initialize_capital(10000.0)
        
        result = self.compounding.execute_compounding(10050.0, datetime(2025, 6, 1))
        
        self.assertFalse(result["compounded"])
        self.assertTrue("below minimum threshold" in result["reason"])
    
    def test_execute_compounding_success(self):
        """Test execute_compounding success."""
        self.compounding.initialize_capital(10000.0)
        
        result = self.compounding.execute_compounding(10200.0, datetime(2025, 6, 1))
        
        self.assertTrue(result["compounded"])
        self.assertEqual(result["capital_before"], 10200.0)
        self.assertEqual(result["profit"], 200.0)
        self.assertEqual(result["reinvested"], 160.0)
        self.assertEqual(result["withdrawn"], 40.0)
        self.assertEqual(result["capital_after"], 10160.0)
        
        # Check that state was updated
        self.assertEqual(self.compounding.initial_capital, 10160.0)
        self.assertEqual(self.compounding.reinvested_profit, 160.0)
        self.assertEqual(self.compounding.withdrawn_profit, 40.0)
        self.assertEqual(self.compounding.last_compounding_date, datetime(2025, 6, 1))
        self.assertEqual(len(self.compounding.history), 1)
    
    def test_get_statistics(self):
        """Test get_statistics."""
        self.compounding.initialize_capital(10000.0)
        self.compounding.update_capital(10200.0)
        self.compounding.execute_compounding(10200.0, datetime(2025, 6, 1))
        
        stats = self.compounding.get_statistics()
        
        self.assertTrue(stats["enabled"])
        self.assertEqual(stats["reinvestment_rate"], 0.8)
        self.assertEqual(stats["frequency"], "monthly")
        self.assertEqual(stats["initial_capital"], 10160.0)
        self.assertEqual(stats["current_capital"], 10200.0)
        self.assertEqual(stats["total_profit"], 200.0)
        self.assertEqual(stats["reinvested_profit"], 160.0)
        self.assertEqual(stats["withdrawn_profit"], 40.0)
        self.assertEqual(stats["compounding_events"], 1)
    
    def test_set_reinvestment_rate(self):
        """Test set_reinvestment_rate."""
        self.compounding.set_reinvestment_rate(0.5)
        self.assertEqual(self.compounding.reinvestment_rate, 0.5)
        
        # Test invalid rate
        self.compounding.set_reinvestment_rate(1.5)
        self.assertEqual(self.compounding.reinvestment_rate, 0.5)  # Should not change
    
    def test_enable_disable(self):
        """Test enable and disable."""
        self.compounding.disable()
        self.assertFalse(self.compounding.enabled)
        
        self.compounding.enable()
        self.assertTrue(self.compounding.enabled)
    
    def test_set_frequency(self):
        """Test set_frequency."""
        self.compounding.set_frequency("weekly")
        self.assertEqual(self.compounding.frequency, "weekly")
        
        self.compounding.set_frequency("daily")
        self.assertEqual(self.compounding.frequency, "daily")
        
        # Test invalid frequency
        self.compounding.set_frequency("yearly")
        self.assertEqual(self.compounding.frequency, "daily")  # Should not change

if __name__ == "__main__":
    unittest.main()
