#!/usr/bin/env python
"""
Enhanced Flash Trading Script with Adjusted Thresholds

This script modifies the flash_trading_fixed.py script to use lower thresholds
for signal generation, enabling more active trading in paper trading mode.
"""

import os
import sys
import re

def enhance_flash_trading():
    """
    Enhance the flash trading script with lower thresholds for more active trading.
    """
    # Path to the fixed flash trading script
    file_path = 'flash_trading_fixed.py'
    
    # Read the current content
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Modify session parameters for US session to use lower thresholds
    pattern = r"'US': \{\s*'imbalance_threshold': ([\d\.]+),\s*'volatility_threshold': ([\d\.]+),\s*'momentum_threshold': ([\d\.]+)"
    
    # Create the replacement with lower thresholds (20% of original values)
    replacement = "'US': {\n      'imbalance_threshold': 0.05,  # Lowered from 0.25\n      'volatility_threshold': 0.016,  # Lowered from 0.08\n      'momentum_threshold': 0.012,  # Lowered from 0.06"
    
    # Replace the thresholds
    new_content = re.sub(pattern, replacement, content)
    
    # Write the enhanced content to a new file
    enhanced_file_path = 'flash_trading_enhanced.py'
    with open(enhanced_file_path, 'w') as file:
        file.write(new_content)
    
    print(f"Enhanced version saved to {enhanced_file_path}")
    return enhanced_file_path

if __name__ == "__main__":
    enhanced_file = enhance_flash_trading()
    print(f"To test the enhanced version with lower thresholds, run: python {enhanced_file} --duration 300 --reset")
