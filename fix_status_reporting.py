#!/usr/bin/env python
"""
Fix for the status reporting type error in flash_trading.py

This script creates a patched version of the _print_status method
that correctly accesses the signal generator statistics.
"""

import os
import re

def fix_print_status_method():
    """
    Fix the _print_status method in flash_trading.py to correctly
    access signal generator statistics without using safe_get_nested
    on a non-dictionary object.
    """
    # Path to the flash trading script
    file_path = 'flash_trading.py'
    
    # Read the current content
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Find the problematic line
    pattern = r'signals = safe_get_nested\(self\.signal_generator, \["stats", "signals_generated"\], 0\)'
    
    # Create the replacement line - directly access stats attribute or method
    replacement = 'signals = getattr(self.signal_generator, "stats", {}).get("signals_generated", 0)'
    
    # Replace the line
    new_content = re.sub(pattern, replacement, content)
    
    # Write the fixed content to a new file
    fixed_file_path = 'flash_trading_fixed.py'
    with open(fixed_file_path, 'w') as file:
        file.write(new_content)
    
    print(f"Fixed version saved to {fixed_file_path}")
    return fixed_file_path

if __name__ == "__main__":
    fixed_file = fix_print_status_method()
    print(f"To test the fixed version, run: python {fixed_file} --duration 300 --reset")
