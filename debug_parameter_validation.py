#!/usr/bin/env python
"""
Debug script for parameter validation issues

This script performs a deep dive into the parameter validation logic
to identify why invalid parameters are being accepted.
"""

import os
import sys
import json
import logging
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import validation functions from parameter_management_api.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from parameter_management_api import (
    VALIDATION_RULES,
    PARAMETER_METADATA,
    validate_parameters,
    load_parameters
)

# Test parameters
TEST_MODULE = 'risk_management'
VALID_PARAMETERS = {
    'risk_level': 'low',
    'max_portfolio_risk_percent': 1.5,
    'max_position_size_usd': 500,
    'stop_loss_percent': 1.5
}
INVALID_PARAMETERS = {
    'risk_level': 'extreme',  # Invalid option
    'max_portfolio_risk_percent': 15.0,  # Out of range
    'max_position_size_usd': -100,  # Negative value
    'stop_loss_percent': 0.0  # Zero value
}

def debug_validation_rules():
    """Debug the validation rules."""
    logger.info("Debugging validation rules...")
    
    # Test numeric_range rule
    test_values = [
        (5, 0, 10, True),
        (0, 0, 10, True),
        (10, 0, 10, True),
        (-1, 0, 10, False),
        (11, 0, 10, False),
        ("5", 0, 10, False),  # String instead of number
        (None, 0, 10, False)  # None value
    ]
    
    for value, min_val, max_val, expected in test_values:
        result = VALIDATION_RULES["numeric_range"](value, min_val, max_val)
        logger.info(f"numeric_range({value}, {min_val}, {max_val}) = {result}, expected: {expected}")
        if result != expected:
            logger.error(f"❌ Validation rule 'numeric_range' failed for value {value}")
    
    # Test option_valid rule
    options = ["low", "medium", "high"]
    test_values = [
        ("low", options, True),
        ("medium", options, True),
        ("high", options, True),
        ("extreme", options, False),
        ("", options, False),
        (None, options, False)
    ]
    
    for value, opts, expected in test_values:
        result = VALIDATION_RULES["option_valid"](value, opts)
        logger.info(f"option_valid({value}, {opts}) = {result}, expected: {expected}")
        if result != expected:
            logger.error(f"❌ Validation rule 'option_valid' failed for value {value}")
    
    # Test array_valid rule
    options = ["BTC/USDC", "ETH/USDC", "SOL/USDC"]
    test_values = [
        (["BTC/USDC"], options, True),
        (["BTC/USDC", "ETH/USDC"], options, True),
        (["BTC/USDC", "ETH/USDC", "SOL/USDC"], options, True),
        (["BTC/USDT"], options, False),
        (["BTC/USDC", "invalid"], options, False),
        ("BTC/USDC", options, False),  # String instead of array
        (None, options, False)  # None value
    ]
    
    for value, opts, expected in test_values:
        try:
            result = VALIDATION_RULES["array_valid"](value, opts)
            logger.info(f"array_valid({value}, {opts}) = {result}, expected: {expected}")
            if result != expected:
                logger.error(f"❌ Validation rule 'array_valid' failed for value {value}")
        except Exception as e:
            logger.error(f"❌ Validation rule 'array_valid' raised exception for value {value}: {e}")

def debug_parameter_metadata():
    """Debug the parameter metadata."""
    logger.info("Debugging parameter metadata...")
    
    # Check if all parameters have metadata
    for param_name in INVALID_PARAMETERS:
        full_param_name = f"{TEST_MODULE}.{param_name}"
        if full_param_name not in PARAMETER_METADATA:
            logger.error(f"❌ No metadata found for parameter {full_param_name}")
            continue
        
        metadata = PARAMETER_METADATA[full_param_name]
        logger.info(f"Metadata for {full_param_name}: {metadata}")
        
        # Check if metadata has required fields
        if metadata["type"] == "numeric":
            if "min" not in metadata or "max" not in metadata:
                logger.error(f"❌ Numeric parameter {full_param_name} missing min/max values in metadata")
        
        elif metadata["type"] == "option":
            if "options" not in metadata:
                logger.error(f"❌ Option parameter {full_param_name} missing options list in metadata")
        
        elif metadata["type"] == "array":
            if "options" not in metadata:
                logger.error(f"❌ Array parameter {full_param_name} missing options list in metadata")

def debug_validate_parameters():
    """Debug the validate_parameters function."""
    logger.info("Debugging validate_parameters function...")
    
    # Test with valid parameters
    errors = validate_parameters(TEST_MODULE, VALID_PARAMETERS)
    logger.info(f"Validation errors for valid parameters: {errors}")
    if errors:
        logger.error(f"❌ Valid parameters were rejected: {errors}")
    
    # Test with invalid parameters
    errors = validate_parameters(TEST_MODULE, INVALID_PARAMETERS)
    logger.info(f"Validation errors for invalid parameters: {errors}")
    if not errors:
        logger.error(f"❌ Invalid parameters were accepted")
    else:
        # Check if all invalid parameters were caught
        for param_name in INVALID_PARAMETERS:
            if param_name not in errors:
                logger.error(f"❌ Invalid parameter {param_name} was not caught")
    
    # Test with individual invalid parameters
    for param_name, param_value in INVALID_PARAMETERS.items():
        errors = validate_parameters(TEST_MODULE, {param_name: param_value})
        logger.info(f"Validation errors for invalid parameter {param_name}: {errors}")
        if not errors:
            logger.error(f"❌ Invalid parameter {param_name} was accepted")
        elif param_name not in errors:
            logger.error(f"❌ Invalid parameter {param_name} was not caught")

def debug_parameter_validation():
    """Debug the parameter validation system."""
    logger.info("Starting parameter validation debugging...")
    
    # Debug validation rules
    debug_validation_rules()
    
    # Debug parameter metadata
    debug_parameter_metadata()
    
    # Debug validate_parameters function
    debug_validate_parameters()
    
    logger.info("Parameter validation debugging completed")

if __name__ == "__main__":
    debug_parameter_validation()
