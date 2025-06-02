#!/usr/bin/env python
"""
Parameter Management System Validation Script

This script validates the parameter management system by:
1. Testing the backend API endpoints
2. Validating parameter updates and validation
3. Testing preset management
4. Ensuring proper error handling
"""

import os
import sys
import json
import requests
import time
import logging
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API base URL
API_BASE_URL = 'http://localhost:5001/api'

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
TEST_PRESET = 'conservative'
CUSTOM_PRESET = {
    'name': 'test_custom_preset',
    'description': 'Test custom preset for validation'
}

def test_api_connection() -> bool:
    """Test connection to the parameter management API."""
    try:
        response = requests.get(f"{API_BASE_URL}/parameters")
        response.raise_for_status()
        logger.info("✅ API connection successful")
        return True
    except Exception as e:
        logger.error(f"❌ API connection failed: {e}")
        return False

def test_get_all_parameters() -> Dict[str, Any]:
    """Test getting all parameters."""
    try:
        response = requests.get(f"{API_BASE_URL}/parameters")
        response.raise_for_status()
        parameters = response.json()
        
        # Validate response structure
        if not isinstance(parameters, dict) or len(parameters) == 0:
            logger.error("❌ Invalid parameters response structure")
            return {}
        
        logger.info(f"✅ Successfully retrieved all parameters ({len(parameters)} modules)")
        return parameters
    except Exception as e:
        logger.error(f"❌ Failed to get all parameters: {e}")
        return {}

def test_get_module_parameters(module: str) -> Dict[str, Any]:
    """Test getting parameters for a specific module."""
    try:
        response = requests.get(f"{API_BASE_URL}/parameters/{module}")
        response.raise_for_status()
        parameters = response.json()
        
        # Validate response structure
        if not isinstance(parameters, dict) or len(parameters) == 0:
            logger.error(f"❌ Invalid module parameters response structure for {module}")
            return {}
        
        logger.info(f"✅ Successfully retrieved parameters for {module} ({len(parameters)} parameters)")
        return parameters
    except Exception as e:
        logger.error(f"❌ Failed to get module parameters for {module}: {e}")
        return {}

def test_update_parameters(module: str, parameters: Dict[str, Any]) -> bool:
    """Test updating parameters for a specific module."""
    try:
        response = requests.put(f"{API_BASE_URL}/parameters/{module}", json=parameters)
        response.raise_for_status()
        result = response.json()
        
        # Validate response structure
        if not isinstance(result, dict) or 'success' not in result:
            logger.error(f"❌ Invalid update parameters response structure for {module}")
            return False
        
        if result['success']:
            logger.info(f"✅ Successfully updated parameters for {module}")
            return True
        else:
            logger.error(f"❌ Failed to update parameters for {module}: {result.get('message', 'Unknown error')}")
            return False
    except Exception as e:
        logger.error(f"❌ Failed to update parameters for {module}: {e}")
        return False

def test_parameter_validation(module: str, parameters: Dict[str, Any]) -> Dict[str, List[str]]:
    """Test parameter validation."""
    try:
        response = requests.post(f"{API_BASE_URL}/parameters/validate", json={
            'module': module,
            'parameters': parameters
        })
        response.raise_for_status()
        result = response.json()
        
        # Validate response structure
        if not isinstance(result, dict) or 'valid' not in result:
            logger.error(f"❌ Invalid parameter validation response structure")
            return {}
        
        if result['valid']:
            logger.info(f"✅ Parameters for {module} are valid")
            return {}
        else:
            logger.info(f"✅ Parameter validation correctly identified invalid parameters for {module}")
            logger.info(f"Validation errors: {result.get('errors', {})}")
            return result.get('errors', {})
    except Exception as e:
        logger.error(f"❌ Failed to validate parameters for {module}: {e}")
        return {}

def test_apply_preset(preset: str) -> bool:
    """Test applying a parameter preset."""
    try:
        response = requests.post(f"{API_BASE_URL}/parameters/presets/{preset}")
        response.raise_for_status()
        result = response.json()
        
        # Validate response structure
        if not isinstance(result, dict) or 'success' not in result:
            logger.error(f"❌ Invalid apply preset response structure for {preset}")
            return False
        
        if result['success']:
            logger.info(f"✅ Successfully applied preset {preset}")
            return True
        else:
            logger.error(f"❌ Failed to apply preset {preset}: {result.get('message', 'Unknown error')}")
            return False
    except Exception as e:
        logger.error(f"❌ Failed to apply preset {preset}: {e}")
        return False

def test_save_custom_preset(preset_data: Dict[str, str]) -> bool:
    """Test saving a custom parameter preset."""
    try:
        response = requests.post(f"{API_BASE_URL}/parameters/presets/custom", json=preset_data)
        response.raise_for_status()
        result = response.json()
        
        # Validate response structure
        if not isinstance(result, dict) or 'success' not in result:
            logger.error(f"❌ Invalid save custom preset response structure")
            return False
        
        if result['success']:
            logger.info(f"✅ Successfully saved custom preset {preset_data['name']}")
            return True
        else:
            logger.error(f"❌ Failed to save custom preset: {result.get('message', 'Unknown error')}")
            return False
    except Exception as e:
        logger.error(f"❌ Failed to save custom preset: {e}")
        return False

def test_get_parameter_metadata() -> Dict[str, Any]:
    """Test getting parameter metadata."""
    try:
        response = requests.get(f"{API_BASE_URL}/parameters/metadata")
        response.raise_for_status()
        metadata = response.json()
        
        # Validate response structure
        if not isinstance(metadata, dict) or len(metadata) == 0:
            logger.error("❌ Invalid parameter metadata response structure")
            return {}
        
        logger.info(f"✅ Successfully retrieved parameter metadata ({len(metadata)} parameters)")
        return metadata
    except Exception as e:
        logger.error(f"❌ Failed to get parameter metadata: {e}")
        return {}

def test_reset_parameters() -> bool:
    """Test resetting parameters to defaults."""
    try:
        response = requests.post(f"{API_BASE_URL}/parameters/reset")
        response.raise_for_status()
        result = response.json()
        
        # Validate response structure
        if not isinstance(result, dict) or 'success' not in result:
            logger.error(f"❌ Invalid reset parameters response structure")
            return False
        
        if result['success']:
            logger.info(f"✅ Successfully reset parameters to defaults")
            return True
        else:
            logger.error(f"❌ Failed to reset parameters: {result.get('message', 'Unknown error')}")
            return False
    except Exception as e:
        logger.error(f"❌ Failed to reset parameters: {e}")
        return False

def test_get_audit_log() -> List[Dict[str, Any]]:
    """Test getting parameter change audit log."""
    try:
        response = requests.get(f"{API_BASE_URL}/parameters/audit-log")
        response.raise_for_status()
        audit_log = response.json()
        
        # Validate response structure
        if not isinstance(audit_log, list):
            logger.error("❌ Invalid audit log response structure")
            return []
        
        logger.info(f"✅ Successfully retrieved audit log ({len(audit_log)} entries)")
        return audit_log
    except Exception as e:
        logger.error(f"❌ Failed to get audit log: {e}")
        return []

def run_validation_tests():
    """Run all validation tests."""
    logger.info("Starting parameter management system validation...")
    
    # Test API connection
    if not test_api_connection():
        logger.error("❌ Validation failed: Cannot connect to API")
        return False
    
    # Test getting all parameters
    all_parameters = test_get_all_parameters()
    if not all_parameters:
        logger.error("❌ Validation failed: Cannot get all parameters")
        return False
    
    # Test getting module parameters
    module_parameters = test_get_module_parameters(TEST_MODULE)
    if not module_parameters:
        logger.error(f"❌ Validation failed: Cannot get parameters for {TEST_MODULE}")
        return False
    
    # Test parameter validation (valid parameters)
    validation_errors = test_parameter_validation(TEST_MODULE, VALID_PARAMETERS)
    if validation_errors:
        logger.error(f"❌ Validation failed: Valid parameters were rejected")
        return False
    
    # Test parameter validation (invalid parameters)
    validation_errors = test_parameter_validation(TEST_MODULE, INVALID_PARAMETERS)
    if not validation_errors:
        logger.error(f"❌ Validation failed: Invalid parameters were accepted")
        return False
    
    # Test updating parameters
    if not test_update_parameters(TEST_MODULE, VALID_PARAMETERS):
        logger.error(f"❌ Validation failed: Cannot update parameters for {TEST_MODULE}")
        return False
    
    # Test applying preset
    if not test_apply_preset(TEST_PRESET):
        logger.error(f"❌ Validation failed: Cannot apply preset {TEST_PRESET}")
        return False
    
    # Test saving custom preset
    if not test_save_custom_preset(CUSTOM_PRESET):
        logger.error(f"❌ Validation failed: Cannot save custom preset")
        return False
    
    # Test getting parameter metadata
    metadata = test_get_parameter_metadata()
    if not metadata:
        logger.error(f"❌ Validation failed: Cannot get parameter metadata")
        return False
    
    # Test getting audit log
    audit_log = test_get_audit_log()
    if audit_log is None:
        logger.error(f"❌ Validation failed: Cannot get audit log")
        return False
    
    # Test resetting parameters
    if not test_reset_parameters():
        logger.error(f"❌ Validation failed: Cannot reset parameters")
        return False
    
    logger.info("✅ All validation tests passed successfully!")
    return True

def start_api_server():
    """Start the parameter management API server."""
    import subprocess
    import time
    
    logger.info("Starting parameter management API server...")
    
    # Start the server in a separate process
    server_process = subprocess.Popen(
        ["python", "parameter_management_api.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for the server to start
    time.sleep(5)
    
    return server_process

def stop_api_server(server_process):
    """Stop the parameter management API server."""
    logger.info("Stopping parameter management API server...")
    server_process.terminate()
    server_process.wait()

if __name__ == "__main__":
    # Start the API server
    server_process = start_api_server()
    
    try:
        # Run validation tests
        success = run_validation_tests()
        
        if success:
            logger.info("Parameter management system validation completed successfully!")
            sys.exit(0)
        else:
            logger.error("Parameter management system validation failed!")
            sys.exit(1)
    finally:
        # Stop the API server
        stop_api_server(server_process)
