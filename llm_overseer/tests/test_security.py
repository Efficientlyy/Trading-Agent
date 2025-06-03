#!/usr/bin/env python
"""
Security validation test for LLM Strategic Overseer.

This module tests the security and authentication mechanisms of the LLM Overseer system,
focusing on Telegram bot authentication, session management, and access control.
"""

import os
import sys
import asyncio
import logging
import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import components
from llm_overseer.telegram.auth import TelegramAuth

class TestSecurity(unittest.TestCase):
    """Test security and authentication mechanisms."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary data directory
        self.test_data_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "test_security_data"
        )
        os.makedirs(self.test_data_dir, exist_ok=True)
        
        # Create a mock config class that returns real values
        self.mock_config = MagicMock()
        def mock_config_get(key, default=None):
            if key == "telegram.allowed_user_ids":
                return [123456789]
            elif key == "telegram.session_timeout":
                return 3600  # 1 hour
            return default
        self.mock_config.get.side_effect = mock_config_get
        
        # Set project_root for the mock config
        self.mock_config.project_root = self.test_data_dir
        
        # Initialize authentication manager
        with patch('llm_overseer.telegram.auth.os.path.join', return_value=os.path.join(self.test_data_dir, "telegram_sessions.json")):
            self.auth = TelegramAuth(self.mock_config)
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove test data directory if it exists
        import shutil
        if os.path.exists(self.test_data_dir):
            shutil.rmtree(self.test_data_dir)
    
    def test_user_authorization(self):
        """Test user authorization."""
        # Allowed user
        self.assertTrue(self.auth.is_user_allowed(123456789))
        
        # Unauthorized user
        self.assertFalse(self.auth.is_user_allowed(987654321))
    
    def test_authentication_flow(self):
        """Test authentication flow."""
        # Start authentication for allowed user
        success, message = self.auth.start_authentication(123456789)
        self.assertTrue(success)
        self.assertIn("Please enter the following code", message)
        
        # Get code directly from pending_auth instead of parsing the message
        user_id_str = str(123456789)
        self.assertIn(user_id_str, self.auth.pending_auth)
        code = self.auth.pending_auth[user_id_str]["code"]
        
        # Verify code
        success, message = self.auth.verify_code(123456789, code)
        self.assertTrue(success)
        self.assertIn("Authentication successful", message)
        
        # Check if user is authenticated
        self.assertTrue(self.auth.is_authenticated(123456789))
        
        # Start authentication for unauthorized user
        success, message = self.auth.start_authentication(987654321)
        self.assertFalse(success)
        self.assertIn("not authorized", message)
    
    def test_invalid_code(self):
        """Test invalid authentication code."""
        # Start authentication
        self.auth.start_authentication(123456789)
        
        # Verify with invalid code
        success, message = self.auth.verify_code(123456789, "INVALID")
        self.assertFalse(success)
        self.assertIn("Invalid code", message)
        
        # Check if user is not authenticated
        self.assertFalse(self.auth.is_authenticated(123456789))
    
    def test_session_expiration(self):
        """Test session expiration."""
        # Start authentication
        self.auth.start_authentication(123456789)
        
        # Extract code from pending_auth
        code = self.auth.pending_auth[str(123456789)]["code"]
        
        # Verify code
        self.auth.verify_code(123456789, code)
        
        # Check if user is authenticated
        self.assertTrue(self.auth.is_authenticated(123456789))
        
        # Manually expire session
        self.auth.sessions[str(123456789)]["expires_at"] = datetime.now().timestamp() - 3600
        
        # Check if user is no longer authenticated
        self.assertFalse(self.auth.is_authenticated(123456789))
    
    def test_session_refresh(self):
        """Test session refresh."""
        # Start authentication
        self.auth.start_authentication(123456789)
        
        # Extract code from pending_auth
        code = self.auth.pending_auth[str(123456789)]["code"]
        
        # Verify code
        self.auth.verify_code(123456789, code)
        
        # Get original expiration time
        original_expires_at = self.auth.sessions[str(123456789)]["expires_at"]
        
        # Wait a moment
        import time
        time.sleep(1)
        
        # Refresh session
        self.auth.refresh_session(123456789)
        
        # Check if expiration time was extended
        new_expires_at = self.auth.sessions[str(123456789)]["expires_at"]
        self.assertGreater(new_expires_at, original_expires_at)
    
    def test_logout(self):
        """Test logout."""
        # Start authentication
        self.auth.start_authentication(123456789)
        
        # Extract code from pending_auth
        code = self.auth.pending_auth[str(123456789)]["code"]
        
        # Verify code
        self.auth.verify_code(123456789, code)
        
        # Check if user is authenticated
        self.assertTrue(self.auth.is_authenticated(123456789))
        
        # Logout
        success, message = self.auth.logout(123456789)
        self.assertTrue(success)
        self.assertIn("logged out", message)
        
        # Check if user is no longer authenticated
        self.assertFalse(self.auth.is_authenticated(123456789))
    
    def test_active_sessions(self):
        """Test active sessions."""
        # Start authentication
        self.auth.start_authentication(123456789)
        
        # Extract code from pending_auth
        code = self.auth.pending_auth[str(123456789)]["code"]
        
        # Verify code
        self.auth.verify_code(123456789, code)
        
        # Get active sessions
        active_sessions = self.auth.get_active_sessions()
        
        # Check if session is active
        self.assertEqual(len(active_sessions), 1)
        self.assertEqual(active_sessions[0]["user_id"], 123456789)
    
    def test_authentication_code_expiration(self):
        """Test authentication code expiration."""
        # Start authentication
        self.auth.start_authentication(123456789)
        
        # Extract code from pending_auth
        code = self.auth.pending_auth[str(123456789)]["code"]
        
        # Manually expire code
        self.auth.pending_auth[str(123456789)]["expires_at"] = datetime.now().timestamp() - 300
        
        # Verify expired code
        success, message = self.auth.verify_code(123456789, code)
        self.assertFalse(success)
        self.assertIn("expired", message)
        
        # Check if user is not authenticated
        self.assertFalse(self.auth.is_authenticated(123456789))

if __name__ == "__main__":
    unittest.main()
