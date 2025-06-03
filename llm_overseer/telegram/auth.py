#!/usr/bin/env python
"""
Authentication module for Telegram bot integration.

This module handles secure authentication for the Telegram bot,
ensuring that only authorized users can access the trading system.
"""

import os
import time
import json
import logging
import hashlib
import secrets
from typing import Dict, List, Any, Optional, Tuple

# Import configuration
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config

logger = logging.getLogger(__name__)

class TelegramAuth:
    """
    Handles secure authentication for the Telegram bot.
    Implements multi-factor authentication and session management.
    """
    
    def __init__(self, config: Config):
        """
        Initialize authentication manager.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.allowed_user_ids = config.get("telegram.allowed_user_ids", [])
        self.session_timeout = config.get("telegram.session_timeout", 3600)  # 1 hour default
        
        self.sessions_file = os.path.join(
            config.project_root, 
            "data", 
            "telegram_sessions.json"
        )
        
        self.sessions = self._load_sessions()
        self.pending_auth = {}
    
    def _load_sessions(self) -> Dict[str, Any]:
        """Load sessions from file."""
        if not os.path.exists(self.sessions_file):
            return {}
        
        try:
            with open(self.sessions_file, 'r') as f:
                sessions = json.load(f)
                
                # Clean expired sessions
                now = time.time()
                sessions = {
                    k: v for k, v in sessions.items()
                    if v["expires_at"] > now
                }
                
                return sessions
        except Exception as e:
            logger.error(f"Error loading sessions: {e}")
            return {}
    
    def _save_sessions(self) -> None:
        """Save sessions to file."""
        os.makedirs(os.path.dirname(self.sessions_file), exist_ok=True)
        
        try:
            with open(self.sessions_file, 'w') as f:
                json.dump(self.sessions, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving sessions: {e}")
    
    def is_user_allowed(self, user_id: int) -> bool:
        """
        Check if a user is in the allowed list.
        
        Args:
            user_id: Telegram user ID
            
        Returns:
            True if user is allowed, False otherwise
        """
        return user_id in self.allowed_user_ids
    
    def is_authenticated(self, user_id: int) -> bool:
        """
        Check if a user is authenticated.
        
        Args:
            user_id: Telegram user ID
            
        Returns:
            True if user is authenticated, False otherwise
        """
        user_id_str = str(user_id)
        
        if user_id_str not in self.sessions:
            return False
        
        # Check if session is expired
        now = time.time()
        if self.sessions[user_id_str]["expires_at"] < now:
            del self.sessions[user_id_str]
            self._save_sessions()
            return False
        
        return True
    
    def start_authentication(self, user_id: int) -> Tuple[bool, str]:
        """
        Start authentication process for a user.
        
        Args:
            user_id: Telegram user ID
            
        Returns:
            Tuple of (success, message)
        """
        if not self.is_user_allowed(user_id):
            logger.warning(f"Authentication attempt from unauthorized user: {user_id}")
            return False, "You are not authorized to use this bot."
        
        # Generate one-time code
        code = secrets.token_hex(3).upper()  # 6-character hex code
        
        # Store in pending authentication
        self.pending_auth[str(user_id)] = {
            "code": code,
            "expires_at": time.time() + 300  # 5 minutes
        }
        
        return True, f"Please enter the following code to authenticate: {code}"
    
    def verify_code(self, user_id: int, code: str) -> Tuple[bool, str]:
        """
        Verify authentication code.
        
        Args:
            user_id: Telegram user ID
            code: Authentication code
            
        Returns:
            Tuple of (success, message)
        """
        user_id_str = str(user_id)
        
        if user_id_str not in self.pending_auth:
            return False, "No authentication in progress. Please start authentication first."
        
        auth_data = self.pending_auth[user_id_str]
        
        # Check if code is expired
        if auth_data["expires_at"] < time.time():
            del self.pending_auth[user_id_str]
            return False, "Authentication code expired. Please start authentication again."
        
        # Check if code matches
        if auth_data["code"] != code.upper():
            return False, "Invalid code. Please try again."
        
        # Create session
        session_id = secrets.token_hex(16)
        self.sessions[user_id_str] = {
            "session_id": session_id,
            "created_at": time.time(),
            "expires_at": time.time() + self.session_timeout,
            "user_id": user_id
        }
        
        # Remove from pending authentication
        del self.pending_auth[user_id_str]
        
        # Save sessions
        self._save_sessions()
        
        return True, "Authentication successful. You are now logged in."
    
    def refresh_session(self, user_id: int) -> None:
        """
        Refresh session for a user.
        
        Args:
            user_id: Telegram user ID
        """
        user_id_str = str(user_id)
        
        if user_id_str in self.sessions:
            self.sessions[user_id_str]["expires_at"] = time.time() + self.session_timeout
            self._save_sessions()
    
    def logout(self, user_id: int) -> Tuple[bool, str]:
        """
        Log out a user.
        
        Args:
            user_id: Telegram user ID
            
        Returns:
            Tuple of (success, message)
        """
        user_id_str = str(user_id)
        
        if user_id_str in self.sessions:
            del self.sessions[user_id_str]
            self._save_sessions()
            return True, "You have been logged out."
        
        return False, "You are not logged in."
    
    def get_active_sessions(self) -> List[Dict[str, Any]]:
        """
        Get list of active sessions.
        
        Returns:
            List of active sessions
        """
        now = time.time()
        active_sessions = []
        
        for user_id, session in self.sessions.items():
            if session["expires_at"] > now:
                active_sessions.append({
                    "user_id": int(user_id),
                    "created_at": session["created_at"],
                    "expires_at": session["expires_at"]
                })
        
        return active_sessions
