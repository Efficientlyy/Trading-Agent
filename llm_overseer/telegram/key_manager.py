#!/usr/bin/env python
"""
Key Manager module for secure API key management via Telegram.
This module implements secure storage, encryption, and retrieval of API keys.
"""
import os
import json
import time
import base64
import logging
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Union
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)

class KeyManager:
    """
    Secure API key management system.
    Handles encryption, storage, and retrieval of API keys.
    """
    
    def __init__(self, bot_token: str, keys_file: str = None):
        """
        Initialize Key Manager.
        
        Args:
            bot_token: Telegram bot token (used for encryption key derivation)
            keys_file: Path to keys file (default: keys.json in the same directory)
        """
        self.keys_file = keys_file or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "secure_keys.json"
        )
        
        # Initialize encryption
        self.fernet = self._initialize_encryption(bot_token)
        
        # Load keys
        self.keys = self._load_keys()
        
        logger.info("Key Manager initialized")
    
    def _initialize_encryption(self, bot_token: str) -> Fernet:
        """
        Initialize encryption with key derived from bot token.
        
        Args:
            bot_token: Telegram bot token
            
        Returns:
            Fernet encryption instance
        """
        # Use PBKDF2 to derive a key from the bot token
        salt = b'telegram_bot_key_manager'  # Fixed salt
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        # Derive key from bot token
        key = base64.urlsafe_b64encode(kdf.derive(bot_token.encode()))
        
        return Fernet(key)
    
    def _load_keys(self) -> Dict[str, Dict[str, Any]]:
        """
        Load keys from file.
        
        Returns:
            Dictionary of keys
        """
        if not os.path.exists(self.keys_file):
            logger.info(f"Keys file not found, creating new one: {self.keys_file}")
            return {}
        
        try:
            with open(self.keys_file, 'r') as f:
                encrypted_data = json.load(f)
            
            # Decrypt keys
            keys = {}
            for service, service_data in encrypted_data.items():
                keys[service] = {}
                for key_type, key_data in service_data.items():
                    if key_type in ['last_updated', 'updated_by']:
                        keys[service][key_type] = key_data
                    else:
                        # Decrypt the value
                        encrypted_value = key_data.encode()
                        decrypted_value = self.fernet.decrypt(encrypted_value).decode()
                        keys[service][key_type] = decrypted_value
            
            logger.info(f"Loaded keys for {len(keys)} services")
            return keys
            
        except Exception as e:
            logger.error(f"Error loading keys: {e}")
            return {}
    
    def _save_keys(self) -> bool:
        """
        Save keys to file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.keys_file), exist_ok=True)
            
            # Encrypt keys
            encrypted_data = {}
            for service, service_data in self.keys.items():
                encrypted_data[service] = {}
                for key_type, value in service_data.items():
                    if key_type in ['last_updated', 'updated_by']:
                        encrypted_data[service][key_type] = value
                    else:
                        # Encrypt the value
                        encrypted_value = self.fernet.encrypt(value.encode())
                        encrypted_data[service][key_type] = encrypted_value.decode()
            
            # Save to file
            with open(self.keys_file, 'w') as f:
                json.dump(encrypted_data, f, indent=2)
            
            logger.info(f"Saved keys to {self.keys_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving keys: {e}")
            return False
    
    def set_key(self, service: str, key_type: str, value: str, user_id: int) -> bool:
        """
        Set an API key.
        
        Args:
            service: Service name (e.g., 'mexc', 'openrouter')
            key_type: Key type (e.g., 'api_key', 'api_secret')
            value: Key value
            user_id: User ID of the user setting the key
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Initialize service if it doesn't exist
            if service not in self.keys:
                self.keys[service] = {}
            
            # Set key
            self.keys[service][key_type] = value
            self.keys[service]['last_updated'] = time.time()
            self.keys[service]['updated_by'] = user_id
            
            # Save keys
            return self._save_keys()
            
        except Exception as e:
            logger.error(f"Error setting key: {e}")
            return False
    
    def get_key(self, service: str, key_type: str) -> Optional[str]:
        """
        Get an API key.
        
        Args:
            service: Service name (e.g., 'mexc', 'openrouter')
            key_type: Key type (e.g., 'api_key', 'api_secret')
            
        Returns:
            Key value or None if not found
        """
        try:
            return self.keys.get(service, {}).get(key_type)
        except Exception as e:
            logger.error(f"Error getting key: {e}")
            return None
    
    def delete_key(self, service: str, key_type: str) -> bool:
        """
        Delete an API key.
        
        Args:
            service: Service name (e.g., 'mexc', 'openrouter')
            key_type: Key type (e.g., 'api_key', 'api_secret')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if service in self.keys and key_type in self.keys[service]:
                del self.keys[service][key_type]
                
                # Remove service if no keys left
                if not any(k for k in self.keys[service] if k not in ['last_updated', 'updated_by']):
                    del self.keys[service]
                
                # Save keys
                return self._save_keys()
            
            return False
            
        except Exception as e:
            logger.error(f"Error deleting key: {e}")
            return False
    
    def list_keys(self) -> Dict[str, List[str]]:
        """
        List all available keys (without values).
        
        Returns:
            Dictionary of services and their key types
        """
        try:
            result = {}
            for service, service_data in self.keys.items():
                result[service] = [
                    key_type for key_type in service_data 
                    if key_type not in ['last_updated', 'updated_by']
                ]
            
            return result
            
        except Exception as e:
            logger.error(f"Error listing keys: {e}")
            return {}
    
    def rotate_keys(self, new_bot_token: str, user_id: int) -> bool:
        """
        Re-encrypt all keys with a new master key.
        
        Args:
            new_bot_token: New Telegram bot token
            user_id: User ID of the user rotating the keys
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create new encryption instance
            new_fernet = self._initialize_encryption(new_bot_token)
            
            # Update all keys with new last_updated and updated_by
            for service in self.keys:
                self.keys[service]['last_updated'] = time.time()
                self.keys[service]['updated_by'] = user_id
            
            # Save current keys
            old_keys_file = self.keys_file + '.bak'
            with open(old_keys_file, 'w') as f:
                json.dump(self.keys, f, indent=2)
            
            # Update encryption instance
            self.fernet = new_fernet
            
            # Save keys with new encryption
            return self._save_keys()
            
        except Exception as e:
            logger.error(f"Error rotating keys: {e}")
            return False
    
    def get_service_keys(self, service: str) -> Dict[str, str]:
        """
        Get all keys for a service.
        
        Args:
            service: Service name (e.g., 'mexc', 'openrouter')
            
        Returns:
            Dictionary of key types and values
        """
        try:
            if service not in self.keys:
                return {}
            
            return {
                key_type: value 
                for key_type, value in self.keys[service].items()
                if key_type not in ['last_updated', 'updated_by']
            }
            
        except Exception as e:
            logger.error(f"Error getting service keys: {e}")
            return {}
