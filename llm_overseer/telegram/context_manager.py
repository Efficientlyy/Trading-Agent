#!/usr/bin/env python
"""
Context Manager for LLM conversations.
This module manages conversation history and context for LLM-powered interactions.
"""
import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Union

logger = logging.getLogger(__name__)

class ContextManager:
    """
    Context Manager for LLM conversations.
    Manages conversation history and context for LLM-powered interactions.
    """
    
    def __init__(self, max_history: int = 10, max_age: int = 3600):
        """
        Initialize Context Manager.
        
        Args:
            max_history: Maximum number of messages to keep in history
            max_age: Maximum age of conversation history in seconds
        """
        self.max_history = max_history
        self.max_age = max_age
        self.conversations = {}  # user_id -> conversation data
        
        logger.info("Context Manager initialized")
    
    def get_conversation(self, user_id: int) -> Dict[str, Any]:
        """
        Get conversation data for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Conversation data
        """
        # Initialize conversation if not exists
        if user_id not in self.conversations:
            self.conversations[user_id] = {
                "history": [],
                "last_updated": time.time(),
                "state": {}
            }
        
        return self.conversations[user_id]
    
    def get_history(self, user_id: int) -> List[Dict[str, str]]:
        """
        Get conversation history for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Conversation history
        """
        conversation = self.get_conversation(user_id)
        
        # Check if history is expired
        if time.time() - conversation["last_updated"] > self.max_age:
            # Reset history if expired
            conversation["history"] = []
            conversation["last_updated"] = time.time()
        
        return conversation["history"]
    
    def add_message(self, user_id: int, role: str, content: str) -> None:
        """
        Add message to conversation history.
        
        Args:
            user_id: User ID
            role: Message role (user or assistant)
            content: Message content
        """
        conversation = self.get_conversation(user_id)
        history = conversation["history"]
        
        # Add message to history
        history.append({
            "role": role,
            "content": content
        })
        
        # Trim history if too long
        if len(history) > self.max_history:
            history = history[-self.max_history:]
            conversation["history"] = history
        
        # Update timestamp
        conversation["last_updated"] = time.time()
    
    def get_state(self, user_id: int) -> Dict[str, Any]:
        """
        Get conversation state for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Conversation state
        """
        conversation = self.get_conversation(user_id)
        return conversation["state"]
    
    def update_state(self, user_id: int, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update conversation state for a user.
        
        Args:
            user_id: User ID
            updates: Updates to apply to state
            
        Returns:
            Updated conversation state
        """
        conversation = self.get_conversation(user_id)
        
        # Update state
        conversation["state"].update(updates)
        
        # Update timestamp
        conversation["last_updated"] = time.time()
        
        return conversation["state"]
    
    def clear_state(self, user_id: int) -> None:
        """
        Clear conversation state for a user.
        
        Args:
            user_id: User ID
        """
        conversation = self.get_conversation(user_id)
        conversation["state"] = {}
        
        # Update timestamp
        conversation["last_updated"] = time.time()
    
    def clear_history(self, user_id: int) -> None:
        """
        Clear conversation history for a user.
        
        Args:
            user_id: User ID
        """
        conversation = self.get_conversation(user_id)
        conversation["history"] = []
        
        # Update timestamp
        conversation["last_updated"] = time.time()
    
    def clear_conversation(self, user_id: int) -> None:
        """
        Clear entire conversation for a user.
        
        Args:
            user_id: User ID
        """
        if user_id in self.conversations:
            del self.conversations[user_id]
