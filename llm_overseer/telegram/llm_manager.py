#!/usr/bin/env python
"""
LLM Manager for OpenRouter API integration.
This module handles communication with the OpenRouter API for natural language understanding.
"""
import os
import json
import time
import logging
import requests
from typing import Dict, List, Any, Optional, Tuple, Union

logger = logging.getLogger(__name__)

class LLMManager:
    """
    LLM Manager for OpenRouter API integration.
    Handles communication with the OpenRouter API for natural language understanding.
    """
    
    def __init__(self, api_key: str = None, key_manager = None):
        """
        Initialize LLM Manager.
        
        Args:
            api_key: OpenRouter API key (optional, can be retrieved from key_manager)
            key_manager: Key Manager instance for retrieving API key (optional)
        """
        self.api_key = api_key
        self.key_manager = key_manager
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.default_model = "anthropic/claude-3-opus-20240229"
        self.fallback_model = "openai/gpt-4-turbo"
        
        logger.info("LLM Manager initialized")
    
    def _get_api_key(self) -> Optional[str]:
        """
        Get OpenRouter API key.
        
        Returns:
            API key or None if not available
        """
        if self.api_key:
            return self.api_key
        
        if self.key_manager:
            api_key = self.key_manager.get_key("openrouter", "api_key")
            if api_key:
                self.api_key = api_key
                return api_key
        
        return os.environ.get("OPENROUTER_API_KEY")
    
    def _create_system_prompt(self) -> str:
        """
        Create system prompt for LLM.
        
        Returns:
            System prompt
        """
        return """
You are KeyGuardian, an AI assistant integrated into a Telegram bot for the Trading-Agent system. Your purpose is to help users manage their API keys securely using natural language.

Your capabilities:
1. Set API keys for various services
2. Retrieve API keys when requested
3. List available keys
4. Delete API keys

Security rules:
1. Only respond to key management requests from authenticated users
2. Never suggest or generate fake API keys
3. Always confirm before executing destructive actions like deletion
4. Maintain user privacy and data confidentiality

Available services and key types:
- MEXC Exchange: api_key, api_secret
- OpenRouter: api_key
- Telegram: bot_token, chat_id
- Render: api_token

For each user message, determine:
1. The intent (set_key, get_key, list_keys, delete_key, or other)
2. For key operations, identify the service and key_type
3. For set_key operations, identify the value to be set

Respond in a helpful, conversational manner while extracting the necessary information for the key management system to execute the request.

IMPORTANT: Always include structured metadata at the end of your response in the following format:

<key_management_metadata>
{
  "intent": "set_key|get_key|list_keys|delete_key|help|unknown",
  "service": "mexc|openrouter|telegram|render|null",
  "key_type": "api_key|api_secret|bot_token|chat_id|api_token|null",
  "value": "actual_value_or_null",
  "confidence": 0.95,
  "requires_confirmation": true|false
}
</key_management_metadata>
"""
    
    def _format_conversation_history(self, history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Format conversation history for LLM.
        
        Args:
            history: List of message dictionaries with 'role' and 'content'
            
        Returns:
            Formatted conversation history
        """
        formatted_history = []
        
        # Add system message
        formatted_history.append({
            "role": "system",
            "content": self._create_system_prompt()
        })
        
        # Add conversation history
        for message in history:
            formatted_history.append({
                "role": message["role"],
                "content": message["content"]
            })
        
        return formatted_history
    
    def _extract_metadata(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Extract key management metadata from LLM response.
        
        Args:
            response: LLM response text
            
        Returns:
            Extracted metadata or None if not found
        """
        try:
            # Extract metadata between tags
            start_tag = "<key_management_metadata>"
            end_tag = "</key_management_metadata>"
            
            start_index = response.find(start_tag)
            end_index = response.find(end_tag)
            
            if start_index == -1 or end_index == -1:
                logger.warning("Metadata tags not found in LLM response")
                return None
            
            # Extract JSON string
            metadata_json = response[start_index + len(start_tag):end_index].strip()
            
            # Parse JSON
            metadata = json.loads(metadata_json)
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata from LLM response: {e}")
            return None
    
    def _clean_response(self, response: str) -> str:
        """
        Clean LLM response by removing metadata.
        
        Args:
            response: LLM response text
            
        Returns:
            Cleaned response
        """
        try:
            # Remove metadata section
            start_tag = "<key_management_metadata>"
            end_tag = "</key_management_metadata>"
            
            start_index = response.find(start_tag)
            
            if start_index != -1:
                return response[:start_index].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Error cleaning LLM response: {e}")
            return response
    
    async def process_message(self, user_message: str, conversation_history: List[Dict[str, str]] = None) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        Process user message with LLM.
        
        Args:
            user_message: User message text
            conversation_history: Previous conversation history (optional)
            
        Returns:
            Tuple of (cleaned response text, metadata)
        """
        api_key = self._get_api_key()
        if not api_key:
            logger.error("OpenRouter API key not found")
            return "I'm sorry, but I can't process natural language requests right now. Please try again later or use the command-based interface.", None
        
        # Initialize conversation history if not provided
        if conversation_history is None:
            conversation_history = []
        
        # Add user message to history
        conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Format conversation history
        messages = self._format_conversation_history(conversation_history)
        
        try:
            # Prepare request
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.default_model,
                "messages": messages,
                "temperature": 0.2,
                "max_tokens": 1024,
                "top_p": 0.95
            }
            
            # Send request
            response = requests.post(
                self.api_url,
                headers=headers,
                json=data,
                timeout=30
            )
            
            # Check response
            if response.status_code == 200:
                result = response.json()
                response_text = result["choices"][0]["message"]["content"]
                
                # Extract metadata
                metadata = self._extract_metadata(response_text)
                
                # Clean response
                cleaned_response = self._clean_response(response_text)
                
                # Add assistant response to history
                conversation_history.append({
                    "role": "assistant",
                    "content": cleaned_response
                })
                
                return cleaned_response, metadata
            else:
                logger.error(f"Error from OpenRouter API: {response.status_code} {response.text}")
                
                # Try fallback model if primary model fails
                if self.default_model != self.fallback_model:
                    logger.info(f"Trying fallback model: {self.fallback_model}")
                    
                    # Update model in request
                    data["model"] = self.fallback_model
                    
                    # Send request with fallback model
                    response = requests.post(
                        self.api_url,
                        headers=headers,
                        json=data,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        response_text = result["choices"][0]["message"]["content"]
                        
                        # Extract metadata
                        metadata = self._extract_metadata(response_text)
                        
                        # Clean response
                        cleaned_response = self._clean_response(response_text)
                        
                        # Add assistant response to history
                        conversation_history.append({
                            "role": "assistant",
                            "content": cleaned_response
                        })
                        
                        return cleaned_response, metadata
                
                return "I'm sorry, but I encountered an error processing your request. Please try again later.", None
                
        except Exception as e:
            logger.error(f"Error processing message with LLM: {e}")
            return "I'm sorry, but I encountered an error processing your request. Please try again later.", None
