#!/usr/bin/env python
"""
Natural Language Handler for Telegram bot API key management.
This module implements natural language processing for key management interactions.
"""
import re
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

logger = logging.getLogger(__name__)

class NaturalLanguageHandler:
    """
    Natural Language Handler for Telegram bot API key management.
    Processes natural language requests for key management.
    """
    
    # Service name mappings and aliases
    SERVICE_MAPPINGS = {
        "mexc": ["mexc", "mexc exchange", "mexc trading"],
        "openrouter": ["openrouter", "open router", "openai", "llm", "language model"],
        "telegram": ["telegram", "telegram bot", "bot", "notification"],
        "render": ["render", "render platform", "hosting", "deployment"]
    }
    
    # Key type mappings and aliases
    KEY_TYPE_MAPPINGS = {
        "api_key": ["api key", "key", "access key", "access token", "token"],
        "api_secret": ["api secret", "secret", "secret key", "private key"],
        "bot_token": ["bot token", "telegram token", "telegram bot token"],
        "chat_id": ["chat id", "telegram id", "user id", "telegram user id"],
        "api_token": ["api token", "access token", "token"]
    }
    
    # Action patterns
    SET_PATTERNS = [
        r"(?:set|update|change|store|save|add|put|configure|use)(?: my| the)? (?P<service>.*?) (?P<key_type>.*?)(?: to| as| with| value| is| =| :)? (?P<value>\S+)",
        r"(?:my|the) (?P<service>.*?) (?P<key_type>.*?)(?: is| =| :| should be) (?P<value>\S+)",
        r"(?:use|save|store|add) (?P<value>\S+)(?: as| for)(?: my| the)? (?P<service>.*?) (?P<key_type>.*)"
    ]
    
    GET_PATTERNS = [
        r"(?:get|show|display|what is|tell me)(?: my| the)? (?P<service>.*?) (?P<key_type>.*?)(?:\?)?$",
        r"(?:what|show me|tell me|display)(?: is| are)(?: my| the)? (?P<service>.*?) (?P<key_type>.*?)(?:\?)?$"
    ]
    
    LIST_PATTERNS = [
        r"(?:list|show|display|what|tell me)(?: all| the| my)? (?:available |configured )?keys",
        r"what keys(?: do I have| are available| are configured| are set)",
        r"show me(?: all| my)? keys"
    ]
    
    DELETE_PATTERNS = [
        r"(?:delete|remove|clear)(?: my| the)? (?P<service>.*?) (?P<key_type>.*?)(?:\?)?$",
        r"(?:get rid of|eliminate|erase)(?: my| the)? (?P<service>.*?) (?P<key_type>.*?)(?:\?)?$"
    ]
    
    HELP_PATTERNS = [
        r"(?:help|assist|support|guide|how to|how do I|instructions)",
        r"(?:how|what|tell me how) (?:can|do) I (?:use|set|get|manage|configure) (?:keys|api keys)"
    ]
    
    def __init__(self, key_manager=None):
        """
        Initialize Natural Language Handler.
        
        Args:
            key_manager: Key Manager instance (optional, can be set later)
        """
        self.key_manager = key_manager
        self.conversation_state = {}  # Store conversation state by user_id
        logger.info("Natural Language Handler initialized")
    
    def set_key_manager(self, key_manager) -> None:
        """
        Set Key Manager instance.
        
        Args:
            key_manager: Key Manager instance
        """
        self.key_manager = key_manager
    
    def _normalize_service_name(self, service_text: str) -> Optional[str]:
        """
        Normalize service name from natural language input.
        
        Args:
            service_text: Service name from natural language
            
        Returns:
            Normalized service name or None if not recognized
        """
        service_text = service_text.lower().strip()
        
        for service, aliases in self.SERVICE_MAPPINGS.items():
            if any(alias in service_text for alias in aliases):
                return service
        
        return None
    
    def _normalize_key_type(self, key_type_text: str) -> Optional[str]:
        """
        Normalize key type from natural language input.
        
        Args:
            key_type_text: Key type from natural language
            
        Returns:
            Normalized key type or None if not recognized
        """
        key_type_text = key_type_text.lower().strip()
        
        for key_type, aliases in self.KEY_TYPE_MAPPINGS.items():
            if any(alias in key_type_text for alias in aliases):
                return key_type
        
        return None
    
    def _extract_set_key_info(self, text: str) -> Optional[Dict[str, str]]:
        """
        Extract service, key type, and value from set key request.
        
        Args:
            text: Natural language text
            
        Returns:
            Dictionary with service, key_type, and value, or None if not recognized
        """
        for pattern in self.SET_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                service_text = match.group("service")
                key_type_text = match.group("key_type")
                value = match.group("value")
                
                service = self._normalize_service_name(service_text)
                key_type = self._normalize_key_type(key_type_text)
                
                if service and key_type:
                    return {
                        "service": service,
                        "key_type": key_type,
                        "value": value
                    }
        
        return None
    
    def _extract_get_key_info(self, text: str) -> Optional[Dict[str, str]]:
        """
        Extract service and key type from get key request.
        
        Args:
            text: Natural language text
            
        Returns:
            Dictionary with service and key_type, or None if not recognized
        """
        for pattern in self.GET_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                service_text = match.group("service")
                key_type_text = match.group("key_type")
                
                service = self._normalize_service_name(service_text)
                key_type = self._normalize_key_type(key_type_text)
                
                if service and key_type:
                    return {
                        "service": service,
                        "key_type": key_type
                    }
        
        return None
    
    def _extract_delete_key_info(self, text: str) -> Optional[Dict[str, str]]:
        """
        Extract service and key type from delete key request.
        
        Args:
            text: Natural language text
            
        Returns:
            Dictionary with service and key_type, or None if not recognized
        """
        for pattern in self.DELETE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                service_text = match.group("service")
                key_type_text = match.group("key_type")
                
                service = self._normalize_service_name(service_text)
                key_type = self._normalize_key_type(key_type_text)
                
                if service and key_type:
                    return {
                        "service": service,
                        "key_type": key_type
                    }
        
        return None
    
    def _is_list_keys_request(self, text: str) -> bool:
        """
        Check if text is a list keys request.
        
        Args:
            text: Natural language text
            
        Returns:
            True if text is a list keys request, False otherwise
        """
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in self.LIST_PATTERNS)
    
    def _is_help_request(self, text: str) -> bool:
        """
        Check if text is a help request.
        
        Args:
            text: Natural language text
            
        Returns:
            True if text is a help request, False otherwise
        """
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in self.HELP_PATTERNS)
    
    def _create_service_selection_keyboard(self) -> InlineKeyboardMarkup:
        """
        Create keyboard for service selection.
        
        Returns:
            InlineKeyboardMarkup with service buttons
        """
        keyboard = []
        for service in self.SERVICE_MAPPINGS.keys():
            keyboard.append([InlineKeyboardButton(service.capitalize(), callback_data=f"service_{service}")])
        
        return InlineKeyboardMarkup(keyboard)
    
    def _create_key_type_selection_keyboard(self, service: str) -> InlineKeyboardMarkup:
        """
        Create keyboard for key type selection.
        
        Args:
            service: Service name
            
        Returns:
            InlineKeyboardMarkup with key type buttons
        """
        keyboard = []
        
        # Determine appropriate key types for the service
        if service == "mexc":
            key_types = ["api_key", "api_secret"]
        elif service == "openrouter":
            key_types = ["api_key"]
        elif service == "telegram":
            key_types = ["bot_token", "chat_id"]
        elif service == "render":
            key_types = ["api_token"]
        else:
            key_types = ["api_key", "api_secret"]
        
        for key_type in key_types:
            display_name = key_type.replace("_", " ").title()
            keyboard.append([InlineKeyboardButton(display_name, callback_data=f"key_type_{key_type}")])
        
        # Add cancel button
        keyboard.append([InlineKeyboardButton("Cancel", callback_data="cancel")])
        
        return InlineKeyboardMarkup(keyboard)
    
    def _create_confirmation_keyboard(self, action: str, service: str, key_type: str) -> InlineKeyboardMarkup:
        """
        Create keyboard for confirmation.
        
        Args:
            action: Action to confirm (e.g., "delete")
            service: Service name
            key_type: Key type
            
        Returns:
            InlineKeyboardMarkup with confirmation buttons
        """
        keyboard = [
            [
                InlineKeyboardButton("Yes", callback_data=f"{action}_confirm_{service}_{key_type}"),
                InlineKeyboardButton("No", callback_data=f"{action}_cancel")
            ]
        ]
        
        return InlineKeyboardMarkup(keyboard)
    
    def start_key_setting_flow(self, user_id: int) -> Dict[str, Any]:
        """
        Start key setting conversation flow.
        
        Args:
            user_id: User ID
            
        Returns:
            Initial conversation state
        """
        state = {
            "flow": "set_key",
            "step": "select_service",
            "service": None,
            "key_type": None,
            "value": None
        }
        
        self.conversation_state[user_id] = state
        return state
    
    def start_key_getting_flow(self, user_id: int) -> Dict[str, Any]:
        """
        Start key getting conversation flow.
        
        Args:
            user_id: User ID
            
        Returns:
            Initial conversation state
        """
        state = {
            "flow": "get_key",
            "step": "select_service",
            "service": None,
            "key_type": None
        }
        
        self.conversation_state[user_id] = state
        return state
    
    def start_key_deleting_flow(self, user_id: int) -> Dict[str, Any]:
        """
        Start key deleting conversation flow.
        
        Args:
            user_id: User ID
            
        Returns:
            Initial conversation state
        """
        state = {
            "flow": "delete_key",
            "step": "select_service",
            "service": None,
            "key_type": None
        }
        
        self.conversation_state[user_id] = state
        return state
    
    def get_conversation_state(self, user_id: int) -> Optional[Dict[str, Any]]:
        """
        Get conversation state for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Conversation state or None if not in conversation
        """
        return self.conversation_state.get(user_id)
    
    def update_conversation_state(self, user_id: int, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update conversation state for a user.
        
        Args:
            user_id: User ID
            updates: Updates to apply to state
            
        Returns:
            Updated conversation state
        """
        if user_id not in self.conversation_state:
            self.conversation_state[user_id] = {}
        
        self.conversation_state[user_id].update(updates)
        return self.conversation_state[user_id]
    
    def clear_conversation_state(self, user_id: int) -> None:
        """
        Clear conversation state for a user.
        
        Args:
            user_id: User ID
        """
        if user_id in self.conversation_state:
            del self.conversation_state[user_id]
    
    async def process_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> Tuple[str, Optional[InlineKeyboardMarkup]]:
        """
        Process natural language message.
        
        Args:
            update: Telegram update
            context: Telegram context
            
        Returns:
            Tuple of (response text, optional keyboard markup)
        """
        user_id = update.effective_user.id
        text = update.message.text
        
        # Check if user is in a conversation flow
        state = self.get_conversation_state(user_id)
        if state:
            return await self._continue_conversation(update, context, state)
        
        # Try to identify intent from natural language
        if set_key_info := self._extract_set_key_info(text):
            # Direct set key request
            service = set_key_info["service"]
            key_type = set_key_info["key_type"]
            value = set_key_info["value"]
            
            if self.key_manager:
                success = self.key_manager.set_key(service, key_type, value, user_id)
                
                if success:
                    # Log the action
                    logger.info(f"User {user_id} set {service}.{key_type} via natural language")
                    
                    # Delete the message containing the key for security
                    await update.message.delete()
                    
                    return f"I've set your {service} {key_type.replace('_', ' ')} successfully.", None
                else:
                    return f"I couldn't set your {service} {key_type.replace('_', ' ')}. Please try again.", None
            else:
                return "I'm sorry, but I can't manage keys right now. Please try again later.", None
                
        elif get_key_info := self._extract_get_key_info(text):
            # Direct get key request
            service = get_key_info["service"]
            key_type = get_key_info["key_type"]
            
            if self.key_manager:
                value = self.key_manager.get_key(service, key_type)
                
                if value:
                    # Log the action
                    logger.info(f"User {user_id} retrieved {service}.{key_type} via natural language")
                    
                    return f"Your {service} {key_type.replace('_', ' ')} is: {value}", None
                else:
                    return f"I couldn't find your {service} {key_type.replace('_', ' ')}. Would you like to set it?", self._create_confirmation_keyboard("set", service, key_type)
            else:
                return "I'm sorry, but I can't retrieve keys right now. Please try again later.", None
                
        elif delete_key_info := self._extract_delete_key_info(text):
            # Direct delete key request
            service = delete_key_info["service"]
            key_type = delete_key_info["key_type"]
            
            return f"Are you sure you want to delete your {service} {key_type.replace('_', ' ')}?", self._create_confirmation_keyboard("delete", service, key_type)
            
        elif self._is_list_keys_request(text):
            # List keys request
            if self.key_manager:
                keys = self.key_manager.list_keys()
                
                if keys:
                    # Format keys
                    keys_text = "Here are your available keys:\n\n"
                    for service, key_types in keys.items():
                        keys_text += f"{service.capitalize()}:\n"
                        for key_type in key_types:
                            keys_text += f"  - {key_type.replace('_', ' ').title()}\n"
                        keys_text += "\n"
                    
                    # Log the action
                    logger.info(f"User {user_id} listed keys via natural language")
                    
                    return keys_text, None
                else:
                    return "You don't have any keys set up yet. Would you like to set one up now?", self._create_service_selection_keyboard()
            else:
                return "I'm sorry, but I can't list keys right now. Please try again later.", None
                
        elif self._is_help_request(text):
            # Help request
            help_text = (
                "I can help you manage your API keys securely. Here's what you can ask me to do:\n\n"
                "To set a key:\n"
                "- \"Set my MEXC API key to abc123\"\n"
                "- \"My OpenRouter API key is xyz789\"\n"
                "- \"Update Telegram bot token to 123:ABC\"\n\n"
                "To get a key:\n"
                "- \"What's my MEXC API key?\"\n"
                "- \"Show me my OpenRouter API key\"\n"
                "- \"Get my Telegram chat ID\"\n\n"
                "To list all keys:\n"
                "- \"List all my keys\"\n"
                "- \"What keys do I have?\"\n"
                "- \"Show me my available keys\"\n\n"
                "To delete a key:\n"
                "- \"Delete my MEXC API key\"\n"
                "- \"Remove my OpenRouter API key\"\n\n"
                "Or you can just tap a button below to start:"
            )
            
            # Create keyboard with options
            keyboard = [
                [InlineKeyboardButton("Set a key", callback_data="flow_set_key")],
                [InlineKeyboardButton("Get a key", callback_data="flow_get_key")],
                [InlineKeyboardButton("List all keys", callback_data="flow_list_keys")],
                [InlineKeyboardButton("Delete a key", callback_data="flow_delete_key")]
            ]
            
            return help_text, InlineKeyboardMarkup(keyboard)
            
        else:
            # Unknown intent
            help_text = (
                "I'm not sure what you're asking about API keys. Here are some things you can say:\n\n"
                "- \"Set my MEXC API key to abc123\"\n"
                "- \"What's my OpenRouter API key?\"\n"
                "- \"List all my keys\"\n"
                "- \"Delete my Telegram bot token\"\n\n"
                "Or you can tap a button below to start:"
            )
            
            # Create keyboard with options
            keyboard = [
                [InlineKeyboardButton("Set a key", callback_data="flow_set_key")],
                [InlineKeyboardButton("Get a key", callback_data="flow_get_key")],
                [InlineKeyboardButton("List all keys", callback_data="flow_list_keys")],
                [InlineKeyboardButton("Delete a key", callback_data="flow_delete_key")]
            ]
            
            return help_text, InlineKeyboardMarkup(keyboard)
    
    async def _continue_conversation(self, update: Update, context: ContextTypes.DEFAULT_TYPE, state: Dict[str, Any]) -> Tuple[str, Optional[InlineKeyboardMarkup]]:
        """
        Continue conversation based on state.
        
        Args:
            update: Telegram update
            context: Telegram context
            state: Conversation state
            
        Returns:
            Tuple of (response text, optional keyboard markup)
        """
        user_id = update.effective_user.id
        text = update.message.text
        
        flow = state["flow"]
        step = state["step"]
        
        if flow == "set_key":
            if step == "select_service":
                # Try to identify service from text
                service = self._normalize_service_name(text)
                if service:
                    # Update state
                    self.update_conversation_state(user_id, {
                        "step": "select_key_type",
                        "service": service
                    })
                    
                    return f"What type of key do you want to set for {service.capitalize()}?", self._create_key_type_selection_keyboard(service)
                else:
                    return "I didn't recognize that service. Please select one:", self._create_service_selection_keyboard()
                    
            elif step == "select_key_type":
                # Try to identify key type from text
                key_type = self._normalize_key_type(text)
                if key_type:
                    # Update state
                    self.update_conversation_state(user_id, {
                        "step": "enter_value",
                        "key_type": key_type
                    })
                    
                    service = state["service"]
                    return f"Please enter your {service} {key_type.replace('_', ' ')}:", None
                else:
                    service = state["service"]
                    return f"I didn't recognize that key type. Please select one for {service.capitalize()}:", self._create_key_type_selection_keyboard(service)
                    
            elif step == "enter_value":
                # Get value from text
                value = text.strip()
                
                # Update state
                self.update_conversation_state(user_id, {
                    "step": "confirm",
                    "value": value
                })
                
                service = state["service"]
                key_type = state["key_type"]
                
                return f"You want to set your {service} {key_type.replace('_', ' ')} to {value}. Is that correct?", self._create_confirmation_keyboard("set", service, key_type)
                
        elif flow == "get_key":
            if step == "select_service":
                # Try to identify service from text
                service = self._normalize_service_name(text)
                if service:
                    # Update state
                    self.update_conversation_state(user_id, {
                        "step": "select_key_type",
                        "service": service
                    })
                    
                    return f"What type of key do you want to get for {service.capitalize()}?", self._create_key_type_selection_keyboard(service)
                else:
                    return "I didn't recognize that service. Please select one:", self._create_service_selection_keyboard()
                    
            elif step == "select_key_type":
                # Try to identify key type from text
                key_type = self._normalize_key_type(text)
                if key_type:
                    # Clear conversation state
                    self.clear_conversation_state(user_id)
                    
                    service = state["service"]
                    
                    if self.key_manager:
                        value = self.key_manager.get_key(service, key_type)
                        
                        if value:
                            # Log the action
                            logger.info(f"User {user_id} retrieved {service}.{key_type} via conversation")
                            
                            return f"Your {service} {key_type.replace('_', ' ')} is: {value}", None
                        else:
                            return f"I couldn't find your {service} {key_type.replace('_', ' ')}. Would you like to set it?", self._create_confirmation_keyboard("set", service, key_type)
                    else:
                        return "I'm sorry, but I can't retrieve keys right now. Please try again later.", None
                else:
                    service = state["service"]
                    return f"I didn't recognize that key type. Please select one for {service.capitalize()}:", self._create_key_type_selection_keyboard(service)
                    
        elif flow == "delete_key":
            if step == "select_service":
                # Try to identify service from text
                service = self._normalize_service_name(text)
                if service:
                    # Update state
                    self.update_conversation_state(user_id, {
                        "step": "select_key_type",
                        "service": service
                    })
                    
                    return f"What type of key do you want to delete for {service.capitalize()}?", self._create_key_type_selection_keyboard(service)
                else:
                    return "I didn't recognize that service. Please select one:", self._create_service_selection_keyboard()
                    
            elif step == "select_key_type":
                # Try to identify key type from text
                key_type = self._normalize_key_type(text)
                if key_type:
                    # Update state
                    self.update_conversation_state(user_id, {
                        "step": "confirm",
                        "key_type": key_type
                    })
                    
                    service = state["service"]
                    
                    return f"Are you sure you want to delete your {service} {key_type.replace('_', ' ')}?", self._create_confirmation_keyboard("delete", service, key_type)
                else:
                    service = state["service"]
                    return f"I didn't recognize that key type. Please select one for {service.capitalize()}:", self._create_key_type_selection_keyboard(service)
        
        # Default response for unknown state
        self.clear_conversation_state(user_id)
        return "I'm not sure what you're asking. Can you try again?", None
    
    async def process_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> Tuple[str, Optional[InlineKeyboardMarkup]]:
        """
        Process callback query.
        
        Args:
            update: Telegram update
            context: Telegram context
            
        Returns:
            Tuple of (response text, optional keyboard markup)
        """
        query = update.callback_query
        user_id = query.from_user.id
        data = query.data
        
        # Flow selection callbacks
        if data.startswith("flow_"):
            flow = data[5:]  # Remove "flow_" prefix
            
            if flow == "set_key":
                state = self.start_key_setting_flow(user_id)
                return "Which service do you want to set a key for?", self._create_service_selection_keyboard()
                
            elif flow == "get_key":
                state = self.start_key_getting_flow(user_id)
                return "Which service do you want to get a key for?", self._create_service_selection_keyboard()
                
            elif flow == "list_keys":
                if self.key_manager:
                    keys = self.key_manager.list_keys()
                    
                    if keys:
                        # Format keys
                        keys_text = "Here are your available keys:\n\n"
                        for service, key_types in keys.items():
                            keys_text += f"{service.capitalize()}:\n"
                            for key_type in key_types:
                                keys_text += f"  - {key_type.replace('_', ' ').title()}\n"
                            keys_text += "\n"
                        
                        # Log the action
                        logger.info(f"User {user_id} listed keys via callback")
                        
                        return keys_text, None
                    else:
                        return "You don't have any keys set up yet. Would you like to set one up now?", self._create_service_selection_keyboard()
                else:
                    return "I'm sorry, but I can't list keys right now. Please try again later.", None
                    
            elif flow == "delete_key":
                state = self.start_key_deleting_flow(user_id)
                return "Which service do you want to delete a key for?", self._create_service_selection_keyboard()
        
        # Service selection callbacks
        elif data.startswith("service_"):
            service = data[8:]  # Remove "service_" prefix
            
            # Get current state
            state = self.get_conversation_state(user_id)
            if not state:
                return "I'm not sure what you're trying to do. Can you try again?", None
            
            # Update state
            self.update_conversation_state(user_id, {
                "step": "select_key_type",
                "service": service
            })
            
            return f"What type of key do you want to {state['flow'].split('_')[0]} for {service.capitalize()}?", self._create_key_type_selection_keyboard(service)
            
        # Key type selection callbacks
        elif data.startswith("key_type_"):
            key_type = data[9:]  # Remove "key_type_" prefix
            
            # Get current state
            state = self.get_conversation_state(user_id)
            if not state:
                return "I'm not sure what you're trying to do. Can you try again?", None
            
            flow = state["flow"]
            service = state["service"]
            
            if flow == "set_key":
                # Update state
                self.update_conversation_state(user_id, {
                    "step": "enter_value",
                    "key_type": key_type
                })
                
                return f"Please enter your {service} {key_type.replace('_', ' ')}:", None
                
            elif flow == "get_key":
                # Clear conversation state
                self.clear_conversation_state(user_id)
                
                if self.key_manager:
                    value = self.key_manager.get_key(service, key_type)
                    
                    if value:
                        # Log the action
                        logger.info(f"User {user_id} retrieved {service}.{key_type} via callback")
                        
                        return f"Your {service} {key_type.replace('_', ' ')} is: {value}", None
                    else:
                        return f"I couldn't find your {service} {key_type.replace('_', ' ')}. Would you like to set it?", self._create_confirmation_keyboard("set", service, key_type)
                else:
                    return "I'm sorry, but I can't retrieve keys right now. Please try again later.", None
                    
            elif flow == "delete_key":
                # Update state
                self.update_conversation_state(user_id, {
                    "step": "confirm",
                    "key_type": key_type
                })
                
                return f"Are you sure you want to delete your {service} {key_type.replace('_', ' ')}?", self._create_confirmation_keyboard("delete", service, key_type)
        
        # Confirmation callbacks
        elif data.startswith("set_confirm_"):
            # Extract service and key_type
            _, _, service, key_type = data.split("_", 3)
            
            # Get current state
            state = self.get_conversation_state(user_id)
            
            # Check if we're in a conversation
            if state and state["flow"] == "set_key" and state["step"] == "confirm":
                value = state["value"]
                
                if self.key_manager:
                    success = self.key_manager.set_key(service, key_type, value, user_id)
                    
                    # Clear conversation state
                    self.clear_conversation_state(user_id)
                    
                    if success:
                        # Log the action
                        logger.info(f"User {user_id} set {service}.{key_type} via conversation")
                        
                        return f"I've set your {service} {key_type.replace('_', ' ')} successfully.", None
                    else:
                        return f"I couldn't set your {service} {key_type.replace('_', ' ')}. Please try again.", None
                else:
                    # Clear conversation state
                    self.clear_conversation_state(user_id)
                    
                    return "I'm sorry, but I can't manage keys right now. Please try again later.", None
            else:
                # Start a new conversation to set the key
                state = self.start_key_setting_flow(user_id)
                self.update_conversation_state(user_id, {
                    "step": "enter_value",
                    "service": service,
                    "key_type": key_type
                })
                
                return f"Please enter your {service} {key_type.replace('_', ' ')}:", None
                
        elif data.startswith("delete_confirm_"):
            # Extract service and key_type
            _, _, service, key_type = data.split("_", 3)
            
            if self.key_manager:
                success = self.key_manager.delete_key(service, key_type)
                
                # Clear conversation state
                self.clear_conversation_state(user_id)
                
                if success:
                    # Log the action
                    logger.info(f"User {user_id} deleted {service}.{key_type} via callback")
                    
                    return f"I've deleted your {service} {key_type.replace('_', ' ')} successfully.", None
                else:
                    return f"I couldn't delete your {service} {key_type.replace('_', ' ')}. It might not exist.", None
            else:
                # Clear conversation state
                self.clear_conversation_state(user_id)
                
                return "I'm sorry, but I can't manage keys right now. Please try again later.", None
                
        elif data == "set_cancel" or data == "delete_cancel":
            # Clear conversation state
            self.clear_conversation_state(user_id)
            
            return "Operation cancelled.", None
            
        elif data == "cancel":
            # Clear conversation state
            self.clear_conversation_state(user_id)
            
            return "Operation cancelled.", None
        
        # Default response for unknown callback
        return "I'm not sure what you're trying to do. Can you try again?", None
