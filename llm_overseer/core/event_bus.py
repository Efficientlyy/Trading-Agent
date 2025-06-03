#!/usr/bin/env python
"""
Event Bus System for LLM Strategic Overseer

This module provides a publish-subscribe event system for coordinating updates
across components in the LLM Strategic Overseer system.
"""

import os
import sys
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("event_bus.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("event_bus")

class EventBus:
    """
    Event Bus for publish-subscribe messaging.
    
    This class provides a topic-based message routing system with
    event prioritization, asynchronous event handling, and error recovery.
    """
    
    def __init__(self):
        """Initialize Event Bus."""
        # Topic-based subscribers
        self.subscribers = defaultdict(list)
        
        # Event prioritization
        self.priority_levels = {
            "emergency": 0,  # Highest priority
            "high": 1,
            "normal": 2,
            "low": 3        # Lowest priority
        }
        
        # Event queue
        self.event_queue = asyncio.Queue()
        
        # Event processing flag
        self.processing = False
        
        # Event processing task
        self.processing_task = None
        
        logger.info("Event Bus initialized")
    
    def subscribe(self, topic: str, callback: Callable[[str, Dict[str, Any]], None]) -> int:
        """
        Subscribe to a topic.
        
        Args:
            topic: Topic to subscribe to
            callback: Callback function to handle events
            
        Returns:
            Subscription ID
        """
        subscription_id = len(self.subscribers[topic])
        self.subscribers[topic].append(callback)
        
        logger.info(f"New subscriber to topic '{topic}': {subscription_id}")
        return subscription_id
    
    def unsubscribe(self, topic: str, subscription_id: int) -> bool:
        """
        Unsubscribe from a topic.
        
        Args:
            topic: Topic to unsubscribe from
            subscription_id: Subscription ID
            
        Returns:
            True if unsubscribed successfully, False otherwise
        """
        if topic not in self.subscribers:
            logger.warning(f"Unknown topic: {topic}")
            return False
        
        if subscription_id >= len(self.subscribers[topic]):
            logger.warning(f"Invalid subscription ID: {subscription_id}")
            return False
        
        self.subscribers[topic][subscription_id] = None
        logger.info(f"Unsubscribed from topic '{topic}': {subscription_id}")
        return True
    
    async def publish(self, topic: str, data: Dict[str, Any], priority: str = "normal") -> None:
        """
        Publish event to topic.
        
        Args:
            topic: Topic to publish to
            data: Event data
            priority: Event priority ("emergency", "high", "normal", "low")
        """
        # Validate priority
        if priority not in self.priority_levels:
            logger.warning(f"Invalid priority: {priority}, using 'normal'")
            priority = "normal"
        
        # Add timestamp if not present
        if "timestamp" not in data:
            data["timestamp"] = datetime.now().isoformat()
        
        # Create event
        event = {
            "topic": topic,
            "data": data,
            "priority": priority,
            "timestamp": data["timestamp"]
        }
        
        # Add to queue with priority
        priority_value = self.priority_levels[priority]
        
        # Use a tuple with a unique identifier as the third element to avoid comparing dicts
        await self.event_queue.put((priority_value, id(event), event))
        
        logger.info(f"Published event to topic '{topic}' with priority '{priority}'")
        
        # Start processing if not already running
        if not self.processing:
            self.start_processing()
    
    def start_processing(self) -> None:
        """Start event processing."""
        if self.processing:
            return
        
        self.processing = True
        self.processing_task = asyncio.create_task(self._process_events())
        
        logger.info("Event processing started")
    
    def stop_processing(self) -> None:
        """Stop event processing."""
        if not self.processing:
            return
        
        self.processing = False
        if self.processing_task:
            self.processing_task.cancel()
        
        logger.info("Event processing stopped")
    
    async def _process_events(self) -> None:
        """Process events from queue."""
        try:
            while self.processing:
                # Get event from queue
                priority, _, event = await self.event_queue.get()
                
                # Process event
                await self._process_event(event)
                
                # Mark task as done
                self.event_queue.task_done()
        except asyncio.CancelledError:
            logger.info("Event processing cancelled")
        except Exception as e:
            logger.error(f"Error in event processing: {e}")
            # Restart processing after error
            self.processing = False
            self.start_processing()
    
    async def _process_event(self, event: Dict[str, Any]) -> None:
        """
        Process single event.
        
        Args:
            event: Event to process
        """
        topic = event["topic"]
        data = event["data"]
        priority = event["priority"]
        
        # Check if topic has subscribers
        if topic not in self.subscribers or not self.subscribers[topic]:
            logger.debug(f"No subscribers for topic '{topic}'")
            return
        
        # Notify subscribers
        for callback in self.subscribers[topic]:
            if callback:
                try:
                    # Call callback with topic and data
                    await asyncio.create_task(self._call_callback(callback, topic, data))
                except Exception as e:
                    logger.error(f"Error in subscriber callback: {e}")
    
    async def _call_callback(self, callback: Callable, topic: str, data: Dict[str, Any]) -> None:
        """
        Call subscriber callback safely.
        
        Args:
            callback: Callback function
            topic: Event topic
            data: Event data
        """
        try:
            # Check if callback is coroutine function
            if asyncio.iscoroutinefunction(callback):
                await callback(topic, data)
            else:
                callback(topic, data)
        except Exception as e:
            logger.error(f"Error in callback for topic '{topic}': {e}")


# For testing
async def test():
    """Test function."""
    # Create event bus
    event_bus = EventBus()
    
    # Define callback
    async def print_event(topic, data):
        print(f"Received event on topic '{topic}': {data}")
    
    # Subscribe to topic
    event_bus.subscribe("test.topic", print_event)
    
    # Publish events with different priorities
    await event_bus.publish("test.topic", {"message": "Normal priority event"}, "normal")
    await event_bus.publish("test.topic", {"message": "High priority event"}, "high")
    await event_bus.publish("test.topic", {"message": "Emergency event"}, "emergency")
    
    # Wait for events to be processed
    await asyncio.sleep(1)
    
    # Stop processing
    event_bus.stop_processing()


if __name__ == "__main__":
    asyncio.run(test())
