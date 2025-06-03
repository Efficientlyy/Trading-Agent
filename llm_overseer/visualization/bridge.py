#!/usr/bin/env python
"""
LLM-Visualization Bridge Module

This module provides a bridge between the LLM Strategic Overseer and the visualization system,
enabling bidirectional communication between strategic decision-making and visual representation.
"""

import os
import sys
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("llm_visualization_bridge.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("llm_visualization_bridge")

class LLMVisualizationBridge:
    """
    Bridge between LLM Strategic Overseer and visualization system.
    
    This class enables bidirectional communication between the LLM Overseer
    and the visualization components, transforming data formats and managing
    subscriptions for real-time updates.
    """
    
    def __init__(self, llm_overseer=None, event_bus=None):
        """
        Initialize LLM-Visualization Bridge.
        
        Args:
            llm_overseer: LLM Overseer instance
            event_bus: Event Bus instance
        """
        self.llm_overseer = llm_overseer
        self.event_bus = event_bus
        
        # Subscription management
        self.subscribers = {
            "strategic_decisions": [],
            "pattern_recognition": [],
            "market_analysis": [],
            "risk_alerts": []
        }
        
        # Cache for recent data
        self.decision_cache = []
        self.pattern_cache = []
        self.analysis_cache = []
        self.alert_cache = []
        
        logger.info("LLM-Visualization Bridge initialized")
    
    def set_llm_overseer(self, llm_overseer):
        """
        Set LLM Overseer instance.
        
        Args:
            llm_overseer: LLM Overseer instance
        """
        self.llm_overseer = llm_overseer
        logger.info("LLM Overseer set")
    
    def set_event_bus(self, event_bus):
        """
        Set Event Bus instance.
        
        Args:
            event_bus: Event Bus instance
        """
        self.event_bus = event_bus
        
        # Register event handlers
        if self.event_bus:
            self.event_bus.subscribe("llm.strategic_decision", self._handle_strategic_decision)
            self.event_bus.subscribe("llm.risk_alert", self._handle_risk_alert)
            self.event_bus.subscribe("visualization.pattern_detected", self._handle_pattern_detected)
            self.event_bus.subscribe("visualization.market_analysis", self._handle_market_analysis)
        
        logger.info("Event Bus set and handlers registered")
    
    def subscribe_to_strategic_decisions(self, callback: Callable[[Dict[str, Any]], None]) -> int:
        """
        Subscribe to strategic decisions.
        
        Args:
            callback: Callback function to handle strategic decisions
            
        Returns:
            Subscription ID
        """
        subscription_id = len(self.subscribers["strategic_decisions"])
        self.subscribers["strategic_decisions"].append(callback)
        
        # Send cached decisions to new subscriber
        for decision in self.decision_cache:
            callback(decision)
        
        logger.info(f"New subscriber to strategic decisions: {subscription_id}")
        return subscription_id
    
    def subscribe_to_pattern_recognition(self, callback: Callable[[Dict[str, Any]], None]) -> int:
        """
        Subscribe to pattern recognition results.
        
        Args:
            callback: Callback function to handle pattern recognition results
            
        Returns:
            Subscription ID
        """
        subscription_id = len(self.subscribers["pattern_recognition"])
        self.subscribers["pattern_recognition"].append(callback)
        
        # Send cached patterns to new subscriber
        for pattern in self.pattern_cache:
            callback(pattern)
        
        logger.info(f"New subscriber to pattern recognition: {subscription_id}")
        return subscription_id
    
    def subscribe_to_market_analysis(self, callback: Callable[[Dict[str, Any]], None]) -> int:
        """
        Subscribe to market analysis results.
        
        Args:
            callback: Callback function to handle market analysis results
            
        Returns:
            Subscription ID
        """
        subscription_id = len(self.subscribers["market_analysis"])
        self.subscribers["market_analysis"].append(callback)
        
        # Send cached analysis to new subscriber
        for analysis in self.analysis_cache:
            callback(analysis)
        
        logger.info(f"New subscriber to market analysis: {subscription_id}")
        return subscription_id
    
    def subscribe_to_risk_alerts(self, callback: Callable[[Dict[str, Any]], None]) -> int:
        """
        Subscribe to risk alerts.
        
        Args:
            callback: Callback function to handle risk alerts
            
        Returns:
            Subscription ID
        """
        subscription_id = len(self.subscribers["risk_alerts"])
        self.subscribers["risk_alerts"].append(callback)
        
        # Send cached alerts to new subscriber
        for alert in self.alert_cache:
            callback(alert)
        
        logger.info(f"New subscriber to risk alerts: {subscription_id}")
        return subscription_id
    
    def unsubscribe(self, topic: str, subscription_id: int) -> bool:
        """
        Unsubscribe from topic.
        
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
        logger.info(f"Unsubscribed from {topic}: {subscription_id}")
        return True
    
    async def publish_strategic_decision(self, decision: Dict[str, Any]) -> None:
        """
        Publish strategic decision to subscribers.
        
        Args:
            decision: Strategic decision data
        """
        # Add timestamp if not present
        if "timestamp" not in decision:
            decision["timestamp"] = datetime.now().isoformat()
        
        # Add to cache
        self.decision_cache.append(decision)
        if len(self.decision_cache) > 10:
            self.decision_cache.pop(0)
        
        # Publish to subscribers
        for callback in self.subscribers["strategic_decisions"]:
            if callback:
                try:
                    callback(decision)
                except Exception as e:
                    logger.error(f"Error in strategic decision callback: {e}")
        
        # Publish to event bus if available
        if self.event_bus:
            await self.event_bus.publish("visualization.strategic_decision", decision)
        
        logger.info(f"Published strategic decision: {decision.get('decision_type', 'unknown')}")
    
    async def publish_pattern_recognition(self, pattern: Dict[str, Any]) -> None:
        """
        Publish pattern recognition result to subscribers.
        
        Args:
            pattern: Pattern recognition data
        """
        # Add timestamp if not present
        if "timestamp" not in pattern:
            pattern["timestamp"] = datetime.now().isoformat()
        
        # Add to cache
        self.pattern_cache.append(pattern)
        if len(self.pattern_cache) > 10:
            self.pattern_cache.pop(0)
        
        # Publish to subscribers
        for callback in self.subscribers["pattern_recognition"]:
            if callback:
                try:
                    callback(pattern)
                except Exception as e:
                    logger.error(f"Error in pattern recognition callback: {e}")
        
        # Publish to event bus if available
        if self.event_bus:
            await self.event_bus.publish("llm.pattern_detected", pattern)
        
        # Update LLM Overseer context if available
        if self.llm_overseer:
            self.llm_overseer.update_market_data({
                "pattern_detected": pattern["pattern_type"],
                "confidence": pattern["confidence"],
                "timestamp": pattern["timestamp"],
                "symbol": pattern["symbol"]
            })
        
        logger.info(f"Published pattern recognition: {pattern.get('pattern_type', 'unknown')}")
    
    async def publish_market_analysis(self, analysis: Dict[str, Any]) -> None:
        """
        Publish market analysis result to subscribers.
        
        Args:
            analysis: Market analysis data
        """
        # Add timestamp if not present
        if "timestamp" not in analysis:
            analysis["timestamp"] = datetime.now().isoformat()
        
        # Add to cache
        self.analysis_cache.append(analysis)
        if len(self.analysis_cache) > 10:
            self.analysis_cache.pop(0)
        
        # Publish to subscribers
        for callback in self.subscribers["market_analysis"]:
            if callback:
                try:
                    callback(analysis)
                except Exception as e:
                    logger.error(f"Error in market analysis callback: {e}")
        
        # Publish to event bus if available
        if self.event_bus:
            await self.event_bus.publish("llm.market_analysis", analysis)
        
        # Update LLM Overseer context if available
        if self.llm_overseer:
            self.llm_overseer.update_market_data({
                "market_analysis": analysis["analysis_type"],
                "result": analysis["result"],
                "timestamp": analysis["timestamp"],
                "symbol": analysis["symbol"]
            })
        
        logger.info(f"Published market analysis: {analysis.get('analysis_type', 'unknown')}")
    
    async def publish_risk_alert(self, alert: Dict[str, Any]) -> None:
        """
        Publish risk alert to subscribers.
        
        Args:
            alert: Risk alert data
        """
        # Add timestamp if not present
        if "timestamp" not in alert:
            alert["timestamp"] = datetime.now().isoformat()
        
        # Add to cache
        self.alert_cache.append(alert)
        if len(self.alert_cache) > 10:
            self.alert_cache.pop(0)
        
        # Publish to subscribers
        for callback in self.subscribers["risk_alerts"]:
            if callback:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Error in risk alert callback: {e}")
        
        # Publish to event bus if available
        if self.event_bus:
            await self.event_bus.publish("visualization.risk_alert", alert)
        
        logger.info(f"Published risk alert: {alert.get('alert_type', 'unknown')}")
    
    async def _handle_strategic_decision(self, topic: str, data: Dict[str, Any]) -> None:
        """
        Handle strategic decision event.
        
        Args:
            topic: Event topic
            data: Event data
        """
        await self.publish_strategic_decision(data)
    
    async def _handle_risk_alert(self, topic: str, data: Dict[str, Any]) -> None:
        """
        Handle risk alert event.
        
        Args:
            topic: Event topic
            data: Event data
        """
        await self.publish_risk_alert(data)
    
    async def _handle_pattern_detected(self, topic: str, data: Dict[str, Any]) -> None:
        """
        Handle pattern detected event.
        
        Args:
            topic: Event topic
            data: Event data
        """
        await self.publish_pattern_recognition(data)
    
    async def _handle_market_analysis(self, topic: str, data: Dict[str, Any]) -> None:
        """
        Handle market analysis event.
        
        Args:
            topic: Event topic
            data: Event data
        """
        await self.publish_market_analysis(data)
    
    def transform_decision_to_visualization(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform strategic decision to visualization format.
        
        Args:
            decision: Strategic decision data
            
        Returns:
            Visualization-ready decision data
        """
        # Extract relevant information for visualization
        viz_data = {
            "type": "strategic_decision",
            "decision_type": decision.get("decision_type", "unknown"),
            "timestamp": decision.get("timestamp", datetime.now().isoformat()),
            "symbol": decision.get("symbol", "BTC/USDC"),
            "confidence": decision.get("confidence", 0.5),
            "direction": decision.get("direction", "neutral"),
            "timeframe": decision.get("timeframe", "1h"),
            "visualization": {
                "color": self._get_decision_color(decision.get("direction", "neutral")),
                "icon": self._get_decision_icon(decision.get("decision_type", "unknown")),
                "position": "chart",
                "display_text": decision.get("summary", "Strategic decision")
            }
        }
        
        return viz_data
    
    def transform_pattern_to_llm_context(self, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform pattern recognition result to LLM context format.
        
        Args:
            pattern: Pattern recognition data
            
        Returns:
            LLM context-ready pattern data
        """
        # Extract relevant information for LLM context
        llm_data = {
            "pattern_type": pattern.get("pattern_type", "unknown"),
            "symbol": pattern.get("symbol", "BTC/USDC"),
            "timeframe": pattern.get("timeframe", "1h"),
            "confidence": pattern.get("confidence", 0.5),
            "direction": pattern.get("direction", "neutral"),
            "price_target": pattern.get("price_target"),
            "stop_loss": pattern.get("stop_loss"),
            "timestamp": pattern.get("timestamp", datetime.now().isoformat()),
            "historical_accuracy": pattern.get("historical_accuracy", 0.0)
        }
        
        return llm_data
    
    def _get_decision_color(self, direction: str) -> str:
        """
        Get color for decision based on direction.
        
        Args:
            direction: Decision direction
            
        Returns:
            Color code
        """
        color_map = {
            "bullish": "#4CAF50",  # Green
            "bearish": "#F44336",  # Red
            "neutral": "#FFC107",  # Amber
            "unknown": "#9E9E9E"   # Gray
        }
        
        return color_map.get(direction.lower(), "#9E9E9E")
    
    def _get_decision_icon(self, decision_type: str) -> str:
        """
        Get icon for decision based on type.
        
        Args:
            decision_type: Decision type
            
        Returns:
            Icon name
        """
        icon_map = {
            "entry": "login",
            "exit": "logout",
            "risk_adjustment": "shield",
            "position_sizing": "resize",
            "market_trend": "trending_up",
            "pattern_confirmation": "check_circle",
            "emergency": "warning"
        }
        
        return icon_map.get(decision_type.lower(), "info")


# For testing
async def test():
    """Test function."""
    bridge = LLMVisualizationBridge()
    
    # Test subscription
    def print_decision(decision):
        print(f"Received decision: {decision}")
    
    bridge.subscribe_to_strategic_decisions(print_decision)
    
    # Test publishing
    await bridge.publish_strategic_decision({
        "decision_type": "entry",
        "symbol": "BTC/USDC",
        "direction": "bullish",
        "confidence": 0.85,
        "summary": "Enter long position based on bullish pattern"
    })
    
    # Test transformation
    decision = {
        "decision_type": "entry",
        "symbol": "BTC/USDC",
        "direction": "bullish",
        "confidence": 0.85,
        "summary": "Enter long position based on bullish pattern"
    }
    
    viz_data = bridge.transform_decision_to_visualization(decision)
    print(f"Visualization data: {viz_data}")


if __name__ == "__main__":
    asyncio.run(test())
