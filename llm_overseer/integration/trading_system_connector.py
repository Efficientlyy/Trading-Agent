#!/usr/bin/env python
"""
Trading System Connector for LLM Strategic Overseer

This module provides a connector between the LLM Strategic Overseer and the
trading system core, enabling bidirectional communication and data flow.
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
        logging.FileHandler("trading_system_connector.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("trading_system_connector")

class TradingSystemConnector:
    """
    Connector between LLM Strategic Overseer and trading system core.
    
    This class enables bidirectional communication between the LLM Overseer
    and the trading system core, facilitating data flow and command execution.
    """
    
    def __init__(self, llm_overseer=None, event_bus=None, data_pipeline=None):
        """
        Initialize Trading System Connector.
        
        Args:
            llm_overseer: LLM Overseer instance
            event_bus: Event Bus instance
            data_pipeline: Unified Data Pipeline instance
        """
        self.llm_overseer = llm_overseer
        self.event_bus = event_bus
        self.data_pipeline = data_pipeline
        
        # Trading system components
        self.flash_trading = None
        self.paper_trading = None
        self.order_book_analytics = None
        self.tick_data_processor = None
        
        # Command handlers
        self.command_handlers = {
            "market_analysis": self._handle_market_analysis_command,
            "execute_trade": self._handle_execute_trade_command,
            "adjust_parameters": self._handle_adjust_parameters_command,
            "cancel_orders": self._handle_cancel_orders_command,
            "emergency_stop": self._handle_emergency_stop_command
        }
        
        # Status tracking
        self.status = {
            "connected": False,
            "flash_trading_active": False,
            "paper_trading_active": False,
            "last_update": None
        }
        
        logger.info("Trading System Connector initialized")
    
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
            self.event_bus.subscribe("llm.command", self._handle_llm_command)
            self.event_bus.subscribe("trading.status_update", self._handle_trading_status_update)
            self.event_bus.subscribe("trading.order_update", self._handle_order_update)
            self.event_bus.subscribe("trading.market_update", self._handle_market_update)
        
        logger.info("Event Bus set and handlers registered")
    
    def set_data_pipeline(self, data_pipeline):
        """
        Set Unified Data Pipeline instance.
        
        Args:
            data_pipeline: Unified Data Pipeline instance
        """
        self.data_pipeline = data_pipeline
        logger.info("Unified Data Pipeline set")
    
    def connect_flash_trading(self, flash_trading):
        """
        Connect Flash Trading module.
        
        Args:
            flash_trading: Flash Trading module instance
        """
        self.flash_trading = flash_trading
        logger.info("Flash Trading module connected")
    
    def connect_paper_trading(self, paper_trading):
        """
        Connect Paper Trading module.
        
        Args:
            paper_trading: Paper Trading module instance
        """
        self.paper_trading = paper_trading
        logger.info("Paper Trading module connected")
    
    def connect_order_book_analytics(self, order_book_analytics):
        """
        Connect Order Book Analytics module.
        
        Args:
            order_book_analytics: Order Book Analytics module instance
        """
        self.order_book_analytics = order_book_analytics
        logger.info("Order Book Analytics module connected")
    
    def connect_tick_data_processor(self, tick_data_processor):
        """
        Connect Tick Data Processor module.
        
        Args:
            tick_data_processor: Tick Data Processor module instance
        """
        self.tick_data_processor = tick_data_processor
        logger.info("Tick Data Processor module connected")
    
    async def start(self):
        """Start the connector."""
        self.status["connected"] = True
        self.status["last_update"] = datetime.now().isoformat()
        
        # Publish status update
        if self.event_bus:
            await self.event_bus.publish("connector.status_update", {
                "status": "connected",
                "timestamp": self.status["last_update"]
            })
        
        # Update LLM Overseer context
        if self.llm_overseer:
            self.llm_overseer.update_system_status({
                "status": "connected",
                "trading_connector": "active",
                "timestamp": self.status["last_update"]
            })
        
        logger.info("Trading System Connector started")
    
    async def stop(self):
        """Stop the connector."""
        self.status["connected"] = False
        self.status["last_update"] = datetime.now().isoformat()
        
        # Publish status update
        if self.event_bus:
            await self.event_bus.publish("connector.status_update", {
                "status": "disconnected",
                "timestamp": self.status["last_update"]
            })
        
        # Update LLM Overseer context
        if self.llm_overseer:
            self.llm_overseer.update_system_status({
                "status": "disconnected",
                "trading_connector": "inactive",
                "timestamp": self.status["last_update"]
            })
        
        logger.info("Trading System Connector stopped")
    
    async def execute_command(self, command_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute command on trading system.
        
        Args:
            command_type: Type of command to execute
            params: Command parameters
            
        Returns:
            Command result
        """
        if not self.status["connected"]:
            logger.warning("Cannot execute command: connector not connected")
            return {
                "success": False,
                "error": "Connector not connected"
            }
        
        # Check if command type is supported
        if command_type not in self.command_handlers:
            logger.warning(f"Unsupported command type: {command_type}")
            return {
                "success": False,
                "error": f"Unsupported command type: {command_type}"
            }
        
        # Execute command
        try:
            result = await self.command_handlers[command_type](params)
            
            # Publish command result
            if self.event_bus:
                await self.event_bus.publish("connector.command_result", {
                    "command_type": command_type,
                    "params": params,
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                })
            
            return result
        except Exception as e:
            logger.error(f"Error executing command {command_type}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _handle_llm_command(self, topic: str, data: Dict[str, Any]):
        """
        Handle LLM command event.
        
        Args:
            topic: Event topic
            data: Event data
        """
        command_type = data.get("command_type")
        params = data.get("params", {})
        
        if not command_type:
            logger.warning("LLM command missing command_type")
            return
        
        # Execute command
        result = await self.execute_command(command_type, params)
        
        # Publish result to event bus
        if self.event_bus:
            await self.event_bus.publish("llm.command_result", {
                "command_type": command_type,
                "params": params,
                "result": result,
                "timestamp": datetime.now().isoformat()
            })
    
    async def _handle_trading_status_update(self, topic: str, data: Dict[str, Any]):
        """
        Handle trading status update event.
        
        Args:
            topic: Event topic
            data: Event data
        """
        # Update status
        if "flash_trading_active" in data:
            self.status["flash_trading_active"] = data["flash_trading_active"]
        
        if "paper_trading_active" in data:
            self.status["paper_trading_active"] = data["paper_trading_active"]
        
        self.status["last_update"] = datetime.now().isoformat()
        
        # Update LLM Overseer context
        if self.llm_overseer:
            self.llm_overseer.update_system_status({
                "flash_trading_active": self.status["flash_trading_active"],
                "paper_trading_active": self.status["paper_trading_active"],
                "timestamp": self.status["last_update"]
            })
    
    async def _handle_order_update(self, topic: str, data: Dict[str, Any]):
        """
        Handle order update event.
        
        Args:
            topic: Event topic
            data: Event data
        """
        # Update LLM Overseer context
        if self.llm_overseer:
            self.llm_overseer.update_trading_history(data)
        
        # Update data pipeline
        if self.data_pipeline:
            order_id = data.get("order_id", f"order_{datetime.now().timestamp()}")
            self.data_pipeline.update_decision_data(order_id, data)
    
    async def _handle_market_update(self, topic: str, data: Dict[str, Any]):
        """
        Handle market update event.
        
        Args:
            topic: Event topic
            data: Event data
        """
        # Update LLM Overseer context
        if self.llm_overseer:
            self.llm_overseer.update_market_data(data)
        
        # Update data pipeline
        if self.data_pipeline:
            symbol = data.get("symbol")
            if symbol:
                self.data_pipeline.update_market_data(symbol, data)
    
    async def _handle_market_analysis_command(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle market analysis command.
        
        Args:
            params: Command parameters
            
        Returns:
            Command result
        """
        symbol = params.get("symbol", "BTC/USDC")
        analysis_type = params.get("analysis_type", "orderbook")
        
        # Check if required modules are connected
        if analysis_type == "orderbook" and not self.order_book_analytics:
            return {
                "success": False,
                "error": "Order Book Analytics module not connected"
            }
        
        if analysis_type == "tick" and not self.tick_data_processor:
            return {
                "success": False,
                "error": "Tick Data Processor module not connected"
            }
        
        # Execute analysis
        try:
            if analysis_type == "orderbook":
                # This would call the actual order book analytics
                # For now, we'll simulate a response
                result = {
                    "symbol": symbol,
                    "timestamp": datetime.now().isoformat(),
                    "bid_ask_imbalance": 0.75,
                    "depth_imbalance": 0.65,
                    "pressure_direction": "buy",
                    "liquidity_score": 0.85
                }
            elif analysis_type == "tick":
                # This would call the actual tick data processor
                # For now, we'll simulate a response
                result = {
                    "symbol": symbol,
                    "timestamp": datetime.now().isoformat(),
                    "tick_direction": "up",
                    "momentum_score": 0.68,
                    "volatility": 0.12,
                    "trend_strength": 0.75
                }
            else:
                return {
                    "success": False,
                    "error": f"Unsupported analysis type: {analysis_type}"
                }
            
            return {
                "success": True,
                "analysis_type": analysis_type,
                "result": result
            }
        except Exception as e:
            logger.error(f"Error executing market analysis: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _handle_execute_trade_command(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle execute trade command.
        
        Args:
            params: Command parameters
            
        Returns:
            Command result
        """
        symbol = params.get("symbol", "BTC/USDC")
        side = params.get("side", "buy")
        quantity = params.get("quantity", 0.0)
        price = params.get("price")
        order_type = params.get("order_type", "limit")
        
        # Validate parameters
        if quantity <= 0:
            return {
                "success": False,
                "error": "Invalid quantity"
            }
        
        if order_type == "limit" and not price:
            return {
                "success": False,
                "error": "Price required for limit orders"
            }
        
        # Check if required modules are connected
        if not self.paper_trading and not self.flash_trading:
            return {
                "success": False,
                "error": "No trading module connected"
            }
        
        # Execute trade
        try:
            # Prefer paper trading if available
            if self.paper_trading:
                # This would call the actual paper trading module
                # For now, we'll simulate a response
                order_id = f"order_{datetime.now().timestamp()}"
                result = {
                    "order_id": order_id,
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "price": price,
                    "order_type": order_type,
                    "status": "placed",
                    "timestamp": datetime.now().isoformat()
                }
            elif self.flash_trading:
                # This would call the actual flash trading module
                # For now, we'll simulate a response
                order_id = f"order_{datetime.now().timestamp()}"
                result = {
                    "order_id": order_id,
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "price": price,
                    "order_type": order_type,
                    "status": "placed",
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "success": False,
                    "error": "No trading module connected"
                }
            
            # Update LLM Overseer context
            if self.llm_overseer:
                self.llm_overseer.update_trading_history(result)
            
            return {
                "success": True,
                "order": result
            }
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _handle_adjust_parameters_command(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle adjust parameters command.
        
        Args:
            params: Command parameters
            
        Returns:
            Command result
        """
        module = params.get("module")
        parameters = params.get("parameters", {})
        
        if not module:
            return {
                "success": False,
                "error": "Module not specified"
            }
        
        if not parameters:
            return {
                "success": False,
                "error": "No parameters specified"
            }
        
        # Check which module to adjust
        try:
            if module == "flash_trading":
                if not self.flash_trading:
                    return {
                        "success": False,
                        "error": "Flash Trading module not connected"
                    }
                
                # This would call the actual flash trading module
                # For now, we'll simulate a response
                result = {
                    "module": "flash_trading",
                    "parameters_updated": list(parameters.keys()),
                    "timestamp": datetime.now().isoformat()
                }
            elif module == "paper_trading":
                if not self.paper_trading:
                    return {
                        "success": False,
                        "error": "Paper Trading module not connected"
                    }
                
                # This would call the actual paper trading module
                # For now, we'll simulate a response
                result = {
                    "module": "paper_trading",
                    "parameters_updated": list(parameters.keys()),
                    "timestamp": datetime.now().isoformat()
                }
            elif module == "order_book_analytics":
                if not self.order_book_analytics:
                    return {
                        "success": False,
                        "error": "Order Book Analytics module not connected"
                    }
                
                # This would call the actual order book analytics module
                # For now, we'll simulate a response
                result = {
                    "module": "order_book_analytics",
                    "parameters_updated": list(parameters.keys()),
                    "timestamp": datetime.now().isoformat()
                }
            elif module == "tick_data_processor":
                if not self.tick_data_processor:
                    return {
                        "success": False,
                        "error": "Tick Data Processor module not connected"
                    }
                
                # This would call the actual tick data processor module
                # For now, we'll simulate a response
                result = {
                    "module": "tick_data_processor",
                    "parameters_updated": list(parameters.keys()),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "success": False,
                    "error": f"Unsupported module: {module}"
                }
            
            # Update LLM Overseer context
            if self.llm_overseer:
                self.llm_overseer.update_system_status({
                    f"{module}_parameters_updated": list(parameters.keys()),
                    "timestamp": datetime.now().isoformat()
                })
            
            return {
                "success": True,
                "result": result
            }
        except Exception as e:
            logger.error(f"Error adjusting parameters: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _handle_cancel_orders_command(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle cancel orders command.
        
        Args:
            params: Command parameters
            
        Returns:
            Command result
        """
        symbol = params.get("symbol")
        order_id = params.get("order_id")
        cancel_all = params.get("cancel_all", False)
        
        if not symbol and not order_id and not cancel_all:
            return {
                "success": False,
                "error": "Symbol, order_id, or cancel_all required"
            }
        
        # Check if required modules are connected
        if not self.paper_trading and not self.flash_trading:
            return {
                "success": False,
                "error": "No trading module connected"
            }
        
        # Execute cancel
        try:
            # Prefer paper trading if available
            if self.paper_trading:
                # This would call the actual paper trading module
                # For now, we'll simulate a response
                result = {
                    "symbol": symbol,
                    "order_id": order_id,
                    "cancel_all": cancel_all,
                    "orders_cancelled": 1,
                    "timestamp": datetime.now().isoformat()
                }
            elif self.flash_trading:
                # This would call the actual flash trading module
                # For now, we'll simulate a response
                result = {
                    "symbol": symbol,
                    "order_id": order_id,
                    "cancel_all": cancel_all,
                    "orders_cancelled": 1,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "success": False,
                    "error": "No trading module connected"
                }
            
            # Update LLM Overseer context
            if self.llm_overseer:
                self.llm_overseer.update_trading_history({
                    "action": "cancel_orders",
                    "symbol": symbol,
                    "order_id": order_id,
                    "cancel_all": cancel_all,
                    "orders_cancelled": result["orders_cancelled"],
                    "timestamp": datetime.now().isoformat()
                })
            
            return {
                "success": True,
                "result": result
            }
        except Exception as e:
            logger.error(f"Error cancelling orders: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _handle_emergency_stop_command(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle emergency stop command.
        
        Args:
            params: Command parameters
            
        Returns:
            Command result
        """
        reason = params.get("reason", "Emergency stop requested")
        
        # Execute emergency stop
        try:
            # Stop flash trading if connected
            if self.flash_trading:
                # This would call the actual flash trading module
                # For now, we'll simulate a response
                flash_result = {
                    "module": "flash_trading",
                    "status": "stopped",
                    "timestamp": datetime.now().isoformat()
                }
            else:
                flash_result = {
                    "module": "flash_trading",
                    "status": "not_connected",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Stop paper trading if connected
            if self.paper_trading:
                # This would call the actual paper trading module
                # For now, we'll simulate a response
                paper_result = {
                    "module": "paper_trading",
                    "status": "stopped",
                    "timestamp": datetime.now().isoformat()
                }
            else:
                paper_result = {
                    "module": "paper_trading",
                    "status": "not_connected",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Update status
            self.status["flash_trading_active"] = False
            self.status["paper_trading_active"] = False
            self.status["last_update"] = datetime.now().isoformat()
            
            # Update LLM Overseer context
            if self.llm_overseer:
                self.llm_overseer.update_system_status({
                    "status": "emergency_stop",
                    "reason": reason,
                    "flash_trading_active": False,
                    "paper_trading_active": False,
                    "timestamp": self.status["last_update"]
                })
            
            # Publish emergency stop event
            if self.event_bus:
                await self.event_bus.publish("trading.emergency_stop", {
                    "reason": reason,
                    "timestamp": datetime.now().isoformat()
                }, priority="emergency")
            
            return {
                "success": True,
                "flash_trading": flash_result,
                "paper_trading": paper_result,
                "reason": reason
            }
        except Exception as e:
            logger.error(f"Error executing emergency stop: {e}")
            return {
                "success": False,
                "error": str(e)
            }


# For testing
async def test():
    """Test function."""
    # Create connector
    connector = TradingSystemConnector()
    
    # Start connector
    await connector.start()
    
    # Test market analysis command
    result = await connector.execute_command("market_analysis", {
        "symbol": "BTC/USDC",
        "analysis_type": "orderbook"
    })
    print(f"Market analysis result: {result}")
    
    # Test execute trade command
    result = await connector.execute_command("execute_trade", {
        "symbol": "BTC/USDC",
        "side": "buy",
        "quantity": 0.1,
        "price": 50000.0,
        "order_type": "limit"
    })
    print(f"Execute trade result: {result}")
    
    # Stop connector
    await connector.stop()


if __name__ == "__main__":
    asyncio.run(test())
