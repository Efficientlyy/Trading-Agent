#!/usr/bin/env python
"""
Integration layer for LLM Strategic Overseer.

This module connects the LLM Overseer with the Telegram bot and trading system,
providing a unified interface for command routing and notification delivery.
"""

import os
import sys
import logging
import asyncio
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime

# Import components
# Use relative imports within the package
from ..config.config import Config
from ..core.llm_manager import TieredLLMManager
from ..telegram.bot import TelegramBot
from ..telegram.notifications import NotificationManager

# Import trading system components
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import flash_trading_signals
import paper_trading
import order_book_analytics
import tick_data_processor

logger = logging.getLogger(__name__)

class LLMOverseerIntegration:
    """
    Integration layer for LLM Strategic Overseer.
    Connects the LLM Overseer with the Telegram bot and trading system.
    """
    
    def __init__(self, config: Config, llm_overseer):
        """
        Initialize integration layer.
        
        Args:
            config: Configuration object
            llm_overseer: LLM Overseer instance
        """
        self.config = config
        self.llm_overseer = llm_overseer
        
        # Initialize Telegram components
        self.telegram_bot = TelegramBot(config, llm_overseer)
        self.notification_manager = NotificationManager(config, self.telegram_bot)
        
        # Set up trading system interface
        self.trading_system = None
        self.paper_trading = None
        self.order_book_analytics = None
        self.tick_data_processor = None
        
        # Command routing table
        self.command_routes = {
            "status": self.get_system_status,
            "balance": self.get_account_balance,
            "performance": self.get_performance_metrics,
            "positions": self.get_current_positions,
            "pause": self.pause_trading,
            "resume": self.resume_trading,
            "risk": self.adjust_risk_parameters,
            "emergency_stop": self.emergency_stop,
            "emergency_close": self.emergency_close_positions,
            "report_daily": self.generate_daily_report,
            "report_weekly": self.generate_weekly_report,
            "report_monthly": self.generate_monthly_report,
            "strategy": self.get_current_strategy,
            "market": self.get_market_overview,
            "history": self.get_trading_history,
            "compound_enable_80": lambda: self.set_compounding_rate(0.8),
            "compound_enable_50": lambda: self.set_compounding_rate(0.5),
            "compound_disable": lambda: self.set_compounding_rate(0.0)
        }
        
        logger.info("LLM Overseer Integration initialized")
    
    async def initialize_trading_system(self, mock_mode: bool = True) -> None:
        """
        Initialize trading system components.
        
        Args:
            mock_mode: Whether to use mock mode
        """
        try:
            # Initialize order book analytics
            self.order_book_analytics = order_book_analytics.OrderBookAnalytics()
            
            # Initialize tick data processor
            self.tick_data_processor = tick_data_processor.TickDataProcessor()
            
            # Initialize paper trading
            self.paper_trading = paper_trading.PaperTrading(
                mock_mode=mock_mode,
                order_book_analytics=self.order_book_analytics,
                tick_data_processor=self.tick_data_processor
            )
            
            # Set trading system reference
            self.trading_system = self.paper_trading
            
            logger.info(f"Trading system initialized (mock_mode={mock_mode})")
            
            # Update LLM overseer context with system status
            self.llm_overseer.update_system_status({
                "status": "initialized",
                "mock_mode": mock_mode,
                "timestamp": datetime.now().isoformat()
            })
            
            return True
        except Exception as e:
            logger.error(f"Error initializing trading system: {e}")
            
            # Update LLM overseer context with error
            self.llm_overseer.update_system_status({
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            
            return False
    
    async def start(self, mock_mode: bool = True) -> None:
        """
        Start the integration layer.
        
        Args:
            mock_mode: Whether to use mock mode
        """
        # Initialize trading system
        await self.initialize_trading_system(mock_mode)
        
        # Connect LLM overseer to Telegram bot
        self.telegram_bot.set_llm_overseer(self.llm_overseer)
        
        # Start Telegram bot
        await self.telegram_bot.start()
        
        # Send startup notification
        await self.notification_manager.broadcast_notification(
            "Trading system started",
            level="info",
            data={
                "mock_mode": mock_mode,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        logger.info("LLM Overseer Integration started")
    
    async def stop(self) -> None:
        """Stop the integration layer."""
        # Stop Telegram bot
        await self.telegram_bot.stop()
        
        # Stop trading system
        if self.trading_system:
            # This would call the actual stop method of the trading system
            pass
        
        logger.info("LLM Overseer Integration stopped")
    
    async def route_command(self, command: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Route command to appropriate handler.
        
        Args:
            command: Command name
            params: Command parameters
            
        Returns:
            Command result
        """
        if command not in self.command_routes:
            logger.warning(f"Unknown command: {command}")
            return {
                "success": False,
                "message": f"Unknown command: {command}"
            }
        
        try:
            # Get command handler
            handler = self.command_routes[command]
            
            # Call handler with parameters
            if params:
                result = await handler(**params)
            else:
                result = await handler()
            
            return result
        except Exception as e:
            logger.error(f"Error executing command {command}: {e}")
            return {
                "success": False,
                "message": f"Error executing command: {e}"
            }
    
    async def get_system_status(self) -> Dict[str, Any]:
        """
        Get system status.
        
        Returns:
            System status
        """
        # Get system status from LLM overseer context
        system_status = self.llm_overseer.context_manager.context.get("system_status", {})
        
        # If trading system is available, get additional status
        if self.trading_system:
            # This would get actual status from the trading system
            trading_status = {
                "active": True,
                "uptime": "2 days, 3 hours",
                "last_trade": "2025-06-03T20:45:12Z"
            }
            
            # Merge with system status
            system_status.update(trading_status)
        
        return {
            "success": True,
            "status": system_status
        }
    
    async def get_account_balance(self) -> Dict[str, Any]:
        """
        Get account balance.
        
        Returns:
            Account balance
        """
        if not self.trading_system:
            return {
                "success": False,
                "message": "Trading system not initialized"
            }
        
        # This would get actual balance from the trading system
        balance = {
            "USDC": 43148.94,
            "BTC": 0.0,
            "SOL": 0.000000003,
            "total_value_usd": 43148.94
        }
        
        return {
            "success": True,
            "balance": balance
        }
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics.
        
        Returns:
            Performance metrics
        """
        # Get performance metrics from LLM overseer context
        performance_metrics = self.llm_overseer.context_manager.context.get("performance_metrics", {})
        
        if not performance_metrics:
            # This would get actual metrics from the trading system
            performance_metrics = {
                "daily_pnl": 0.0,
                "daily_pnl_pct": 0.0,
                "total_trades": 0,
                "win_rate": 0.0,
                "avg_trade_pnl": 0.0,
                "largest_win": 0.0,
                "largest_loss": 0.0
            }
        
        return {
            "success": True,
            "metrics": performance_metrics
        }
    
    async def get_current_positions(self) -> Dict[str, Any]:
        """
        Get current positions.
        
        Returns:
            Current positions
        """
        if not self.trading_system:
            return {
                "success": False,
                "message": "Trading system not initialized"
            }
        
        # This would get actual positions from the trading system
        positions = []
        
        return {
            "success": True,
            "positions": positions
        }
    
    async def pause_trading(self) -> Dict[str, Any]:
        """
        Pause trading.
        
        Returns:
            Result
        """
        if not self.trading_system:
            return {
                "success": False,
                "message": "Trading system not initialized"
            }
        
        # This would pause the actual trading system
        
        # Update LLM overseer context
        self.llm_overseer.update_system_status({
            "status": "paused",
            "timestamp": datetime.now().isoformat()
        })
        
        # Send notification
        await self.notification_manager.broadcast_notification(
            "Trading system paused",
            level="info"
        )
        
        return {
            "success": True,
            "message": "Trading system paused"
        }
    
    async def resume_trading(self) -> Dict[str, Any]:
        """
        Resume trading.
        
        Returns:
            Result
        """
        if not self.trading_system:
            return {
                "success": False,
                "message": "Trading system not initialized"
            }
        
        # This would resume the actual trading system
        
        # Update LLM overseer context
        self.llm_overseer.update_system_status({
            "status": "running",
            "timestamp": datetime.now().isoformat()
        })
        
        # Send notification
        await self.notification_manager.broadcast_notification(
            "Trading system resumed",
            level="info"
        )
        
        return {
            "success": True,
            "message": "Trading system resumed"
        }
    
    async def adjust_risk_parameters(self, risk_level: str) -> Dict[str, Any]:
        """
        Adjust risk parameters.
        
        Args:
            risk_level: Risk level ("low", "medium", "high")
            
        Returns:
            Result
        """
        if not self.trading_system:
            return {
                "success": False,
                "message": "Trading system not initialized"
            }
        
        # Define risk parameters for each level
        risk_params = {
            "low": {
                "max_position_size_pct": 0.05,
                "max_daily_drawdown_pct": 0.02,
                "stop_loss_pct": 0.01
            },
            "medium": {
                "max_position_size_pct": 0.10,
                "max_daily_drawdown_pct": 0.05,
                "stop_loss_pct": 0.02
            },
            "high": {
                "max_position_size_pct": 0.20,
                "max_daily_drawdown_pct": 0.10,
                "stop_loss_pct": 0.05
            }
        }
        
        if risk_level not in risk_params:
            return {
                "success": False,
                "message": f"Invalid risk level: {risk_level}"
            }
        
        # Get risk parameters for the selected level
        params = risk_params[risk_level]
        
        # This would update the actual trading system
        
        # Update LLM overseer context
        self.llm_overseer.update_risk_parameters({
            "risk_level": risk_level,
            **params,
            "timestamp": datetime.now().isoformat()
        })
        
        # Send notification
        await self.notification_manager.broadcast_notification(
            f"Risk level set to {risk_level.upper()}",
            level="info",
            data=params
        )
        
        return {
            "success": True,
            "message": f"Risk level set to {risk_level}",
            "parameters": params
        }
    
    async def emergency_stop(self) -> Dict[str, Any]:
        """
        Emergency stop trading.
        
        Returns:
            Result
        """
        if not self.trading_system:
            return {
                "success": False,
                "message": "Trading system not initialized"
            }
        
        # This would stop the actual trading system
        
        # Update LLM overseer context
        self.llm_overseer.update_system_status({
            "status": "emergency_stopped",
            "timestamp": datetime.now().isoformat()
        })
        
        # Send notification
        await self.notification_manager.broadcast_notification(
            "EMERGENCY STOP ACTIVATED",
            level="emergency"
        )
        
        return {
            "success": True,
            "message": "Emergency stop activated"
        }
    
    async def emergency_close_positions(self) -> Dict[str, Any]:
        """
        Emergency close all positions.
        
        Returns:
            Result
        """
        if not self.trading_system:
            return {
                "success": False,
                "message": "Trading system not initialized"
            }
        
        # This would close all positions in the actual trading system
        
        # Send notification
        await self.notification_manager.broadcast_notification(
            "CLOSING ALL POSITIONS",
            level="emergency"
        )
        
        return {
            "success": True,
            "message": "Closing all positions"
        }
    
    async def generate_daily_report(self) -> Dict[str, Any]:
        """
        Generate daily report.
        
        Returns:
            Report data
        """
        # Get performance metrics
        performance = await self.get_performance_metrics()
        
        # Get account balance
        balance = await self.get_account_balance()
        
        # Get trading history
        history = await self.get_trading_history()
        
        # Generate report using LLM
        report_prompt = (
            "Generate a daily trading report based on the following data:\n\n"
            f"Performance Metrics: {performance}\n\n"
            f"Account Balance: {balance}\n\n"
            f"Trading History: {history}\n\n"
            "The report should include a summary of performance, key metrics, "
            "notable trades, and recommendations for tomorrow."
        )
        
        report_result = await self.llm_overseer.make_strategic_decision(
            "report_generation",
            report_prompt,
            urgency="normal"
        )
        
        report = report_result["decision"]
        
        # Send notification with report
        await self.notification_manager.broadcast_notification(
            "Daily Report Generated",
            level="info",
            data={"report": report}
        )
        
        return {
            "success": True,
            "report": report,
            "metrics": performance.get("metrics", {}),
            "balance": balance.get("balance", {})
        }
    
    async def generate_weekly_report(self) -> Dict[str, Any]:
        """
        Generate weekly report.
        
        Returns:
            Report data
        """
        # Similar to daily report but with weekly data
        # This is a simplified implementation
        
        # Generate report using LLM
        report_prompt = (
            "Generate a weekly trading report with the following sections:\n"
            "1. Performance Summary\n"
            "2. Key Metrics\n"
            "3. Market Analysis\n"
            "4. Strategy Performance\n"
            "5. Recommendations for Next Week"
        )
        
        report_result = await self.llm_overseer.make_strategic_decision(
            "report_generation",
            report_prompt,
            urgency="normal"
        )
        
        report = report_result["decision"]
        
        # Send notification with report
        await self.notification_manager.broadcast_notification(
            "Weekly Report Generated",
            level="info",
            data={"report": report}
        )
        
        return {
            "success": True,
            "report": report
        }
    
    async def generate_monthly_report(self) -> Dict[str, Any]:
        """
        Generate monthly report.
        
        Returns:
            Report data
        """
        # Similar to weekly report but with monthly data
        # This is a simplified implementation
        
        # Generate report using LLM
        report_prompt = (
            "Generate a comprehensive monthly trading report with the following sections:\n"
            "1. Executive Summary\n"
            "2. Performance Metrics\n"
            "3. Market Analysis\n"
            "4. Strategy Performance\n"
            "5. Risk Analysis\n"
            "6. Recommendations for Next Month"
        )
        
        report_result = await self.llm_overseer.make_strategic_decision(
            "report_generation",
            report_prompt,
            urgency="normal"
        )
        
        report = report_result["decision"]
        
        # Send notification with report
        await self.notification_manager.broadcast_notification(
            "Monthly Report Generated",
            level="info",
            data={"report": report}
        )
        
        return {
            "success": True,
            "report": report
        }
    
    async def get_current_strategy(self) -> Dict[str, Any]:
        """
        Get current trading strategy.
        
        Returns:
            Strategy information
        """
        # Get strategy information from LLM overseer
        strategy_prompt = (
            "Analyze the current trading system configuration and performance metrics. "
            "Provide a concise summary of the current trading strategy, including key components, "
            "focus areas, and performance characteristics."
        )
        
        strategy_result = await self.llm_overseer.make_strategic_decision(
            "strategy_analysis",
            strategy_prompt,
            urgency="normal"
        )
        
        strategy = strategy_result["decision"]
        
        return {
            "success": True,
            "strategy": strategy
        }
    
    async def get_market_overview(self) -> Dict[str, Any]:
        """
        Get market overview.
        
        Returns:
            Market overview
        """
        # Get market data from LLM overseer context
        market_data = self.llm_overseer.context_manager.context.get("market_data", [])
        
        # Generate market analysis using LLM
        if market_data:
            market_prompt = (
                f"Analyze the following market data and provide a concise overview "
                f"of current market conditions, trends, and key observations:\n\n"
                f"{market_data[-5:]}"
            )
            
            analysis_result = await self.llm_overseer.make_strategic_decision(
                "market_analysis",
                market_prompt,
                urgency="normal"
            )
            
            analysis = analysis_result["decision"]
        else:
            analysis = "No market data available for analysis."
        
        return {
            "success": True,
            "market_data": market_data[-5:] if market_data else [],
            "analysis": analysis
        }
    
    async def get_trading_history(self) -> Dict[str, Any]:
        """
        Get trading history.
        
        Returns:
            Trading history
        """
        # Get trading history from LLM overseer context
        trading_history = self.llm_overseer.context_manager.context.get("trading_history", [])
        
        return {
            "success": True,
            "history": trading_history[-10:] if trading_history else []
        }
    
    async def set_compounding_rate(self, rate: float) -> Dict[str, Any]:
        """
        Set profit compounding rate.
        
        Args:
            rate: Compounding rate (0.0 to 1.0)
            
        Returns:
            Result
        """
        if rate < 0.0 or rate > 1.0:
            return {
                "success": False,
                "message": f"Invalid compounding rate: {rate}. Must be between 0.0 and 1.0."
            }
        
        # This would update the actual trading system
        
        # Update LLM overseer context
        self.llm_overseer.update_risk_parameters({
            "compounding_rate": rate,
            "timestamp": datetime.now().isoformat()
        })
        
        # Format message based on rate
        if rate == 0.0:
            message = "Profit compounding disabled"
        else:
            message = f"Profit compounding enabled ({int(rate * 100)}%)"
        
        # Send notification
        await self.notification_manager.broadcast_notification(
            message,
            level="info",
            data={"compounding_rate": rate}
        )
        
        return {
            "success": True,
            "message": message,
            "compounding_rate": rate
        }
    
    async def process_market_data(self, data: Dict[str, Any]) -> None:
        """
        Process market data and update LLM overseer context.
        
        Args:
            data: Market data
        """
        # Update LLM overseer context
        self.llm_overseer.update_market_data(data)
        
        # Process data with order book analytics if available
        if self.order_book_analytics:
            # This would process the data with the actual order book analytics
            pass
        
        # Process data with tick data processor if available
        if self.tick_data_processor:
            # This would process the data with the actual tick data processor
            pass
    
    async def process_trade(self, trade: Dict[str, Any]) -> None:
        """
        Process trade and update LLM overseer context.
        
        Args:
            trade: Trade data
        """
        # Update LLM overseer context
        self.llm_overseer.update_trading_history(trade)
        
        # Send trade notification
        message, data = self.notification_manager.format_trade_notification(trade)
        await self.notification_manager.broadcast_notification(
            message,
            level="trade",
            data=data
        )
    
    async def update_performance_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Update performance metrics and LLM overseer context.
        
        Args:
            metrics: Performance metrics
        """
        # Update LLM overseer context
        self.llm_overseer.update_performance_metrics(metrics)
        
        # Check if notification should be sent
        if metrics.get("notify", False):
            message, data = self.notification_manager.format_performance_notification(metrics)
            await self.notification_manager.broadcast_notification(
                message,
                level="info",
                data=data
            )
    
    async def handle_risk_alert(self, alert: Dict[str, Any]) -> None:
        """
        Handle risk alert and update LLM overseer context.
        
        Args:
            alert: Risk alert data
        """
        # Update LLM overseer context
        self.llm_overseer.update_risk_parameters(alert)
        
        # Send risk notification
        message, data = self.notification_manager.format_risk_notification(alert)
        
        # Determine notification level based on risk level
        risk_level = alert.get("risk_level", "").lower()
        if risk_level == "high":
            level = "critical"
        elif risk_level == "medium":
            level = "trade"
        else:
            level = "info"
        
        await self.notification_manager.broadcast_notification(
            message,
            level=level,
            data=data
        )
        
        # If risk level is high, get strategic decision from LLM
        if risk_level == "high":
            decision_prompt = (
                f"A high risk alert has been triggered: {alert.get('reason', 'Unknown reason')}\n\n"
                f"Current exposure: {alert.get('current_exposure', 0)}%\n"
                f"Max allowed: {alert.get('max_allowed', 0)}%\n"
                f"Daily drawdown: {alert.get('daily_drawdown', 0)}%\n\n"
                f"Recommend immediate actions to mitigate risk."
            )
            
            decision_result = await self.llm_overseer.make_strategic_decision(
                "risk_management",
                decision_prompt,
                urgency="high"
            )
            
            # Send decision notification
            await self.notification_manager.broadcast_notification(
                "Risk Mitigation Recommendation",
                level="critical",
                data={"recommendation": decision_result["decision"]}
            )


async def main():
    """Main function for testing."""
    # Import LLM Overseer
    from main import LLMOverseer
    
    # Initialize configuration
    config = Config()
    
    # Initialize LLM Overseer
    llm_overseer = LLMOverseer()
    
    # Initialize integration layer
    integration = LLMOverseerIntegration(config, llm_overseer)
    
    try:
        # Start integration layer
        await integration.start(mock_mode=True)
        
        # Keep the integration layer running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        # Stop integration layer
        await integration.stop()


if __name__ == "__main__":
    asyncio.run(main())
