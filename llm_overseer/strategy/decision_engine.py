#!/usr/bin/env python
"""
Decision engine module for LLM Strategic Overseer.

This module implements the strategic decision-making logic for trading
optimization, market analysis, and risk management.
"""

import os
import sys
import logging
import json
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class DecisionEngine:
    """
    Decision engine for strategic trading decisions.
    
    Implements decision-making logic for market analysis, trading strategy
    optimization, and risk management based on LLM insights.
    """
    
    def __init__(self, config, llm_manager, context_manager):
        """
        Initialize decision engine.
        
        Args:
            config: Configuration object
            llm_manager: LLM manager for generating insights
            context_manager: Context manager for market data
        """
        self.config = config
        self.llm_manager = llm_manager
        self.context_manager = context_manager
        
        # Load decision engine configuration
        self.decision_threshold = self.config.get("strategy.decision_threshold", 0.75)
        self.max_position_size = self.config.get("strategy.max_position_size", 0.1)  # 10% of capital
        self.risk_tolerance = self.config.get("strategy.risk_tolerance", "medium")
        
        # Initialize tracking variables
        self.decisions_history = []
        self.current_strategy = None
        self.last_analysis_time = None
        
        # Load historical data if available
        self.data_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "decisions_history.json"
        )
        self._load_history()
        
        logger.info(f"Decision engine initialized with {self.risk_tolerance} risk tolerance")
    
    def _load_history(self) -> None:
        """Load decisions history from file."""
        os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
        
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    self.decisions_history = data.get("history", [])
                    self.current_strategy = data.get("current_strategy")
                    
                    last_time = data.get("last_analysis_time")
                    if last_time:
                        self.last_analysis_time = datetime.fromisoformat(last_time)
                    
                    logger.info(f"Loaded decisions history: {len(self.decisions_history)} records")
            except Exception as e:
                logger.error(f"Error loading decisions history: {e}")
    
    def _save_history(self) -> None:
        """Save decisions history to file."""
        try:
            data = {
                "history": self.decisions_history[-100:],  # Keep last 100 decisions
                "current_strategy": self.current_strategy,
                "last_analysis_time": self.last_analysis_time.isoformat() if self.last_analysis_time else None
            }
            
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Saved decisions history: {len(self.decisions_history)} records")
        except Exception as e:
            logger.error(f"Error saving decisions history: {e}")
    
    async def analyze_market_trend(self, symbol: str, timeframe: str = "1h") -> Dict[str, Any]:
        """
        Analyze market trend for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDC")
            timeframe: Analysis timeframe (e.g., "1h", "4h", "1d")
            
        Returns:
            Market trend analysis result
        """
        # Get market data from context
        market_data = self.context_manager.get_market_data(symbol, timeframe)
        
        # Prepare prompt for LLM
        prompt = f"""
        Analyze the current market trend for {symbol} based on the following data:
        
        Price: {market_data.get('price')}
        24h Change: {market_data.get('change_24h')}%
        Volume: {market_data.get('volume_24h')}
        
        Recent price movements:
        {market_data.get('price_history', [])}
        
        Recent volume:
        {market_data.get('volume_history', [])}
        
        Recent trades:
        {market_data.get('recent_trades', [])}
        
        Order book imbalance:
        {market_data.get('order_book_imbalance')}
        
        Based on this data, provide:
        1. Current market trend (bullish, bearish, or neutral)
        2. Key support and resistance levels
        3. Volume analysis
        4. Short-term price prediction (1-4 hours)
        5. Recommended trading strategy
        """
        
        # Generate analysis using LLM
        result = await self.llm_manager.generate_response(
            prompt,
            model_tier=2,  # Use tier 2 model for market analysis
            max_tokens=1000
        )
        
        if not result["success"]:
            logger.error(f"Failed to analyze market trend: {result.get('error')}")
            return {
                "success": False,
                "error": result.get("error", "Unknown error"),
                "timestamp": datetime.now().isoformat()
            }
        
        # Parse LLM response
        analysis = {
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "analysis": result["response"],
            "model": result["model"],
            "timestamp": datetime.now().isoformat()
        }
        
        # Update tracking variables
        self.last_analysis_time = datetime.now()
        self.decisions_history.append({
            "type": "market_analysis",
            "symbol": symbol,
            "timeframe": timeframe,
            "timestamp": self.last_analysis_time.isoformat(),
            "result": analysis
        })
        
        # Save history
        self._save_history()
        
        logger.info(f"Market trend analysis completed for {symbol} ({timeframe})")
        
        return analysis
    
    async def optimize_trading_parameters(self, symbol: str) -> Dict[str, Any]:
        """
        Optimize trading parameters for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDC")
            
        Returns:
            Optimized trading parameters
        """
        # Get trading history and performance from context
        trading_history = self.context_manager.get_trading_history(symbol)
        performance = self.context_manager.get_performance_metrics(symbol)
        
        # Prepare prompt for LLM
        prompt = f"""
        Optimize trading parameters for {symbol} based on the following data:
        
        Trading history:
        {trading_history}
        
        Performance metrics:
        - Win rate: {performance.get('win_rate')}%
        - Average profit: {performance.get('avg_profit')}%
        - Average loss: {performance.get('avg_loss')}%
        - Profit factor: {performance.get('profit_factor')}
        - Sharpe ratio: {performance.get('sharpe_ratio')}
        
        Current parameters:
        - Entry threshold: {performance.get('entry_threshold')}
        - Exit threshold: {performance.get('exit_threshold')}
        - Stop loss: {performance.get('stop_loss')}%
        - Take profit: {performance.get('take_profit')}%
        - Position size: {performance.get('position_size')}%
        
        Based on this data, provide optimized trading parameters to improve performance:
        1. Entry threshold
        2. Exit threshold
        3. Stop loss percentage
        4. Take profit percentage
        5. Position size percentage (max {self.max_position_size * 100}%)
        6. Reasoning for each parameter
        """
        
        # Generate optimization using LLM
        result = await self.llm_manager.generate_response(
            prompt,
            model_tier=2,  # Use tier 2 model for parameter optimization
            max_tokens=1000
        )
        
        if not result["success"]:
            logger.error(f"Failed to optimize trading parameters: {result.get('error')}")
            return {
                "success": False,
                "error": result.get("error", "Unknown error"),
                "timestamp": datetime.now().isoformat()
            }
        
        # Parse LLM response
        optimization = {
            "success": True,
            "symbol": symbol,
            "parameters": result["response"],
            "model": result["model"],
            "timestamp": datetime.now().isoformat()
        }
        
        # Update tracking variables
        self.decisions_history.append({
            "type": "parameter_optimization",
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "result": optimization
        })
        
        # Save history
        self._save_history()
        
        logger.info(f"Trading parameters optimization completed for {symbol}")
        
        return optimization
    
    async def evaluate_risk(self, symbol: str, position_size: float, entry_price: float) -> Dict[str, Any]:
        """
        Evaluate risk for a potential trade.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDC")
            position_size: Position size as percentage of capital
            entry_price: Entry price
            
        Returns:
            Risk evaluation result
        """
        # Get market data and portfolio from context
        market_data = self.context_manager.get_market_data(symbol)
        portfolio = self.context_manager.get_portfolio()
        
        # Calculate risk metrics
        capital = portfolio.get("total_capital", 0)
        position_value = capital * position_size
        current_exposure = portfolio.get("current_exposure", 0)
        total_exposure = current_exposure + position_value
        exposure_pct = total_exposure / capital if capital > 0 else 0
        
        # Prepare prompt for LLM
        prompt = f"""
        Evaluate risk for a potential {symbol} trade with the following parameters:
        
        Trade details:
        - Symbol: {symbol}
        - Position size: {position_size * 100}% of capital (${position_value:.2f})
        - Entry price: {entry_price}
        
        Portfolio:
        - Total capital: ${capital:.2f}
        - Current exposure: ${current_exposure:.2f} ({current_exposure / capital * 100 if capital > 0 else 0:.2f}% of capital)
        - Total exposure after trade: ${total_exposure:.2f} ({exposure_pct * 100:.2f}% of capital)
        
        Market conditions:
        - Current price: {market_data.get('price')}
        - 24h volatility: {market_data.get('volatility_24h')}%
        - Market trend: {market_data.get('trend')}
        
        Risk tolerance: {self.risk_tolerance}
        
        Based on this data, provide a risk assessment:
        1. Risk level (low, medium, high, extreme)
        2. Maximum recommended position size
        3. Suggested stop loss level
        4. Risk-reward ratio
        5. Risk mitigation recommendations
        """
        
        # Generate risk evaluation using LLM
        result = await self.llm_manager.generate_response(
            prompt,
            model_tier=1,  # Use tier 1 model for risk evaluation
            max_tokens=800
        )
        
        if not result["success"]:
            logger.error(f"Failed to evaluate risk: {result.get('error')}")
            return {
                "success": False,
                "error": result.get("error", "Unknown error"),
                "timestamp": datetime.now().isoformat()
            }
        
        # Parse LLM response
        evaluation = {
            "success": True,
            "symbol": symbol,
            "position_size": position_size,
            "entry_price": entry_price,
            "evaluation": result["response"],
            "model": result["model"],
            "timestamp": datetime.now().isoformat()
        }
        
        # Update tracking variables
        self.decisions_history.append({
            "type": "risk_evaluation",
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "result": evaluation
        })
        
        # Save history
        self._save_history()
        
        logger.info(f"Risk evaluation completed for {symbol} trade")
        
        return evaluation
    
    async def generate_trading_strategy(self, symbol: str, timeframe: str = "1d") -> Dict[str, Any]:
        """
        Generate a comprehensive trading strategy for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDC")
            timeframe: Strategy timeframe (e.g., "1h", "4h", "1d")
            
        Returns:
            Trading strategy result
        """
        # Get market data, trading history, and performance from context
        market_data = self.context_manager.get_market_data(symbol, timeframe)
        trading_history = self.context_manager.get_trading_history(symbol)
        performance = self.context_manager.get_performance_metrics(symbol)
        
        # Prepare prompt for LLM
        prompt = f"""
        Generate a comprehensive trading strategy for {symbol} on {timeframe} timeframe based on the following data:
        
        Market data:
        - Current price: {market_data.get('price')}
        - 24h Change: {market_data.get('change_24h')}%
        - Volume: {market_data.get('volume_24h')}
        - Market trend: {market_data.get('trend')}
        
        Performance metrics:
        - Win rate: {performance.get('win_rate')}%
        - Average profit: {performance.get('avg_profit')}%
        - Average loss: {performance.get('avg_loss')}%
        - Profit factor: {performance.get('profit_factor')}
        - Sharpe ratio: {performance.get('sharpe_ratio')}
        
        Based on this data, provide a comprehensive trading strategy:
        1. Overall market assessment
        2. Entry conditions (specific indicators and values)
        3. Exit conditions (specific indicators and values)
        4. Position sizing recommendations
        5. Risk management rules
        6. Timeframe-specific considerations
        7. Key levels to monitor
        """
        
        # Generate strategy using LLM
        result = await self.llm_manager.generate_response(
            prompt,
            model_tier=3,  # Use tier 3 model for comprehensive strategy
            max_tokens=1500
        )
        
        if not result["success"]:
            logger.error(f"Failed to generate trading strategy: {result.get('error')}")
            return {
                "success": False,
                "error": result.get("error", "Unknown error"),
                "timestamp": datetime.now().isoformat()
            }
        
        # Parse LLM response
        strategy = {
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "strategy": result["response"],
            "model": result["model"],
            "timestamp": datetime.now().isoformat()
        }
        
        # Update tracking variables
        self.current_strategy = strategy
        self.decisions_history.append({
            "type": "trading_strategy",
            "symbol": symbol,
            "timeframe": timeframe,
            "timestamp": datetime.now().isoformat(),
            "result": strategy
        })
        
        # Save history
        self._save_history()
        
        logger.info(f"Trading strategy generated for {symbol} ({timeframe})")
        
        return strategy
    
    async def evaluate_trade_opportunity(self, symbol: str, direction: str, entry_price: float, 
                                        position_size: float) -> Dict[str, Any]:
        """
        Evaluate a specific trade opportunity.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDC")
            direction: Trade direction ("buy" or "sell")
            entry_price: Entry price
            position_size: Position size as percentage of capital
            
        Returns:
            Trade opportunity evaluation result
        """
        # Get market data and current strategy from context
        market_data = self.context_manager.get_market_data(symbol)
        
        # Prepare prompt for LLM
        prompt = f"""
        Evaluate the following trade opportunity:
        
        Trade details:
        - Symbol: {symbol}
        - Direction: {direction.upper()}
        - Entry price: {entry_price}
        - Position size: {position_size * 100}% of capital
        
        Market conditions:
        - Current price: {market_data.get('price')}
        - 24h Change: {market_data.get('change_24h')}%
        - Volume: {market_data.get('volume_24h')}
        - Market trend: {market_data.get('trend')}
        
        Current strategy:
        {self.current_strategy['strategy'] if self.current_strategy else 'No current strategy'}
        
        Based on this data, evaluate this trade opportunity:
        1. Alignment with current strategy (0-100%)
        2. Probability of success (0-100%)
        3. Recommended entry price adjustments
        4. Recommended position size adjustments
        5. Suggested stop loss and take profit levels
        6. Overall recommendation (execute, adjust, or avoid)
        """
        
        # Generate evaluation using LLM
        result = await self.llm_manager.generate_response(
            prompt,
            model_tier=2,  # Use tier 2 model for trade evaluation
            max_tokens=1000
        )
        
        if not result["success"]:
            logger.error(f"Failed to evaluate trade opportunity: {result.get('error')}")
            return {
                "success": False,
                "error": result.get("error", "Unknown error"),
                "timestamp": datetime.now().isoformat()
            }
        
        # Parse LLM response
        evaluation = {
            "success": True,
            "symbol": symbol,
            "direction": direction,
            "entry_price": entry_price,
            "position_size": position_size,
            "evaluation": result["response"],
            "model": result["model"],
            "timestamp": datetime.now().isoformat()
        }
        
        # Update tracking variables
        self.decisions_history.append({
            "type": "trade_evaluation",
            "symbol": symbol,
            "direction": direction,
            "timestamp": datetime.now().isoformat(),
            "result": evaluation
        })
        
        # Save history
        self._save_history()
        
        logger.info(f"Trade opportunity evaluation completed for {direction.upper()} {symbol}")
        
        return evaluation
    
    def get_decision_history(self, decision_type: Optional[str] = None, 
                           limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get decision history.
        
        Args:
            decision_type: Filter by decision type (optional)
            limit: Maximum number of records to return
            
        Returns:
            Decision history records
        """
        if decision_type:
            filtered_history = [d for d in self.decisions_history if d["type"] == decision_type]
            return filtered_history[-limit:]
        
        return self.decisions_history[-limit:]
    
    def get_current_strategy(self) -> Optional[Dict[str, Any]]:
        """
        Get current trading strategy.
        
        Returns:
            Current trading strategy or None if not available
        """
        return self.current_strategy
    
    def set_risk_tolerance(self, risk_tolerance: str) -> None:
        """
        Set risk tolerance level.
        
        Args:
            risk_tolerance: Risk tolerance level ("low", "medium", or "high")
        """
        if risk_tolerance not in ["low", "medium", "high"]:
            logger.warning(f"Invalid risk tolerance: {risk_tolerance}, must be 'low', 'medium', or 'high'")
            return
        
        self.risk_tolerance = risk_tolerance
        logger.info(f"Risk tolerance set to {risk_tolerance}")
    
    def set_max_position_size(self, max_position_size: float) -> None:
        """
        Set maximum position size.
        
        Args:
            max_position_size: Maximum position size as percentage of capital (0.0 to 1.0)
        """
        if max_position_size < 0.0 or max_position_size > 1.0:
            logger.warning(f"Invalid max position size: {max_position_size}, must be between 0.0 and 1.0")
            return
        
        self.max_position_size = max_position_size
        logger.info(f"Maximum position size set to {max_position_size * 100:.0f}%")
