#!/usr/bin/env python
"""
Market analyzer module for LLM Strategic Overseer.

This module implements market analysis logic for identifying trends,
patterns, and trading opportunities.
"""

import os
import sys
import logging
import json
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class MarketAnalyzer:
    """
    Market analyzer for identifying trading opportunities.
    
    Implements market analysis logic for identifying trends, patterns,
    and trading opportunities based on order book data and market microstructure.
    """
    
    def __init__(self, config):
        """
        Initialize market analyzer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # Load market analyzer configuration
        self.analysis_timeframes = self.config.get("strategy.analysis_timeframes", ["1m", "5m", "15m", "1h", "4h", "1d"])
        self.min_order_book_depth = self.config.get("strategy.min_order_book_depth", 20)
        self.volume_significance_threshold = self.config.get("strategy.volume_significance_threshold", 1.5)
        
        # Initialize tracking variables
        self.market_states = {}
        self.support_resistance_levels = {}
        self.volume_profiles = {}
        self.last_analysis_time = {}
        
        # Load historical data if available
        self.data_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "market_analysis.json"
        )
        self._load_history()
        
        logger.info(f"Market analyzer initialized with {len(self.analysis_timeframes)} timeframes")
    
    def _load_history(self) -> None:
        """Load market analysis history from file."""
        os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
        
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    self.market_states = data.get("market_states", {})
                    self.support_resistance_levels = data.get("support_resistance_levels", {})
                    self.volume_profiles = data.get("volume_profiles", {})
                    
                    last_times = data.get("last_analysis_time", {})
                    for symbol, time_str in last_times.items():
                        self.last_analysis_time[symbol] = datetime.fromisoformat(time_str)
                    
                    logger.info(f"Loaded market analysis history for {len(self.market_states)} symbols")
            except Exception as e:
                logger.error(f"Error loading market analysis history: {e}")
    
    def _save_history(self) -> None:
        """Save market analysis history to file."""
        try:
            # Convert datetime objects to ISO format strings
            last_times = {}
            for symbol, time in self.last_analysis_time.items():
                last_times[symbol] = time.isoformat()
            
            data = {
                "market_states": self.market_states,
                "support_resistance_levels": self.support_resistance_levels,
                "volume_profiles": self.volume_profiles,
                "last_analysis_time": last_times
            }
            
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Saved market analysis history for {len(self.market_states)} symbols")
        except Exception as e:
            logger.error(f"Error saving market analysis history: {e}")
    
    def analyze_order_book(self, symbol: str, order_book: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze order book for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDC")
            order_book: Order book data
            
        Returns:
            Order book analysis result
        """
        # Extract bids and asks
        bids = order_book.get("bids", [])
        asks = order_book.get("asks", [])
        
        # Check if order book has sufficient depth
        if len(bids) < self.min_order_book_depth or len(asks) < self.min_order_book_depth:
            logger.warning(f"Order book for {symbol} has insufficient depth: {len(bids)} bids, {len(asks)} asks")
            return {
                "success": False,
                "error": "Insufficient order book depth",
                "symbol": symbol,
                "timestamp": datetime.now().isoformat()
            }
        
        # Calculate bid-ask spread
        best_bid = bids[0][0] if bids else 0
        best_ask = asks[0][0] if asks else 0
        spread = best_ask - best_bid
        spread_pct = (spread / best_bid) * 100 if best_bid > 0 else 0
        
        # Calculate order book imbalance
        bid_volume = sum(bid[1] for bid in bids[:self.min_order_book_depth])
        ask_volume = sum(ask[1] for ask in asks[:self.min_order_book_depth])
        total_volume = bid_volume + ask_volume
        imbalance = (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0
        
        # Identify significant levels
        significant_bids = []
        significant_asks = []
        
        # Group nearby levels
        grouped_bids = self._group_price_levels(bids)
        grouped_asks = self._group_price_levels(asks)
        
        # Find levels with significant volume
        for price, volume in grouped_bids:
            if volume > bid_volume / len(grouped_bids) * self.volume_significance_threshold:
                significant_bids.append((price, volume))
        
        for price, volume in grouped_asks:
            if volume > ask_volume / len(grouped_asks) * self.volume_significance_threshold:
                significant_asks.append((price, volume))
        
        # Update support and resistance levels
        if symbol not in self.support_resistance_levels:
            self.support_resistance_levels[symbol] = {
                "support": [],
                "resistance": []
            }
        
        # Add significant bid levels as support
        for price, volume in significant_bids:
            if price not in [level["price"] for level in self.support_resistance_levels[symbol]["support"]]:
                self.support_resistance_levels[symbol]["support"].append({
                    "price": price,
                    "volume": volume,
                    "strength": volume / bid_volume,
                    "timestamp": datetime.now().isoformat()
                })
        
        # Add significant ask levels as resistance
        for price, volume in significant_asks:
            if price not in [level["price"] for level in self.support_resistance_levels[symbol]["resistance"]]:
                self.support_resistance_levels[symbol]["resistance"].append({
                    "price": price,
                    "volume": volume,
                    "strength": volume / ask_volume,
                    "timestamp": datetime.now().isoformat()
                })
        
        # Limit the number of levels
        self.support_resistance_levels[symbol]["support"] = sorted(
            self.support_resistance_levels[symbol]["support"],
            key=lambda x: x["strength"],
            reverse=True
        )[:10]
        
        self.support_resistance_levels[symbol]["resistance"] = sorted(
            self.support_resistance_levels[symbol]["resistance"],
            key=lambda x: x["strength"],
            reverse=True
        )[:10]
        
        # Prepare analysis result
        analysis = {
            "success": True,
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "bid_ask_spread": {
                "absolute": spread,
                "percentage": spread_pct
            },
            "imbalance": imbalance,
            "significant_levels": {
                "support": significant_bids,
                "resistance": significant_asks
            },
            "market_depth": {
                "bid_volume": bid_volume,
                "ask_volume": ask_volume,
                "total_volume": total_volume
            }
        }
        
        # Update market state
        if symbol not in self.market_states:
            self.market_states[symbol] = {}
        
        self.market_states[symbol]["order_book"] = {
            "imbalance": imbalance,
            "spread": spread_pct,
            "timestamp": datetime.now().isoformat()
        }
        
        # Update last analysis time
        self.last_analysis_time[symbol] = datetime.now()
        
        # Save history
        self._save_history()
        
        logger.info(f"Order book analysis completed for {symbol}")
        
        return analysis
    
    def analyze_market_microstructure(self, symbol: str, tick_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze market microstructure for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDC")
            tick_data: Tick-by-tick trade data
            
        Returns:
            Market microstructure analysis result
        """
        # Check if tick data is sufficient
        if len(tick_data) < 100:
            logger.warning(f"Tick data for {symbol} is insufficient: {len(tick_data)} ticks")
            return {
                "success": False,
                "error": "Insufficient tick data",
                "symbol": symbol,
                "timestamp": datetime.now().isoformat()
            }
        
        # Extract trade information
        prices = [tick["price"] for tick in tick_data]
        volumes = [tick["volume"] for tick in tick_data]
        sides = [tick["side"] for tick in tick_data]
        timestamps = [datetime.fromisoformat(tick["timestamp"]) for tick in tick_data]
        
        # Calculate basic statistics
        avg_price = sum(prices) / len(prices)
        avg_volume = sum(volumes) / len(volumes)
        buy_volume = sum(vol for vol, side in zip(volumes, sides) if side == "buy")
        sell_volume = sum(vol for vol, side in zip(volumes, sides) if side == "sell")
        buy_count = sum(1 for side in sides if side == "buy")
        sell_count = sum(1 for side in sides if side == "sell")
        
        # Calculate time-based metrics
        time_diffs = [(timestamps[i] - timestamps[i-1]).total_seconds() for i in range(1, len(timestamps))]
        avg_time_between_trades = sum(time_diffs) / len(time_diffs) if time_diffs else 0
        
        # Calculate price impact
        price_changes = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
        volume_weighted_price_impact = sum(change * vol for change, vol in zip(price_changes, volumes[1:])) / sum(volumes[1:]) if volumes[1:] else 0
        
        # Calculate trade flow imbalance
        trade_flow_imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume) if (buy_volume + sell_volume) > 0 else 0
        
        # Update volume profile
        if symbol not in self.volume_profiles:
            self.volume_profiles[symbol] = {}
        
        # Group trades by price level
        price_levels = {}
        for price, volume in zip(prices, volumes):
            price_level = round(price, 2)  # Round to 2 decimal places for grouping
            if price_level not in price_levels:
                price_levels[price_level] = 0
            price_levels[price_level] += volume
        
        # Update volume profile with new data
        for price_level, volume in price_levels.items():
            if str(price_level) not in self.volume_profiles[symbol]:
                self.volume_profiles[symbol][str(price_level)] = 0
            self.volume_profiles[symbol][str(price_level)] += volume
        
        # Limit the number of price levels in the volume profile
        if len(self.volume_profiles[symbol]) > 100:
            # Keep only the top 100 levels by volume
            sorted_levels = sorted(self.volume_profiles[symbol].items(), key=lambda x: float(x[1]), reverse=True)[:100]
            self.volume_profiles[symbol] = {k: v for k, v in sorted_levels}
        
        # Update market state
        if symbol not in self.market_states:
            self.market_states[symbol] = {}
        
        self.market_states[symbol]["microstructure"] = {
            "trade_flow_imbalance": trade_flow_imbalance,
            "avg_time_between_trades": avg_time_between_trades,
            "volume_weighted_price_impact": volume_weighted_price_impact,
            "timestamp": datetime.now().isoformat()
        }
        
        # Prepare analysis result
        analysis = {
            "success": True,
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "basic_stats": {
                "avg_price": avg_price,
                "avg_volume": avg_volume,
                "buy_volume": buy_volume,
                "sell_volume": sell_volume,
                "buy_count": buy_count,
                "sell_count": sell_count
            },
            "time_metrics": {
                "avg_time_between_trades": avg_time_between_trades
            },
            "price_impact": {
                "volume_weighted": volume_weighted_price_impact
            },
            "trade_flow": {
                "imbalance": trade_flow_imbalance
            }
        }
        
        # Update last analysis time
        self.last_analysis_time[symbol] = datetime.now()
        
        # Save history
        self._save_history()
        
        logger.info(f"Market microstructure analysis completed for {symbol}")
        
        return analysis
    
    def analyze_market_regime(self, symbol: str, price_data: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Analyze market regime for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDC")
            price_data: Price data for different timeframes
            
        Returns:
            Market regime analysis result
        """
        # Check if price data is sufficient
        if not price_data or not all(len(prices) >= 20 for prices in price_data.values()):
            logger.warning(f"Price data for {symbol} is insufficient")
            return {
                "success": False,
                "error": "Insufficient price data",
                "symbol": symbol,
                "timestamp": datetime.now().isoformat()
            }
        
        # Calculate volatility for each timeframe
        volatility = {}
        for timeframe, prices in price_data.items():
            # Calculate returns
            returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
            
            # Calculate volatility (standard deviation of returns)
            mean_return = sum(returns) / len(returns)
            variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
            std_dev = variance ** 0.5
            
            volatility[timeframe] = std_dev * 100  # Convert to percentage
        
        # Calculate trend strength for each timeframe
        trend_strength = {}
        for timeframe, prices in price_data.items():
            # Simple trend strength calculation
            price_change = (prices[-1] - prices[0]) / prices[0] * 100
            max_drawdown = 0
            peak = prices[0]
            
            for price in prices:
                if price > peak:
                    peak = price
                drawdown = (peak - price) / peak * 100
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
            
            # Trend strength is positive price change minus maximum drawdown
            trend_strength[timeframe] = abs(price_change) - max_drawdown
        
        # Determine market regime for each timeframe
        regimes = {}
        for timeframe in price_data.keys():
            vol = volatility.get(timeframe, 0)
            strength = trend_strength.get(timeframe, 0)
            
            # Classify regime based on volatility and trend strength
            if vol < 1.0:  # Low volatility
                if strength > 5.0:
                    regimes[timeframe] = "trending_low_vol"
                else:
                    regimes[timeframe] = "ranging_low_vol"
            elif vol < 3.0:  # Medium volatility
                if strength > 10.0:
                    regimes[timeframe] = "trending_medium_vol"
                else:
                    regimes[timeframe] = "ranging_medium_vol"
            else:  # High volatility
                if strength > 15.0:
                    regimes[timeframe] = "trending_high_vol"
                else:
                    regimes[timeframe] = "ranging_high_vol"
        
        # Update market state
        if symbol not in self.market_states:
            self.market_states[symbol] = {}
        
        self.market_states[symbol]["regime"] = {
            "regimes": regimes,
            "volatility": volatility,
            "trend_strength": trend_strength,
            "timestamp": datetime.now().isoformat()
        }
        
        # Prepare analysis result
        analysis = {
            "success": True,
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "volatility": volatility,
            "trend_strength": trend_strength,
            "regimes": regimes
        }
        
        # Update last analysis time
        self.last_analysis_time[symbol] = datetime.now()
        
        # Save history
        self._save_history()
        
        logger.info(f"Market regime analysis completed for {symbol}")
        
        return analysis
    
    def get_market_state(self, symbol: str) -> Dict[str, Any]:
        """
        Get current market state for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDC")
            
        Returns:
            Current market state
        """
        if symbol not in self.market_states:
            return {
                "success": False,
                "error": f"No market state available for {symbol}",
                "symbol": symbol,
                "timestamp": datetime.now().isoformat()
            }
        
        return {
            "success": True,
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "state": self.market_states[symbol]
        }
    
    def get_support_resistance_levels(self, symbol: str) -> Dict[str, Any]:
        """
        Get support and resistance levels for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDC")
            
        Returns:
            Support and resistance levels
        """
        if symbol not in self.support_resistance_levels:
            return {
                "success": False,
                "error": f"No support/resistance levels available for {symbol}",
                "symbol": symbol,
                "timestamp": datetime.now().isoformat()
            }
        
        return {
            "success": True,
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "levels": self.support_resistance_levels[symbol]
        }
    
    def get_volume_profile(self, symbol: str) -> Dict[str, Any]:
        """
        Get volume profile for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDC")
            
        Returns:
            Volume profile
        """
        if symbol not in self.volume_profiles:
            return {
                "success": False,
                "error": f"No volume profile available for {symbol}",
                "symbol": symbol,
                "timestamp": datetime.now().isoformat()
            }
        
        return {
            "success": True,
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "profile": self.volume_profiles[symbol]
        }
    
    def _group_price_levels(self, levels: List[List[float]], group_threshold: float = 0.001) -> List[List[float]]:
        """
        Group nearby price levels.
        
        Args:
            levels: List of [price, volume] pairs
            group_threshold: Threshold for grouping (as percentage of price)
            
        Returns:
            Grouped price levels
        """
        if not levels:
            return []
        
        # Sort by price
        sorted_levels = sorted(levels, key=lambda x: x[0])
        
        # Group nearby levels
        grouped_levels = []
        current_group = [sorted_levels[0][0], sorted_levels[0][1]]
        
        for price, volume in sorted_levels[1:]:
            # If price is within threshold of current group, add to group
            if abs(price - current_group[0]) / current_group[0] <= group_threshold:
                # Update group price (weighted average)
                total_volume = current_group[1] + volume
                current_group[0] = (current_group[0] * current_group[1] + price * volume) / total_volume
                current_group[1] = total_volume
            else:
                # Add current group to result and start a new group
                grouped_levels.append(current_group)
                current_group = [price, volume]
        
        # Add the last group
        grouped_levels.append(current_group)
        
        return grouped_levels
