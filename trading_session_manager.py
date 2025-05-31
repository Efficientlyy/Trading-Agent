#!/usr/bin/env python
"""
Trading Session Manager

This module provides functionality to identify, track, and manage different
global trading sessions (Asia, Europe, US) for the flash trading system.
It enables session-specific parameter adjustments and performance tracking.
"""

import time
import logging
import json
import os
from datetime import datetime, timezone, timedelta
from threading import RLock
import pytz

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("trading_session_manager")

class TradingSession:
    """Represents a global trading session with specific time range and characteristics"""
    
    def __init__(self, name, start_hour_utc, end_hour_utc, description=None):
        """Initialize a trading session with UTC time boundaries"""
        self.name = name
        self.start_hour_utc = start_hour_utc
        self.end_hour_utc = end_hour_utc
        self.description = description or f"{name} Trading Session"
        
        # Performance metrics for this session
        self.metrics = {
            "trades_count": 0,
            "profitable_trades": 0,
            "losing_trades": 0,
            "total_profit_loss": 0.0,
            "win_rate": 0.0,
            "avg_profit_per_trade": 0.0,
            "avg_loss_per_trade": 0.0,
            "max_drawdown": 0.0,
            "volatility": 0.0
        }
    
    def is_active(self, current_hour_utc=None):
        """Check if this session is currently active"""
        if current_hour_utc is None:
            current_hour_utc = datetime.now(timezone.utc).hour
        
        # Handle sessions that span across midnight
        if self.start_hour_utc <= self.end_hour_utc:
            return self.start_hour_utc <= current_hour_utc < self.end_hour_utc
        else:
            return current_hour_utc >= self.start_hour_utc or current_hour_utc < self.end_hour_utc
    
    def get_progress(self, current_hour_utc=None, current_minute=None):
        """Get session progress as a percentage (0-100%)"""
        if current_hour_utc is None:
            now = datetime.now(timezone.utc)
            current_hour_utc = now.hour
            current_minute = now.minute
        
        # Calculate total minutes in session
        if self.start_hour_utc <= self.end_hour_utc:
            total_minutes = (self.end_hour_utc - self.start_hour_utc) * 60
        else:
            total_minutes = (24 - self.start_hour_utc + self.end_hour_utc) * 60
        
        # Calculate elapsed minutes
        if self.start_hour_utc <= current_hour_utc:
            elapsed_minutes = (current_hour_utc - self.start_hour_utc) * 60 + current_minute
        else:
            elapsed_minutes = (24 - self.start_hour_utc + current_hour_utc) * 60 + current_minute
        
        # Ensure elapsed minutes doesn't exceed total (in case of calculation edge cases)
        elapsed_minutes = min(elapsed_minutes, total_minutes)
        
        # Calculate progress percentage
        return (elapsed_minutes / total_minutes) * 100 if total_minutes > 0 else 0
    
    def update_metrics(self, trade_result):
        """Update session performance metrics with a new trade result"""
        self.metrics["trades_count"] += 1
        
        profit_loss = trade_result.get("profit_loss", 0.0)
        self.metrics["total_profit_loss"] += profit_loss
        
        if profit_loss > 0:
            self.metrics["profitable_trades"] += 1
        elif profit_loss < 0:
            self.metrics["losing_trades"] += 1
        
        # Update win rate
        if self.metrics["trades_count"] > 0:
            self.metrics["win_rate"] = (self.metrics["profitable_trades"] / self.metrics["trades_count"]) * 100
        
        # Update average profit/loss
        if self.metrics["profitable_trades"] > 0:
            self.metrics["avg_profit_per_trade"] = self.metrics["total_profit_loss"] / self.metrics["profitable_trades"]
        
        if self.metrics["losing_trades"] > 0:
            self.metrics["avg_loss_per_trade"] = self.metrics["total_profit_loss"] / self.metrics["losing_trades"]
    
    def to_dict(self):
        """Convert session to dictionary for serialization"""
        return {
            "name": self.name,
            "start_hour_utc": self.start_hour_utc,
            "end_hour_utc": self.end_hour_utc,
            "description": self.description,
            "metrics": self.metrics
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create session from dictionary"""
        session = cls(
            name=data["name"],
            start_hour_utc=data["start_hour_utc"],
            end_hour_utc=data["end_hour_utc"],
            description=data.get("description")
        )
        session.metrics = data.get("metrics", session.metrics)
        return session


class TradingSessionManager:
    """Manages global trading sessions and provides session-specific parameters"""
    
    def __init__(self, config_path=None):
        """Initialize the trading session manager"""
        self.sessions = {}
        self.session_parameters = {}
        self.current_session = None
        self.lock = RLock()
        
        # Default sessions - Modified for testing to ensure at least one session is always active
        # ASIA: 0-8 UTC, EUROPE: 8-16 UTC, US: 16-24 UTC (no gaps or overlaps)
        self.default_sessions = {
            "ASIA": TradingSession("ASIA", 0, 8, "Asian Trading Session (00:00-08:00 UTC)"),
            "EUROPE": TradingSession("EUROPE", 8, 16, "European Trading Session (08:00-16:00 UTC)"),
            "US": TradingSession("US", 16, 0, "US Trading Session (16:00-24:00 UTC)")  # Note: end_hour 0 means midnight
        }
        
        # Default session parameters
        self.default_parameters = {
            "ASIA": {
                "imbalance_threshold": 0.15,      # Lower threshold for typically lower volume
                "volatility_threshold": 0.12,      # Higher threshold for typically higher volatility
                "momentum_threshold": 0.04,        # Lower threshold for momentum
                "position_size_factor": 0.8,       # Smaller positions due to higher volatility
                "take_profit_bps": 25.0,           # Higher take profit due to higher volatility
                "stop_loss_bps": 15.0             # Higher stop loss due to higher volatility
            },
            "EUROPE": {
                "imbalance_threshold": 0.2,        # Standard threshold
                "volatility_threshold": 0.1,       # Standard threshold
                "momentum_threshold": 0.05,        # Standard threshold
                "position_size_factor": 1.0,       # Standard position size
                "take_profit_bps": 20.0,           # Standard take profit
                "stop_loss_bps": 10.0             # Standard stop loss
            },
            "US": {
                "imbalance_threshold": 0.25,       # Higher threshold for higher liquidity
                "volatility_threshold": 0.08,      # Lower threshold for typically lower volatility
                "momentum_threshold": 0.06,        # Higher threshold for stronger trends
                "position_size_factor": 1.2,       # Larger positions due to higher liquidity
                "take_profit_bps": 15.0,           # Lower take profit due to lower volatility
                "stop_loss_bps": 8.0              # Lower stop loss due to lower volatility
            }
        }
        
        # Session priority - custom sessions have higher priority than default sessions
        self.session_priority = {
            "ASIA": 10,
            "EUROPE": 20,
            "US": 30
        }
        
        # Initialize with default sessions and parameters
        self.sessions = self.default_sessions.copy()
        self.session_parameters = self.default_parameters.copy()
        
        # Load configuration if provided
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
        
        # Update current session
        self.update_current_session()
    
    def update_current_session(self):
        """Update the current active session based on UTC time"""
        with self.lock:
            current_hour_utc = datetime.now(timezone.utc).hour
            
            # Find active sessions
            active_sessions = []
            for session_name, session in self.sessions.items():
                if session.is_active(current_hour_utc):
                    active_sessions.append(session_name)
            
            # Determine primary session (if multiple are active)
            if len(active_sessions) == 1:
                self.current_session = active_sessions[0]
            elif len(active_sessions) > 1:
                # Sort by priority (higher priority wins)
                # Custom sessions (not in session_priority) get highest priority (1000)
                active_sessions.sort(key=lambda s: self.session_priority.get(s, 1000), reverse=True)
                self.current_session = active_sessions[0]
            else:
                # Default to closest upcoming session
                next_session = self._find_next_session(current_hour_utc)
                self.current_session = next_session
            
            logger.info(f"Current trading session: {self.current_session}")
            return self.current_session
    
    def _find_next_session(self, current_hour_utc):
        """Find the next upcoming session"""
        min_hours = 24
        next_session = None
        
        for session_name, session in self.sessions.items():
            # Calculate hours until session starts
            if current_hour_utc < session.start_hour_utc:
                hours_until = session.start_hour_utc - current_hour_utc
            else:
                hours_until = 24 - current_hour_utc + session.start_hour_utc
            
            if hours_until < min_hours:
                min_hours = hours_until
                next_session = session_name
        
        return next_session or "EUROPE"  # Default to EUROPE if no sessions defined
    
    def get_current_session_name(self):
        """Get the name of the current trading session"""
        self.update_current_session()
        return self.current_session
    
    def get_current_session(self):
        """Get the current trading session object"""
        session_name = self.get_current_session_name()
        return self.sessions.get(session_name)
    
    def get_session_parameter(self, parameter_name, default=None):
        """Get a parameter value for the current session"""
        with self.lock:
            session_name = self.get_current_session_name()
            session_params = self.session_parameters.get(session_name, {})
            return session_params.get(parameter_name, default)
    
    def get_all_session_parameters(self):
        """Get all parameters for the current session"""
        with self.lock:
            session_name = self.get_current_session_name()
            return self.session_parameters.get(session_name, {}).copy()
    
    def update_session_parameter(self, session_name, parameter_name, value):
        """Update a specific parameter for a session"""
        with self.lock:
            if session_name not in self.sessions:
                logger.warning(f"Session {session_name} not found")
                return False
            
            if session_name not in self.session_parameters:
                self.session_parameters[session_name] = {}
            
            self.session_parameters[session_name][parameter_name] = value
            return True
    
    def add_session(self, name, start_hour_utc, end_hour_utc, description=None):
        """Add a new trading session"""
        with self.lock:
            self.sessions[name] = TradingSession(name, start_hour_utc, end_hour_utc, description)
            if name not in self.session_parameters:
                # Copy parameters from most similar existing session
                if start_hour_utc >= 0 and start_hour_utc < 8:
                    self.session_parameters[name] = self.session_parameters.get("ASIA", {}).copy()
                elif start_hour_utc >= 8 and start_hour_utc < 16:
                    self.session_parameters[name] = self.session_parameters.get("EUROPE", {}).copy()
                else:
                    self.session_parameters[name] = self.session_parameters.get("US", {}).copy()
            
            # Set priority for custom session (higher than default sessions)
            if name not in self.session_priority:
                self.session_priority[name] = 1000  # Custom sessions get highest priority
            
            self.update_current_session()
            return True
    
    def remove_session(self, name):
        """Remove a trading session"""
        with self.lock:
            if name in self.sessions:
                del self.sessions[name]
                if name in self.session_parameters:
                    del self.session_parameters[name]
                if name in self.session_priority:
                    del self.session_priority[name]
                
                self.update_current_session()
                return True
            return False
    
    def record_trade_result(self, trade_result):
        """Record a trade result for the current session's metrics"""
        with self.lock:
            session_name = self.get_current_session_name()
            if session_name in self.sessions:
                self.sessions[session_name].update_metrics(trade_result)
                return True
            return False
    
    def get_session_metrics(self, session_name=None):
        """Get performance metrics for a specific session or current session"""
        with self.lock:
            if session_name is None:
                session_name = self.get_current_session_name()
            
            if session_name in self.sessions:
                return self.sessions[session_name].metrics.copy()
            return {}
    
    def get_all_sessions_metrics(self):
        """Get performance metrics for all sessions"""
        with self.lock:
            return {name: session.metrics.copy() for name, session in self.sessions.items()}
    
    def save_config(self, config_path):
        """Save session configuration to file"""
        with self.lock:
            config = {
                "sessions": {name: session.to_dict() for name, session in self.sessions.items()},
                "parameters": self.session_parameters,
                "priority": self.session_priority
            }
            
            try:
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                logger.info(f"Session configuration saved to {config_path}")
                return True
            except Exception as e:
                logger.error(f"Error saving session configuration: {str(e)}")
                return False
    
    def load_config(self, config_path):
        """Load session configuration from file"""
        with self.lock:
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Load sessions
                if "sessions" in config:
                    self.sessions = {}
                    for name, session_data in config["sessions"].items():
                        self.sessions[name] = TradingSession.from_dict(session_data)
                
                # Load parameters
                if "parameters" in config:
                    self.session_parameters = config["parameters"]
                
                # Load priority
                if "priority" in config:
                    self.session_priority = config["priority"]
                
                self.update_current_session()
                logger.info(f"Session configuration loaded from {config_path}")
                return True
            except Exception as e:
                logger.error(f"Error loading session configuration: {str(e)}")
                # Fall back to defaults
                self.sessions = self.default_sessions.copy()
                self.session_parameters = self.default_parameters.copy()
                self.session_priority = {"ASIA": 10, "EUROPE": 20, "US": 30}
                self.update_current_session()
                return False
    
    def reset_to_defaults(self):
        """Reset sessions and parameters to defaults"""
        with self.lock:
            self.sessions = self.default_sessions.copy()
            self.session_parameters = self.default_parameters.copy()
            self.session_priority = {"ASIA": 10, "EUROPE": 20, "US": 30}
            self.update_current_session()
            return True


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Trading Session Manager')
    parser.add_argument('--config', default="trading_session_config.json", help='Path to session configuration file')
    parser.add_argument('--save', action='store_true', help='Save default configuration to file')
    parser.add_argument('--reset', action='store_true', help='Reset to default configuration')
    
    args = parser.parse_args()
    
    # Create session manager
    session_manager = TradingSessionManager(args.config if not args.reset else None)
    
    # Save configuration if requested
    if args.save:
        session_manager.save_config(args.config)
    
    # Print current session information
    current_session_name = session_manager.get_current_session_name()
    current_session = session_manager.get_current_session()
    
    print(f"Current UTC time: {datetime.now(timezone.utc).strftime('%H:%M:%S')}")
    print(f"Current trading session: {current_session_name}")
    
    if current_session:
        print(f"Session period: {current_session.start_hour_utc:02d}:00-{current_session.end_hour_utc:02d}:00 UTC")
        print(f"Session progress: {current_session.get_progress():.1f}%")
        print(f"Session description: {current_session.description}")
    
    # Print session parameters
    print("\nSession Parameters:")
    params = session_manager.get_all_session_parameters()
    for param_name, value in params.items():
        print(f"  {param_name}: {value}")
    
    # Print all sessions
    print("\nAll Trading Sessions:")
    for name, session in session_manager.sessions.items():
        status = "ACTIVE" if session.is_active() else "INACTIVE"
        print(f"  {name}: {session.start_hour_utc:02d}:00-{session.end_hour_utc:02d}:00 UTC [{status}]")
