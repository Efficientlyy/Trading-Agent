#!/usr/bin/env python
"""
Local Render-like Environment Test Script

This script tests the Trading Agent system in a local environment that
simulates the Render deployment environment. It verifies that all components
load correctly and can communicate with each other.
"""

import os
import sys
import time
import logging
import subprocess
import signal
import threading
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import environment configuration
from env_config import load_env, is_production, is_paper_trading, get_port

class LocalRenderTest:
    """
    Local Render-like Environment Test.
    
    This class simulates a Render deployment environment locally to test
    all components of the Trading Agent system.
    """
    
    def __init__(self):
        """Initialize Local Render Test."""
        # Load environment variables
        if not load_env():
            logger.error("Failed to load environment variables")
            sys.exit(1)
        
        # Initialize process dictionary
        self.processes = {}
        
        # Initialize status
        self.status = {
            "dashboard": False,
            "trading_engine": False,
            "llm_overseer": False
        }
        
        logger.info("Local Render Test initialized")
    
    def start_dashboard(self):
        """Start the dashboard component."""
        logger.info("Starting dashboard component...")
        
        try:
            # Start dashboard process
            process = subprocess.Popen(
                ["python", "mexc_dashboard_production.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Store process
            self.processes["dashboard"] = process
            
            # Wait for startup
            time.sleep(2)
            
            # Check if process is still running
            if process.poll() is None:
                logger.info(f"Dashboard started successfully on port {get_port()}")
                self.status["dashboard"] = True
            else:
                stdout, stderr = process.communicate()
                logger.error(f"Dashboard failed to start: {stderr}")
                self.status["dashboard"] = False
        
        except Exception as e:
            logger.error(f"Error starting dashboard: {e}")
            self.status["dashboard"] = False
    
    def start_trading_engine(self):
        """Start the trading engine component."""
        logger.info("Starting trading engine component...")
        
        try:
            # Start trading engine process
            process = subprocess.Popen(
                ["python", "flash_trading.py", "--env", ".env-secure/.env", "--mode", "production"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Store process
            self.processes["trading_engine"] = process
            
            # Wait for startup
            time.sleep(2)
            
            # Check if process is still running
            if process.poll() is None:
                logger.info("Trading engine started successfully")
                self.status["trading_engine"] = True
            else:
                stdout, stderr = process.communicate()
                logger.error(f"Trading engine failed to start: {stderr}")
                self.status["trading_engine"] = False
        
        except Exception as e:
            logger.error(f"Error starting trading engine: {e}")
            self.status["trading_engine"] = False
    
    def start_llm_overseer(self):
        """Start the LLM overseer component."""
        logger.info("Starting LLM overseer component...")
        
        try:
            # Start LLM overseer process
            process = subprocess.Popen(
                ["python", "-m", "llm_overseer.main"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Store process
            self.processes["llm_overseer"] = process
            
            # Wait for startup
            time.sleep(2)
            
            # Check if process is still running
            if process.poll() is None:
                logger.info("LLM overseer started successfully")
                self.status["llm_overseer"] = True
            else:
                stdout, stderr = process.communicate()
                logger.error(f"LLM overseer failed to start: {stderr}")
                self.status["llm_overseer"] = False
        
        except Exception as e:
            logger.error(f"Error starting LLM overseer: {e}")
            self.status["llm_overseer"] = False
    
    def start_all(self):
        """Start all components."""
        logger.info("Starting all components...")
        
        # Start components in separate threads
        threads = [
            threading.Thread(target=self.start_dashboard),
            threading.Thread(target=self.start_trading_engine),
            threading.Thread(target=self.start_llm_overseer)
        ]
        
        # Start threads
        for thread in threads:
            thread.start()
        
        # Wait for threads to complete
        for thread in threads:
            thread.join()
        
        # Print status
        self.print_status()
    
    def stop_all(self):
        """Stop all components."""
        logger.info("Stopping all components...")
        
        # Stop each process
        for name, process in self.processes.items():
            if process.poll() is None:
                logger.info(f"Stopping {name}...")
                process.send_signal(signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning(f"{name} did not terminate gracefully, forcing...")
                    process.kill()
        
        logger.info("All components stopped")
    
    def print_status(self):
        """Print status of all components."""
        print("\n=== Component Status ===")
        
        all_success = True
        for name, status in self.status.items():
            status_str = "✅ RUNNING" if status else "❌ FAILED"
            print(f"{name}: {status_str}")
            all_success = all_success and status
        
        print("\nOverall Status: " + ("✅ ALL COMPONENTS RUNNING" if all_success else "❌ SOME COMPONENTS FAILED"))
        
        # Print environment info
        env_type = "Production" if is_production() else "Development"
        trading_mode = "Paper Trading" if is_paper_trading() else "Real Trading"
        print(f"\nEnvironment: {env_type}")
        print(f"Trading Mode: {trading_mode}")
        
        if all_success:
            print(f"\nDashboard URL: http://localhost:{get_port()}")
        
        print("\nPress Ctrl+C to stop all components")
    
    def run_test(self, duration=30):
        """
        Run the local Render test.
        
        Args:
            duration: Test duration in seconds
        """
        try:
            # Start all components
            self.start_all()
            
            # Wait for specified duration
            logger.info(f"Running test for {duration} seconds...")
            time.sleep(duration)
            
        except KeyboardInterrupt:
            logger.info("Test interrupted by user")
        
        finally:
            # Stop all components
            self.stop_all()
            
            # Print final status
            print("\n=== Final Status ===")
            all_success = all(self.status.values())
            print("Test Result: " + ("✅ PASSED" if all_success else "❌ FAILED"))
            
            # Return success status
            return all_success


if __name__ == "__main__":
    print("=== Local Render-like Environment Test ===\n")
    print("This script tests the Trading Agent system in a local environment")
    print("that simulates the Render deployment environment.\n")
    
    # Parse duration argument
    duration = 30
    if len(sys.argv) > 1:
        try:
            duration = int(sys.argv[1])
        except ValueError:
            print(f"Invalid duration: {sys.argv[1]}, using default of 30 seconds")
    
    # Create and run test
    test = LocalRenderTest()
    success = test.run_test(duration)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
