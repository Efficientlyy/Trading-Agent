#!/usr/bin/env python
"""
Test script to validate zero-fee handling for BTC/USDC trading pair

This script tests that the system correctly applies zero fees for BTC/USDC trades
while still applying normal fees for other trading pairs.
"""

import logging
import json
from mock_exchange_client import MockExchangeClient
from execution_optimization import Order, OrderType, OrderSide, OrderRouter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("zero_fee_test")

def test_zero_fee_for_btcusdc():
    """Test that BTC/USDC trades have zero fees"""
    logger.info("Starting zero-fee test for BTC/USDC")
    
    # Initialize mock client
    client = MockExchangeClient()
    
    # Test market order for BTC/USDC (should have zero fee)
    btc_market_order = client.create_market_order(
        symbol="BTC/USDC",
        side="buy",
        quantity=0.1
    )
    
    # Test market order for ETH/USDT (should have normal fee)
    eth_market_order = client.create_market_order(
        symbol="ETH/USDT",
        side="buy",
        quantity=1.0
    )
    
    # Test limit order for BTC/USDC (should have zero fee)
    btc_limit_order = client.create_limit_order(
        symbol="BTC/USDC",
        side="buy",
        quantity=0.1,
        price=50000.0
    )
    
    # Test limit order for ETH/USDT (should have normal fee)
    eth_limit_order = client.create_limit_order(
        symbol="ETH/USDT",
        side="buy",
        quantity=1.0,
        price=3000.0
    )
    
    # Validate fees
    logger.info(f"BTC/USDC market order fee: {btc_market_order['fee']['cost']}")
    logger.info(f"ETH/USDT market order fee: {eth_market_order['fee']['cost']}")
    logger.info(f"BTC/USDC limit order fee: {btc_limit_order['fee']['cost']}")
    logger.info(f"ETH/USDT limit order fee: {eth_limit_order['fee']['cost']}")
    
    # Assert zero fee for BTC/USDC
    assert btc_market_order['fee']['cost'] == 0.0, "BTC/USDC market order should have zero fee"
    assert btc_limit_order['fee']['cost'] == 0.0, "BTC/USDC limit order should have zero fee"
    
    # Assert normal fee for ETH/USDT
    assert eth_market_order['fee']['cost'] > 0.0, "ETH/USDT market order should have normal fee"
    assert eth_limit_order['fee']['cost'] > 0.0, "ETH/USDT limit order should have normal fee"
    
    logger.info("Zero-fee test for BTC/USDC passed!")
    
    # Test with OrderRouter
    logger.info("Testing OrderRouter with zero-fee handling")
    
    # Initialize OrderRouter with mock client
    router = OrderRouter(client_instance=client)
    
    # Create BTC/USDC order
    btc_order = Order(
        symbol="BTC/USDC",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=0.1
    )
    
    # Create ETH/USDT order
    eth_order = Order(
        symbol="ETH/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=1.0
    )
    
    # Submit orders
    btc_result = router.submit_order(btc_order)
    eth_result = router.submit_order(eth_order)
    
    # Save results to file for inspection
    with open("zero_fee_test_results.json", "w") as f:
        json.dump({
            "btc_market_order": btc_market_order,
            "eth_market_order": eth_market_order,
            "btc_limit_order": btc_limit_order,
            "eth_limit_order": eth_limit_order,
            "btc_router_order": btc_result.to_dict(),
            "eth_router_order": eth_result.to_dict()
        }, f, indent=2)
    
    logger.info("Test results saved to zero_fee_test_results.json")
    logger.info("All zero-fee tests completed successfully!")

if __name__ == "__main__":
    test_zero_fee_for_btcusdc()
