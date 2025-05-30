import os
import logging
from typing import Dict, List, Optional, Union
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import grpc
import redis
import pika
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Signal Generator",
    description="Technical Analysis Signal Generator for MEXC Trading System",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
class Config:
    MARKET_DATA_SERVICE_URL = os.getenv("MARKET_DATA_SERVICE_URL", "localhost:50051")
    RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/mexc_trading")

config = Config()

# Models
class Signal(BaseModel):
    symbol: str
    signal_type: str
    direction: str  # "buy" or "sell"
    strength: float = Field(..., ge=0.0, le=1.0)
    timestamp: int
    metadata: Dict[str, Union[str, int, float, bool]] = {}

class SignalResponse(BaseModel):
    success: bool
    message: str
    signal_id: Optional[str] = None

# Services
class MarketDataService:
    def __init__(self):
        self.channel = grpc.insecure_channel(config.MARKET_DATA_SERVICE_URL)
        # Initialize gRPC client here
        # self.stub = market_data_pb2_grpc.MarketDataServiceStub(self.channel)
        
    def get_order_book(self, symbol: str, depth: int = 10):
        # Implement gRPC call to get order book
        # request = market_data_pb2.GetOrderBookRequest(symbol=symbol, depth=depth)
        # return self.stub.GetOrderBook(request)
        pass
    
    def get_recent_trades(self, symbol: str, limit: int = 100):
        # Implement gRPC call to get recent trades
        # request = market_data_pb2.GetRecentTradesRequest(symbol=symbol, limit=limit)
        # return self.stub.GetRecentTrades(request)
        pass

class SignalPublisher:
    def __init__(self):
        self.connection = pika.BlockingConnection(
            pika.URLParameters(config.RABBITMQ_URL)
        )
        self.channel = self.connection.channel()
        self.channel.exchange_declare(
            exchange="signals", exchange_type="direct", durable=True
        )
        
    def publish_signal(self, signal: Signal):
        self.channel.basic_publish(
            exchange="signals",
            routing_key=signal.signal_type,
            body=signal.json(),
            properties=pika.BasicProperties(
                delivery_mode=2,  # make message persistent
                content_type="application/json",
            ),
        )
        
    def close(self):
        if self.connection.is_open:
            self.connection.close()

# Dependencies
def get_market_data_service():
    service = MarketDataService()
    return service

def get_signal_publisher():
    publisher = SignalPublisher()
    try:
        yield publisher
    finally:
        publisher.close()

# Routes
@app.get("/")
async def root():
    return {"message": "Signal Generator API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/signals", response_model=SignalResponse)
async def create_signal(
    signal: Signal,
    publisher: SignalPublisher = Depends(get_signal_publisher)
):
    try:
        # Publish signal to message queue
        publisher.publish_signal(signal)
        
        # In a real implementation, you would also store the signal in a database
        # and return the ID
        
        return SignalResponse(
            success=True,
            message="Signal published successfully",
            signal_id="sample-id-123",  # Replace with actual ID
        )
    except Exception as e:
        logger.error(f"Error publishing signal: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/symbols")
async def get_symbols():
    # In a real implementation, you would fetch this from the market data service
    return {
        "symbols": [
            "BTCUSDT",
            "ETHUSDT",
            "SOLUSDT",
        ]
    }

# Main entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
