# Python Signal Generator Boilerplate

This is a boilerplate for the Signal Generator component of the MEXC trading system, implemented in Python with FastAPI.

## Project Structure

```
signal-generator/
├── app/
│   ├── api/              # API endpoints
│   │   ├── __init__.py   # API package initialization
│   │   ├── router.py     # FastAPI router
│   │   └── endpoints/    # API endpoint modules
│   │       ├── __init__.py
│   │       ├── health.py # Health check endpoints
│   │       └── signals.py # Signal endpoints
│   ├── core/             # Core application code
│   │   ├── __init__.py   # Core package initialization
│   │   ├── config.py     # Configuration management
│   │   └── logging.py    # Logging configuration
│   ├── grpc/             # gRPC client implementations
│   │   ├── __init__.py   # gRPC package initialization
│   │   └── market_data.py # Market data service client
│   ├── models/           # Data models
│   │   ├── __init__.py   # Models package initialization
│   │   └── signals.py    # Signal models
│   ├── services/         # Business logic services
│   │   ├── __init__.py   # Services package initialization
│   │   ├── technical_analysis.py # Technical analysis service
│   │   └── signal_publisher.py # Signal publishing service
│   ├── utils/            # Utility functions
│   │   ├── __init__.py   # Utils package initialization
│   │   └── metrics.py    # Metrics collection
│   ├── __init__.py       # Application package initialization
│   └── main.py           # Application entry point
├── tests/                # Tests
│   ├── __init__.py       # Tests package initialization
│   ├── conftest.py       # Test configuration
│   ├── test_api/         # API tests
│   └── test_services/    # Service tests
├── proto/                # Protocol buffer definitions
│   └── market_data.proto # Market data service definition
├── .dockerignore         # Docker ignore file
├── .env.example          # Example environment variables
├── .gitignore            # Git ignore file
├── Dockerfile            # Docker build configuration
├── pyproject.toml        # Python project configuration
├── README.md             # Project documentation
└── requirements.txt      # Python dependencies
```

## Getting Started

### Prerequisites

- Python 3.9 or later
- pip
- Docker and Docker Compose (for containerized development)

### Development Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```
3. Activate the virtual environment:
   - On Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - On Unix or MacOS:
     ```bash
     source .venv/bin/activate
     ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
6. Run the application:
   ```bash
   uvicorn app.main:app --reload
   ```

### Docker Development

For containerized development:

```bash
docker-compose up signal-generator
```

## Configuration

The service is configured via environment variables:

- `LOG_LEVEL`: Logging level
- `MARKET_DATA_SERVICE_URL`: Market Data Service gRPC URL
- `RABBITMQ_URL`: RabbitMQ connection string
- `REDIS_URL`: Redis connection string
- `DATABASE_URL`: PostgreSQL connection string

## Windows Development Notes

When developing on Windows, consider the following:

- Use WSL2 for better Docker performance
- Use forward slashes in paths, even on Windows
- Set Git to use LF line endings:
  ```bash
  git config --global core.autocrlf input
  ```
- Use the VSCode Remote - Containers extension for a consistent development experience
- For virtual environments on Windows, use `.venv\Scripts\activate` instead of `.venv/bin/activate`
- If you encounter issues with path separators in Python code, use `os.path.join()` instead of hardcoded slashes
- For native Windows development, ensure Python is properly installed and included in your PATH
