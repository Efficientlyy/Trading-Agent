# Node.js Decision Service Boilerplate

This is a boilerplate for the Decision Service component of the MEXC trading system, implemented in Node.js with TypeScript.

## Project Structure

```
decision-service/
├── src/
│   ├── config/           # Configuration management
│   │   ├── index.ts      # Configuration loader
│   │   └── validation.ts # Configuration validation
│   ├── controllers/      # API controllers
│   │   ├── decision.ts   # Decision controller
│   │   ├── health.ts     # Health check controller
│   │   └── index.ts      # Controller exports
│   ├── grpc/             # gRPC client implementations
│   │   ├── market-data.ts # Market data service client
│   │   ├── order-execution.ts # Order execution client
│   │   └── index.ts      # gRPC client exports
│   ├── llm/              # LLM integration
│   │   ├── client.ts     # LLM client
│   │   ├── prompts.ts    # LLM prompts
│   │   └── index.ts      # LLM module exports
│   ├── middleware/       # Express middleware
│   │   ├── error.ts      # Error handling middleware
│   │   ├── logging.ts    # Request logging middleware
│   │   └── index.ts      # Middleware exports
│   ├── models/           # Data models
│   │   ├── decision.ts   # Decision model
│   │   ├── signal.ts     # Trading signal model
│   │   └── index.ts      # Model exports
│   ├── routes/           # API routes
│   │   ├── decision.ts   # Decision routes
│   │   ├── health.ts     # Health check routes
│   │   └── index.ts      # Route exports
│   ├── services/         # Business logic services
│   │   ├── decision.ts   # Decision making service
│   │   ├── signal.ts     # Signal aggregation service
│   │   └── index.ts      # Service exports
│   ├── utils/            # Utility functions
│   │   ├── logger.ts     # Logging utility
│   │   ├── metrics.ts    # Metrics collection
│   │   └── index.ts      # Utility exports
│   ├── app.ts            # Express application setup
│   └── index.ts          # Application entry point
├── test/                 # Tests
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   └── setup.ts          # Test setup
├── proto/                # Protocol buffer definitions
│   ├── market_data.proto # Market data service definition
│   └── order_execution.proto # Order execution service definition
├── .dockerignore         # Docker ignore file
├── .env.example          # Example environment variables
├── .eslintrc.js          # ESLint configuration
├── .gitignore            # Git ignore file
├── .prettierrc           # Prettier configuration
├── Dockerfile            # Docker build configuration
├── jest.config.js        # Jest configuration
├── nodemon.json          # Nodemon configuration
├── package.json          # NPM package configuration
├── README.md             # Project documentation
├── tsconfig.json         # TypeScript configuration
└── tsconfig.build.json   # TypeScript build configuration
```

## Getting Started

### Prerequisites

- Node.js (v16 or later)
- npm or yarn
- Docker and Docker Compose (for containerized development)

### Development Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   npm install
   # or
   yarn install
   ```
3. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
4. Start the development server:
   ```bash
   npm run dev
   # or
   yarn dev
   ```

### Docker Development

For containerized development:

```bash
docker-compose up decision-service
```

## Configuration

The service is configured via environment variables:

- `NODE_ENV`: Environment (`development`, `test`, `production`)
- `PORT`: HTTP server port
- `LOG_LEVEL`: Logging level
- `MARKET_DATA_SERVICE_URL`: Market Data Service gRPC URL
- `ORDER_EXECUTION_URL`: Order Execution Service gRPC URL
- `RABBITMQ_URL`: RabbitMQ connection string
- `REDIS_URL`: Redis connection string
- `DATABASE_URL`: PostgreSQL connection string
- `LLM_API_KEY`: API key for LLM service
- `LLM_API_URL`: URL for LLM service

## Windows Development Notes

When developing on Windows, consider the following:

- Use WSL2 for better Docker performance
- Use forward slashes in paths, even on Windows
- Set Git to use LF line endings:
  ```bash
  git config --global core.autocrlf input
  ```
- Use the VSCode Remote - Containers extension for a consistent development experience
- If not using Docker, ensure Node.js is properly installed on Windows and included in your PATH
- For native Windows development, you may need to adjust some npm scripts in package.json to use Windows-compatible commands
