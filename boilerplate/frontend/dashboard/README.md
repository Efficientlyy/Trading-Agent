# Dashboard Frontend Boilerplate

This is a boilerplate for the Dashboard frontend component of the MEXC trading system, implemented with React, TypeScript, and Vite.

## Project Structure

```
dashboard/
├── public/             # Static assets
│   ├── favicon.ico     # Favicon
│   └── robots.txt      # Robots file
├── src/
│   ├── api/            # API client code
│   │   ├── apiClient.ts # Axios client setup
│   │   ├── endpoints.ts # API endpoint definitions
│   │   └── index.ts    # API exports
│   ├── assets/         # Static assets for the app
│   │   ├── icons/      # Icon assets
│   │   └── images/     # Image assets
│   ├── components/     # Reusable components
│   │   ├── common/     # Common UI components
│   │   ├── charts/     # Chart components
│   │   ├── layout/     # Layout components
│   │   └── index.ts    # Component exports
│   ├── context/        # React context providers
│   │   ├── AuthContext.tsx # Authentication context
│   │   └── index.ts    # Context exports
│   ├── hooks/          # Custom React hooks
│   │   ├── useMarketData.ts # Hook for market data
│   │   ├── useWebSocket.ts  # Hook for WebSocket
│   │   └── index.ts    # Hook exports
│   ├── pages/          # Page components
│   │   ├── Dashboard/  # Dashboard page
│   │   ├── Markets/    # Markets page
│   │   ├── Signals/    # Signals page
│   │   ├── Settings/   # Settings page
│   │   └── index.ts    # Page exports
│   ├── store/          # State management
│   │   ├── marketStore.ts # Market data store
│   │   ├── signalStore.ts # Trading signals store
│   │   └── index.ts    # Store exports
│   ├── types/          # TypeScript type definitions
│   │   ├── market.ts   # Market data types
│   │   ├── signal.ts   # Signal types
│   │   └── index.ts    # Type exports
│   ├── utils/          # Utility functions
│   │   ├── formatting.ts # Data formatting utilities
│   │   ├── time.ts     # Time utilities
│   │   └── index.ts    # Utility exports
│   ├── App.tsx         # Main App component
│   ├── main.tsx        # Application entry point
│   └── vite-env.d.ts   # Vite environment types
├── .dockerignore       # Docker ignore file
├── .env.example        # Example environment variables
├── .eslintrc.cjs       # ESLint configuration
├── .gitignore          # Git ignore file
├── .prettierrc         # Prettier configuration
├── Dockerfile          # Docker build configuration
├── index.html          # HTML entry point
├── package.json        # NPM package configuration
├── README.md           # Project documentation
├── tsconfig.json       # TypeScript configuration
├── tsconfig.node.json  # TypeScript Node configuration
└── vite.config.ts      # Vite configuration
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
docker-compose up dashboard
```

## Configuration

The application is configured via environment variables:

- `VITE_API_URL`: API Gateway URL
- `VITE_WS_URL`: WebSocket server URL
- `VITE_ENVIRONMENT`: Environment (`development`, `test`, `production`)

## Building for Production

```bash
npm run build
# or
yarn build
```

The build artifacts will be stored in the `dist/` directory.

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
- When running Vite on Windows, you might need to use `cross-env` for environment variables
