# Paper Trading Dashboard

A modern React-based dashboard for monitoring and controlling the Paper Trading module of the Trading Agent system.

## Features

- **Real-time Portfolio Tracking**: Monitor balances, positions, and performance metrics
- **Market Data Visualization**: Interactive price charts with various timeframes
- **Order Management**: Place and cancel orders, view order history
- **Performance Analytics**: Track profit/loss, win rate, drawdown, and other key metrics
- **Paper Trading Configuration**: Customize paper trading settings to simulate different market conditions

## Getting Started

### Prerequisites

- Node.js (v14 or higher)
- npm or yarn
- Running instance of the Market Data Processor backend

### Installation

1. Navigate to the dashboard directory:
   ```bash
   cd market-data-processor/dashboard
   ```

2. Install dependencies:
   ```bash
   npm install
   # or
   yarn install
   ```

3. Start the development server:
   ```bash
   npm start
   # or
   yarn start
   ```

4. Open your browser and navigate to [http://localhost:3000](http://localhost:3000)

## Development

### API Integration

The dashboard communicates with the Rust backend through the API service defined in `src/services/apiService.js`. During development, the service provides mock data for testing without requiring a running backend.

When running in production, make sure the backend API server is running on the expected endpoint (default: `/api`).

### Component Structure

- **Layout**: Main application layout with navigation sidebar
- **Dashboard**: Main overview page with portfolio summary and real-time data
- **PortfolioSummary**: Displays account balances and allocations
- **PerformanceMetrics**: Shows key trading performance indicators
- **PriceChart**: Interactive chart for price visualization
- **ActiveOrders**: Displays and manages current open orders
- **RecentTrades**: Shows recent executed trades
- **OrderHistory**: Searchable history of all orders with filtering
- **PaperTradingSettings**: Configure paper trading parameters

## Building for Production

To build the dashboard for production:

```bash
npm run build
# or
yarn build
```

The build artifacts will be stored in the `build/` directory, ready to be served by a static file server or integrated with the Rust backend.

## Integration with the Rust Backend

The dashboard is designed to be served by the Market Data Processor's web server. After building the dashboard, the backend can serve these static files.

To enable this integration:

1. Build the dashboard as described above
2. Configure the Market Data Processor to serve the static files from the `build/` directory
3. The API endpoints should be implemented in the backend to match the expected format in `apiService.js`

## Docker Integration

For containerized deployment, the dashboard can be included in the same Docker image as the Market Data Processor. Update the Dockerfile to:

1. Build the React application
2. Copy the build artifacts to the appropriate location
3. Configure the backend to serve these files

## Customization

- **Theme**: The application uses a dark theme by default, which can be customized in `App.js`
- **Trading Pairs**: Add or remove trading pairs in the settings and API service
- **Charts**: Customize chart appearance and timeframes in the `PriceChart` component
