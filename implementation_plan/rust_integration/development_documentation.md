# MEXC Trading System Development Documentation

## Overview

This document serves as the main entry point for all documentation related to the MEXC trading system development. It provides links to all relevant documents and guides needed to start building the system.

## System Architecture

- [Modular Trading System Architecture](../architecture/modular_trading_system_architecture.md) - Comprehensive overview of the system's modular architecture
- [MEXC API Component Mapping](../architecture/mexc_api_component_mapping.md) - Detailed mapping between system components and MEXC API features
- [Rust Integration Architecture](./rust_architecture.md) - Architecture for integrating Rust components into the system

## Implementation Plans

### Core Implementation

- [Implementation Priorities](../implementation_priorities.md) - Core priorities and phased approach for development
- [Architecture and Frameworks](../architecture_and_frameworks.md) - Detailed technical architecture with framework recommendations
- [Backend Implementation Plan](../backend/step_by_step_implementation.md) - Step-by-step plan for backend development
- [Frontend Implementation Plan](../frontend/step_by_step_implementation.md) - Step-by-step plan for frontend development
- [LLM Developer Guidelines](../llm_developer_guidelines.md) - Guidelines for LLM-based development

### Rust Implementation

- [Market Data Processor Implementation](./market_data_processor_implementation.md) - Detailed implementation plan for the Rust-based market data processor
- [Order Execution Implementation](./order_execution_implementation.md) - Detailed implementation plan for the Rust-based order execution module
- [Interoperability Guidelines](./interoperability_guidelines.md) - Guidelines for ensuring seamless integration between Rust, Node.js, and Python components

## Getting Started

### Prerequisites

Before starting development, ensure you have the following installed:

1. **Rust Toolchain**
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. **Node.js (v16+)**
   ```bash
   curl -fsSL https://deb.nodesource.com/setup_16.x | sudo -E bash -
   sudo apt-get install -y nodejs
   ```

3. **Python (3.9+)**
   ```bash
   sudo apt-get install python3.9 python3.9-dev python3.9-venv
   ```

4. **Docker and Docker Compose**
   ```bash
   sudo apt-get install docker.io docker-compose
   ```

5. **Development Tools**
   ```bash
   sudo apt-get install build-essential git
   ```

### Setting Up the Development Environment

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-org/mexc-trading-system.git
   cd mexc-trading-system
   ```

2. **Set Up Rust Components**
   ```bash
   cd rust
   cargo build
   ```

3. **Set Up Node.js Components**
   ```bash
   cd ../nodejs
   npm install
   ```

4. **Set Up Python Components**
   ```bash
   cd ../python
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

5. **Start Development Environment**
   ```bash
   cd ..
   docker-compose up -d
   ```

## Development Workflow

### Rust Development

1. Navigate to the Rust component directory
   ```bash
   cd rust/market-data-processor
   ```

2. Run tests
   ```bash
   cargo test
   ```

3. Build in release mode
   ```bash
   cargo build --release
   ```

4. Run the component
   ```bash
   cargo run --release
   ```

### Node.js Development

1. Navigate to the Node.js component directory
   ```bash
   cd nodejs/decision-service
   ```

2. Run tests
   ```bash
   npm test
   ```

3. Start the development server
   ```bash
   npm run dev
   ```

### Python Development

1. Navigate to the Python component directory
   ```bash
   cd python/signal-generator
   ```

2. Activate the virtual environment
   ```bash
   source ../../venv/bin/activate
   ```

3. Run tests
   ```bash
   pytest
   ```

4. Start the component
   ```bash
   python main.py
   ```

## API Documentation

- [MEXC API Documentation](https://mexcdevelop.github.io/apidocs/spot_v3_en/) - Official MEXC API documentation
- [System API Documentation](./api_docs/README.md) - Documentation for internal system APIs

## Testing

- [Testing Strategy](./testing/testing_strategy.md) - Overall testing strategy for the system
- [Integration Testing](./testing/integration_testing.md) - Guide for integration testing between components
- [Performance Testing](./testing/performance_testing.md) - Guide for performance testing

## Deployment

- [Deployment Guide](./deployment/deployment_guide.md) - Guide for deploying the system
- [Docker Configuration](./deployment/docker_configuration.md) - Docker configuration for the system
- [Monitoring Setup](./deployment/monitoring_setup.md) - Setting up monitoring for the system

## Troubleshooting

- [Common Issues](./troubleshooting/common_issues.md) - Solutions for common development issues
- [Debugging Guide](./troubleshooting/debugging_guide.md) - Guide for debugging the system

## Next Steps

1. Review the [Modular Trading System Architecture](../architecture/modular_trading_system_architecture.md) to understand the overall system design
2. Familiarize yourself with the [MEXC API Component Mapping](../architecture/mexc_api_component_mapping.md) to understand how the system interacts with MEXC
3. Start with the [Market Data Processor Implementation](./market_data_processor_implementation.md) to begin building the Rust components
4. Follow the [Interoperability Guidelines](./interoperability_guidelines.md) to ensure seamless integration between components

## Support

For questions or issues, please contact the development team at dev-support@example.com or open an issue in the GitHub repository.
