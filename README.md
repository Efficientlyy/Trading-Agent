# MEXC Trading System Documentation

## Overview

This is the comprehensive documentation for the MEXC Trading System, a modular automated trading platform designed to work with the MEXC exchange. This documentation covers all aspects of the system, from architecture and setup to development guidelines and operational procedures.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Getting Started](#getting-started)
3. [Development Environment](#development-environment)
4. [Core Components](#core-components)
5. [Paper Trading Mode](#paper-trading-mode)
6. [Performance Benchmarking](#performance-benchmarking)
7. [Monitoring and Observability](#monitoring-and-observability)
8. [Development Guidelines](#development-guidelines)
9. [Troubleshooting](#troubleshooting)

## System Architecture

The MEXC Trading System follows a modular architecture with the following key components:

- **Market Data Processor** (Rust): High-performance processing of real-time market data
- **Signal Generator** (Python): Technical analysis and signal generation
- **Decision Service** (Node.js): LLM-based trading decision making
- **Order Execution** (Rust): Reliable and efficient order execution
- **Dashboard** (React): Real-time visualization and monitoring

For detailed architecture information, see:
- [Modular Trading System Architecture](./docs/architecture/modular_trading_system_architecture.md)
- [MEXC API Component Mapping](./docs/architecture/mexc_api_component_mapping.md)

## Getting Started

### Prerequisites

- Docker Desktop with WSL2 (for Windows)
- Git
- Visual Studio Code with Remote Containers extension
- MEXC API keys (for live trading)

### Quick Start

1. Clone the repository
2. Open in VS Code with Remote Containers
3. Start the development environment
4. Access the dashboard at http://localhost:8080

For detailed setup instructions, see:
- [Development Environment Setup Guide](./docs/development/setup/setup_guide.md)

## Development Environment

The system uses a containerized development environment to ensure consistency across platforms:

- Docker Compose for service orchestration
- VS Code devcontainers for integrated development
- Cross-platform compatibility (Windows, macOS, Linux)

For detailed environment information, see:
- [Containerized Environment Setup Guide](./docs/development/setup/setup_guide.md)

## Core Components

### Market Data Processor (Rust)

High-performance component for processing real-time market data from MEXC:

- WebSocket connection management
- Order book maintenance
- Market data normalization
- gRPC API for other components

For implementation details, see:
- [Market Data Processor Overview](./docs/components/rust/market_data_processor.md)
- [Market Data Processor Implementation](./docs/components/rust/market_data_processor_implementation.md)

### Signal Generator (Python)

Technical analysis component for generating trading signals:

- Pattern recognition
- Indicator calculation
- Signal generation and publishing
- FastAPI interface

For implementation details, see:
- [Signal Generator Overview](./docs/components/python/signal_generator.md)

### Decision Service (Node.js)

LLM-based decision making component:

- Signal aggregation
- LLM integration for decision making
- Trading strategy implementation
- Express.js API

For implementation details, see:
- [Decision Service Overview](./docs/components/nodejs/decision_service.md)

### Order Execution (Rust)

Reliable and efficient order execution component:

- MEXC API integration
- Order management
- Position tracking
- Risk management

For implementation details, see:
- [Order Execution Implementation](./docs/components/rust/order_execution_implementation.md)

### Dashboard (React)

Real-time visualization and monitoring interface:

- Trading charts
- System status
- Performance metrics
- Trading controls

For implementation details, see:
- [Dashboard Overview](./docs/components/frontend/dashboard.md)

## Paper Trading Mode

The system includes a paper trading mode for safe development and testing:

- Real market data with simulated execution
- Virtual account management
- Realistic order matching
- Configurable via environment variables

For detailed information, see:
- [Paper Trading Design and Implementation](./docs/development/implementation/design_and_implementation.md)

## Performance Benchmarking

Comprehensive benchmarking tools for performance validation:

- Component-level benchmarks
- System-wide load testing
- Profiling tools
- Continuous performance testing

For detailed information, see:
- [Benchmarking Toolkit](./docs/development/benchmarking_toolkit.md)

## Monitoring and Observability

Complete monitoring solution for system visibility:

- Grafana dashboards
- Prometheus metrics
- Distributed tracing
- Alerting rules

For detailed information, see:
- [Grafana Dashboard Template](./docs/operations/monitoring/grafana_dashboard_template.md)

## Development Guidelines

Guidelines for consistent development:

- Coding standards
- Testing requirements
- Documentation practices
- Pull request process

For detailed information, see:
- [LLM Developer Guidelines](./docs/development/guidelines/llm_developer_guidelines.md)
- [Interoperability Guidelines](./docs/development/guidelines/interoperability_guidelines.md)

## Troubleshooting

Common issues and solutions:

- Environment setup problems
- Docker and WSL2 issues
- Component-specific troubleshooting
- Performance problems

For detailed information, see:
- [Containerized Environment Setup Guide](./docs/development/setup/setup_guide.md) (Troubleshooting section)

## Documentation Structure

The documentation is organized into the following directories:

- **[docs/architecture](./docs/architecture/)** - System architecture and design
- **[docs/components](./docs/components/)** - Component-specific documentation
- **[docs/development](./docs/development/)** - Development guides and implementation details
- **[docs/operations](./docs/operations/)** - Operational procedures and monitoring
- **[docs/reference](./docs/reference/)** - Reference materials and API documentation

For a complete index of all documentation, see:
- [Documentation Index](./docs/index.md)
- [Cross-Reference Map](./docs/cross_reference_map.md)
- [Documentation Guide](./docs/documentation_guide.md)

## Additional Resources

- [MEXC API Report](./docs/reference/mexc_api_report.md)
- [Implementation Priorities](./docs/reference/implementation_priorities.md)
- [Architecture and Frameworks](./docs/reference/architecture_and_frameworks.md)
