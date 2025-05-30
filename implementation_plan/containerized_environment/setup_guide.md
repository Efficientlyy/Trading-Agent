# Containerized Development Environment

This document provides a comprehensive guide for setting up and using the containerized development environment for the MEXC trading system, with special considerations for Windows development.

## Overview

The containerized development environment uses Docker Compose and VS Code devcontainers to provide a consistent, reproducible development experience across all platforms (Windows, macOS, and Linux). This approach ensures that all developers work with identical dependencies, configurations, and tooling regardless of their host operating system.

## Prerequisites

Before setting up the development environment, ensure you have the following installed:

### For Windows

1. **Docker Desktop for Windows**
   - Download and install from [Docker Hub](https://hub.docker.com/editions/community/docker-ce-desktop-windows)
   - Ensure WSL 2 integration is enabled for better performance
   - Allocate sufficient resources (at least 8GB RAM, 4 CPUs)

2. **Visual Studio Code**
   - Download and install from [VS Code website](https://code.visualstudio.com/)
   - Install the "Remote - Containers" extension

3. **Windows Subsystem for Linux (WSL 2)**
   - Install WSL 2 following [Microsoft's instructions](https://docs.microsoft.com/en-us/windows/wsl/install)
   - Install Ubuntu 20.04 or later from the Microsoft Store

4. **Git for Windows**
   - Download and install from [Git website](https://git-scm.com/download/win)
   - Configure Git to use LF line endings:
     ```bash
     git config --global core.autocrlf input
     ```

### For macOS and Linux

1. **Docker and Docker Compose**
   - macOS: Install Docker Desktop from [Docker Hub](https://hub.docker.com/editions/community/docker-ce-desktop-mac)
   - Linux: Follow the [Docker installation guide](https://docs.docker.com/engine/install/) for your distribution

2. **Visual Studio Code**
   - Download and install from [VS Code website](https://code.visualstudio.com/)
   - Install the "Remote - Containers" extension

## Repository Structure

The repository is organized as a monorepo with the following structure:

```
mexc-trading-system/
├── .devcontainer/                # VS Code devcontainer configuration
│   ├── devcontainer.json         # Main devcontainer configuration
│   ├── docker-compose.extend.yml # Docker Compose extension for development
│   └── post-create.sh            # Setup script that runs after container creation
├── docker/                       # Docker configuration files
│   ├── grafana/                  # Grafana configuration
│   ├── postgres/                 # PostgreSQL initialization scripts
│   └── prometheus/               # Prometheus configuration
├── rust/                         # Rust components
│   ├── market-data-processor/    # Market data processor service
│   └── order-execution/          # Order execution service
├── nodejs/                       # Node.js components
│   ├── decision-service/         # Decision making service
│   └── api-gateway/              # API gateway service
├── python/                       # Python components
│   └── signal-generator/         # Signal generator service
├── frontend/                     # Frontend components
│   └── dashboard/                # Trading dashboard
├── benchmarks/                   # System-wide benchmarks
├── docker-compose.yml            # Main Docker Compose configuration
├── docker-compose.benchmark.yml  # Docker Compose for benchmarking
└── README.md                     # Project documentation
```

## Getting Started

### Clone the Repository

```bash
git clone https://github.com/your-org/mexc-trading-system.git
cd mexc-trading-system
```

### Open in VS Code with Devcontainers

1. Open VS Code
2. Open the command palette (Ctrl+Shift+P or Cmd+Shift+P)
3. Select "Remote-Containers: Open Folder in Container..."
4. Navigate to the cloned repository and select it

VS Code will build and start the development containers, which may take several minutes the first time. The `post-create.sh` script will automatically set up the development environment.

### Manual Docker Compose Setup (Alternative)

If you prefer not to use VS Code devcontainers, you can manually start the development environment:

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down
```

## Development Workflow

### Working with Rust Components

```bash
# Enter the market-data-processor container
docker-compose exec market-data-processor bash

# Build the project
cargo build

# Run tests
cargo test

# Run the service
cargo run
```

### Working with Node.js Components

```bash
# Enter the decision-service container
docker-compose exec decision-service bash

# Install dependencies
npm install

# Run in development mode
npm run dev

# Run tests
npm test
```

### Working with Python Components

```bash
# Enter the signal-generator container
docker-compose exec signal-generator bash

# Activate virtual environment
source .venv/bin/activate

# Run the service
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run tests
pytest
```

### Working with Frontend Components

```bash
# Enter the dashboard container
docker-compose exec dashboard bash

# Install dependencies
npm install

# Run in development mode
npm run dev

# Build for production
npm run build
```

## Windows-Specific Considerations

### File Permissions

When using Docker with Windows, file permission issues can occur. The devcontainer configuration includes settings to mitigate these issues, but if you encounter permission problems:

1. Ensure you're using WSL 2 backend for Docker Desktop
2. Check that the files are owned by your user in WSL
3. If necessary, run the following command in the WSL terminal:
   ```bash
   sudo chown -R $(id -u):$(id -g) /path/to/mexc-trading-system
   ```

### Performance Optimization

For better performance on Windows:

1. Store the repository in the WSL filesystem, not the Windows filesystem
2. Increase the memory and CPU allocation for Docker Desktop
3. Use the dedicated volumes for dependencies (like `cargo-registry` and `node_modules`)
4. Consider using the WSL 2 terminal for commands instead of Windows Command Prompt

### Line Endings

To avoid issues with line endings on Windows:

1. Configure Git as mentioned in the prerequisites
2. Add a `.gitattributes` file to the repository:
   ```
   * text=auto eol=lf
   *.{cmd,[cC][mM][dD]} text eol=crlf
   *.{bat,[bB][aA][tT]} text eol=crlf
   ```
3. If you encounter issues with shell scripts, run:
   ```bash
   dos2unix /path/to/script.sh
   ```

## Accessing Services

Once the development environment is running, you can access the services at:

- Dashboard: http://localhost:8080
- API Gateway: http://localhost:3000
- Decision Service: http://localhost:3001
- Signal Generator: http://localhost:8000
- Grafana: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9090
- RabbitMQ Management: http://localhost:15672 (guest/guest)
- Jaeger UI: http://localhost:16686

## Troubleshooting

### Common Issues on Windows

1. **Docker containers fail to start**
   - Ensure WSL 2 is properly configured
   - Restart Docker Desktop
   - Check Windows Defender Firewall settings

2. **Slow performance**
   - Move the repository to the WSL filesystem
   - Increase Docker Desktop resource allocation
   - Ensure you're using the optimized volume mounts in docker-compose.yml

3. **Permission denied errors**
   - Fix permissions in WSL as described above
   - Ensure Docker has permission to access the necessary directories

4. **Port conflicts**
   - Check if any services are already using the required ports
   - Modify the port mappings in docker-compose.yml if needed

### General Troubleshooting

1. **Container fails to build**
   - Check the build logs: `docker-compose logs -f [service_name]`
   - Ensure all required files are present
   - Try rebuilding: `docker-compose build --no-cache [service_name]`

2. **Service crashes or doesn't start**
   - Check the logs: `docker-compose logs -f [service_name]`
   - Ensure environment variables are correctly set
   - Verify that dependent services are running

3. **Database connection issues**
   - Ensure PostgreSQL is running: `docker-compose ps postgres`
   - Check the connection string in the service's configuration
   - Try connecting manually: `docker-compose exec postgres psql -U postgres -d mexc_trading`

## Extending the Environment

### Adding New Services

To add a new service to the development environment:

1. Create a new directory for the service
2. Add a Dockerfile for the service
3. Add the service to docker-compose.yml
4. Add the service to .devcontainer/docker-compose.extend.yml if needed
5. Rebuild the containers: `docker-compose up -d --build`

### Customizing VS Code Settings

The devcontainer.json file includes VS Code settings and extensions. To customize:

1. Modify the "settings" section to change editor settings
2. Add or remove extensions in the "extensions" section
3. Rebuild the devcontainer to apply changes

## Conclusion

This containerized development environment provides a consistent, reproducible way to develop and test the MEXC trading system across all platforms, with special considerations for Windows development. By following this guide, you can set up and use the environment effectively, avoiding common issues and optimizing performance.
