# WSL2 Configuration Guide for MEXC Trading System

This comprehensive guide will help you set up Windows Subsystem for Linux 2 (WSL2) for optimal Docker performance with the MEXC Trading System.

## Table of Contents

1. [WSL2 Installation and Configuration](#wsl2-installation-and-configuration)
2. [Docker Desktop Integration](#docker-desktop-integration)
3. [Volume Mounting Best Practices](#volume-mounting-best-practices)
4. [Network Configuration](#network-configuration)
5. [Troubleshooting Common Issues](#troubleshooting-common-issues)

## WSL2 Installation and Configuration

### Prerequisites

- Windows 10 version 2004 and higher (Build 19041 and higher) or Windows 11
- Administrator access
- At least 8GB of RAM (16GB recommended)

### Step 1: Enable WSL2 Features

Open PowerShell as Administrator and run:

```powershell
# Enable Windows Subsystem for Linux
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart

# Enable Virtual Machine Platform
dism.exe /online /enable-feature /featurename:VirtualMachineplatform /all /norestart

# Restart your computer to complete the WSL installation
```

### Step 2: Set WSL2 as Default

After restarting, open PowerShell as Administrator and run:

```powershell
# Set WSL 2 as the default version
wsl --set-default-version 2
```

### Step 3: Install a Linux Distribution

1. Open the Microsoft Store
2. Search for "Ubuntu" (Ubuntu 20.04 LTS or newer recommended)
3. Click "Get" to install
4. Launch Ubuntu to complete setup
5. Create a username and password when prompted

### Step 4: Verify Installation

```powershell
# Check WSL version and installed distributions
wsl -l -v
```

Should show:
```
  NAME      STATE           VERSION
* Ubuntu    Running         2
```

### Step 5: Configure WSL2 Resources

Create a `.wslconfig` file in your Windows home directory (`%USERPROFILE%`) to limit resource usage:

```
[wsl2]
memory=8GB
processors=4
swap=2GB
```

Adjust values based on your system capabilities. For trading systems, allocate at least 8GB of memory for optimal performance.

## Docker Desktop Integration

### Step 1: Install Docker Desktop

1. Download Docker Desktop from [https://www.docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop)
2. Install with default settings
3. Restart your computer if prompted

### Step 2: Configure Docker Desktop to use WSL2

1. Open Docker Desktop
2. Go to Settings > General
3. Ensure "Use the WSL 2 based engine" is checked
4. Go to Settings > Resources > WSL Integration
5. Enable integration with your Ubuntu distribution
6. Click "Apply & Restart"

### Step 3: Verify Docker in WSL2

1. Open your Ubuntu terminal
2. Run: `docker --version`
3. Run: `docker-compose --version`

Both commands should return version information without errors.

### Step 4: Configure Docker Resources

In Docker Desktop:

1. Go to Settings > Resources
2. Allocate at least 6GB of memory (8GB+ recommended for the Trading System)
3. Allocate at least 4 CPU cores
4. Set swap to at least 2GB
5. Click "Apply & Restart"

## Volume Mounting Best Practices

### Understanding Path Translation

WSL2 mounts Windows drives under `/mnt/` directory. For example:

- Windows path: `C:\Users\username\Projects\Trading Agent`
- WSL2 path: `/mnt/c/Users/username/Projects/Trading Agent`

### Best Practice 1: Use Linux Paths in Docker Compose

When your project is stored on Windows filesystem:

```yaml
# GOOD: Use WSL2 paths in docker-compose.yml
volumes:
  - /mnt/c/Users/username/Projects/Trading-Agent/logs:/app/logs
  - /mnt/c/Users/username/Projects/Trading-Agent/config:/app/config
```

```yaml
# BAD: Don't use Windows paths directly
volumes:
  - C:\Users\username\Projects\Trading-Agent\logs:/app/logs  # Will not work correctly
```

### Best Practice 2: Use Environment Variables for Paths

```yaml
# BEST: Use environment variables for flexibility
volumes:
  - ${PWD}/logs:/app/logs
  - ${PWD}/config:/app/config
```

In WSL2, `${PWD}` will resolve to the correct path format automatically.

### Best Practice 3: Store Project in WSL2 Filesystem for Best Performance

For optimal I/O performance (crucial for trading systems):

1. Store project files in the WSL2 filesystem, not Windows
2. Access via: `\\wsl$\Ubuntu\home\username\projects` in Windows Explorer
3. Use standard Linux paths in docker-compose.yml:

```yaml
volumes:
  - ./logs:/app/logs  # Will use native Linux filesystem
  - ./config:/app/config
```

### Example Docker Compose for MEXC Trading System

```yaml
version: '3.8'
name: trading-agent-system

services:
  market-data-processor:
    build:
      context: .
      dockerfile: Dockerfile
    volumes:
      - ./logs:/app/logs
      - ./config:/app/config
      - ./data:/app/data
    ports:
      - "8080:8080"
      - "50051:50051"
    environment:
      - HTTP_SERVER_ADDR=0.0.0.0:8080
      - GRPC_SERVER_ADDR=0.0.0.0:50051
      - PAPER_TRADING=true

  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./monitoring/prometheus:/etc/prometheus
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana:latest
    volumes:
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning
      - grafana_data:/var/lib/grafana
    ports:
      - "3000:3000"
    depends_on:
      - prometheus

volumes:
  prometheus_data:
  grafana_data:

networks:
  default:
    driver: bridge
```

## Network Configuration

### Docker Networking in WSL2

WSL2 creates a virtual network adapter that bridges Windows and Linux environments, allowing Docker containers to communicate.

### Service Discovery Between Containers

Containers on the same Docker network can discover each other by service name:

```yaml
services:
  market-data-processor:
    # Configuration...
    depends_on:
      - prometheus
  
  prometheus:
    # Configuration...
```

The `market-data-processor` can reach `prometheus` at `http://prometheus:9090`.

### Port Forwarding Chain

When running services in Docker within WSL2:

1. Container port (e.g., 8080)
2. Exposed to WSL2 (e.g., 8080)
3. Automatically forwarded to Windows host (e.g., 8080)

No additional configuration needed for standard port forwarding.

### Fixed IP Addresses

If your trading system requires fixed IP addresses:

```yaml
networks:
  trading_net:
    driver: bridge
    ipam:
      config:
        - subnet: 172.28.0.0/16

services:
  market-data-processor:
    networks:
      trading_net:
        ipv4_address: 172.28.1.2
```

### Accessing Host Services from Containers

To access services running on your Windows host:

1. Use `host.docker.internal` as the hostname
2. Example: `http://host.docker.internal:8000`

### Example Prometheus Configuration for Service Discovery

```yaml
# monitoring/prometheus/prometheus.yml
scrape_configs:
  - job_name: 'market_data_processor'
    metrics_path: '/metrics'
    static_configs:
      - targets: ['market-data-processor:8080']
    scrape_interval: 5s
```

## Troubleshooting Common Issues

### Issue 1: Docker Containers Can't Access Internet

**Symptoms:**
- Container can't download packages
- API calls to external services fail

**Solutions:**
1. Check WSL2 DNS:
   ```bash
   cat /etc/resolv.conf
   ```
2. If DNS is missing, create `/etc/wsl.conf`:
   ```
   [network]
   generateResolvConf = true
   ```
3. Restart WSL: `wsl --shutdown` and relaunch terminal

### Issue 2: Volume Mounts Not Working

**Symptoms:**
- Files not appearing inside container
- Permission denied errors

**Solutions:**
1. Verify paths are correct in WSL2 format
2. Check if paths use `/mnt/c/` prefix for Windows drives
3. Verify file permissions:
   ```bash
   sudo chown -R $(id -u):$(id -g) ./data
   ```

### Issue 3: Poor Performance with Windows-Based Volumes

**Symptoms:**
- Slow container startup
- High I/O latency

**Solutions:**
1. Move project files to WSL2 filesystem
2. Update docker-compose.yml to use relative paths
3. Increase resource allocation in Docker Desktop settings

### Issue 4: WSL2 Using Too Much Memory

**Symptoms:**
- System slowdown
- WSL2 not releasing memory

**Solutions:**
1. Limit memory in `.wslconfig` as shown earlier
2. Release memory:
   ```powershell
   wsl --shutdown
   ```

### Issue 5: Container Health Checks Failing

**Symptoms:**
- Services marked unhealthy
- Dependent services not starting

**Solutions:**
1. Verify curl is installed in container:
   ```Dockerfile
   RUN apt-get update && \
       apt-get install -y curl && \
       rm -rf /var/lib/apt/lists/*
   ```
2. Check health check URLs are accessible within container
3. Increase health check timeout/retries:
   ```yaml
   healthcheck:
     test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
     interval: 30s
     timeout: 10s
     retries: 5
     start_period: 30s
   ```

## Verification Steps

After completing setup, verify your environment:

1. Open WSL2 terminal and navigate to project directory
2. Run `docker-compose build`
3. Run `docker-compose up -d`
4. Check container status: `docker-compose ps`
5. Verify service access:
   - Dashboard: http://localhost:8080
   - Prometheus: http://localhost:9090
   - Grafana: http://localhost:3000

If all services are running and accessible, your WSL2 environment is correctly configured for the MEXC Trading System.

## Additional Resources

- [Official WSL2 Documentation](https://docs.microsoft.com/en-us/windows/wsl/)
- [Docker Desktop WSL2 Backend](https://docs.docker.com/desktop/windows/wsl/)
- [WSL2 Linux Kernel GitHub](https://github.com/microsoft/WSL2-Linux-Kernel)
