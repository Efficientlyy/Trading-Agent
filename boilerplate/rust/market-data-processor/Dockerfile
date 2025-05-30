# Multi-stage build for the Market Data Processor
FROM rust:1.70-slim as builder

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create a new empty project
WORKDIR /app
RUN USER=root cargo new --bin market-data-processor
WORKDIR /app/market-data-processor

# Copy manifests
COPY Cargo.toml ./

# Build dependencies (this will be cached if dependencies don't change)
RUN cargo build --release
RUN rm src/*.rs

# Copy source code
COPY src ./src

# Build the application
RUN cargo build --release

# Runtime stage
FROM debian:bullseye-slim

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the binary from the builder stage
COPY --from=builder /app/market-data-processor/target/release/market-data-processor /usr/local/bin/

# Copy the static dashboard files
COPY dashboard/build /app/dashboard/build

# Set the working directory
WORKDIR /app

# Expose the port
EXPOSE 8080

# Run the binary
CMD ["market-data-processor"]
