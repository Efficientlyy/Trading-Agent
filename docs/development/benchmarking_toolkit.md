# Benchmarking and Profiling Toolkit

This document outlines the benchmarking and profiling toolkit for the MEXC trading system, with a focus on performance validation for Rust components and cross-component system performance.

## Overview

Performance is critical for a trading system, especially for components handling market data processing and order execution. This toolkit provides standardized approaches for:

1. **Microbenchmarking** - Testing individual functions and algorithms
2. **Component benchmarking** - Testing entire components under load
3. **System benchmarking** - Testing the entire system end-to-end
4. **Profiling** - Identifying performance bottlenecks
5. **Continuous performance testing** - Tracking performance over time

## Rust Component Benchmarking

### Criterion Framework

For Rust components, we use the Criterion framework for reliable, statistically sound benchmarks:

```rust
// In market-data-processor/benches/order_book_benchmarks.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use market_data_processor::models::order_book::{OrderBook, OrderBookEntry};

fn bench_order_book_update(c: &mut Criterion) {
    // Setup test data
    let mut order_book = OrderBook::new("BTCUSDT".to_string());
    let entries = vec![
        OrderBookEntry { price: "45000.0".parse().unwrap(), quantity: "1.5".parse().unwrap() },
        OrderBookEntry { price: "45001.0".parse().unwrap(), quantity: "2.5".parse().unwrap() },
        // More entries...
    ];
    
    // Benchmark the update operation
    c.bench_function("order_book_update_10_entries", |b| {
        b.iter(|| {
            let mut test_book = order_book.clone();
            black_box(test_book.update(black_box(entries.clone())));
        })
    });
}

criterion_group!(order_book_benches, bench_order_book_update);
criterion_main!(order_book_benches);
```

### Running Rust Benchmarks

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark
cargo bench -- order_book_update

# Run benchmarks with different sample sizes
cargo bench -- --sample-size 100
```

### Benchmark Configuration

Criterion configuration in `Cargo.toml`:

```toml
[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }

[[bench]]
name = "order_book_benchmarks"
harness = false

[[bench]]
name = "websocket_benchmarks"
harness = false
```

## Node.js Component Benchmarking

### Autocannon for HTTP Benchmarking

For Node.js API components, we use Autocannon:

```javascript
// In decision-service/benchmarks/api_benchmarks.js
const autocannon = require('autocannon');

async function runBenchmark() {
  const result = await autocannon({
    url: 'http://localhost:3001/api/decisions',
    connections: 10,
    duration: 10,
    method: 'POST',
    headers: {
      'content-type': 'application/json'
    },
    body: JSON.stringify({
      symbol: 'BTCUSDT',
      signals: [
        { type: 'technical', value: 0.75, direction: 'buy' },
        { type: 'sentiment', value: 0.6, direction: 'buy' }
      ]
    })
  });
  
  console.log(result);
}

runBenchmark();
```

### Running Node.js Benchmarks

```bash
# Install autocannon
npm install -g autocannon

# Run benchmark
node benchmarks/api_benchmarks.js
```

## Python Component Benchmarking

### Locust for Load Testing

For Python components, we use Locust for load testing:

```python
# In signal-generator/benchmarks/locustfile.py
from locust import HttpUser, task, between

class SignalGeneratorUser(HttpUser):
    wait_time = between(1, 3)
    
    @task
    def generate_signal(self):
        self.client.post("/signals", json={
            "symbol": "BTCUSDT",
            "signal_type": "technical",
            "direction": "buy",
            "strength": 0.75,
            "timestamp": 1620000000000,
            "metadata": {
                "indicator": "RSI",
                "value": 30
            }
        })
    
    @task
    def get_symbols(self):
        self.client.get("/symbols")
```

### Running Python Benchmarks

```bash
# Install locust
pip install locust

# Run benchmark
locust -f benchmarks/locustfile.py --headless -u 10 -r 1 -t 30s --host http://localhost:8000
```

## System-Wide Benchmarking with k6

For end-to-end system benchmarking, we use k6:

```javascript
// In benchmarks/system/trading_flow.js
import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  vus: 10,
  duration: '30s',
};

export default function() {
  // Step 1: Get market data
  const marketDataRes = http.get('http://localhost:3000/api/market/BTCUSDT');
  check(marketDataRes, {
    'market data status is 200': (r) => r.status === 200,
  });
  
  // Step 2: Generate signal
  const signalRes = http.post('http://localhost:3000/api/signals', JSON.stringify({
    symbol: 'BTCUSDT',
    signal_type: 'technical',
    direction: 'buy',
    strength: 0.75,
    timestamp: Date.now(),
    metadata: { indicator: 'RSI', value: 30 }
  }), { headers: { 'Content-Type': 'application/json' } });
  check(signalRes, {
    'signal status is 200': (r) => r.status === 200,
  });
  
  // Step 3: Make decision
  const decisionRes = http.post('http://localhost:3000/api/decisions', JSON.stringify({
    symbol: 'BTCUSDT',
    signals: [
      { type: 'technical', value: 0.75, direction: 'buy' }
    ]
  }), { headers: { 'Content-Type': 'application/json' } });
  check(decisionRes, {
    'decision status is 200': (r) => r.status === 200,
  });
  
  // Step 4: Place order
  const orderRes = http.post('http://localhost:3000/api/orders', JSON.stringify({
    symbol: 'BTCUSDT',
    side: 'BUY',
    type: 'MARKET',
    quantity: '0.001'
  }), { headers: { 'Content-Type': 'application/json' } });
  check(orderRes, {
    'order status is 200': (r) => r.status === 200,
  });
  
  sleep(1);
}
```

### Running System Benchmarks

```bash
# Install k6
# On Windows: choco install k6
# On Linux: snap install k6

# Run benchmark
k6 run benchmarks/system/trading_flow.js
```

## Profiling Tools

### Rust Profiling with perf and flamegraph

For Rust components, we use perf and flamegraph:

```bash
# Install perf (Linux)
sudo apt-get install linux-tools-common linux-tools-generic

# Install flamegraph
cargo install flamegraph

# Run profiling
cargo flamegraph --bin market-data-processor

# On Windows with WSL2
cargo flamegraph --bin market-data-processor --output market-data-processor-flamegraph.svg
```

### Node.js Profiling with Clinic.js

For Node.js components:

```bash
# Install clinic
npm install -g clinic

# Run profiling
clinic doctor -- node dist/index.js
```

### Python Profiling with py-spy

For Python components:

```bash
# Install py-spy
pip install py-spy

# Run profiling
py-spy record -o profile.svg -- python -m app.main
```

## Distributed Tracing with OpenTelemetry

For tracing across components:

```rust
// In Rust components
use opentelemetry::trace::{Tracer, TracerProvider};
use opentelemetry::sdk::trace::TracerProvider as SdkTracerProvider;
use opentelemetry::sdk::trace::Config;
use opentelemetry_jaeger::new_pipeline;

fn init_tracer() -> Result<impl Tracer, Box<dyn Error>> {
    let provider = SdkTracerProvider::builder()
        .with_config(Config::default())
        .build();
    let tracer = provider.get_tracer("market-data-processor");
    
    Ok(tracer)
}
```

```javascript
// In Node.js components
const { NodeTracerProvider } = require('@opentelemetry/node');
const { JaegerExporter } = require('@opentelemetry/exporter-jaeger');
const { registerInstrumentations } = require('@opentelemetry/instrumentation');
const { ExpressInstrumentation } = require('@opentelemetry/instrumentation-express');
const { HttpInstrumentation } = require('@opentelemetry/instrumentation-http');

const provider = new NodeTracerProvider();
const exporter = new JaegerExporter({
  serviceName: 'decision-service',
});

provider.addSpanProcessor(new SimpleSpanProcessor(exporter));
provider.register();

registerInstrumentations({
  instrumentations: [
    new HttpInstrumentation(),
    new ExpressInstrumentation(),
  ],
});
```

```python
# In Python components
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

trace.set_tracer_provider(
    TracerProvider(
        resource=Resource.create({SERVICE_NAME: "signal-generator"})
    )
)

jaeger_exporter = JaegerExporter(
    agent_host_name="jaeger",
    agent_port=6831,
)

trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(jaeger_exporter)
)

tracer = trace.get_tracer(__name__)
```

## Continuous Performance Testing

We integrate performance testing into the CI/CD pipeline:

```yaml
# In .github/workflows/performance.yml
name: Performance Testing

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
          
      - name: Run Rust benchmarks
        run: cargo bench -- --output-format bencher | tee output.txt
        
      - name: Store benchmark result
        uses: benchmark-action/github-action-benchmark@v1
        with:
          name: Rust Benchmark
          tool: 'cargo'
          output-file-path: output.txt
          github-token: ${{ secrets.GITHUB_TOKEN }}
          auto-push: true
          # Show alert with commit comment on detecting possible performance regression
          alert-threshold: '200%'
          comment-on-alert: true
          fail-on-alert: true
```

## Windows Development Considerations

When running benchmarks and profiling on Windows:

1. **Rust Benchmarks**: Use Criterion which works well on Windows
   ```bash
   cargo bench
   ```

2. **Flamegraphs on Windows**: Use WSL2 or the Windows-compatible approach
   ```bash
   # Using WSL2
   wsl cargo flamegraph --bin market-data-processor
   
   # Windows-native alternative
   cargo install cargo-instruments  # Uses ETW on Windows
   cargo instruments --bin market-data-processor --template cpu-time
   ```

3. **Node.js Profiling on Windows**: Use Clinic.js which is cross-platform
   ```bash
   npx clinic doctor -- node dist/index.js
   ```

4. **k6 on Windows**: Install via Chocolatey
   ```bash
   choco install k6
   k6 run benchmarks/system/trading_flow.js
   ```

5. **Docker-based benchmarking**: For consistent results across platforms
   ```bash
   docker-compose -f docker-compose.benchmark.yml up
   ```

## Example Benchmark Results

Here's an example of what benchmark results look like:

```
order_book_update_10_entries
                        time:   [12.345 µs 12.567 µs 12.789 µs]
                        thrpt:  [78.19 MiB/s 79.57 MiB/s 81.00 MiB/s]
```

## Interpreting Results

Guidelines for interpreting benchmark results:

1. **Latency targets**:
   - Market data processing: < 1ms per update
   - Signal generation: < 10ms per signal
   - Decision making: < 50ms per decision
   - Order execution: < 100ms per order

2. **Throughput targets**:
   - Market data processing: > 1000 updates/second
   - Signal generation: > 100 signals/second
   - Decision making: > 50 decisions/second
   - Order execution: > 10 orders/second

3. **Memory usage targets**:
   - Market data processor: < 500MB
   - Signal generator: < 300MB
   - Decision service: < 400MB
   - Order execution: < 300MB

## Conclusion

This benchmarking and profiling toolkit provides comprehensive tools for ensuring the MEXC trading system meets its performance requirements. By integrating these tools into the development workflow, we can identify and address performance issues early, ensuring the system remains responsive and efficient.
