# Grafana Monitoring Dashboard Template

This document provides a comprehensive Grafana monitoring dashboard template for the MEXC trading system, covering system metrics, trading performance, and business operations.

## Overview

Effective monitoring is critical for a trading system to ensure optimal performance, detect issues early, and track trading outcomes. The provided Grafana dashboard templates offer a complete monitoring solution that integrates with the containerized development environment.

## Dashboard Components

The monitoring solution includes several specialized dashboards:

1. **System Overview** - High-level health and performance metrics
2. **Market Data Pipeline** - Metrics for market data acquisition and processing
3. **Trading Performance** - Trading outcomes, execution quality, and P&L
4. **Technical Analysis** - Signal generation and pattern recognition metrics
5. **Infrastructure** - Detailed system resource utilization

## Setup Instructions

### Prerequisites

- Running Docker Compose environment with:
  - Prometheus for metrics collection
  - Grafana for visualization
  - Node exporters for system metrics
  - Application metrics exporters

### Import Dashboards

1. Access Grafana at http://localhost:3000 (default credentials: admin/admin)
2. Navigate to Dashboards > Import
3. Upload the JSON files from the `grafana/dashboards` directory or paste their contents
4. Select the appropriate Prometheus data source
5. Click Import

## Dashboard Details

### 1. System Overview Dashboard

![System Overview](https://example.com/system_overview.png)

This dashboard provides a high-level view of the entire system's health and performance:

- System uptime and service status
- Error rates and alert status
- Key performance indicators
- Resource utilization summary
- Recent trading activity

```json
{
  "annotations": {
    "list": [
      {
        "builtIn": 1,
        "datasource": "-- Grafana --",
        "enable": true,
        "hide": true,
        "iconColor": "rgba(0, 211, 255, 1)",
        "name": "Annotations & Alerts",
        "type": "dashboard"
      }
    ]
  },
  "editable": true,
  "gnetId": null,
  "graphTooltip": 0,
  "id": 1,
  "links": [],
  "panels": [
    {
      "datasource": null,
      "gridPos": {
        "h": 8,
        "w": 12,
        "x": 0,
        "y": 0
      },
      "id": 2,
      "options": {
        "content": "# MEXC Trading System\n\nSystem overview dashboard showing key metrics and status information.",
        "mode": "markdown"
      },
      "pluginVersion": "7.5.5",
      "title": "Dashboard Overview",
      "type": "text"
    },
    {
      "datasource": "Prometheus",
      "description": "",
      "fieldConfig": {
        "defaults": {
          "color": {
            "mode": "thresholds"
          },
          "mappings": [],
          "thresholds": {
            "mode": "absolute",
            "steps": [
              {
                "color": "green",
                "value": null
              },
              {
                "color": "red",
                "value": 80
              }
            ]
          }
        },
        "overrides": []
      },
      "gridPos": {
        "h": 8,
        "w": 12,
        "x": 12,
        "y": 0
      },
      "id": 4,
      "options": {
        "colorMode": "value",
        "graphMode": "area",
        "justifyMode": "auto",
        "orientation": "auto",
        "reduceOptions": {
          "calcs": [
            "lastNotNull"
          ],
          "fields": "",
          "values": false
        },
        "text": {},
        "textMode": "auto"
      },
      "pluginVersion": "7.5.5",
      "targets": [
        {
          "exemplar": true,
          "expr": "sum(up{job=~\".*\"})",
          "interval": "",
          "legendFormat": "",
          "refId": "A"
        }
      ],
      "title": "Services Up",
      "type": "stat"
    }
  ],
  "schemaVersion": 27,
  "style": "dark",
  "tags": [],
  "templating": {
    "list": []
  },
  "time": {
    "from": "now-6h",
    "to": "now"
  },
  "timepicker": {},
  "timezone": "",
  "title": "System Overview",
  "uid": "system-overview",
  "version": 1
}
```

### 2. Market Data Pipeline Dashboard

This dashboard focuses on the market data acquisition and processing pipeline:

- WebSocket connection status
- Market data throughput and latency
- Order book update rates
- Data processing queue lengths
- Error rates by data type

```json
{
  "annotations": {
    "list": [
      {
        "builtIn": 1,
        "datasource": "-- Grafana --",
        "enable": true,
        "hide": true,
        "iconColor": "rgba(0, 211, 255, 1)",
        "name": "Annotations & Alerts",
        "type": "dashboard"
      }
    ]
  },
  "editable": true,
  "gnetId": null,
  "graphTooltip": 0,
  "id": 2,
  "links": [],
  "panels": [
    {
      "aliasColors": {},
      "bars": false,
      "dashLength": 10,
      "dashes": false,
      "datasource": "Prometheus",
      "fieldConfig": {
        "defaults": {},
        "overrides": []
      },
      "fill": 1,
      "fillGradient": 0,
      "gridPos": {
        "h": 8,
        "w": 12,
        "x": 0,
        "y": 0
      },
      "hiddenSeries": false,
      "id": 2,
      "legend": {
        "avg": false,
        "current": false,
        "max": false,
        "min": false,
        "show": true,
        "total": false,
        "values": false
      },
      "lines": true,
      "linewidth": 1,
      "nullPointMode": "null",
      "options": {
        "alertThreshold": true
      },
      "percentage": false,
      "pluginVersion": "7.5.5",
      "pointradius": 2,
      "points": false,
      "renderer": "flot",
      "seriesOverrides": [],
      "spaceLength": 10,
      "stack": false,
      "steppedLine": false,
      "targets": [
        {
          "exemplar": true,
          "expr": "rate(market_data_messages_received_total[1m])",
          "interval": "",
          "legendFormat": "{{instance}}",
          "refId": "A"
        }
      ],
      "thresholds": [],
      "timeFrom": null,
      "timeRegions": [],
      "timeShift": null,
      "title": "Market Data Messages Rate",
      "tooltip": {
        "shared": true,
        "sort": 0,
        "value_type": "individual"
      },
      "type": "graph",
      "xaxis": {
        "buckets": null,
        "mode": "time",
        "name": null,
        "show": true,
        "values": []
      },
      "yaxes": [
        {
          "format": "short",
          "label": null,
          "logBase": 1,
          "max": null,
          "min": null,
          "show": true
        },
        {
          "format": "short",
          "label": null,
          "logBase": 1,
          "max": null,
          "min": null,
          "show": true
        }
      ],
      "yaxis": {
        "align": false,
        "alignLevel": null
      }
    }
  ],
  "schemaVersion": 27,
  "style": "dark",
  "tags": [],
  "templating": {
    "list": []
  },
  "time": {
    "from": "now-6h",
    "to": "now"
  },
  "timepicker": {},
  "timezone": "",
  "title": "Market Data Pipeline",
  "uid": "market-data-pipeline",
  "version": 1
}
```

### 3. Trading Performance Dashboard

This dashboard tracks trading outcomes and execution quality:

- P&L by trading pair
- Win/loss ratio
- Order execution latency
- Slippage metrics
- Position size and exposure
- Trading volume

```json
{
  "annotations": {
    "list": [
      {
        "builtIn": 1,
        "datasource": "-- Grafana --",
        "enable": true,
        "hide": true,
        "iconColor": "rgba(0, 211, 255, 1)",
        "name": "Annotations & Alerts",
        "type": "dashboard"
      }
    ]
  },
  "editable": true,
  "gnetId": null,
  "graphTooltip": 0,
  "id": 3,
  "links": [],
  "panels": [
    {
      "datasource": "Prometheus",
      "fieldConfig": {
        "defaults": {
          "color": {
            "mode": "palette-classic"
          },
          "custom": {
            "axisLabel": "",
            "axisPlacement": "auto",
            "barAlignment": 0,
            "drawStyle": "line",
            "fillOpacity": 10,
            "gradientMode": "none",
            "hideFrom": {
              "legend": false,
              "tooltip": false,
              "viz": false
            },
            "lineInterpolation": "linear",
            "lineWidth": 1,
            "pointSize": 5,
            "scaleDistribution": {
              "type": "linear"
            },
            "showPoints": "never",
            "spanNulls": true,
            "stacking": {
              "group": "A",
              "mode": "none"
            },
            "thresholdsStyle": {
              "mode": "off"
            }
          },
          "mappings": [],
          "thresholds": {
            "mode": "absolute",
            "steps": [
              {
                "color": "green",
                "value": null
              },
              {
                "color": "red",
                "value": 80
              }
            ]
          },
          "unit": "currencyUSD"
        },
        "overrides": []
      },
      "gridPos": {
        "h": 8,
        "w": 12,
        "x": 0,
        "y": 0
      },
      "id": 2,
      "options": {
        "legend": {
          "calcs": [],
          "displayMode": "list",
          "placement": "bottom"
        },
        "tooltip": {
          "mode": "single"
        }
      },
      "pluginVersion": "7.5.5",
      "targets": [
        {
          "exemplar": true,
          "expr": "trading_pnl{symbol=\"BTCUSDT\"}",
          "interval": "",
          "legendFormat": "BTC/USDT",
          "refId": "A"
        },
        {
          "exemplar": true,
          "expr": "trading_pnl{symbol=\"ETHUSDT\"}",
          "hide": false,
          "interval": "",
          "legendFormat": "ETH/USDT",
          "refId": "B"
        }
      ],
      "title": "P&L by Trading Pair",
      "type": "timeseries"
    }
  ],
  "schemaVersion": 27,
  "style": "dark",
  "tags": [],
  "templating": {
    "list": []
  },
  "time": {
    "from": "now-24h",
    "to": "now"
  },
  "timepicker": {},
  "timezone": "",
  "title": "Trading Performance",
  "uid": "trading-performance",
  "version": 1
}
```

### 4. Technical Analysis Dashboard

This dashboard monitors signal generation and pattern recognition:

- Signal counts by type
- Signal accuracy metrics
- Pattern detection rates
- Indicator values over time
- Signal-to-noise ratio

```json
{
  "annotations": {
    "list": [
      {
        "builtIn": 1,
        "datasource": "-- Grafana --",
        "enable": true,
        "hide": true,
        "iconColor": "rgba(0, 211, 255, 1)",
        "name": "Annotations & Alerts",
        "type": "dashboard"
      }
    ]
  },
  "editable": true,
  "gnetId": null,
  "graphTooltip": 0,
  "id": 4,
  "links": [],
  "panels": [
    {
      "datasource": "Prometheus",
      "fieldConfig": {
        "defaults": {
          "color": {
            "mode": "palette-classic"
          },
          "custom": {
            "axisLabel": "",
            "axisPlacement": "auto",
            "axisSoftMin": 0,
            "fillOpacity": 80,
            "gradientMode": "none",
            "hideFrom": {
              "legend": false,
              "tooltip": false,
              "viz": false
            },
            "lineWidth": 1
          },
          "mappings": [],
          "thresholds": {
            "mode": "absolute",
            "steps": [
              {
                "color": "green",
                "value": null
              },
              {
                "color": "red",
                "value": 80
              }
            ]
          }
        },
        "overrides": []
      },
      "gridPos": {
        "h": 8,
        "w": 12,
        "x": 0,
        "y": 0
      },
      "id": 2,
      "options": {
        "barWidth": 0.97,
        "groupWidth": 0.7,
        "legend": {
          "calcs": [],
          "displayMode": "list",
          "placement": "bottom"
        },
        "orientation": "auto",
        "showValue": "auto",
        "text": {
          "valueSize": 12
        },
        "tooltip": {
          "mode": "single"
        }
      },
      "pluginVersion": "7.5.5",
      "targets": [
        {
          "exemplar": true,
          "expr": "sum(signals_generated_total) by (type)",
          "interval": "",
          "legendFormat": "{{type}}",
          "refId": "A"
        }
      ],
      "title": "Signals by Type",
      "type": "barchart"
    }
  ],
  "schemaVersion": 27,
  "style": "dark",
  "tags": [],
  "templating": {
    "list": []
  },
  "time": {
    "from": "now-6h",
    "to": "now"
  },
  "timepicker": {},
  "timezone": "",
  "title": "Technical Analysis",
  "uid": "technical-analysis",
  "version": 1
}
```

### 5. Infrastructure Dashboard

This dashboard provides detailed system resource utilization metrics:

- CPU, memory, and disk usage
- Network I/O
- Container metrics
- Database performance
- Message queue metrics

```json
{
  "annotations": {
    "list": [
      {
        "builtIn": 1,
        "datasource": "-- Grafana --",
        "enable": true,
        "hide": true,
        "iconColor": "rgba(0, 211, 255, 1)",
        "name": "Annotations & Alerts",
        "type": "dashboard"
      }
    ]
  },
  "editable": true,
  "gnetId": null,
  "graphTooltip": 0,
  "id": 5,
  "links": [],
  "panels": [
    {
      "aliasColors": {},
      "bars": false,
      "dashLength": 10,
      "dashes": false,
      "datasource": "Prometheus",
      "fieldConfig": {
        "defaults": {},
        "overrides": []
      },
      "fill": 1,
      "fillGradient": 0,
      "gridPos": {
        "h": 8,
        "w": 12,
        "x": 0,
        "y": 0
      },
      "hiddenSeries": false,
      "id": 2,
      "legend": {
        "avg": false,
        "current": false,
        "max": false,
        "min": false,
        "show": true,
        "total": false,
        "values": false
      },
      "lines": true,
      "linewidth": 1,
      "nullPointMode": "null",
      "options": {
        "alertThreshold": true
      },
      "percentage": false,
      "pluginVersion": "7.5.5",
      "pointradius": 2,
      "points": false,
      "renderer": "flot",
      "seriesOverrides": [],
      "spaceLength": 10,
      "stack": false,
      "steppedLine": false,
      "targets": [
        {
          "exemplar": true,
          "expr": "sum by (container_name) (rate(container_cpu_usage_seconds_total{container_name=~\"mexc-.*\"}[1m]))",
          "interval": "",
          "legendFormat": "{{container_name}}",
          "refId": "A"
        }
      ],
      "thresholds": [],
      "timeFrom": null,
      "timeRegions": [],
      "timeShift": null,
      "title": "CPU Usage by Container",
      "tooltip": {
        "shared": true,
        "sort": 0,
        "value_type": "individual"
      },
      "type": "graph",
      "xaxis": {
        "buckets": null,
        "mode": "time",
        "name": null,
        "show": true,
        "values": []
      },
      "yaxes": [
        {
          "format": "short",
          "label": null,
          "logBase": 1,
          "max": null,
          "min": null,
          "show": true
        },
        {
          "format": "short",
          "label": null,
          "logBase": 1,
          "max": null,
          "min": null,
          "show": true
        }
      ],
      "yaxis": {
        "align": false,
        "alignLevel": null
      }
    }
  ],
  "schemaVersion": 27,
  "style": "dark",
  "tags": [],
  "templating": {
    "list": []
  },
  "time": {
    "from": "now-6h",
    "to": "now"
  },
  "timepicker": {},
  "timezone": "",
  "title": "Infrastructure",
  "uid": "infrastructure",
  "version": 1
}
```

## Metrics Collection

### Prometheus Configuration

The monitoring system uses Prometheus to collect metrics. Here's the configuration:

```yaml
# /docker/prometheus/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'market-data-processor'
    static_configs:
      - targets: ['market-data-processor:9091']

  - job_name: 'order-execution'
    static_configs:
      - targets: ['order-execution:9092']

  - job_name: 'decision-service'
    static_configs:
      - targets: ['decision-service:9093']

  - job_name: 'signal-generator'
    static_configs:
      - targets: ['signal-generator:9094']

  - job_name: 'api-gateway'
    static_configs:
      - targets: ['api-gateway:9095']

  - job_name: 'cadvisor'
    static_configs:
      - targets: ['cadvisor:8080']

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
```

### Application Metrics

Each component exposes metrics via a Prometheus endpoint:

#### Rust Components

```rust
use prometheus::{register_counter, register_histogram, Counter, Histogram};
use lazy_static::lazy_static;

lazy_static! {
    pub static ref MARKET_DATA_MESSAGES_RECEIVED: Counter = register_counter!(
        "market_data_messages_received_total",
        "Total number of market data messages received"
    ).unwrap();
    
    pub static ref ORDER_BOOK_UPDATE_DURATION: Histogram = register_histogram!(
        "order_book_update_duration_seconds",
        "Time taken to update the order book",
        vec![0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    ).unwrap();
}
```

#### Node.js Components

```javascript
const client = require('prom-client');

// Create a Registry to register the metrics
const register = new client.Registry();

// Add a default label to all metrics
client.collectDefaultMetrics({ register });

// Create a custom counter metric
const decisionsTotal = new client.Counter({
  name: 'decisions_total',
  help: 'Total number of trading decisions made',
  labelNames: ['symbol', 'direction']
});

// Register the custom metrics
register.registerMetric(decisionsTotal);

// Expose metrics endpoint
app.get('/metrics', async (req, res) => {
  res.set('Content-Type', register.contentType);
  res.end(await register.metrics());
});
```

#### Python Components

```python
from prometheus_client import Counter, Histogram, start_http_server

# Create metrics
SIGNALS_GENERATED = Counter(
    'signals_generated_total',
    'Total number of trading signals generated',
    ['type', 'symbol', 'direction']
)

SIGNAL_GENERATION_TIME = Histogram(
    'signal_generation_duration_seconds',
    'Time taken to generate a trading signal',
    ['type'],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
)

# Start metrics server
start_http_server(9094)
```

## Alerting

The monitoring system includes alerting rules for critical conditions:

```yaml
# /docker/prometheus/alert_rules.yml
groups:
  - name: trading_system
    rules:
      - alert: ServiceDown
        expr: up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Service {{ $labels.job }} is down"
          description: "Service {{ $labels.job }} has been down for more than 1 minute."

      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate for {{ $labels.job }}"
          description: "Error rate is above 10% for 5 minutes."

      - alert: HighLatency
        expr: histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le, job)) > 1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High latency for {{ $labels.job }}"
          description: "95th percentile latency is above 1 second for 5 minutes."
```

## Windows Development Considerations

When using the monitoring dashboard on Windows:

1. Ensure Docker Desktop has enough resources allocated
2. Access Grafana through localhost (http://localhost:3000) in your browser
3. If using WSL2, you may need to access using the WSL2 IP address instead
4. For persistent storage, ensure the Docker volumes are properly configured

## Extending the Dashboards

To add custom metrics and panels:

1. Define new metrics in your application code
2. Expose them via the Prometheus endpoint
3. Add new panels to the Grafana dashboards using the Grafana UI
4. Export the updated dashboard JSON
5. Save it to the appropriate file in the repository

## Conclusion

This monitoring solution provides comprehensive visibility into the MEXC trading system's performance, health, and business metrics. By importing these dashboard templates into Grafana, you'll have immediate access to critical information about your trading system's operation.

The dashboards are designed to work seamlessly with the containerized development environment and can be extended as needed to monitor additional aspects of the system.
