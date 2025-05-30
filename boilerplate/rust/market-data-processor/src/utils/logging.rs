use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use tracing::{Level, info};
use opentelemetry::{global, sdk::Resource};
use opentelemetry::sdk::trace::{Sampler, BatchConfig};
use opentelemetry_jaeger::new_pipeline;
use std::str::FromStr;

use crate::utils::config::Config;

pub fn init(config: &Config) -> Result<(), Box<dyn std::error::Error>> {
    let log_level = Level::from_str(&config.log_level).unwrap_or(Level::INFO);
    
    // Create base subscriber with env filter
    let subscriber = tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| format!("market_data_processor={}", config.log_level).into()),
        )
        .with(tracing_subscriber::fmt::layer());
    
    // Add telemetry if enabled
    if config.enable_telemetry {
        info!("Telemetry enabled, connecting to Jaeger at {}", config.jaeger_endpoint);
        
        let tracer = new_pipeline()
            .with_service_name("market-data-processor")
            .with_endpoint(&config.jaeger_endpoint)
            .with_trace_config(
                opentelemetry::sdk::trace::config()
                    .with_sampler(Sampler::AlwaysOn)
                    .with_resource(Resource::new(vec![
                        opentelemetry::KeyValue::new("service.name", "market-data-processor"),
                        opentelemetry::KeyValue::new("service.version", env!("CARGO_PKG_VERSION")),
                    ]))
            )
            .install_batch(BatchConfig::default())?;
            
        let telemetry = tracing_opentelemetry::layer().with_tracer(tracer);
        subscriber.with(telemetry).init();
    } else {
        subscriber.init();
    }
    
    info!("Logging initialized at level {}", log_level);
    Ok(())
}

// Clean up telemetry on shutdown
pub fn shutdown() {
    global::shutdown_tracer_provider();
}
