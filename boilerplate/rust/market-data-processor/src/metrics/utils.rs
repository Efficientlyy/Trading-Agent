use std::time::{Duration, Instant};
use std::{thread, time};
use std::sync::atomic::{AtomicUsize, Ordering};

/// A timer utility for measuring execution durations and reporting them as metrics
pub struct PerformanceTimer {
    start: Instant,
    name: String,
    labels: Vec<String>,
}

impl PerformanceTimer {
    /// Create a new timer with the given name and labels
    pub fn new(name: &str, labels: Vec<&str>) -> Self {
        Self {
            start: Instant::now(),
            name: name.to_string(),
            labels: labels.into_iter().map(|s| s.to_string()).collect(),
        }
    }

    /// Record the elapsed time in the appropriate histogram metric
    pub fn observe_execution_time(&self) -> Duration {
        let elapsed = self.start.elapsed();
        
        // Match the metric name to the correct histogram
        match self.name.as_str() {
            "order_execution" => {
                if self.labels.len() >= 3 {
                    super::ORDER_EXECUTION_LATENCY
                        .with_label_values(&[&self.labels[0], &self.labels[1], &self.labels[2]])
                        .observe(elapsed.as_secs_f64());
                }
            }
            "order_round_trip" => {
                if self.labels.len() >= 3 {
                    super::ORDER_ROUND_TRIP_TIME
                        .with_label_values(&[&self.labels[0], &self.labels[1], &self.labels[2]])
                        .observe(elapsed.as_secs_f64());
                }
            }
            "market_data_processing" => {
                if self.labels.len() >= 2 {
                    super::MARKET_DATA_PROCESSING_TIME
                        .with_label_values(&[&self.labels[0], &self.labels[1]])
                        .observe(elapsed.as_secs_f64());
                }
            }
            "signal_generation" => {
                if self.labels.len() >= 2 {
                    super::SIGNAL_GENERATION_TIME
                        .with_label_values(&[&self.labels[0], &self.labels[1]])
                        .observe(elapsed.as_secs_f64());
                }
            }
            "api_request" => {
                if self.labels.len() >= 3 {
                    super::API_REQUEST_DURATION
                        .with_label_values(&[&self.labels[0], &self.labels[1], &self.labels[2]])
                        .observe(elapsed.as_secs_f64());
                }
            }
            "websocket_message" => {
                if self.labels.len() >= 1 {
                    super::WEBSOCKET_MESSAGE_DURATION
                        .with_label_values(&[&self.labels[0]])
                        .observe(elapsed.as_secs_f64());
                }
            }
            _ => {} // Unknown metric name, do nothing
        }
        
        elapsed
    }
}

/// A helper to track market data throughput
pub struct MarketDataTracker {
    pair: String,
    update_type: String,
}

impl MarketDataTracker {
    /// Create a new market data tracker
    pub fn new(pair: &str, update_type: &str) -> Self {
        Self {
            pair: pair.to_string(),
            update_type: update_type.to_string(),
        }
    }
    
    /// Increment the counter for market data updates
    pub fn record_update(&self) {
        super::MARKET_DATA_THROUGHPUT
            .with_label_values(&[&self.pair, &self.update_type])
            .inc();
    }
    
    /// Increment the counter by a specific amount
    pub fn record_updates(&self, count: u64) {
        super::MARKET_DATA_THROUGHPUT
            .with_label_values(&[&self.pair, &self.update_type])
            .inc_by(count as f64);
    }
}

/// A periodic resource usage reporter
pub struct ResourceReporter {
    component: String,
    running: AtomicUsize,
}

impl ResourceReporter {
    /// Create a new resource reporter for the given component
    pub fn new(component: &str) -> Self {
        Self {
            component: component.to_string(),
            running: AtomicUsize::new(0),
        }
    }
    
    /// Start a background thread to periodically report resource usage
    pub fn start_reporting(&self, interval_ms: u64) {
        let component = self.component.clone();
        let running = &self.running;
        
        running.store(1, Ordering::SeqCst);
        
        thread::spawn(move || {
            while running.load(Ordering::SeqCst) == 1 {
                // Report CPU usage
                if let Some(cpu_usage) = get_cpu_usage() {
                    super::CPU_USAGE_PERCENT
                        .with_label_values(&[&component])
                        .set(cpu_usage);
                }
                
                // Report memory usage
                if let Some(memory_usage) = get_memory_usage() {
                    super::MEMORY_USAGE_BYTES
                        .with_label_values(&[&component])
                        .set(memory_usage as f64);
                }
                
                thread::sleep(time::Duration::from_millis(interval_ms));
            }
        });
    }
    
    /// Stop the resource reporting thread
    pub fn stop_reporting(&self) {
        self.running.store(0, Ordering::SeqCst);
    }
}

// Helper to get current CPU usage (platform-dependent)
fn get_cpu_usage() -> Option<f64> {
    // This is a placeholder; actual implementation would use 
    // platform-specific APIs or libraries to get CPU usage
    #[cfg(target_os = "linux")]
    {
        // Linux implementation
        None
    }
    
    #[cfg(target_os = "windows")]
    {
        // Windows implementation
        None
    }
    
    #[cfg(target_os = "macos")]
    {
        // macOS implementation
        None
    }
    
    // Default placeholder
    Some(0.0)
}

// Helper to get current memory usage (platform-dependent)
fn get_memory_usage() -> Option<usize> {
    // This is a placeholder; actual implementation would use 
    // platform-specific APIs or libraries to get memory usage
    #[cfg(target_os = "linux")]
    {
        // Linux implementation
        None
    }
    
    #[cfg(target_os = "windows")]
    {
        // Windows implementation
        None
    }
    
    #[cfg(target_os = "macos")]
    {
        // macOS implementation
        None
    }
    
    // Default placeholder
    Some(0)
}
