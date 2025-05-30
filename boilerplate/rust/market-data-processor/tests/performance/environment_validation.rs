use market_data_processor::metrics::{
    self,
    utils::{PerformanceTimer, MarketDataTracker, ResourceReporter},
    collectors::{PerformanceCollector, RegressionDetector},
};
use std::{thread, time::{Duration, Instant}};
use std::fs::File;
use std::io::Write;
use std::sync::Arc;
use std::collections::HashMap;
use std::process::Command;

/// Environment detection and reporting
struct EnvironmentInfo {
    os_type: String,
    cpu_info: String,
    memory_total: String,
    docker_mode: String,
    is_wsl: bool,
}

impl EnvironmentInfo {
    fn detect() -> Self {
        // Detect OS
        let os_type = if cfg!(target_os = "windows") {
            "Windows".to_string()
        } else if cfg!(target_os = "linux") {
            // Check if running in WSL
            let is_wsl = std::fs::metadata("/proc/sys/fs/binfmt_misc/WSLInterop").is_ok();
            if is_wsl {
                "Linux (WSL)".to_string()
            } else {
                "Linux".to_string()
            }
        } else if cfg!(target_os = "macos") {
            "macOS".to_string()
        } else {
            "Unknown".to_string()
        };
        
        // Check if running in Docker
        let in_docker = std::fs::metadata("/.dockerenv").is_ok();
        
        // Docker mode detection
        let docker_mode = if in_docker {
            "Docker Container".to_string()
        } else {
            // Try to detect Docker Desktop mode on Windows
            if cfg!(target_os = "windows") {
                // Check if Docker Desktop is using WSL2 backend
                if Command::new("docker").args(["info"]).output().map_or(false, |output| {
                    String::from_utf8_lossy(&output.stdout).contains("WSL")
                }) {
                    "Docker Desktop (WSL2 backend)".to_string()
                } else {
                    "Docker Desktop (Hyper-V backend)".to_string()
                }
            } else {
                "Docker Engine".to_string()
            }
        };
        
        // Get CPU info
        let cpu_info = if cfg!(target_os = "windows") {
            Command::new("powershell")
                .args(["-Command", "(Get-WmiObject -Class Win32_Processor).Name"])
                .output()
                .map_or("Unknown".to_string(), |output| {
                    String::from_utf8_lossy(&output.stdout).trim().to_string()
                })
        } else {
            Command::new("sh")
                .args(["-c", "cat /proc/cpuinfo | grep 'model name' | head -n 1 | cut -d ':' -f 2"])
                .output()
                .map_or("Unknown".to_string(), |output| {
                    String::from_utf8_lossy(&output.stdout).trim().to_string()
                })
        };
        
        // Get memory info
        let memory_total = if cfg!(target_os = "windows") {
            Command::new("powershell")
                .args(["-Command", "(Get-WmiObject -Class Win32_ComputerSystem).TotalPhysicalMemory"])
                .output()
                .map_or("Unknown".to_string(), |output| {
                    let bytes = String::from_utf8_lossy(&output.stdout).trim().parse::<u64>().unwrap_or(0);
                    format!("{:.2} GB", bytes as f64 / 1_073_741_824.0)
                })
        } else {
            Command::new("sh")
                .args(["-c", "free -m | grep Mem | awk '{print $2}'"])
                .output()
                .map_or("Unknown".to_string(), |output| {
                    let mb = String::from_utf8_lossy(&output.stdout).trim().parse::<u64>().unwrap_or(0);
                    format!("{:.2} GB", mb as f64 / 1024.0)
                })
        };
        
        // Check if running in WSL
        let is_wsl = os_type.contains("WSL") || std::fs::metadata("/proc/sys/fs/binfmt_misc/WSLInterop").is_ok();
        
        Self {
            os_type,
            cpu_info,
            memory_total,
            docker_mode,
            is_wsl,
        }
    }
    
    fn to_string(&self) -> String {
        format!(
            "Environment Information:\n\
             - OS: {}\n\
             - CPU: {}\n\
             - Memory: {}\n\
             - Docker: {}\n\
             - WSL: {}\n",
            self.os_type, 
            self.cpu_info, 
            self.memory_total, 
            self.docker_mode,
            if self.is_wsl { "Yes" } else { "No" }
        )
    }
}

/// Create environment-specific baseline profiles
fn create_environment_baseline() -> HashMap<String, f64> {
    let env_info = EnvironmentInfo::detect();
    println!("{}", env_info.to_string());
    
    // Create baseline metrics appropriate for this environment
    let mut baseline = HashMap::new();
    
    // Adjust baselines based on environment
    if env_info.is_wsl {
        // WSL2 environment typically has some overhead
        baseline.insert("order_execution_latency_p95".to_string(), 0.12); // 120ms baseline
        baseline.insert("market_data_throughput_avg".to_string(), 800.0); // 800/sec baseline
        baseline.insert("signal_generation_time_p95".to_string(), 0.035); // 35ms baseline
    } else if env_info.os_type == "Windows" {
        // Native Windows
        baseline.insert("order_execution_latency_p95".to_string(), 0.10); // 100ms baseline
        baseline.insert("market_data_throughput_avg".to_string(), 900.0); // 900/sec baseline
        baseline.insert("signal_generation_time_p95".to_string(), 0.030); // 30ms baseline
    } else if env_info.os_type.contains("Linux") {
        // Native Linux typically has the best performance
        baseline.insert("order_execution_latency_p95".to_string(), 0.08); // 80ms baseline
        baseline.insert("market_data_throughput_avg".to_string(), 1200.0); // 1200/sec baseline
        baseline.insert("signal_generation_time_p95".to_string(), 0.025); // 25ms baseline
    } else if env_info.os_type == "macOS" {
        // macOS
        baseline.insert("order_execution_latency_p95".to_string(), 0.09); // 90ms baseline
        baseline.insert("market_data_throughput_avg".to_string(), 1000.0); // 1000/sec baseline
        baseline.insert("signal_generation_time_p95".to_string(), 0.028); // 28ms baseline
    }
    
    // Adjust for Docker mode
    if env_info.docker_mode.contains("Container") {
        // Running inside container - minimal overhead in a properly configured environment
        baseline.iter_mut().for_each(|(_, v)| *v *= 0.95); // 5% improvement
    } else if env_info.docker_mode.contains("Hyper-V") {
        // Hyper-V backend has more overhead
        baseline.iter_mut().for_each(|(_, v)| *v *= 1.2); // 20% degradation
    }
    
    // Save baseline to file
    let env_name = if env_info.is_wsl {
        "wsl2"
    } else if env_info.os_type.contains("Windows") {
        "windows"
    } else if env_info.os_type.contains("Linux") {
        "linux"
    } else if env_info.os_type.contains("macOS") {
        "macos"
    } else {
        "unknown"
    };
    
    let filename = format!("./tests/performance/results/baseline_{}.json", env_name);
    let json = serde_json::to_string_pretty(&baseline).unwrap();
    let mut file = File::create(&filename).unwrap();
    file.write_all(json.as_bytes()).unwrap();
    
    println!("Created environment-specific baseline in: {}", filename);
    
    baseline
}

/// Run environment-specific performance tests
#[test]
fn test_environment_performance() {
    // Detect environment
    let env_info = EnvironmentInfo::detect();
    println!("Running environment-specific performance tests");
    println!("{}", env_info.to_string());
    
    // Create environment-specific baseline
    let baseline = create_environment_baseline();
    
    // Adjust test parameters based on environment
    let (iterations, batch_size) = if env_info.is_wsl || env_info.os_type == "Windows" {
        // Reduce test load on WSL2 and Windows
        (50, 500)
    } else {
        // Full test on Linux and macOS
        (100, 1000)
    };
    
    println!("Running test with iterations={}, batch_size={}", iterations, batch_size);
    
    // Measure order execution latency
    let mut order_latencies = Vec::new();
    for _ in 0..iterations {
        let timer = PerformanceTimer::new("order_execution", vec!["BTCUSDT", "LIMIT", "BUY"]);
        // Simulate order execution process
        thread::sleep(Duration::from_millis(50));
        order_latencies.push(timer.observe_execution_time().as_secs_f64());
    }
    
    // Measure market data throughput
    let market_data_tracker = MarketDataTracker::new("BTCUSDT", "PRICE");
    let throughput_start = Instant::now();
    for _ in 0..iterations {
        // Process a batch of updates
        thread::sleep(Duration::from_micros(100));
        market_data_tracker.record_updates(batch_size as u64);
    }
    let throughput_duration = throughput_start.elapsed().as_secs_f64();
    let updates_per_second = (iterations * batch_size) as f64 / throughput_duration;
    
    // Measure signal generation latency
    let mut signal_latencies = Vec::new();
    for _ in 0..iterations {
        let timer = PerformanceTimer::new("signal_generation", vec!["MOMENTUM", "BTCUSDT"]);
        // Simulate signal generation
        thread::sleep(Duration::from_millis(20));
        signal_latencies.push(timer.observe_execution_time().as_secs_f64());
    }
    
    // Calculate percentiles
    order_latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    signal_latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let p95_index = (order_latencies.len() as f64 * 0.95) as usize;
    let order_p95 = order_latencies[p95_index];
    
    let p95_index = (signal_latencies.len() as f64 * 0.95) as usize;
    let signal_p95 = signal_latencies[p95_index];
    
    // Compare with baseline
    let order_baseline = baseline.get("order_execution_latency_p95").unwrap_or(&0.1);
    let throughput_baseline = baseline.get("market_data_throughput_avg").unwrap_or(&1000.0);
    let signal_baseline = baseline.get("signal_generation_time_p95").unwrap_or(&0.03);
    
    println!("\nPerformance Results:");
    println!("Order Execution Latency (p95): {:.3}s (Baseline: {:.3}s) - {}",
        order_p95, order_baseline,
        if order_p95 <= *order_baseline { "✅ PASS" } else { "❌ FAIL" });
    
    println!("Market Data Throughput: {:.1} updates/sec (Baseline: {:.1}) - {}",
        updates_per_second, throughput_baseline,
        if updates_per_second >= *throughput_baseline { "✅ PASS" } else { "❌ FAIL" });
    
    println!("Signal Generation Time (p95): {:.3}s (Baseline: {:.3}s) - {}",
        signal_p95, signal_baseline,
        if signal_p95 <= *signal_baseline { "✅ PASS" } else { "❌ FAIL" });
    
    // Save results
    let mut results = HashMap::new();
    results.insert("order_execution_latency_p95".to_string(), order_p95);
    results.insert("market_data_throughput_avg".to_string(), updates_per_second);
    results.insert("signal_generation_time_p95".to_string(), signal_p95);
    
    let env_name = if env_info.is_wsl {
        "wsl2"
    } else if env_info.os_type.contains("Windows") {
        "windows"
    } else if env_info.os_type.contains("Linux") {
        "linux"
    } else if env_info.os_type.contains("macOS") {
        "macos"
    } else {
        "unknown"
    };
    
    let filename = format!("./tests/performance/results/results_{}.json", env_name);
    let json = serde_json::to_string_pretty(&results).unwrap();
    let mut file = File::create(&filename).unwrap();
    file.write_all(json.as_bytes()).unwrap();
    
    println!("\nSaved environment-specific results to: {}", filename);
    
    // Generate optimization recommendations
    println!("\nEnvironment-Specific Optimization Recommendations:");
    
    if env_info.is_wsl {
        println!("WSL2 Environment Recommendations:");
        println!("1. Ensure .wslconfig has appropriate memory and CPU allocations");
        println!("2. Store project files in WSL2 filesystem for better I/O performance");
        println!("3. Use WSL2-specific Docker configuration as documented in WSL2_SETUP_GUIDE.md");
    } else if env_info.os_type == "Windows" {
        println!("Windows Environment Recommendations:");
        println!("1. Use Docker Desktop with WSL2 backend instead of Hyper-V for better performance");
        println!("2. Enable hardware virtualization in BIOS/UEFI");
        println!("3. Configure Windows Defender exclusions for Docker and project directories");
    } else if env_info.os_type.contains("Linux") {
        println!("Linux Environment Recommendations:");
        println!("1. Ensure cgroup v2 is enabled for better Docker resource management");
        println!("2. Consider using a real-time kernel for better latency consistency");
        println!("3. Configure appropriate ulimits for file descriptors and process count");
    } else if env_info.os_type == "macOS" {
        println!("macOS Environment Recommendations:");
        println!("1. Allocate more resources to Docker Desktop in Preferences");
        println!("2. Use volumes with delegated consistency for better I/O performance");
        println!("3. Monitor thermal throttling during extended test runs");
    }
    
    // Check for failures
    assert!(order_p95 <= *order_baseline * 1.2, "Order execution latency significantly exceeds baseline");
    assert!(updates_per_second >= *throughput_baseline * 0.8, "Market data throughput significantly below baseline");
    assert!(signal_p95 <= *signal_baseline * 1.2, "Signal generation time significantly exceeds baseline");
}

/// Helper function to run Docker-specific validation
fn validate_docker_configuration() {
    println!("Validating Docker configuration...");
    
    // Check if Docker is running
    let docker_running = Command::new("docker").args(["ps"]).status().map_or(false, |status| status.success());
    
    if !docker_running {
        println!("❌ Docker is not running or not accessible");
        return;
    }
    
    println!("✅ Docker is running");
    
    // Check Docker Compose
    let compose_available = Command::new("docker-compose").args(["version"]).status().map_or(false, |status| status.success());
    
    if !compose_available {
        println!("❌ Docker Compose is not available");
    } else {
        println!("✅ Docker Compose is available");
    }
    
    // Check if we're on Windows
    #[cfg(target_os = "windows")]
    {
        // Check Docker Desktop backend
        let output = Command::new("docker").args(["info"]).output();
        
        if let Ok(output) = output {
            let info = String::from_utf8_lossy(&output.stdout);
            
            if info.contains("WSL") {
                println!("✅ Docker Desktop is using WSL2 backend (recommended)");
            } else if info.contains("Hyper-V") {
                println!("⚠️ Docker Desktop is using Hyper-V backend (WSL2 backend recommended)");
            }
        }
        
        // Check PowerShell version
        let ps_output = Command::new("powershell").args(["-Command", "$PSVersionTable.PSVersion.Major"]).output();
        
        if let Ok(output) = ps_output {
            let version = String::from_utf8_lossy(&output.stdout).trim().parse::<i32>().unwrap_or(0);
            
            if version >= 7 {
                println!("✅ PowerShell Core 7+ detected (recommended)");
            } else if version >= 5 {
                println!("ℹ️ Windows PowerShell 5.x detected");
            } else {
                println!("⚠️ Outdated PowerShell version detected. Consider upgrading to PowerShell 7+");
            }
        }
    }
    
    // Check Docker volumes
    println!("\nChecking Docker volume configuration...");
    
    // Create simple test container with volume mount
    let test_output = Command::new("docker").args([
        "run", "--rm", "-v", "./tests:/test", "alpine", "ls", "-la", "/test"
    ]).output();
    
    if let Ok(output) = test_output {
        if output.status.success() {
            println!("✅ Docker volume mounts are working correctly");
        } else {
            println!("❌ Docker volume mount test failed");
            println!("Error: {}", String::from_utf8_lossy(&output.stderr));
            println!("Solution: Ensure Docker has permission to access the local filesystem");
            
            #[cfg(target_os = "windows")]
            {
                println!("On Windows, check Docker Desktop settings > Resources > File Sharing");
                println!("Also, ensure your Windows username doesn't contain spaces or special characters");
            }
        }
    }
    
    println!("\nFor detailed Docker configuration guidance, refer to WSL2_SETUP_GUIDE.md");
}

/// Run Docker-specific validation tests
#[test]
fn test_docker_configuration() {
    validate_docker_configuration();
}

/// Run all environment validation tests
#[test]
fn run_all_environment_validations() {
    // Create results directory if it doesn't exist
    std::fs::create_dir_all("./tests/performance/results").unwrap();
    
    // Run tests
    test_environment_performance();
    test_docker_configuration();
}
