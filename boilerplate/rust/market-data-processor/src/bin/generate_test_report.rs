use std::fs;
use std::path::Path;
use chrono::Utc;
use clap::Parser;
use serde::{Deserialize, Serialize};
use serde_json::Value;

// Import our test modules for the metrics types
mod tests {
    pub use market_data_processor_tests::integration::metrics::*;
}

use tests::{PerformanceMetrics, MetricsReporter};

#[derive(Parser, Debug)]
#[clap(
    name = "generate_test_report",
    about = "Generate HTML and Markdown reports from integration test results"
)]
struct Args {
    /// Input JSON metrics file
    #[clap(short, long, value_parser)]
    input: String,
    
    /// Output report file (Markdown)
    #[clap(short, long, value_parser)]
    output: String,
    
    /// Optional HTML output file
    #[clap(short = 'H', long, value_parser)]
    html_output: Option<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    
    println!("Generating test report from: {}", args.input);
    
    // Read metrics file
    let json_content = fs::read_to_string(&args.input)?;
    let metrics: Vec<PerformanceMetrics> = serde_json::from_str(&json_content)?;
    
    // Create reporter
    let mut reporter = MetricsReporter::new();
    for metric in metrics {
        reporter.add_metrics(metric);
    }
    
    // Generate Markdown report
    let markdown_report = reporter.generate_summary();
    fs::write(&args.output, markdown_report)?;
    println!("Markdown report written to: {}", args.output);
    
    // Generate HTML report if requested
    if let Some(html_path) = args.html_output {
        let html_report = generate_html_report(&metrics);
        fs::write(&html_path, html_report)?;
        println!("HTML report written to: {}", html_path);
    }
    
    Ok(())
}

fn generate_html_report(metrics: &[PerformanceMetrics]) -> String {
    let mut html = String::new();
    
    // HTML header
    html.push_str(r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Paper Trading Test Report</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        h1, h2, h3, h4 {
            color: #0066cc;
        }
        .card {
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 15px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .metric-box {
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 4px;
            text-align: center;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            margin: 10px 0;
        }
        .metric-label {
            font-size: 14px;
            color: #666;
        }
        .positive {
            color: green;
        }
        .negative {
            color: red;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 10px;
            border: 1px solid #ddd;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .chart-container {
            height: 300px;
            margin: 20px 0;
        }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <h1>Paper Trading Integration Test Report</h1>
    <p>Generated on "#);
    
    // Add timestamp
    html.push_str(&Utc::now().format("%Y-%m-%d %H:%M:%S UTC").to_string());
    html.push_str("</p>\n");
    
    // Summary section
    html.push_str(r#"
    <div class="card">
        <h2>Summary</h2>
        <div class="metrics-grid">
"#);

    // Calculate overall metrics
    let total_scenarios = metrics.len();
    let total_pnl: f64 = metrics.iter().map(|m| m.account.total_pnl_usdt).sum();
    let avg_win_rate: f64 = if !metrics.is_empty() {
        metrics.iter().map(|m| m.trading.win_rate).sum::<f64>() / metrics.len() as f64
    } else {
        0.0
    };
    let max_drawdown: f64 = metrics.iter().map(|m| m.risk.max_drawdown_percent).fold(0.0, f64::max);
    
    // Summary metrics
    html.push_str(&format!(r#"
            <div class="metric-box">
                <div class="metric-label">Total Scenarios</div>
                <div class="metric-value">{}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Total P&L</div>
                <div class="metric-value {}">$ {:.2}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Average Win Rate</div>
                <div class="metric-value">{:.2}%</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Max Drawdown</div>
                <div class="metric-value negative">{:.2}%</div>
            </div>
        </div>
    </div>
    "#, 
        total_scenarios,
        if total_pnl >= 0.0 { "positive" } else { "negative" },
        total_pnl,
        avg_win_rate,
        max_drawdown
    ));
    
    // Per-scenario sections
    for (i, metric) in metrics.iter().enumerate() {
        html.push_str(&format!(r#"
    <div class="card">
        <h2>Scenario {}: {}</h2>
        <p>{}</p>
        
        <h3>Account Metrics</h3>
        <div class="metrics-grid">
            <div class="metric-box">
                <div class="metric-label">Initial Value</div>
                <div class="metric-value">$ {:.2}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Final Value</div>
                <div class="metric-value">$ {:.2}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Total P&L</div>
                <div class="metric-value {}">$ {:.2}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">P&L Percentage</div>
                <div class="metric-value {}">{:.2}%</div>
            </div>
        </div>
        
        <h3>Trading Metrics</h3>
        <div class="metrics-grid">
            <div class="metric-box">
                <div class="metric-label">Total Orders</div>
                <div class="metric-value">{}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Win Rate</div>
                <div class="metric-value">{:.2}%</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Avg. Win</div>
                <div class="metric-value positive">$ {:.2}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Avg. Loss</div>
                <div class="metric-value negative">$ {:.2}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Profit Factor</div>
                <div class="metric-value">{:.2}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Avg. Holding Time</div>
                <div class="metric-value">{:.1} min</div>
            </div>
        </div>
        
        <h3>Risk Metrics</h3>
        <div class="metrics-grid">
            <div class="metric-box">
                <div class="metric-label">Max Drawdown</div>
                <div class="metric-value negative">{:.2}%</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Max Drawdown Amount</div>
                <div class="metric-value negative">$ {:.2}</div>
            </div>"#,
            i + 1,
            metric.scenario_name,
            "Market simulation test", // Description placeholder
            metric.account.initial_value_usdt,
            metric.account.final_value_usdt,
            if metric.account.total_pnl_usdt >= 0.0 { "positive" } else { "negative" },
            metric.account.total_pnl_usdt,
            if metric.account.total_pnl_percent >= 0.0 { "positive" } else { "negative" },
            metric.account.total_pnl_percent,
            metric.trading.total_orders,
            metric.trading.win_rate,
            metric.trading.avg_win_profit,
            metric.trading.avg_loss,
            metric.trading.profit_factor,
            metric.trading.avg_holding_time_seconds / 60.0,
            metric.risk.max_drawdown_percent,
            metric.risk.max_drawdown_usdt
        ));
        
        // Add Sharpe and Calmar ratios if available
        if let Some(sharpe) = metric.risk.sharpe_ratio {
            html.push_str(&format!(r#"
            <div class="metric-box">
                <div class="metric-label">Sharpe Ratio</div>
                <div class="metric-value">{:.2}</div>
            </div>"#,
                sharpe
            ));
        }
        
        if let Some(calmar) = metric.risk.calmar_ratio {
            html.push_str(&format!(r#"
            <div class="metric-box">
                <div class="metric-label">Calmar Ratio</div>
                <div class="metric-value">{:.2}</div>
            </div>"#,
                calmar
            ));
        }
        
        html.push_str("</div>"); // Close risk metrics grid
        
        // Symbol performance table if available
        if !metric.symbols.is_empty() {
            html.push_str(r#"
        <h3>Symbol Performance</h3>
        <table>
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th>Trades</th>
                    <th>P&L</th>
                    <th>Win Rate</th>
                    <th>Avg. Position</th>
                    <th>Avg. Holding Time</th>
                    <th>Max Profit</th>
                    <th>Max Loss</th>
                </tr>
            </thead>
            <tbody>"#);
            
            for (_, symbol) in &metric.symbols {
                html.push_str(&format!(r#"
                <tr>
                    <td>{}</td>
                    <td>{}</td>
                    <td class="{}">$ {:.2}</td>
                    <td>{:.2}%</td>
                    <td>{:.4}</td>
                    <td>{:.1} min</td>
                    <td class="positive">$ {:.2}</td>
                    <td class="negative">$ {:.2}</td>
                </tr>"#,
                    symbol.symbol,
                    symbol.trade_count,
                    if symbol.total_pnl >= 0.0 { "positive" } else { "negative" },
                    symbol.total_pnl,
                    symbol.win_rate,
                    symbol.avg_position_size,
                    symbol.avg_holding_time_seconds / 60.0,
                    symbol.max_profit,
                    symbol.max_loss
                ));
            }
            
            html.push_str(r#"
            </tbody>
        </table>"#);
        }
        
        html.push_str("</div>"); // Close scenario card
    }
    
    // Chart placeholders for future enhancements
    html.push_str(r#"
    <div class="card">
        <h2>Performance Charts</h2>
        <p>Note: The actual chart generation would require actual data series from test execution.</p>
        <div class="chart-container">
            <canvas id="pnlChart"></canvas>
        </div>
    </div>
    
    <script>
        // This is a placeholder for actual chart generation
        // In a real implementation, we would pass data from the test execution
        const ctx = document.getElementById('pnlChart').getContext('2d');
        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: ['Scenario 1', 'Scenario 2', 'Scenario 3', 'Scenario 4', 'Scenario 5'],
                datasets: [{
                    label: 'P&L by Scenario',
                    data: [300, -150, 500, 250, -100],
                    borderColor: 'rgb(75, 192, 192)',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    </script>
</body>
</html>"#);
    
    html
}
