/**
 * Main application initialization and control
 */

// Current symbol and interval
let currentSymbol = 'BTCUSDT';
let currentInterval = '1m';

// Update frequency (milliseconds)
const UPDATE_INTERVAL = 2000;

// Initialize all components
document.addEventListener('DOMContentLoaded', () => {
    // Initialize charts
    initializePriceChart();
    initializePortfolioChart();
    initializePerformanceChart();
    initializeActivityChart();
    initializeAssetPerformanceChart();
    
    // Setup trading interface
    setupTradingInterface();
    
    // Setup navigation
    setupNavigation();
    
    // Setup symbol selector
    setupSymbolSelector();
    
    // Setup timeframe buttons
    setupTimeframeButtons();
    
    // Start data updates
    startDataUpdates();
});

// Setup navigation between views
function setupNavigation() {
    const dashboardTab = document.getElementById('dashboard-tab');
    const portfolioTab = document.getElementById('portfolio-tab');
    const analyticsTab = document.getElementById('analytics-tab');
    
    const dashboardView = document.getElementById('dashboard-view');
    const portfolioView = document.getElementById('portfolio-view');
    const analyticsView = document.getElementById('analytics-view');
    
    if (dashboardTab && portfolioTab && analyticsTab) {
        // Dashboard tab click
        dashboardTab.addEventListener('click', (e) => {
            e.preventDefault();
            
            // Update active tab
            dashboardTab.classList.add('active');
            portfolioTab.classList.remove('active');
            analyticsTab.classList.remove('active');
            
            // Show dashboard view
            dashboardView.classList.add('active');
            portfolioView.classList.remove('active');
            analyticsView.classList.remove('active');
        });
        
        // Portfolio tab click
        portfolioTab.addEventListener('click', (e) => {
            e.preventDefault();
            
            // Update active tab
            dashboardTab.classList.remove('active');
            portfolioTab.classList.add('active');
            analyticsTab.classList.remove('active');
            
            // Show portfolio view
            dashboardView.classList.remove('active');
            portfolioView.classList.add('active');
            analyticsView.classList.remove('active');
            
            // Update portfolio data
            updatePortfolioAssets();
        });
        
        // Analytics tab click
        analyticsTab.addEventListener('click', (e) => {
            e.preventDefault();
            
            // Update active tab
            dashboardTab.classList.remove('active');
            portfolioTab.classList.remove('active');
            analyticsTab.classList.add('active');
            
            // Show analytics view
            dashboardView.classList.remove('active');
            portfolioView.classList.remove('active');
            analyticsView.classList.add('active');
        });
    }
}

// Setup symbol selector
function setupSymbolSelector() {
    const symbolSelect = document.getElementById('trading-pair-select');
    
    if (symbolSelect) {
        symbolSelect.addEventListener('change', () => {
            currentSymbol = symbolSelect.value;
            
            // Update data for new symbol
            updateTicker(currentSymbol);
            updateOrderbook(currentSymbol);
            updateRecentTrades(currentSymbol);
            updatePriceChart(currentSymbol, currentInterval);
        });
    }
}

// Setup timeframe buttons
function setupTimeframeButtons() {
    const timeframeButtons = document.querySelectorAll('.timeframe-btn');
    
    timeframeButtons.forEach(button => {
        button.addEventListener('click', () => {
            // Update active button
            timeframeButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');
            
            // Update current interval
            currentInterval = button.dataset.interval;
            
            // Update price chart
            updatePriceChart(currentSymbol, currentInterval);
        });
    });
}

// Start periodic data updates
function startDataUpdates() {
    // Initial updates
    updateTicker(currentSymbol);
    updateOrderbook(currentSymbol);
    updateRecentTrades(currentSymbol);
    updatePriceChart(currentSymbol, currentInterval);
    updateBalances();
    updatePositions();
    updateTradeHistory();
    
    // Setup periodic updates
    setInterval(() => {
        updateTicker(currentSymbol);
        updateOrderbook(currentSymbol);
        updateRecentTrades(currentSymbol);
    }, UPDATE_INTERVAL);
    
    // Update chart less frequently
    setInterval(() => {
        updatePriceChart(currentSymbol, currentInterval);
    }, UPDATE_INTERVAL * 2);
    
    // Update account data less frequently
    setInterval(() => {
        updateBalances();
        updatePositions();
        updateTradeHistory();
    }, UPDATE_INTERVAL * 3);
    
    // Update system status
    updateSystemStatus();
    setInterval(updateSystemStatus, 10000);
}

// Update system status
function updateSystemStatus() {
    fetchAPI('/api/system_metrics')
        .then(data => {
            if (!data) return;
            
            const statusElement = document.getElementById('system-status');
            if (statusElement) {
                const uptime = data.uptime ? Math.floor(data.uptime / 60) : 0;
                statusElement.textContent = `System Online (Uptime: ${uptime}m)`;
            }
        });
}
