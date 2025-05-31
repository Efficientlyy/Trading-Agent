#!/usr/bin/env python
"""
MEXC Trading System Dashboard - Minimal Chart Test Server

This script creates a Flask server that serves a minimal chart test page
to debug LightweightCharts rendering issues.
"""

import os
from flask import Flask, send_from_directory
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__, static_folder='static')
CORS(app)

# Flask routes
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'minimal-chart-test.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8082, debug=True)
