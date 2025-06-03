#!/bin/bash
echo "Stopping Trading-Agent System..."
kill 89373 89381 89390 89395 2>/dev/null
echo "All components stopped."
