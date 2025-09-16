#!/usr/bin/env python3
"""
Step 1: Daily Scan Runner
Execute daily scan and save results
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from scanner import run_daily_scan

if __name__ == "__main__":
    print("🚀 Starting Daily Scan...")
    results = run_daily_scan()
    print("\n✅ Daily scan complete!")