#!/usr/bin/env python3
"""
Step 1: Signal Verification Script
Verify historical signal performance
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from validator import SignalValidator


def main():
    parser = argparse.ArgumentParser(description='Verify signal performance')
    parser.add_argument('--period', type=int, default=30, 
                       help='Number of days to analyze (default: 30)')
    parser.add_argument('--validate-days', type=int, default=7,
                       help='Forward-looking days for validation (default: 7)')
    
    args = parser.parse_args()
    
    print(f"📊 Verifying signals for the last {args.period} days")
    print(f"   Using {args.validate_days}-day forward returns")
    print("="*60)
    
    validator = SignalValidator()
    
    # Generate performance report
    report = validator.generate_performance_report(days=args.period)
    
    if 'error' in report:
        print(f"❌ Error: {report['error']}")
        return
    
    # Print results
    print("\n📈 PERFORMANCE REPORT")
    print("-"*60)
    
    print(f"Analysis Period: {report['analysis_period_days']} days")
    print(f"Total Scans: {report['total_scans']}")
    print(f"Total Signals: {report['total_signals_generated']}")
    
    if 'validation_metrics' in report:
        val = report['validation_metrics']
        print(f"\n🎯 Signal Accuracy:")
        print(f"   Overall Hit Rate: {val.get('overall_hit_rate', 0)*100:.1f}%")
        print(f"   Average Return: {val.get('average_return', 0):.2f}%")
        print(f"   Validated Signals: {val.get('validated_signals', 0)}/{val.get('total_signals', 0)}")
        
        if 'buy_signals' in val:
            buy = val['buy_signals']
            print(f"\n   Buy Signals:")
            print(f"      Total: {buy.get('total', 0)}")
            print(f"      Hit Rate: {buy.get('hit_rate', 0)*100:.1f}%")
        
        if 'sell_signals' in val:
            sell = val['sell_signals']
            print(f"\n   Sell Signals:")
            print(f"      Total: {sell.get('total', 0)}")
            print(f"      Hit Rate: {sell.get('hit_rate', 0)*100:.1f}%")
    
    if 'performance_trends' in report:
        trends = report['performance_trends']
        print(f"\n📊 Performance Trends:")
        print(f"   Avg Scan Time: {trends.get('average_scan_time', 0):.2f}s")
        print(f"   Avg Data Quality: {trends.get('average_data_quality', 0)*100:.1f}%")
        print(f"   Avg Signals/Scan: {trends.get('average_signals_per_scan', 0):.1f}")
    
    print("\n" + "="*60)
    
    # Check if requirements are met
    if val.get('overall_hit_rate', 0) >= 0.55:
        print("✅ Hit rate requirement met (>55%)")
    else:
        print("⚠️  Hit rate below 55% threshold")
    
    if trends.get('average_scan_time', 0) < 2.0:
        print("✅ Speed requirement met (<2s)")
    else:
        print("⚠️  Scan time exceeds 2s threshold")
    
    print("="*60)


if __name__ == "__main__":
    main()