#!/usr/bin/env python3
"""
Step 1: Performance Benchmark
Ensures scanner meets speed requirements (<2 seconds for 100 stocks)
"""

import sys
import time
import json
import statistics
from pathlib import Path
from datetime import datetime
import psutil

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from scanner import MinimalScanner
from data_fetcher import DataFetcher
from indicators import TechnicalIndicators


class PerformanceBenchmark:
    """Comprehensive performance testing for Step 1"""
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'tests': {},
            'system_info': self._get_system_info()
        }
    
    def _get_system_info(self):
        """Get system information for benchmark context"""
        return {
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq().current if psutil.cpu_freq() else None,
            'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'memory_available_gb': round(psutil.virtual_memory().available / (1024**3), 2)
        }
    
    def benchmark_data_fetching(self, runs: int = 3):
        """Benchmark data fetching performance"""
        print("\n📊 Benchmarking Data Fetching...")
        
        fetcher = DataFetcher()
        symbols = fetcher.symbols[:100]
        
        times = []
        for i in range(runs):
            print(f"   Run {i+1}/{runs}...", end="")
            start = time.time()
            data = fetcher.fetch_batch(symbols[:20], period="1mo")  # Test with 20 symbols
            elapsed = time.time() - start
            times.append(elapsed)
            print(f" {elapsed:.2f}s")
        
        self.results['tests']['data_fetching'] = {
            'symbols_tested': 20,
            'runs': runs,
            'times': times,
            'average': statistics.mean(times),
            'std_dev': statistics.stdev(times) if len(times) > 1 else 0,
            'min': min(times),
            'max': max(times)
        }
        
        return statistics.mean(times)
    
    def benchmark_indicator_calculation(self, runs: int = 5):
        """Benchmark technical indicator calculations"""
        print("\n📈 Benchmarking Indicator Calculations...")
        
        fetcher = DataFetcher()
        
        # Get sample data
        sample_data = fetcher.fetch_batch(["AAPL", "MSFT", "GOOGL"], period="3mo")
        
        times = []
        for i in range(runs):
            print(f"   Run {i+1}/{runs}...", end="")
            start = time.time()
            
            # Calculate indicators for each symbol
            for symbol, data in sample_data.items():
                if data is not None and not data.empty:
                    indicators = TechnicalIndicators(data)
                    _ = indicators.get_signal_features()
            
            elapsed = time.time() - start
            times.append(elapsed)
            print(f" {elapsed:.3f}s")
        
        self.results['tests']['indicator_calculation'] = {
            'symbols_tested': len(sample_data),
            'runs': runs,
            'times': times,
            'average': statistics.mean(times),
            'std_dev': statistics.stdev(times) if len(times) > 1 else 0,
            'min': min(times),
            'max': max(times)
        }
        
        return statistics.mean(times)
    
    def benchmark_full_scan(self, symbol_counts: list = [10, 25, 50, 100]):
        """Benchmark full scanning with different symbol counts"""
        print("\n🔍 Benchmarking Full Scan Performance...")
        
        scanner = MinimalScanner(max_symbols=100)
        
        for count in symbol_counts:
            print(f"\n   Testing with {count} symbols:")
            
            times = []
            signal_counts = []
            
            for i in range(3):  # 3 runs per count
                print(f"      Run {i+1}/3...", end="")
                
                # Clear cache for fair comparison
                scanner.fetcher.clear_cache()
                
                start = time.time()
                results = scanner.scan(scanner.symbols[:count])
                elapsed = time.time() - start
                
                times.append(elapsed)
                signal_counts.append(results['signals_generated'])
                
                print(f" {elapsed:.2f}s ({results['signals_generated']} signals)")
            
            self.results['tests'][f'full_scan_{count}_symbols'] = {
                'symbol_count': count,
                'times': times,
                'average_time': statistics.mean(times),
                'average_signals': statistics.mean(signal_counts),
                'data_quality': results.get('data_quality_score', 0)
            }
    
    def benchmark_memory_usage(self):
        """Benchmark memory usage during scanning"""
        print("\n💾 Benchmarking Memory Usage...")
        
        process = psutil.Process()
        
        # Get baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run scan
        scanner = MinimalScanner(max_symbols=100)
        results = scanner.scan()
        
        # Get peak memory
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        self.results['tests']['memory_usage'] = {
            'baseline_mb': round(baseline_memory, 2),
            'peak_mb': round(peak_memory, 2),
            'increase_mb': round(peak_memory - baseline_memory, 2)
        }
        
        print(f"   Baseline: {baseline_memory:.2f} MB")
        print(f"   Peak: {peak_memory:.2f} MB")
        print(f"   Increase: {peak_memory - baseline_memory:.2f} MB")
    
    def verify_requirements(self):
        """Verify Step 1 requirements are met"""
        print("\n✅ Verifying Step 1 Requirements...")
        
        requirements_met = True
        
        # Requirement 1: Process 100 stocks in <2 seconds
        if 'full_scan_100_symbols' in self.results['tests']:
            scan_time = self.results['tests']['full_scan_100_symbols']['average_time']
            if scan_time < 2.0:
                print(f"   ✅ Speed Requirement: {scan_time:.2f}s < 2.0s")
            else:
                print(f"   ❌ Speed Requirement: {scan_time:.2f}s > 2.0s")
                requirements_met = False
        
        # Requirement 2: Data quality >95%
        if 'full_scan_100_symbols' in self.results['tests']:
            quality = self.results['tests']['full_scan_100_symbols'].get('data_quality', 0)
            if quality > 0.95:
                print(f"   ✅ Data Quality: {quality*100:.1f}% > 95%")
            else:
                print(f"   ❌ Data Quality: {quality*100:.1f}% < 95%")
                requirements_met = False
        
        # Requirement 3: Generate signals
        if 'full_scan_100_symbols' in self.results['tests']:
            signals = self.results['tests']['full_scan_100_symbols']['average_signals']
            if signals > 0:
                print(f"   ✅ Signal Generation: {signals:.0f} signals generated")
            else:
                print(f"   ❌ Signal Generation: No signals generated")
                requirements_met = False
        
        # Requirement 4: Memory usage reasonable (<500MB increase)
        if 'memory_usage' in self.results['tests']:
            memory_increase = self.results['tests']['memory_usage']['increase_mb']
            if memory_increase < 500:
                print(f"   ✅ Memory Usage: {memory_increase:.2f} MB < 500 MB")
            else:
                print(f"   ⚠️  Memory Usage: {memory_increase:.2f} MB > 500 MB")
        
        self.results['requirements_met'] = requirements_met
        return requirements_met
    
    def save_results(self):
        """Save benchmark results to file"""
        output_dir = Path(__file__).parent.parent / "data"
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"benchmark_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
        return str(output_file)
    
    def print_summary(self):
        """Print benchmark summary"""
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        
        if 'full_scan_100_symbols' in self.results['tests']:
            test = self.results['tests']['full_scan_100_symbols']
            print(f"Full Scan (100 symbols):")
            print(f"  Average Time: {test['average_time']:.2f}s")
            print(f"  Average Signals: {test['average_signals']:.0f}")
            print(f"  Data Quality: {test.get('data_quality', 0)*100:.1f}%")
        
        if 'memory_usage' in self.results['tests']:
            mem = self.results['tests']['memory_usage']
            print(f"\nMemory Usage:")
            print(f"  Peak: {mem['peak_mb']:.2f} MB")
            print(f"  Increase: {mem['increase_mb']:.2f} MB")
        
        print("\n" + "="*60)
        
        if self.results.get('requirements_met'):
            print("🎉 ALL REQUIREMENTS MET!")
        else:
            print("⚠️  Some requirements not met. See details above.")
        
        print("="*60)


def run_benchmark():
    """Run complete benchmark suite"""
    benchmark = PerformanceBenchmark()
    
    print("🚀 Starting Step 1 Performance Benchmark")
    print("="*60)
    
    # Run benchmarks
    benchmark.benchmark_data_fetching(runs=3)
    benchmark.benchmark_indicator_calculation(runs=5)
    benchmark.benchmark_full_scan(symbol_counts=[10, 25, 50, 100])
    benchmark.benchmark_memory_usage()
    
    # Verify requirements
    benchmark.verify_requirements()
    
    # Save and summarize
    benchmark.save_results()
    benchmark.print_summary()
    
    return benchmark.results


if __name__ == "__main__":
    run_benchmark()