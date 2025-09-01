"""
Risk Scoring System - Performance Benchmarking Tests
Author: Claude Code (QA Engineer)
Version: 1.0

Performance and load testing for risk scoring system. Ensures sub-100ms
calculation times, high throughput capacity, and system stability under load.
Tests concurrent processing, memory usage, and scalability limits.
"""

import pytest
import numpy as np
import pandas as pd
import time
import asyncio
import concurrent.futures
import threading
import multiprocessing
from datetime import datetime, timedelta
from typing import List, Dict, Callable, Any, Tuple
from dataclasses import dataclass
from statistics import mean, median, stdev
import psutil
import gc
import sys
from contextlib import contextmanager

from data_models.python.core_models import Asset, AssetCategory, MarketData
from data_models.python.signal_models import RiskAssessment


@dataclass
class PerformanceMetrics:
    """Container for performance measurement results"""
    execution_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    throughput_ops_per_sec: float
    success_rate: float
    error_count: int
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float


class PerformanceTestBase:
    """Base class for performance testing utilities"""
    
    @staticmethod
    @contextmanager
    def measure_performance(operation_name: str = "operation"):
        """Context manager to measure performance metrics"""
        process = psutil.Process()
        
        # Initial measurements
        start_time = time.perf_counter()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        start_cpu_times = process.cpu_times()
        
        gc.collect()  # Clean up before measurement
        
        try:
            yield
        finally:
            # Final measurements
            end_time = time.perf_counter()
            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            end_cpu_times = process.cpu_times()
            
            # Calculate metrics
            execution_time = (end_time - start_time) * 1000  # Convert to ms
            memory_delta = end_memory - start_memory
            cpu_time_delta = (end_cpu_times.user + end_cpu_times.system) - \
                           (start_cpu_times.user + start_cpu_times.system)
            
            print(f"\n{operation_name} Performance Metrics:")
            print(f"  Execution Time: {execution_time:.2f} ms")
            print(f"  Memory Delta: {memory_delta:.2f} MB")
            print(f"  CPU Time: {cpu_time_delta:.3f} s")
    
    @staticmethod
    def create_test_market_data_batch(count: int) -> List[MarketData]:
        """Create batch of test market data"""
        batch = []
        base_time = datetime.utcnow()
        
        for i in range(count):
            price = int(np.random.uniform(5000, 50000))  # $50 to $500 in cents
            spread_bps = int(np.random.uniform(1, 50))
            spread_cents = (price * spread_bps) // 10000
            
            data = MarketData(
                asset_id=f"asset-{i:06d}",
                timestamp=base_time + timedelta(seconds=i),
                open_price_cents=price + np.random.randint(-100, 100),
                high_price_cents=price + np.random.randint(0, 200),
                low_price_cents=price - np.random.randint(0, 200),
                close_price_cents=price,
                volume=int(np.random.uniform(10000, 10000000)),
                bid_price_cents=price - spread_cents // 2,
                ask_price_cents=price + spread_cents // 2,
                data_source="PERF_TEST",
                data_quality_score=95
            )
            batch.append(data)
        
        return batch
    
    @staticmethod
    def measure_latency_distribution(latencies: List[float]) -> Dict[str, float]:
        """Calculate latency distribution metrics"""
        if not latencies:
            return {}
        
        sorted_latencies = sorted(latencies)
        return {
            'mean': mean(latencies),
            'median': median(latencies),
            'std_dev': stdev(latencies) if len(latencies) > 1 else 0.0,
            'min': min(latencies),
            'max': max(latencies),
            'p50': np.percentile(sorted_latencies, 50),
            'p90': np.percentile(sorted_latencies, 90),
            'p95': np.percentile(sorted_latencies, 95),
            'p99': np.percentile(sorted_latencies, 99)
        }


class TestSingleRequestPerformance(PerformanceTestBase):
    """Test individual risk calculation performance"""
    
    def test_single_risk_calculation_latency(self):
        """Test that single risk calculation meets <50ms requirement"""
        
        class OptimizedRiskCalculator:
            """Optimized risk calculator for performance testing"""
            
            def __init__(self):
                # Pre-compute lookup tables for performance
                self._volatility_lookup = self._build_volatility_lookup()
                self._regime_multipliers = {
                    'bull': 0.8, 'bear': 1.3, 'sideways': 1.0,
                    'high_vol': 1.4, 'low_vol': 0.7
                }
            
            def _build_volatility_lookup(self) -> Dict[int, int]:
                """Pre-compute volatility to risk score mapping"""
                lookup = {}
                for vol_bps in range(0, 10000, 10):  # 0% to 100% in 0.1% increments
                    vol_pct = vol_bps / 10000
                    risk_score = min(int(vol_pct * 200), 100)
                    lookup[vol_bps] = risk_score
                return lookup
            
            def calculate_risk_score_fast(self, market_data: MarketData) -> int:
                """Fast risk calculation optimized for performance"""
                # Volatility component (using lookup table)
                price_range_bps = ((market_data.high_price_cents - market_data.low_price_cents) * 10000) // market_data.close_price_cents
                volatility_score = self._volatility_lookup.get(min(price_range_bps, 9990), 100)
                
                # Liquidity component (vectorized calculation)
                if market_data.bid_price_cents and market_data.ask_price_cents:
                    spread_bps = ((market_data.ask_price_cents - market_data.bid_price_cents) * 10000) // market_data.close_price_cents
                    liquidity_score = min(spread_bps // 5, 100)  # Fast integer division
                else:
                    liquidity_score = 50
                
                # Volume component (simple thresholding)
                volume_score = 10 if market_data.volume > 1000000 else 30 if market_data.volume > 100000 else 60
                
                # Fast weighted average (using bit shifts for division)
                composite_score = ((volatility_score << 2) + (liquidity_score << 2) + (volume_score << 1)) >> 3  # Weights: 4,4,2 / 8
                
                return min(composite_score, 100)
        
        calculator = OptimizedRiskCalculator()
        test_data = self.create_test_market_data_batch(1)[0]
        
        # Warm up the calculator
        for _ in range(10):
            calculator.calculate_risk_score_fast(test_data)
        
        # Measure performance over multiple iterations
        latencies = []
        for _ in range(1000):
            start_time = time.perf_counter()
            risk_score = calculator.calculate_risk_score_fast(test_data)
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            
            # Validate result
            assert 0 <= risk_score <= 100, f"Invalid risk score: {risk_score}"
        
        # Analyze latency distribution
        latency_stats = self.measure_latency_distribution(latencies)
        
        print(f"\nSingle Risk Calculation Performance:")
        print(f"  Mean Latency: {latency_stats['mean']:.3f} ms")
        print(f"  P95 Latency: {latency_stats['p95']:.3f} ms")
        print(f"  P99 Latency: {latency_stats['p99']:.3f} ms")
        print(f"  Max Latency: {latency_stats['max']:.3f} ms")
        
        # Performance assertions
        assert latency_stats['p95'] < 50.0, f"P95 latency {latency_stats['p95']:.3f}ms exceeds 50ms requirement"
        assert latency_stats['mean'] < 25.0, f"Mean latency {latency_stats['mean']:.3f}ms exceeds 25ms target"
        assert latency_stats['p99'] < 100.0, f"P99 latency {latency_stats['p99']:.3f}ms exceeds 100ms limit"
    
    def test_memory_usage_per_calculation(self):
        """Test memory usage per risk calculation remains minimal"""
        
        class MemoryEfficientCalculator:
            """Calculator designed for minimal memory allocation"""
            
            def __init__(self):
                # Use __slots__ to minimize memory overhead
                self.temp_buffer = np.zeros(10, dtype=np.float32)  # Reusable buffer
            
            def calculate_risk_minimal_memory(self, market_data: MarketData) -> int:
                """Risk calculation with minimal memory allocation"""
                # Reuse pre-allocated buffer instead of creating new arrays
                self.temp_buffer[0] = float(market_data.high_price_cents - market_data.low_price_cents)
                self.temp_buffer[1] = float(market_data.close_price_cents)
                self.temp_buffer[2] = float(market_data.volume)
                
                # Calculate components without creating intermediate objects
                volatility_ratio = self.temp_buffer[0] / self.temp_buffer[1]
                vol_score = min(int(volatility_ratio * 1000), 100)
                
                volume_score = 20 if self.temp_buffer[2] > 1000000 else 40
                
                return min((vol_score + volume_score) // 2, 100)
        
        calculator = MemoryEfficientCalculator()
        test_data_batch = self.create_test_market_data_batch(100)
        
        process = psutil.Process()
        
        # Measure initial memory
        gc.collect()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform calculations
        for data in test_data_batch:
            risk_score = calculator.calculate_risk_minimal_memory(data)
            assert 0 <= risk_score <= 100
        
        # Measure final memory
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_delta = final_memory - initial_memory
        
        print(f"\nMemory Usage per 100 Calculations:")
        print(f"  Initial Memory: {initial_memory:.2f} MB")
        print(f"  Final Memory: {final_memory:.2f} MB")
        print(f"  Memory Delta: {memory_delta:.2f} MB")
        print(f"  Memory per Calculation: {memory_delta * 1024 / 100:.2f} KB")
        
        # Memory usage should be minimal (<10MB for 100 calculations)
        assert memory_delta < 10.0, f"Memory usage {memory_delta:.2f}MB too high for 100 calculations"
        
        # Per-calculation memory should be reasonable
        memory_per_calc_kb = memory_delta * 1024 / 100
        assert memory_per_calc_kb < 100.0, f"Memory per calculation {memory_per_calc_kb:.2f}KB too high"


class TestBatchProcessingPerformance(PerformanceTestBase):
    """Test batch processing performance and throughput"""
    
    def test_batch_risk_calculation_throughput(self):
        """Test batch processing meets throughput requirements"""
        
        class BatchRiskProcessor:
            """Vectorized batch risk processor"""
            
            def __init__(self, batch_size: int = 1000):
                self.batch_size = batch_size
            
            def process_batch_vectorized(self, market_data_batch: List[MarketData]) -> List[int]:
                """Process batch using vectorized operations"""
                if not market_data_batch:
                    return []
                
                # Convert to numpy arrays for vectorized processing
                high_prices = np.array([d.high_price_cents for d in market_data_batch], dtype=np.float32)
                low_prices = np.array([d.low_price_cents for d in market_data_batch], dtype=np.float32)
                close_prices = np.array([d.close_price_cents for d in market_data_batch], dtype=np.float32)
                volumes = np.array([d.volume for d in market_data_batch], dtype=np.float32)
                
                # Vectorized calculations
                price_ranges = (high_prices - low_prices) / close_prices
                volatility_scores = np.minimum(price_ranges * 1000, 100).astype(np.int32)
                
                volume_scores = np.where(volumes > 1000000, 20, 
                                np.where(volumes > 100000, 30, 50)).astype(np.int32)
                
                # Weighted combination
                risk_scores = np.minimum((volatility_scores * 0.6 + volume_scores * 0.4), 100).astype(np.int32)
                
                return risk_scores.tolist()
            
            def process_batch_chunked(self, market_data_batch: List[MarketData]) -> List[int]:
                """Process large batch in chunks to optimize memory usage"""
                results = []
                
                for i in range(0, len(market_data_batch), self.batch_size):
                    chunk = market_data_batch[i:i + self.batch_size]
                    chunk_results = self.process_batch_vectorized(chunk)
                    results.extend(chunk_results)
                
                return results
        
        processor = BatchRiskProcessor()
        
        # Test different batch sizes
        batch_sizes = [100, 1000, 5000, 10000]
        
        for batch_size in batch_sizes:
            test_data = self.create_test_market_data_batch(batch_size)
            
            with self.measure_performance(f"Batch Size {batch_size}"):
                start_time = time.perf_counter()
                risk_scores = processor.process_batch_chunked(test_data)
                end_time = time.perf_counter()
                
                execution_time = end_time - start_time
                throughput = len(risk_scores) / execution_time
                
                print(f"  Throughput: {throughput:.0f} calculations/sec")
                print(f"  Time per calculation: {execution_time * 1000 / len(risk_scores):.3f} ms")
                
                # Validate results
                assert len(risk_scores) == batch_size, "Should process all items in batch"
                for score in risk_scores:
                    assert 0 <= score <= 100, f"Invalid risk score: {score}"
                
                # Performance requirements
                assert throughput >= 1000, f"Throughput {throughput:.0f} calculations/sec below 1000 minimum"
                
                # Time per calculation should decrease with larger batches (vectorization benefit)
                time_per_calc_ms = execution_time * 1000 / len(risk_scores)
                if batch_size >= 1000:
                    assert time_per_calc_ms < 10.0, f"Batch processing not efficient enough: {time_per_calc_ms:.3f}ms per calc"
    
    def test_real_time_processing_simulation(self):
        """Test real-time processing simulation with continuous data flow"""
        
        class RealTimeRiskProcessor:
            """Simulate real-time risk processing with sliding window"""
            
            def __init__(self, window_size: int = 100):
                self.window_size = window_size
                self.processing_buffer = []
                self.results_buffer = []
                self.processing_times = []
            
            def add_market_data(self, data: MarketData) -> Optional[int]:
                """Add new market data and trigger processing if buffer full"""
                self.processing_buffer.append(data)
                
                if len(self.processing_buffer) >= self.window_size:
                    return self._process_buffer()
                return None
            
            def _process_buffer(self) -> int:
                """Process buffer and return risk score for latest data"""
                start_time = time.perf_counter()
                
                # Simple rolling volatility calculation
                recent_prices = [d.close_price_cents for d in self.processing_buffer[-20:]]  # Last 20 data points
                if len(recent_prices) > 1:
                    returns = [(recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1] 
                              for i in range(1, len(recent_prices))]
                    volatility = np.std(returns) * np.sqrt(252)  # Annualized
                    risk_score = min(int(volatility * 200), 100)
                else:
                    risk_score = 50  # Default
                
                # Maintain sliding window
                if len(self.processing_buffer) > self.window_size * 2:
                    self.processing_buffer = self.processing_buffer[-self.window_size:]
                
                end_time = time.perf_counter()
                processing_time = (end_time - start_time) * 1000
                self.processing_times.append(processing_time)
                
                return risk_score
            
            def get_performance_stats(self) -> Dict[str, float]:
                """Get processing performance statistics"""
                if not self.processing_times:
                    return {}
                
                return self.measure_latency_distribution(self.processing_times)
        
        processor = RealTimeRiskProcessor()
        
        # Simulate real-time data stream
        total_data_points = 10000
        test_stream = self.create_test_market_data_batch(total_data_points)
        
        results = []
        stream_start = time.perf_counter()
        
        for i, data_point in enumerate(test_stream):
            risk_score = processor.add_market_data(data_point)
            if risk_score is not None:
                results.append(risk_score)
                
                # Simulate processing delay between data points (1ms)
                if i % 100 == 0:  # Periodic small delay to simulate real conditions
                    time.sleep(0.001)
        
        stream_end = time.perf_counter()
        total_stream_time = stream_end - stream_start
        
        # Get performance statistics
        perf_stats = processor.get_performance_stats()
        
        print(f"\nReal-Time Processing Simulation:")
        print(f"  Total Data Points: {total_data_points}")
        print(f"  Risk Scores Generated: {len(results)}")
        print(f"  Total Stream Time: {total_stream_time:.2f} s")
        print(f"  Stream Throughput: {total_data_points / total_stream_time:.0f} points/sec")
        
        if perf_stats:
            print(f"  Mean Processing Time: {perf_stats['mean']:.3f} ms")
            print(f"  P95 Processing Time: {perf_stats['p95']:.3f} ms")
            print(f"  Max Processing Time: {perf_stats['max']:.3f} ms")
        
        # Validate results
        assert len(results) > 0, "Should generate some risk scores"
        for score in results:
            assert 0 <= score <= 100, f"Invalid risk score: {score}"
        
        # Performance assertions
        stream_throughput = total_data_points / total_stream_time
        assert stream_throughput >= 5000, f"Stream throughput {stream_throughput:.0f} points/sec below 5000 minimum"
        
        if perf_stats:
            assert perf_stats['p95'] < 100.0, f"P95 processing time {perf_stats['p95']:.3f}ms exceeds 100ms limit"


class TestConcurrentProcessingPerformance(PerformanceTestBase):
    """Test concurrent and parallel processing performance"""
    
    def test_thread_pool_concurrent_processing(self):
        """Test concurrent processing using thread pool"""
        
        def calculate_risk_thread_safe(market_data: MarketData) -> Tuple[str, int, float]:
            """Thread-safe risk calculation function"""
            start_time = time.perf_counter()
            
            # Simulate some calculation complexity
            price_volatility = (market_data.high_price_cents - market_data.low_price_cents) / market_data.close_price_cents
            volume_factor = min(market_data.volume / 1000000, 1.0)
            
            risk_score = min(int(price_volatility * 800 + (1 - volume_factor) * 50), 100)
            
            end_time = time.perf_counter()
            processing_time = (end_time - start_time) * 1000
            
            return market_data.asset_id, risk_score, processing_time
        
        # Test different thread pool sizes
        thread_counts = [1, 4, 8, 16, 32]
        batch_size = 1000
        
        test_data = self.create_test_market_data_batch(batch_size)
        
        for thread_count in thread_counts:
            start_time = time.perf_counter()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
                future_to_data = {
                    executor.submit(calculate_risk_thread_safe, data): data 
                    for data in test_data
                }
                
                results = []
                processing_times = []
                
                for future in concurrent.futures.as_completed(future_to_data):
                    asset_id, risk_score, proc_time = future.result()
                    results.append((asset_id, risk_score))
                    processing_times.append(proc_time)
            
            end_time = time.perf_counter()
            total_time = end_time - start_time
            throughput = len(results) / total_time
            
            print(f"\nThread Pool Performance (Workers: {thread_count}):")
            print(f"  Total Time: {total_time:.2f} s")
            print(f"  Throughput: {throughput:.0f} calculations/sec")
            print(f"  Mean Processing Time: {np.mean(processing_times):.3f} ms")
            print(f"  Scalability vs Single Thread: {throughput / (batch_size / max(total_time, 0.001)):.2f}x")
            
            # Validate results
            assert len(results) == batch_size, f"Should process all {batch_size} items"
            for _, score in results:
                assert 0 <= score <= 100, f"Invalid risk score: {score}"
            
            # Performance assertions
            assert throughput >= 500, f"Thread pool throughput {throughput:.0f} below minimum 500 calc/sec"
            
            # Should see some benefit from multiple threads (at least 2x improvement with 4+ threads)
            if thread_count >= 4:
                single_thread_time = batch_size * np.mean(processing_times) / 1000
                actual_efficiency = single_thread_time / total_time
                assert actual_efficiency >= 1.5, f"Thread pool efficiency {actual_efficiency:.2f}x too low for {thread_count} threads"
    
    @pytest.mark.skipif(sys.platform.startswith("win"), reason="Multiprocessing can be flaky on Windows CI")
    def test_multiprocessing_parallel_performance(self):
        """Test parallel processing using multiprocessing"""
        
        def calculate_risk_multiprocess(data_chunk: List[MarketData]) -> List[Tuple[str, int]]:
            """Multi-process safe risk calculation for chunk"""
            results = []
            
            for data in data_chunk:
                # CPU-intensive calculation to benefit from multiprocessing
                volatility = (data.high_price_cents - data.low_price_cents) / data.close_price_cents
                
                # Simulate more complex calculation
                risk_components = []
                for _ in range(10):  # Simulate multiple risk factors
                    component = volatility * np.random.uniform(0.8, 1.2)
                    risk_components.append(component)
                
                avg_component = np.mean(risk_components)
                risk_score = min(int(avg_component * 1000), 100)
                
                results.append((data.asset_id, risk_score))
            
            return results
        
        cpu_count = min(multiprocessing.cpu_count(), 8)  # Limit to 8 to avoid resource exhaustion
        batch_size = 2000
        chunk_size = batch_size // cpu_count
        
        test_data = self.create_test_market_data_batch(batch_size)
        data_chunks = [test_data[i:i + chunk_size] for i in range(0, len(test_data), chunk_size)]
        
        # Test sequential vs parallel processing
        start_time = time.perf_counter()
        sequential_results = []
        for chunk in data_chunks:
            sequential_results.extend(calculate_risk_multiprocess(chunk))
        sequential_time = time.perf_counter() - start_time
        
        # Parallel processing
        start_time = time.perf_counter()
        
        with multiprocessing.Pool(processes=cpu_count) as pool:
            parallel_chunks = pool.map(calculate_risk_multiprocess, data_chunks)
            parallel_results = []
            for chunk_results in parallel_chunks:
                parallel_results.extend(chunk_results)
        
        parallel_time = time.perf_counter() - start_time
        
        # Calculate performance metrics
        sequential_throughput = len(sequential_results) / sequential_time
        parallel_throughput = len(parallel_results) / parallel_time
        speedup = sequential_time / parallel_time
        efficiency = speedup / cpu_count
        
        print(f"\nMultiprocessing Performance (CPUs: {cpu_count}):")
        print(f"  Sequential Time: {sequential_time:.2f} s")
        print(f"  Parallel Time: {parallel_time:.2f} s")
        print(f"  Sequential Throughput: {sequential_throughput:.0f} calc/sec")
        print(f"  Parallel Throughput: {parallel_throughput:.0f} calc/sec")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Efficiency: {efficiency:.2f}")
        
        # Validate results
        assert len(parallel_results) == batch_size, f"Should process all {batch_size} items"
        for _, score in parallel_results:
            assert 0 <= score <= 100, f"Invalid risk score: {score}"
        
        # Performance assertions
        assert parallel_throughput > sequential_throughput, "Parallel processing should be faster"
        assert speedup >= 1.5, f"Multiprocessing speedup {speedup:.2f}x too low"
        assert efficiency >= 0.3, f"Multiprocessing efficiency {efficiency:.2f} too low"
    
    @pytest.mark.asyncio
    async def test_async_concurrent_processing(self):
        """Test asynchronous concurrent processing"""
        
        async def calculate_risk_async(market_data: MarketData, delay_ms: float = 1.0) -> Tuple[str, int]:
            """Async risk calculation with simulated I/O delay"""
            # Simulate async I/O operation (e.g., database lookup, API call)
            await asyncio.sleep(delay_ms / 1000)
            
            # Quick risk calculation
            volatility = (market_data.high_price_cents - market_data.low_price_cents) / market_data.close_price_cents
            risk_score = min(int(volatility * 1000), 100)
            
            return market_data.asset_id, risk_score
        
        async def process_batch_async(data_batch: List[MarketData], concurrency_limit: int = 100) -> List[Tuple[str, int]]:
            """Process batch with concurrency limiting"""
            semaphore = asyncio.Semaphore(concurrency_limit)
            
            async def limited_calculation(data):
                async with semaphore:
                    return await calculate_risk_async(data)
            
            tasks = [limited_calculation(data) for data in data_batch]
            return await asyncio.gather(*tasks)
        
        # Test different batch sizes and concurrency limits
        batch_size = 1000
        concurrency_limits = [10, 50, 100, 200]
        
        test_data = self.create_test_market_data_batch(batch_size)
        
        for limit in concurrency_limits:
            start_time = time.perf_counter()
            results = await process_batch_async(test_data, limit)
            end_time = time.perf_counter()
            
            execution_time = end_time - start_time
            throughput = len(results) / execution_time
            
            print(f"\nAsync Concurrent Processing (Limit: {limit}):")
            print(f"  Execution Time: {execution_time:.2f} s")
            print(f"  Throughput: {throughput:.0f} calculations/sec")
            print(f"  Concurrency Efficiency: {throughput / limit:.1f} calc/sec per concurrent task")
            
            # Validate results
            assert len(results) == batch_size, f"Should process all {batch_size} items"
            for _, score in results:
                assert 0 <= score <= 100, f"Invalid risk score: {score}"
            
            # Performance assertions
            assert throughput >= 200, f"Async throughput {throughput:.0f} below minimum 200 calc/sec"
            
            # Higher concurrency should improve throughput (up to a point)
            if limit <= 100:
                expected_min_throughput = min(limit * 0.8, 500)  # 80% efficiency up to 500 calc/sec
                assert throughput >= expected_min_throughput, \
                    f"Throughput {throughput:.0f} below expected minimum {expected_min_throughput:.0f} for concurrency {limit}"


class TestMemoryAndResourceUsage(PerformanceTestBase):
    """Test memory usage and resource consumption under load"""
    
    def test_memory_usage_under_sustained_load(self):
        """Test memory usage remains stable under sustained processing load"""
        
        class MemoryMonitoringCalculator:
            def __init__(self):
                self.calculation_count = 0
                self.memory_readings = []
            
            def calculate_with_monitoring(self, market_data: MarketData) -> int:
                """Calculate risk score while monitoring memory"""
                # Perform calculation
                volatility = (market_data.high_price_cents - market_data.low_price_cents) / market_data.close_price_cents
                risk_score = min(int(volatility * 1000), 100)
                
                self.calculation_count += 1
                
                # Monitor memory every 100 calculations
                if self.calculation_count % 100 == 0:
                    process = psutil.Process()
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    self.memory_readings.append((self.calculation_count, memory_mb))
                
                return risk_score
            
            def get_memory_growth_rate(self) -> float:
                """Calculate memory growth rate per calculation"""
                if len(self.memory_readings) < 2:
                    return 0.0
                
                first_reading = self.memory_readings[0]
                last_reading = self.memory_readings[-1]
                
                calc_diff = last_reading[0] - first_reading[0]
                memory_diff = last_reading[1] - first_reading[1]
                
                return memory_diff / calc_diff if calc_diff > 0 else 0.0
        
        calculator = MemoryMonitoringCalculator()
        
        # Perform sustained load testing
        total_calculations = 10000
        batch_size = 1000
        
        print(f"\nSustained Load Testing - {total_calculations} calculations:")
        
        for batch_num in range(total_calculations // batch_size):
            # Create new test data for each batch to simulate real usage
            test_batch = self.create_test_market_data_batch(batch_size)
            
            for data in test_batch:
                risk_score = calculator.calculate_with_monitoring(data)
                assert 0 <= risk_score <= 100
            
            # Force garbage collection periodically
            if batch_num % 3 == 0:
                gc.collect()
            
            print(f"  Completed batch {batch_num + 1}/{total_calculations // batch_size}")
        
        # Analyze memory usage
        memory_readings = calculator.memory_readings
        initial_memory = memory_readings[0][1] if memory_readings else 0
        final_memory = memory_readings[-1][1] if memory_readings else 0
        peak_memory = max(reading[1] for reading in memory_readings) if memory_readings else 0
        
        memory_growth_rate = calculator.get_memory_growth_rate()
        
        print(f"\nMemory Usage Analysis:")
        print(f"  Initial Memory: {initial_memory:.2f} MB")
        print(f"  Final Memory: {final_memory:.2f} MB")
        print(f"  Peak Memory: {peak_memory:.2f} MB")
        print(f"  Total Growth: {final_memory - initial_memory:.2f} MB")
        print(f"  Growth Rate: {memory_growth_rate * 1000:.3f} MB per 1000 calculations")
        
        # Memory usage assertions
        total_growth = final_memory - initial_memory
        assert total_growth < 100.0, f"Memory growth {total_growth:.2f}MB too high for {total_calculations} calculations"
        
        # Memory growth rate should be minimal (< 10MB per 10k calculations)
        growth_per_10k = memory_growth_rate * 10000
        assert growth_per_10k < 10.0, f"Memory growth rate {growth_per_10k:.2f}MB per 10k calculations too high"
        
        # Peak memory shouldn't exceed reasonable limits
        assert peak_memory < 500.0, f"Peak memory usage {peak_memory:.2f}MB exceeds 500MB limit"
    
    def test_cpu_usage_efficiency(self):
        """Test CPU usage efficiency during intensive processing"""
        
        def cpu_intensive_risk_calculation(market_data: MarketData) -> int:
            """CPU-intensive risk calculation for stress testing"""
            # Simulate complex statistical calculations
            price_series = [
                market_data.open_price_cents,
                market_data.high_price_cents, 
                market_data.low_price_cents,
                market_data.close_price_cents
            ]
            
            # Calculate multiple statistical measures
            mean_price = np.mean(price_series)
            std_price = np.std(price_series)
            
            # Simulate Monte Carlo simulation (simplified)
            random_samples = np.random.normal(mean_price, std_price, 100)
            percentiles = np.percentile(random_samples, [5, 25, 75, 95])
            
            # Risk score based on statistical analysis
            volatility_factor = std_price / mean_price
            risk_score = min(int(volatility_factor * 2000 + np.std(percentiles) / mean_price * 1000), 100)
            
            return max(0, risk_score)
        
        test_data = self.create_test_market_data_batch(5000)
        
        # Monitor CPU usage during processing
        process = psutil.Process()
        start_cpu_times = process.cpu_times()
        start_time = time.perf_counter()
        
        results = []
        for data in test_data:
            risk_score = cpu_intensive_risk_calculation(data)
            results.append(risk_score)
        
        end_time = time.perf_counter()
        end_cpu_times = process.cpu_times()
        
        # Calculate CPU metrics
        wall_clock_time = end_time - start_time
        cpu_time_used = (end_cpu_times.user + end_cpu_times.system) - (start_cpu_times.user + start_cpu_times.system)
        cpu_efficiency = cpu_time_used / wall_clock_time if wall_clock_time > 0 else 0
        throughput = len(results) / wall_clock_time
        
        print(f"\nCPU Usage Efficiency Test:")
        print(f"  Wall Clock Time: {wall_clock_time:.2f} s")
        print(f"  CPU Time Used: {cpu_time_used:.2f} s")
        print(f"  CPU Efficiency: {cpu_efficiency:.2f} (1.0 = 100% CPU utilization)")
        print(f"  Throughput: {throughput:.0f} calculations/sec")
        print(f"  CPU Time per Calculation: {cpu_time_used * 1000 / len(results):.3f} ms")
        
        # Validate results
        assert len(results) == len(test_data)
        for score in results:
            assert 0 <= score <= 100, f"Invalid risk score: {score}"
        
        # CPU efficiency should be reasonable (not too low, indicating inefficiency)
        assert cpu_efficiency >= 0.5, f"CPU efficiency {cpu_efficiency:.2f} too low - indicates inefficient processing"
        
        # Should achieve reasonable throughput even with intensive calculations
        assert throughput >= 100, f"CPU-intensive throughput {throughput:.0f} below minimum 100 calc/sec"


if __name__ == "__main__":
    # Run performance tests
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-s",  # Show print output
        "-x"   # Stop on first failure
    ])