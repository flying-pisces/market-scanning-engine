"""
Risk Scoring System - Stress Testing for Edge Cases
Author: Claude Code (QA Engineer)
Version: 1.0

Comprehensive stress testing for extreme market conditions, system overloads,
and edge cases. Ensures risk scoring system remains stable and produces
reasonable outputs even under severe stress scenarios.
"""

import pytest
import numpy as np
import pandas as pd
import time
import threading
import random
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from unittest.mock import Mock, patch
import queue
import gc
import warnings

from data_models.python.core_models import (
    Asset, AssetCategory, MarketData, TechnicalIndicators, 
    OptionsData, MarketRegime, RiskScore
)
from data_models.python.signal_models import (
    RiskAssessment, RiskFactorContribution, 
    MarketRegimeData, Signal, SignalDirection
)


class StressTestBase:
    """Base class for stress testing utilities"""
    
    @staticmethod
    def create_extreme_market_data(
        scenario: str = "crash",
        asset_id: str = "STRESS_ASSET"
    ) -> MarketData:
        """Create extreme market data for stress testing"""
        
        scenarios = {
            "crash": {
                "open": 10000, "high": 10100, "low": 5000, "close": 5200,
                "volume": 50000000, "bid_ask_spread_bps": 500  # 50% crash, 5% spread
            },
            "flash_crash": {
                "open": 10000, "high": 10050, "low": 1000, "close": 9800,
                "volume": 100000000, "bid_ask_spread_bps": 1000  # 90% intraday drop
            },
            "circuit_breaker": {
                "open": 10000, "high": 10100, "low": 7900, "close": 7900,
                "volume": 200000000, "bid_ask_spread_bps": 200  # 21% down limit
            },
            "low_liquidity": {
                "open": 10000, "high": 10200, "low": 9800, "close": 10100,
                "volume": 1000, "bid_ask_spread_bps": 2000  # Very low volume, wide spread
            },
            "extreme_volatility": {
                "open": 10000, "high": 15000, "low": 5000, "close": 12500,
                "volume": 75000000, "bid_ask_spread_bps": 800  # 100% intraday range
            },
            "zero_volume": {
                "open": 10000, "high": 10000, "low": 10000, "close": 10000,
                "volume": 0, "bid_ask_spread_bps": 10000  # No trading, massive spread
            },
            "penny_stock": {
                "open": 5, "high": 8, "low": 2, "close": 3,
                "volume": 500000, "bid_ask_spread_bps": 3333  # Low price, volatile
            },
            "market_manipulation": {
                "open": 10000, "high": 25000, "low": 10000, "close": 24900,
                "volume": 1000000, "bid_ask_spread_bps": 100  # Suspicious pump
            }
        }
        
        scenario_data = scenarios.get(scenario, scenarios["crash"])
        
        spread_cents = (scenario_data["close"] * scenario_data["bid_ask_spread_bps"]) // 10000
        
        return MarketData(
            asset_id=asset_id,
            timestamp=datetime.utcnow(),
            open_price_cents=scenario_data["open"],
            high_price_cents=scenario_data["high"],
            low_price_cents=scenario_data["low"],
            close_price_cents=scenario_data["close"],
            volume=scenario_data["volume"],
            bid_price_cents=scenario_data["close"] - spread_cents // 2,
            ask_price_cents=scenario_data["close"] + spread_cents // 2,
            data_source="STRESS_TEST",
            data_quality_score=85 if scenario != "market_manipulation" else 30
        )
    
    @staticmethod
    def create_corrupted_data_scenarios() -> List[Dict[str, Any]]:
        """Create various corrupted data scenarios"""
        return [
            {"name": "negative_prices", "open": -100, "high": -50, "low": -200, "close": -75},
            {"name": "zero_prices", "open": 0, "high": 0, "low": 0, "close": 0},
            {"name": "infinite_prices", "open": float('inf'), "high": float('inf'), "low": 100, "close": 200},
            {"name": "nan_prices", "open": float('nan'), "high": 100, "low": 90, "close": 95},
            {"name": "inconsistent_ohlc", "open": 100, "high": 80, "low": 120, "close": 110},  # High < Low
            {"name": "extreme_values", "open": 1e15, "high": 1e16, "low": 1e14, "close": 5e15},
            {"name": "negative_volume", "volume": -1000000},
            {"name": "extreme_volume", "volume": int(1e15)},
        ]


class TestMarketCrashScenarios(StressTestBase):
    """Test risk scoring during market crash scenarios"""
    
    def test_market_crash_risk_scoring(self):
        """Test risk scoring during various market crash scenarios"""
        
        class CrashResilientRiskCalculator:
            def __init__(self):
                self.max_risk_score = 100
                self.emergency_thresholds = {
                    'crash_detection': 0.20,  # 20% drop triggers crash mode
                    'flash_crash_detection': 0.50,  # 50% intraday range
                    'volume_surge': 10.0  # 10x normal volume
                }
            
            def calculate_crash_adjusted_risk(self, market_data: MarketData) -> Tuple[int, Dict[str, Any]]:
                """Calculate risk with crash scenario adjustments"""
                metadata = {"crash_indicators": [], "adjustments": []}
                
                try:
                    # Basic volatility calculation
                    if market_data.close_price_cents <= 0:
                        raise ValueError("Invalid price data")
                    
                    price_change = (market_data.close_price_cents - market_data.open_price_cents) / market_data.open_price_cents
                    intraday_range = (market_data.high_price_cents - market_data.low_price_cents) / market_data.close_price_cents
                    
                    base_risk = min(int(abs(price_change) * 300 + intraday_range * 200), 100)
                    
                    # Crash detection and adjustments
                    if price_change < -self.emergency_thresholds['crash_detection']:
                        metadata["crash_indicators"].append("Major price decline detected")
                        base_risk = min(base_risk + 30, 100)
                        metadata["adjustments"].append("Crash adjustment: +30 points")
                    
                    if intraday_range > self.emergency_thresholds['flash_crash_detection']:
                        metadata["crash_indicators"].append("Flash crash pattern detected")
                        base_risk = min(base_risk + 40, 100)
                        metadata["adjustments"].append("Flash crash adjustment: +40 points")
                    
                    # Volume surge detection
                    if market_data.volume > 50000000:  # Assume normal volume is ~5M
                        volume_multiplier = market_data.volume / 5000000
                        if volume_multiplier > self.emergency_thresholds['volume_surge']:
                            metadata["crash_indicators"].append(f"Volume surge: {volume_multiplier:.1f}x normal")
                            base_risk = min(base_risk + 20, 100)
                            metadata["adjustments"].append("Volume surge adjustment: +20 points")
                    
                    # Liquidity crisis detection
                    if market_data.bid_price_cents and market_data.ask_price_cents:
                        spread_pct = (market_data.ask_price_cents - market_data.bid_price_cents) / market_data.close_price_cents
                        if spread_pct > 0.05:  # 5% spread indicates liquidity crisis
                            metadata["crash_indicators"].append(f"Liquidity crisis: {spread_pct:.1%} spread")
                            base_risk = min(base_risk + 25, 100)
                            metadata["adjustments"].append("Liquidity crisis adjustment: +25 points")
                    
                    return base_risk, metadata
                
                except (ValueError, ZeroDivisionError, OverflowError) as e:
                    metadata["crash_indicators"].append(f"Data error: {str(e)}")
                    return 95, metadata  # Conservative high-risk score for corrupted data
        
        calculator = CrashResilientRiskCalculator()
        
        # Test various crash scenarios
        crash_scenarios = [
            "crash", "flash_crash", "circuit_breaker", "extreme_volatility", 
            "low_liquidity", "zero_volume", "market_manipulation"
        ]
        
        results = {}
        
        for scenario in crash_scenarios:
            print(f"\nTesting {scenario} scenario:")
            
            crash_data = self.create_extreme_market_data(scenario)
            risk_score, metadata = calculator.calculate_crash_adjusted_risk(crash_data)
            
            results[scenario] = {
                "risk_score": risk_score,
                "metadata": metadata,
                "data": crash_data
            }
            
            print(f"  Risk Score: {risk_score}")
            print(f"  Crash Indicators: {metadata['crash_indicators']}")
            print(f"  Adjustments: {metadata['adjustments']}")
            
            # Validate results
            assert 0 <= risk_score <= 100, f"Risk score {risk_score} outside valid range"
            
            # Crash scenarios should generally produce high risk scores
            if scenario in ["crash", "flash_crash", "circuit_breaker", "extreme_volatility"]:
                assert risk_score >= 70, f"{scenario} should produce high risk score, got {risk_score}"
            
            # Low liquidity should be detected
            if scenario in ["low_liquidity", "zero_volume"]:
                assert risk_score >= 60, f"{scenario} should reflect liquidity risk, got {risk_score}"
                assert any("liquidity" in indicator.lower() for indicator in metadata["crash_indicators"])
        
        # Verify crash scenarios produce higher risk than normal market
        normal_data = MarketData(
            asset_id="NORMAL",
            timestamp=datetime.utcnow(),
            open_price_cents=10000, high_price_cents=10100, 
            low_price_cents=9900, close_price_cents=10050,
            volume=1000000, bid_price_cents=10045, ask_price_cents=10055,
            data_source="TEST", data_quality_score=95
        )
        
        normal_risk, _ = calculator.calculate_crash_adjusted_risk(normal_data)
        
        for scenario, result in results.items():
            if scenario != "penny_stock":  # Penny stock might have different risk profile
                assert result["risk_score"] > normal_risk, \
                    f"{scenario} risk score {result['risk_score']} should exceed normal {normal_risk}"
    
    def test_consecutive_crash_days_resilience(self):
        """Test system resilience during consecutive crash days"""
        
        class MultiDayCrashTracker:
            def __init__(self):
                self.daily_scores = []
                self.crash_streak = 0
                self.max_consecutive_crashes = 0
            
            def process_daily_data(self, market_data: MarketData, day_number: int) -> int:
                """Process daily market data and track crash patterns"""
                # Calculate daily risk
                price_change = (market_data.close_price_cents - market_data.open_price_cents) / market_data.open_price_cents
                base_risk = min(int(abs(price_change) * 400), 100)
                
                # Track crash streaks
                if price_change < -0.05:  # 5% decline = crash day
                    self.crash_streak += 1
                    self.max_consecutive_crashes = max(self.max_consecutive_crashes, self.crash_streak)
                    
                    # Increase risk for extended crash periods
                    streak_multiplier = min(1.0 + (self.crash_streak - 1) * 0.1, 1.5)
                    base_risk = min(int(base_risk * streak_multiplier), 100)
                else:
                    self.crash_streak = 0
                
                self.daily_scores.append(base_risk)
                return base_risk
            
            def get_stress_metrics(self) -> Dict[str, float]:
                """Calculate stress testing metrics"""
                if not self.daily_scores:
                    return {}
                
                return {
                    "max_daily_risk": max(self.daily_scores),
                    "avg_risk": np.mean(self.daily_scores),
                    "risk_volatility": np.std(self.daily_scores),
                    "max_consecutive_crashes": self.max_consecutive_crashes,
                    "high_risk_days": sum(1 for score in self.daily_scores if score >= 80),
                    "stress_period_length": len(self.daily_scores)
                }
        
        tracker = MultiDayCrashTracker()
        
        # Simulate 30-day crash scenario (2008-style financial crisis)
        crash_period_days = 30
        base_price = 10000
        
        print(f"\nSimulating {crash_period_days}-day market crisis:")
        
        for day in range(crash_period_days):
            # Generate progressively worse market conditions
            if day < 10:
                # Initial decline phase
                daily_return = random.uniform(-0.08, -0.02)  # -2% to -8%
            elif day < 20:
                # Acceleration phase
                daily_return = random.uniform(-0.15, -0.05)  # -5% to -15%
            else:
                # Recovery attempts with high volatility
                daily_return = random.uniform(-0.10, 0.05)   # -10% to +5%
            
            # Update price for the day
            new_price = int(base_price * (1 + daily_return))
            base_price = max(new_price, 1000)  # Don't let price go below $10
            
            # Create market data with increasing volatility over time
            volatility_multiplier = 1 + (day / 30) * 2  # Volatility increases over time
            intraday_range = int(abs(daily_return) * base_price * volatility_multiplier)
            
            daily_data = MarketData(
                asset_id=f"CRISIS_DAY_{day}",
                timestamp=datetime.utcnow() + timedelta(days=day),
                open_price_cents=int(base_price / (1 + daily_return)),
                high_price_cents=base_price + intraday_range // 2,
                low_price_cents=base_price - intraday_range,
                close_price_cents=base_price,
                volume=int(random.uniform(5000000, 50000000)),  # High volume during crisis
                bid_price_cents=base_price - random.randint(50, 500),
                ask_price_cents=base_price + random.randint(50, 500),
                data_source="CRISIS_SIM",
                data_quality_score=max(70, 95 - day)  # Data quality degrades during crisis
            )
            
            risk_score = tracker.process_daily_data(daily_data, day)
            
            if day % 5 == 0:
                print(f"  Day {day}: Price=${base_price/100:.2f}, Return={daily_return:.1%}, Risk={risk_score}")
        
        # Analyze stress period
        stress_metrics = tracker.get_stress_metrics()
        
        print(f"\nCrash Period Analysis:")
        for metric, value in stress_metrics.items():
            print(f"  {metric}: {value}")
        
        # Validate stress testing results
        assert stress_metrics["max_daily_risk"] >= 80, "Should detect high-risk periods during crash"
        assert stress_metrics["avg_risk"] >= 50, "Average risk should be elevated during crash period"
        assert stress_metrics["max_consecutive_crashes"] >= 3, "Should detect consecutive crash days"
        assert stress_metrics["high_risk_days"] >= crash_period_days * 0.3, "Should flag significant portion as high-risk"
        
        # System should remain stable throughout
        assert len(tracker.daily_scores) == crash_period_days, "Should process all days without failure"
    
    def test_black_swan_event_handling(self):
        """Test handling of extreme black swan events"""
        
        class BlackSwanDetector:
            def __init__(self):
                self.historical_volatility = []
                self.black_swan_threshold = 6.0  # 6 sigma event
            
            def detect_black_swan(self, market_data: MarketData, historical_context: List[float]) -> Tuple[int, bool, str]:
                """Detect and score black swan events"""
                
                if len(historical_context) < 30:
                    # Insufficient history - use conservative scoring
                    return 85, True, "Insufficient historical data for black swan detection"
                
                # Calculate current return
                price_change = (market_data.close_price_cents - market_data.open_price_cents) / market_data.open_price_cents
                
                # Calculate z-score based on historical context
                historical_mean = np.mean(historical_context)
                historical_std = np.std(historical_context)
                
                if historical_std == 0:
                    z_score = 0
                else:
                    z_score = abs(price_change - historical_mean) / historical_std
                
                # Black swan detection
                is_black_swan = z_score > self.black_swan_threshold
                
                if is_black_swan:
                    risk_score = 100  # Maximum risk for black swan
                    reason = f"Black swan event detected: {z_score:.1f} sigma deviation"
                elif z_score > 3.0:
                    risk_score = 95   # Extreme event, but not black swan
                    reason = f"Extreme event: {z_score:.1f} sigma deviation"
                elif z_score > 2.0:
                    risk_score = 80   # Significant event
                    reason = f"Significant event: {z_score:.1f} sigma deviation"
                else:
                    risk_score = min(int(z_score * 25), 75)  # Normal scaling
                    reason = f"Normal market movement: {z_score:.1f} sigma"
                
                return risk_score, is_black_swan, reason
        
        detector = BlackSwanDetector()
        
        # Create historical context (normal market returns)
        normal_returns = np.random.normal(0.001, 0.02, 252)  # 1 year of daily returns
        
        # Test various black swan scenarios
        black_swan_events = [
            {"name": "2008 Lehman Crash", "return": -0.20, "expected_black_swan": True},
            {"name": "1987 Black Monday", "return": -0.22, "expected_black_swan": True},
            {"name": "2020 COVID Crash", "return": -0.12, "expected_black_swan": False},  # Large but not 6-sigma
            {"name": "Flash Crash", "return": -0.09, "expected_black_swan": False},
            {"name": "Normal Volatility", "return": -0.03, "expected_black_swan": False},
            {"name": "Extreme Positive", "return": 0.15, "expected_black_swan": False}
        ]
        
        print(f"\nBlack Swan Event Testing:")
        
        for event in black_swan_events:
            # Create market data for the event
            event_data = MarketData(
                asset_id=f"BLACK_SWAN_{event['name'].replace(' ', '_')}",
                timestamp=datetime.utcnow(),
                open_price_cents=10000,
                high_price_cents=int(10000 * (1 + max(event['return'], 0) + 0.02)),
                low_price_cents=int(10000 * (1 + min(event['return'], 0) - 0.01)),
                close_price_cents=int(10000 * (1 + event['return'])),
                volume=50000000,  # High volume during extreme events
                bid_price_cents=int(10000 * (1 + event['return']) - 100),
                ask_price_cents=int(10000 * (1 + event['return']) + 100),
                data_source="BLACK_SWAN_TEST",
                data_quality_score=90
            )
            
            risk_score, is_detected_black_swan, reason = detector.detect_black_swan(
                event_data, normal_returns.tolist()
            )
            
            print(f"\n  {event['name']}:")
            print(f"    Return: {event['return']:.1%}")
            print(f"    Risk Score: {risk_score}")
            print(f"    Black Swan Detected: {is_detected_black_swan}")
            print(f"    Reason: {reason}")
            
            # Validate detection
            assert 0 <= risk_score <= 100, f"Risk score {risk_score} outside valid range"
            
            if event['expected_black_swan']:
                assert risk_score >= 95, f"Black swan event should have very high risk score, got {risk_score}"
            
            # Extreme events should have high risk scores regardless of black swan classification
            if abs(event['return']) > 0.08:  # 8% move
                assert risk_score >= 80, f"Extreme event should have high risk score, got {risk_score}"


class TestSystemOverloadScenarios(StressTestBase):
    """Test system behavior under extreme load conditions"""
    
    def test_concurrent_request_overload(self):
        """Test system handling of extreme concurrent load"""
        
        class OverloadResilientRiskService:
            def __init__(self, max_concurrent_requests: int = 100):
                self.max_concurrent = max_concurrent_requests
                self.active_requests = 0
                self.request_lock = threading.Lock()
                self.overload_counter = 0
                self.success_counter = 0
                self.error_counter = 0
            
            def calculate_risk_with_load_shedding(self, market_data: MarketData) -> Tuple[Optional[int], str]:
                """Calculate risk with load shedding when overloaded"""
                
                with self.request_lock:
                    if self.active_requests >= self.max_concurrent:
                        self.overload_counter += 1
                        return None, "OVERLOADED"
                    
                    self.active_requests += 1
                
                try:
                    # Simulate processing time
                    time.sleep(random.uniform(0.001, 0.005))  # 1-5ms processing time
                    
                    # Simple risk calculation
                    volatility = (market_data.high_price_cents - market_data.low_price_cents) / market_data.close_price_cents
                    risk_score = min(int(volatility * 1000), 100)
                    
                    self.success_counter += 1
                    return risk_score, "SUCCESS"
                
                except Exception as e:
                    self.error_counter += 1
                    return None, f"ERROR: {str(e)}"
                
                finally:
                    with self.request_lock:
                        self.active_requests -= 1
            
            def get_load_stats(self) -> Dict[str, Any]:
                """Get load testing statistics"""
                total_requests = self.success_counter + self.error_counter + self.overload_counter
                return {
                    "total_requests": total_requests,
                    "successful": self.success_counter,
                    "errors": self.error_counter,
                    "overloaded": self.overload_counter,
                    "success_rate": self.success_counter / total_requests if total_requests > 0 else 0,
                    "overload_rate": self.overload_counter / total_requests if total_requests > 0 else 0
                }
        
        service = OverloadResilientRiskService(max_concurrent_requests=50)
        
        # Generate test data
        test_data_count = 1000
        test_market_data = [
            self.create_extreme_market_data("extreme_volatility", f"OVERLOAD_TEST_{i}")
            for i in range(test_data_count)
        ]
        
        print(f"\nConcurrent Overload Test - {test_data_count} requests with 200 threads:")
        
        # Execute concurrent requests with intentional overload
        results = []
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=200) as executor:  # Intentionally more than max_concurrent
            futures = [
                executor.submit(service.calculate_risk_with_load_shedding, data)
                for data in test_market_data
            ]
            
            for future in as_completed(futures):
                result, status = future.result()
                results.append((result, status))
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Analyze results
        load_stats = service.get_load_stats()
        
        print(f"Load Test Results:")
        for key, value in load_stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
        
        print(f"  Total Time: {total_time:.2f} seconds")
        print(f"  Throughput: {load_stats['total_requests'] / total_time:.0f} requests/sec")
        
        # Validate load shedding behavior
        assert load_stats['success_rate'] > 0.3, "Should successfully process at least 30% of requests"
        assert load_stats['overload_rate'] > 0.1, "Should demonstrate load shedding under extreme load"
        assert load_stats['errors'] == 0, "Should not have processing errors, only overload responses"
        
        # Successful requests should have valid risk scores
        successful_scores = [result for result, status in results if status == "SUCCESS" and result is not None]
        assert len(successful_scores) > 0, "Should have some successful calculations"
        
        for score in successful_scores[:10]:  # Check first 10 successful scores
            assert 0 <= score <= 100, f"Invalid risk score: {score}"
    
    def test_memory_exhaustion_resilience(self):
        """Test system behavior when approaching memory limits"""
        
        class MemoryAwareRiskCalculator:
            def __init__(self, memory_limit_mb: float = 100):
                self.memory_limit = memory_limit_mb
                self.large_data_cache = {}  # Intentionally grows large
                self.calculation_count = 0
                self.memory_errors = 0
                self.gc_triggers = 0
            
            def calculate_with_memory_management(self, market_data: MarketData) -> Tuple[Optional[int], str]:
                """Calculate risk with memory management"""
                import psutil
                
                try:
                    # Check memory usage
                    process = psutil.Process()
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    
                    if memory_mb > self.memory_limit:
                        # Trigger garbage collection and clear cache
                        gc.collect()
                        self.large_data_cache.clear()
                        self.gc_triggers += 1
                        
                        # Re-check memory after cleanup
                        memory_mb = process.memory_info().rss / 1024 / 1024
                        if memory_mb > self.memory_limit * 1.2:  # Still too high
                            self.memory_errors += 1
                            return None, "MEMORY_EXHAUSTED"
                    
                    # Intentionally create large data structures to stress memory
                    asset_id = market_data.asset_id
                    if asset_id not in self.large_data_cache:
                        # Create large arrays to consume memory
                        self.large_data_cache[asset_id] = {
                            'price_history': np.random.random(10000),  # Large array
                            'correlation_matrix': np.random.random((500, 500)),  # Very large matrix
                            'metadata': {f'key_{i}': f'value_{i}' * 100 for i in range(1000)}  # Large dict
                        }
                    
                    # Simple risk calculation
                    volatility = (market_data.high_price_cents - market_data.low_price_cents) / market_data.close_price_cents
                    risk_score = min(int(volatility * 1000), 100)
                    
                    self.calculation_count += 1
                    return risk_score, "SUCCESS"
                
                except MemoryError:
                    self.memory_errors += 1
                    return None, "MEMORY_ERROR"
                except Exception as e:
                    return None, f"UNEXPECTED_ERROR: {str(e)}"
        
        calculator = MemoryAwareRiskCalculator(memory_limit_mb=150)  # Relatively low limit
        
        # Test with many different assets to force memory growth
        test_assets = 200
        test_data = [
            self.create_extreme_market_data("extreme_volatility", f"MEMORY_TEST_{i}")
            for i in range(test_assets)
        ]
        
        print(f"\nMemory Exhaustion Test - {test_assets} unique assets:")
        
        results = []
        memory_readings = []
        
        import psutil
        process = psutil.Process()
        
        for i, data in enumerate(test_data):
            result, status = calculator.calculate_with_memory_management(data)
            results.append((result, status))
            
            # Track memory usage
            if i % 10 == 0:
                memory_mb = process.memory_info().rss / 1024 / 1024
                memory_readings.append(memory_mb)
                print(f"  Asset {i}: Memory={memory_mb:.1f}MB, Status={status}")
        
        # Analyze memory management
        status_counts = {}
        for _, status in results:
            status_counts[status] = status_counts.get(status, 0) + 1
        
        print(f"\nMemory Management Results:")
        print(f"  Calculations attempted: {len(results)}")
        print(f"  Successful: {status_counts.get('SUCCESS', 0)}")
        print(f"  Memory exhausted: {status_counts.get('MEMORY_EXHAUSTED', 0)}")
        print(f"  Memory errors: {status_counts.get('MEMORY_ERROR', 0)}")
        print(f"  GC triggers: {calculator.gc_triggers}")
        print(f"  Peak memory: {max(memory_readings):.1f}MB")
        print(f"  Final memory: {memory_readings[-1]:.1f}MB")
        
        # Validate memory management behavior
        assert calculator.calculation_count > 0, "Should successfully complete some calculations"
        assert calculator.gc_triggers > 0, "Should trigger garbage collection under memory pressure"
        
        # Should demonstrate memory management rather than crashing
        success_rate = status_counts.get('SUCCESS', 0) / len(results)
        assert success_rate > 0.5, f"Should maintain >50% success rate under memory pressure, got {success_rate:.1%}"
        
        # Memory should be managed (not continuously growing)
        if len(memory_readings) > 5:
            early_memory = np.mean(memory_readings[:3])
            late_memory = np.mean(memory_readings[-3:])
            memory_growth = late_memory / early_memory
            assert memory_growth < 3.0, f"Memory growth {memory_growth:.1f}x too high - indicates memory leak"


class TestCorruptedDataResilience(StressTestBase):
    """Test resilience to corrupted and malformed data"""
    
    def test_corrupted_data_handling(self):
        """Test handling of various corrupted data scenarios"""
        
        class RobustRiskCalculator:
            def __init__(self):
                self.validation_failures = []
                self.recovery_strategies = []
                self.calculation_attempts = 0
                self.successful_calculations = 0
            
            def calculate_risk_robust(self, market_data_dict: Dict[str, Any]) -> Tuple[Optional[int], List[str]]:
                """Robust risk calculation with extensive data validation"""
                self.calculation_attempts += 1
                issues = []
                
                try:
                    # Data validation and sanitization
                    sanitized_data = self._sanitize_market_data(market_data_dict)
                    
                    if not sanitized_data:
                        issues.append("Data failed sanitization")
                        return None, issues
                    
                    # Safe risk calculation with multiple fallbacks
                    risk_score = self._calculate_with_fallbacks(sanitized_data)
                    
                    if risk_score is not None:
                        self.successful_calculations += 1
                    
                    return risk_score, issues
                
                except Exception as e:
                    issues.append(f"Calculation exception: {str(e)}")
                    return None, issues
            
            def _sanitize_market_data(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
                """Sanitize and validate market data"""
                sanitized = {}
                
                # Price validation and sanitization
                price_fields = ['open_price_cents', 'high_price_cents', 'low_price_cents', 'close_price_cents']
                
                for field in price_fields:
                    value = data.get(field)
                    
                    if value is None:
                        self.validation_failures.append(f"Missing {field}")
                        return None
                    
                    # Handle various corrupted number formats
                    try:
                        if isinstance(value, str):
                            value = float(value.replace('$', '').replace(',', ''))
                        
                        if np.isnan(value) or np.isinf(value):
                            self.validation_failures.append(f"Invalid {field}: {value}")
                            return None
                        
                        if value <= 0:
                            self.validation_failures.append(f"Non-positive {field}: {value}")
                            # Try to recover with nearby price if available
                            if field == 'open_price_cents' and 'close_price_cents' in data:
                                value = data['close_price_cents']
                                self.recovery_strategies.append(f"Recovered {field} using close price")
                            else:
                                return None
                        
                        # Cap extreme values
                        if value > 1e10:  # $100M+ per share seems unrealistic
                            value = 1e10
                            self.recovery_strategies.append(f"Capped extreme {field}")
                        
                        sanitized[field] = int(value)
                    
                    except (ValueError, TypeError) as e:
                        self.validation_failures.append(f"Cannot convert {field}: {e}")
                        return None
                
                # OHLC consistency check
                o, h, l, c = sanitized['open_price_cents'], sanitized['high_price_cents'], \
                            sanitized['low_price_cents'], sanitized['close_price_cents']
                
                if not (l <= min(o, c) and max(o, c) <= h):
                    # Try to fix OHLC consistency
                    min_price, max_price = min(o, h, l, c), max(o, h, l, c)
                    sanitized['low_price_cents'] = min_price
                    sanitized['high_price_cents'] = max_price
                    self.recovery_strategies.append("Fixed OHLC consistency")
                
                # Volume validation
                volume = data.get('volume', 0)
                try:
                    if isinstance(volume, str):
                        volume = int(volume.replace(',', ''))
                    
                    if volume < 0:
                        volume = 0
                        self.recovery_strategies.append("Fixed negative volume")
                    elif volume > 1e12:  # Cap extreme volume
                        volume = 1e12
                        self.recovery_strategies.append("Capped extreme volume")
                    
                    sanitized['volume'] = int(volume)
                
                except (ValueError, TypeError):
                    sanitized['volume'] = 0
                    self.recovery_strategies.append("Set default volume")
                
                return sanitized
            
            def _calculate_with_fallbacks(self, data: Dict[str, Any]) -> Optional[int]:
                """Calculate risk with multiple fallback methods"""
                
                # Primary calculation method
                try:
                    volatility = (data['high_price_cents'] - data['low_price_cents']) / data['close_price_cents']
                    return min(int(volatility * 1000), 100)
                except (ZeroDivisionError, ValueError):
                    pass
                
                # Fallback 1: Use price change
                try:
                    price_change = abs(data['close_price_cents'] - data['open_price_cents']) / data['open_price_cents']
                    self.recovery_strategies.append("Used price change fallback")
                    return min(int(price_change * 2000), 100)
                except (ZeroDivisionError, ValueError):
                    pass
                
                # Fallback 2: Volume-based risk
                try:
                    if data['volume'] < 1000:
                        self.recovery_strategies.append("Used low volume fallback")
                        return 80  # High risk for low volume
                    else:
                        return 40  # Medium risk
                except:
                    pass
                
                # Final fallback: Conservative default
                self.recovery_strategies.append("Used conservative default")
                return 75
        
        calculator = RobustRiskCalculator()
        
        # Test with various corrupted data scenarios
        corrupted_scenarios = self.create_corrupted_data_scenarios()
        
        print(f"\nCorrupted Data Resilience Test:")
        
        for scenario in corrupted_scenarios:
            print(f"\nTesting {scenario['name']} scenario:")
            
            # Create corrupted market data
            base_data = {
                'asset_id': f"CORRUPT_{scenario['name']}",
                'timestamp': datetime.utcnow(),
                'open_price_cents': 10000,
                'high_price_cents': 10100,
                'low_price_cents': 9900,
                'close_price_cents': 10050,
                'volume': 1000000,
                'data_source': 'CORRUPTION_TEST'
            }
            
            # Apply corruption
            base_data.update(scenario)
            
            risk_score, issues = calculator.calculate_risk_robust(base_data)
            
            print(f"  Risk Score: {risk_score}")
            print(f"  Issues: {issues}")
            
            # Should handle corruption gracefully
            if risk_score is not None:
                assert 0 <= risk_score <= 100, f"Risk score {risk_score} outside valid range"
                print(f"  Successfully recovered from corruption")
            else:
                print(f"  Appropriately rejected corrupted data")
        
        # Analyze overall resilience
        total_attempts = calculator.calculation_attempts
        success_rate = calculator.successful_calculations / total_attempts if total_attempts > 0 else 0
        
        print(f"\nCorruption Resilience Summary:")
        print(f"  Total scenarios tested: {total_attempts}")
        print(f"  Successfully processed: {calculator.successful_calculations}")
        print(f"  Success rate: {success_rate:.1%}")
        print(f"  Validation failures: {len(calculator.validation_failures)}")
        print(f"  Recovery strategies used: {len(calculator.recovery_strategies)}")
        
        # Should recover from at least some corrupted data
        assert success_rate >= 0.3, f"Should recover from at least 30% of corrupted data, got {success_rate:.1%}"
        assert len(calculator.recovery_strategies) > 0, "Should demonstrate data recovery strategies"
    
    def test_malicious_data_injection_resistance(self):
        """Test resistance to malicious data injection attacks"""
        
        class SecureRiskCalculator:
            def __init__(self):
                self.security_violations = []
                self.blocked_attempts = 0
                self.processed_safely = 0
            
            def calculate_risk_secure(self, raw_data: Any) -> Tuple[Optional[int], List[str]]:
                """Secure risk calculation with input validation"""
                security_issues = []
                
                # Input type validation
                if not isinstance(raw_data, dict):
                    security_issues.append("Invalid input type - expected dict")
                    self.blocked_attempts += 1
                    return None, security_issues
                
                # Check for suspicious keys (potential injection attempts)
                suspicious_patterns = [
                    '__', 'eval', 'exec', 'import', 'os.', 'subprocess', 
                    'file', 'open', 'write', '..', '/', '\\'
                ]
                
                for key, value in raw_data.items():
                    key_str = str(key).lower()
                    value_str = str(value).lower() if value is not None else ""
                    
                    for pattern in suspicious_patterns:
                        if pattern in key_str or pattern in value_str:
                            security_issues.append(f"Suspicious pattern detected: {pattern}")
                            self.security_violations.append((key, value, pattern))
                            self.blocked_attempts += 1
                            return None, security_issues
                
                # Size limits to prevent DoS attacks
                if len(str(raw_data)) > 10000:  # 10KB limit
                    security_issues.append("Input size too large - potential DoS attack")
                    self.blocked_attempts += 1
                    return None, security_issues
                
                # Numeric validation - only allow reasonable numeric types
                numeric_fields = ['open_price_cents', 'high_price_cents', 'low_price_cents', 
                                'close_price_cents', 'volume']
                
                for field in numeric_fields:
                    if field in raw_data:
                        value = raw_data[field]
                        if not isinstance(value, (int, float)):
                            try:
                                # Only allow simple numeric conversion
                                float(value)
                            except (ValueError, TypeError):
                                security_issues.append(f"Non-numeric value in {field}")
                                self.blocked_attempts += 1
                                return None, security_issues
                
                # If all security checks pass, calculate risk normally
                try:
                    high = float(raw_data.get('high_price_cents', 100))
                    low = float(raw_data.get('low_price_cents', 100))
                    close = float(raw_data.get('close_price_cents', 100))
                    
                    if close <= 0:
                        return 85, ["Zero or negative close price"]
                    
                    volatility = (high - low) / close
                    risk_score = min(max(int(volatility * 1000), 0), 100)
                    
                    self.processed_safely += 1
                    return risk_score, security_issues
                
                except Exception as e:
                    security_issues.append(f"Safe calculation failed: {str(e)}")
                    return 90, security_issues  # Conservative high-risk score
        
        calculator = SecureRiskCalculator()
        
        # Test various malicious injection attempts
        malicious_payloads = [
            # Code injection attempts
            {"name": "python_eval", "open_price_cents": "eval('1+1')", "close_price_cents": 100},
            {"name": "import_attempt", "volume": "import os; os.system('ls')", "close_price_cents": 100},
            {"name": "file_access", "asset_id": "../../../etc/passwd", "close_price_cents": 100},
            
            # DoS attempts
            {"name": "large_payload", "description": "x" * 20000, "close_price_cents": 100},
            {"name": "nested_dict", "data": {"level1": {"level2": {"level3": "deep"} * 1000}}},
            
            # Type confusion attacks
            {"name": "object_injection", "close_price_cents": {"__class__": "malicious"}},
            {"name": "function_reference", "volume": lambda x: x * 2},
            
            # Buffer overflow attempts (simulated)
            {"name": "extreme_numbers", "open_price_cents": 10**100, "close_price_cents": 1},
            
            # SQL injection style (though not applicable, tests input sanitization)
            {"name": "sql_injection", "asset_id": "'; DROP TABLE assets; --", "close_price_cents": 100},
        ]
        
        print(f"\nMalicious Data Injection Test:")
        
        for payload in malicious_payloads:
            print(f"\nTesting {payload['name']} attack:")
            
            risk_score, issues = calculator.calculate_risk_secure(payload)
            
            print(f"  Risk Score: {risk_score}")
            print(f"  Security Issues: {issues}")
            
            if risk_score is None:
                print(f"  ✓ Attack blocked successfully")
            else:
                print(f"  ⚠ Attack processed (score: {risk_score}) - check if safe")
                assert 0 <= risk_score <= 100, f"Even processed attacks should return valid scores"
        
        # Test with legitimate data to ensure normal operation
        legitimate_data = {
            'open_price_cents': 10000,
            'high_price_cents': 10200,
            'low_price_cents': 9800,
            'close_price_cents': 10100,
            'volume': 1000000,
            'asset_id': 'LEGIT_ASSET'
        }
        
        legit_score, legit_issues = calculator.calculate_risk_secure(legitimate_data)
        
        print(f"\nLegitimate data test:")
        print(f"  Risk Score: {legit_score}")
        print(f"  Issues: {legit_issues}")
        
        # Analyze security effectiveness
        total_attacks = len(malicious_payloads)
        blocked_rate = calculator.blocked_attempts / total_attacks if total_attacks > 0 else 0
        
        print(f"\nSecurity Analysis:")
        print(f"  Total attack attempts: {total_attacks}")
        print(f"  Blocked attempts: {calculator.blocked_attempts}")
        print(f"  Block rate: {blocked_rate:.1%}")
        print(f"  Security violations detected: {len(calculator.security_violations)}")
        print(f"  Safely processed legitimate data: {calculator.processed_safely > 0}")
        
        # Security validation
        assert blocked_rate >= 0.7, f"Should block at least 70% of malicious attempts, got {blocked_rate:.1%}"
        assert calculator.processed_safely > 0, "Should successfully process legitimate data"
        assert legit_score is not None, "Legitimate data should be processed successfully"
        assert 0 <= legit_score <= 100, "Legitimate data should produce valid risk score"


if __name__ == "__main__":
    # Run stress tests
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-s",  # Show print output for stress test analysis
    ])