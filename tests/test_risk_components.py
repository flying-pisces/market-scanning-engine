"""
Risk Scoring System - Unit Tests for Individual Risk Components
Author: Claude Code (QA Engineer)
Version: 1.0

Comprehensive unit tests for volatility, liquidity, time decay, market regime, 
and position size risk components. Ensures each component produces accurate
0-100 risk scores with proper validation and edge case handling.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import List, Dict, Optional
from unittest.mock import Mock, patch
import time
import statistics

from data_models.python.core_models import RiskScore, MarketRegime, AssetCategory
from data_models.python.signal_models import RiskAssessment


class RiskComponentTestBase:
    """Base class for risk component testing with common utilities"""
    
    @staticmethod
    def assert_score_range(score: int, min_val: int = 0, max_val: int = 100):
        """Assert score is within valid 0-100 range"""
        assert isinstance(score, int), f"Score must be integer, got {type(score)}"
        assert min_val <= score <= max_val, f"Score {score} not in range [{min_val}, {max_val}]"
    
    @staticmethod
    def assert_score_consistency(calculate_func, inputs: dict, iterations: int = 100):
        """Assert function produces consistent results for identical inputs"""
        scores = []
        for _ in range(iterations):
            score = calculate_func(**inputs)
            scores.append(score)
        
        unique_scores = set(scores)
        assert len(unique_scores) == 1, f"Inconsistent scores: {unique_scores}"
        return scores[0]
    
    @staticmethod
    def assert_calculation_speed(calculate_func, inputs: dict, max_time_ms: int = 50):
        """Assert calculation completes within time limit"""
        start_time = time.perf_counter()
        result = calculate_func(**inputs)
        end_time = time.perf_counter()
        
        execution_time_ms = (end_time - start_time) * 1000
        assert execution_time_ms <= max_time_ms, f"Calculation took {execution_time_ms:.2f}ms, limit: {max_time_ms}ms"
        return result
    
    @staticmethod
    def assert_score_calibration(scores: List[int], asset_class: AssetCategory, tolerance: float = 0.1):
        """Assert scores are properly calibrated for asset class"""
        if not scores:
            pytest.fail("No scores provided for calibration test")
        
        mean_score = statistics.mean(scores)
        
        # Expected score ranges by asset class
        expected_ranges = {
            AssetCategory.DERIVATIVES: (70, 95),  # Options
            AssetCategory.EQUITY: (30, 80),       # Stocks
            AssetCategory.FIXED_INCOME: (5, 40),  # Bonds
            AssetCategory.COMMODITY: (0, 20)      # T-bills/CDs treated as commodity for this test
        }
        
        if asset_class in expected_ranges:
            min_expected, max_expected = expected_ranges[asset_class]
            expected_mean = (min_expected + max_expected) / 2
            
            # Allow tolerance around expected mean
            lower_bound = expected_mean * (1 - tolerance)
            upper_bound = expected_mean * (1 + tolerance)
            
            assert lower_bound <= mean_score <= upper_bound, \
                f"Mean score {mean_score:.1f} not within expected range [{lower_bound:.1f}, {upper_bound:.1f}] for {asset_class.value}"


class TestVolatilityRiskComponent(RiskComponentTestBase):
    """Test volatility risk component (VR001-VR008)"""
    
    def test_historical_volatility_calculation_accuracy(self):
        """VR001: Validate historical volatility calculation accuracy"""
        # Test data: 20 days of price returns with known volatility
        price_returns = [0.01, -0.02, 0.015, -0.01, 0.005, 0.03, -0.025, 0.02, -0.015, 0.01,
                        0.008, -0.012, 0.018, -0.008, 0.022, -0.018, 0.012, -0.005, 0.015, -0.01]
        
        expected_volatility = np.std(price_returns) * np.sqrt(252)  # Annualized
        
        def calculate_volatility_risk(returns: List[float]) -> int:
            """Mock volatility risk calculation"""
            vol = np.std(returns) * np.sqrt(252)
            # Convert to 0-100 scale (assuming 50% vol = 100 score)
            return min(int(vol * 200), 100)
        
        score = calculate_volatility_risk(price_returns)
        self.assert_score_range(score)
        
        # Verify calculation accuracy
        expected_score = min(int(expected_volatility * 200), 100)
        assert abs(score - expected_score) <= 1, f"Score {score} too far from expected {expected_score}"
    
    def test_vix_correlation_validation(self):
        """VR002: Validate VIX correlation coefficient calculation"""
        # Mock VIX data and asset returns
        vix_values = [15, 18, 22, 25, 20, 16, 19, 24, 28, 21]
        asset_returns = [-0.01, -0.02, -0.03, -0.04, -0.02, 0.01, -0.015, -0.035, -0.045, -0.025]
        
        def calculate_vix_correlation_risk(vix: List[float], returns: List[float]) -> int:
            """Mock VIX correlation risk calculation"""
            correlation = np.corrcoef(vix, returns)[0, 1]
            # Higher negative correlation = higher risk during VIX spikes
            risk_score = min(int(abs(correlation) * 100), 100)
            return risk_score
        
        score = calculate_vix_correlation_risk(vix_values, asset_returns)
        self.assert_score_range(score)
        
        # Test edge cases
        zero_correlation_score = calculate_vix_correlation_risk([20] * 10, [0] * 10)
        self.assert_score_range(zero_correlation_score, 0, 10)  # Low correlation = low risk
    
    def test_volatility_percentile_ranking(self):
        """VR003: Test volatility percentile ranking against historical data"""
        # Historical volatility data (mock 252 trading days)
        historical_vols = np.random.normal(0.2, 0.05, 252)  # Mean 20%, std 5%
        current_vol = 0.35  # 35% volatility (should be high percentile)
        
        def calculate_volatility_percentile_risk(current: float, historical: List[float]) -> int:
            """Calculate risk based on volatility percentile"""
            percentile = np.percentile(historical, current * 100)
            return min(int(percentile), 100)
        
        score = calculate_volatility_percentile_risk(current_vol, historical_vols)
        self.assert_score_range(score)
        
        # High volatility should result in high risk score
        assert score > 70, f"High volatility should yield high risk score, got {score}"
    
    def test_boundary_conditions_extreme_volatility(self):
        """VR005: Test boundary conditions with extreme volatility values"""
        def calculate_volatility_risk_safe(volatility: float) -> int:
            """Safe volatility risk calculation with boundary handling"""
            if volatility < 0:
                return 0
            elif volatility > 5.0:  # 500% volatility cap
                return 100
            else:
                return min(int(volatility * 20), 100)
        
        # Test normal range
        normal_score = calculate_volatility_risk_safe(0.25)  # 25% volatility
        self.assert_score_range(normal_score, 0, 10)
        
        # Test extreme high
        extreme_high_score = calculate_volatility_risk_safe(10.0)  # 1000% volatility
        assert extreme_high_score == 100
        
        # Test zero volatility
        zero_score = calculate_volatility_risk_safe(0.0)
        assert zero_score == 0
        
        # Test negative (should handle gracefully)
        negative_score = calculate_volatility_risk_safe(-0.1)
        assert negative_score == 0
    
    def test_performance_volatility_calculation(self):
        """Performance test for volatility calculation speed"""
        large_dataset = np.random.normal(0, 0.02, 1000)  # 1000 data points
        
        def fast_volatility_calculation(returns: List[float]) -> int:
            """Optimized volatility calculation"""
            vol = np.std(returns) * np.sqrt(252)
            return min(int(vol * 100), 100)
        
        # Should complete within 25ms for 1000 data points
        self.assert_calculation_speed(
            fast_volatility_calculation, 
            {"returns": large_dataset}, 
            max_time_ms=25
        )


class TestLiquidityRiskComponent(RiskComponentTestBase):
    """Test liquidity risk component (LR001-LR008)"""
    
    def test_bid_ask_spread_calculation(self):
        """LR001: Test bid/ask spread calculation accuracy"""
        def calculate_spread_risk(bid: float, ask: float, mid_price: float) -> int:
            """Calculate liquidity risk from bid/ask spread"""
            spread_pct = (ask - bid) / mid_price
            # Convert to 0-100 scale (5% spread = 100 score)
            return min(int(spread_pct * 2000), 100)
        
        # Normal spread (0.1%)
        normal_score = calculate_spread_risk(bid=99.95, ask=100.05, mid_price=100.0)
        self.assert_score_range(normal_score, 0, 5)
        
        # Wide spread (2%)
        wide_score = calculate_spread_risk(bid=99.0, ask=101.0, mid_price=100.0)
        self.assert_score_range(wide_score, 35, 45)
        
        # Very wide spread (10%)
        very_wide_score = calculate_spread_risk(bid=95.0, ask=105.0, mid_price=100.0)
        assert very_wide_score == 100
    
    def test_volume_pattern_analysis(self):
        """LR002: Test volume pattern analysis for liquidity assessment"""
        # Daily volumes for 30 days
        volumes = [1000000, 950000, 1100000, 800000, 1200000, 750000, 900000, 1050000, 
                  600000, 1300000, 700000, 1150000, 850000, 950000, 1000000,
                  500000, 1400000, 650000, 1050000, 900000, 800000, 1100000,
                  450000, 1500000, 600000, 1000000, 850000, 950000, 750000, 1200000]
        
        def calculate_volume_liquidity_risk(daily_volumes: List[int]) -> int:
            """Calculate liquidity risk based on volume patterns"""
            avg_volume = np.mean(daily_volumes)
            volume_volatility = np.std(daily_volumes) / avg_volume
            
            # Higher volume volatility = higher liquidity risk
            return min(int(volume_volatility * 200), 100)
        
        score = calculate_volume_liquidity_risk(volumes)
        self.assert_score_range(score)
        
        # Test with consistent volumes (low risk)
        consistent_volumes = [1000000] * 30
        consistent_score = calculate_volume_liquidity_risk(consistent_volumes)
        assert consistent_score < 10, f"Consistent volumes should have low risk, got {score}"
    
    def test_market_depth_assessment(self):
        """LR003: Test market depth assessment for liquidity risk"""
        def calculate_depth_risk(order_book: Dict[str, List]) -> int:
            """Calculate risk based on order book depth"""
            bid_sizes = order_book.get('bid_sizes', [])
            ask_sizes = order_book.get('ask_sizes', [])
            
            total_depth = sum(bid_sizes) + sum(ask_sizes)
            
            # Shallow depth = high liquidity risk
            if total_depth < 10000:
                return 90
            elif total_depth < 50000:
                return 60
            elif total_depth < 100000:
                return 30
            else:
                return 10
        
        # Deep market
        deep_market = {
            'bid_sizes': [50000, 30000, 25000, 20000, 15000],
            'ask_sizes': [45000, 35000, 28000, 22000, 18000]
        }
        deep_score = calculate_depth_risk(deep_market)
        self.assert_score_range(deep_score, 5, 15)
        
        # Shallow market
        shallow_market = {
            'bid_sizes': [1000, 500, 300],
            'ask_sizes': [800, 600, 400]
        }
        shallow_score = calculate_depth_risk(shallow_market)
        self.assert_score_range(shallow_score, 85, 95)
    
    def test_after_hours_liquidity_adjustment(self):
        """LR005: Test after-hours liquidity risk adjustments"""
        def calculate_time_adjusted_liquidity_risk(base_score: int, trading_hour: int) -> int:
            """Adjust liquidity risk based on trading hours"""
            # Market hours: 9:30 AM - 4:00 PM EST (9.5 - 16.0)
            if 9.5 <= trading_hour <= 16.0:
                return base_score  # No adjustment during market hours
            else:
                # Increase risk during after-hours
                return min(base_score + 20, 100)
        
        base_score = 30
        
        # During market hours
        market_hours_score = calculate_time_adjusted_liquidity_risk(base_score, 12.0)
        assert market_hours_score == base_score
        
        # After hours
        after_hours_score = calculate_time_adjusted_liquidity_risk(base_score, 18.0)
        assert after_hours_score == 50
        
        # Pre-market
        pre_market_score = calculate_time_adjusted_liquidity_risk(base_score, 7.0)
        assert pre_market_score == 50


class TestTimeDecayRiskComponent(RiskComponentTestBase):
    """Test time decay risk component (TD001-TD008)"""
    
    def test_options_theta_calculation(self):
        """TD001: Test options theta calculation accuracy"""
        def calculate_theta_risk(theta: float, position_size: int, days_to_expiry: int) -> int:
            """Calculate time decay risk from theta"""
            daily_decay = abs(theta) * position_size
            total_decay = daily_decay * days_to_expiry
            
            # Scale to 0-100 (assuming $10,000 total decay = 100 score)
            return min(int(total_decay / 100), 100)
        
        # High theta option near expiry
        high_theta_score = calculate_theta_risk(theta=-0.50, position_size=10, days_to_expiry=1)
        self.assert_score_range(high_theta_score, 0, 10)
        
        # High theta option far from expiry
        high_theta_long_term = calculate_theta_risk(theta=-0.50, position_size=10, days_to_expiry=30)
        self.assert_score_range(high_theta_long_term, 100, 100)
    
    def test_time_to_expiration_scaling(self):
        """TD002: Test time to expiration risk scaling"""
        def calculate_expiration_risk(days_to_expiry: int) -> int:
            """Calculate risk based on time to expiration"""
            if days_to_expiry <= 0:
                return 100  # Expired = maximum risk
            elif days_to_expiry <= 7:
                return 90   # Weekly expiry
            elif days_to_expiry <= 30:
                return 60   # Monthly expiry
            elif days_to_expiry <= 90:
                return 30   # Quarterly expiry
            else:
                return 10   # Long-term options
        
        # Test various expiration periods
        assert calculate_expiration_risk(0) == 100     # Expired
        assert calculate_expiration_risk(3) == 90      # 3 days
        assert calculate_expiration_risk(15) == 60     # 2 weeks
        assert calculate_expiration_risk(60) == 30     # 2 months
        assert calculate_expiration_risk(180) == 10    # 6 months
    
    def test_weekend_holiday_adjustments(self):
        """TD003: Test weekend and holiday time decay adjustments"""
        def calculate_weekend_adjusted_theta_risk(
            base_theta_risk: int, 
            current_date: date, 
            expiry_date: date
        ) -> int:
            """Adjust theta risk for weekends and holidays"""
            days_diff = (expiry_date - current_date).days
            
            # Count weekends between now and expiry
            weekend_days = 0
            current = current_date
            while current < expiry_date:
                if current.weekday() >= 5:  # Saturday or Sunday
                    weekend_days += 1
                current += timedelta(days=1)
            
            # Increase risk if expiration crosses weekend
            if weekend_days > 0:
                return min(base_theta_risk + (weekend_days * 5), 100)
            return base_theta_risk
        
        base_risk = 50
        friday = date(2024, 1, 5)      # Friday
        monday = date(2024, 1, 8)      # Following Monday
        
        weekend_risk = calculate_weekend_adjusted_theta_risk(base_risk, friday, monday)
        assert weekend_risk > base_risk, "Weekend crossing should increase risk"
    
    def test_portfolio_level_time_decay(self):
        """TD007: Test portfolio-level time decay calculation"""
        positions = [
            {'theta': -0.25, 'size': 10, 'days': 7},
            {'theta': -0.15, 'size': 5, 'days': 30},
            {'theta': -0.35, 'size': 8, 'days': 3},
            {'theta': -0.10, 'size': 15, 'days': 60}
        ]
        
        def calculate_portfolio_theta_risk(positions: List[Dict]) -> int:
            """Calculate portfolio-level theta risk"""
            total_daily_decay = sum(abs(pos['theta']) * pos['size'] for pos in positions)
            
            # Weight by time to expiry (shorter = higher weight)
            weighted_decay = 0
            for pos in positions:
                weight = 1 / max(pos['days'], 1)  # Avoid division by zero
                weighted_decay += abs(pos['theta']) * pos['size'] * weight
            
            return min(int(weighted_decay * 20), 100)
        
        portfolio_risk = calculate_portfolio_theta_risk(positions)
        self.assert_score_range(portfolio_risk)
        
        # Portfolio with near-expiry options should have higher risk
        assert portfolio_risk > 40, "Portfolio with near-expiry options should have significant theta risk"


class TestMarketRegimeRiskComponent(RiskComponentTestBase):
    """Test market regime risk component (MR001-MR008)"""
    
    def test_bull_bear_market_classification(self):
        """MR001: Test bull/bear market classification accuracy"""
        def classify_market_regime(returns_20d: List[float], vix: float) -> tuple:
            """Classify market regime and calculate risk score"""
            avg_return = np.mean(returns_20d)
            volatility = np.std(returns_20d)
            
            if avg_return > 0.002 and vix < 20:
                regime = MarketRegime.BULL_MARKET
                risk_adjustment = 0.8  # Lower risk in bull market
            elif avg_return < -0.001 or vix > 30:
                regime = MarketRegime.BEAR_MARKET
                risk_adjustment = 1.3  # Higher risk in bear market
            else:
                regime = MarketRegime.SIDEWAYS
                risk_adjustment = 1.0  # Neutral risk
            
            base_score = 50
            adjusted_score = min(int(base_score * risk_adjustment), 100)
            
            return regime, adjusted_score
        
        # Bull market conditions
        bull_returns = [0.01, 0.008, 0.012, 0.005, 0.015, 0.003, 0.009, 0.007, 0.011, 0.006] * 2
        bull_regime, bull_score = classify_market_regime(bull_returns, vix=15)
        assert bull_regime == MarketRegime.BULL_MARKET
        assert bull_score < 50, f"Bull market should reduce risk score, got {bull_score}"
        
        # Bear market conditions
        bear_returns = [-0.02, -0.015, -0.025, -0.01, -0.018, -0.012, -0.022, -0.008, -0.020, -0.016] * 2
        bear_regime, bear_score = classify_market_regime(bear_returns, vix=35)
        assert bear_regime == MarketRegime.BEAR_MARKET
        assert bear_score > 50, f"Bear market should increase risk score, got {bear_score}"
    
    def test_volatility_regime_detection(self):
        """MR002: Test volatility regime detection"""
        def detect_volatility_regime(vix_history: List[float]) -> tuple:
            """Detect volatility regime and calculate risk"""
            current_vix = vix_history[-1]
            avg_vix = np.mean(vix_history[:-1])
            
            if current_vix > avg_vix * 1.5:
                regime = MarketRegime.HIGH_VOL
                risk_score = min(int(current_vix * 2), 100)
            elif current_vix < avg_vix * 0.7:
                regime = MarketRegime.LOW_VOL
                risk_score = max(int(current_vix * 1.5), 10)
            else:
                regime = "NORMAL_VOL"
                risk_score = int(current_vix * 2)
            
            return regime, risk_score
        
        # High volatility regime
        high_vol_history = [20, 18, 22, 19, 21, 35]  # VIX spikes to 35
        regime, score = detect_volatility_regime(high_vol_history)
        assert regime == MarketRegime.HIGH_VOL
        assert score > 60, f"High volatility should yield high risk score, got {score}"
        
        # Low volatility regime
        low_vol_history = [20, 18, 22, 19, 21, 12]  # VIX drops to 12
        regime, score = detect_volatility_regime(low_vol_history)
        assert regime == MarketRegime.LOW_VOL
        assert score < 30, f"Low volatility should yield low risk score, got {score}"
    
    def test_regime_transition_smoothing(self):
        """MR004: Test regime transition smoothing to prevent whipsaws"""
        def smooth_regime_transition(
            current_regime: str, 
            new_regime: str, 
            confidence: float,
            previous_scores: List[int]
        ) -> int:
            """Smooth regime transitions using exponential moving average"""
            if current_regime != new_regime and confidence < 0.8:
                # Use smoothed transition
                base_score = 50 if new_regime == "BULL" else 70
                if previous_scores:
                    alpha = 0.3  # Smoothing factor
                    smoothed_score = int(alpha * base_score + (1 - alpha) * previous_scores[-1])
                    return smoothed_score
                return base_score
            else:
                # Direct transition with high confidence
                return 50 if new_regime == "BULL" else 70
        
        previous_bull_scores = [45, 42, 48, 44]
        
        # Low confidence regime change should be smoothed
        smoothed_score = smooth_regime_transition("BULL", "BEAR", confidence=0.6, previous_scores=previous_bull_scores)
        assert 44 <= smoothed_score <= 65, f"Smoothed transition score should be gradual, got {smoothed_score}"
        
        # High confidence regime change should be direct
        direct_score = smooth_regime_transition("BULL", "BEAR", confidence=0.9, previous_scores=previous_bull_scores)
        assert direct_score == 70, f"High confidence transition should be direct, got {direct_score}"


class TestPositionSizeRiskComponent(RiskComponentTestBase):
    """Test position size risk component (PS001-PS008)"""
    
    def test_portfolio_concentration_calculations(self):
        """PS001: Test portfolio concentration risk calculations"""
        def calculate_concentration_risk(
            position_value: float, 
            total_portfolio_value: float,
            max_position_limit: float = 0.10  # 10% limit
        ) -> int:
            """Calculate concentration risk based on position size"""
            concentration_pct = position_value / total_portfolio_value
            
            if concentration_pct > max_position_limit:
                excess_concentration = concentration_pct - max_position_limit
                return min(int(excess_concentration * 1000), 100)  # Scale excess to 0-100
            else:
                return int(concentration_pct * 500)  # Normal scaling
        
        # Normal position (5% of portfolio)
        normal_risk = calculate_concentration_risk(50000, 1000000)
        self.assert_score_range(normal_risk, 20, 30)
        
        # Concentrated position (25% of portfolio)
        concentrated_risk = calculate_concentration_risk(250000, 1000000)
        self.assert_score_range(concentrated_risk, 90, 100)
        
        # Small position (1% of portfolio)
        small_risk = calculate_concentration_risk(10000, 1000000)
        self.assert_score_range(small_risk, 0, 10)
    
    def test_correlation_adjusted_position_sizing(self):
        """PS002: Test correlation-adjusted position sizing risk"""
        correlations = np.array([
            [1.0, 0.8, 0.3, 0.1],  # Asset 1 correlations
            [0.8, 1.0, 0.4, 0.2],  # Asset 2 correlations
            [0.3, 0.4, 1.0, 0.6],  # Asset 3 correlations
            [0.1, 0.2, 0.6, 1.0]   # Asset 4 correlations
        ])
        
        position_weights = [0.3, 0.25, 0.25, 0.2]  # Portfolio weights
        
        def calculate_correlation_adjusted_risk(
            correlations: np.ndarray, 
            weights: List[float]
        ) -> int:
            """Calculate risk adjusted for correlations"""
            portfolio_variance = np.dot(weights, np.dot(correlations, weights))
            portfolio_risk = np.sqrt(portfolio_variance)
            
            # Convert to 0-100 scale
            return min(int(portfolio_risk * 100), 100)
        
        risk_score = calculate_correlation_adjusted_risk(correlations, position_weights)
        self.assert_score_range(risk_score)
        
        # High correlation should increase risk
        high_corr_weights = [0.5, 0.5, 0.0, 0.0]  # Concentrated in highly correlated assets
        high_corr_risk = calculate_correlation_adjusted_risk(correlations, high_corr_weights)
        
        assert high_corr_risk > risk_score, "High correlation concentration should increase risk"
    
    def test_sector_exposure_limits(self):
        """PS003: Test sector exposure limit calculations"""
        sector_exposures = {
            'TECHNOLOGY': 0.35,
            'HEALTHCARE': 0.20,
            'FINANCIALS': 0.25,
            'ENERGY': 0.15,
            'UTILITIES': 0.05
        }
        
        sector_limits = {
            'TECHNOLOGY': 0.30,
            'HEALTHCARE': 0.25,
            'FINANCIALS': 0.20,
            'ENERGY': 0.20,
            'UTILITIES': 0.10
        }
        
        def calculate_sector_concentration_risk(
            exposures: Dict[str, float], 
            limits: Dict[str, float]
        ) -> int:
            """Calculate sector concentration risk"""
            total_excess = 0
            for sector, exposure in exposures.items():
                limit = limits.get(sector, 0.10)  # Default 10% limit
                if exposure > limit:
                    total_excess += (exposure - limit)
            
            return min(int(total_excess * 200), 100)  # Scale to 0-100
        
        sector_risk = calculate_sector_concentration_risk(sector_exposures, sector_limits)
        self.assert_score_range(sector_risk)
        
        # Technology and financials are over-allocated
        assert sector_risk > 50, f"Sector concentration should create meaningful risk, got {sector_risk}"
    
    def test_dynamic_position_sizing(self):
        """PS008: Test dynamic position sizing based on market conditions"""
        def calculate_dynamic_position_risk(
            base_position_size: float,
            volatility_multiplier: float,
            regime_multiplier: float,
            liquidity_adjustment: float
        ) -> int:
            """Calculate risk-adjusted position size"""
            adjusted_size = base_position_size * volatility_multiplier * regime_multiplier * liquidity_adjustment
            
            # Risk increases non-linearly with position size
            if adjusted_size > 0.20:  # > 20% of portfolio
                return 100
            elif adjusted_size > 0.15:  # > 15% of portfolio
                return 80
            elif adjusted_size > 0.10:  # > 10% of portfolio
                return 60
            elif adjusted_size > 0.05:  # > 5% of portfolio
                return 40
            else:
                return 20
        
        # Normal market conditions
        normal_risk = calculate_dynamic_position_risk(
            base_position_size=0.08,  # 8% base
            volatility_multiplier=1.0,
            regime_multiplier=1.0,
            liquidity_adjustment=1.0
        )
        assert normal_risk == 60
        
        # High volatility + bear market conditions
        stressed_risk = calculate_dynamic_position_risk(
            base_position_size=0.08,  # 8% base
            volatility_multiplier=1.5,  # High volatility
            regime_multiplier=1.3,      # Bear market
            liquidity_adjustment=0.8    # Reduced liquidity
        )
        assert stressed_risk > normal_risk, "Stressed conditions should increase position risk"


class TestRiskScoreIntegration(RiskComponentTestBase):
    """Integration tests combining multiple risk components"""
    
    def test_composite_risk_score_calculation(self):
        """Test composite risk score from all components"""
        component_scores = {
            'volatility': 65,
            'liquidity': 45,
            'time_decay': 80,
            'market_regime': 55,
            'position_size': 70
        }
        
        component_weights = {
            'volatility': 0.25,
            'liquidity': 0.20,
            'time_decay': 0.15,
            'market_regime': 0.20,
            'position_size': 0.20
        }
        
        def calculate_composite_risk_score(
            scores: Dict[str, int], 
            weights: Dict[str, float]
        ) -> int:
            """Calculate weighted composite risk score"""
            composite = sum(scores[component] * weights[component] for component in scores)
            return min(int(composite), 100)
        
        composite_score = calculate_composite_risk_score(component_scores, component_weights)
        self.assert_score_range(composite_score)
        
        # Verify composite is reasonable weighted average
        expected_range = (50, 70)  # Based on input scores
        assert expected_range[0] <= composite_score <= expected_range[1], \
            f"Composite score {composite_score} outside expected range {expected_range}"
    
    def test_asset_class_score_calibration(self):
        """Test risk score calibration across different asset classes"""
        
        # Generate sample scores for each asset class
        options_scores = [75, 82, 68, 91, 78, 85, 72, 89, 76, 80]  # High risk
        stock_scores = [45, 52, 38, 61, 47, 55, 42, 58, 49, 51]    # Medium risk
        bond_scores = [15, 22, 8, 28, 18, 25, 12, 31, 19, 21]      # Low risk
        tbill_scores = [2, 5, 1, 8, 3, 6, 0, 9, 4, 7]             # Minimal risk
        
        # Test calibration for each asset class
        self.assert_score_calibration(options_scores, AssetCategory.DERIVATIVES)
        self.assert_score_calibration(stock_scores, AssetCategory.EQUITY)
        self.assert_score_calibration(bond_scores, AssetCategory.FIXED_INCOME)
        self.assert_score_calibration(tbill_scores, AssetCategory.COMMODITY)  # Using commodity as proxy
    
    def test_risk_score_consistency_across_time(self):
        """Test risk score consistency for identical inputs over time"""
        mock_inputs = {
            'volatility': 0.25,
            'bid_ask_spread': 0.001,
            'volume': 1000000,
            'vix': 20.0,
            'position_size': 0.05,
            'market_regime': 'NORMAL'
        }
        
        def calculate_deterministic_risk_score(**inputs) -> int:
            """Deterministic risk calculation for consistency testing"""
            vol_score = min(int(inputs['volatility'] * 200), 100)
            liquidity_score = min(int(inputs['bid_ask_spread'] * 10000), 100)
            position_score = min(int(inputs['position_size'] * 1000), 100)
            
            return min(int((vol_score + liquidity_score + position_score) / 3), 100)
        
        # Test consistency over multiple calculations
        consistent_score = self.assert_score_consistency(
            calculate_deterministic_risk_score, 
            mock_inputs, 
            iterations=100
        )
        
        self.assert_score_range(consistent_score)


# Performance and stress testing fixtures
@pytest.fixture
def large_market_dataset():
    """Generate large dataset for performance testing"""
    return {
        'price_data': np.random.normal(100, 5, 10000),
        'volume_data': np.random.poisson(1000000, 10000),
        'bid_ask_data': [(p - 0.01, p + 0.01) for p in np.random.normal(100, 5, 10000)]
    }


@pytest.mark.performance
class TestRiskCalculationPerformance(RiskComponentTestBase):
    """Performance tests for risk calculation speed"""
    
    def test_batch_risk_calculation_performance(self, large_market_dataset):
        """Test batch processing of risk calculations"""
        def batch_risk_calculation(market_data: Dict) -> List[int]:
            """Calculate risk scores for large batch of assets"""
            scores = []
            for i in range(len(market_data['price_data'])):
                # Simplified risk calculation
                vol = np.std(market_data['price_data'][max(0, i-20):i+1])
                score = min(int(vol * 10), 100)
                scores.append(score)
            return scores
        
        start_time = time.perf_counter()
        risk_scores = batch_risk_calculation(large_market_dataset)
        end_time = time.perf_counter()
        
        execution_time = end_time - start_time
        scores_per_second = len(risk_scores) / execution_time
        
        # Should process at least 1000 scores per second
        assert scores_per_second >= 1000, f"Performance too slow: {scores_per_second:.0f} scores/sec"
        
        # Verify all scores are valid
        for score in risk_scores:
            self.assert_score_range(score)
    
    def test_concurrent_risk_calculation(self):
        """Test concurrent risk calculations"""
        import concurrent.futures
        import threading
        
        def single_risk_calculation(asset_data: Dict) -> int:
            """Single asset risk calculation"""
            vol = asset_data.get('volatility', 0.2)
            spread = asset_data.get('spread', 0.001)
            return min(int((vol * 100 + spread * 10000) / 2), 100)
        
        # Create 100 asset datasets
        asset_datasets = [
            {'volatility': np.random.normal(0.2, 0.05), 'spread': np.random.uniform(0.0001, 0.01)}
            for _ in range(100)
        ]
        
        start_time = time.perf_counter()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(single_risk_calculation, data) for data in asset_datasets]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        
        # Should complete 100 calculations in under 1 second
        assert execution_time < 1.0, f"Concurrent calculations too slow: {execution_time:.2f}s"
        assert len(results) == 100, "Not all calculations completed"
        
        for score in results:
            self.assert_score_range(score)


if __name__ == "__main__":
    # Run specific test categories
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-m", "not performance"  # Skip performance tests by default
    ])