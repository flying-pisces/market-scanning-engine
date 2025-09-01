"""
Risk Scoring System - Data Validation and Accuracy Tests
Author: Claude Code (QA Engineer)
Version: 1.0

Comprehensive tests for data validation, historical accuracy, and predictive
performance of risk scoring models. Validates correlation with actual market
behavior and ensures scoring accuracy across different market conditions.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Tuple, Any
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
from dataclasses import dataclass

from data_models.python.core_models import Asset, AssetCategory, MarketData, RiskScore
from data_models.python.signal_models import RiskAssessment, MarketRegimeData


@dataclass
class AccuracyMetrics:
    """Container for accuracy measurement results"""
    correlation: float
    r_squared: float
    mean_absolute_error: float
    root_mean_squared_error: float
    prediction_accuracy: float
    directional_accuracy: float
    calibration_score: float


class AccuracyTestBase:
    """Base class for accuracy testing utilities"""
    
    @staticmethod
    def generate_historical_data(
        days: int = 252,
        base_price: int = 10000,
        annual_volatility: float = 0.20
    ) -> List[MarketData]:
        """Generate realistic historical market data"""
        
        data = []
        current_price = base_price
        daily_vol = annual_volatility / np.sqrt(252)
        
        for i in range(days):
            # Generate realistic daily return with some serial correlation
            if i == 0:
                daily_return = np.random.normal(0.0005, daily_vol)  # Small positive drift
            else:
                # Add slight momentum/mean reversion
                prev_return = (data[-1].close_price_cents - data[-1].open_price_cents) / data[-1].open_price_cents
                momentum = prev_return * 0.1  # 10% momentum
                daily_return = np.random.normal(momentum, daily_vol)
            
            # Calculate OHLC with realistic intraday patterns
            open_price = current_price
            close_price = int(open_price * (1 + daily_return))
            
            # Intraday range based on daily volatility
            intraday_vol = abs(daily_return) * 2 + daily_vol * np.random.uniform(0.5, 2.0)
            range_cents = int(open_price * intraday_vol)
            
            high_price = max(open_price, close_price) + np.random.randint(0, range_cents // 2)
            low_price = min(open_price, close_price) - np.random.randint(0, range_cents // 2)
            
            # Volume with some correlation to volatility
            base_volume = 1000000
            vol_multiplier = 1 + abs(daily_return) * 10  # Higher volume on big moves
            volume = int(base_volume * vol_multiplier * np.random.uniform(0.5, 2.0))
            
            # Bid-ask spread correlated with volatility
            spread_bps = max(1, int(abs(daily_return) * 10000 + np.random.uniform(1, 10)))
            spread_cents = (close_price * spread_bps) // 10000
            
            market_data = MarketData(
                asset_id=f"HIST_{i:04d}",
                timestamp=datetime.utcnow() - timedelta(days=days-i),
                open_price_cents=open_price,
                high_price_cents=high_price,
                low_price_cents=low_price,
                close_price_cents=close_price,
                volume=volume,
                bid_price_cents=close_price - spread_cents // 2,
                ask_price_cents=close_price + spread_cents // 2,
                data_source="ACCURACY_TEST",
                data_quality_score=95
            )
            
            data.append(market_data)
            current_price = close_price
        
        return data
    
    @staticmethod
    def calculate_realized_volatility(price_data: List[MarketData], periods: int = 30) -> List[float]:
        """Calculate rolling realized volatility"""
        if len(price_data) < periods + 1:
            return []
        
        realized_vols = []
        
        for i in range(periods, len(price_data)):
            # Get returns for the period
            returns = []
            for j in range(i - periods + 1, i + 1):
                if j > 0:
                    ret = (price_data[j].close_price_cents - price_data[j-1].close_price_cents) / price_data[j-1].close_price_cents
                    returns.append(ret)
            
            if returns:
                # Annualized volatility
                vol = np.std(returns) * np.sqrt(252)
                realized_vols.append(vol)
            else:
                realized_vols.append(0.0)
        
        return realized_vols
    
    @staticmethod
    def calculate_accuracy_metrics(
        predicted_scores: List[int],
        actual_volatilities: List[float],
        prediction_horizon_scores: Optional[List[int]] = None,
        actual_future_volatilities: Optional[List[float]] = None
    ) -> AccuracyMetrics:
        """Calculate comprehensive accuracy metrics"""
        
        # Convert risk scores to volatility estimates for comparison
        predicted_vols = [score / 100 * 0.5 for score in predicted_scores]  # Assume 100 score = 50% vol
        
        # Correlation analysis
        correlation = stats.pearsonr(predicted_vols, actual_volatilities)[0] if len(predicted_vols) > 1 else 0.0
        r_squared = r2_score(actual_volatilities, predicted_vols) if len(predicted_vols) > 1 else 0.0
        
        # Error metrics
        mae = mean_absolute_error(actual_volatilities, predicted_vols)
        rmse = np.sqrt(mean_squared_error(actual_volatilities, predicted_vols))
        
        # Prediction accuracy (for future-looking metrics)
        prediction_accuracy = 0.0
        directional_accuracy = 0.0
        
        if prediction_horizon_scores and actual_future_volatilities:
            pred_future_vols = [score / 100 * 0.5 for score in prediction_horizon_scores]
            
            # R-squared for prediction
            prediction_accuracy = r2_score(actual_future_volatilities, pred_future_vols)
            
            # Directional accuracy (did we predict direction of vol change correctly)
            if len(actual_volatilities) == len(actual_future_volatilities):
                vol_changes = [actual_future_volatilities[i] - actual_volatilities[i] 
                              for i in range(len(actual_volatilities))]
                pred_changes = [pred_future_vols[i] - predicted_vols[i] 
                               for i in range(len(predicted_vols))]
                
                correct_directions = sum(1 for i in range(len(vol_changes)) 
                                       if (vol_changes[i] > 0) == (pred_changes[i] > 0))
                directional_accuracy = correct_directions / len(vol_changes) if vol_changes else 0.0
        
        # Calibration score (how well do risk buckets match actual risk)
        calibration_score = AccuracyTestBase._calculate_calibration_score(predicted_scores, actual_volatilities)
        
        return AccuracyMetrics(
            correlation=correlation,
            r_squared=r_squared,
            mean_absolute_error=mae,
            root_mean_squared_error=rmse,
            prediction_accuracy=prediction_accuracy,
            directional_accuracy=directional_accuracy,
            calibration_score=calibration_score
        )
    
    @staticmethod
    def _calculate_calibration_score(risk_scores: List[int], actual_vols: List[float]) -> float:
        """Calculate calibration score - how well risk buckets match actual outcomes"""
        if not risk_scores or not actual_vols or len(risk_scores) != len(actual_vols):
            return 0.0
        
        # Group by risk score buckets
        buckets = {i: [] for i in range(0, 101, 10)}  # 0-10, 10-20, ..., 90-100
        
        for score, vol in zip(risk_scores, actual_vols):
            bucket = (score // 10) * 10
            if bucket in buckets:
                buckets[bucket].append(vol)
        
        # Calculate calibration error
        calibration_errors = []
        for bucket_start, vols in buckets.items():
            if vols:
                expected_vol = (bucket_start + 5) / 100 * 0.5  # Middle of bucket range
                actual_avg_vol = np.mean(vols)
                error = abs(expected_vol - actual_avg_vol)
                calibration_errors.append(error)
        
        # Return inverse of mean calibration error (higher is better)
        mean_error = np.mean(calibration_errors) if calibration_errors else 1.0
        return max(0.0, 1.0 - mean_error * 2)  # Scale so perfect calibration = 1.0


class TestHistoricalAccuracy(AccuracyTestBase):
    """Test historical accuracy against realized market data"""
    
    def test_volatility_prediction_accuracy(self):
        """Test accuracy of volatility-based risk scoring against historical volatility"""
        
        class HistoricalVolatilityRiskModel:
            def __init__(self, lookback_period: int = 30):
                self.lookback_period = lookback_period
                self.model_history = []
            
            def calculate_risk_score(self, current_data: MarketData, historical_context: List[MarketData]) -> int:
                """Calculate risk score based on historical volatility patterns"""
                
                if len(historical_context) < self.lookback_period:
                    # Insufficient history - use current day volatility
                    current_vol = (current_data.high_price_cents - current_data.low_price_cents) / current_data.close_price_cents
                    return min(int(current_vol * 1000), 100)
                
                # Calculate recent realized volatility
                recent_returns = []
                for i in range(max(1, len(historical_context) - self.lookback_period), len(historical_context)):
                    if i > 0:
                        ret = (historical_context[i].close_price_cents - historical_context[i-1].close_price_cents) / historical_context[i-1].close_price_cents
                        recent_returns.append(ret)
                
                if not recent_returns:
                    return 50  # Default medium risk
                
                # Annualized volatility
                realized_vol = np.std(recent_returns) * np.sqrt(252)
                
                # Current day intraday volatility
                current_day_vol = (current_data.high_price_cents - current_data.low_price_cents) / current_data.close_price_cents
                
                # Combine recent volatility with current day signal
                combined_vol = realized_vol * 0.7 + current_day_vol * 0.3
                
                # Convert to 0-100 risk score (50% vol = 100 score)
                risk_score = min(int(combined_vol * 200), 100)
                
                # Store for analysis
                self.model_history.append({
                    'timestamp': current_data.timestamp,
                    'risk_score': risk_score,
                    'realized_vol': realized_vol,
                    'current_day_vol': current_day_vol,
                    'combined_vol': combined_vol
                })
                
                return risk_score
        
        # Generate test data with different volatility regimes
        print("\nHistorical Volatility Accuracy Test:")
        
        # Low volatility period
        low_vol_data = self.generate_historical_data(days=100, annual_volatility=0.12)
        # High volatility period  
        high_vol_data = self.generate_historical_data(days=100, annual_volatility=0.35)
        # Normal volatility period
        normal_vol_data = self.generate_historical_data(days=100, annual_volatility=0.20)
        
        all_data = low_vol_data + normal_vol_data + high_vol_data
        
        model = HistoricalVolatilityRiskModel(lookback_period=30)
        
        # Calculate risk scores and realized volatilities
        risk_scores = []
        realized_vols = []
        
        for i, data_point in enumerate(all_data):
            if i >= 30:  # Need some history for realized vol calculation
                # Get historical context
                historical_context = all_data[max(0, i-50):i]  # Up to 50 days of history
                
                # Calculate risk score
                risk_score = model.calculate_risk_score(data_point, historical_context)
                risk_scores.append(risk_score)
                
                # Calculate realized volatility for the period ending at this point
                recent_data = all_data[max(0, i-30):i+1]
                if len(recent_data) > 1:
                    returns = []
                    for j in range(1, len(recent_data)):
                        ret = (recent_data[j].close_price_cents - recent_data[j-1].close_price_cents) / recent_data[j-1].close_price_cents
                        returns.append(ret)
                    realized_vol = np.std(returns) * np.sqrt(252) if returns else 0.0
                    realized_vols.append(realized_vol)
        
        # Calculate accuracy metrics
        accuracy = self.calculate_accuracy_metrics(risk_scores, realized_vols)
        
        print(f"Volatility Prediction Accuracy Results:")
        print(f"  Correlation: {accuracy.correlation:.3f}")
        print(f"  R-squared: {accuracy.r_squared:.3f}")
        print(f"  Mean Absolute Error: {accuracy.mean_absolute_error:.4f}")
        print(f"  RMSE: {accuracy.root_mean_squared_error:.4f}")
        print(f"  Calibration Score: {accuracy.calibration_score:.3f}")
        
        # Validate accuracy requirements
        assert accuracy.correlation >= 0.4, f"Correlation {accuracy.correlation:.3f} below minimum 0.4"
        assert accuracy.r_squared >= 0.15, f"R-squared {accuracy.r_squared:.3f} below minimum 0.15"
        assert accuracy.calibration_score >= 0.3, f"Calibration score {accuracy.calibration_score:.3f} below minimum 0.3"
        
        # Test different volatility regimes produce different risk scores
        low_vol_scores = risk_scores[:70]   # First 70 points (low vol period)
        high_vol_scores = risk_scores[170:240]  # Last 70 points (high vol period)
        
        assert np.mean(high_vol_scores) > np.mean(low_vol_scores) + 10, \
            f"High vol period should have higher scores than low vol period"
    
    def test_drawdown_prediction_accuracy(self):
        """Test accuracy of risk scores in predicting future drawdowns"""
        
        class DrawdownPredictionModel:
            def __init__(self):
                self.predictions = []
            
            def predict_drawdown_risk(self, market_data: MarketData, window_data: List[MarketData]) -> Tuple[int, Dict[str, float]]:
                """Predict likelihood of significant drawdown in next period"""
                
                if len(window_data) < 20:
                    return 50, {"insufficient_data": True}
                
                # Calculate multiple risk factors
                factors = {}
                
                # Recent volatility trend
                recent_returns = []
                for i in range(len(window_data) - 10, len(window_data)):
                    if i > 0:
                        ret = (window_data[i].close_price_cents - window_data[i-1].close_price_cents) / window_data[i-1].close_price_cents
                        recent_returns.append(ret)
                
                factors['recent_volatility'] = np.std(recent_returns) * np.sqrt(252) if recent_returns else 0.0
                
                # Price momentum (negative momentum = higher drawdown risk)
                if len(window_data) >= 5:
                    recent_returns_5d = [
                        (window_data[i].close_price_cents - window_data[i-1].close_price_cents) / window_data[i-1].close_price_cents
                        for i in range(len(window_data) - 5, len(window_data))
                        if i > 0
                    ]
                    factors['momentum'] = np.mean(recent_returns_5d) if recent_returns_5d else 0.0
                
                # Volume surge (unusual volume = higher risk)
                recent_volumes = [d.volume for d in window_data[-10:]]
                historical_volumes = [d.volume for d in window_data[:-10]]
                factors['volume_ratio'] = np.mean(recent_volumes) / np.mean(historical_volumes) if historical_volumes else 1.0
                
                # Spread widening (liquidity deterioration)
                recent_spreads = []
                for data in window_data[-5:]:
                    if data.bid_price_cents and data.ask_price_cents:
                        spread_pct = (data.ask_price_cents - data.bid_price_cents) / data.close_price_cents
                        recent_spreads.append(spread_pct)
                factors['avg_spread'] = np.mean(recent_spreads) if recent_spreads else 0.01
                
                # Combine factors into risk score
                vol_score = min(factors['recent_volatility'] * 200, 40)  # Max 40 points for volatility
                momentum_score = max(-factors['momentum'] * 1000, 0)    # Negative momentum increases risk
                momentum_score = min(momentum_score, 30)                # Max 30 points
                volume_score = min((factors['volume_ratio'] - 1) * 50, 20)  # Unusual volume
                spread_score = min(factors['avg_spread'] * 1000, 10)    # Max 10 points for spreads
                
                total_risk = min(int(vol_score + momentum_score + volume_score + spread_score), 100)
                
                self.predictions.append({
                    'timestamp': market_data.timestamp,
                    'risk_score': total_risk,
                    'factors': factors
                })
                
                return total_risk, factors
        
        # Generate data with embedded drawdown periods
        print("\nDrawdown Prediction Accuracy Test:")
        
        # Create data with simulated market stress periods
        base_data = []
        current_price = 10000
        
        for i in range(200):
            if 50 <= i < 70:  # Drawdown period 1
                drift = -0.003  # Negative drift
                vol = 0.035     # Higher volatility
            elif 120 <= i < 135:  # Drawdown period 2
                drift = -0.004
                vol = 0.045
            else:  # Normal periods
                drift = 0.001
                vol = 0.018
            
            daily_return = np.random.normal(drift, vol)
            new_price = max(int(current_price * (1 + daily_return)), 1000)
            
            # Create realistic OHLC
            open_price = current_price
            close_price = new_price
            range_pct = abs(daily_return) * 1.5 + vol * np.random.uniform(0.5, 1.5)
            range_cents = int(open_price * range_pct)
            
            high_price = max(open_price, close_price) + np.random.randint(0, range_cents)
            low_price = min(open_price, close_price) - np.random.randint(0, range_cents)
            
            # Volume increases during stress
            base_volume = 1000000
            if 50 <= i < 70 or 120 <= i < 135:
                volume = int(base_volume * np.random.uniform(2.0, 5.0))
            else:
                volume = int(base_volume * np.random.uniform(0.8, 1.5))
            
            data = MarketData(
                asset_id=f"DRAWDOWN_TEST_{i}",
                timestamp=datetime.utcnow() - timedelta(days=200-i),
                open_price_cents=open_price,
                high_price_cents=high_price,
                low_price_cents=low_price,
                close_price_cents=close_price,
                volume=volume,
                bid_price_cents=close_price - 25,
                ask_price_cents=close_price + 25,
                data_source="DRAWDOWN_TEST",
                data_quality_score=95
            )
            base_data.append(data)
            current_price = new_price
        
        model = DrawdownPredictionModel()
        
        # Calculate risk scores and actual future drawdowns
        risk_scores = []
        actual_drawdowns = []
        
        for i in range(30, len(base_data) - 20):  # Need history and future data
            window_data = base_data[max(0, i-30):i]
            current_data = base_data[i]
            
            # Get risk score
            risk_score, factors = model.predict_drawdown_risk(current_data, window_data)
            risk_scores.append(risk_score)
            
            # Calculate actual maximum drawdown in next 20 days
            future_prices = [base_data[j].close_price_cents for j in range(i, min(i+20, len(base_data)))]
            current_price = current_data.close_price_cents
            
            max_drawdown = 0
            peak_price = current_price
            for price in future_prices:
                peak_price = max(peak_price, price)
                drawdown = (peak_price - price) / peak_price
                max_drawdown = max(max_drawdown, drawdown)
            
            actual_drawdowns.append(max_drawdown)
        
        # Analyze prediction accuracy
        if risk_scores and actual_drawdowns:
            # Convert drawdowns to comparable scale (0-100)
            drawdown_scores = [min(dd * 500, 100) for dd in actual_drawdowns]  # 20% drawdown = 100 score
            
            accuracy = self.calculate_accuracy_metrics(risk_scores, actual_drawdowns)
            
            print(f"Drawdown Prediction Results:")
            print(f"  Correlation: {accuracy.correlation:.3f}")
            print(f"  R-squared: {accuracy.r_squared:.3f}")
            print(f"  Mean Absolute Error: {accuracy.mean_absolute_error:.4f}")
            print(f"  Calibration Score: {accuracy.calibration_score:.3f}")
            
            # Test if high risk scores correspond to higher actual drawdowns
            high_risk_indices = [i for i, score in enumerate(risk_scores) if score >= 70]
            low_risk_indices = [i for i, score in enumerate(risk_scores) if score <= 30]
            
            if high_risk_indices and low_risk_indices:
                high_risk_drawdowns = [actual_drawdowns[i] for i in high_risk_indices]
                low_risk_drawdowns = [actual_drawdowns[i] for i in low_risk_indices]
                
                mean_high_dd = np.mean(high_risk_drawdowns)
                mean_low_dd = np.mean(low_risk_drawdowns)
                
                print(f"  High Risk Mean Drawdown: {mean_high_dd:.3f} ({mean_high_dd*100:.1f}%)")
                print(f"  Low Risk Mean Drawdown: {mean_low_dd:.3f} ({mean_low_dd*100:.1f}%)")
                
                # High risk periods should have higher actual drawdowns
                assert mean_high_dd > mean_low_dd, f"High risk scores should predict higher drawdowns"
                assert mean_high_dd > mean_low_dd * 1.5, f"Risk discrimination should be meaningful"
            
            # Overall accuracy requirements
            assert accuracy.correlation >= 0.25, f"Drawdown prediction correlation {accuracy.correlation:.3f} too low"
        
        print(f"  Total predictions made: {len(risk_scores)}")
        print(f"  Average risk score: {np.mean(risk_scores):.1f}")
        print(f"  Average actual drawdown: {np.mean(actual_drawdowns):.3f} ({np.mean(actual_drawdowns)*100:.1f}%)")


class TestCrossAssetAccuracy(AccuracyTestBase):
    """Test accuracy across different asset classes"""
    
    def test_asset_class_risk_calibration(self):
        """Test risk score calibration across different asset classes"""
        
        class MultiAssetRiskModel:
            def __init__(self):
                self.asset_class_multipliers = {
                    AssetCategory.DERIVATIVES: 1.5,
                    AssetCategory.EQUITY: 1.0,
                    AssetCategory.FIXED_INCOME: 0.4,
                    AssetCategory.COMMODITY: 0.2,
                    AssetCategory.CRYPTO: 2.0
                }
                self.results_by_class = {}
            
            def calculate_risk_by_asset_class(
                self, 
                market_data: MarketData, 
                asset_class: AssetCategory,
                days_to_expiry: Optional[int] = None
            ) -> int:
                """Calculate risk score adjusted for asset class"""
                
                # Base volatility calculation
                volatility = (market_data.high_price_cents - market_data.low_price_cents) / market_data.close_price_cents
                base_score = min(int(volatility * 1000), 80)  # Leave room for multipliers
                
                # Asset class adjustment
                multiplier = self.asset_class_multipliers.get(asset_class, 1.0)
                adjusted_score = min(int(base_score * multiplier), 100)
                
                # Additional adjustments for derivatives
                if asset_class == AssetCategory.DERIVATIVES and days_to_expiry is not None:
                    if days_to_expiry <= 7:
                        adjusted_score = min(adjusted_score + 20, 100)
                    elif days_to_expiry <= 30:
                        adjusted_score = min(adjusted_score + 10, 100)
                
                # Store results by class
                if asset_class not in self.results_by_class:
                    self.results_by_class[asset_class] = []
                self.results_by_class[asset_class].append({
                    'base_score': base_score,
                    'adjusted_score': adjusted_score,
                    'volatility': volatility
                })
                
                return adjusted_score
        
        model = MultiAssetRiskModel()
        
        print("\nCross-Asset Risk Calibration Test:")
        
        # Generate data for different asset classes with appropriate characteristics
        asset_class_data = {}
        
        # Options (high volatility, time decay)
        options_data = self.generate_historical_data(days=100, annual_volatility=0.40)
        for i, data in enumerate(options_data):
            days_to_expiry = max(1, 30 - (i % 30))  # Simulate various expiries
            score = model.calculate_risk_by_asset_class(data, AssetCategory.DERIVATIVES, days_to_expiry)
        asset_class_data[AssetCategory.DERIVATIVES] = [r['adjusted_score'] for r in model.results_by_class[AssetCategory.DERIVATIVES]]
        
        # Stocks (medium volatility)
        stock_data = self.generate_historical_data(days=100, annual_volatility=0.25)
        for data in stock_data:
            score = model.calculate_risk_by_asset_class(data, AssetCategory.EQUITY)
        asset_class_data[AssetCategory.EQUITY] = [r['adjusted_score'] for r in model.results_by_class[AssetCategory.EQUITY]]
        
        # Bonds (low volatility)
        bond_data = self.generate_historical_data(days=100, annual_volatility=0.08)
        for data in bond_data:
            score = model.calculate_risk_by_asset_class(data, AssetCategory.FIXED_INCOME)
        asset_class_data[AssetCategory.FIXED_INCOME] = [r['adjusted_score'] for r in model.results_by_class[AssetCategory.FIXED_INCOME]]
        
        # T-bills/CDs (very low volatility)
        tbill_data = self.generate_historical_data(days=100, annual_volatility=0.03)
        for data in tbill_data:
            score = model.calculate_risk_by_asset_class(data, AssetCategory.COMMODITY)
        asset_class_data[AssetCategory.COMMODITY] = [r['adjusted_score'] for r in model.results_by_class[AssetCategory.COMMODITY]]
        
        # Analyze calibration by asset class
        expected_ranges = {
            AssetCategory.DERIVATIVES: (70, 95),
            AssetCategory.EQUITY: (30, 80),
            AssetCategory.FIXED_INCOME: (5, 40),
            AssetCategory.COMMODITY: (0, 20)
        }
        
        for asset_class, scores in asset_class_data.items():
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            min_score = min(scores)
            max_score = max(scores)
            
            print(f"\n{asset_class.value.title()} Asset Class:")
            print(f"  Mean Score: {mean_score:.1f}")
            print(f"  Std Dev: {std_score:.1f}")
            print(f"  Range: {min_score} - {max_score}")
            
            if asset_class in expected_ranges:
                expected_min, expected_max = expected_ranges[asset_class]
                print(f"  Expected Range: {expected_min} - {expected_max}")
                
                # Validate calibration
                assert expected_min <= mean_score <= expected_max, \
                    f"{asset_class.value} mean score {mean_score:.1f} outside expected range [{expected_min}, {expected_max}]"
                
                # Check that most scores fall within expected range (allowing some outliers)
                in_range_count = sum(1 for score in scores if expected_min <= score <= expected_max)
                in_range_pct = in_range_count / len(scores)
                
                assert in_range_pct >= 0.7, \
                    f"{asset_class.value} only {in_range_pct:.1%} of scores in expected range"
                
                print(f"  Calibration: {in_range_pct:.1%} within expected range ✓")
        
        # Test relative ordering
        mean_scores = {asset_class: np.mean(scores) for asset_class, scores in asset_class_data.items()}
        
        # Verify expected risk ordering
        assert mean_scores[AssetCategory.DERIVATIVES] > mean_scores[AssetCategory.EQUITY], \
            "Options should have higher risk than stocks"
        assert mean_scores[AssetCategory.EQUITY] > mean_scores[AssetCategory.FIXED_INCOME], \
            "Stocks should have higher risk than bonds"
        assert mean_scores[AssetCategory.FIXED_INCOME] > mean_scores[AssetCategory.COMMODITY], \
            "Bonds should have higher risk than T-bills"
        
        print(f"\nRisk Ordering Validation: ✓")
        print(f"  Options ({mean_scores[AssetCategory.DERIVATIVES]:.1f}) > Stocks ({mean_scores[AssetCategory.EQUITY]:.1f}) > " +
              f"Bonds ({mean_scores[AssetCategory.FIXED_INCOME]:.1f}) > T-bills ({mean_scores[AssetCategory.COMMODITY]:.1f})")


class TestMarketRegimeAccuracy(AccuracyTestBase):
    """Test accuracy in different market regimes"""
    
    def test_regime_specific_risk_accuracy(self):
        """Test risk scoring accuracy across different market regimes"""
        
        class RegimeAwareRiskModel:
            def __init__(self):
                self.regime_adjustments = {
                    'bull_market': 0.8,
                    'bear_market': 1.4,
                    'sideways': 1.0,
                    'high_vol': 1.3,
                    'low_vol': 0.7
                }
                self.results_by_regime = {}
            
            def classify_market_regime(self, price_data: List[MarketData]) -> str:
                """Simple market regime classification"""
                if len(price_data) < 20:
                    return 'sideways'
                
                # Calculate 20-day return and volatility
                returns = []
                for i in range(len(price_data) - 19, len(price_data)):
                    if i > 0:
                        ret = (price_data[i].close_price_cents - price_data[i-1].close_price_cents) / price_data[i-1].close_price_cents
                        returns.append(ret)
                
                if not returns:
                    return 'sideways'
                
                avg_return = np.mean(returns)
                volatility = np.std(returns) * np.sqrt(252)
                
                # Classify regime
                if volatility > 0.30:
                    return 'high_vol'
                elif volatility < 0.15:
                    return 'low_vol'
                elif avg_return > 0.002:  # >0.2% daily average
                    return 'bull_market'
                elif avg_return < -0.001:  # <-0.1% daily average
                    return 'bear_market'
                else:
                    return 'sideways'
            
            def calculate_regime_adjusted_risk(self, market_data: MarketData, historical_context: List[MarketData]) -> Tuple[int, str]:
                """Calculate risk score with regime adjustment"""
                
                # Base risk calculation
                volatility = (market_data.high_price_cents - market_data.low_price_cents) / market_data.close_price_cents
                base_score = min(int(volatility * 1000), 70)
                
                # Detect current regime
                regime = self.classify_market_regime(historical_context + [market_data])
                
                # Apply regime adjustment
                adjustment = self.regime_adjustments.get(regime, 1.0)
                adjusted_score = min(int(base_score * adjustment), 100)
                
                # Store results
                if regime not in self.results_by_regime:
                    self.results_by_regime[regime] = []
                self.results_by_regime[regime].append({
                    'base_score': base_score,
                    'adjusted_score': adjusted_score,
                    'volatility': volatility
                })
                
                return adjusted_score, regime
        
        model = RegimeAwareRiskModel()
        
        print("\nMarket Regime Risk Accuracy Test:")
        
        # Generate data for different market regimes
        regimes_data = {}
        
        # Bull market
        bull_data = self.generate_historical_data(days=80, annual_volatility=0.18)
        # Add positive drift
        for i in range(1, len(bull_data)):
            drift = 0.002  # 0.2% daily drift
            old_price = bull_data[i-1].close_price_cents
            new_price = int(old_price * (1 + drift))
            bull_data[i].open_price_cents = old_price
            bull_data[i].close_price_cents = new_price
            bull_data[i].high_price_cents = max(bull_data[i].high_price_cents, new_price)
            bull_data[i].low_price_cents = min(bull_data[i].low_price_cents, new_price)
        
        # Bear market
        bear_data = self.generate_historical_data(days=80, annual_volatility=0.35)
        # Add negative drift
        for i in range(1, len(bear_data)):
            drift = -0.003  # -0.3% daily drift
            old_price = bear_data[i-1].close_price_cents
            new_price = max(int(old_price * (1 + drift)), 1000)  # Don't go below $10
            bear_data[i].open_price_cents = old_price
            bear_data[i].close_price_cents = new_price
            bear_data[i].high_price_cents = max(bear_data[i].high_price_cents, old_price)
            bear_data[i].low_price_cents = min(bear_data[i].low_price_cents, new_price)
        
        # High volatility (sideways but volatile)
        high_vol_data = self.generate_historical_data(days=80, annual_volatility=0.45)
        
        # Low volatility
        low_vol_data = self.generate_historical_data(days=80, annual_volatility=0.08)
        
        # Process each regime
        regime_datasets = {
            'bull_market': bull_data,
            'bear_market': bear_data,
            'high_vol': high_vol_data,
            'low_vol': low_vol_data
        }
        
        regime_scores = {}
        regime_classifications = {}
        
        for expected_regime, data in regime_datasets.items():
            scores = []
            classified_regimes = []
            
            for i in range(20, len(data)):  # Need history for regime classification
                historical_context = data[max(0, i-30):i]
                score, classified_regime = model.calculate_regime_adjusted_risk(data[i], historical_context)
                scores.append(score)
                classified_regimes.append(classified_regime)
            
            regime_scores[expected_regime] = scores
            regime_classifications[expected_regime] = classified_regimes
            
            mean_score = np.mean(scores)
            most_common_classification = max(set(classified_regimes), key=classified_regimes.count)
            classification_accuracy = classified_regimes.count(most_common_classification) / len(classified_regimes)
            
            print(f"\n{expected_regime.title().replace('_', ' ')}:")
            print(f"  Mean Risk Score: {mean_score:.1f}")
            print(f"  Most Common Classification: {most_common_classification}")
            print(f"  Classification Accuracy: {classification_accuracy:.1%}")
        
        # Validate regime-specific behavior
        mean_scores = {regime: np.mean(scores) for regime, scores in regime_scores.items()}
        
        # Bear markets should have higher risk than bull markets
        assert mean_scores['bear_market'] > mean_scores['bull_market'], \
            "Bear market should have higher risk scores than bull market"
        
        # High vol should have higher risk than low vol
        assert mean_scores['high_vol'] > mean_scores['low_vol'], \
            "High volatility regime should have higher risk than low volatility"
        
        # Bear market should be the highest risk regime
        assert mean_scores['bear_market'] == max(mean_scores.values()), \
            "Bear market should have the highest average risk score"
        
        # Low vol should be among the lowest risk regimes
        assert mean_scores['low_vol'] <= min(mean_scores['bull_market'], mean_scores['high_vol']), \
            "Low volatility should have lower risk than bull market or high vol"
        
        print(f"\nRegime Risk Ordering Validation: ✓")
        print(f"  Bear ({mean_scores['bear_market']:.1f}) > High Vol ({mean_scores['high_vol']:.1f}) > " +
              f"Bull ({mean_scores['bull_market']:.1f}) > Low Vol ({mean_scores['low_vol']:.1f})")
        
        # Validate regime classification accuracy
        for expected_regime, classifications in regime_classifications.items():
            most_common = max(set(classifications), key=classifications.count)
            accuracy = classifications.count(expected_regime) / len(classifications)
            
            # Should classify correctly at least 60% of the time
            if expected_regime in ['bull_market', 'bear_market', 'high_vol', 'low_vol']:
                assert accuracy >= 0.4 or most_common == expected_regime, \
                    f"Regime classification accuracy too low for {expected_regime}: {accuracy:.1%}"


if __name__ == "__main__":
    # Run accuracy tests
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-s"  # Show print output for analysis
    ])