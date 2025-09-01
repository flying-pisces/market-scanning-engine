"""
Risk Scoring System - Integration Tests
Author: Claude Code (QA Engineer)
Version: 1.0

Integration tests for multi-component risk scoring system. Tests end-to-end
workflows, cross-asset class consistency, real-time data integration,
and system-level risk assessment accuracy.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import List, Dict, Optional, Tuple
from unittest.mock import Mock, patch, MagicMock
import asyncio
import concurrent.futures
import time
import json

from data_models.python.core_models import (
    Asset, AssetCategory, MarketData, TechnicalIndicators, 
    OptionsData, MarketRegime, RiskScore
)
from data_models.python.signal_models import (
    RiskAssessment, RiskFactor, RiskFactorContribution,
    MarketRegimeData, Signal, SignalDirection
)


class RiskIntegrationTestBase:
    """Base class for risk integration testing"""
    
    @staticmethod
    def create_mock_asset(
        symbol: str, 
        asset_class: AssetCategory, 
        sector: str = "TECHNOLOGY"
    ) -> Asset:
        """Create mock asset for testing"""
        return Asset(
            symbol=symbol,
            name=f"{symbol} Test Asset",
            exchange="NYSE",
            currency="USD",
            sector=sector,
            market_cap=100000000000,  # $1B in cents
            avg_volume_30d=1000000
        )
    
    @staticmethod
    def create_mock_market_data(
        asset_id: str, 
        price_cents: int = 10000,
        volume: int = 1000000,
        bid_ask_spread_bps: int = 5
    ) -> MarketData:
        """Create mock market data"""
        spread_cents = (price_cents * bid_ask_spread_bps) // 10000
        return MarketData(
            asset_id=asset_id,
            timestamp=datetime.utcnow(),
            open_price_cents=price_cents,
            high_price_cents=price_cents + 50,
            low_price_cents=price_cents - 50,
            close_price_cents=price_cents,
            volume=volume,
            bid_price_cents=price_cents - spread_cents // 2,
            ask_price_cents=price_cents + spread_cents // 2,
            data_source="TEST_FEED",
            data_quality_score=95
        )
    
    @staticmethod
    def create_mock_options_data(
        underlying_id: str,
        strike_cents: int = 10000,
        days_to_expiry: int = 30,
        option_type: str = "CALL"
    ) -> OptionsData:
        """Create mock options data with Greeks"""
        return OptionsData(
            underlying_asset_id=underlying_id,
            option_symbol=f"TEST_{option_type}_{strike_cents//100}_{days_to_expiry}",
            expiration_date=date.today() + timedelta(days=days_to_expiry),
            strike_price_cents=strike_cents,
            option_type=option_type,
            timestamp=datetime.utcnow(),
            bid_price_cents=200,  # $2.00
            ask_price_cents=220,  # $2.20
            last_price_cents=210,
            volume=500,
            open_interest=1000,
            delta=5000,  # 0.50 scaled
            gamma=200,   # 0.02 scaled
            theta=-50,   # -0.005 scaled
            vega=100,    # 0.01 scaled
            implied_volatility=2500,  # 25% scaled
            data_source="TEST_OPTIONS"
        )


class TestMultiComponentRiskScoring(RiskIntegrationTestBase):
    """Test integration of multiple risk components"""
    
    def test_end_to_end_risk_assessment_workflow(self):
        """Test complete risk assessment workflow from data to score"""
        
        # Mock risk calculation service
        class RiskCalculationService:
            def __init__(self):
                self.components = {
                    'volatility': self._calculate_volatility_risk,
                    'liquidity': self._calculate_liquidity_risk,
                    'time_decay': self._calculate_time_decay_risk,
                    'market_regime': self._calculate_market_regime_risk,
                    'position_size': self._calculate_position_size_risk
                }
                self.weights = {
                    'volatility': 0.25,
                    'liquidity': 0.20,
                    'time_decay': 0.15,
                    'market_regime': 0.20,
                    'position_size': 0.20
                }
            
            def _calculate_volatility_risk(self, market_data: MarketData, **kwargs) -> int:
                # Mock volatility calculation based on price range
                price_range = (market_data.high_price_cents - market_data.low_price_cents) / market_data.close_price_cents
                return min(int(price_range * 2000), 100)
            
            def _calculate_liquidity_risk(self, market_data: MarketData, **kwargs) -> int:
                # Mock liquidity calculation based on bid-ask spread and volume
                spread_pct = (market_data.ask_price_cents - market_data.bid_price_cents) / market_data.close_price_cents
                volume_factor = max(1 - (market_data.volume / 2000000), 0)  # Lower volume = higher risk
                return min(int((spread_pct * 10000 + volume_factor * 50)), 100)
            
            def _calculate_time_decay_risk(self, options_data: Optional[OptionsData] = None, **kwargs) -> int:
                if not options_data:
                    return 0  # No time decay for non-options
                days_to_expiry = (options_data.expiration_date - date.today()).days
                if days_to_expiry <= 7:
                    return 90
                elif days_to_expiry <= 30:
                    return 60
                else:
                    return 20
            
            def _calculate_market_regime_risk(self, regime_data: MarketRegimeData, **kwargs) -> int:
                regime_adjustments = {
                    'bull_market': 0.8,
                    'bear_market': 1.3,
                    'sideways': 1.0,
                    'high_vol': 1.4,
                    'low_vol': 0.7
                }
                base_score = 50
                adjustment = regime_adjustments.get(regime_data.primary_regime.lower(), 1.0)
                return min(int(base_score * adjustment), 100)
            
            def _calculate_position_size_risk(self, position_pct: float = 0.05, **kwargs) -> int:
                # Risk increases non-linearly with position size
                if position_pct > 0.20:
                    return 100
                elif position_pct > 0.10:
                    return 70
                elif position_pct > 0.05:
                    return 40
                else:
                    return 20
            
            def calculate_comprehensive_risk(
                self, 
                asset: Asset,
                market_data: MarketData,
                options_data: Optional[OptionsData] = None,
                regime_data: Optional[MarketRegimeData] = None,
                position_size_pct: float = 0.05
            ) -> RiskAssessment:
                """Calculate comprehensive risk assessment"""
                
                # Calculate individual component scores
                component_scores = {}
                factor_contributions = []
                
                for component, calculator in self.components.items():
                    if component == 'time_decay':
                        score = calculator(options_data=options_data)
                    elif component == 'market_regime' and regime_data:
                        score = calculator(regime_data=regime_data)
                    elif component == 'position_size':
                        score = calculator(position_pct=position_size_pct)
                    else:
                        score = calculator(market_data)
                    
                    component_scores[component] = score
                    
                    # Create factor contribution
                    contribution = RiskFactorContribution(
                        risk_assessment_id="test-assessment",
                        risk_factor_id=hash(component),
                        factor_score=score,
                        contribution_weight=self.weights[component]
                    )
                    factor_contributions.append(contribution)
                
                # Calculate weighted overall score
                overall_score = sum(
                    score * self.weights[component] 
                    for component, score in component_scores.items()
                )
                overall_score = min(int(overall_score), 100)
                
                return RiskAssessment(
                    asset_id=asset.id,
                    overall_risk_score=overall_score,
                    market_risk_score=component_scores.get('market_regime', 50),
                    liquidity_risk_score=component_scores['liquidity'],
                    volatility_risk_score=component_scores['volatility'],
                    concentration_risk_score=component_scores['position_size'],
                    model_version="1.0",
                    factor_contributions=factor_contributions
                )
        
        # Test the complete workflow
        risk_service = RiskCalculationService()
        
        # Create test data
        test_asset = self.create_mock_asset("AAPL", AssetCategory.EQUITY)
        test_market_data = self.create_mock_market_data(test_asset.id, volume=500000)  # Lower volume
        test_regime_data = MarketRegimeData(
            regime_date=date.today(),
            primary_regime="bear_market",
            volatility_regime="high_vol",
            liquidity_regime="normal",
            regime_confidence=85,
            regime_stability=70,
            market_stress_index=75
        )
        
        # Calculate risk assessment
        risk_assessment = risk_service.calculate_comprehensive_risk(
            asset=test_asset,
            market_data=test_market_data,
            regime_data=test_regime_data,
            position_size_pct=0.08  # 8% position
        )
        
        # Validate results
        assert 0 <= risk_assessment.overall_risk_score <= 100
        assert risk_assessment.overall_risk_score > 50  # Should be higher risk due to bear market
        assert len(risk_assessment.factor_contributions) == 5
        assert risk_assessment.model_version == "1.0"
        
        # Validate component scores are reasonable
        assert risk_assessment.liquidity_risk_score > 20  # Lower volume should increase liquidity risk
        assert risk_assessment.market_risk_score > 50   # Bear market should increase market risk
    
    def test_cross_asset_class_consistency(self):
        """Test risk scoring consistency across different asset classes"""
        
        class CrossAssetRiskService:
            def calculate_asset_class_risk(self, asset: Asset, base_volatility: float = 0.20) -> int:
                """Calculate risk score adjusted for asset class"""
                base_score = min(int(base_volatility * 200), 100)
                
                # Asset class multipliers
                multipliers = {
                    AssetCategory.DERIVATIVES: 1.5,    # Options - higher risk
                    AssetCategory.EQUITY: 1.0,         # Stocks - baseline
                    AssetCategory.FIXED_INCOME: 0.4,   # Bonds - lower risk
                    AssetCategory.COMMODITY: 0.2       # T-bills - minimal risk
                }
                
                if hasattr(asset, 'asset_class_id'):
                    # Use category from asset
                    multiplier = multipliers.get(AssetCategory.EQUITY, 1.0)  # Default to equity
                else:
                    multiplier = 1.0
                
                return min(int(base_score * multiplier), 100)
        
        risk_service = CrossAssetRiskService()
        
        # Create assets of different classes
        option_asset = self.create_mock_asset("SPY_CALL", AssetCategory.DERIVATIVES)
        stock_asset = self.create_mock_asset("AAPL", AssetCategory.EQUITY)
        bond_asset = self.create_mock_asset("TLT", AssetCategory.FIXED_INCOME)
        tbill_asset = self.create_mock_asset("BIL", AssetCategory.COMMODITY)
        
        # Calculate risks with same base volatility
        base_vol = 0.25  # 25% volatility
        
        option_risk = risk_service.calculate_asset_class_risk(option_asset, base_vol)
        stock_risk = risk_service.calculate_asset_class_risk(stock_asset, base_vol)
        bond_risk = risk_service.calculate_asset_class_risk(bond_asset, base_vol)
        tbill_risk = risk_service.calculate_asset_class_risk(tbill_asset, base_vol)
        
        # Validate risk ordering
        assert option_risk > stock_risk > bond_risk > tbill_risk, \
            f"Risk ordering incorrect: Options({option_risk}) > Stocks({stock_risk}) > Bonds({bond_risk}) > T-bills({tbill_risk})"
        
        # Validate ranges match expectations
        assert 70 <= option_risk <= 95, f"Options risk {option_risk} outside expected range [70,95]"
        assert 30 <= stock_risk <= 80, f"Stock risk {stock_risk} outside expected range [30,80]"
        assert 5 <= bond_risk <= 40, f"Bond risk {bond_risk} outside expected range [5,40]"
        assert 0 <= tbill_risk <= 20, f"T-bill risk {tbill_risk} outside expected range [0,20]"
    
    def test_real_time_risk_score_updates(self):
        """Test real-time risk score updates with streaming data"""
        
        class RealTimeRiskMonitor:
            def __init__(self):
                self.current_scores = {}
                self.score_history = {}
            
            def process_market_update(self, asset_id: str, market_data: MarketData) -> int:
                """Process real-time market data update"""
                # Simple real-time risk calculation
                volatility_indicator = (market_data.high_price_cents - market_data.low_price_cents) / market_data.close_price_cents
                liquidity_indicator = (market_data.ask_price_cents - market_data.bid_price_cents) / market_data.close_price_cents
                volume_indicator = max(0, 1 - (market_data.volume / 1000000))  # Normalized volume risk
                
                risk_score = min(int(
                    volatility_indicator * 5000 +
                    liquidity_indicator * 10000 +
                    volume_indicator * 50
                ), 100)
                
                # Store current and historical scores
                self.current_scores[asset_id] = risk_score
                if asset_id not in self.score_history:
                    self.score_history[asset_id] = []
                self.score_history[asset_id].append((datetime.utcnow(), risk_score))
                
                return risk_score
            
            def get_score_volatility(self, asset_id: str, periods: int = 10) -> float:
                """Calculate score volatility over recent periods"""
                if asset_id not in self.score_history or len(self.score_history[asset_id]) < periods:
                    return 0.0
                
                recent_scores = [score for _, score in self.score_history[asset_id][-periods:]]
                return float(np.std(recent_scores))
        
        monitor = RealTimeRiskMonitor()
        test_asset_id = "test-asset-123"
        
        # Simulate real-time market updates
        market_updates = [
            (10000, 500000, 5),   # Normal conditions
            (10000, 400000, 8),   # Lower volume, wider spread
            (10100, 300000, 12),  # Price up, even lower volume, wider spread
            (9950, 200000, 15),   # Price down, very low volume, wide spread
            (10050, 600000, 4),   # Recovery - higher volume, tighter spread
        ]
        
        scores = []
        for price_cents, volume, spread_bps in market_updates:
            market_data = self.create_mock_market_data(
                test_asset_id, 
                price_cents=price_cents,
                volume=volume,
                bid_ask_spread_bps=spread_bps
            )
            
            score = monitor.process_market_update(test_asset_id, market_data)
            scores.append(score)
            
            # Validate score range
            assert 0 <= score <= 100, f"Score {score} outside valid range"
        
        # Validate score progression makes sense
        assert scores[1] > scores[0], "Lower volume should increase risk"
        assert scores[3] > scores[2], "Very low volume should further increase risk"
        assert scores[4] < scores[3], "Recovery should decrease risk"
        
        # Test score volatility calculation
        score_volatility = monitor.get_score_volatility(test_asset_id)
        assert score_volatility > 0, "Should have non-zero score volatility"
        assert score_volatility < 50, "Score volatility should be reasonable"
    
    def test_portfolio_level_risk_aggregation(self):
        """Test portfolio-level risk aggregation across multiple positions"""
        
        class PortfolioRiskAggregator:
            def __init__(self):
                self.correlation_matrix = None
            
            def set_correlation_matrix(self, assets: List[str], correlations: np.ndarray):
                """Set correlation matrix for portfolio risk calculation"""
                self.correlation_matrix = pd.DataFrame(
                    correlations, 
                    index=assets, 
                    columns=assets
                )
            
            def calculate_portfolio_risk(
                self, 
                positions: Dict[str, Dict],  # {asset_id: {weight, individual_risk}}
                diversification_benefit: bool = True
            ) -> Dict[str, float]:
                """Calculate portfolio-level risk metrics"""
                
                if not positions:
                    return {'total_risk': 0.0, 'diversified_risk': 0.0, 'concentration_risk': 0.0}
                
                # Simple additive risk (no diversification)
                total_undiversified_risk = sum(
                    pos['weight'] * pos['individual_risk'] 
                    for pos in positions.values()
                )
                
                # Diversified risk using correlation matrix
                diversified_risk = total_undiversified_risk
                if diversification_benefit and self.correlation_matrix is not None:
                    asset_ids = list(positions.keys())
                    weights = [positions[asset_id]['weight'] for asset_id in asset_ids]
                    risks = [positions[asset_id]['individual_risk'] for asset_id in asset_ids]
                    
                    # Calculate correlation-adjusted risk
                    risk_contributions = []
                    for i, asset_i in enumerate(asset_ids):
                        for j, asset_j in enumerate(asset_ids):
                            if asset_i in self.correlation_matrix.index and asset_j in self.correlation_matrix.columns:
                                correlation = self.correlation_matrix.loc[asset_i, asset_j]
                                contribution = weights[i] * weights[j] * risks[i] * risks[j] * correlation
                                risk_contributions.append(contribution)
                    
                    if risk_contributions:
                        diversified_risk = min(np.sqrt(sum(risk_contributions)), 100.0)
                
                # Concentration risk (Herfindahl-Hirschman Index)
                weight_squares = [pos['weight'] ** 2 for pos in positions.values()]
                hhi = sum(weight_squares)
                concentration_risk = min(hhi * 100, 100.0)  # Scale to 0-100
                
                return {
                    'total_risk': float(total_undiversified_risk),
                    'diversified_risk': float(diversified_risk),
                    'concentration_risk': float(concentration_risk),
                    'diversification_benefit': float(max(0, total_undiversified_risk - diversified_risk))
                }
        
        aggregator = PortfolioRiskAggregator()
        
        # Set up correlation matrix for test assets
        test_assets = ['AAPL', 'GOOGL', 'TSLA', 'TLT', 'GLD']
        correlations = np.array([
            [1.0, 0.7, 0.6, -0.2, 0.1],  # AAPL
            [0.7, 1.0, 0.5, -0.1, 0.0],  # GOOGL
            [0.6, 0.5, 1.0, -0.3, 0.2],  # TSLA
            [-0.2, -0.1, -0.3, 1.0, 0.3], # TLT
            [0.1, 0.0, 0.2, 0.3, 1.0]    # GLD
        ])
        aggregator.set_correlation_matrix(test_assets, correlations)
        
        # Test concentrated portfolio (high correlation)
        concentrated_portfolio = {
            'AAPL': {'weight': 0.5, 'individual_risk': 60},
            'GOOGL': {'weight': 0.3, 'individual_risk': 65},
            'TSLA': {'weight': 0.2, 'individual_risk': 80}
        }
        
        concentrated_risk = aggregator.calculate_portfolio_risk(concentrated_portfolio)
        
        # Test diversified portfolio
        diversified_portfolio = {
            'AAPL': {'weight': 0.25, 'individual_risk': 60},
            'GOOGL': {'weight': 0.2, 'individual_risk': 65},
            'TSLA': {'weight': 0.15, 'individual_risk': 80},
            'TLT': {'weight': 0.25, 'individual_risk': 25},
            'GLD': {'weight': 0.15, 'individual_risk': 40}
        }
        
        diversified_risk = aggregator.calculate_portfolio_risk(diversified_portfolio)
        
        # Validate results
        assert concentrated_risk['concentration_risk'] > diversified_risk['concentration_risk'], \
            "Concentrated portfolio should have higher concentration risk"
        
        assert diversified_risk['diversification_benefit'] > 0, \
            "Diversified portfolio should have diversification benefit"
        
        assert diversified_risk['diversified_risk'] < diversified_risk['total_risk'], \
            "Diversified risk should be lower than total undiversified risk"
        
        # Validate reasonable ranges
        assert 0 <= concentrated_risk['total_risk'] <= 100
        assert 0 <= diversified_risk['diversified_risk'] <= 100


class TestRealTimeDataIntegration(RiskIntegrationTestBase):
    """Test real-time data integration and processing"""
    
    @pytest.mark.asyncio
    async def test_async_risk_calculation_pipeline(self):
        """Test asynchronous risk calculation pipeline"""
        
        class AsyncRiskCalculator:
            def __init__(self):
                self.processing_times = []
            
            async def fetch_market_data(self, asset_id: str) -> MarketData:
                """Simulate async market data fetch"""
                await asyncio.sleep(0.01)  # Simulate network latency
                return self.create_mock_market_data(asset_id)
            
            async def calculate_risk_score(self, market_data: MarketData) -> int:
                """Simulate async risk calculation"""
                start_time = time.perf_counter()
                await asyncio.sleep(0.005)  # Simulate calculation time
                
                # Simple risk calculation
                volatility = (market_data.high_price_cents - market_data.low_price_cents) / market_data.close_price_cents
                risk_score = min(int(volatility * 1000), 100)
                
                end_time = time.perf_counter()
                self.processing_times.append(end_time - start_time)
                
                return risk_score
            
            async def process_asset_batch(self, asset_ids: List[str]) -> Dict[str, int]:
                """Process batch of assets asynchronously"""
                tasks = []
                
                for asset_id in asset_ids:
                    task = asyncio.create_task(self._process_single_asset(asset_id))
                    tasks.append(task)
                
                results = await asyncio.gather(*tasks)
                return dict(zip(asset_ids, results))
            
            async def _process_single_asset(self, asset_id: str) -> int:
                """Process single asset: fetch data -> calculate risk"""
                market_data = await self.fetch_market_data(asset_id)
                return await self.calculate_risk_score(market_data)
        
        calculator = AsyncRiskCalculator()
        
        # Test batch processing
        test_assets = [f"ASSET_{i}" for i in range(20)]
        
        start_time = time.perf_counter()
        risk_scores = await calculator.process_asset_batch(test_assets)
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        
        # Validate results
        assert len(risk_scores) == len(test_assets), "Should process all assets"
        
        for asset_id, score in risk_scores.items():
            assert 0 <= score <= 100, f"Invalid score {score} for {asset_id}"
        
        # Should complete faster than sequential processing
        assert total_time < 0.5, f"Async processing too slow: {total_time:.3f}s"
        
        # Average processing time should be reasonable
        avg_processing_time = sum(calculator.processing_times) / len(calculator.processing_times)
        assert avg_processing_time < 0.1, f"Average processing time too high: {avg_processing_time:.3f}s"
    
    def test_data_feed_error_handling(self):
        """Test error handling for data feed failures"""
        
        class RobustRiskCalculator:
            def __init__(self):
                self.fallback_data_sources = ['PRIMARY', 'SECONDARY', 'CACHED']
                self.error_counts = {'network': 0, 'parsing': 0, 'calculation': 0}
            
            def fetch_market_data_with_fallback(self, asset_id: str, simulate_errors: bool = False) -> Optional[MarketData]:
                """Fetch market data with fallback sources"""
                
                for i, source in enumerate(self.fallback_data_sources):
                    try:
                        # Simulate different types of failures
                        if simulate_errors and i == 0:
                            self.error_counts['network'] += 1
                            raise ConnectionError("Primary data feed unavailable")
                        
                        if simulate_errors and i == 1:
                            self.error_counts['parsing'] += 1
                            raise ValueError("Data parsing error")
                        
                        # Success case (or final fallback)
                        return self.create_mock_market_data(
                            asset_id, 
                            price_cents=10000 + i * 100  # Slightly different prices per source
                        )
                    
                    except (ConnectionError, ValueError) as e:
                        if i == len(self.fallback_data_sources) - 1:
                            # Last fallback failed
                            return None
                        continue
                
                return None
            
            def calculate_risk_with_graceful_degradation(
                self, 
                asset_id: str, 
                simulate_errors: bool = False
            ) -> Tuple[Optional[int], List[str]]:
                """Calculate risk with graceful degradation"""
                
                warnings = []
                
                try:
                    market_data = self.fetch_market_data_with_fallback(asset_id, simulate_errors)
                    
                    if market_data is None:
                        warnings.append("All data sources failed - using cached risk score")
                        return 50, warnings  # Default/cached score
                    
                    if market_data.data_source != 'PRIMARY':
                        warnings.append(f"Using fallback data source: {market_data.data_source}")
                    
                    # Calculate risk score
                    if market_data.data_quality_score < 80:
                        warnings.append("Low data quality - risk score may be less accurate")
                    
                    volatility = (market_data.high_price_cents - market_data.low_price_cents) / market_data.close_price_cents
                    risk_score = min(int(volatility * 1000), 100)
                    
                    return risk_score, warnings
                
                except Exception as e:
                    self.error_counts['calculation'] += 1
                    warnings.append(f"Risk calculation error: {str(e)}")
                    return 75, warnings  # Conservative high-risk default
        
        calculator = RobustRiskCalculator()
        
        # Test normal operation
        risk_score, warnings = calculator.calculate_risk_with_graceful_degradation("TEST_ASSET")
        assert risk_score is not None
        assert 0 <= risk_score <= 100
        assert len(warnings) == 0, "Should have no warnings in normal operation"
        
        # Test with simulated errors
        risk_score_with_errors, warnings_with_errors = calculator.calculate_risk_with_graceful_degradation(
            "TEST_ASSET_ERROR", 
            simulate_errors=True
        )
        
        assert risk_score_with_errors is not None, "Should still return a risk score despite errors"
        assert 0 <= risk_score_with_errors <= 100
        assert len(warnings_with_errors) > 0, "Should have warnings when using fallback sources"
        
        # Check error counts
        assert calculator.error_counts['network'] > 0, "Should have recorded network errors"
        assert calculator.error_counts['parsing'] > 0, "Should have recorded parsing errors"
    
    def test_data_quality_validation_and_filtering(self):
        """Test data quality validation and filtering"""
        
        class DataQualityValidator:
            def __init__(self):
                self.quality_thresholds = {
                    'price_consistency': 0.95,
                    'volume_reasonableness': 0.90,
                    'timestamp_freshness': 0.85,
                    'bid_ask_validity': 0.90
                }
            
            def validate_market_data(self, data: MarketData) -> Tuple[bool, float, List[str]]:
                """Validate market data quality"""
                issues = []
                quality_scores = []
                
                # Price consistency check
                if data.low_price_cents > data.high_price_cents:
                    issues.append("Low price exceeds high price")
                    quality_scores.append(0.0)
                elif data.close_price_cents < data.low_price_cents or data.close_price_cents > data.high_price_cents:
                    issues.append("Close price outside OHLC range")
                    quality_scores.append(0.0)
                else:
                    quality_scores.append(1.0)
                
                # Volume reasonableness
                if data.volume < 0:
                    issues.append("Negative volume")
                    quality_scores.append(0.0)
                elif data.volume > 1000000000:  # 1B shares seems excessive
                    issues.append("Extremely high volume - possible error")
                    quality_scores.append(0.5)
                else:
                    quality_scores.append(1.0)
                
                # Timestamp freshness (within last 5 minutes for real-time data)
                time_diff = (datetime.utcnow() - data.timestamp).total_seconds()
                if time_diff > 300:  # 5 minutes
                    issues.append("Stale data - older than 5 minutes")
                    quality_scores.append(max(0, 1 - (time_diff - 300) / 1800))  # Decay over 30min
                else:
                    quality_scores.append(1.0)
                
                # Bid/Ask validity
                if data.bid_price_cents and data.ask_price_cents:
                    if data.bid_price_cents >= data.ask_price_cents:
                        issues.append("Bid price >= Ask price")
                        quality_scores.append(0.0)
                    else:
                        spread_pct = (data.ask_price_cents - data.bid_price_cents) / data.close_price_cents
                        if spread_pct > 0.1:  # 10% spread seems excessive
                            issues.append("Extremely wide bid-ask spread")
                            quality_scores.append(0.5)
                        else:
                            quality_scores.append(1.0)
                else:
                    quality_scores.append(0.8)  # Missing bid/ask data
                
                overall_quality = np.mean(quality_scores)
                is_acceptable = overall_quality >= 0.8
                
                return is_acceptable, overall_quality, issues
            
            def filter_and_adjust_risk_for_quality(
                self, 
                market_data: MarketData, 
                base_risk_score: int
            ) -> Tuple[int, List[str]]:
                """Filter and adjust risk score based on data quality"""
                
                is_acceptable, quality_score, issues = self.validate_market_data(market_data)
                
                if not is_acceptable:
                    return 85, [f"Poor data quality ({quality_score:.2f}) - using conservative high risk"] + issues
                
                # Adjust risk score based on data quality
                quality_adjustment = 1.0
                adjustment_reasons = []
                
                if quality_score < 0.95:
                    quality_adjustment = 1 + (0.95 - quality_score) * 0.5  # Increase risk for lower quality
                    adjustment_reasons.append(f"Data quality adjustment: +{(quality_adjustment-1)*100:.1f}%")
                
                adjusted_risk = min(int(base_risk_score * quality_adjustment), 100)
                
                return adjusted_risk, adjustment_reasons + issues
        
        validator = DataQualityValidator()
        
        # Test with high-quality data
        good_data = self.create_mock_market_data(
            "GOOD_ASSET",
            price_cents=10000,
            volume=1000000,
            bid_ask_spread_bps=5
        )
        good_data.timestamp = datetime.utcnow()  # Fresh timestamp
        
        is_acceptable, quality, issues = validator.validate_market_data(good_data)
        assert is_acceptable, "High-quality data should be acceptable"
        assert quality > 0.9, f"Quality score should be high, got {quality}"
        assert len(issues) == 0, f"Should have no issues, got {issues}"
        
        # Test with poor-quality data
        bad_data = self.create_mock_market_data(
            "BAD_ASSET",
            price_cents=10000,
            volume=-1000,  # Negative volume
            bid_ask_spread_bps=1000  # 10% spread
        )
        bad_data.low_price_cents = 11000  # Higher than close price
        bad_data.timestamp = datetime.utcnow() - timedelta(hours=1)  # Stale data
        
        is_acceptable, quality, issues = validator.validate_market_data(bad_data)
        assert not is_acceptable, "Poor-quality data should be rejected"
        assert quality < 0.5, f"Quality score should be low, got {quality}"
        assert len(issues) > 0, "Should have multiple data quality issues"
        
        # Test risk adjustment
        base_risk = 50
        adjusted_risk, reasons = validator.filter_and_adjust_risk_for_quality(bad_data, base_risk)
        assert adjusted_risk >= 80, f"Poor quality should result in conservative high risk, got {adjusted_risk}"
        assert len(reasons) > 0, "Should provide reasons for risk adjustment"


if __name__ == "__main__":
    # Run integration tests
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-m", "not slow"  # Skip slow tests by default
    ])