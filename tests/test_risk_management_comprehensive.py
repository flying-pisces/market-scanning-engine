"""
Comprehensive Risk Management Tests
Tests for VaR calculations, stress testing, and risk monitoring
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, AsyncMock
from decimal import Decimal

from app.risk.management import RiskManager, VaRMethod, StressTest, RiskMetrics
from app.risk.monitoring import RiskMonitoringService
from app.models.user import User, RiskTolerance
from app.models.portfolio import Portfolio, Position


class TestRiskManager:
    """Test risk management calculations"""
    
    @pytest.fixture
    def risk_manager(self):
        """Create risk manager instance"""
        return RiskManager()
    
    @pytest.fixture
    def sample_returns(self):
        """Create sample return data"""
        dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
        np.random.seed(42)
        
        returns = pd.DataFrame({
            'AAPL': np.random.normal(0.001, 0.02, len(dates)),
            'GOOGL': np.random.normal(0.0008, 0.025, len(dates)),
            'TSLA': np.random.normal(0.002, 0.04, len(dates)),
            'SPY': np.random.normal(0.0005, 0.015, len(dates))
        }, index=dates)
        
        return returns
    
    @pytest.fixture
    def sample_portfolio(self):
        """Create sample portfolio"""
        positions = [
            Position(
                symbol="AAPL",
                quantity=100,
                average_price=150.0,
                current_price=155.0,
                market_value=15500.0
            ),
            Position(
                symbol="GOOGL",
                quantity=10,
                average_price=2500.0,
                current_price=2600.0,
                market_value=26000.0
            ),
            Position(
                symbol="TSLA",
                quantity=50,
                average_price=200.0,
                current_price=210.0,
                market_value=10500.0
            )
        ]
        
        return Portfolio(
            user_id=1,
            positions=positions,
            total_value=52000.0,
            cash_balance=48000.0
        )
    
    def test_historical_var_calculation(self, risk_manager, sample_returns, sample_portfolio):
        """Test historical VaR calculation"""
        var_result = risk_manager.calculate_var(
            portfolio=sample_portfolio,
            returns_data=sample_returns,
            confidence_level=0.95,
            method=VaRMethod.HISTORICAL,
            time_horizon=1
        )
        
        assert var_result is not None
        assert var_result.var_value > 0
        assert var_result.confidence_level == 0.95
        assert var_result.method == VaRMethod.HISTORICAL
        assert var_result.time_horizon == 1
        assert var_result.currency == "USD"
    
    def test_parametric_var_calculation(self, risk_manager, sample_returns, sample_portfolio):
        """Test parametric VaR calculation"""
        var_result = risk_manager.calculate_var(
            portfolio=sample_portfolio,
            returns_data=sample_returns,
            confidence_level=0.99,
            method=VaRMethod.PARAMETRIC,
            time_horizon=1
        )
        
        assert var_result is not None
        assert var_result.var_value > 0
        assert var_result.confidence_level == 0.99
        assert var_result.method == VaRMethod.PARAMETRIC
        assert var_result.expected_shortfall > var_result.var_value
    
    def test_monte_carlo_var_calculation(self, risk_manager, sample_returns, sample_portfolio):
        """Test Monte Carlo VaR calculation"""
        var_result = risk_manager.calculate_var(
            portfolio=sample_portfolio,
            returns_data=sample_returns,
            confidence_level=0.95,
            method=VaRMethod.MONTE_CARLO,
            time_horizon=10,
            simulations=1000
        )
        
        assert var_result is not None
        assert var_result.var_value > 0
        assert var_result.confidence_level == 0.95
        assert var_result.method == VaRMethod.MONTE_CARLO
        assert var_result.time_horizon == 10
        assert var_result.simulations == 1000
    
    def test_conditional_var_calculation(self, risk_manager, sample_returns, sample_portfolio):
        """Test Conditional VaR (Expected Shortfall) calculation"""
        cvar_result = risk_manager.calculate_conditional_var(
            portfolio=sample_portfolio,
            returns_data=sample_returns,
            confidence_level=0.95,
            method=VaRMethod.HISTORICAL
        )
        
        assert cvar_result is not None
        assert cvar_result.cvar_value > 0
        assert cvar_result.confidence_level == 0.95
        # CVaR should be higher than VaR
        assert hasattr(cvar_result, 'var_value')
        if cvar_result.var_value:
            assert cvar_result.cvar_value >= cvar_result.var_value
    
    def test_stress_testing(self, risk_manager, sample_portfolio):
        """Test portfolio stress testing"""
        # Define stress scenarios
        stress_scenarios = [
            StressTest(
                name="Market Crash",
                scenario_type="historical",
                shocks={'AAPL': -0.20, 'GOOGL': -0.25, 'TSLA': -0.30, 'SPY': -0.15}
            ),
            StressTest(
                name="Tech Selloff",
                scenario_type="hypothetical",
                shocks={'AAPL': -0.15, 'GOOGL': -0.18, 'TSLA': -0.25, 'SPY': -0.08}
            ),
            StressTest(
                name="Interest Rate Shock",
                scenario_type="factor_based",
                shocks={'AAPL': -0.10, 'GOOGL': -0.12, 'TSLA': -0.08, 'SPY': -0.06}
            )
        ]
        
        stress_results = risk_manager.run_stress_tests(
            portfolio=sample_portfolio,
            stress_scenarios=stress_scenarios
        )
        
        assert len(stress_results) == 3
        
        for result in stress_results:
            assert result.scenario_name in ["Market Crash", "Tech Selloff", "Interest Rate Shock"]
            assert result.portfolio_loss < 0  # Should show losses
            assert result.portfolio_value_after > 0
            assert len(result.position_impacts) > 0
    
    def test_correlation_analysis(self, risk_manager, sample_returns):
        """Test portfolio correlation analysis"""
        correlation_matrix = risk_manager.calculate_correlation_matrix(sample_returns)
        
        assert isinstance(correlation_matrix, pd.DataFrame)
        assert correlation_matrix.shape[0] == correlation_matrix.shape[1]
        assert all(correlation_matrix.columns == sample_returns.columns)
        
        # Diagonal should be 1.0 (perfect self-correlation)
        for symbol in correlation_matrix.columns:
            assert abs(correlation_matrix.loc[symbol, symbol] - 1.0) < 0.001
        
        # Matrix should be symmetric
        for i in correlation_matrix.index:
            for j in correlation_matrix.columns:
                assert abs(correlation_matrix.loc[i, j] - correlation_matrix.loc[j, i]) < 0.001
    
    def test_beta_calculation(self, risk_manager, sample_returns):
        """Test portfolio beta calculation"""
        # Calculate beta relative to SPY
        betas = risk_manager.calculate_portfolio_beta(
            portfolio_returns=sample_returns[['AAPL', 'GOOGL', 'TSLA']],
            market_returns=sample_returns['SPY'],
            weights={'AAPL': 0.4, 'GOOGL': 0.4, 'TSLA': 0.2}
        )
        
        assert 'portfolio_beta' in betas
        assert 'individual_betas' in betas
        assert len(betas['individual_betas']) == 3
        assert all(isinstance(beta, float) for beta in betas['individual_betas'].values())
    
    def test_drawdown_analysis(self, risk_manager):
        """Test maximum drawdown calculation"""
        # Create sample portfolio values with drawdowns
        portfolio_values = pd.Series([
            100000, 105000, 103000, 108000, 106000,
            104000, 95000, 92000, 98000, 102000,
            110000, 108000, 105000, 112000, 115000
        ], index=pd.date_range('2023-01-01', periods=15, freq='D'))
        
        drawdown_analysis = risk_manager.calculate_drawdown_metrics(portfolio_values)
        
        assert 'max_drawdown' in drawdown_analysis
        assert 'max_drawdown_duration' in drawdown_analysis
        assert 'recovery_time' in drawdown_analysis
        assert 'drawdown_series' in drawdown_analysis
        
        # Max drawdown should be negative
        assert drawdown_analysis['max_drawdown'] < 0
        assert drawdown_analysis['max_drawdown_duration'] >= 0
    
    def test_liquidity_risk_assessment(self, risk_manager, sample_portfolio):
        """Test liquidity risk assessment"""
        # Mock liquidity data
        liquidity_data = {
            'AAPL': {'avg_daily_volume': 50000000, 'bid_ask_spread': 0.01},
            'GOOGL': {'avg_daily_volume': 1500000, 'bid_ask_spread': 0.05},
            'TSLA': {'avg_daily_volume': 25000000, 'bid_ask_spread': 0.02}
        }
        
        liquidity_risk = risk_manager.assess_liquidity_risk(
            portfolio=sample_portfolio,
            liquidity_data=liquidity_data,
            liquidation_timeframe=5  # days
        )
        
        assert 'overall_liquidity_score' in liquidity_risk
        assert 'position_liquidity_scores' in liquidity_risk
        assert 'estimated_liquidation_cost' in liquidity_risk
        assert 'time_to_liquidate' in liquidity_risk
        
        assert 0 <= liquidity_risk['overall_liquidity_score'] <= 1
        assert liquidity_risk['estimated_liquidation_cost'] >= 0


class TestRiskMonitoringService:
    """Test risk monitoring service"""
    
    @pytest.fixture
    def monitoring_service(self):
        """Create risk monitoring service instance"""
        return RiskMonitoringService()
    
    @pytest.fixture
    def sample_user(self):
        """Create sample user"""
        return User(
            id=1,
            username="test_user",
            risk_tolerance=RiskTolerance.MODERATE,
            investment_amount=Decimal("100000.00")
        )
    
    @pytest.mark.asyncio
    async def test_service_initialization(self, monitoring_service):
        """Test service initialization"""
        assert not monitoring_service.is_running
        assert len(monitoring_service.monitored_portfolios) == 0
        assert monitoring_service.risk_limits is not None
    
    @pytest.mark.asyncio
    async def test_portfolio_monitoring_setup(self, monitoring_service, sample_user, sample_portfolio):
        """Test portfolio monitoring setup"""
        await monitoring_service.add_portfolio_monitoring(
            user=sample_user,
            portfolio=sample_portfolio,
            risk_limits={
                'max_var_95': 5000.0,
                'max_drawdown': -0.15,
                'max_position_concentration': 0.30,
                'min_liquidity_score': 0.7
            }
        )
        
        assert sample_user.id in monitoring_service.monitored_portfolios
        portfolio_config = monitoring_service.monitored_portfolios[sample_user.id]
        assert portfolio_config['risk_limits']['max_var_95'] == 5000.0
    
    @pytest.mark.asyncio
    async def test_risk_limit_violation_detection(self, monitoring_service, sample_user):
        """Test risk limit violation detection"""
        # Setup portfolio with risk limits
        monitoring_service.monitored_portfolios[sample_user.id] = {
            'risk_limits': {
                'max_var_95': 1000.0,  # Very low limit
                'max_drawdown': -0.05,  # Very strict drawdown
                'max_position_concentration': 0.20
            },
            'portfolio': sample_portfolio,
            'last_check': datetime.now(timezone.utc)
        }
        
        # Mock current risk metrics that violate limits
        current_metrics = RiskMetrics(
            var_95=2000.0,  # Exceeds limit
            current_drawdown=-0.08,  # Exceeds limit
            max_position_concentration=0.35,  # Exceeds limit
            portfolio_value=100000.0
        )
        
        violations = monitoring_service._check_risk_violations(
            user_id=sample_user.id,
            current_metrics=current_metrics
        )
        
        assert len(violations) == 3  # All three limits violated
        assert any(v['limit_type'] == 'max_var_95' for v in violations)
        assert any(v['limit_type'] == 'max_drawdown' for v in violations)
        assert any(v['limit_type'] == 'max_position_concentration' for v in violations)
    
    @pytest.mark.asyncio
    async def test_alert_generation(self, monitoring_service, sample_user):
        """Test risk alert generation"""
        violation = {
            'limit_type': 'max_var_95',
            'current_value': 2000.0,
            'limit_value': 1000.0,
            'severity': 'HIGH',
            'message': 'Portfolio VaR exceeds risk limit'
        }
        
        # Mock alert system
        with patch.object(monitoring_service, '_send_alert') as mock_alert:
            await monitoring_service._generate_alert(sample_user.id, violation)
            mock_alert.assert_called_once()
    
    def test_risk_score_calculation(self, monitoring_service, sample_portfolio):
        """Test overall risk score calculation"""
        risk_metrics = RiskMetrics(
            var_95=3000.0,
            expected_shortfall=4500.0,
            current_drawdown=-0.06,
            volatility=0.15,
            beta=1.2,
            max_position_concentration=0.25,
            liquidity_score=0.8,
            portfolio_value=100000.0
        )
        
        risk_score = monitoring_service.calculate_risk_score(
            portfolio=sample_portfolio,
            risk_metrics=risk_metrics,
            risk_tolerance=RiskTolerance.MODERATE
        )
        
        assert 0 <= risk_score <= 100
        assert isinstance(risk_score, (int, float))
    
    @pytest.mark.asyncio
    async def test_automated_rebalancing_trigger(self, monitoring_service, sample_user):
        """Test automated rebalancing trigger"""
        # Setup monitoring with rebalancing enabled
        monitoring_service.monitored_portfolios[sample_user.id] = {
            'auto_rebalancing': True,
            'rebalance_threshold': 0.05,  # 5% drift threshold
            'target_allocation': {'AAPL': 0.4, 'GOOGL': 0.3, 'TSLA': 0.3}
        }
        
        # Mock current allocation that exceeds threshold
        current_allocation = {'AAPL': 0.5, 'GOOGL': 0.3, 'TSLA': 0.2}  # AAPL drifted +10%
        
        with patch.object(monitoring_service, '_trigger_rebalancing') as mock_rebalance:
            should_rebalance = monitoring_service._check_rebalancing_needed(
                user_id=sample_user.id,
                current_allocation=current_allocation
            )
            
            assert should_rebalance
            if should_rebalance:
                await monitoring_service._trigger_rebalancing(sample_user.id)
                mock_rebalance.assert_called_once()
    
    def test_risk_metrics_aggregation(self, monitoring_service):
        """Test risk metrics aggregation across portfolios"""
        # Mock multiple portfolio risk metrics
        portfolio_metrics = {
            1: RiskMetrics(var_95=2000.0, volatility=0.15, portfolio_value=100000.0),
            2: RiskMetrics(var_95=1500.0, volatility=0.12, portfolio_value=75000.0),
            3: RiskMetrics(var_95=3000.0, volatility=0.20, portfolio_value=150000.0)
        }
        
        aggregated = monitoring_service.aggregate_risk_metrics(portfolio_metrics)
        
        assert 'total_var' in aggregated
        assert 'average_volatility' in aggregated
        assert 'total_portfolio_value' in aggregated
        assert 'portfolio_count' in aggregated
        
        assert aggregated['total_portfolio_value'] == 325000.0
        assert aggregated['portfolio_count'] == 3


class TestRiskIntegration:
    """Integration tests for risk management system"""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_end_to_end_risk_monitoring(self):
        """Test complete risk monitoring workflow"""
        risk_manager = RiskManager()
        monitoring_service = RiskMonitoringService()
        
        # Create test data
        sample_user = User(
            id=1,
            username="integration_test",
            risk_tolerance=RiskTolerance.MODERATE,
            investment_amount=Decimal("100000.00")
        )
        
        sample_portfolio = Portfolio(
            user_id=1,
            positions=[
                Position(
                    symbol="AAPL",
                    quantity=100,
                    average_price=150.0,
                    current_price=155.0,
                    market_value=15500.0
                )
            ],
            total_value=15500.0,
            cash_balance=84500.0
        )
        
        # Setup monitoring
        await monitoring_service.add_portfolio_monitoring(
            user=sample_user,
            portfolio=sample_portfolio,
            risk_limits={
                'max_var_95': 2000.0,
                'max_drawdown': -0.10,
                'max_position_concentration': 0.25
            }
        )
        
        # Mock market data for risk calculations
        returns_data = pd.DataFrame({
            'AAPL': np.random.normal(0.001, 0.02, 252)
        }, index=pd.date_range('2023-01-01', periods=252))
        
        # Calculate current risk metrics
        var_result = risk_manager.calculate_var(
            portfolio=sample_portfolio,
            returns_data=returns_data,
            confidence_level=0.95,
            method=VaRMethod.HISTORICAL
        )
        
        assert var_result is not None
        assert sample_user.id in monitoring_service.monitored_portfolios
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_kafka_integration(self):
        """Test Kafka integration for risk alerts"""
        monitoring_service = RiskMonitoringService()
        
        # Mock Kafka producer
        with patch('app.risk.monitoring.get_producer') as mock_producer:
            mock_producer.return_value = AsyncMock()
            
            # Mock risk alert
            risk_alert = {
                "user_id": 1,
                "alert_type": "RISK_LIMIT_VIOLATION",
                "severity": "HIGH",
                "message": "Portfolio VaR exceeds limit",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            await monitoring_service._publish_risk_alert(risk_alert)
            
            # Verify Kafka message was sent
            mock_producer.return_value.send.assert_called_once()


class TestRiskPerformance:
    """Test risk calculation performance"""
    
    @pytest.mark.performance
    def test_var_calculation_performance(self):
        """Test VaR calculation performance with large datasets"""
        risk_manager = RiskManager()
        
        # Create large returns dataset
        dates = pd.date_range('2018-01-01', '2024-01-01', freq='D')
        n_assets = 100
        
        np.random.seed(42)
        returns_data = pd.DataFrame(
            np.random.multivariate_normal(
                mean=np.random.normal(0.0005, 0.001, n_assets),
                cov=np.eye(n_assets) * 0.0001,  # Diagonal covariance matrix
                size=len(dates)
            ),
            index=dates,
            columns=[f"ASSET_{i}" for i in range(n_assets)]
        )
        
        # Create large portfolio
        positions = [
            Position(
                symbol=f"ASSET_{i}",
                quantity=100,
                average_price=100.0,
                current_price=105.0,
                market_value=10500.0
            ) for i in range(n_assets)
        ]
        
        large_portfolio = Portfolio(
            user_id=1,
            positions=positions,
            total_value=n_assets * 10500.0,
            cash_balance=0.0
        )
        
        start_time = datetime.now()
        var_result = risk_manager.calculate_var(
            portfolio=large_portfolio,
            returns_data=returns_data,
            confidence_level=0.95,
            method=VaRMethod.HISTORICAL
        )
        calculation_time = (datetime.now() - start_time).total_seconds()
        
        assert var_result is not None
        assert calculation_time < 30  # Should complete within 30 seconds
    
    @pytest.mark.performance
    def test_stress_test_performance(self):
        """Test stress testing performance"""
        risk_manager = RiskManager()
        
        # Create portfolio with many positions
        positions = [
            Position(
                symbol=f"STOCK_{i}",
                quantity=100,
                average_price=100.0,
                current_price=105.0,
                market_value=10500.0
            ) for i in range(50)
        ]
        
        portfolio = Portfolio(
            user_id=1,
            positions=positions,
            total_value=50 * 10500.0,
            cash_balance=0.0
        )
        
        # Create multiple stress scenarios
        stress_scenarios = [
            StressTest(
                name=f"Scenario_{i}",
                scenario_type="hypothetical",
                shocks={f"STOCK_{j}": np.random.normal(-0.1, 0.05) for j in range(50)}
            ) for i in range(10)
        ]
        
        start_time = datetime.now()
        stress_results = risk_manager.run_stress_tests(
            portfolio=portfolio,
            stress_scenarios=stress_scenarios
        )
        stress_time = (datetime.now() - start_time).total_seconds()
        
        assert len(stress_results) == 10
        assert stress_time < 10  # Should complete within 10 seconds


class TestRiskErrorHandling:
    """Test risk management error handling"""
    
    def test_var_with_insufficient_data(self):
        """Test VaR calculation with insufficient data"""
        risk_manager = RiskManager()
        
        # Very small dataset
        small_returns = pd.DataFrame({
            'AAPL': [0.01, -0.02, 0.015]
        })
        
        portfolio = Portfolio(
            user_id=1,
            positions=[Position(symbol="AAPL", quantity=100, average_price=100.0, 
                               current_price=105.0, market_value=10500.0)],
            total_value=10500.0,
            cash_balance=0.0
        )
        
        var_result = risk_manager.calculate_var(
            portfolio=portfolio,
            returns_data=small_returns,
            confidence_level=0.95,
            method=VaRMethod.HISTORICAL
        )
        
        # Should handle gracefully or return appropriate warning
        assert var_result is None or var_result.var_value >= 0
    
    def test_stress_test_with_missing_symbols(self):
        """Test stress testing with missing symbols"""
        risk_manager = RiskManager()
        
        portfolio = Portfolio(
            user_id=1,
            positions=[Position(symbol="AAPL", quantity=100, average_price=100.0,
                               current_price=105.0, market_value=10500.0)],
            total_value=10500.0,
            cash_balance=0.0
        )
        
        # Stress scenario with symbol not in portfolio
        stress_scenario = StressTest(
            name="Missing Symbol Test",
            scenario_type="hypothetical",
            shocks={'MISSING_STOCK': -0.20}  # Symbol not in portfolio
        )
        
        stress_results = risk_manager.run_stress_tests(
            portfolio=portfolio,
            stress_scenarios=[stress_scenario]
        )
        
        # Should handle gracefully
        assert len(stress_results) == 1
        result = stress_results[0]
        assert result.portfolio_loss == 0  # No impact since symbol not in portfolio
    
    def test_invalid_confidence_level(self):
        """Test VaR calculation with invalid confidence level"""
        risk_manager = RiskManager()
        
        returns_data = pd.DataFrame({
            'AAPL': np.random.normal(0.001, 0.02, 100)
        })
        
        portfolio = Portfolio(
            user_id=1,
            positions=[Position(symbol="AAPL", quantity=100, average_price=100.0,
                               current_price=105.0, market_value=10500.0)],
            total_value=10500.0,
            cash_balance=0.0
        )
        
        # Invalid confidence levels
        for invalid_confidence in [-0.5, 0.0, 1.0, 1.5]:
            with pytest.raises((ValueError, AssertionError)):
                risk_manager.calculate_var(
                    portfolio=portfolio,
                    returns_data=returns_data,
                    confidence_level=invalid_confidence,
                    method=VaRMethod.HISTORICAL
                )


if __name__ == "__main__":
    # Run risk management tests
    pytest.main([__file__, "-v", "--tb=short"])