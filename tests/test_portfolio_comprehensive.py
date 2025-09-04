"""
Comprehensive Portfolio Optimization Tests
Tests for portfolio optimization, allocation, and risk management services
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, AsyncMock
from decimal import Decimal

from app.portfolio.optimization import PortfolioOptimizer, OptimizationMethod, OptimizationResult
from app.portfolio.allocation_service import PortfolioAllocationService
from app.models.user import User, RiskTolerance
from app.models.signal import SignalCreate, SignalType, AssetClass, TimeFrame


class TestPortfolioOptimization:
    """Test portfolio optimization algorithms"""
    
    @pytest.fixture
    def optimizer(self):
        """Create optimizer instance"""
        return PortfolioOptimizer()
    
    @pytest.fixture
    def sample_signals(self):
        """Create sample trading signals"""
        return [
            SignalCreate(
                symbol="AAPL",
                signal_type=SignalType.BUY,
                confidence=0.8,
                target_price=150.0,
                asset_class=AssetClass.EQUITY,
                timeframe=TimeFrame.DAILY
            ),
            SignalCreate(
                symbol="GOOGL",
                signal_type=SignalType.BUY,
                confidence=0.7,
                target_price=2500.0,
                asset_class=AssetClass.EQUITY,
                timeframe=TimeFrame.DAILY
            ),
            SignalCreate(
                symbol="TSLA",
                signal_type=SignalType.SELL,
                confidence=0.6,
                target_price=200.0,
                asset_class=AssetClass.EQUITY,
                timeframe=TimeFrame.DAILY
            )
        ]
    
    @pytest.fixture
    def sample_user(self):
        """Create sample user"""
        return User(
            id=1,
            username="test_user",
            risk_tolerance=RiskTolerance.MODERATE,
            investment_amount=Decimal("100000.00")
        )
    
    @pytest.fixture
    def returns_data(self):
        """Create sample returns data"""
        dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
        np.random.seed(42)
        
        returns = pd.DataFrame({
            'AAPL': np.random.normal(0.001, 0.02, len(dates)),
            'GOOGL': np.random.normal(0.0008, 0.025, len(dates)),
            'TSLA': np.random.normal(0.002, 0.04, len(dates))
        }, index=dates)
        
        return returns
    
    def test_mean_variance_optimization(self, optimizer, returns_data):
        """Test mean-variance optimization"""
        result = optimizer.optimize_portfolio(
            returns_data=returns_data,
            method=OptimizationMethod.MEAN_VARIANCE,
            target_return=0.12
        )
        
        assert isinstance(result, OptimizationResult)
        assert len(result.weights) == len(returns_data.columns)
        assert abs(sum(result.weights.values()) - 1.0) < 0.001
        assert all(w >= 0 for w in result.weights.values())
        assert result.expected_return > 0
        assert result.volatility > 0
        assert result.sharpe_ratio is not None
    
    def test_black_litterman_optimization(self, optimizer, returns_data):
        """Test Black-Litterman optimization"""
        # Create market views
        views = {
            'AAPL': 0.15,  # Expected 15% return
            'GOOGL': 0.12  # Expected 12% return
        }
        
        result = optimizer.optimize_portfolio(
            returns_data=returns_data,
            method=OptimizationMethod.BLACK_LITTERMAN,
            views=views
        )
        
        assert isinstance(result, OptimizationResult)
        assert len(result.weights) == len(returns_data.columns)
        assert abs(sum(result.weights.values()) - 1.0) < 0.001
        assert result.expected_return > 0
        assert result.volatility > 0
    
    def test_risk_parity_optimization(self, optimizer, returns_data):
        """Test risk parity optimization"""
        result = optimizer.optimize_portfolio(
            returns_data=returns_data,
            method=OptimizationMethod.RISK_PARITY
        )
        
        assert isinstance(result, OptimizationResult)
        assert len(result.weights) == len(returns_data.columns)
        assert abs(sum(result.weights.values()) - 1.0) < 0.001
        assert all(w >= 0 for w in result.weights.values())
        
        # Risk parity should have relatively balanced risk contributions
        risk_contributions = result.risk_contributions
        assert all(rc > 0 for rc in risk_contributions.values())
    
    def test_kelly_criterion_optimization(self, optimizer, returns_data):
        """Test Kelly criterion optimization"""
        result = optimizer.optimize_portfolio(
            returns_data=returns_data,
            method=OptimizationMethod.KELLY_CRITERION
        )
        
        assert isinstance(result, OptimizationResult)
        assert len(result.weights) == len(returns_data.columns)
        assert abs(sum(result.weights.values()) - 1.0) < 0.001
        # Kelly can have negative weights for shorting
        assert result.expected_return is not None
        assert result.volatility > 0
    
    def test_optimization_with_constraints(self, optimizer, returns_data):
        """Test optimization with constraints"""
        constraints = {
            'max_weight': 0.4,
            'min_weight': 0.1,
            'sector_limits': {'TECH': 0.6}
        }
        
        result = optimizer.optimize_portfolio(
            returns_data=returns_data,
            method=OptimizationMethod.MEAN_VARIANCE,
            constraints=constraints
        )
        
        assert isinstance(result, OptimizationResult)
        assert all(w <= 0.4 for w in result.weights.values())
        assert all(w >= 0.1 for w in result.weights.values())
    
    def test_optimization_error_handling(self, optimizer):
        """Test optimization error handling"""
        # Empty returns data
        empty_data = pd.DataFrame()
        
        result = optimizer.optimize_portfolio(
            returns_data=empty_data,
            method=OptimizationMethod.MEAN_VARIANCE
        )
        
        assert result is None or len(result.weights) == 0


class TestPortfolioAllocationService:
    """Test portfolio allocation service"""
    
    @pytest.fixture
    def allocation_service(self):
        """Create allocation service instance"""
        return PortfolioAllocationService()
    
    @pytest.fixture
    def mock_optimizer(self):
        """Create mock optimizer"""
        optimizer = Mock()
        optimizer.optimize_portfolio.return_value = OptimizationResult(
            weights={'AAPL': 0.4, 'GOOGL': 0.3, 'TSLA': 0.3},
            expected_return=0.12,
            volatility=0.15,
            sharpe_ratio=0.8,
            method=OptimizationMethod.MEAN_VARIANCE
        )
        return optimizer
    
    @pytest.mark.asyncio
    async def test_service_initialization(self, allocation_service):
        """Test service initialization"""
        assert not allocation_service.is_running
        assert len(allocation_service.user_portfolios) == 0
        assert allocation_service.rebalance_queue is not None
    
    @pytest.mark.asyncio
    async def test_portfolio_creation(self, allocation_service, sample_user, sample_signals):
        """Test portfolio creation for user"""
        with patch.object(allocation_service, 'optimizer') as mock_optimizer:
            mock_optimizer.optimize_portfolio.return_value = OptimizationResult(
                weights={'AAPL': 0.5, 'GOOGL': 0.3, 'TSLA': 0.2},
                expected_return=0.12,
                volatility=0.15,
                sharpe_ratio=0.8,
                method=OptimizationMethod.MEAN_VARIANCE
            )
            
            allocation = await allocation_service.create_portfolio_allocation(
                user=sample_user,
                signals=sample_signals
            )
            
            assert allocation is not None
            assert allocation.user_id == sample_user.id
            assert len(allocation.positions) > 0
            assert allocation.total_allocation == 1.0
    
    @pytest.mark.asyncio
    async def test_portfolio_rebalancing(self, allocation_service, sample_user):
        """Test portfolio rebalancing"""
        # Mock existing portfolio
        allocation_service.user_portfolios[sample_user.id] = {
            'weights': {'AAPL': 0.4, 'GOOGL': 0.6},
            'last_rebalance': datetime.now(timezone.utc) - timedelta(days=1),
            'performance_metrics': {}
        }
        
        with patch.object(allocation_service, 'optimizer') as mock_optimizer:
            mock_optimizer.optimize_portfolio.return_value = OptimizationResult(
                weights={'AAPL': 0.5, 'GOOGL': 0.5},
                expected_return=0.12,
                volatility=0.15,
                sharpe_ratio=0.8,
                method=OptimizationMethod.MEAN_VARIANCE
            )
            
            rebalanced = await allocation_service.rebalance_portfolio(sample_user.id)
            
            assert rebalanced
            assert sample_user.id in allocation_service.user_portfolios
    
    @pytest.mark.asyncio
    async def test_risk_based_allocation(self, allocation_service, sample_user, sample_signals):
        """Test risk-based portfolio allocation"""
        # Conservative user
        conservative_user = User(
            id=2,
            username="conservative_user",
            risk_tolerance=RiskTolerance.CONSERVATIVE,
            investment_amount=Decimal("50000.00")
        )
        
        with patch.object(allocation_service, 'optimizer') as mock_optimizer:
            def mock_optimize(returns_data, method, **kwargs):
                # Conservative allocation should have lower volatility
                if method == OptimizationMethod.RISK_PARITY:
                    return OptimizationResult(
                        weights={'AAPL': 0.33, 'GOOGL': 0.33, 'TSLA': 0.34},
                        expected_return=0.08,
                        volatility=0.10,
                        sharpe_ratio=0.8,
                        method=method
                    )
                return OptimizationResult(
                    weights={'AAPL': 0.4, 'GOOGL': 0.6},
                    expected_return=0.10,
                    volatility=0.12,
                    sharpe_ratio=0.83,
                    method=method
                )
            
            mock_optimizer.optimize_portfolio.side_effect = mock_optimize
            
            allocation = await allocation_service.create_portfolio_allocation(
                user=conservative_user,
                signals=sample_signals
            )
            
            assert allocation.expected_volatility <= 0.15  # Conservative limit
    
    def test_performance_tracking(self, allocation_service, sample_user):
        """Test portfolio performance tracking"""
        # Add mock portfolio
        allocation_service.user_portfolios[sample_user.id] = {
            'weights': {'AAPL': 0.5, 'GOOGL': 0.5},
            'initial_value': 100000.0,
            'current_value': 105000.0,
            'last_updated': datetime.now(timezone.utc)
        }
        
        stats = allocation_service.get_portfolio_performance(sample_user.id)
        
        assert stats is not None
        assert 'total_return' in stats
        assert 'current_value' in stats
        assert 'positions' in stats


class TestPortfolioIntegration:
    """Integration tests for portfolio system"""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_end_to_end_portfolio_workflow(self):
        """Test complete portfolio workflow"""
        # Create services
        allocation_service = PortfolioAllocationService()
        
        # Mock dependencies
        sample_user = User(
            id=1,
            username="integration_test",
            risk_tolerance=RiskTolerance.MODERATE,
            investment_amount=Decimal("100000.00")
        )
        
        sample_signals = [
            SignalCreate(
                symbol="AAPL",
                signal_type=SignalType.BUY,
                confidence=0.8,
                target_price=150.0,
                asset_class=AssetClass.EQUITY,
                timeframe=TimeFrame.DAILY
            )
        ]
        
        with patch.object(allocation_service, 'optimizer') as mock_optimizer:
            mock_optimizer.optimize_portfolio.return_value = OptimizationResult(
                weights={'AAPL': 1.0},
                expected_return=0.12,
                volatility=0.20,
                sharpe_ratio=0.6,
                method=OptimizationMethod.MEAN_VARIANCE
            )
            
            # Create allocation
            allocation = await allocation_service.create_portfolio_allocation(
                user=sample_user,
                signals=sample_signals
            )
            
            assert allocation is not None
            
            # Check performance
            stats = allocation_service.get_portfolio_performance(sample_user.id)
            assert stats is not None
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_kafka_integration(self):
        """Test Kafka integration for portfolio updates"""
        allocation_service = PortfolioAllocationService()
        
        # Mock Kafka producer
        with patch('app.portfolio.allocation_service.get_producer') as mock_producer:
            mock_producer.return_value = AsyncMock()
            
            # Mock portfolio update
            portfolio_update = {
                "user_id": 1,
                "action": "rebalance",
                "weights": {"AAPL": 0.6, "GOOGL": 0.4},
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            await allocation_service._publish_portfolio_update(portfolio_update)
            
            # Verify Kafka message was sent
            mock_producer.return_value.send.assert_called_once()


class TestPortfolioPerformance:
    """Test portfolio performance calculations"""
    
    @pytest.mark.performance
    def test_optimization_performance(self):
        """Test optimization algorithm performance"""
        optimizer = PortfolioOptimizer()
        
        # Create large dataset
        dates = pd.date_range('2020-01-01', '2024-01-01', freq='D')
        n_assets = 50
        
        # Generate synthetic returns
        np.random.seed(42)
        returns_data = pd.DataFrame(
            np.random.multivariate_normal(
                mean=np.random.normal(0.0005, 0.001, n_assets),
                cov=np.random.rand(n_assets, n_assets) * 0.0001,
                size=len(dates)
            ),
            index=dates,
            columns=[f"ASSET_{i}" for i in range(n_assets)]
        )
        
        start_time = datetime.now()
        result = optimizer.optimize_portfolio(
            returns_data=returns_data,
            method=OptimizationMethod.MEAN_VARIANCE
        )
        optimization_time = (datetime.now() - start_time).total_seconds()
        
        assert result is not None
        assert optimization_time < 60  # Should complete within 1 minute
        assert len(result.weights) == n_assets
    
    @pytest.mark.performance
    def test_memory_usage(self):
        """Test memory usage during portfolio operations"""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create multiple optimizers and portfolios
        optimizers = []
        for i in range(10):
            optimizer = PortfolioOptimizer()
            optimizers.append(optimizer)
        
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = current_memory - initial_memory
        
        # Clean up
        del optimizers
        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        assert memory_increase < 200  # Less than 200MB increase
        assert final_memory - initial_memory < 50  # Most memory cleaned up


class TestPortfolioErrorHandling:
    """Test portfolio system error handling"""
    
    def test_optimization_with_invalid_data(self):
        """Test optimization with invalid data"""
        optimizer = PortfolioOptimizer()
        
        # Invalid returns data (NaN values)
        invalid_data = pd.DataFrame({
            'AAPL': [np.nan, np.nan, np.nan],
            'GOOGL': [0.01, np.nan, -0.02]
        })
        
        result = optimizer.optimize_portfolio(
            returns_data=invalid_data,
            method=OptimizationMethod.MEAN_VARIANCE
        )
        
        # Should handle gracefully
        assert result is None or len(result.weights) == 0
    
    @pytest.mark.asyncio
    async def test_allocation_with_no_signals(self):
        """Test allocation with no signals"""
        allocation_service = PortfolioAllocationService()
        
        sample_user = User(
            id=1,
            username="test_user",
            risk_tolerance=RiskTolerance.MODERATE,
            investment_amount=Decimal("100000.00")
        )
        
        allocation = await allocation_service.create_portfolio_allocation(
            user=sample_user,
            signals=[]
        )
        
        # Should handle gracefully
        assert allocation is None or len(allocation.positions) == 0
    
    def test_invalid_optimization_method(self):
        """Test invalid optimization method handling"""
        optimizer = PortfolioOptimizer()
        
        returns_data = pd.DataFrame({
            'AAPL': [0.01, -0.005, 0.02],
            'GOOGL': [-0.01, 0.015, -0.008]
        })
        
        with pytest.raises((ValueError, AttributeError)):
            optimizer.optimize_portfolio(
                returns_data=returns_data,
                method="INVALID_METHOD"
            )


if __name__ == "__main__":
    # Run portfolio tests
    pytest.main([__file__, "-v", "--tb=short"])