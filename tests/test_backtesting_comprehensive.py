"""
Comprehensive Backtesting System Tests
Tests for backtesting engine, execution models, and strategy evaluation
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, AsyncMock
from decimal import Decimal

from app.backtesting.engine import BacktestEngine, ExecutionModel, BacktestResult, TradeResult
from app.backtesting.service import BacktestingService
from app.models.signal import SignalCreate, SignalType, AssetClass, TimeFrame
from app.models.user import User, RiskTolerance


class TestBacktestEngine:
    """Test backtesting engine functionality"""
    
    @pytest.fixture
    def backtest_engine(self):
        """Create backtest engine instance"""
        return BacktestEngine()
    
    @pytest.fixture
    def sample_signals(self):
        """Create sample trading signals"""
        base_time = datetime.now(timezone.utc) - timedelta(days=30)
        
        return [
            SignalCreate(
                symbol="AAPL",
                signal_type=SignalType.BUY,
                confidence=0.8,
                target_price=150.0,
                asset_class=AssetClass.EQUITY,
                timeframe=TimeFrame.DAILY,
                timestamp=base_time
            ),
            SignalCreate(
                symbol="AAPL",
                signal_type=SignalType.SELL,
                confidence=0.7,
                target_price=155.0,
                asset_class=AssetClass.EQUITY,
                timeframe=TimeFrame.DAILY,
                timestamp=base_time + timedelta(days=5)
            ),
            SignalCreate(
                symbol="GOOGL",
                signal_type=SignalType.BUY,
                confidence=0.75,
                target_price=2500.0,
                asset_class=AssetClass.EQUITY,
                timeframe=TimeFrame.DAILY,
                timestamp=base_time + timedelta(days=2)
            )
        ]
    
    @pytest.fixture
    def market_data(self):
        """Create sample market data"""
        dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
        np.random.seed(42)
        
        # Generate realistic OHLCV data
        returns = np.random.normal(0.0005, 0.02, len(dates))
        prices = 100 * np.cumprod(1 + returns)
        
        data = {}
        for symbol in ['AAPL', 'GOOGL', 'TSLA']:
            symbol_returns = np.random.normal(0.0005, 0.02, len(dates))
            symbol_prices = 100 * np.cumprod(1 + symbol_returns)
            
            data[symbol] = pd.DataFrame({
                'open': symbol_prices * np.random.uniform(0.99, 1.01, len(dates)),
                'high': symbol_prices * np.random.uniform(1.0, 1.03, len(dates)),
                'low': symbol_prices * np.random.uniform(0.97, 1.0, len(dates)),
                'close': symbol_prices,
                'volume': np.random.randint(100000, 10000000, len(dates)),
            }, index=dates)
        
        return data
    
    @pytest.mark.asyncio
    async def test_perfect_execution_backtest(self, backtest_engine, sample_signals, market_data):
        """Test backtest with perfect execution"""
        result = await backtest_engine.run_backtest(
            signals=sample_signals,
            market_data=market_data,
            initial_capital=100000.0,
            execution_model=ExecutionModel.PERFECT,
            strategy_name="Perfect Execution Test"
        )
        
        assert isinstance(result, BacktestResult)
        assert result.strategy_name == "Perfect Execution Test"
        assert result.initial_capital == 100000.0
        assert len(result.trades) > 0
        assert result.total_return is not None
        assert result.sharpe_ratio is not None
        assert result.max_drawdown is not None
    
    @pytest.mark.asyncio
    async def test_realistic_execution_backtest(self, backtest_engine, sample_signals, market_data):
        """Test backtest with realistic execution"""
        result = await backtest_engine.run_backtest(
            signals=sample_signals,
            market_data=market_data,
            initial_capital=100000.0,
            execution_model=ExecutionModel.REALISTIC,
            strategy_name="Realistic Execution Test",
            commission=5.0,
            slippage=0.001
        )
        
        assert isinstance(result, BacktestResult)
        assert result.total_commission > 0  # Should have commission costs
        assert len(result.trades) > 0
        
        # Realistic execution should have slightly worse performance than perfect
        for trade in result.trades:
            assert isinstance(trade, TradeResult)
            assert trade.commission >= 0
            assert trade.slippage_cost >= 0
    
    @pytest.mark.asyncio
    async def test_pessimistic_execution_backtest(self, backtest_engine, sample_signals, market_data):
        """Test backtest with pessimistic execution"""
        result = await backtest_engine.run_backtest(
            signals=sample_signals,
            market_data=market_data,
            initial_capital=100000.0,
            execution_model=ExecutionModel.PESSIMISTIC,
            strategy_name="Pessimistic Execution Test",
            commission=10.0,
            slippage=0.002
        )
        
        assert isinstance(result, BacktestResult)
        assert result.total_commission > 0
        assert len(result.trades) > 0
        
        # Pessimistic should have highest costs
        for trade in result.trades:
            assert trade.commission >= 0
            assert trade.slippage_cost >= 0
    
    def test_performance_metrics_calculation(self, backtest_engine):
        """Test performance metrics calculation"""
        # Mock portfolio values over time
        portfolio_values = pd.Series([
            100000, 105000, 103000, 108000, 106000, 
            112000, 109000, 115000, 118000, 116000
        ], index=pd.date_range('2023-01-01', periods=10, freq='D'))
        
        metrics = backtest_engine._calculate_performance_metrics(
            portfolio_values=portfolio_values,
            initial_capital=100000.0,
            trades=[]
        )
        
        assert 'total_return' in metrics
        assert 'annual_return' in metrics
        assert 'volatility' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'win_rate' in metrics
        
        # Verify calculations
        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
        assert abs(metrics['total_return'] - total_return) < 0.001
    
    def test_trade_execution_simulation(self, backtest_engine, market_data):
        """Test trade execution simulation"""
        signal = SignalCreate(
            symbol="AAPL",
            signal_type=SignalType.BUY,
            confidence=0.8,
            target_price=150.0,
            asset_class=AssetClass.EQUITY,
            timeframe=TimeFrame.DAILY
        )
        
        # Mock current market price
        current_price = 148.0
        
        trade_result = backtest_engine._execute_trade(
            signal=signal,
            current_price=current_price,
            available_capital=10000.0,
            execution_model=ExecutionModel.REALISTIC,
            commission=5.0,
            slippage=0.001
        )
        
        assert isinstance(trade_result, TradeResult)
        assert trade_result.symbol == "AAPL"
        assert trade_result.signal_type == SignalType.BUY
        assert trade_result.quantity > 0
        assert trade_result.execution_price > 0
        assert trade_result.commission == 5.0
        assert trade_result.slippage_cost >= 0
    
    def test_position_sizing(self, backtest_engine):
        """Test position sizing calculations"""
        # Equal weight position sizing
        position_size = backtest_engine._calculate_position_size(
            signal_confidence=0.8,
            available_capital=100000.0,
            target_price=150.0,
            sizing_method="equal_weight",
            max_positions=10
        )
        
        assert position_size == 10000.0  # 100k / 10 positions
        
        # Confidence-based sizing
        position_size = backtest_engine._calculate_position_size(
            signal_confidence=0.8,
            available_capital=100000.0,
            target_price=150.0,
            sizing_method="confidence_based",
            max_positions=10
        )
        
        assert position_size > 0
        assert position_size <= 100000.0


class TestBacktestingService:
    """Test backtesting service"""
    
    @pytest.fixture
    def backtesting_service(self):
        """Create backtesting service instance"""
        return BacktestingService()
    
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
    async def test_service_initialization(self, backtesting_service):
        """Test service initialization"""
        assert not backtesting_service.is_running
        assert len(backtesting_service.active_backtests) == 0
        assert backtesting_service.backtest_queue is not None
    
    @pytest.mark.asyncio
    async def test_backtest_job_creation(self, backtesting_service, sample_user, sample_signals):
        """Test backtest job creation"""
        job_id = await backtesting_service.create_backtest_job(
            user=sample_user,
            signals=sample_signals,
            strategy_name="Test Strategy",
            start_date=datetime.now() - timedelta(days=365),
            end_date=datetime.now(),
            initial_capital=100000.0
        )
        
        assert job_id is not None
        assert len(backtesting_service.backtest_queue) == 1
    
    @pytest.mark.asyncio
    async def test_strategy_comparison(self, backtesting_service):
        """Test strategy comparison functionality"""
        # Mock backtest results
        strategy1_result = BacktestResult(
            strategy_name="Strategy 1",
            total_return=0.15,
            annual_return=0.12,
            sharpe_ratio=1.2,
            max_drawdown=-0.08,
            win_rate=0.65,
            trades=[],
            portfolio_values=pd.Series([100000, 115000]),
            initial_capital=100000.0
        )
        
        strategy2_result = BacktestResult(
            strategy_name="Strategy 2",
            total_return=0.18,
            annual_return=0.14,
            sharpe_ratio=1.1,
            max_drawdown=-0.12,
            win_rate=0.62,
            trades=[],
            portfolio_values=pd.Series([100000, 118000]),
            initial_capital=100000.0
        )
        
        comparison = backtesting_service.compare_strategies([strategy1_result, strategy2_result])
        
        assert 'strategies' in comparison
        assert 'best_strategy' in comparison
        assert 'comparison_metrics' in comparison
        assert len(comparison['strategies']) == 2
    
    @pytest.mark.asyncio
    async def test_backtest_result_storage(self, backtesting_service, sample_user):
        """Test backtest result storage and retrieval"""
        # Mock backtest result
        mock_result = BacktestResult(
            strategy_name="Mock Strategy",
            total_return=0.10,
            annual_return=0.08,
            sharpe_ratio=0.9,
            max_drawdown=-0.05,
            win_rate=0.6,
            trades=[],
            portfolio_values=pd.Series([100000, 110000]),
            initial_capital=100000.0
        )
        
        # Store result
        result_id = await backtesting_service.store_backtest_result(
            user_id=sample_user.id,
            result=mock_result
        )
        
        assert result_id is not None
        
        # Retrieve result
        stored_result = await backtesting_service.get_backtest_result(result_id)
        assert stored_result is not None
        assert stored_result.strategy_name == "Mock Strategy"
    
    def test_performance_ranking(self, backtesting_service):
        """Test strategy performance ranking"""
        results = [
            BacktestResult(
                strategy_name="Strategy A",
                total_return=0.15,
                sharpe_ratio=1.2,
                max_drawdown=-0.08,
                win_rate=0.65,
                trades=[], portfolio_values=pd.Series([]), initial_capital=100000.0
            ),
            BacktestResult(
                strategy_name="Strategy B",
                total_return=0.12,
                sharpe_ratio=1.5,
                max_drawdown=-0.05,
                win_rate=0.7,
                trades=[], portfolio_values=pd.Series([]), initial_capital=100000.0
            ),
            BacktestResult(
                strategy_name="Strategy C",
                total_return=0.18,
                sharpe_ratio=0.9,
                max_drawdown=-0.15,
                win_rate=0.55,
                trades=[], portfolio_values=pd.Series([]), initial_capital=100000.0
            )
        ]
        
        ranked = backtesting_service.rank_strategies(results, metric="sharpe_ratio")
        
        assert len(ranked) == 3
        assert ranked[0].strategy_name == "Strategy B"  # Highest Sharpe ratio
        assert ranked[1].sharpe_ratio >= ranked[2].sharpe_ratio  # Properly sorted
    
    @pytest.mark.asyncio
    async def test_user_backtest_history(self, backtesting_service, sample_user):
        """Test user backtest history retrieval"""
        # Mock user backtest history
        backtesting_service.user_backtests[sample_user.id] = [
            {"strategy_name": "Strategy 1", "result_id": "123", "timestamp": datetime.now()},
            {"strategy_name": "Strategy 2", "result_id": "456", "timestamp": datetime.now()}
        ]
        
        history = backtesting_service.get_user_backtest_history(sample_user.id)
        
        assert len(history) == 2
        assert history[0]["strategy_name"] == "Strategy 1"
        assert history[1]["strategy_name"] == "Strategy 2"


class TestBacktestIntegration:
    """Integration tests for backtesting system"""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_end_to_end_backtest_workflow(self):
        """Test complete backtesting workflow"""
        # Create services
        backtesting_service = BacktestingService()
        backtest_engine = BacktestEngine()
        
        # Mock user and signals
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
        
        # Mock market data provider
        with patch('app.backtesting.service.BacktestingService._get_market_data') as mock_data:
            mock_data.return_value = {
                'AAPL': pd.DataFrame({
                    'close': [145, 148, 152, 150, 155],
                    'volume': [1000000] * 5
                }, index=pd.date_range('2023-01-01', periods=5))
            }
            
            # Create and run backtest
            job_id = await backtesting_service.create_backtest_job(
                user=sample_user,
                signals=sample_signals,
                strategy_name="Integration Test",
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 12, 31),
                initial_capital=100000.0
            )
            
            assert job_id is not None
            
            # Simulate backtest completion
            # In real implementation, this would be processed by background worker
            assert len(backtesting_service.backtest_queue) == 1
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_kafka_integration(self):
        """Test Kafka integration for backtest results"""
        backtesting_service = BacktestingService()
        
        # Mock Kafka producer
        with patch('app.backtesting.service.get_producer') as mock_producer:
            mock_producer.return_value = AsyncMock()
            
            # Mock backtest completion
            backtest_result = {
                "user_id": 1,
                "strategy_name": "Kafka Test Strategy",
                "total_return": 0.12,
                "sharpe_ratio": 1.1,
                "completed_at": datetime.now(timezone.utc).isoformat()
            }
            
            await backtesting_service._publish_backtest_completion(backtest_result)
            
            # Verify Kafka message was sent
            mock_producer.return_value.send.assert_called_once()


class TestBacktestPerformance:
    """Test backtesting performance"""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_large_backtest_performance(self):
        """Test performance with large dataset"""
        backtest_engine = BacktestEngine()
        
        # Create large signal dataset
        signals = []
        base_time = datetime(2020, 1, 1, tzinfo=timezone.utc)
        
        for i in range(1000):  # 1000 signals
            signals.append(SignalCreate(
                symbol=f"STOCK_{i % 10}",
                signal_type=SignalType.BUY if i % 2 == 0 else SignalType.SELL,
                confidence=0.5 + (i % 5) * 0.1,
                target_price=100 + i * 0.1,
                asset_class=AssetClass.EQUITY,
                timeframe=TimeFrame.DAILY,
                timestamp=base_time + timedelta(days=i // 10)
            ))
        
        # Create market data for multiple symbols
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        market_data = {}
        
        for i in range(10):
            symbol = f"STOCK_{i}"
            returns = np.random.normal(0.0005, 0.02, len(dates))
            prices = 100 * np.cumprod(1 + returns)
            
            market_data[symbol] = pd.DataFrame({
                'open': prices * 0.99,
                'high': prices * 1.01,
                'low': prices * 0.98,
                'close': prices,
                'volume': np.random.randint(100000, 1000000, len(dates))
            }, index=dates)
        
        start_time = datetime.now()
        result = await backtest_engine.run_backtest(
            signals=signals,
            market_data=market_data,
            initial_capital=1000000.0,
            execution_model=ExecutionModel.REALISTIC,
            strategy_name="Large Performance Test"
        )
        backtest_time = (datetime.now() - start_time).total_seconds()
        
        assert result is not None
        assert backtest_time < 300  # Should complete within 5 minutes
        assert len(result.trades) > 0
    
    @pytest.mark.performance
    def test_memory_usage_during_backtest(self):
        """Test memory usage during backtesting"""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create multiple backtest engines
        engines = []
        for i in range(5):
            engine = BacktestEngine()
            engines.append(engine)
        
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = current_memory - initial_memory
        
        # Clean up
        del engines
        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        assert memory_increase < 300  # Less than 300MB increase
        assert final_memory - initial_memory < 100  # Most memory cleaned up


class TestBacktestErrorHandling:
    """Test backtesting error handling"""
    
    @pytest.mark.asyncio
    async def test_backtest_with_missing_data(self):
        """Test backtest with missing market data"""
        backtest_engine = BacktestEngine()
        
        signals = [
            SignalCreate(
                symbol="MISSING_STOCK",
                signal_type=SignalType.BUY,
                confidence=0.8,
                target_price=100.0,
                asset_class=AssetClass.EQUITY,
                timeframe=TimeFrame.DAILY
            )
        ]
        
        # Empty market data
        market_data = {}
        
        result = await backtest_engine.run_backtest(
            signals=signals,
            market_data=market_data,
            initial_capital=100000.0,
            strategy_name="Missing Data Test"
        )
        
        # Should handle gracefully
        assert result is None or len(result.trades) == 0
    
    def test_invalid_execution_model(self):
        """Test invalid execution model handling"""
        backtest_engine = BacktestEngine()
        
        with pytest.raises((ValueError, AttributeError)):
            backtest_engine._execute_trade(
                signal=Mock(),
                current_price=100.0,
                available_capital=10000.0,
                execution_model="INVALID_MODEL",
                commission=5.0,
                slippage=0.001
            )
    
    @pytest.mark.asyncio
    async def test_backtest_with_extreme_parameters(self):
        """Test backtest with extreme parameters"""
        backtest_engine = BacktestEngine()
        
        signals = [
            SignalCreate(
                symbol="AAPL",
                signal_type=SignalType.BUY,
                confidence=0.8,
                target_price=1000000.0,  # Unrealistic price
                asset_class=AssetClass.EQUITY,
                timeframe=TimeFrame.DAILY
            )
        ]
        
        market_data = {
            'AAPL': pd.DataFrame({
                'close': [150],
                'volume': [1000000]
            }, index=[datetime.now()])
        }
        
        result = await backtest_engine.run_backtest(
            signals=signals,
            market_data=market_data,
            initial_capital=1000.0,  # Very small capital
            commission=1000.0,  # Very high commission
            strategy_name="Extreme Parameters Test"
        )
        
        # Should handle gracefully
        assert result is not None
        # With high commission and low capital, no trades should execute
        assert len([t for t in result.trades if t.quantity > 0]) == 0


if __name__ == "__main__":
    # Run backtesting tests
    pytest.main([__file__, "-v", "--tb=short"])