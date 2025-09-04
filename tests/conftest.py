"""
Pytest Configuration and Shared Fixtures
Global configuration and fixtures for the comprehensive test suite
"""

import pytest
import asyncio
import os
import tempfile
import shutil
from unittest.mock import Mock, AsyncMock
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

# Set test environment
os.environ['ENVIRONMENT'] = 'testing'

# Configure pytest markers
def pytest_configure(config):
    """Configure custom pytest markers"""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "external: mark test as requiring external services"
    )

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)

@pytest.fixture
def mock_database():
    """Mock database connection"""
    mock_db = AsyncMock()
    mock_db.execute.return_value = Mock()
    mock_db.fetch.return_value = []
    mock_db.fetchrow.return_value = None
    return mock_db

@pytest.fixture
def mock_kafka_producer():
    """Mock Kafka producer"""
    producer = AsyncMock()
    producer.send.return_value = Mock()
    producer.send.return_value.get = Mock(return_value=Mock())
    return producer

@pytest.fixture
def mock_kafka_consumer():
    """Mock Kafka consumer"""
    consumer = AsyncMock()
    consumer.subscribe.return_value = None
    consumer.poll.return_value = {}
    consumer.commit.return_value = None
    return consumer

@pytest.fixture
def mock_redis():
    """Mock Redis client"""
    redis = AsyncMock()
    redis.get.return_value = None
    redis.set.return_value = True
    redis.delete.return_value = 1
    redis.exists.return_value = False
    redis.ping.return_value = True
    return redis

@pytest.fixture
def sample_market_data():
    """Generate sample market data for testing"""
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
    symbols = ['AAPL', 'GOOGL', 'TSLA', 'MSFT', 'AMZN']
    
    data = {}
    np.random.seed(42)  # For reproducible tests
    
    for symbol in symbols:
        # Generate realistic price data
        returns = np.random.normal(0.0005, 0.02, len(dates))
        prices = 100 * np.cumprod(1 + returns)
        
        data[symbol] = pd.DataFrame({
            'open': prices * np.random.uniform(0.995, 1.005, len(dates)),
            'high': prices * np.random.uniform(1.0, 1.025, len(dates)),
            'low': prices * np.random.uniform(0.975, 1.0, len(dates)),
            'close': prices,
            'volume': np.random.randint(1000000, 50000000, len(dates)),
            'timestamp': dates
        }, index=dates)
    
    return data

@pytest.fixture
def sample_returns_data(sample_market_data):
    """Generate returns data from market data"""
    returns_data = {}
    
    for symbol, df in sample_market_data.items():
        returns_data[symbol] = df['close'].pct_change().dropna()
    
    return pd.DataFrame(returns_data)

@pytest.fixture
def mock_external_api():
    """Mock external API responses"""
    class MockAPI:
        def __init__(self):
            self.call_count = 0
            
        async def get_market_data(self, symbol, **kwargs):
            self.call_count += 1
            return {
                'symbol': symbol,
                'price': 150.0 + np.random.normal(0, 5),
                'volume': np.random.randint(1000000, 10000000),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        
        async def get_financial_data(self, symbol, **kwargs):
            self.call_count += 1
            return {
                'symbol': symbol,
                'pe_ratio': np.random.uniform(10, 30),
                'market_cap': np.random.uniform(100e9, 3000e9),
                'dividend_yield': np.random.uniform(0, 0.05)
            }
    
    return MockAPI()

@pytest.fixture
def mock_ml_model():
    """Mock ML model for testing"""
    model = Mock()
    model.predict.return_value = np.array([0.65])  # Prediction confidence
    model.predict_proba.return_value = np.array([[0.35, 0.65]])
    model.feature_importances_ = np.array([0.3, 0.2, 0.15, 0.1, 0.25])
    return model

@pytest.fixture
def mock_portfolio_optimizer():
    """Mock portfolio optimizer"""
    optimizer = Mock()
    
    def mock_optimize(*args, **kwargs):
        from app.portfolio.optimization import OptimizationResult, OptimizationMethod
        return OptimizationResult(
            weights={'AAPL': 0.3, 'GOOGL': 0.3, 'TSLA': 0.4},
            expected_return=0.12,
            volatility=0.18,
            sharpe_ratio=0.67,
            method=OptimizationMethod.MEAN_VARIANCE,
            risk_contributions={'AAPL': 0.33, 'GOOGL': 0.33, 'TSLA': 0.34}
        )
    
    optimizer.optimize_portfolio = mock_optimize
    return optimizer

@pytest.fixture
def test_config():
    """Test configuration settings"""
    return {
        'database_url': 'sqlite+aiosqlite:///:memory:',
        'redis_url': 'redis://localhost:6379/15',
        'kafka_bootstrap_servers': 'localhost:9092',
        'test_mode': True,
        'mock_market_data': True,
        'ml_max_training_time': 60,
        'backtest_max_concurrent': 1
    }

# Async fixtures for services
@pytest.fixture
async def mock_signal_service():
    """Mock signal generation service"""
    service = AsyncMock()
    
    from app.models.signal import SignalCreate, SignalType, AssetClass, TimeFrame
    
    service.generate_signal.return_value = SignalCreate(
        symbol="AAPL",
        signal_type=SignalType.BUY,
        confidence=0.75,
        target_price=155.0,
        asset_class=AssetClass.EQUITY,
        timeframe=TimeFrame.DAILY,
        timestamp=datetime.now(timezone.utc)
    )
    
    return service

@pytest.fixture
async def mock_user_service():
    """Mock user management service"""
    service = AsyncMock()
    
    from app.models.user import User, RiskTolerance
    from decimal import Decimal
    
    service.get_user.return_value = User(
        id=1,
        username="test_user",
        email="test@example.com",
        risk_tolerance=RiskTolerance.MODERATE,
        investment_amount=Decimal("100000.00"),
        created_at=datetime.now(timezone.utc)
    )
    
    return service

# Performance testing fixtures
@pytest.fixture
def performance_timer():
    """Timer for performance tests"""
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = datetime.now()
        
        def stop(self):
            self.end_time = datetime.now()
        
        @property
        def elapsed_seconds(self):
            if self.start_time and self.end_time:
                return (self.end_time - self.start_time).total_seconds()
            return None
    
    return Timer()

# Memory usage fixtures
@pytest.fixture
def memory_profiler():
    """Memory usage profiler for performance tests"""
    try:
        import psutil
        
        class MemoryProfiler:
            def __init__(self):
                self.process = psutil.Process()
                self.initial_memory = None
                self.peak_memory = None
            
            def start(self):
                self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
                self.peak_memory = self.initial_memory
            
            def update_peak(self):
                current_memory = self.process.memory_info().rss / 1024 / 1024
                if current_memory > self.peak_memory:
                    self.peak_memory = current_memory
            
            @property
            def memory_increase(self):
                current_memory = self.process.memory_info().rss / 1024 / 1024
                return current_memory - self.initial_memory if self.initial_memory else 0
        
        return MemoryProfiler()
    except ImportError:
        # Return mock if psutil not available
        class MockMemoryProfiler:
            def start(self): pass
            def update_peak(self): pass
            @property
            def memory_increase(self): return 0
        
        return MockMemoryProfiler()

# Test data cleanup
@pytest.fixture(autouse=True)
def cleanup_test_data():
    """Automatically cleanup test data after each test"""
    yield
    # Cleanup code can go here if needed
    pass

# Skip conditions for external dependencies
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add skip conditions"""
    skip_external = pytest.mark.skip(reason="External service not available")
    
    for item in items:
        # Skip external tests if services not available
        if "external" in item.keywords:
            # Could add logic here to check if external services are available
            # For now, just run them
            pass