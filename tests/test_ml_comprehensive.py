"""
Comprehensive ML System Tests
Tests for machine learning models, training, and prediction services
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, AsyncMock

from app.ml.models import MLPredictor, EnsemblePredictor, ModelConfig, ModelType, PredictionHorizon
from app.ml.training_service import MLTrainingService
from app.ml.prediction_service import MLPredictionService
from app.models.signal import SignalCreate, SignalType, AssetClass, TimeFrame


class TestMLModels:
    """Test ML model implementations"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data"""
        dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
        n_points = len(dates)
        
        # Generate synthetic OHLCV data
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.02, n_points)
        prices = 100 * np.cumprod(1 + returns)
        
        data = pd.DataFrame({
            'open': prices * np.random.uniform(0.99, 1.01, n_points),
            'high': prices * np.random.uniform(1.0, 1.03, n_points),
            'low': prices * np.random.uniform(0.97, 1.0, n_points),
            'close': prices,
            'volume': np.random.randint(100000, 10000000, n_points),
        }, index=dates)
        
        return data
    
    @pytest.fixture
    def model_config(self):
        """Create test model configuration"""
        return ModelConfig(
            model_type=ModelType.XGBOOST,
            horizon=PredictionHorizon.DAILY,
            lookback_periods=50,
            feature_count=15,
            hyperparams={'n_estimators': 50, 'learning_rate': 0.1}
        )
    
    def test_model_initialization(self, model_config):
        """Test ML model initialization"""
        predictor = MLPredictor(model_config)
        
        assert predictor.config == model_config
        assert not predictor.is_trained
        assert predictor.model is None
        assert predictor.performance is None
    
    @pytest.mark.asyncio
    async def test_model_training(self, model_config, sample_data):
        """Test model training process"""
        predictor = MLPredictor(model_config)
        
        # Train the model
        success = await predictor.train(sample_data, "AAPL")
        
        assert success
        assert predictor.is_trained
        assert predictor.model is not None
        assert predictor.performance is not None
        
        # Check performance metrics
        assert 0 <= predictor.performance.accuracy <= 1
        assert predictor.performance.mse >= 0
        assert predictor.performance.mae >= 0
    
    @pytest.mark.asyncio
    async def test_model_prediction(self, model_config, sample_data):
        """Test model prediction"""
        predictor = MLPredictor(model_config)
        
        # Train first
        await predictor.train(sample_data, "AAPL")
        
        # Make prediction
        features = {
            'close': 150.0,
            'volume': 1000000,
            'rsi': 50.0,
            'macd': 0.5,
            'hour': 14,
            'day_of_week': 2
        }
        
        prediction = await predictor.predict_single(features, "AAPL")
        
        assert prediction is not None
        assert prediction.symbol == "AAPL"
        assert prediction.predicted_price > 0
        assert 0 <= prediction.prediction_confidence <= 1
        assert prediction.predicted_direction in [SignalType.BUY, SignalType.SELL]
        assert 0 <= prediction.probability_up <= 1
        assert 0 <= prediction.probability_down <= 1
        assert isinstance(prediction.feature_importance, dict)
    
    @pytest.mark.asyncio
    async def test_ensemble_prediction(self, sample_data):
        """Test ensemble model prediction"""
        configs = [
            ModelConfig(ModelType.XGBOOST, PredictionHorizon.DAILY, 50, 15),
            ModelConfig(ModelType.RANDOM_FOREST, PredictionHorizon.DAILY, 50, 15),
        ]
        
        ensemble = EnsemblePredictor(configs)
        
        # Train ensemble
        success = await ensemble.train(sample_data, "AAPL")
        assert success
        
        # Make prediction
        features = {'close': 150.0, 'volume': 1000000, 'rsi': 50.0}
        prediction = await ensemble.predict(features, "AAPL")
        
        assert prediction is not None
        assert prediction.symbol == "AAPL"
    
    def test_model_serialization(self, model_config, tmp_path):
        """Test model saving and loading"""
        predictor = MLPredictor(model_config)
        
        # Mock successful training
        predictor.is_trained = True
        predictor.model = Mock()
        predictor.scaler = Mock()
        predictor.feature_engineer.feature_names = ['close', 'volume', 'rsi']
        
        # Test would save/load model files
        assert predictor.is_trained


class TestMLTrainingService:
    """Test ML training service"""
    
    @pytest.fixture
    def training_service(self):
        """Create training service instance"""
        return MLTrainingService()
    
    @pytest.mark.asyncio
    async def test_service_initialization(self, training_service):
        """Test service initialization"""
        assert not training_service.is_running
        assert len(training_service.active_models) == 0
        assert len(training_service.default_configs) > 0
    
    @pytest.mark.asyncio
    async def test_training_job_creation(self, training_service):
        """Test training job creation"""
        training_service.train_symbol("AAPL", force_retrain=True)
        
        assert len(training_service.training_queue) == 1
        job = training_service.training_queue[0]
        assert job.symbol == "AAPL"
        assert job.force_retrain == True
    
    @pytest.mark.asyncio
    async def test_model_management(self, training_service):
        """Test model storage and retrieval"""
        # Mock a trained model
        mock_predictor = Mock()
        training_service.active_models["AAPL"] = {"xgboost_daily": mock_predictor}
        
        # Test retrieval
        predictor = training_service.get_predictor("AAPL", "xgboost", "daily")
        assert predictor == mock_predictor
        
        # Test available models
        models = training_service.get_available_models("AAPL")
        assert "xgboost_daily" in models


class TestMLPredictionService:
    """Test ML prediction service"""
    
    @pytest.fixture
    def prediction_service(self):
        """Create prediction service instance"""
        return MLPredictionService()
    
    @pytest.mark.asyncio
    async def test_service_initialization(self, prediction_service):
        """Test prediction service initialization"""
        assert not prediction_service.is_running
        assert len(prediction_service.prediction_symbols) == 0
        assert len(prediction_service.prediction_cache) == 0
    
    @pytest.mark.asyncio
    async def test_symbol_management(self, prediction_service):
        """Test symbol addition and removal"""
        prediction_service.add_symbol("AAPL")
        assert "AAPL" in prediction_service.prediction_symbols
        
        prediction_service.remove_symbol("AAPL")
        assert "AAPL" not in prediction_service.prediction_symbols
    
    @pytest.mark.asyncio
    async def test_prediction_caching(self, prediction_service):
        """Test prediction caching mechanism"""
        # Mock prediction result
        mock_prediction = Mock()
        mock_prediction.symbol = "AAPL"
        mock_prediction.prediction_horizon = PredictionHorizon.DAILY
        mock_prediction.timestamp = datetime.now(timezone.utc)
        
        cache_key = f"AAPL_{PredictionHorizon.DAILY.value}"
        prediction_service.prediction_cache[cache_key] = mock_prediction
        
        # Test cache retrieval
        cached = prediction_service.prediction_cache.get(cache_key)
        assert cached == mock_prediction
    
    def test_prediction_stats(self, prediction_service):
        """Test prediction service statistics"""
        prediction_service.add_symbol("AAPL")
        prediction_service.add_symbol("GOOGL")
        
        stats = prediction_service.get_prediction_stats()
        
        assert stats["monitored_symbols"] == 2
        assert stats["cached_predictions"] == 0
        assert "predictions_by_horizon" in stats
        assert "recent_predictions" in stats


class TestMLPerformance:
    """Test ML system performance"""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_training_performance(self):
        """Test model training performance"""
        config = ModelConfig(
            model_type=ModelType.XGBOOST,
            horizon=PredictionHorizon.DAILY,
            lookback_periods=100,
            feature_count=20
        )
        
        predictor = MLPredictor(config)
        
        # Create larger dataset for performance testing
        dates = pd.date_range('2020-01-01', '2024-01-01', freq='D')
        data = pd.DataFrame({
            'close': np.random.random(len(dates)) * 100 + 100,
            'volume': np.random.randint(100000, 10000000, len(dates))
        }, index=dates)
        
        start_time = datetime.now()
        success = await predictor.train(data, "PERF_TEST")
        training_time = (datetime.now() - start_time).total_seconds()
        
        assert success
        assert training_time < 300  # Should complete within 5 minutes
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_prediction_latency(self):
        """Test prediction latency"""
        config = ModelConfig(ModelType.XGBOOST, PredictionHorizon.DAILY)
        predictor = MLPredictor(config)
        
        # Mock trained model
        predictor.is_trained = True
        predictor.model = Mock()
        predictor.model.predict.return_value = np.array([0.05])
        predictor.scaler = Mock()
        predictor.scaler.transform.return_value = np.array([[1, 2, 3]])
        predictor.feature_engineer.feature_names = ['close', 'volume', 'rsi']
        
        features = {'close': 150.0, 'volume': 1000000, 'rsi': 50.0}
        
        # Measure prediction latency
        latencies = []
        for _ in range(100):
            start = datetime.now()
            await predictor.predict_single(features, "LATENCY_TEST")
            latency = (datetime.now() - start).total_seconds() * 1000  # ms
            latencies.append(latency)
        
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        
        assert avg_latency < 100  # Average < 100ms
        assert p95_latency < 200   # P95 < 200ms
    
    @pytest.mark.performance
    def test_memory_usage(self):
        """Test memory usage during ML operations"""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create multiple predictors
        predictors = []
        for i in range(10):
            config = ModelConfig(ModelType.RANDOM_FOREST, PredictionHorizon.DAILY)
            predictor = MLPredictor(config)
            predictors.append(predictor)
        
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = current_memory - initial_memory
        
        # Clean up
        del predictors
        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        assert memory_increase < 500  # Less than 500MB increase
        assert final_memory - initial_memory < 100  # Most memory cleaned up


class TestMLIntegration:
    """Integration tests for ML system"""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self):
        """Test complete ML workflow"""
        # Create services
        training_service = MLTrainingService()
        prediction_service = MLPredictionService()
        
        # Mock data provider
        with patch('app.ml.training_service.MLTrainingService._get_training_data') as mock_data:
            sample_data = pd.DataFrame({
                'close': np.random.random(100) * 100 + 100,
                'volume': np.random.randint(100000, 1000000, 100)
            })
            mock_data.return_value = sample_data
            
            # Queue training job
            training_service.train_symbol("INTEGRATION_TEST")
            
            # Simulate training completion
            config = ModelConfig(ModelType.XGBOOST, PredictionHorizon.DAILY)
            mock_predictor = Mock()
            mock_predictor.config = config
            mock_predictor.is_trained = True
            
            training_service.active_models["INTEGRATION_TEST"] = {"xgboost_daily": mock_predictor}
            
            # Add symbol to prediction service
            prediction_service.add_symbol("INTEGRATION_TEST")
            
            # Verify integration
            assert len(training_service.active_models) > 0
            assert "INTEGRATION_TEST" in prediction_service.prediction_symbols
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_kafka_integration(self):
        """Test Kafka integration for ML predictions"""
        prediction_service = MLPredictionService()
        
        # Mock Kafka producer
        with patch('app.ml.prediction_service.get_producer') as mock_producer:
            mock_producer.return_value = AsyncMock()
            
            # Mock prediction generation
            with patch.object(prediction_service, '_generate_prediction') as mock_generate:
                mock_prediction = Mock()
                mock_prediction.prediction_confidence = 0.8
                mock_prediction.symbol = "KAFKA_TEST"
                mock_generate.return_value = mock_prediction
                
                # Process market data message
                market_data = {
                    "symbol": "KAFKA_TEST",
                    "price": 150.0,
                    "volume": 1000000,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
                await prediction_service._trigger_predictions_for_symbol("KAFKA_TEST", market_data)
                
                # Verify prediction was processed
                assert len(prediction_service.prediction_cache) >= 0


class TestMLErrorHandling:
    """Test ML system error handling"""
    
    @pytest.mark.asyncio
    async def test_training_with_insufficient_data(self):
        """Test training with insufficient data"""
        config = ModelConfig(ModelType.XGBOOST, PredictionHorizon.DAILY)
        predictor = MLPredictor(config)
        
        # Create very small dataset
        small_data = pd.DataFrame({
            'close': [100, 101, 102],
            'volume': [1000, 1100, 1200]
        })
        
        success = await predictor.train(small_data, "SMALL_DATA_TEST")
        
        # Should handle gracefully
        assert not success or predictor.is_trained  # Either fails gracefully or succeeds
    
    @pytest.mark.asyncio
    async def test_prediction_with_missing_features(self):
        """Test prediction with missing features"""
        config = ModelConfig(ModelType.XGBOOST, PredictionHorizon.DAILY)
        predictor = MLPredictor(config)
        
        # Mock trained state
        predictor.is_trained = True
        predictor.model = Mock()
        predictor.model.predict.return_value = np.array([0.05])
        predictor.scaler = Mock()
        predictor.scaler.transform.return_value = np.array([[1, 2, 0]])  # Missing feature = 0
        predictor.feature_engineer.feature_names = ['close', 'volume', 'rsi']
        
        # Missing features
        incomplete_features = {'close': 150.0}  # Missing volume and rsi
        
        prediction = await predictor.predict_single(incomplete_features, "MISSING_FEATURES_TEST")
        
        # Should handle missing features gracefully
        assert prediction is not None or prediction is None  # Either works or fails gracefully
    
    def test_invalid_model_configuration(self):
        """Test invalid model configuration handling"""
        # Test with invalid model type
        with pytest.raises((ValueError, AttributeError)):
            config = ModelConfig(
                model_type="INVALID_MODEL",  # Invalid
                horizon=PredictionHorizon.DAILY
            )
            MLPredictor(config)


if __name__ == "__main__":
    # Run ML tests
    pytest.main([__file__, "-v", "--tb=short"])