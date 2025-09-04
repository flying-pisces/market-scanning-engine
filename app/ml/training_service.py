"""
ML Model Training Service
Handles model training, retraining, and model lifecycle management
"""

import asyncio
import logging
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass

from app.ml.models import MLPredictor, EnsemblePredictor, ModelConfig, ModelType, PredictionHorizon
from app.core.cache import get_cache, get_market_cache
from app.core.kafka_client import get_producer, KafkaTopics
from app.services.background_jobs import get_job_processor, JobPriority
from app.database import get_db
from app.models.signal import Signal
from sqlalchemy import select, and_

logger = logging.getLogger(__name__)


@dataclass
class TrainingJob:
    """ML model training job"""
    symbol: str
    model_configs: List[ModelConfig]
    priority: JobPriority = JobPriority.NORMAL
    force_retrain: bool = False
    
    
class MLTrainingService:
    """Centralized ML model training and management service"""
    
    def __init__(self):
        self.active_models: Dict[str, Dict[str, MLPredictor]] = {}  # symbol -> {model_type_horizon: predictor}
        self.ensemble_models: Dict[str, EnsemblePredictor] = {}  # symbol -> ensemble
        self.training_queue: List[TrainingJob] = []
        self.is_running = False
        
        # Default model configurations
        self.default_configs = self._create_default_configs()
    
    def _create_default_configs(self) -> List[ModelConfig]:
        """Create default model configurations"""
        configs = []
        
        # Fast models for intraday predictions
        configs.append(ModelConfig(
            model_type=ModelType.XGBOOST,
            horizon=PredictionHorizon.INTRADAY,
            lookback_periods=50,
            feature_count=15,
            retrain_frequency=6,  # 6 hours
            hyperparams={
                'n_estimators': 50,
                'learning_rate': 0.1,
                'max_depth': 4
            }
        ))
        
        configs.append(ModelConfig(
            model_type=ModelType.LIGHTGBM,
            horizon=PredictionHorizon.INTRADAY,
            lookback_periods=50,
            feature_count=15,
            retrain_frequency=6,
            hyperparams={
                'n_estimators': 50,
                'learning_rate': 0.1,
                'max_depth': 4
            }
        ))
        
        # Medium-term models for daily predictions
        configs.append(ModelConfig(
            model_type=ModelType.RANDOM_FOREST,
            horizon=PredictionHorizon.DAILY,
            lookback_periods=100,
            feature_count=20,
            retrain_frequency=24,  # 24 hours
            hyperparams={
                'n_estimators': 100,
                'max_depth': 8
            }
        ))
        
        configs.append(ModelConfig(
            model_type=ModelType.XGBOOST,
            horizon=PredictionHorizon.DAILY,
            lookback_periods=100,
            feature_count=20,
            retrain_frequency=24,
            hyperparams={
                'n_estimators': 100,
                'learning_rate': 0.05,
                'max_depth': 6
            }
        ))
        
        # Longer-term models for weekly predictions
        configs.append(ModelConfig(
            model_type=ModelType.GRADIENT_BOOSTING,
            horizon=PredictionHorizon.WEEKLY,
            lookback_periods=200,
            feature_count=25,
            retrain_frequency=48,  # 48 hours
            hyperparams={
                'n_estimators': 150,
                'learning_rate': 0.03,
                'max_depth': 8
            }
        ))
        
        return configs
    
    async def start(self):
        """Start the ML training service"""
        if self.is_running:
            return
        
        self.is_running = True
        logger.info("ML Training Service started")
        
        # Start background tasks
        tasks = [
            self._process_training_queue(),
            self._schedule_retraining(),
            self._monitor_model_performance()
        ]
        
        await asyncio.gather(*tasks)
    
    async def stop(self):
        """Stop the ML training service"""
        self.is_running = False
        logger.info("ML Training Service stopped")
    
    def add_training_job(self, job: TrainingJob):
        """Add a training job to the queue"""
        self.training_queue.append(job)
        logger.info(f"Added training job for {job.symbol} with {len(job.model_configs)} models")
    
    def train_symbol(self, symbol: str, force_retrain: bool = False, priority: JobPriority = JobPriority.NORMAL):
        """Queue training for a specific symbol"""
        job = TrainingJob(
            symbol=symbol,
            model_configs=self.default_configs.copy(),
            priority=priority,
            force_retrain=force_retrain
        )
        self.add_training_job(job)
    
    def train_multiple_symbols(self, symbols: List[str], force_retrain: bool = False):
        """Queue training for multiple symbols"""
        for symbol in symbols:
            self.train_symbol(symbol, force_retrain)
    
    async def _process_training_queue(self):
        """Process training jobs from the queue"""
        while self.is_running:
            try:
                if self.training_queue:
                    # Sort by priority
                    self.training_queue.sort(key=lambda x: x.priority.value, reverse=True)
                    job = self.training_queue.pop(0)
                    
                    await self._execute_training_job(job)
                else:
                    await asyncio.sleep(10)  # Wait before checking again
                    
            except Exception as e:
                logger.error(f"Error processing training queue: {e}")
                await asyncio.sleep(30)
    
    async def _execute_training_job(self, job: TrainingJob):
        """Execute a training job"""
        try:
            logger.info(f"Executing training job for {job.symbol}")
            
            # Check if retraining is needed
            if not job.force_retrain and not await self._needs_retraining(job.symbol):
                logger.info(f"Skipping training for {job.symbol} - models are up to date")
                return
            
            # Get training data
            data = await self._get_training_data(job.symbol)
            if data is None or len(data) < 100:
                logger.warning(f"Insufficient data for training {job.symbol}: {len(data) if data is not None else 0} samples")
                return
            
            # Train individual models
            predictors = []
            for config in job.model_configs:
                predictor = MLPredictor(config)
                success = await predictor.train(data, job.symbol)
                
                if success:
                    predictors.append(predictor)
                    model_key = f"{config.model_type.value}_{config.horizon.value}"
                    
                    # Store in active models
                    if job.symbol not in self.active_models:
                        self.active_models[job.symbol] = {}
                    self.active_models[job.symbol][model_key] = predictor
                    
                    logger.info(f"Successfully trained {model_key} for {job.symbol}")
                else:
                    logger.warning(f"Failed to train {config.model_type.value} for {job.symbol}")
            
            # Create ensemble if we have multiple models
            if len(predictors) > 1:
                ensemble = EnsemblePredictor([p.config for p in predictors])
                ensemble.predictors = predictors  # Use already trained predictors
                ensemble._calculate_weights()
                
                self.ensemble_models[job.symbol] = ensemble
                logger.info(f"Created ensemble model for {job.symbol} with {len(predictors)} models")
            
            # Publish training completion event
            await self._publish_training_event(job.symbol, len(predictors))
            
            # Cache model metadata
            await self._cache_model_metadata(job.symbol, predictors)
            
        except Exception as e:
            logger.error(f"Error executing training job for {job.symbol}: {e}")
    
    async def _get_training_data(self, symbol: str, days: int = 365) -> Optional[pd.DataFrame]:
        """Get training data for a symbol"""
        try:
            # Try to get from cache first
            market_cache = get_market_cache()
            if market_cache:
                cached_data = await market_cache.get_price_history(symbol, "daily")
                if cached_data and len(cached_data) > 200:
                    return self._format_training_data(cached_data, symbol)
            
            # If not in cache, would normally fetch from data provider
            # For now, generate synthetic data for demonstration
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days)
            
            # Generate synthetic OHLCV data
            dates = pd.date_range(start_date, end_date, freq='D')
            n_points = len(dates)
            
            # Random walk for price
            np.random.seed(42)  # For reproducible results
            returns = np.random.normal(0.0005, 0.02, n_points)  # ~0.05% daily return, 2% volatility
            prices = 100 * np.cumprod(1 + returns)
            
            # Generate OHLC from prices
            high_factor = np.random.uniform(1.0, 1.05, n_points)
            low_factor = np.random.uniform(0.95, 1.0, n_points)
            
            data = pd.DataFrame({
                'timestamp': dates,
                'open': prices * np.random.uniform(0.98, 1.02, n_points),
                'high': prices * high_factor,
                'low': prices * low_factor,
                'close': prices,
                'volume': np.random.randint(100000, 10000000, n_points)
            })
            
            data.set_index('timestamp', inplace=True)
            return data
            
        except Exception as e:
            logger.error(f"Error getting training data for {symbol}: {e}")
            return None
    
    def _format_training_data(self, data: Any, symbol: str) -> pd.DataFrame:
        """Format cached data for training"""
        # This would format the cached data into the expected DataFrame structure
        # For now, return as-is assuming it's already formatted
        return data
    
    async def _needs_retraining(self, symbol: str) -> bool:
        """Check if models need retraining"""
        cache = get_cache()
        if not cache:
            return True
        
        # Check last training time
        model_metadata = await cache.get("ml_models", f"{symbol}_last_trained")
        if not model_metadata:
            return True
        
        last_trained = datetime.fromisoformat(model_metadata['trained_at'])
        hours_since_training = (datetime.now(timezone.utc) - last_trained).total_seconds() / 3600
        
        # Retrain if more than 24 hours have passed
        return hours_since_training > 24
    
    async def _schedule_retraining(self):
        """Schedule periodic retraining of models"""
        while self.is_running:
            try:
                # Get active symbols from database
                async with get_db() as db:
                    # Get symbols that have had recent signals
                    result = await db.execute(
                        select(Signal.symbol.distinct())
                        .where(Signal.created_at >= datetime.now(timezone.utc) - timedelta(days=7))
                    )
                    active_symbols = [row[0] for row in result.fetchall()]
                
                # Check each symbol for retraining needs
                for symbol in active_symbols:
                    if await self._needs_retraining(symbol):
                        logger.info(f"Scheduling retraining for {symbol}")
                        self.train_symbol(symbol, force_retrain=False, priority=JobPriority.LOW)
                
                # Wait 6 hours before next check
                await asyncio.sleep(6 * 3600)
                
            except Exception as e:
                logger.error(f"Error in retraining scheduler: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour on error
    
    async def _monitor_model_performance(self):
        """Monitor model performance and flag underperforming models"""
        while self.is_running:
            try:
                for symbol, models in self.active_models.items():
                    for model_key, predictor in models.items():
                        if predictor.performance:
                            # Flag models with poor performance
                            if predictor.performance.accuracy < 0.5 or predictor.performance.r2 < 0.1:
                                logger.warning(f"Poor performance detected for {symbol} {model_key}: "
                                             f"accuracy={predictor.performance.accuracy:.3f}, "
                                             f"r2={predictor.performance.r2:.3f}")
                                
                                # Schedule retraining with high priority
                                self.train_symbol(symbol, force_retrain=True, priority=JobPriority.HIGH)
                
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                logger.error(f"Error monitoring model performance: {e}")
                await asyncio.sleep(3600)
    
    async def _publish_training_event(self, symbol: str, model_count: int):
        """Publish model training completion event"""
        producer = get_producer()
        if producer:
            event = {
                "type": "model_training_completed",
                "symbol": symbol,
                "model_count": model_count,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            await producer.send_message(
                KafkaTopics.SYSTEM_METRICS,
                event,
                key=f"training_{symbol}"
            )
    
    async def _cache_model_metadata(self, symbol: str, predictors: List[MLPredictor]):
        """Cache model metadata for quick access"""
        cache = get_cache()
        if not cache:
            return
        
        metadata = {
            "symbol": symbol,
            "model_count": len(predictors),
            "models": [],
            "trained_at": datetime.now(timezone.utc).isoformat()
        }
        
        for predictor in predictors:
            model_info = {
                "type": predictor.config.model_type.value,
                "horizon": predictor.config.horizon.value,
                "performance": {
                    "accuracy": predictor.performance.accuracy if predictor.performance else 0.0,
                    "r2": predictor.performance.r2 if predictor.performance else 0.0,
                    "sharpe_ratio": predictor.performance.sharpe_ratio if predictor.performance else 0.0
                } if predictor.performance else None
            }
            metadata["models"].append(model_info)
        
        # Cache metadata
        await cache.set("ml_models", f"{symbol}_metadata", metadata, ttl=86400)
        await cache.set("ml_models", f"{symbol}_last_trained", metadata, ttl=86400)
    
    def get_predictor(self, symbol: str, model_type: str, horizon: str) -> Optional[MLPredictor]:
        """Get specific predictor for symbol"""
        if symbol not in self.active_models:
            return None
        
        model_key = f"{model_type}_{horizon}"
        return self.active_models[symbol].get(model_key)
    
    def get_ensemble(self, symbol: str) -> Optional[EnsemblePredictor]:
        """Get ensemble predictor for symbol"""
        return self.ensemble_models.get(symbol)
    
    def get_available_models(self, symbol: str) -> List[str]:
        """Get list of available models for symbol"""
        if symbol not in self.active_models:
            return []
        return list(self.active_models[symbol].keys())
    
    async def get_model_performance(self, symbol: str) -> Dict[str, Any]:
        """Get performance summary for symbol's models"""
        if symbol not in self.active_models:
            return {"error": "No models found for symbol"}
        
        performance_summary = {
            "symbol": symbol,
            "models": {},
            "ensemble_available": symbol in self.ensemble_models
        }
        
        for model_key, predictor in self.active_models[symbol].items():
            if predictor.performance:
                performance_summary["models"][model_key] = {
                    "accuracy": predictor.performance.accuracy,
                    "r2": predictor.performance.r2,
                    "mse": predictor.performance.mse,
                    "sharpe_ratio": predictor.performance.sharpe_ratio,
                    "win_rate": predictor.performance.win_rate
                }
            else:
                performance_summary["models"][model_key] = {"status": "no_performance_data"}
        
        return performance_summary


# Global training service instance
ml_training_service: Optional[MLTrainingService] = None


async def init_ml_training_service() -> MLTrainingService:
    """Initialize ML training service"""
    global ml_training_service
    
    ml_training_service = MLTrainingService()
    logger.info("ML Training Service initialized")
    return ml_training_service


def get_ml_training_service() -> Optional[MLTrainingService]:
    """Get global ML training service instance"""
    return ml_training_service


# Convenience functions for background job integration

async def schedule_model_training(symbol: str, priority: str = "normal"):
    """Schedule model training via background jobs"""
    job_processor = get_job_processor()
    training_service = get_ml_training_service()
    
    if job_processor and training_service:
        priority_map = {
            "low": JobPriority.LOW,
            "normal": JobPriority.NORMAL,
            "high": JobPriority.HIGH,
            "urgent": JobPriority.URGENT
        }
        
        job_priority = priority_map.get(priority, JobPriority.NORMAL)
        
        job_processor.add_job(
            name=f"Train ML Models - {symbol}",
            handler_name="train_ml_models",
            args=[symbol],
            kwargs={"priority": priority},
            priority=job_priority
        )


async def bulk_train_models(symbols: List[str]):
    """Schedule bulk training for multiple symbols"""
    training_service = get_ml_training_service()
    if training_service:
        training_service.train_multiple_symbols(symbols)