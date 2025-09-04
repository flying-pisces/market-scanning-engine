"""
ML Prediction Service
Handles real-time predictions using trained ML models
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

from app.ml.models import PredictionResult, PredictionHorizon
from app.ml.training_service import get_ml_training_service
from app.core.kafka_client import get_producer, KafkaConsumerClient, KafkaTopics, kafka_config
from app.core.cache import get_cache
from app.models.market import MarketDataPoint
from app.models.signal import SignalCreate, SignalType, TimeFrame, AssetClass
from app.services.signal_validation import get_signal_validator, ValidationLevel

logger = logging.getLogger(__name__)


class MLPredictionService:
    """Real-time ML prediction service"""
    
    def __init__(self):
        self.is_running = False
        self.prediction_cache: Dict[str, PredictionResult] = {}
        self.producer = None
        
        # Prediction settings
        self.prediction_symbols = set()  # Symbols to generate predictions for
        self.min_prediction_confidence = 0.6
        self.prediction_intervals = {
            PredictionHorizon.INTRADAY: 300,  # 5 minutes
            PredictionHorizon.DAILY: 3600,    # 1 hour
            PredictionHorizon.WEEKLY: 14400,  # 4 hours
        }
    
    async def start(self):
        """Start the prediction service"""
        if self.is_running:
            return
        
        self.is_running = True
        self.producer = get_producer()
        
        logger.info("ML Prediction Service started")
        
        # Start prediction tasks
        tasks = [
            self._consume_market_data(),
            self._generate_periodic_predictions(),
            self._cleanup_old_predictions()
        ]
        
        await asyncio.gather(*tasks)
    
    async def stop(self):
        """Stop the prediction service"""
        self.is_running = False
        logger.info("ML Prediction Service stopped")
    
    def add_symbol(self, symbol: str):
        """Add symbol for prediction monitoring"""
        self.prediction_symbols.add(symbol)
        logger.info(f"Added {symbol} to prediction monitoring")
    
    def remove_symbol(self, symbol: str):
        """Remove symbol from prediction monitoring"""
        self.prediction_symbols.discard(symbol)
        logger.info(f"Removed {symbol} from prediction monitoring")
    
    async def get_prediction(self, symbol: str, horizon: PredictionHorizon = PredictionHorizon.DAILY) -> Optional[PredictionResult]:
        """Get latest prediction for symbol and horizon"""
        # Try cache first
        cache_key = f"{symbol}_{horizon.value}"
        
        if cache_key in self.prediction_cache:
            cached_pred = self.prediction_cache[cache_key]
            
            # Check if prediction is still fresh (within last prediction interval)
            age_seconds = (datetime.now(timezone.utc) - cached_pred.timestamp).total_seconds()
            if age_seconds < self.prediction_intervals.get(horizon, 3600):
                return cached_pred
        
        # Generate new prediction
        return await self._generate_prediction(symbol, horizon)
    
    async def _consume_market_data(self):
        """Consume market data and trigger predictions"""
        consumer = KafkaConsumerClient(
            config=kafka_config,
            group_id="ml-prediction-market-data",
            topics=[KafkaTopics.MARKET_DATA_RAW]
        )
        
        async def handle_market_data(topic: str, message: Dict[str, Any]):
            try:
                symbol = message.get("symbol")
                if symbol in self.prediction_symbols:
                    # Trigger prediction for this symbol
                    await self._trigger_predictions_for_symbol(symbol, message)
                    
            except Exception as e:
                logger.error(f"Error processing market data for predictions: {e}")
        
        async for message in consumer.consume_messages(handle_market_data):
            if not self.is_running:
                break
    
    async def _trigger_predictions_for_symbol(self, symbol: str, market_data: Dict[str, Any]):
        """Trigger predictions for a symbol based on new market data"""
        try:
            # Check if we need new predictions
            current_time = datetime.now(timezone.utc)
            
            for horizon in [PredictionHorizon.INTRADAY, PredictionHorizon.DAILY]:
                cache_key = f"{symbol}_{horizon.value}"
                
                should_predict = True
                if cache_key in self.prediction_cache:
                    last_prediction = self.prediction_cache[cache_key]
                    age_seconds = (current_time - last_prediction.timestamp).total_seconds()
                    should_predict = age_seconds >= self.prediction_intervals[horizon]
                
                if should_predict:
                    prediction = await self._generate_prediction(symbol, horizon, market_data)
                    if prediction:
                        await self._process_prediction(prediction)
                        
        except Exception as e:
            logger.error(f"Error triggering predictions for {symbol}: {e}")
    
    async def _generate_prediction(self, symbol: str, horizon: PredictionHorizon, market_data: Dict[str, Any] = None) -> Optional[PredictionResult]:
        """Generate ML prediction for symbol and horizon"""
        try:
            training_service = get_ml_training_service()
            if not training_service:
                logger.warning("ML training service not available")
                return None
            
            # Try ensemble first
            ensemble = training_service.get_ensemble(symbol)
            if ensemble:
                features = await self._prepare_features(symbol, market_data)
                if features:
                    prediction = await ensemble.predict(features, symbol)
                    if prediction and prediction.prediction_horizon == horizon:
                        return prediction
            
            # Fall back to individual models
            model_key = f"{horizon.value}"
            available_models = training_service.get_available_models(symbol)
            
            for model_name in available_models:
                if horizon.value in model_name:
                    # Extract model type and horizon
                    parts = model_name.split('_')
                    if len(parts) >= 2:
                        model_type = parts[0]
                        predictor = training_service.get_predictor(symbol, model_type, horizon.value)
                        
                        if predictor:
                            features = await self._prepare_features(symbol, market_data)
                            if features:
                                prediction = await predictor.predict_single(features, symbol)
                                if prediction:
                                    return prediction
            
            logger.debug(f"No suitable model found for {symbol} {horizon.value}")
            return None
            
        except Exception as e:
            logger.error(f"Error generating prediction for {symbol}: {e}")
            return None
    
    async def _prepare_features(self, symbol: str, market_data: Dict[str, Any] = None) -> Optional[Dict[str, float]]:
        """Prepare features for ML prediction"""
        try:
            features = {}
            
            # Use provided market data if available
            if market_data:
                features.update({
                    'close': float(market_data.get('price', 0)),
                    'high': float(market_data.get('high', 0)),
                    'low': float(market_data.get('low', 0)),
                    'volume': float(market_data.get('volume', 0)),
                    'open': float(market_data.get('open_price', 0))
                })
            else:
                # Try to get latest market data from cache
                from app.core.cache import get_market_cache
                market_cache = get_market_cache()
                if market_cache:
                    cached_quote = await market_cache.get_quote(symbol)
                    if cached_quote:
                        features.update({
                            'close': float(cached_quote.get('price', 0)),
                            'high': float(cached_quote.get('high', 0)),
                            'low': float(cached_quote.get('low', 0)),
                            'volume': float(cached_quote.get('volume', 0)),
                            'open': float(cached_quote.get('open', 0))
                        })
            
            # Add technical indicators (simplified - would normally calculate from price history)
            if 'close' in features and features['close'] > 0:
                close_price = features['close']
                
                # Mock technical indicators
                features.update({
                    'rsi': 50.0,  # Would calculate actual RSI
                    'macd': 0.0,
                    'bb_upper': close_price * 1.02,
                    'bb_lower': close_price * 0.98,
                    'ma_20': close_price,
                    'ma_50': close_price,
                    'volatility_20d': 0.02,
                    'volume_ratio_20': 1.0,
                    'return_1d': 0.001,
                    'return_5d': 0.005
                })
                
                # Time features
                import numpy as np
                now = datetime.now(timezone.utc)
                features.update({
                    'hour': now.hour,
                    'day_of_week': now.weekday(),
                    'hour_sin': np.sin(2 * np.pi * now.hour / 24),
                    'hour_cos': np.cos(2 * np.pi * now.hour / 24)
                })
                
                return features
            
            return None
            
        except Exception as e:
            logger.error(f"Error preparing features for {symbol}: {e}")
            return None
    
    async def _process_prediction(self, prediction: PredictionResult):
        """Process and distribute ML prediction"""
        try:
            # Cache the prediction
            cache_key = f"{prediction.symbol}_{prediction.prediction_horizon.value}"
            self.prediction_cache[cache_key] = prediction
            
            # Convert to trading signal if confidence is high enough
            if prediction.prediction_confidence >= self.min_prediction_confidence:
                signal = await self._prediction_to_signal(prediction)
                if signal:
                    # Validate signal
                    validator = get_signal_validator(ValidationLevel.STANDARD)
                    validation_result = await validator.validate_signal(signal)
                    
                    if validation_result.result.value in ['approved', 'warning']:
                        # Publish validated signal
                        await self._publish_ml_signal(signal, prediction, validation_result)
            
            # Publish prediction event
            await self._publish_prediction_event(prediction)
            
            # Cache prediction
            await self._cache_prediction(prediction)
            
        except Exception as e:
            logger.error(f"Error processing prediction: {e}")
    
    async def _prediction_to_signal(self, prediction: PredictionResult) -> Optional[SignalCreate]:
        """Convert ML prediction to trading signal"""
        try:
            # Determine signal timeframe
            timeframe_map = {
                PredictionHorizon.INTRADAY: TimeFrame.INTRADAY,
                PredictionHorizon.DAILY: TimeFrame.DAILY,
                PredictionHorizon.WEEKLY: TimeFrame.WEEKLY,
                PredictionHorizon.MONTHLY: TimeFrame.MONTHLY
            }
            
            timeframe = timeframe_map.get(prediction.prediction_horizon, TimeFrame.DAILY)
            
            # Estimate current price (would get from market data)
            current_price = prediction.predicted_price / (1 + 0.01)  # Rough estimate
            
            # Calculate target and stop loss
            if prediction.predicted_direction == SignalType.BUY:
                target_price = prediction.predicted_price
                stop_loss = current_price * 0.95  # 5% stop loss
            else:
                target_price = prediction.predicted_price
                stop_loss = current_price * 1.05  # 5% stop loss
            
            # Determine asset class (simplified)
            asset_class = AssetClass.STOCKS  # Would determine from symbol
            
            # Calculate risk score based on prediction volatility
            risk_score = min(95, max(20, int(50 + (1 - prediction.prediction_confidence) * 40)))
            
            # Create signal
            signal = SignalCreate(
                symbol=prediction.symbol,
                asset_class=asset_class,
                signal_type=prediction.predicted_direction,
                entry_price=current_price,
                target_price=target_price,
                stop_loss=stop_loss,
                confidence=prediction.prediction_confidence,
                risk_score=risk_score,
                timeframe=timeframe,
                description=f"ML Prediction - {prediction.prediction_horizon.value} horizon "
                           f"({prediction.model_accuracy:.1%} accuracy)",
                metadata={
                    "prediction_source": "ml_model",
                    "model_accuracy": prediction.model_accuracy,
                    "probability_up": prediction.probability_up,
                    "feature_importance": prediction.feature_importance
                }
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error converting prediction to signal: {e}")
            return None
    
    async def _publish_ml_signal(self, signal: SignalCreate, prediction: PredictionResult, validation_result):
        """Publish ML-generated signal"""
        if not self.producer:
            return
        
        signal_data = {
            "symbol": signal.symbol,
            "asset_class": signal.asset_class.value,
            "signal_type": signal.signal_type.value,
            "entry_price": signal.entry_price,
            "target_price": signal.target_price,
            "stop_loss": signal.stop_loss,
            "confidence": signal.confidence,
            "risk_score": signal.risk_score,
            "timeframe": signal.timeframe.value,
            "description": signal.description,
            "metadata": {
                **signal.metadata,
                "ml_prediction": {
                    "predicted_price": prediction.predicted_price,
                    "prediction_confidence": prediction.prediction_confidence,
                    "model_accuracy": prediction.model_accuracy
                },
                "validation": {
                    "result": validation_result.result.value,
                    "quality_score": validation_result.quality_metrics.overall_quality
                }
            },
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": "ml_prediction"
        }
        
        await self.producer.send_message(
            KafkaTopics.SIGNALS_GENERATED,
            signal_data,
            key=signal.symbol
        )
        
        logger.info(f"Published ML signal for {signal.symbol}: {signal.signal_type.value} "
                   f"({signal.confidence:.1%} confidence)")
    
    async def _publish_prediction_event(self, prediction: PredictionResult):
        """Publish prediction event for monitoring"""
        if not self.producer:
            return
        
        event = {
            "type": "ml_prediction",
            "symbol": prediction.symbol,
            "predicted_price": prediction.predicted_price,
            "prediction_confidence": prediction.prediction_confidence,
            "predicted_direction": prediction.predicted_direction.value,
            "horizon": prediction.prediction_horizon.value,
            "model_accuracy": prediction.model_accuracy,
            "timestamp": prediction.timestamp.isoformat()
        }
        
        await self.producer.send_message(
            KafkaTopics.SYSTEM_METRICS,
            event,
            key=f"prediction_{prediction.symbol}"
        )
    
    async def _cache_prediction(self, prediction: PredictionResult):
        """Cache prediction for API access"""
        cache = get_cache()
        if not cache:
            return
        
        cache_key = f"{prediction.symbol}_{prediction.prediction_horizon.value}"
        prediction_data = {
            "symbol": prediction.symbol,
            "predicted_price": prediction.predicted_price,
            "prediction_confidence": prediction.prediction_confidence,
            "predicted_direction": prediction.predicted_direction.value,
            "probability_up": prediction.probability_up,
            "model_accuracy": prediction.model_accuracy,
            "timestamp": prediction.timestamp.isoformat()
        }
        
        await cache.set(
            "ml_predictions",
            cache_key,
            prediction_data,
            ttl=self.prediction_intervals.get(prediction.prediction_horizon, 3600)
        )
    
    async def _generate_periodic_predictions(self):
        """Generate predictions on a schedule"""
        while self.is_running:
            try:
                # Generate predictions for all monitored symbols
                for symbol in list(self.prediction_symbols):  # Copy to avoid modification during iteration
                    try:
                        # Generate daily predictions every hour
                        prediction = await self._generate_prediction(symbol, PredictionHorizon.DAILY)
                        if prediction:
                            await self._process_prediction(prediction)
                        
                        # Brief delay between symbols
                        await asyncio.sleep(1)
                        
                    except Exception as e:
                        logger.error(f"Error generating periodic prediction for {symbol}: {e}")
                
                # Wait before next round
                await asyncio.sleep(3600)  # 1 hour
                
            except Exception as e:
                logger.error(f"Error in periodic prediction generation: {e}")
                await asyncio.sleep(1800)  # 30 minutes on error
    
    async def _cleanup_old_predictions(self):
        """Clean up old cached predictions"""
        while self.is_running:
            try:
                current_time = datetime.now(timezone.utc)
                expired_keys = []
                
                for cache_key, prediction in list(self.prediction_cache.items()):
                    age_hours = (current_time - prediction.timestamp).total_seconds() / 3600
                    
                    # Remove predictions older than 24 hours
                    if age_hours > 24:
                        expired_keys.append(cache_key)
                
                # Remove expired predictions
                for key in expired_keys:
                    del self.prediction_cache[key]
                
                if expired_keys:
                    logger.info(f"Cleaned up {len(expired_keys)} expired predictions")
                
                await asyncio.sleep(3600)  # Clean up every hour
                
            except Exception as e:
                logger.error(f"Error cleaning up predictions: {e}")
                await asyncio.sleep(3600)
    
    def get_prediction_stats(self) -> Dict[str, Any]:
        """Get prediction service statistics"""
        current_time = datetime.now(timezone.utc)
        
        stats = {
            "monitored_symbols": len(self.prediction_symbols),
            "cached_predictions": len(self.prediction_cache),
            "predictions_by_horizon": {},
            "recent_predictions": 0
        }
        
        # Count predictions by horizon
        for prediction in self.prediction_cache.values():
            horizon = prediction.prediction_horizon.value
            stats["predictions_by_horizon"][horizon] = stats["predictions_by_horizon"].get(horizon, 0) + 1
            
            # Count recent predictions (last hour)
            if (current_time - prediction.timestamp).total_seconds() < 3600:
                stats["recent_predictions"] += 1
        
        return stats


# Global prediction service instance
ml_prediction_service: Optional[MLPredictionService] = None


async def init_ml_prediction_service() -> MLPredictionService:
    """Initialize ML prediction service"""
    global ml_prediction_service
    
    ml_prediction_service = MLPredictionService()
    logger.info("ML Prediction Service initialized")
    return ml_prediction_service


def get_ml_prediction_service() -> Optional[MLPredictionService]:
    """Get global ML prediction service instance"""
    return ml_prediction_service