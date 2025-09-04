"""
Advanced machine learning models for signal prediction
Implements various ML approaches for market signal generation and price prediction
"""

import asyncio
import logging
import numpy as np
import pandas as pd
import joblib
import json
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

# ML Libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb

# Time Series
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from arch import arch_model

# Feature Engineering
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import ta  # Technical Analysis library

from app.models.signal import SignalType, TimeFrame, AssetClass
from app.core.cache import get_cache

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Types of ML models available"""
    LINEAR = "linear"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    SVM = "svm"
    ENSEMBLE = "ensemble"
    LSTM = "lstm"
    ARIMA = "arima"
    GARCH = "garch"


class PredictionHorizon(Enum):
    """Prediction time horizons"""
    INTRADAY = "1h"      # Next hour
    DAILY = "1d"         # Next day
    WEEKLY = "1w"        # Next week
    MONTHLY = "1m"       # Next month


@dataclass
class ModelConfig:
    """ML model configuration"""
    model_type: ModelType
    horizon: PredictionHorizon
    lookback_periods: int = 100
    feature_count: int = 20
    retrain_frequency: int = 24  # hours
    validation_split: float = 0.2
    hyperparams: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.hyperparams is None:
            self.hyperparams = {}


@dataclass
class PredictionResult:
    """ML prediction result"""
    symbol: str
    predicted_price: float
    prediction_confidence: float
    prediction_horizon: PredictionHorizon
    predicted_direction: SignalType
    probability_up: float
    probability_down: float
    feature_importance: Dict[str, float]
    model_accuracy: float
    timestamp: datetime


@dataclass
class ModelPerformance:
    """Model performance metrics"""
    mse: float
    mae: float
    r2: float
    accuracy: float
    precision: float
    recall: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float


class FeatureEngineer:
    """Advanced feature engineering for financial time series"""
    
    def __init__(self):
        self.scaler = RobustScaler()
        self.feature_names = []
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive feature set"""
        features_df = df.copy()
        
        # Price-based features
        features_df = self._add_price_features(features_df)
        
        # Technical indicators
        features_df = self._add_technical_indicators(features_df)
        
        # Volume features
        features_df = self._add_volume_features(features_df)
        
        # Time-based features
        features_df = self._add_time_features(features_df)
        
        # Statistical features
        features_df = self._add_statistical_features(features_df)
        
        # Market microstructure features
        features_df = self._add_microstructure_features(features_df)
        
        return features_df.dropna()
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features"""
        # Returns
        for period in [1, 5, 10, 20]:
            df[f'return_{period}d'] = df['close'].pct_change(period)
            df[f'log_return_{period}d'] = np.log(df['close'] / df['close'].shift(period))
        
        # Price ratios
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        df['price_range'] = (df['high'] - df['low']) / df['close']
        
        # Price position within day's range
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            df[f'ma_{window}'] = df['close'].rolling(window).mean()
            df[f'price_ma_ratio_{window}'] = df['close'] / df[f'ma_{window}']
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical analysis indicators"""
        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
        df['rsi_sma'] = df['rsi'].rolling(14).mean()
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['close'])
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # Williams %R
        df['williams_r'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r()
        
        # Average True Range
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        df['atr_ratio'] = df['atr'] / df['close']
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features"""
        if 'volume' not in df.columns:
            return df
        
        # Volume moving averages
        for window in [5, 10, 20]:
            df[f'volume_ma_{window}'] = df['volume'].rolling(window).mean()
            df[f'volume_ratio_{window}'] = df['volume'] / df[f'volume_ma_{window}']
        
        # Volume-price indicators
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        df['price_vwap_ratio'] = df['close'] / df['vwap']
        
        # On-Balance Volume
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        
        # Volume Rate of Change
        df['volume_roc'] = df['volume'].pct_change(10)
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features"""
        # Rolling statistics
        for window in [5, 10, 20]:
            df[f'volatility_{window}d'] = df['close'].rolling(window).std()
            df[f'skew_{window}d'] = df['close'].rolling(window).skew()
            df[f'kurt_{window}d'] = df['close'].rolling(window).kurt()
        
        # Z-scores
        for window in [20, 50]:
            mean = df['close'].rolling(window).mean()
            std = df['close'].rolling(window).std()
            df[f'zscore_{window}d'] = (df['close'] - mean) / std
        
        return df
    
    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features"""
        # Bid-ask spread proxies
        df['hl_spread'] = (df['high'] - df['low']) / df['close']
        df['co_spread'] = abs(df['close'] - df['open']) / df['close']
        
        # Price impact
        df['price_impact'] = abs(df['close'].shift(1) - df['open']) / df['close'].shift(1)
        
        # Gaps
        df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        df['gap_filled'] = (df['gap'] > 0) & (df['low'] <= df['close'].shift(1))
        
        return df
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, k: int = 20) -> List[str]:
        """Select top k features using statistical tests"""
        # Remove non-numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_cols]
        
        # Handle missing values
        X_numeric = X_numeric.fillna(X_numeric.median())
        
        # Feature selection
        selector = SelectKBest(score_func=f_regression, k=min(k, len(numeric_cols)))
        selector.fit(X_numeric, y)
        
        selected_features = X_numeric.columns[selector.get_support()].tolist()
        self.feature_names = selected_features
        
        return selected_features


class MLPredictor:
    """Advanced ML predictor for market signals"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.feature_engineer = FeatureEngineer()
        self.model = None
        self.scaler = RobustScaler()
        self.is_trained = False
        self.performance: Optional[ModelPerformance] = None
        
        # Model cache directory
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)
    
    async def train(self, data: pd.DataFrame, symbol: str) -> bool:
        """Train the ML model"""
        try:
            logger.info(f"Training {self.config.model_type.value} model for {symbol}")
            
            # Feature engineering
            features_df = self.feature_engineer.create_features(data)
            
            # Create target variable
            target = self._create_target(features_df)
            
            # Feature selection
            feature_cols = self.feature_engineer.select_features(
                features_df, target, self.config.feature_count
            )
            
            X = features_df[feature_cols].fillna(features_df[feature_cols].median())
            y = target
            
            # Remove any remaining NaN values
            mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[mask]
            y = y[mask]
            
            if len(X) < 50:
                logger.warning(f"Insufficient data for training: {len(X)} samples")
                return False
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train-validation split (time series aware)
            split_idx = int(len(X) * (1 - self.config.validation_split))
            X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Initialize and train model
            self.model = self._initialize_model()
            
            if self.config.model_type in [ModelType.ARIMA, ModelType.GARCH]:
                # Time series models
                self._train_time_series_model(y_train)
            else:
                # ML models
                self.model.fit(X_train, y_train)
            
            # Evaluate model
            if len(X_val) > 0:
                predictions = self.predict_batch(X_val)
                self.performance = self._calculate_performance(y_val, predictions)
            
            self.is_trained = True
            
            # Save model
            await self._save_model(symbol)
            
            logger.info(f"Model training completed for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error training model for {symbol}: {e}")
            return False
    
    def _create_target(self, df: pd.DataFrame) -> pd.Series:
        """Create target variable based on prediction horizon"""
        if self.config.horizon == PredictionHorizon.INTRADAY:
            periods = 1
        elif self.config.horizon == PredictionHorizon.DAILY:
            periods = 1
        elif self.config.horizon == PredictionHorizon.WEEKLY:
            periods = 5
        else:  # MONTHLY
            periods = 20
        
        # Future returns
        future_returns = df['close'].pct_change(periods).shift(-periods)
        return future_returns
    
    def _initialize_model(self):
        """Initialize the ML model based on configuration"""
        hyperparams = self.config.hyperparams
        
        if self.config.model_type == ModelType.LINEAR:
            return Ridge(alpha=hyperparams.get('alpha', 1.0))
        
        elif self.config.model_type == ModelType.RANDOM_FOREST:
            return RandomForestRegressor(
                n_estimators=hyperparams.get('n_estimators', 100),
                max_depth=hyperparams.get('max_depth', 10),
                random_state=42
            )
        
        elif self.config.model_type == ModelType.GRADIENT_BOOSTING:
            return GradientBoostingRegressor(
                n_estimators=hyperparams.get('n_estimators', 100),
                learning_rate=hyperparams.get('learning_rate', 0.1),
                max_depth=hyperparams.get('max_depth', 6),
                random_state=42
            )
        
        elif self.config.model_type == ModelType.XGBOOST:
            return xgb.XGBRegressor(
                n_estimators=hyperparams.get('n_estimators', 100),
                learning_rate=hyperparams.get('learning_rate', 0.1),
                max_depth=hyperparams.get('max_depth', 6),
                random_state=42
            )
        
        elif self.config.model_type == ModelType.LIGHTGBM:
            return lgb.LGBMRegressor(
                n_estimators=hyperparams.get('n_estimators', 100),
                learning_rate=hyperparams.get('learning_rate', 0.1),
                max_depth=hyperparams.get('max_depth', 6),
                random_state=42,
                verbose=-1
            )
        
        elif self.config.model_type == ModelType.SVM:
            return SVR(
                kernel=hyperparams.get('kernel', 'rbf'),
                C=hyperparams.get('C', 1.0),
                gamma=hyperparams.get('gamma', 'scale')
            )
        
        else:
            # Default to Random Forest
            return RandomForestRegressor(n_estimators=100, random_state=42)
    
    def _train_time_series_model(self, y_train: pd.Series):
        """Train time series specific models"""
        if self.config.model_type == ModelType.ARIMA:
            self.model = ARIMA(y_train, order=(1, 1, 1)).fit()
        elif self.config.model_type == ModelType.GARCH:
            # Convert returns to percentage for GARCH
            returns_pct = y_train * 100
            self.model = arch_model(returns_pct, vol='Garch', p=1, q=1).fit(disp='off')
    
    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Make batch predictions"""
        if not self.is_trained or self.model is None:
            return np.zeros(len(X))
        
        try:
            if self.config.model_type in [ModelType.ARIMA, ModelType.GARCH]:
                # Time series models
                forecast = self.model.forecast(steps=len(X))
                return np.array(forecast).flatten()
            else:
                # ML models
                return self.model.predict(X)
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return np.zeros(len(X))
    
    async def predict_single(self, features: Dict[str, float], symbol: str) -> Optional[PredictionResult]:
        """Make single prediction"""
        if not self.is_trained:
            await self._load_model(symbol)
        
        if not self.is_trained:
            return None
        
        try:
            # Prepare features
            feature_vector = np.array([[
                features.get(name, 0.0) for name in self.feature_engineer.feature_names
            ]])
            
            # Scale features
            feature_vector_scaled = self.scaler.transform(feature_vector)
            
            # Make prediction
            predicted_return = self.predict_batch(feature_vector_scaled)[0]
            
            # Convert to price prediction (assuming features contain current price)
            current_price = features.get('close', 100.0)
            predicted_price = current_price * (1 + predicted_return)
            
            # Determine direction and probabilities
            probability_up = max(0.5, min(0.9, 0.5 + predicted_return * 10))
            probability_down = 1 - probability_up
            
            predicted_direction = SignalType.BUY if predicted_return > 0 else SignalType.SELL
            
            # Confidence based on model performance and prediction magnitude
            base_confidence = 0.6 if self.performance else 0.5
            magnitude_boost = min(0.3, abs(predicted_return) * 20)
            prediction_confidence = base_confidence + magnitude_boost
            
            # Feature importance (simplified)
            feature_importance = {}
            if hasattr(self.model, 'feature_importances_'):
                for name, importance in zip(self.feature_engineer.feature_names, 
                                          self.model.feature_importances_):
                    feature_importance[name] = float(importance)
            
            return PredictionResult(
                symbol=symbol,
                predicted_price=predicted_price,
                prediction_confidence=prediction_confidence,
                prediction_horizon=self.config.horizon,
                predicted_direction=predicted_direction,
                probability_up=probability_up,
                probability_down=probability_down,
                feature_importance=feature_importance,
                model_accuracy=self.performance.r2 if self.performance else 0.0,
                timestamp=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            logger.error(f"Error making single prediction: {e}")
            return None
    
    def _calculate_performance(self, y_true: pd.Series, y_pred: np.ndarray) -> ModelPerformance:
        """Calculate model performance metrics"""
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Direction accuracy
        direction_true = (y_true > 0).astype(int)
        direction_pred = (y_pred > 0).astype(int)
        accuracy = (direction_true == direction_pred).mean()
        
        # Precision/Recall for buy signals
        buy_signals_true = direction_true == 1
        buy_signals_pred = direction_pred == 1
        
        precision = (buy_signals_true & buy_signals_pred).sum() / buy_signals_pred.sum() if buy_signals_pred.sum() > 0 else 0
        recall = (buy_signals_true & buy_signals_pred).sum() / buy_signals_true.sum() if buy_signals_true.sum() > 0 else 0
        
        # Financial metrics (simplified)
        returns = y_true * direction_pred
        sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0
        
        cumulative_returns = (1 + returns).cumprod()
        max_drawdown = ((cumulative_returns / cumulative_returns.expanding().max()) - 1).min()
        
        win_rate = (returns > 0).mean()
        
        return ModelPerformance(
            mse=mse,
            mae=mae,
            r2=r2,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=abs(max_drawdown),
            win_rate=win_rate
        )
    
    async def _save_model(self, symbol: str):
        """Save trained model to disk"""
        model_path = self.model_dir / f"{symbol}_{self.config.model_type.value}_{self.config.horizon.value}.joblib"
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_engineer.feature_names,
            'config': asdict(self.config),
            'performance': asdict(self.performance) if self.performance else None,
            'trained_at': datetime.now(timezone.utc)
        }
        
        joblib.dump(model_data, model_path)
        
        # Cache model metadata
        cache = get_cache()
        if cache:
            await cache.set(
                "ml_models", 
                f"{symbol}_{self.config.model_type.value}_{self.config.horizon.value}",
                {
                    'symbol': symbol,
                    'model_type': self.config.model_type.value,
                    'horizon': self.config.horizon.value,
                    'performance': asdict(self.performance) if self.performance else None,
                    'trained_at': datetime.now(timezone.utc).isoformat()
                },
                ttl=86400  # 24 hours
            )
    
    async def _load_model(self, symbol: str) -> bool:
        """Load trained model from disk"""
        model_path = self.model_dir / f"{symbol}_{self.config.model_type.value}_{self.config.horizon.value}.joblib"
        
        if not model_path.exists():
            return False
        
        try:
            model_data = joblib.load(model_path)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_engineer.feature_names = model_data['feature_names']
            
            if model_data.get('performance'):
                self.performance = ModelPerformance(**model_data['performance'])
            
            self.is_trained = True
            return True
            
        except Exception as e:
            logger.error(f"Error loading model for {symbol}: {e}")
            return False


class EnsemblePredictor:
    """Ensemble of multiple ML models for robust predictions"""
    
    def __init__(self, configs: List[ModelConfig]):
        self.predictors = [MLPredictor(config) for config in configs]
        self.weights = None
    
    async def train(self, data: pd.DataFrame, symbol: str) -> bool:
        """Train all ensemble models"""
        results = []
        for predictor in self.predictors:
            success = await predictor.train(data, symbol)
            results.append(success)
        
        # Calculate ensemble weights based on performance
        self._calculate_weights()
        
        return any(results)
    
    def _calculate_weights(self):
        """Calculate ensemble weights based on model performance"""
        weights = []
        for predictor in self.predictors:
            if predictor.performance and predictor.performance.r2 > 0:
                # Weight based on R² score
                weights.append(max(0.1, predictor.performance.r2))
            else:
                weights.append(0.1)  # Minimum weight
        
        # Normalize weights
        total_weight = sum(weights)
        self.weights = [w / total_weight for w in weights] if total_weight > 0 else [1.0 / len(weights)] * len(weights)
    
    async def predict(self, features: Dict[str, float], symbol: str) -> Optional[PredictionResult]:
        """Make ensemble prediction"""
        predictions = []
        
        # Get predictions from all models
        for predictor in self.predictors:
            pred = await predictor.predict_single(features, symbol)
            if pred:
                predictions.append(pred)
        
        if not predictions:
            return None
        
        # Combine predictions using weights
        return self._combine_predictions(predictions, symbol)
    
    def _combine_predictions(self, predictions: List[PredictionResult], symbol: str) -> PredictionResult:
        """Combine predictions from multiple models"""
        if not predictions:
            return None
        
        if len(predictions) == 1:
            return predictions[0]
        
        # Use equal weights if not calculated
        if not self.weights or len(self.weights) != len(predictions):
            weights = [1.0 / len(predictions)] * len(predictions)
        else:
            weights = self.weights[:len(predictions)]
        
        # Weighted average of predictions
        weighted_price = sum(p.predicted_price * w for p, w in zip(predictions, weights))
        weighted_confidence = sum(p.prediction_confidence * w for p, w in zip(predictions, weights))
        weighted_prob_up = sum(p.probability_up * w for p, w in zip(predictions, weights))
        
        # Majority vote for direction
        buy_votes = sum(1 for p in predictions if p.predicted_direction == SignalType.BUY)
        predicted_direction = SignalType.BUY if buy_votes > len(predictions) / 2 else SignalType.SELL
        
        # Average model accuracy
        avg_accuracy = sum(p.model_accuracy for p in predictions) / len(predictions)
        
        # Combine feature importance
        combined_importance = {}
        for pred in predictions:
            for feature, importance in pred.feature_importance.items():
                combined_importance[feature] = combined_importance.get(feature, 0) + importance
        
        # Normalize combined importance
        total_importance = sum(combined_importance.values())
        if total_importance > 0:
            combined_importance = {k: v / total_importance for k, v in combined_importance.items()}
        
        return PredictionResult(
            symbol=symbol,
            predicted_price=weighted_price,
            prediction_confidence=weighted_confidence,
            prediction_horizon=predictions[0].prediction_horizon,
            predicted_direction=predicted_direction,
            probability_up=weighted_prob_up,
            probability_down=1 - weighted_prob_up,
            feature_importance=combined_importance,
            model_accuracy=avg_accuracy,
            timestamp=datetime.now(timezone.utc)
        )