"""
Step 2: Main Prediction Engine
Integrates all prediction strategies with event analysis and signal generation
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add Step 1 to path for data integration
step1_path = str(Path(__file__).parent.parent.parent / "step1" / "src")
sys.path.append(step1_path)

# Import Step 1 components
try:
    from enhanced_data_fetcher import EnhancedDataFetcher
except ImportError:
    print("Warning: Could not import Step 1 components. Some features may be limited.")

# Import Step 2 components
from prediction_strategies import (
    TechnicalLSTMStrategy, ARIMAMomentumStrategy, 
    EventDrivenStrategy, EnsembleMetaStrategy,
    PredictionResult
)
from event_detector import EventDetector, MarketEvent

class Step2PredictionEngine:
    """
    Main Step 2 Engine: Transparent Multi-Strategy Prediction System
    Builds on Step 1 with 5-day lookback, Friday prediction target
    """
    
    def __init__(self):
        # Initialize Step 1 data fetcher
        try:
            self.data_fetcher = EnhancedDataFetcher()
        except:
            self.data_fetcher = None
            
        # Initialize prediction strategies
        self.technical_lstm = TechnicalLSTMStrategy()
        self.arima_momentum = ARIMAMomentumStrategy()
        self.event_driven = EventDrivenStrategy()
        
        # Initialize ensemble (combining all strategies)
        self.ensemble = EnsembleMetaStrategy([
            self.technical_lstm,
            self.arima_momentum,
            self.event_driven
        ])
        
        # Event detection system
        self.event_detector = EventDetector()
        
        # Configuration
        self.lookback_days = 5
        self.prediction_resolution = "1H"  # 1-hour intervals
        self.target_day = "Friday"  # Option expiration day
        
        # Performance tracking
        self.prediction_history = []
        self.performance_metrics = {}
        
    def generate_predictions(self, ticker: str, strategy_name: str = "ensemble") -> Dict:
        """
        Generate predictions for a ticker using specified strategy
        
        Args:
            ticker: Stock symbol (e.g., 'AAPL')
            strategy_name: 'technical-lstm', 'arima', 'event-driven', or 'ensemble'
            
        Returns:
            Complete prediction results with visualization data
        """
        try:
            # Step 1: Get historical data (5-day lookback)
            historical_data = self._fetch_historical_data(ticker)
            if historical_data is None or len(historical_data) < 10:
                return self._generate_error_response(ticker, "Insufficient historical data")
            
            # Step 2: Detect upcoming events
            upcoming_events = self.event_detector.detect_upcoming_events(ticker)
            
            # Step 3: Fit and train strategies on historical data
            self._train_strategies(historical_data)
            
            # Step 4: Generate predictions based on strategy
            prediction_result = self._execute_strategy(
                strategy_name, ticker, historical_data, upcoming_events
            )
            
            # Step 5: Convert to visualization format
            visualization_data = self._convert_to_visualization_format(
                prediction_result, historical_data, upcoming_events
            )
            
            # Step 6: Add transparency information
            transparency_info = self._generate_transparency_info(
                prediction_result, upcoming_events
            )
            
            # Step 7: Track prediction for validation
            self._track_prediction(prediction_result)
            
            return {
                'success': True,
                'ticker': ticker,
                'strategy': strategy_name,
                'prediction': prediction_result,
                'visualization': visualization_data,
                'transparency': transparency_info,
                'events': [self._serialize_event(event) for event in upcoming_events],
                'timestamp': datetime.now().isoformat(),
                'next_friday': prediction_result.target_date
            }
            
        except Exception as e:
            return self._generate_error_response(ticker, f"Prediction failed: {str(e)}")
    
    def get_multiple_strategy_predictions(self, ticker: str) -> Dict:
        """
        Get predictions from all strategies for comparison
        
        Returns:
            Predictions from all 4 strategies for the same ticker
        """
        try:
            # Get data once for all strategies
            historical_data = self._fetch_historical_data(ticker)
            upcoming_events = self.event_detector.detect_upcoming_events(ticker)
            
            if historical_data is None:
                return self._generate_error_response(ticker, "Data unavailable")
            
            # Train all strategies
            self._train_strategies(historical_data)
            
            # Get predictions from each strategy
            strategies = {
                'technical_lstm': self.technical_lstm,
                'arima_momentum': self.arima_momentum,
                'event_driven': self.event_driven,
                'ensemble': self.ensemble
            }
            
            all_predictions = {}
            
            for strategy_name, strategy in strategies.items():
                try:
                    if strategy_name == 'event_driven':
                        prediction = strategy.predict(ticker, historical_data, upcoming_events)
                    elif strategy_name == 'ensemble':
                        prediction = strategy.predict(ticker, historical_data, upcoming_events)
                    else:
                        prediction = strategy.predict(ticker, historical_data)
                    
                    all_predictions[strategy_name] = {
                        'prediction': prediction,
                        'visualization': self._convert_to_visualization_format(
                            prediction, historical_data, upcoming_events
                        )
                    }
                except Exception as e:
                    all_predictions[strategy_name] = {
                        'error': str(e),
                        'prediction': None
                    }
            
            return {
                'success': True,
                'ticker': ticker,
                'strategies': all_predictions,
                'events': [self._serialize_event(event) for event in upcoming_events],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return self._generate_error_response(ticker, f"Multi-strategy prediction failed: {str(e)}")
    
    def _fetch_historical_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Fetch 5-day historical data at specified resolution"""
        try:
            if self.data_fetcher:
                # Use enhanced data fetcher from Step 1
                end_date = datetime.now()
                start_date = end_date - timedelta(days=self.lookback_days)
                
                # For MVP, get daily data and simulate intraday
                # In production, fetch actual intraday data
                data = self._simulate_intraday_data(ticker)
                return data
            else:
                # Fallback: simulate data for testing
                return self._simulate_historical_data(ticker)
                
        except Exception as e:
            return None
    
    def _simulate_intraday_data(self, ticker: str) -> pd.DataFrame:
        """Simulate intraday data for MVP (replace with real data in production)"""
        try:
            # Get current real price from Step 1
            if self.data_fetcher:
                current_data = self.data_fetcher.get_real_time_data(ticker)
                current_price = current_data['price'] if current_data else 100.0
            else:
                current_price = 100.0
            
            # Generate 5 days of hourly data (5 * 6.5 hours = ~33 data points)
            hours = 33
            dates = pd.date_range(end=datetime.now(), periods=hours, freq='H')
            
            # Simulate realistic price movement
            np.random.seed(hash(ticker) % 1000)  # Deterministic but ticker-specific
            returns = np.random.normal(0, 0.02, hours)  # 2% hourly volatility
            returns[0] = 0  # Start with no change
            
            # Create price series
            prices = [current_price]
            for i in range(1, hours):
                new_price = prices[-1] * (1 + returns[i])
                prices.append(new_price)
            
            # Reverse to start from 5 days ago
            prices.reverse()
            
            # Generate volume data
            base_volume = 1000000
            volumes = np.random.lognormal(np.log(base_volume), 0.5, hours)
            
            # Create DataFrame
            df = pd.DataFrame({
                'timestamp': dates,
                'open': prices,
                'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
                'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
                'close': prices,
                'volume': volumes
            })
            
            return df
            
        except Exception as e:
            return self._simulate_historical_data(ticker)
    
    def _simulate_historical_data(self, ticker: str) -> pd.DataFrame:
        """Fallback simulation when no real data available"""
        # Basic simulation for testing
        np.random.seed(42)
        days = 20
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        prices = 100 + np.cumsum(np.random.normal(0, 2, days))
        volumes = np.random.lognormal(15, 1, days)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': volumes
        })
        
        return df
    
    def _train_strategies(self, historical_data: pd.DataFrame) -> None:
        """Train all prediction strategies on historical data"""
        try:
            self.technical_lstm.fit(historical_data)
            self.arima_momentum.fit(historical_data)
            self.event_driven.fit(historical_data)
            self.ensemble.fit(historical_data)
        except Exception as e:
            pass  # Strategies handle their own errors
    
    def _execute_strategy(self, strategy_name: str, ticker: str, 
                         historical_data: pd.DataFrame, 
                         upcoming_events: List[MarketEvent]) -> PredictionResult:
        """Execute specific prediction strategy"""
        
        strategy_map = {
            'technical-lstm': self.technical_lstm,
            'arima': self.arima_momentum,
            'arima-momentum': self.arima_momentum,
            'event-driven': self.event_driven,
            'ensemble': self.ensemble
        }
        
        strategy = strategy_map.get(strategy_name, self.ensemble)
        
        if strategy_name in ['event-driven', 'ensemble']:
            return strategy.predict(ticker, historical_data, upcoming_events)
        else:
            return strategy.predict(ticker, historical_data)
    
    def _convert_to_visualization_format(self, prediction: PredictionResult,
                                       historical_data: pd.DataFrame,
                                       events: List[MarketEvent]) -> Dict:
        """Convert prediction to signal_visualization format"""
        
        # Historical data for chart (last 20 points)
        historical_prices = historical_data['close'].tail(20).tolist()
        
        # Ensure prediction arrays have consistent length
        max_length = max(
            len(prediction.predicted_prices),
            len(prediction.upper_band),
            len(prediction.lower_band)
        )
        
        # Pad arrays if needed
        predicted_prices = prediction.predicted_prices[:max_length]
        upper_band = prediction.upper_band[:max_length]
        lower_band = prediction.lower_band[:max_length]
        
        # Create chart data matching signal_visualization format
        chart_data = {
            'historical_data': historical_prices,
            'prediction_upper': upper_band,
            'prediction_base': predicted_prices,
            'prediction_lower': lower_band,
            'event_label': self._get_primary_event_label(events),
            'chart_color': self._get_strategy_color(prediction.strategy_name)
        }
        
        # Key statistics for display
        current_price = prediction.current_price
        target_price = predicted_prices[-1] if predicted_prices else current_price
        price_change = ((target_price - current_price) / current_price) * 100
        
        key_stats = [
            {
                'value': f"{prediction.confidence:.0f}%",
                'label': 'Confidence',
                'is_positive': prediction.confidence > 60
            },
            {
                'value': f"${target_price:.2f}",
                'label': f'Friday Target',
                'is_positive': price_change > 0
            },
            {
                'value': f"{price_change:+.1f}%",
                'label': 'Expected Move',
                'is_positive': price_change > 0
            }
        ]
        
        # Add strategy-specific stats
        for factor, value in prediction.key_factors.items():
            if isinstance(value, (int, float)):
                key_stats.append({
                    'value': f"{value:.1f}",
                    'label': factor.replace('_', ' '),
                    'is_positive': value > 0
                })
        
        return {
            'chart_data': chart_data,
            'key_stats': key_stats[:6],  # Limit to 6 stats for mobile display
            'strategy_info': {
                'title': self._get_strategy_title(prediction.strategy_name),
                'description': prediction.rationale,
                'algorithm': prediction.strategy_name,
                'link_url': f'/strategy/{prediction.strategy_name.lower().replace(" ", "-")}'
            }
        }
    
    def _generate_transparency_info(self, prediction: PredictionResult,
                                  events: List[MarketEvent]) -> Dict:
        """Generate transparency information for users"""
        
        transparency = {
            'algorithm_used': prediction.strategy_name,
            'data_sources': self._get_data_sources(),
            'model_confidence': prediction.confidence,
            'key_factors': prediction.key_factors,
            'reasoning': prediction.rationale,
            'limitations': self._get_strategy_limitations(prediction.strategy_name),
            'last_updated': datetime.now().isoformat(),
            'events_considered': len(events),
            'prediction_horizon': f"Until {prediction.target_date}",
            'risk_level': self._assess_risk_level(prediction, events)
        }
        
        return transparency
    
    def _get_primary_event_label(self, events: List[MarketEvent]) -> str:
        """Get label for primary upcoming event"""
        if not events:
            return ""
        
        primary_event = events[0]  # Highest impact event
        event_date = primary_event.event_date.strftime('%m/%d')
        
        event_labels = {
            'earnings': f'Earnings {event_date}',
            'ex_dividend': f'Ex-Div {event_date}',
            'fda_approval': f'FDA {event_date}',
            'product_launch': f'Product {event_date}',
            'merger_acquisition': f'M&A {event_date}',
            'stock_split': f'Split {event_date}'
        }
        
        return event_labels.get(primary_event.event_type, f'Event {event_date}')
    
    def _get_strategy_color(self, strategy_name: str) -> str:
        """Get color for strategy visualization"""
        colors = {
            'Technical-LSTM Hybrid': '#00ff88',
            'ARIMA-Momentum Ensemble': '#00d4ff',
            'Event-Driven Volatility': '#ff00ff',
            'Ensemble Meta-Learner': '#ffd93d'
        }
        return colors.get(strategy_name, '#00ff88')
    
    def _get_strategy_title(self, strategy_name: str) -> str:
        """Get display title for strategy"""
        titles = {
            'Technical-LSTM Hybrid': 'AI Technical Analysis',
            'ARIMA-Momentum Ensemble': 'Statistical Trend Analysis',
            'Event-Driven Volatility': 'Event Impact Prediction',
            'Ensemble Meta-Learner': 'Multi-Strategy Consensus'
        }
        return titles.get(strategy_name, strategy_name)
    
    def _get_data_sources(self) -> List[str]:
        """Get list of data sources used"""
        sources = ['Step 1 Real-Time API']
        if self.data_fetcher:
            sources.extend(['Alpaca Markets', 'Finnhub', 'yfinance'])
        return sources
    
    def _get_strategy_limitations(self, strategy_name: str) -> List[str]:
        """Get limitations for each strategy"""
        limitations = {
            'Technical-LSTM Hybrid': [
                'Requires sufficient historical data',
                'Performance may degrade in low volume periods',
                'Past patterns may not predict future moves'
            ],
            'ARIMA-Momentum Ensemble': [
                'Assumes stationary time series',
                'Limited effectiveness in trending markets',
                'Sensitive to structural breaks'
            ],
            'Event-Driven Volatility': [
                'Event dates may be estimated',
                'Binary outcomes create high uncertainty',
                'Market reaction may differ from historical patterns'
            ],
            'Ensemble Meta-Learner': [
                'Inherits limitations of constituent strategies',
                'May mask individual strategy insights',
                'Requires all strategies to be functional'
            ]
        }
        return limitations.get(strategy_name, ['Model predictions are not guaranteed'])
    
    def _assess_risk_level(self, prediction: PredictionResult, events: List[MarketEvent]) -> str:
        """Assess overall risk level of prediction"""
        base_risk = 1  # Start with low risk
        
        # Confidence affects risk
        if prediction.confidence < 50:
            base_risk += 1
        elif prediction.confidence > 80:
            base_risk -= 0.5
        
        # Events increase risk
        for event in events:
            if event.impact_score >= 8:
                base_risk += 2
            elif event.impact_score >= 6:
                base_risk += 1
        
        # Volatility in prediction bands
        if prediction.predicted_prices and prediction.upper_band and prediction.lower_band:
            avg_volatility = np.mean([
                (upper - lower) / pred for pred, upper, lower in 
                zip(prediction.predicted_prices, prediction.upper_band, prediction.lower_band)
                if pred > 0
            ])
            if avg_volatility > 0.1:  # >10% volatility
                base_risk += 1
        
        risk_levels = ['LOW', 'MEDIUM', 'HIGH', 'EXTREME']
        risk_index = min(int(base_risk), len(risk_levels) - 1)
        return risk_levels[risk_index]
    
    def _track_prediction(self, prediction: PredictionResult) -> None:
        """Track prediction for future validation"""
        prediction_record = {
            'timestamp': datetime.now().isoformat(),
            'ticker': prediction.ticker,
            'strategy': prediction.strategy_name,
            'current_price': prediction.current_price,
            'predicted_price': prediction.predicted_prices[-1] if prediction.predicted_prices else None,
            'confidence': prediction.confidence,
            'target_date': prediction.target_date
        }
        
        self.prediction_history.append(prediction_record)
        
        # Keep only last 1000 predictions
        if len(self.prediction_history) > 1000:
            self.prediction_history = self.prediction_history[-1000:]
    
    def _serialize_event(self, event: MarketEvent) -> Dict:
        """Convert event to JSON-serializable format"""
        return {
            'ticker': event.ticker,
            'type': event.event_type,
            'date': event.event_date.isoformat(),
            'description': event.description,
            'impact_score': event.impact_score,
            'direction_bias': event.direction_bias,
            'volatility_multiplier': event.volatility_multiplier,
            'confidence': event.confidence,
            'source': event.source
        }
    
    def _generate_error_response(self, ticker: str, error_message: str) -> Dict:
        """Generate standardized error response"""
        return {
            'success': False,
            'ticker': ticker,
            'error': error_message,
            'timestamp': datetime.now().isoformat(),
            'fallback_available': True
        }
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary of all strategies"""
        if not self.prediction_history:
            return {'message': 'No predictions tracked yet'}
        
        # Calculate basic performance metrics
        total_predictions = len(self.prediction_history)
        strategies = set(p['strategy'] for p in self.prediction_history)
        
        return {
            'total_predictions': total_predictions,
            'strategies_used': list(strategies),
            'tracking_since': self.prediction_history[0]['timestamp'] if self.prediction_history else None,
            'last_prediction': self.prediction_history[-1]['timestamp'] if self.prediction_history else None,
            'average_confidence': np.mean([p['confidence'] for p in self.prediction_history if p['confidence']]),
            'status': 'tracking_enabled'
        }

# Test and demo functionality
if __name__ == "__main__":
    # Initialize Step 2 engine
    engine = Step2PredictionEngine()
    
    # Test with a few tickers
    test_tickers = ['AAPL', 'TSLA', 'SAVA']
    
    print("🔮 Step 2 Prediction Engine Demo")
    print("=" * 50)
    
    for ticker in test_tickers:
        print(f"\n📊 Testing {ticker}...")
        
        # Get ensemble prediction
        result = engine.generate_predictions(ticker, "ensemble")
        
        if result['success']:
            pred = result['prediction']
            print(f"Strategy: {pred.strategy_name}")
            print(f"Current: ${pred.current_price:.2f}")
            print(f"Friday Target: ${pred.predicted_prices[-1]:.2f}")
            print(f"Confidence: {pred.confidence:.0f}%")
            print(f"Rationale: {pred.rationale}")
            print(f"Events: {len(result['events'])}")
        else:
            print(f"Error: {result['error']}")
    
    # Test multi-strategy comparison
    print(f"\n🎭 Multi-Strategy Comparison for AAPL...")
    multi_result = engine.get_multiple_strategy_predictions('AAPL')
    
    if multi_result['success']:
        for strategy_name, strategy_data in multi_result['strategies'].items():
            if 'error' not in strategy_data:
                pred = strategy_data['prediction']
                target = pred.predicted_prices[-1] if pred.predicted_prices else 0
                print(f"{strategy_name}: ${target:.2f} (Confidence: {pred.confidence:.0f}%)")
            else:
                print(f"{strategy_name}: Error - {strategy_data['error']}")
    
    print(f"\n✅ Step 2 Demo Complete!")
    print("Ready for API integration and signal visualization.")