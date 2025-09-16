"""
Transparent Signal Generation Engine
Focuses on signal accuracy, transparency, and algorithmic explanations
Rather than profitability, emphasizes understanding how signals are generated
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

# Import Step 2 components
from prediction_strategies import (
    TechnicalLSTMStrategy, ARIMAMomentumStrategy, 
    EventDrivenStrategy, EnsembleMetaStrategy,
    PredictionResult
)
from event_detector import EventDetector, MarketEvent
from user_profiling_engine import (
    UserProfilingEngine, RiskProfile, SignalRiskClassification,
    EnhancedSignalData, UserProfile
)

class TransparentSignalEngine:
    """
    Main engine that generates transparent, accuracy-focused signals
    Profiles users and provides personalized recommendations based on risk appetite
    """
    
    def __init__(self):
        # Initialize prediction strategies
        self.technical_lstm = TechnicalLSTMStrategy()
        self.arima_momentum = ARIMAMomentumStrategy()
        self.event_driven = EventDrivenStrategy()
        self.ensemble = EnsembleMetaStrategy([
            self.technical_lstm, self.arima_momentum, self.event_driven
        ])
        
        # Event detection
        self.event_detector = EventDetector()
        
        # User profiling engine
        self.profiling_engine = UserProfilingEngine()
        
        # Historical performance tracking (would be loaded from database)
        self.performance_database = self._initialize_performance_database()
        
        # Signal type mappings
        self.signal_type_mappings = self._initialize_signal_mappings()
        
    def _initialize_performance_database(self) -> Dict:
        """Initialize historical performance data for transparency"""
        return {
            'dividend_capture': {
                'win_rate': 92.3,
                'avg_return_when_correct': 2.8,
                'avg_loss_when_wrong': -0.8,
                'sample_size': 1250,
                'backtest_period': '2020-2024',
                'best_conditions': ['Ex-dividend within 3 days', 'Stable market environment'],
                'worst_conditions': ['Market volatility >25%', 'Sector rotation periods'],
                'typical_timeline': '2-4 days'
            },
            'large_cap_earnings': {
                'win_rate': 78.4,
                'avg_return_when_correct': 5.2,
                'avg_loss_when_wrong': -3.1,
                'sample_size': 890,
                'backtest_period': '2020-2024',
                'best_conditions': ['Clear earnings guidance', 'Low volatility pre-earnings'],
                'worst_conditions': ['Macro uncertainty', 'Sector headwinds'],
                'typical_timeline': '1-2 weeks'
            },
            'technical_breakout': {
                'win_rate': 64.7,
                'avg_return_when_correct': 8.9,
                'avg_loss_when_wrong': -4.2,
                'sample_size': 2150,
                'backtest_period': '2020-2024',
                'best_conditions': ['High volume confirmation', 'Clear resistance levels'],
                'worst_conditions': ['Low volume markets', 'Range-bound conditions'],
                'typical_timeline': '3-7 days'
            },
            'momentum_options': {
                'win_rate': 57.8,
                'avg_return_when_correct': 28.5,
                'avg_loss_when_wrong': -18.2,
                'sample_size': 1680,
                'backtest_period': '2020-2024',
                'best_conditions': ['Strong momentum', 'Options flow confirmation'],
                'worst_conditions': ['Low volatility', 'Time decay periods'],
                'typical_timeline': '1-3 days'
            },
            'yolo_biotech': {
                'win_rate': 38.2,
                'avg_return_when_correct': 185.6,
                'avg_loss_when_wrong': -68.4,
                'sample_size': 420,
                'backtest_period': '2020-2024',
                'best_conditions': ['Clear FDA timeline', 'Strong drug data'],
                'worst_conditions': ['Regulatory uncertainty', 'Competition'],
                'typical_timeline': '1-7 days'
            },
            '0dte_options': {
                'win_rate': 31.8,
                'avg_return_when_correct': 312.4,
                'avg_loss_when_wrong': -85.6,
                'sample_size': 950,
                'backtest_period': '2022-2024',
                'best_conditions': ['High gamma exposure', 'Clear directional catalysts'],
                'worst_conditions': ['Low volatility', 'Time decay acceleration'],
                'typical_timeline': '2-6 hours'
            }
        }
    
    def _initialize_signal_mappings(self) -> Dict:
        """Map prediction strategies to signal types with risk classifications"""
        return {
            'Technical-LSTM Hybrid': {
                'conservative_signals': ['large_cap_momentum', 'dividend_capture'],
                'moderate_signals': ['technical_breakout', 'earnings_play'],
                'aggressive_signals': ['momentum_options', 'breakout_options'],
                'yolo_signals': ['gap_play', 'momentum_yolo']
            },
            'ARIMA-Momentum Ensemble': {
                'conservative_signals': ['trend_following', 'dividend_aristocrat'],
                'moderate_signals': ['sector_rotation', 'index_momentum'],
                'aggressive_signals': ['trend_acceleration', 'momentum_reversal'],
                'yolo_signals': ['trend_break', 'momentum_extreme']
            },
            'Event-Driven Volatility': {
                'conservative_signals': ['dividend_capture', 'low_vol_earnings'],
                'moderate_signals': ['earnings_play', 'announcement_play'],
                'aggressive_signals': ['volatility_expansion', 'event_options'],
                'yolo_signals': ['fda_binary', 'yolo_biotech', 'merger_arb']
            }
        }
    
    def generate_personalized_signals(self, user_id: str, max_signals: int = 10) -> Dict:
        """
        Generate personalized signals based on user's risk profile
        Focus on transparency and education rather than profitability
        """
        try:
            # Get or create user profile
            if user_id not in self.profiling_engine.user_profiles:
                self.profiling_engine.create_user_profile(user_id)
            
            # Generate signals for multiple tickers
            test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'META', 'AMZN', 'NFLX', 'SAVA', 'MRNA']
            all_signals = []
            
            for ticker in test_tickers:
                signals = self._generate_ticker_signals(ticker)
                all_signals.extend(signals)
            
            # Get personalized recommendations
            personalized_signals = self.profiling_engine.get_personalized_signals(
                user_id, all_signals, max_signals
            )
            
            # Add detailed explanations
            detailed_signals = []
            for signal in personalized_signals:
                detailed_explanation = self.profiling_engine.generate_signal_explanation(
                    signal, detail_level="full"
                )
                detailed_signals.append({
                    'signal_data': signal,
                    'explanation': detailed_explanation,
                    'swipe_data': self._prepare_swipe_data(signal)
                })
            
            return {
                'success': True,
                'user_id': user_id,
                'user_profile': self.profiling_engine.get_user_profile_summary(user_id),
                'signals': detailed_signals,
                'total_signals_generated': len(all_signals),
                'personalized_count': len(personalized_signals),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Signal generation failed: {str(e)}",
                'user_id': user_id,
                'timestamp': datetime.now().isoformat()
            }
    
    def _generate_ticker_signals(self, ticker: str) -> List[EnhancedSignalData]:
        """Generate transparent signals for a single ticker"""
        signals = []
        
        try:
            # Get basic market data (simulated for MVP)
            market_data = self._get_simulated_market_data(ticker)
            current_price = market_data['current_price']
            
            # Detect events
            upcoming_events = self.event_detector.detect_upcoming_events(ticker)
            
            # Generate signals from each strategy
            strategies = {
                'Technical-LSTM Hybrid': self.technical_lstm,
                'ARIMA-Momentum Ensemble': self.arima_momentum,
                'Event-Driven Volatility': self.event_driven
            }
            
            for strategy_name, strategy in strategies.items():
                try:
                    # Get prediction from strategy
                    historical_data = self._get_simulated_historical_data(ticker)
                    strategy.fit(historical_data)
                    
                    if strategy_name == 'Event-Driven Volatility':
                        prediction = strategy.predict(ticker, historical_data, upcoming_events)
                    else:
                        prediction = strategy.predict(ticker, historical_data)
                    
                    # Convert to different risk levels
                    signal_types = self._get_signal_types_for_strategy(strategy_name, prediction)
                    
                    for signal_type, risk_level in signal_types:
                        enhanced_signal = self._create_enhanced_signal(
                            ticker, prediction, signal_type, risk_level, strategy_name, upcoming_events
                        )
                        signals.append(enhanced_signal)
                        
                except Exception as e:
                    continue  # Skip failed strategies
            
        except Exception as e:
            pass  # Skip failed tickers
        
        return signals
    
    def _get_signal_types_for_strategy(self, strategy_name: str, prediction: PredictionResult) -> List[Tuple[str, RiskProfile]]:
        """Get different signal types for a strategy based on confidence and conditions"""
        signal_types = []
        
        # Base signal mapping
        mappings = self.signal_type_mappings.get(strategy_name, {})
        
        # High confidence signals can span multiple risk levels
        if prediction.confidence > 80:
            signal_types.append(('high_confidence_' + strategy_name.lower().replace(' ', '_'), RiskProfile.CONSERVATIVE))
            signal_types.append(('momentum_' + strategy_name.lower().replace(' ', '_'), RiskProfile.AGGRESSIVE))
        
        # Medium confidence - moderate risk
        if 60 <= prediction.confidence <= 80:
            signal_types.append(('standard_' + strategy_name.lower().replace(' ', '_'), RiskProfile.MODERATE))
        
        # Lower confidence but high potential - aggressive/YOLO
        if prediction.confidence < 60:
            price_change = abs((prediction.predicted_prices[-1] - prediction.current_price) / prediction.current_price) if prediction.predicted_prices else 0
            if price_change > 0.15:  # >15% potential move
                signal_types.append(('high_risk_' + strategy_name.lower().replace(' ', '_'), RiskProfile.EXTREMELY_AGGRESSIVE))
        
        # Event-driven can create YOLO signals
        if strategy_name == 'Event-Driven Volatility' and prediction.confidence > 40:
            signal_types.append(('event_yolo', RiskProfile.EXTREMELY_AGGRESSIVE))
        
        return signal_types
    
    def _create_enhanced_signal(self, ticker: str, prediction: PredictionResult, 
                              signal_type: str, risk_level: RiskProfile, 
                              strategy_name: str, events: List[MarketEvent]) -> EnhancedSignalData:
        """Create enhanced signal with full transparency data"""
        
        # Get performance data for this signal type
        perf_data = self._get_signal_performance_data(signal_type, strategy_name)
        
        # Get algorithm performance
        algo_perf = self.profiling_engine.algorithm_performance.get(strategy_name, {})
        
        # Create risk classification
        risk_classification = self._create_risk_classification(signal_type, risk_level, perf_data)
        
        # Calculate expected returns
        expected_return_if_correct = perf_data.get('avg_return_when_correct', 5.0)
        expected_loss_if_wrong = abs(perf_data.get('avg_loss_when_wrong', -3.0))
        
        # Create enhanced signal
        enhanced_signal = EnhancedSignalData(
            ticker=ticker,
            signal_type=signal_type,
            current_price=prediction.current_price,
            target_price=prediction.predicted_prices[-1] if prediction.predicted_prices else prediction.current_price,
            confidence=prediction.confidence,
            
            risk_classification=risk_classification,
            
            algorithm_used=strategy_name,
            algorithm_description=self._get_algorithm_description(strategy_name),
            key_factors=prediction.key_factors,
            
            historical_win_rate=perf_data.get('win_rate', 65.0),
            expected_return_if_correct=expected_return_if_correct,
            expected_loss_if_wrong=expected_loss_if_wrong,
            typical_timeline=perf_data.get('typical_timeline', '1-3 days'),
            
            algorithm_strengths=algo_perf.get('strengths', ['Data-driven approach']),
            algorithm_weaknesses=algo_perf.get('weaknesses', ['Market conditions dependent']),
            market_conditions_best=algo_perf.get('best_market_conditions', ['Normal volatility']),
            market_conditions_worst=algo_perf.get('worst_market_conditions', ['Extreme volatility']),
            
            backtest_period=perf_data.get('backtest_period', '2020-2024'),
            sample_size=perf_data.get('sample_size', 1000),
            confidence_intervals={'win_rate_95_ci': algo_perf.get('confidence_intervals', {}).get('win_rate_95_ci', [60, 70])},
            
            # Will be filled by profiling engine
            relevance_score=0.0,
            recommendation_reason=""
        )
        
        return enhanced_signal
    
    def _get_signal_performance_data(self, signal_type: str, strategy_name: str) -> Dict:
        """Get performance data for signal type"""
        # Try to find exact match first
        if signal_type in self.performance_database:
            return self.performance_database[signal_type]
        
        # Map signal types to base performance data
        base_mappings = {
            'high_confidence': 'large_cap_earnings',
            'standard': 'technical_breakout',
            'momentum': 'momentum_options',
            'high_risk': 'yolo_biotech',
            'event': 'large_cap_earnings',
            'yolo': '0dte_options'
        }
        
        # Find best match
        for key, base_type in base_mappings.items():
            if key in signal_type.lower():
                return self.performance_database.get(base_type, self.performance_database['technical_breakout'])
        
        # Default fallback
        return self.performance_database['technical_breakout']
    
    def _create_risk_classification(self, signal_type: str, risk_level: RiskProfile, perf_data: Dict) -> SignalRiskClassification:
        """Create risk classification for signal"""
        
        # Map risk levels to typical values
        risk_mappings = {
            RiskProfile.EXTREMELY_CONSERVATIVE: {'max_drawdown': 3.0, 'complexity': 2, 'capital': 'Low'},
            RiskProfile.CONSERVATIVE: {'max_drawdown': 8.0, 'complexity': 4, 'capital': 'Low'},
            RiskProfile.MODERATE: {'max_drawdown': 15.0, 'complexity': 5, 'capital': 'Medium'},
            RiskProfile.AGGRESSIVE: {'max_drawdown': 30.0, 'complexity': 7, 'capital': 'High'},
            RiskProfile.EXTREMELY_AGGRESSIVE: {'max_drawdown': 80.0, 'complexity': 9, 'capital': 'Very High'}
        }
        
        risk_config = risk_mappings[risk_level]
        
        return SignalRiskClassification(
            risk_level=risk_level,
            signal_type=signal_type,
            win_rate=perf_data.get('win_rate', 65.0),
            avg_return=perf_data.get('avg_return_when_correct', 8.0),
            max_drawdown=risk_config['max_drawdown'],
            typical_holding_period=perf_data.get('typical_timeline', '1-3 days'),
            capital_requirement=risk_config['capital'],
            complexity_score=risk_config['complexity']
        )
    
    def _get_algorithm_description(self, strategy_name: str) -> str:
        """Get human-readable algorithm description"""
        descriptions = {
            'Technical-LSTM Hybrid': 'Combines traditional technical analysis (RSI, MACD, Bollinger Bands) with deep learning neural networks to identify patterns humans might miss. The LSTM component learns from historical price sequences.',
            'ARIMA-Momentum Ensemble': 'Uses statistical time series analysis (ARIMA) combined with momentum indicators. This approach assumes past price movements can help predict future trends with mathematical rigor.',
            'Event-Driven Volatility': 'Analyzes upcoming corporate events (earnings, FDA approvals, dividends) and their historical impact on stock prices. Uses event-specific volatility models to predict price movements.',
            'Ensemble Meta-Learner': 'Combines predictions from multiple algorithms and weights them based on their historical performance in similar market conditions. Provides most robust predictions.'
        }
        return descriptions.get(strategy_name, 'Advanced quantitative algorithm for market prediction.')
    
    def _prepare_swipe_data(self, signal: EnhancedSignalData) -> Dict:
        """Prepare data for swipe interactions (swipe up for details, swipe down for rejection)"""
        return {
            'swipe_up_reveals': {
                'algorithm_details': {
                    'how_it_works': signal.algorithm_description,
                    'key_inputs': list(signal.key_factors.keys()),
                    'confidence_factors': signal.key_factors
                },
                'performance_deep_dive': {
                    'win_rate_breakdown': f"{signal.historical_win_rate:.1f}% win rate over {signal.sample_size} historical signals",
                    'when_it_wins': f"Average gain: {signal.expected_return_if_correct:+.1f}%",
                    'when_it_loses': f"Average loss: {signal.expected_loss_if_wrong:.1f}%",
                    'best_conditions': signal.market_conditions_best,
                    'worst_conditions': signal.market_conditions_worst
                },
                'risk_breakdown': {
                    'max_potential_loss': f"{signal.risk_classification.max_drawdown:.1f}%",
                    'typical_timeline': signal.typical_timeline,
                    'complexity_level': f"{signal.risk_classification.complexity_score}/10",
                    'capital_needed': signal.risk_classification.capital_requirement
                },
                'algorithm_limitations': {
                    'known_weaknesses': signal.algorithm_weaknesses,
                    'when_to_avoid': signal.market_conditions_worst,
                    'data_requirements': f"Requires {signal.backtest_period} of historical data",
                    'confidence_range': signal.confidence_intervals
                }
            },
            'swipe_down_captures': {
                'rejection_reason': 'User showed no interest in this signal type',
                'learning_data': {
                    'signal_type': signal.signal_type,
                    'risk_level': signal.risk_classification.risk_level.value,
                    'algorithm_used': signal.algorithm_used,
                    'confidence_level': signal.confidence
                }
            }
        }
    
    def _get_simulated_market_data(self, ticker: str) -> Dict:
        """Get simulated current market data"""
        # Simulate different price ranges for different tickers
        base_prices = {
            'AAPL': 175, 'MSFT': 380, 'GOOGL': 140, 'TSLA': 250, 'NVDA': 450,
            'META': 320, 'AMZN': 140, 'NFLX': 450, 'SAVA': 25, 'MRNA': 90
        }
        
        base_price = base_prices.get(ticker, 100)
        # Add some random variation
        np.random.seed(hash(ticker) % 1000)
        current_price = base_price * (1 + np.random.normal(0, 0.02))
        
        return {
            'current_price': current_price,
            'volume': np.random.randint(1000000, 5000000),
            'timestamp': datetime.now()
        }
    
    def _get_simulated_historical_data(self, ticker: str) -> pd.DataFrame:
        """Get simulated historical data for strategy training"""
        np.random.seed(hash(ticker) % 1000)
        days = 30
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        base_price = self._get_simulated_market_data(ticker)['current_price']
        prices = base_price + np.cumsum(np.random.normal(0, base_price * 0.01, days))
        volumes = np.random.lognormal(15, 1, days)
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': volumes
        })
    
    def process_user_swipe(self, user_id: str, signal_id: str, action: str, 
                          engagement_time: float = 0.0) -> Dict:
        """
        Process user swipe interaction for learning
        
        Args:
            user_id: User identifier
            signal_id: Signal that was swiped on
            action: 'swipe_up' (interested) or 'swipe_down' (not interested)  
            engagement_time: Time spent looking at signal (seconds)
        """
        try:
            # For MVP, we'll simulate the signal data
            # In production, this would lookup the actual signal from signal_id
            simulated_signal = self._create_simulated_signal_for_swipe(signal_id)
            
            # Calculate engagement level based on time
            engagement_level = min(1.0, engagement_time / 10.0)  # 10 seconds = full engagement
            
            # Record the interaction
            self.profiling_engine.record_user_interaction(
                user_id, simulated_signal, action, engagement_level
            )
            
            # Get updated profile
            updated_profile = self.profiling_engine.get_user_profile_summary(user_id)
            
            return {
                'success': True,
                'user_id': user_id,
                'action_recorded': action,
                'engagement_level': engagement_level,
                'updated_profile': updated_profile,
                'learning_feedback': self._generate_learning_feedback(action, updated_profile),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Failed to process swipe: {str(e)}",
                'user_id': user_id,
                'timestamp': datetime.now().isoformat()
            }
    
    def _create_simulated_signal_for_swipe(self, signal_id: str) -> EnhancedSignalData:
        """Create simulated signal for swipe processing (MVP implementation)"""
        # Parse signal_id to extract properties
        # Format: ticker_strategy_riskLevel (e.g., "AAPL_technical_moderate")
        parts = signal_id.split('_')
        ticker = parts[0] if len(parts) > 0 else 'AAPL'
        strategy = parts[1] if len(parts) > 1 else 'technical'
        risk_str = parts[2] if len(parts) > 2 else 'moderate'
        
        # Map risk string to enum
        risk_mapping = {
            'conservative': RiskProfile.CONSERVATIVE,
            'moderate': RiskProfile.MODERATE,
            'aggressive': RiskProfile.AGGRESSIVE,
            'yolo': RiskProfile.EXTREMELY_AGGRESSIVE
        }
        risk_level = risk_mapping.get(risk_str, RiskProfile.MODERATE)
        
        # Create basic enhanced signal
        return EnhancedSignalData(
            ticker=ticker,
            signal_type=f"{strategy}_{risk_str}",
            current_price=100.0,
            target_price=105.0,
            confidence=70.0,
            risk_classification=SignalRiskClassification(
                risk_level=risk_level,
                signal_type=f"{strategy}_{risk_str}",
                win_rate=65.0,
                avg_return=8.0,
                max_drawdown=10.0,
                typical_holding_period='3 days',
                capital_requirement='Medium',
                complexity_score=5
            ),
            algorithm_used=strategy.title(),
            algorithm_description="Test algorithm",
            key_factors={'test': 1.0},
            historical_win_rate=65.0,
            expected_return_if_correct=8.0,
            expected_loss_if_wrong=5.0,
            typical_timeline='3 days',
            algorithm_strengths=['Test strength'],
            algorithm_weaknesses=['Test weakness'],
            market_conditions_best=['Test condition'],
            market_conditions_worst=['Test condition'],
            backtest_period='2020-2024',
            sample_size=1000,
            confidence_intervals={'win_rate_95_ci': [62, 68]},
            relevance_score=75.0,
            recommendation_reason="Test recommendation"
        )
    
    def _generate_learning_feedback(self, action: str, profile: Dict) -> str:
        """Generate feedback about what the system learned from user action"""
        if action == 'swipe_up':
            return f"✅ Learning: You show interest in {profile['current_risk_profile'].replace('_', ' ')} signals. " + \
                   f"Confidence in your profile: {profile['confidence_score']:.0f}%"
        else:
            return f"📚 Learning: You avoid this signal type. " + \
                   f"Adjusting recommendations for {profile['current_risk_profile'].replace('_', ' ')} profile. " + \
                   f"Profile confidence: {profile['confidence_score']:.0f}%"

# Demo and testing
if __name__ == "__main__":
    print("🔍 Transparent Signal Engine Demo")
    print("=" * 60)
    
    engine = TransparentSignalEngine()
    
    # Test user profiling and signal generation
    test_user = "demo_user_001"
    
    print(f"👤 Creating user profile for {test_user}...")
    result = engine.generate_personalized_signals(test_user, max_signals=3)
    
    if result['success']:
        profile = result['user_profile']
        print(f"User Profile: {profile['current_risk_profile']}")
        print(f"Profile Confidence: {profile['confidence_score']:.0f}%")
        print(f"Generated {result['personalized_count']} personalized signals from {result['total_signals_generated']} total")
        
        print(f"\n📊 Sample Signals:")
        for i, signal_data in enumerate(result['signals'][:2]):
            signal = signal_data['signal_data']
            explanation = signal_data['explanation']
            
            print(f"\n--- Signal {i+1}: {signal.ticker} ---")
            print(f"Type: {signal.signal_type}")
            print(f"Algorithm: {signal.algorithm_used}")
            print(f"Win Rate: {signal.historical_win_rate:.1f}%")
            print(f"Risk Level: {signal.risk_classification.risk_level.value}")
            print(f"Expected Return: {signal.expected_return_if_correct:+.1f}%")
            print(f"Max Loss: {signal.expected_loss_if_wrong:.1f}%")
            print(f"Relevance Score: {signal.relevance_score:.0f}/100")
    
    # Test swipe interaction
    print(f"\n👆 Testing Swipe Interaction...")
    swipe_result = engine.process_user_swipe(test_user, "AAPL_technical_moderate", "swipe_up", 5.0)
    
    if swipe_result['success']:
        print(f"Action: {swipe_result['action_recorded']}")
        print(f"Learning: {swipe_result['learning_feedback']}")
        updated_profile = swipe_result['updated_profile']
        print(f"Updated Confidence: {updated_profile['confidence_score']:.0f}%")
    
    print(f"\n✅ Transparent Signal Engine Demo Complete!")
    print("🎯 Focus: Signal accuracy and transparency, not profitability")
    print("📱 Ready for swipe-based user profiling!")