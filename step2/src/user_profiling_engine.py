"""
User Risk Profiling & Signal Recommendation Engine
Profiles users from Conservative to YOLO based on signal interactions
Focuses on signal accuracy and transparency rather than profitability
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json

class RiskProfile(Enum):
    """User risk appetite categories"""
    EXTREMELY_CONSERVATIVE = "extremely_conservative"  # Bond-type, no principal loss
    CONSERVATIVE = "conservative"                      # Blue chip, dividend stocks
    MODERATE = "moderate"                             # SP500 index level risk
    AGGRESSIVE = "aggressive"                         # Growth stocks, options
    EXTREMELY_AGGRESSIVE = "extremely_aggressive"     # 0DTE, YOLO plays, 2x or lose all

@dataclass
class SignalRiskClassification:
    """Risk classification for trading signals"""
    risk_level: RiskProfile
    signal_type: str
    win_rate: float                    # Historical win rate %
    avg_return: float                  # Average return when correct
    max_drawdown: float               # Maximum loss potential
    typical_holding_period: str       # "1 day", "1 week", etc.
    capital_requirement: str          # "Low", "Medium", "High"
    complexity_score: int             # 1-10 (simple to complex)
    
@dataclass
class UserProfile:
    """Complete user risk profile and preferences"""
    user_id: str
    current_risk_profile: RiskProfile
    confidence_score: float           # How confident we are in the profile (0-100%)
    interaction_history: List[Dict]   # Swipe history and signal interactions
    preferred_signal_types: List[str] # Signal types user has shown interest in
    avoided_signal_types: List[str]   # Signal types user consistently rejects
    total_interactions: int
    profile_stability: float          # How stable the profile is over time
    last_updated: datetime
    
@dataclass 
class EnhancedSignalData:
    """Signal with complete transparency and risk information"""
    # Core signal data
    ticker: str
    signal_type: str
    current_price: float
    target_price: float
    confidence: float
    
    # Risk classification
    risk_classification: SignalRiskClassification
    
    # Algorithm transparency
    algorithm_used: str
    algorithm_description: str
    key_factors: Dict[str, float]
    
    # Performance metrics
    historical_win_rate: float
    expected_return_if_correct: float
    expected_loss_if_wrong: float
    typical_timeline: str
    
    # Algorithm limitations
    algorithm_strengths: List[str]
    algorithm_weaknesses: List[str]
    market_conditions_best: List[str]
    market_conditions_worst: List[str]
    
    # Backtest data
    backtest_period: str
    sample_size: int
    confidence_intervals: Dict[str, float]
    
    # User-specific relevance
    relevance_score: float            # How relevant for this user's profile
    recommendation_reason: str        # Why this signal matches user profile

class UserProfilingEngine:
    """
    Main engine for user risk profiling and signal recommendation
    """
    
    def __init__(self):
        # Risk profile definitions
        self.risk_profiles = self._initialize_risk_profiles()
        
        # Signal risk classifications
        self.signal_classifications = self._initialize_signal_classifications()
        
        # Algorithm performance data (would be loaded from backtesting)
        self.algorithm_performance = self._initialize_algorithm_performance()
        
        # User profiles storage (in production, this would be a database)
        self.user_profiles = {}
        
    def _initialize_risk_profiles(self) -> Dict[RiskProfile, Dict]:
        """Define characteristics of each risk profile"""
        return {
            RiskProfile.EXTREMELY_CONSERVATIVE: {
                'description': 'Cannot lose principal at all - bond-type investments only',
                'max_acceptable_loss': 0.02,        # 2% max loss
                'preferred_win_rate': 0.95,         # Needs 95%+ win rate
                'preferred_holding_period': ['1 week', '1 month', '3 months'],
                'acceptable_complexity': [1, 2, 3], # Simple strategies only
                'preferred_signal_types': ['dividend_play', 'bond_substitute', 'defensive_stock']
            },
            RiskProfile.CONSERVATIVE: {
                'description': 'Blue chip, dividend stocks - steady growth',
                'max_acceptable_loss': 0.05,        # 5% max loss
                'preferred_win_rate': 0.80,         # Needs 80%+ win rate
                'preferred_holding_period': ['1 week', '1 month'],
                'acceptable_complexity': [1, 2, 3, 4],
                'preferred_signal_types': ['earnings_play', 'dividend_capture', 'large_cap_momentum']
            },
            RiskProfile.MODERATE: {
                'description': 'SP500 index level risk - balanced approach',
                'max_acceptable_loss': 0.10,        # 10% max loss
                'preferred_win_rate': 0.65,         # Needs 65%+ win rate
                'preferred_holding_period': ['1 day', '1 week', '2 weeks'],
                'acceptable_complexity': [1, 2, 3, 4, 5, 6],
                'preferred_signal_types': ['technical_breakout', 'earnings_play', 'sector_rotation']
            },
            RiskProfile.AGGRESSIVE: {
                'description': 'Growth stocks, options - chasing higher returns',
                'max_acceptable_loss': 0.25,        # 25% max loss
                'preferred_win_rate': 0.55,         # Needs 55%+ win rate
                'preferred_holding_period': ['1 day', '3 days', '1 week'],
                'acceptable_complexity': [3, 4, 5, 6, 7, 8],
                'preferred_signal_types': ['momentum_play', 'options_flow', 'growth_breakout']
            },
            RiskProfile.EXTREMELY_AGGRESSIVE: {
                'description': '0DTE, YOLO plays - 2x or lose all mentality',
                'max_acceptable_loss': 1.0,         # Can lose everything
                'preferred_win_rate': 0.35,         # Accept 35%+ win rate for big wins
                'preferred_holding_period': ['1 hour', '1 day', '3 days'],
                'acceptable_complexity': [5, 6, 7, 8, 9, 10],
                'preferred_signal_types': ['yolo_play', '0dte_options', 'meme_momentum', 'biotech_binary']
            }
        }
    
    def _initialize_signal_classifications(self) -> Dict[str, SignalRiskClassification]:
        """Define risk classification for each signal type"""
        return {
            'dividend_capture': SignalRiskClassification(
                risk_level=RiskProfile.EXTREMELY_CONSERVATIVE,
                signal_type='dividend_capture',
                win_rate=92.0,
                avg_return=2.5,
                max_drawdown=3.0,
                typical_holding_period='3 days',
                capital_requirement='Low',
                complexity_score=2
            ),
            'large_cap_earnings': SignalRiskClassification(
                risk_level=RiskProfile.CONSERVATIVE,
                signal_type='large_cap_earnings',
                win_rate=78.0,
                avg_return=4.2,
                max_drawdown=7.5,
                typical_holding_period='1 week',
                capital_requirement='Medium',
                complexity_score=4
            ),
            'technical_breakout': SignalRiskClassification(
                risk_level=RiskProfile.MODERATE,
                signal_type='technical_breakout',
                win_rate=65.0,
                avg_return=8.5,
                max_drawdown=12.0,
                typical_holding_period='3 days',
                capital_requirement='Medium',
                complexity_score=5
            ),
            'momentum_options': SignalRiskClassification(
                risk_level=RiskProfile.AGGRESSIVE,
                signal_type='momentum_options',
                win_rate=58.0,
                avg_return=25.0,
                max_drawdown=35.0,
                typical_holding_period='1 day',
                capital_requirement='High',
                complexity_score=7
            ),
            'yolo_biotech': SignalRiskClassification(
                risk_level=RiskProfile.EXTREMELY_AGGRESSIVE,
                signal_type='yolo_biotech',
                win_rate=38.0,
                avg_return=180.0,
                max_drawdown=85.0,
                typical_holding_period='1 day',
                capital_requirement='High',
                complexity_score=9
            ),
            '0dte_options': SignalRiskClassification(
                risk_level=RiskProfile.EXTREMELY_AGGRESSIVE,
                signal_type='0dte_options',
                win_rate=32.0,
                avg_return=300.0,
                max_drawdown=95.0,
                typical_holding_period='4 hours',
                capital_requirement='Very High',
                complexity_score=10
            )
        }
    
    def _initialize_algorithm_performance(self) -> Dict[str, Dict]:
        """Initialize algorithm performance data (from backtesting)"""
        return {
            'Technical-LSTM Hybrid': {
                'overall_win_rate': 68.5,
                'strengths': [
                    'Excellent at detecting technical pattern reversals',
                    'Strong performance in trending markets',
                    'Good at identifying volume-based signals'
                ],
                'weaknesses': [
                    'Struggles during low volatility periods',
                    'Can generate false signals in choppy markets',
                    'Requires sufficient historical data (minimum 30 days)'
                ],
                'best_market_conditions': [
                    'High volume trending markets',
                    'Clear technical patterns present',
                    'Moderate to high volatility (VIX 15-35)'
                ],
                'worst_market_conditions': [
                    'Extremely low volatility (VIX < 12)',
                    'News-driven markets with gaps',
                    'Holiday/low volume periods'
                ],
                'backtest_period': '2020-2024',
                'sample_size': 15420,
                'confidence_intervals': {'win_rate_95_ci': [65.2, 71.8]}
            },
            'ARIMA-Momentum Ensemble': {
                'overall_win_rate': 71.2,
                'strengths': [
                    'Excellent trend following capabilities',
                    'Strong statistical foundation',
                    'Performs well in sustained trends'
                ],
                'weaknesses': [
                    'Slow to react to sudden reversals',
                    'Poor performance in sideways markets',
                    'Assumes market efficiency (often violated)'
                ],
                'best_market_conditions': [
                    'Clear trending markets (up or down)',
                    'Low to moderate volatility',
                    'Fundamental-driven moves'
                ],
                'worst_market_conditions': [
                    'Range-bound/sideways markets',
                    'High volatility with frequent reversals',
                    'Event-driven binary outcomes'
                ],
                'backtest_period': '2020-2024',
                'sample_size': 12850,
                'confidence_intervals': {'win_rate_95_ci': [68.7, 73.7]}
            },
            'Event-Driven Volatility': {
                'overall_win_rate': 59.8,
                'strengths': [
                    'Excellent at predicting volatility expansion',
                    'Strong performance around known events',
                    'Good at sizing for risk/reward scenarios'
                ],
                'weaknesses': [
                    'Vulnerable to unexpected news',
                    'Event timing can be imprecise',
                    'High variability in outcomes'
                ],
                'best_market_conditions': [
                    'Known upcoming events (earnings, FDA)',
                    'Stable macro environment',
                    'Clear event calendars'
                ],
                'worst_market_conditions': [
                    'Surprise news/events',
                    'Multiple overlapping events',
                    'Macro uncertainty periods'
                ],
                'backtest_period': '2020-2024',
                'sample_size': 8450,
                'confidence_intervals': {'win_rate_95_ci': [56.3, 63.3]}
            },
            'Ensemble Meta-Learner': {
                'overall_win_rate': 72.1,
                'strengths': [
                    'Most consistent performance across conditions',
                    'Reduces single-algorithm bias',
                    'Adapts to changing market regimes'
                ],
                'weaknesses': [
                    'Can mask important individual signals',
                    'Slower to adapt to new patterns',
                    'Complex to interpret for users'
                ],
                'best_market_conditions': [
                    'Mixed market conditions',
                    'Normal volatility environments',
                    'Diverse signal types available'
                ],
                'worst_market_conditions': [
                    'Extreme market stress',
                    'When all algorithms fail simultaneously',
                    'Very short time horizons'
                ],
                'backtest_period': '2020-2024',
                'sample_size': 18200,
                'confidence_intervals': {'win_rate_95_ci': [69.8, 74.4]}
            }
        }
    
    def create_user_profile(self, user_id: str, initial_risk_assessment: Optional[RiskProfile] = None) -> UserProfile:
        """Create a new user profile"""
        profile = UserProfile(
            user_id=user_id,
            current_risk_profile=initial_risk_assessment or RiskProfile.MODERATE,
            confidence_score=20.0 if initial_risk_assessment else 10.0,  # Low initial confidence
            interaction_history=[],
            preferred_signal_types=[],
            avoided_signal_types=[],
            total_interactions=0,
            profile_stability=0.0,
            last_updated=datetime.now()
        )
        
        self.user_profiles[user_id] = profile
        return profile
    
    def record_user_interaction(self, user_id: str, signal_data: EnhancedSignalData, 
                              action: str, engagement_level: float = 0.5) -> None:
        """
        Record user interaction with a signal
        
        Args:
            user_id: User identifier
            signal_data: The signal user interacted with
            action: 'swipe_up' (interested) or 'swipe_down' (not interested)
            engagement_level: 0-1, how much time/attention user gave
        """
        if user_id not in self.user_profiles:
            self.create_user_profile(user_id)
        
        profile = self.user_profiles[user_id]
        
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'signal_type': signal_data.signal_type,
            'risk_level': signal_data.risk_classification.risk_level.value,
            'action': action,
            'engagement_level': engagement_level,
            'signal_confidence': signal_data.confidence,
            'algorithm_used': signal_data.algorithm_used
        }
        
        profile.interaction_history.append(interaction)
        profile.total_interactions += 1
        profile.last_updated = datetime.now()
        
        # Update preferences based on action
        if action == 'swipe_up':
            if signal_data.signal_type not in profile.preferred_signal_types:
                profile.preferred_signal_types.append(signal_data.signal_type)
            # Remove from avoided if previously there
            if signal_data.signal_type in profile.avoided_signal_types:
                profile.avoided_signal_types.remove(signal_data.signal_type)
        elif action == 'swipe_down':
            if signal_data.signal_type not in profile.avoided_signal_types:
                profile.avoided_signal_types.append(signal_data.signal_type)
            # Remove from preferred if previously there
            if signal_data.signal_type in profile.preferred_signal_types:
                profile.preferred_signal_types.remove(signal_data.signal_type)
        
        # Re-evaluate risk profile
        self._update_risk_profile(user_id)
    
    def _update_risk_profile(self, user_id: str) -> None:
        """Update user's risk profile based on interaction history"""
        profile = self.user_profiles[user_id]
        
        if profile.total_interactions < 5:
            # Not enough data for reliable profiling
            profile.confidence_score = min(30.0, profile.total_interactions * 6)
            return
        
        # Analyze interaction patterns
        recent_interactions = profile.interaction_history[-20:]  # Last 20 interactions
        
        # Calculate risk tolerance based on signals user engaged with
        risk_scores = []
        for interaction in recent_interactions:
            if interaction['action'] == 'swipe_up':
                # User showed interest - extract risk level
                risk_level = interaction['risk_level']
                risk_score = self._risk_level_to_score(risk_level)
                risk_scores.append(risk_score)
        
        if not risk_scores:
            return  # No positive interactions yet
        
        # Calculate average risk preference
        avg_risk_score = np.mean(risk_scores)
        new_risk_profile = self._score_to_risk_level(avg_risk_score)
        
        # Update profile if changed
        if new_risk_profile != profile.current_risk_profile:
            profile.current_risk_profile = new_risk_profile
            profile.profile_stability = self._calculate_profile_stability(profile)
        
        # Update confidence based on number of interactions and consistency
        profile.confidence_score = min(95.0, 
            30 + (profile.total_interactions * 2) + (profile.profile_stability * 30)
        )
    
    def _risk_level_to_score(self, risk_level: str) -> float:
        """Convert risk level string to numeric score"""
        mapping = {
            'extremely_conservative': 1.0,
            'conservative': 2.0,
            'moderate': 3.0,
            'aggressive': 4.0,
            'extremely_aggressive': 5.0
        }
        return mapping.get(risk_level, 3.0)
    
    def _score_to_risk_level(self, score: float) -> RiskProfile:
        """Convert numeric score to risk profile"""
        if score <= 1.5:
            return RiskProfile.EXTREMELY_CONSERVATIVE
        elif score <= 2.5:
            return RiskProfile.CONSERVATIVE
        elif score <= 3.5:
            return RiskProfile.MODERATE
        elif score <= 4.5:
            return RiskProfile.AGGRESSIVE
        else:
            return RiskProfile.EXTREMELY_AGGRESSIVE
    
    def _calculate_profile_stability(self, profile: UserProfile) -> float:
        """Calculate how stable the user's profile has been over time"""
        if len(profile.interaction_history) < 10:
            return 0.0
        
        # Look at last 20 interactions and see consistency
        recent_interactions = profile.interaction_history[-20:]
        risk_scores = []
        
        for interaction in recent_interactions:
            if interaction['action'] == 'swipe_up':
                risk_score = self._risk_level_to_score(interaction['risk_level'])
                risk_scores.append(risk_score)
        
        if len(risk_scores) < 3:
            return 0.0
        
        # Calculate coefficient of variation (stability measure)
        std_dev = np.std(risk_scores)
        mean_score = np.mean(risk_scores)
        
        if mean_score == 0:
            return 0.0
        
        # Convert to stability score (lower variation = higher stability)
        cv = std_dev / mean_score
        stability = max(0.0, 1.0 - cv)
        
        return stability
    
    def get_personalized_signals(self, user_id: str, all_signals: List[EnhancedSignalData], 
                                max_signals: int = 10) -> List[EnhancedSignalData]:
        """
        Get personalized signal recommendations for a user
        
        Args:
            user_id: User identifier
            all_signals: All available signals
            max_signals: Maximum number of signals to return
            
        Returns:
            Personalized and ranked signals for the user
        """
        if user_id not in self.user_profiles:
            # New user - return moderate risk signals
            profile = self.create_user_profile(user_id)
        else:
            profile = self.user_profiles[user_id]
        
        # Calculate relevance score for each signal
        scored_signals = []
        
        for signal in all_signals:
            relevance_score = self._calculate_signal_relevance(profile, signal)
            
            # Add user-specific information to signal
            enhanced_signal = self._enhance_signal_for_user(signal, profile, relevance_score)
            scored_signals.append((relevance_score, enhanced_signal))
        
        # Sort by relevance score and return top signals
        scored_signals.sort(key=lambda x: x[0], reverse=True)
        
        return [signal for _, signal in scored_signals[:max_signals]]
    
    def _calculate_signal_relevance(self, profile: UserProfile, signal: EnhancedSignalData) -> float:
        """Calculate how relevant a signal is for a specific user"""
        relevance_score = 0.0
        
        # Risk profile match (most important factor)
        user_risk_preferences = self.risk_profiles[profile.current_risk_profile]
        
        # Check if signal risk level matches user preference
        signal_risk_score = self._risk_level_to_score(signal.risk_classification.risk_level.value)
        user_risk_score = self._risk_level_to_score(profile.current_risk_profile.value)
        
        risk_distance = abs(signal_risk_score - user_risk_score)
        risk_match_score = max(0, 1.0 - (risk_distance / 4.0))  # 0-1 scale
        relevance_score += risk_match_score * 40  # 40% weight
        
        # Win rate preference
        if signal.historical_win_rate >= user_risk_preferences['preferred_win_rate']:
            relevance_score += 20  # 20% weight
        else:
            # Penalize based on how far below preferred win rate
            win_rate_deficit = user_risk_preferences['preferred_win_rate'] - signal.historical_win_rate
            relevance_score += max(0, 20 - (win_rate_deficit * 100))
        
        # Max loss tolerance
        if signal.risk_classification.max_drawdown <= user_risk_preferences['max_acceptable_loss'] * 100:
            relevance_score += 15  # 15% weight
        else:
            # Penalize if potential loss exceeds tolerance
            relevance_score += max(0, 15 - 30)  # Significant penalty
        
        # Signal type preference based on history
        if signal.signal_type in profile.preferred_signal_types:
            relevance_score += 15  # 15% weight
        elif signal.signal_type in profile.avoided_signal_types:
            relevance_score -= 20  # Penalty for avoided types
        
        # Complexity preference
        if signal.risk_classification.complexity_score in user_risk_preferences['acceptable_complexity']:
            relevance_score += 10  # 10% weight
        
        return max(0, min(100, relevance_score))
    
    def _enhance_signal_for_user(self, signal: EnhancedSignalData, profile: UserProfile, 
                               relevance_score: float) -> EnhancedSignalData:
        """Add user-specific information to signal"""
        # Create recommendation reason
        risk_match = "perfectly matches" if relevance_score > 80 else "aligns with" if relevance_score > 60 else "may interest"
        recommendation_reason = f"This signal {risk_match} your {profile.current_risk_profile.value.replace('_', ' ')} risk profile"
        
        # Create a copy with user-specific data
        enhanced_signal = EnhancedSignalData(
            # Copy all existing data
            ticker=signal.ticker,
            signal_type=signal.signal_type,
            current_price=signal.current_price,
            target_price=signal.target_price,
            confidence=signal.confidence,
            risk_classification=signal.risk_classification,
            algorithm_used=signal.algorithm_used,
            algorithm_description=signal.algorithm_description,
            key_factors=signal.key_factors,
            historical_win_rate=signal.historical_win_rate,
            expected_return_if_correct=signal.expected_return_if_correct,
            expected_loss_if_wrong=signal.expected_loss_if_wrong,
            typical_timeline=signal.typical_timeline,
            algorithm_strengths=signal.algorithm_strengths,
            algorithm_weaknesses=signal.algorithm_weaknesses,
            market_conditions_best=signal.market_conditions_best,
            market_conditions_worst=signal.market_conditions_worst,
            backtest_period=signal.backtest_period,
            sample_size=signal.sample_size,
            confidence_intervals=signal.confidence_intervals,
            
            # Add user-specific data
            relevance_score=relevance_score,
            recommendation_reason=recommendation_reason
        )
        
        return enhanced_signal
    
    def get_user_profile_summary(self, user_id: str) -> Dict:
        """Get comprehensive user profile summary"""
        if user_id not in self.user_profiles:
            return {'error': 'User profile not found'}
        
        profile = self.user_profiles[user_id]
        
        return {
            'user_id': user_id,
            'current_risk_profile': profile.current_risk_profile.value,
            'risk_description': self.risk_profiles[profile.current_risk_profile]['description'],
            'confidence_score': profile.confidence_score,
            'total_interactions': profile.total_interactions,
            'profile_stability': profile.profile_stability,
            'preferred_signal_types': profile.preferred_signal_types,
            'avoided_signal_types': profile.avoided_signal_types,
            'last_updated': profile.last_updated.isoformat(),
            'recommendations': {
                'max_acceptable_loss': self.risk_profiles[profile.current_risk_profile]['max_acceptable_loss'],
                'preferred_win_rate': self.risk_profiles[profile.current_risk_profile]['preferred_win_rate'],
                'preferred_holding_periods': self.risk_profiles[profile.current_risk_profile]['preferred_holding_period']
            }
        }
    
    def generate_signal_explanation(self, signal: EnhancedSignalData, detail_level: str = "full") -> Dict:
        """
        Generate comprehensive signal explanation for transparency
        
        Args:
            signal: The signal to explain
            detail_level: "summary", "detailed", or "full"
        """
        explanation = {
            'signal_overview': {
                'ticker': signal.ticker,
                'signal_type': signal.signal_type,
                'current_price': signal.current_price,
                'target_price': signal.target_price,
                'expected_move': f"{((signal.target_price - signal.current_price) / signal.current_price) * 100:+.1f}%",
                'timeline': signal.typical_timeline
            }
        }
        
        if detail_level in ["detailed", "full"]:
            explanation.update({
                'algorithm_details': {
                    'algorithm_used': signal.algorithm_used,
                    'description': signal.algorithm_description,
                    'key_factors': signal.key_factors,
                    'confidence': signal.confidence
                },
                'performance_data': {
                    'historical_win_rate': f"{signal.historical_win_rate:.1f}%",
                    'expected_return_if_correct': f"{signal.expected_return_if_correct:+.1f}%",
                    'expected_loss_if_wrong': f"{signal.expected_loss_if_wrong:.1f}%",
                    'backtest_period': signal.backtest_period,
                    'sample_size': signal.sample_size
                },
                'risk_assessment': {
                    'risk_level': signal.risk_classification.risk_level.value,
                    'max_drawdown': f"{signal.risk_classification.max_drawdown:.1f}%",
                    'capital_requirement': signal.risk_classification.capital_requirement,
                    'complexity_score': f"{signal.risk_classification.complexity_score}/10"
                }
            })
        
        if detail_level == "full":
            explanation.update({
                'algorithm_strengths': signal.algorithm_strengths,
                'algorithm_weaknesses': signal.algorithm_weaknesses,
                'best_market_conditions': signal.market_conditions_best,
                'worst_market_conditions': signal.market_conditions_worst,
                'statistical_confidence': signal.confidence_intervals,
                'user_relevance': {
                    'relevance_score': f"{signal.relevance_score:.0f}/100",
                    'recommendation_reason': signal.recommendation_reason
                }
            })
        
        return explanation

# Demo and testing
if __name__ == "__main__":
    print("👤 User Profiling Engine Demo")
    print("=" * 50)
    
    engine = UserProfilingEngine()
    
    # Create test user
    user_id = "test_user_001"
    profile = engine.create_user_profile(user_id, RiskProfile.MODERATE)
    
    print(f"Created user profile: {profile.current_risk_profile.value}")
    print(f"Initial confidence: {profile.confidence_score:.1f}%")
    
    # Show risk profile characteristics
    risk_info = engine.risk_profiles[profile.current_risk_profile]
    print(f"\nRisk Profile: {risk_info['description']}")
    print(f"Max acceptable loss: {risk_info['max_acceptable_loss']*100:.1f}%")
    print(f"Preferred win rate: {risk_info['preferred_win_rate']*100:.1f}%")
    
    print(f"\n✅ User Profiling Engine Ready!")
    print("System will learn user preferences through swipe interactions")