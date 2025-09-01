"""
User-Signal Matching Service
Core logic for matching trading signals to users based on risk tolerance
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_

from app.models.database import User, Signal, UserSignalMatch
from app.schemas.user import UserRiskProfile
from app.core.exceptions import ValidationError

logger = logging.getLogger(__name__)


class SignalMatcher:
    """Core signal-to-user matching algorithm"""
    
    # Matching weights for different factors
    MATCHING_WEIGHTS = {
        "risk_compatibility": 0.50,  # Primary factor - risk alignment
        "asset_preference": 0.25,    # User's asset class preferences  
        "confidence_boost": 0.15,    # Higher confidence signals get boost
        "profit_potential": 0.10,    # Higher profit potential gets boost
    }
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def find_compatible_users(
        self,
        signal: Signal,
        max_matches: int = 100,
        min_match_score: float = 60.0
    ) -> List[Dict[str, Any]]:
        """
        Find users compatible with a given signal
        
        Args:
            signal: The trading signal to match
            max_matches: Maximum number of matches to return
            min_match_score: Minimum match score threshold (0-100)
            
        Returns:
            List of user matches with scores and explanations
        """
        
        try:
            # Get active users within reasonable risk range
            risk_tolerance_range = 30  # Allow wider range for more matches in MVP
            
            risk_min = max(0, signal.risk_score - risk_tolerance_range)
            risk_max = min(100, signal.risk_score + risk_tolerance_range)
            
            query = select(User).where(
                and_(
                    User.is_active == True,
                    User.risk_tolerance >= risk_min,
                    User.risk_tolerance <= risk_max,
                    User.max_position_size_cents >= signal.min_position_size_cents
                )
            )
            
            result = await self.db.execute(query)
            candidate_users = result.scalars().all()
            
            logger.info(f"Found {len(candidate_users)} candidate users for signal {signal.id}")
            
            # Calculate match scores for each user
            user_matches = []
            
            for user in candidate_users:
                match_score, explanation = self._calculate_match_score(signal, user)
                
                if match_score >= min_match_score:
                    user_matches.append({
                        "user_id": user.id,
                        "user_email": user.email,
                        "user_risk_tolerance": user.risk_tolerance,
                        "match_score": round(match_score, 2),
                        "explanation": explanation,
                        "signal_id": signal.id,
                        "signal_symbol": signal.symbol,
                        "signal_risk_score": signal.risk_score,
                    })
            
            # Sort by match score (highest first) and limit results
            user_matches.sort(key=lambda x: x["match_score"], reverse=True)
            
            logger.info(f"Generated {len(user_matches)} matches for signal {signal.id}")
            
            return user_matches[:max_matches]
            
        except Exception as e:
            logger.error(f"Failed to find compatible users for signal {signal.id}: {e}")
            raise
    
    async def find_signals_for_user(
        self,
        user: User,
        max_signals: int = 50,
        min_match_score: float = 70.0
    ) -> List[Dict[str, Any]]:
        """
        Find signals compatible with a specific user
        
        Args:
            user: The user to find signals for
            max_signals: Maximum number of signals to return
            min_match_score: Minimum match score threshold
            
        Returns:
            List of signal matches with scores
        """
        
        try:
            # Get active signals within user's risk tolerance range
            risk_range = user.max_risk_deviation or 25
            
            risk_min = max(0, user.risk_tolerance - risk_range)
            risk_max = min(100, user.risk_tolerance + risk_range)
            
            query = select(Signal).where(
                and_(
                    Signal.is_active == True,
                    Signal.expires_at > datetime.utcnow(),
                    Signal.risk_score >= risk_min,
                    Signal.risk_score <= risk_max,
                    Signal.max_position_size_cents <= user.max_position_size_cents
                )
            )
            
            result = await self.db.execute(query)
            candidate_signals = result.scalars().all()
            
            logger.info(f"Found {len(candidate_signals)} candidate signals for user {user.id}")
            
            # Calculate match scores
            signal_matches = []
            
            for signal in candidate_signals:
                match_score, explanation = self._calculate_match_score(signal, user)
                
                if match_score >= min_match_score:
                    signal_matches.append({
                        "signal_id": signal.id,
                        "signal_name": signal.signal_name,
                        "symbol": signal.symbol,
                        "asset_class": signal.asset_class,
                        "risk_score": signal.risk_score,
                        "confidence_score": signal.confidence_score,
                        "profit_potential_score": signal.profit_potential_score,
                        "match_score": round(match_score, 2),
                        "explanation": explanation,
                        "created_at": signal.created_at,
                        "expires_at": signal.expires_at,
                    })
            
            # Sort by match score
            signal_matches.sort(key=lambda x: x["match_score"], reverse=True)
            
            logger.info(f"Generated {len(signal_matches)} signal matches for user {user.id}")
            
            return signal_matches[:max_signals]
            
        except Exception as e:
            logger.error(f"Failed to find signals for user {user.id}: {e}")
            raise
    
    def _calculate_match_score(self, signal: Signal, user: User) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate match score between signal and user
        
        Returns:
            Tuple of (match_score, explanation_dict)
        """
        
        # 1. Risk compatibility (most important factor)
        risk_compatibility = self._calculate_risk_compatibility(
            signal.risk_score, user.risk_tolerance, user.max_risk_deviation or 25
        )
        
        # 2. Asset class preference
        asset_preference = self._calculate_asset_preference(
            signal.asset_class, user.asset_preferences or {}
        )
        
        # 3. Confidence boost (higher confidence = better match)
        confidence_boost = min(100, signal.confidence_score * 1.2)  # Scale up slightly
        
        # 4. Profit potential factor
        profit_potential = signal.profit_potential_score
        
        # Calculate weighted score
        match_score = (
            risk_compatibility * self.MATCHING_WEIGHTS["risk_compatibility"] +
            asset_preference * self.MATCHING_WEIGHTS["asset_preference"] +
            confidence_boost * self.MATCHING_WEIGHTS["confidence_boost"] +
            profit_potential * self.MATCHING_WEIGHTS["profit_potential"]
        )
        
        # Position size compatibility check
        position_compatible = (
            signal.min_position_size_cents <= user.max_position_size_cents
        )
        
        if not position_compatible:
            match_score *= 0.5  # Significant penalty for position size mismatch
        
        # Create explanation
        explanation = {
            "risk_compatibility": round(risk_compatibility, 1),
            "asset_preference": round(asset_preference, 1),
            "confidence_boost": round(confidence_boost, 1),
            "profit_potential": round(profit_potential, 1),
            "position_compatible": position_compatible,
            "risk_difference": abs(signal.risk_score - user.risk_tolerance),
            "primary_factors": []
        }
        
        # Add primary matching reasons
        if risk_compatibility >= 80:
            explanation["primary_factors"].append("excellent_risk_match")
        elif risk_compatibility >= 60:
            explanation["primary_factors"].append("good_risk_match")
        
        if asset_preference >= 80:
            explanation["primary_factors"].append("preferred_asset_class")
        
        if confidence_boost >= 85:
            explanation["primary_factors"].append("high_confidence_signal")
        
        if profit_potential >= 80:
            explanation["primary_factors"].append("high_profit_potential")
        
        return max(0, min(100, match_score)), explanation
    
    def _calculate_risk_compatibility(
        self, 
        signal_risk: float, 
        user_tolerance: int, 
        user_deviation: int
    ) -> float:
        """Calculate risk compatibility score (0-100)"""
        
        risk_difference = abs(signal_risk - user_tolerance)
        
        # Perfect match
        if risk_difference == 0:
            return 100.0
        
        # Within tolerance
        if risk_difference <= user_deviation:
            # Linear decay within tolerance range
            compatibility = 100.0 * (1 - (risk_difference / user_deviation) * 0.3)
            return max(70.0, compatibility)
        
        # Outside tolerance but still reasonable
        if risk_difference <= user_deviation * 2:
            # Steep decay beyond tolerance
            excess_diff = risk_difference - user_deviation
            max_excess = user_deviation
            compatibility = 70.0 * (1 - (excess_diff / max_excess))
            return max(20.0, compatibility)
        
        # Very poor match
        return max(0.0, 20.0 * (1 - (risk_difference - user_deviation * 2) / 50.0))
    
    def _calculate_asset_preference(
        self, 
        signal_asset_class: str, 
        user_preferences: Dict[str, float]
    ) -> float:
        """Calculate asset class preference score (0-100)"""
        
        if not user_preferences:
            return 50.0  # Neutral score if no preferences set
        
        # Direct preference for this asset class
        if signal_asset_class in user_preferences:
            preference = user_preferences[signal_asset_class]
            return preference * 100  # Convert 0-1 scale to 0-100
        
        # Default score for unspecified asset classes
        return 30.0
    
    async def create_user_signal_match(
        self,
        user_id: UUID,
        signal_id: UUID,
        match_score: float,
        explanation: Dict[str, Any]
    ) -> UserSignalMatch:
        """Create a user-signal match record in the database"""
        
        try:
            # Check if match already exists
            existing_match = await self.db.execute(
                select(UserSignalMatch).where(
                    and_(
                        UserSignalMatch.user_id == user_id,
                        UserSignalMatch.signal_id == signal_id
                    )
                )
            )
            
            if existing_match.scalar_one_or_none():
                logger.warning(f"Match already exists for user {user_id} and signal {signal_id}")
                return existing_match.scalar_one()
            
            # Create new match
            match = UserSignalMatch(
                user_id=user_id,
                signal_id=signal_id,
                match_score=match_score,
                compatibility_reason=f"Risk compatibility: {explanation.get('risk_compatibility', 0):.1f}, Asset preference: {explanation.get('asset_preference', 0):.1f}"
            )
            
            self.db.add(match)
            await self.db.flush()
            await self.db.refresh(match)
            
            logger.info(f"Created match between user {user_id} and signal {signal_id} (score: {match_score})")
            
            return match
            
        except Exception as e:
            logger.error(f"Failed to create user-signal match: {e}")
            raise


# Global matcher instance (will be initialized with DB session as needed)
async def get_signal_matcher(db: AsyncSession) -> SignalMatcher:
    """Get signal matcher instance with database session"""
    return SignalMatcher(db)