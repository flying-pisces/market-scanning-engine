"""
Signal matching endpoints
Connect users with compatible trading signals
"""

import logging
from typing import List, Dict, Any, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, Query, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from app.core.database import get_db
from app.core.exceptions import NotFoundError, ValidationError
from app.models.database import User, Signal, UserSignalMatch
from app.services.matching import get_signal_matcher
from app.schemas.user import UserRiskProfile

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/matching", tags=["Matching"])


@router.post("/signals/{signal_id}/users")
async def match_signal_to_users(
    signal_id: UUID,
    max_matches: int = Query(50, ge=1, le=500, description="Maximum matches to return"),
    min_match_score: float = Query(60.0, ge=0, le=100, description="Minimum match score"),
    db: AsyncSession = Depends(get_db)
) -> List[Dict[str, Any]]:
    """Find users compatible with a specific signal"""
    
    try:
        # Get signal
        signal_result = await db.execute(
            select(Signal).where(Signal.id == signal_id)
        )
        signal = signal_result.scalar_one_or_none()
        
        if not signal:
            raise NotFoundError("Signal", str(signal_id))
        
        if not signal.is_active:
            raise ValidationError("Signal is not active")
        
        # Get matcher and find compatible users
        matcher = await get_signal_matcher(db)
        matches = await matcher.find_compatible_users(
            signal=signal,
            max_matches=max_matches,
            min_match_score=min_match_score
        )
        
        logger.info(f"Found {len(matches)} user matches for signal {signal_id}")
        return matches
        
    except (NotFoundError, ValidationError):
        raise
    except Exception as e:
        logger.error(f"Failed to match signal to users: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to find user matches"
        )


@router.post("/users/{user_id}/signals")
async def match_user_to_signals(
    user_id: UUID,
    max_signals: int = Query(25, ge=1, le=100, description="Maximum signals to return"),
    min_match_score: float = Query(70.0, ge=0, le=100, description="Minimum match score"),
    asset_classes: Optional[List[str]] = Query(None, description="Filter by asset classes"),
    db: AsyncSession = Depends(get_db)
) -> List[Dict[str, Any]]:
    """Find signals compatible with a specific user"""
    
    try:
        # Get user
        user_result = await db.execute(
            select(User).where(User.id == user_id)
        )
        user = user_result.scalar_one_or_none()
        
        if not user:
            raise NotFoundError("User", str(user_id))
        
        if not user.is_active:
            raise ValidationError("User is not active")
        
        # Get matcher and find compatible signals
        matcher = await get_signal_matcher(db)
        matches = await matcher.find_signals_for_user(
            user=user,
            max_signals=max_signals,
            min_match_score=min_match_score
        )
        
        # Filter by asset classes if requested
        if asset_classes:
            matches = [
                match for match in matches 
                if match["asset_class"] in asset_classes
            ]
        
        logger.info(f"Found {len(matches)} signal matches for user {user_id}")
        return matches
        
    except (NotFoundError, ValidationError):
        raise
    except Exception as e:
        logger.error(f"Failed to match user to signals: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to find signal matches"
        )


@router.post("/create-match", response_model=Dict[str, Any])
async def create_user_signal_match(
    user_id: UUID,
    signal_id: UUID,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Create a user-signal match record"""
    
    try:
        # Verify user and signal exist
        user_result = await db.execute(
            select(User).where(User.id == user_id)
        )
        user = user_result.scalar_one_or_none()
        
        signal_result = await db.execute(
            select(Signal).where(Signal.id == signal_id)
        )
        signal = signal_result.scalar_one_or_none()
        
        if not user:
            raise NotFoundError("User", str(user_id))
        if not signal:
            raise NotFoundError("Signal", str(signal_id))
        
        # Calculate match score
        matcher = await get_signal_matcher(db)
        match_score, explanation = matcher._calculate_match_score(signal, user)
        
        # Create match record
        match_record = await matcher.create_user_signal_match(
            user_id=user_id,
            signal_id=signal_id,
            match_score=match_score,
            explanation=explanation
        )
        
        # Background task: could send notification here
        # background_tasks.add_task(send_match_notification, user, signal, match_score)
        
        return {
            "match_id": match_record.id,
            "user_id": user_id,
            "signal_id": signal_id,
            "match_score": match_score,
            "explanation": explanation,
            "created_at": match_record.created_at,
            "status": "created"
        }
        
    except (NotFoundError, ValidationError):
        raise
    except Exception as e:
        logger.error(f"Failed to create match: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user-signal match"
        )


@router.get("/users/{user_id}/matches", response_model=List[Dict[str, Any]])
async def get_user_matches(
    user_id: UUID,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    include_viewed: bool = Query(False, description="Include already viewed matches"),
    db: AsyncSession = Depends(get_db)
) -> List[Dict[str, Any]]:
    """Get existing matches for a user"""
    
    try:
        # Verify user exists
        user_result = await db.execute(
            select(User).where(User.id == user_id)
        )
        if not user_result.scalar_one_or_none():
            raise NotFoundError("User", str(user_id))
        
        # Build query
        query = select(UserSignalMatch, Signal).join(
            Signal, UserSignalMatch.signal_id == Signal.id
        ).where(UserSignalMatch.user_id == user_id)
        
        if not include_viewed:
            query = query.where(UserSignalMatch.viewed_at.is_(None))
        
        query = query.order_by(UserSignalMatch.created_at.desc()).offset(skip).limit(limit)
        
        result = await db.execute(query)
        matches_with_signals = result.all()
        
        # Format response
        response = []
        for match, signal in matches_with_signals:
            response.append({
                "match_id": match.id,
                "signal_id": signal.id,
                "signal_name": signal.signal_name,
                "symbol": signal.symbol,
                "asset_class": signal.asset_class,
                "risk_score": signal.risk_score,
                "match_score": match.match_score,
                "compatibility_reason": match.compatibility_reason,
                "created_at": match.created_at,
                "viewed_at": match.viewed_at,
                "user_action": match.user_action,
                "signal_expires_at": signal.expires_at,
                "signal_active": signal.is_active,
            })
        
        logger.info(f"Retrieved {len(response)} matches for user {user_id}")
        return response
        
    except NotFoundError:
        raise
    except Exception as e:
        logger.error(f"Failed to get user matches: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user matches"
        )


@router.put("/matches/{match_id}/view")
async def mark_match_viewed(
    match_id: UUID,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Mark a match as viewed by the user"""
    
    try:
        result = await db.execute(
            select(UserSignalMatch).where(UserSignalMatch.id == match_id)
        )
        match = result.scalar_one_or_none()
        
        if not match:
            raise NotFoundError("Match", str(match_id))
        
        # Update viewed timestamp
        from datetime import datetime
        match.viewed_at = datetime.utcnow()
        if not match.user_action:
            match.user_action = "viewed"
        
        await db.flush()
        
        return {
            "match_id": match_id,
            "viewed_at": match.viewed_at,
            "status": "viewed"
        }
        
    except NotFoundError:
        raise
    except Exception as e:
        logger.error(f"Failed to mark match as viewed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update match"
        )


@router.post("/batch-match")
async def batch_generate_matches(
    background_tasks: BackgroundTasks,
    max_matches_per_signal: int = Query(100, ge=1, le=500),
    min_match_score: float = Query(60.0, ge=0, le=100),
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Generate matches for all active signals (background processing)"""
    
    try:
        # Get all active signals without many matches
        signals_result = await db.execute(
            select(Signal).where(
                and_(
                    Signal.is_active == True,
                    Signal.expires_at > datetime.utcnow()
                )
            ).limit(50)  # Process in batches for MVP
        )
        signals = signals_result.scalars().all()
        
        # Add background task to generate matches
        background_tasks.add_task(
            _batch_generate_matches_task,
            signals,
            max_matches_per_signal,
            min_match_score
        )
        
        return {
            "status": "processing",
            "signals_to_process": len(signals),
            "max_matches_per_signal": max_matches_per_signal,
            "min_match_score": min_match_score,
            "message": "Batch matching started in background"
        }
        
    except Exception as e:
        logger.error(f"Failed to start batch matching: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start batch matching"
        )


async def _batch_generate_matches_task(
    signals: List[Signal],
    max_matches_per_signal: int,
    min_match_score: float
):
    """Background task to generate matches"""
    
    # This would typically use a separate DB session for background processing
    # For MVP, we'll log the intent
    
    logger.info(f"Background matching task started for {len(signals)} signals")
    
    try:
        # In a real implementation, you'd:
        # 1. Create new DB session
        # 2. Process each signal
        # 3. Generate matches
        # 4. Store in database
        # 5. Optionally send notifications
        
        for signal in signals:
            logger.info(f"Would generate matches for signal {signal.id} ({signal.symbol})")
        
        logger.info("Background matching task completed")
        
    except Exception as e:
        logger.error(f"Background matching task failed: {e}")