"""
User management endpoints
CRUD operations for user profiles and risk preferences
"""

import logging
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload

from app.core.database import get_db
from app.core.exceptions import NotFoundError, ValidationError
from app.models.database import User, UserSignalMatch
from app.schemas.user import UserCreate, UserUpdate, UserResponse, UserStats
from app.services.risk_scoring import AssetClass

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/users", tags=["Users"])


@router.post("/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db)
) -> UserResponse:
    """Create a new user profile"""
    
    try:
        # Check if user already exists
        existing_user = await db.execute(
            select(User).where(User.email == user_data.email)
        )
        if existing_user.scalar_one_or_none():
            raise ValidationError(f"User with email {user_data.email} already exists")
        
        # Create new user
        user = User(
            email=user_data.email,
            display_name=user_data.display_name,
            risk_tolerance=user_data.risk_tolerance,
            max_risk_deviation=user_data.max_risk_deviation or 25,
            asset_preferences=user_data.asset_preferences or {},
            max_position_size_cents=user_data.max_position_size_cents or 100000,
            daily_loss_limit_cents=user_data.daily_loss_limit_cents or 50000,
        )
        
        db.add(user)
        await db.flush()  # Get the ID
        await db.refresh(user)
        
        logger.info(f"Created user {user.id} with email {user.email}")
        return UserResponse.from_orm(user)
        
    except ValidationError:
        raise
    except Exception as e:
        logger.error(f"Failed to create user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user"
        )


@router.get("/", response_model=List[UserResponse])
async def list_users(
    skip: int = Query(0, ge=0, description="Number of users to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Number of users to return"),
    risk_tolerance_min: Optional[int] = Query(None, ge=0, le=100, description="Minimum risk tolerance"),
    risk_tolerance_max: Optional[int] = Query(None, ge=0, le=100, description="Maximum risk tolerance"),
    active_only: bool = Query(True, description="Return only active users"),
    db: AsyncSession = Depends(get_db)
) -> List[UserResponse]:
    """List users with optional filtering"""
    
    try:
        query = select(User)
        
        # Apply filters
        if active_only:
            query = query.where(User.is_active == True)
        
        if risk_tolerance_min is not None:
            query = query.where(User.risk_tolerance >= risk_tolerance_min)
        
        if risk_tolerance_max is not None:
            query = query.where(User.risk_tolerance <= risk_tolerance_max)
        
        # Add pagination
        query = query.offset(skip).limit(limit).order_by(User.created_at.desc())
        
        result = await db.execute(query)
        users = result.scalars().all()
        
        logger.info(f"Retrieved {len(users)} users")
        return [UserResponse.from_orm(user) for user in users]
        
    except Exception as e:
        logger.error(f"Failed to list users: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve users"
        )


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: UUID,
    db: AsyncSession = Depends(get_db)
) -> UserResponse:
    """Get user by ID"""
    
    try:
        result = await db.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        
        if not user:
            raise NotFoundError("User", str(user_id))
        
        return UserResponse.from_orm(user)
        
    except NotFoundError:
        raise
    except Exception as e:
        logger.error(f"Failed to get user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user"
        )


@router.put("/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: UUID,
    user_data: UserUpdate,
    db: AsyncSession = Depends(get_db)
) -> UserResponse:
    """Update user profile"""
    
    try:
        result = await db.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        
        if not user:
            raise NotFoundError("User", str(user_id))
        
        # Update fields that were provided
        update_data = user_data.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(user, field, value)
        
        await db.flush()
        await db.refresh(user)
        
        logger.info(f"Updated user {user_id}")
        return UserResponse.from_orm(user)
        
    except NotFoundError:
        raise
    except Exception as e:
        logger.error(f"Failed to update user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user"
        )


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: UUID,
    db: AsyncSession = Depends(get_db)
) -> None:
    """Soft delete user (deactivate)"""
    
    try:
        result = await db.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        
        if not user:
            raise NotFoundError("User", str(user_id))
        
        # Soft delete - just deactivate the user
        user.is_active = False
        
        await db.flush()
        logger.info(f"Deactivated user {user_id}")
        
    except NotFoundError:
        raise
    except Exception as e:
        logger.error(f"Failed to delete user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete user"
        )


@router.get("/{user_id}/stats", response_model=UserStats)
async def get_user_stats(
    user_id: UUID,
    db: AsyncSession = Depends(get_db)
) -> UserStats:
    """Get user statistics and activity summary"""
    
    try:
        # Verify user exists
        user_result = await db.execute(
            select(User).where(User.id == user_id)
        )
        user = user_result.scalar_one_or_none()
        
        if not user:
            raise NotFoundError("User", str(user_id))
        
        # Get match statistics
        stats_query = select(
            func.count(UserSignalMatch.id).label("total_signals"),
            func.count(UserSignalMatch.viewed_at).label("signals_viewed"),
            func.count().filter(UserSignalMatch.user_action.isnot(None)).label("signals_acted_on"),
            func.avg(UserSignalMatch.match_score).label("avg_match_score"),
            func.max(UserSignalMatch.created_at).label("last_activity")
        ).where(UserSignalMatch.user_id == user_id)
        
        stats_result = await db.execute(stats_query)
        stats_row = stats_result.first()
        
        # Get preferred asset classes from user preferences
        preferred_classes = []
        if user.asset_preferences:
            # Sort by preference value and take top 3
            sorted_prefs = sorted(
                user.asset_preferences.items(),
                key=lambda x: x[1],
                reverse=True
            )
            preferred_classes = [asset_class for asset_class, _ in sorted_prefs[:3]]
        
        return UserStats(
            user_id=user_id,
            total_signals_received=stats_row.total_signals or 0,
            signals_viewed=stats_row.signals_viewed or 0,
            signals_acted_on=stats_row.signals_acted_on or 0,
            average_match_score=float(stats_row.avg_match_score) if stats_row.avg_match_score else None,
            last_activity=stats_row.last_activity,
            preferred_asset_classes=preferred_classes
        )
        
    except NotFoundError:
        raise
    except Exception as e:
        logger.error(f"Failed to get stats for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user statistics"
        )


@router.post("/{user_id}/asset-preferences", response_model=UserResponse)
async def update_asset_preferences(
    user_id: UUID,
    asset_preferences: dict[str, float],
    db: AsyncSession = Depends(get_db)
) -> UserResponse:
    """Update user's asset class preferences"""
    
    try:
        result = await db.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        
        if not user:
            raise NotFoundError("User", str(user_id))
        
        # Validate preferences
        valid_asset_classes = {e.value for e in AssetClass}
        for asset_class in asset_preferences.keys():
            if asset_class not in valid_asset_classes:
                raise ValidationError(f"Invalid asset class: {asset_class}")
        
        for value in asset_preferences.values():
            if not 0 <= value <= 1:
                raise ValidationError("Asset preferences must be between 0 and 1")
        
        total = sum(asset_preferences.values())
        if total > 1.1:
            raise ValidationError("Total asset preferences cannot exceed 100%")
        
        user.asset_preferences = asset_preferences
        await db.flush()
        await db.refresh(user)
        
        logger.info(f"Updated asset preferences for user {user_id}")
        return UserResponse.from_orm(user)
        
    except (NotFoundError, ValidationError):
        raise
    except Exception as e:
        logger.error(f"Failed to update asset preferences for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update asset preferences"
        )