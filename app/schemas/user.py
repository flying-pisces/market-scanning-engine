"""
Pydantic schemas for user-related endpoints
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from uuid import UUID

from pydantic import BaseModel, Field, validator, EmailStr


class UserBase(BaseModel):
    """Base user schema"""
    email: str = Field(..., description="User email address")
    display_name: Optional[str] = Field(None, max_length=100, description="Display name")
    risk_tolerance: int = Field(..., ge=0, le=100, description="Risk tolerance (0=YOLO, 100=conservative)")
    max_risk_deviation: Optional[int] = Field(25, ge=0, le=50, description="Max deviation from risk tolerance")
    asset_preferences: Optional[Dict[str, float]] = Field(default_factory=dict, description="Asset class preferences")
    max_position_size_cents: Optional[int] = Field(100000, ge=1000, description="Max position size in cents")
    daily_loss_limit_cents: Optional[int] = Field(50000, ge=1000, description="Daily loss limit in cents")


class UserCreate(UserBase):
    """Schema for creating a new user"""
    
    @validator('asset_preferences')
    def validate_asset_preferences(cls, v):
        if v:
            # Ensure all values are between 0 and 1
            for asset_class, preference in v.items():
                if not 0 <= preference <= 1:
                    raise ValueError(f"Asset preference for {asset_class} must be between 0 and 1")
            
            # Ensure total doesn't exceed 1.0 (with some tolerance)
            total = sum(v.values())
            if total > 1.1:
                raise ValueError("Total asset preferences cannot exceed 100%")
        
        return v


class UserUpdate(BaseModel):
    """Schema for updating user profile"""
    display_name: Optional[str] = Field(None, max_length=100)
    risk_tolerance: Optional[int] = Field(None, ge=0, le=100)
    max_risk_deviation: Optional[int] = Field(None, ge=0, le=50)
    asset_preferences: Optional[Dict[str, float]] = None
    max_position_size_cents: Optional[int] = Field(None, ge=1000)
    daily_loss_limit_cents: Optional[int] = Field(None, ge=1000)
    
    @validator('asset_preferences')
    def validate_asset_preferences(cls, v):
        if v:
            for asset_class, preference in v.items():
                if not 0 <= preference <= 1:
                    raise ValueError(f"Asset preference for {asset_class} must be between 0 and 1")
            
            total = sum(v.values())
            if total > 1.1:
                raise ValueError("Total asset preferences cannot exceed 100%")
        
        return v


class UserResponse(UserBase):
    """Schema for user response"""
    id: UUID
    created_at: datetime
    updated_at: Optional[datetime] = None
    is_active: bool
    last_login_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class UserRiskProfile(BaseModel):
    """Simplified risk profile for matching"""
    user_id: UUID
    risk_tolerance: int
    max_risk_deviation: int
    asset_preferences: Dict[str, float]
    max_position_size_cents: int
    
    class Config:
        from_attributes = True


class UserStats(BaseModel):
    """User statistics and activity"""
    user_id: UUID
    total_signals_received: int
    signals_viewed: int
    signals_acted_on: int
    average_match_score: Optional[float] = None
    last_activity: Optional[datetime] = None
    preferred_asset_classes: List[str] = []