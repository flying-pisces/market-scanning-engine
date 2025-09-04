"""
User models for user management and authentication
"""

from enum import Enum
from typing import Optional, Dict, Any, List
from datetime import datetime
from decimal import Decimal
from pydantic import BaseModel, Field, EmailStr
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Enum as SQLEnum, Boolean
from sqlalchemy.dialects.postgresql import UUID
import uuid

from .database import Base


class RiskTolerance(str, Enum):
    """User risk tolerance levels"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    VERY_AGGRESSIVE = "very_aggressive"


class UserStatus(str, Enum):
    """User account status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING_VERIFICATION = "pending_verification"


class SubscriptionTier(str, Enum):
    """User subscription tiers"""
    FREE = "free"
    BASIC = "basic"
    PREMIUM = "premium"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


class TradingExperience(str, Enum):
    """User trading experience levels"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


# Database Models
class UserExtended(Base):
    """Extended user database model"""
    __tablename__ = "users_extended"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    
    # Profile information
    first_name = Column(String(100))
    last_name = Column(String(100))
    phone = Column(String(20))
    
    # Account settings
    status = Column(SQLEnum(UserStatus), default=UserStatus.PENDING_VERIFICATION)
    subscription_tier = Column(SQLEnum(SubscriptionTier), default=SubscriptionTier.FREE)
    risk_tolerance = Column(SQLEnum(RiskTolerance), default=RiskTolerance.MODERATE)
    trading_experience = Column(SQLEnum(TradingExperience), default=TradingExperience.BEGINNER)
    
    # Financial information
    investment_amount = Column(String(20))  # Store as string to maintain precision
    annual_income = Column(String(20))
    net_worth = Column(String(20))
    
    # Preferences
    preferred_asset_classes = Column(Text)  # JSON string
    notification_preferences = Column(Text)  # JSON string
    trading_preferences = Column(Text)  # JSON string
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime)
    email_verified_at = Column(DateTime)
    
    # Security
    failed_login_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime)
    two_factor_enabled = Column(Boolean, default=False)
    two_factor_secret = Column(String(32))
    
    def __repr__(self):
        return f"<User(username={self.username}, email={self.email})>"


class UserProfile(Base):
    """Extended user profile information"""
    __tablename__ = "user_profiles"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    
    # Trading preferences
    max_position_size = Column(Float, default=0.1)  # As percentage of portfolio
    max_daily_loss = Column(Float, default=0.02)   # As percentage of portfolio
    preferred_timeframes = Column(Text)  # JSON array
    preferred_markets = Column(Text)     # JSON array
    
    # Risk management
    stop_loss_percentage = Column(Float, default=0.05)
    take_profit_percentage = Column(Float, default=0.10)
    max_open_positions = Column(Integer, default=10)
    correlation_limit = Column(Float, default=0.3)
    
    # Performance tracking
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    total_pnl = Column(String(20), default="0")
    best_trade = Column(String(20), default="0")
    worst_trade = Column(String(20), default="0")
    
    # Analytics
    avg_holding_period = Column(Float)  # In hours
    preferred_entry_time = Column(String(5))  # HH:MM format
    most_traded_symbols = Column(Text)  # JSON array
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# Pydantic Models
class UserBase(BaseModel):
    """Base user schema"""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    first_name: Optional[str] = Field(None, max_length=100)
    last_name: Optional[str] = Field(None, max_length=100)
    phone: Optional[str] = Field(None, max_length=20)
    risk_tolerance: RiskTolerance = RiskTolerance.MODERATE
    trading_experience: TradingExperience = TradingExperience.BEGINNER
    investment_amount: Optional[Decimal] = Field(None, gt=0)
    annual_income: Optional[Decimal] = Field(None, gt=0)
    net_worth: Optional[Decimal] = Field(None, gt=0)


class UserCreate(UserBase):
    """User creation schema"""
    password: str = Field(..., min_length=8)
    confirm_password: str = Field(..., min_length=8)
    
    def validate_passwords(self):
        if self.password != self.confirm_password:
            raise ValueError("Passwords do not match")


class UserUpdate(BaseModel):
    """User update schema"""
    first_name: Optional[str] = Field(None, max_length=100)
    last_name: Optional[str] = Field(None, max_length=100)
    phone: Optional[str] = Field(None, max_length=20)
    risk_tolerance: Optional[RiskTolerance] = None
    trading_experience: Optional[TradingExperience] = None
    investment_amount: Optional[Decimal] = Field(None, gt=0)
    annual_income: Optional[Decimal] = Field(None, gt=0)
    net_worth: Optional[Decimal] = Field(None, gt=0)
    preferred_asset_classes: Optional[List[str]] = None
    notification_preferences: Optional[Dict[str, Any]] = None
    trading_preferences: Optional[Dict[str, Any]] = None


class UserResponse(UserBase):
    """User response schema"""
    id: uuid.UUID
    status: UserStatus
    subscription_tier: SubscriptionTier
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime] = None
    email_verified: bool = False
    two_factor_enabled: bool = False
    
    class Config:
        from_attributes = True


class UserLogin(BaseModel):
    """User login schema"""
    username: str
    password: str
    remember_me: bool = False
    two_factor_code: Optional[str] = Field(None, min_length=6, max_length=6)


class UserPreferences(BaseModel):
    """User preferences schema"""
    preferred_asset_classes: Optional[List[str]] = None
    notification_preferences: Optional[Dict[str, Any]] = None
    trading_preferences: Optional[Dict[str, Any]] = None
    risk_management_settings: Optional[Dict[str, Any]] = None


class UserProfileUpdate(BaseModel):
    """User profile update schema"""
    max_position_size: Optional[float] = Field(None, gt=0, le=1.0)
    max_daily_loss: Optional[float] = Field(None, gt=0, le=1.0)
    preferred_timeframes: Optional[List[str]] = None
    preferred_markets: Optional[List[str]] = None
    stop_loss_percentage: Optional[float] = Field(None, gt=0, le=1.0)
    take_profit_percentage: Optional[float] = Field(None, gt=0, le=1.0)
    max_open_positions: Optional[int] = Field(None, gt=0)
    correlation_limit: Optional[float] = Field(None, gt=0, le=1.0)


class UserStats(BaseModel):
    """User trading statistics"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: Decimal = Decimal("0")
    best_trade: Decimal = Decimal("0")
    worst_trade: Decimal = Decimal("0")
    avg_holding_period: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    profit_factor: Optional[float] = None


class PasswordReset(BaseModel):
    """Password reset schema"""
    token: str
    new_password: str = Field(..., min_length=8)
    confirm_password: str = Field(..., min_length=8)
    
    def validate_passwords(self):
        if self.new_password != self.confirm_password:
            raise ValueError("Passwords do not match")


class EmailVerification(BaseModel):
    """Email verification schema"""
    token: str
    
    
class TwoFactorSetup(BaseModel):
    """Two-factor authentication setup"""
    secret: str
    verification_code: str = Field(..., min_length=6, max_length=6)


class UserActivity(BaseModel):
    """User activity tracking"""
    user_id: uuid.UUID
    activity_type: str
    description: str
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None