"""
Database models for the Market Scanning Engine MVP
Basic SQLAlchemy models for users, signals, and matches
"""

from datetime import datetime, date
from typing import Optional, List
from decimal import Decimal

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, Text, 
    ForeignKey, Numeric, Date, JSON, Index
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from app.core.database import Base


class User(Base):
    """User profile model"""
    __tablename__ = "users"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Basic profile
    email = Column(String(255), unique=True, nullable=False)
    display_name = Column(String(100))
    is_active = Column(Boolean, default=True)
    
    # Risk preferences (0-100 scale)
    risk_tolerance = Column(Integer, nullable=False)  # 0=YOLO, 100=ultra-conservative
    max_risk_deviation = Column(Integer, default=25)  # How far from risk tolerance to match
    
    # Asset class preferences (JSON for MVP)
    asset_preferences = Column(JSON, default=dict)  # {"stocks": 0.4, "options": 0.3, ...}
    
    # Trading constraints
    max_position_size_cents = Column(Integer, default=100000)  # $1000 default
    daily_loss_limit_cents = Column(Integer, default=50000)    # $500 default
    
    # User activity
    last_login_at = Column(DateTime(timezone=True))
    
    # Relationships
    signals = relationship("UserSignalMatch", back_populates="user")
    
    # Indexes
    __table_args__ = (
        Index("ix_users_risk_tolerance", "risk_tolerance"),
        Index("ix_users_active_email", "email", "is_active"),
    )


class Signal(Base):
    """Trading signal model"""
    __tablename__ = "signals"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Signal identification
    signal_name = Column(String(255), nullable=False)
    signal_type = Column(String(50), nullable=False)  # "technical", "options_flow", etc.
    
    # Asset information
    asset_class = Column(String(50), nullable=False)  # "stocks", "daily_options", etc.
    symbol = Column(String(20), nullable=False)
    asset_name = Column(String(255))
    
    # Risk and scoring (0-100 scale)
    risk_score = Column(Numeric(5, 2), nullable=False)  # 0.00 to 100.00
    confidence_score = Column(Numeric(5, 2), nullable=False)  # 0.00 to 100.00
    profit_potential_score = Column(Numeric(5, 2), nullable=False)  # 0.00 to 100.00
    
    # Signal details
    direction = Column(String(10))  # "bullish", "bearish", "neutral"
    entry_price_cents = Column(Integer)  # Price in cents
    target_price_cents = Column(Integer)  # Target price in cents
    stop_loss_price_cents = Column(Integer)  # Stop loss in cents
    
    # Position sizing
    min_position_size_cents = Column(Integer, default=10000)  # $100 minimum
    max_position_size_cents = Column(Integer, default=100000)  # $1000 maximum
    
    # Signal validity
    expires_at = Column(DateTime(timezone=True))
    is_active = Column(Boolean, default=True)
    
    # Signal metadata
    generation_method = Column(String(100))  # How this signal was generated
    market_context = Column(JSON, default=dict)  # Additional market data
    
    # Processing status
    status = Column(String(20), default="pending")  # "pending", "matched", "expired"
    
    # Relationships
    matches = relationship("UserSignalMatch", back_populates="signal")
    
    # Indexes
    __table_args__ = (
        Index("ix_signals_risk_score", "risk_score"),
        Index("ix_signals_asset_class", "asset_class"),
        Index("ix_signals_symbol", "symbol"),
        Index("ix_signals_active", "is_active", "created_at"),
        Index("ix_signals_expires", "expires_at"),
    )


class UserSignalMatch(Base):
    """User-to-signal matching results"""
    __tablename__ = "user_signal_matches"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Foreign keys
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    signal_id = Column(UUID(as_uuid=True), ForeignKey("signals.id"), nullable=False)
    
    # Match scoring
    match_score = Column(Numeric(5, 2), nullable=False)  # 0.00 to 100.00
    compatibility_reason = Column(Text)  # Why this signal matched the user
    
    # User interaction
    viewed_at = Column(DateTime(timezone=True))
    user_action = Column(String(20))  # "viewed", "ignored", "favorited", "traded"
    user_feedback = Column(Integer)  # 1-5 rating
    
    # Notification status
    notification_sent_at = Column(DateTime(timezone=True))
    notification_method = Column(String(50))  # "email", "sms", "push", etc.
    
    # Relationships
    user = relationship("User", back_populates="signals")
    signal = relationship("Signal", back_populates="matches")
    
    # Constraints and indexes
    __table_args__ = (
        Index("ix_match_user_signal", "user_id", "signal_id", unique=True),
        Index("ix_match_score", "match_score"),
        Index("ix_match_created", "created_at"),
    )