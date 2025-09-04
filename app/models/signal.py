"""
Signal models for trading signal management
"""

from enum import Enum
from typing import Optional, Dict, Any, List
from datetime import datetime
from decimal import Decimal
from pydantic import BaseModel, Field
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID
import uuid

from .database import Base


class SignalType(str, Enum):
    """Trading signal types"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class AssetClass(str, Enum):
    """Asset classification"""
    EQUITY = "equity"
    BOND = "bond"
    COMMODITY = "commodity"
    CRYPTOCURRENCY = "cryptocurrency"
    FOREX = "forex"
    OPTION = "option"
    FUTURE = "future"
    ETF = "etf"


class TimeFrame(str, Enum):
    """Signal time frame"""
    MINUTE = "1m"
    FIVE_MINUTE = "5m"
    FIFTEEN_MINUTE = "15m"
    HOURLY = "1h"
    DAILY = "1d"
    WEEKLY = "1w"
    MONTHLY = "1M"


class SignalSource(str, Enum):
    """Signal generation source"""
    ML_MODEL = "ml_model"
    TECHNICAL_ANALYSIS = "technical_analysis"
    FUNDAMENTAL_ANALYSIS = "fundamental_analysis"
    NEWS_SENTIMENT = "news_sentiment"
    MANUAL = "manual"
    EXTERNAL_API = "external_api"


# Database Models
class SignalExtended(Base):
    """Extended signal database model"""
    __tablename__ = "signals_extended"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symbol = Column(String(20), nullable=False, index=True)
    signal_type = Column(SQLEnum(SignalType), nullable=False)
    confidence = Column(Float, nullable=False)
    target_price = Column(Float)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    asset_class = Column(SQLEnum(AssetClass), nullable=False)
    timeframe = Column(SQLEnum(TimeFrame), nullable=False)
    source = Column(SQLEnum(SignalSource), nullable=False, default=SignalSource.ML_MODEL)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)
    is_active = Column(Integer, default=1)
    
    # Analysis data
    analysis_data = Column(Text)  # JSON string
    market_conditions = Column(Text)  # JSON string
    
    # Performance tracking
    entry_price = Column(Float)
    exit_price = Column(Float)
    realized_pnl = Column(Float)
    
    def __repr__(self):
        return f"<Signal(symbol={self.symbol}, type={self.signal_type}, confidence={self.confidence})>"


# Pydantic Models
class SignalBase(BaseModel):
    """Base signal schema"""
    symbol: str = Field(..., min_length=1, max_length=20)
    signal_type: SignalType
    confidence: float = Field(..., ge=0.0, le=1.0)
    target_price: Optional[float] = Field(None, gt=0)
    stop_loss: Optional[float] = Field(None, gt=0)
    take_profit: Optional[float] = Field(None, gt=0)
    asset_class: AssetClass
    timeframe: TimeFrame
    source: SignalSource = SignalSource.ML_MODEL
    analysis_data: Optional[Dict[str, Any]] = None
    market_conditions: Optional[Dict[str, Any]] = None


class SignalCreate(SignalBase):
    """Signal creation schema"""
    expires_at: Optional[datetime] = None


class SignalUpdate(BaseModel):
    """Signal update schema"""
    signal_type: Optional[SignalType] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    target_price: Optional[float] = Field(None, gt=0)
    stop_loss: Optional[float] = Field(None, gt=0)
    take_profit: Optional[float] = Field(None, gt=0)
    is_active: Optional[bool] = None
    entry_price: Optional[float] = Field(None, gt=0)
    exit_price: Optional[float] = Field(None, gt=0)
    realized_pnl: Optional[float] = None


class SignalResponse(SignalBase):
    """Signal response schema"""
    id: uuid.UUID
    created_at: datetime
    expires_at: Optional[datetime] = None
    is_active: bool = True
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    realized_pnl: Optional[float] = None
    
    class Config:
        from_attributes = True


class SignalAnalysis(BaseModel):
    """Signal analysis results"""
    technical_indicators: Optional[Dict[str, float]] = None
    market_sentiment: Optional[float] = Field(None, ge=-1.0, le=1.0)
    volume_analysis: Optional[Dict[str, float]] = None
    price_momentum: Optional[float] = None
    volatility_score: Optional[float] = Field(None, ge=0.0)
    correlation_factors: Optional[Dict[str, float]] = None


class SignalPerformance(BaseModel):
    """Signal performance metrics"""
    signal_id: uuid.UUID
    symbol: str
    entry_time: datetime
    exit_time: Optional[datetime] = None
    entry_price: float
    exit_price: Optional[float] = None
    return_pct: Optional[float] = None
    duration_hours: Optional[float] = None
    max_favorable_excursion: Optional[float] = None
    max_adverse_excursion: Optional[float] = None
    
    
class SignalBatch(BaseModel):
    """Batch signal processing"""
    signals: List[SignalCreate]
    batch_id: Optional[str] = None
    processing_options: Optional[Dict[str, Any]] = None


class SignalFilter(BaseModel):
    """Signal filtering criteria"""
    symbols: Optional[List[str]] = None
    signal_types: Optional[List[SignalType]] = None
    asset_classes: Optional[List[AssetClass]] = None
    timeframes: Optional[List[TimeFrame]] = None
    min_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    max_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    sources: Optional[List[SignalSource]] = None
    active_only: bool = True
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    expires_after: Optional[datetime] = None
    expires_before: Optional[datetime] = None