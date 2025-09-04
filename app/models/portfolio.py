"""
Portfolio models for portfolio management and tracking
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


class PositionType(str, Enum):
    """Position types"""
    LONG = "long"
    SHORT = "short"


class PositionStatus(str, Enum):
    """Position status"""
    OPEN = "open"
    CLOSED = "closed"
    PARTIAL = "partial"


class OrderType(str, Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(str, Enum):
    """Order status"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


# Database Models
class Portfolio(Base):
    """Portfolio database model"""
    __tablename__ = "portfolios"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    
    # Portfolio metrics
    total_value = Column(String(20), default="0")  # Store as string for precision
    cash_balance = Column(String(20), default="0")
    invested_amount = Column(String(20), default="0")
    unrealized_pnl = Column(String(20), default="0")
    realized_pnl = Column(String(20), default="0")
    
    # Risk metrics
    beta = Column(Float)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    volatility = Column(Float)
    var_95 = Column(Float)
    
    # Settings
    rebalance_frequency = Column(String(20), default="monthly")
    auto_rebalancing = Column(Integer, default=0)  # Boolean as integer
    risk_target = Column(Float, default=0.15)
    max_position_size = Column(Float, default=0.10)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<Portfolio(user_id={self.user_id}, name={self.name})>"


class Position(Base):
    """Position database model"""
    __tablename__ = "positions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    portfolio_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    
    # Position details
    position_type = Column(SQLEnum(PositionType), default=PositionType.LONG)
    status = Column(SQLEnum(PositionStatus), default=PositionStatus.OPEN)
    quantity = Column(String(20), nullable=False)  # Store as string for precision
    average_price = Column(String(20), nullable=False)
    current_price = Column(String(20))
    
    # Calculated values
    market_value = Column(String(20))
    unrealized_pnl = Column(String(20), default="0")
    unrealized_pnl_pct = Column(Float, default=0.0)
    
    # Entry/Exit tracking
    entry_date = Column(DateTime, default=datetime.utcnow)
    exit_date = Column(DateTime)
    holding_period = Column(Integer)  # Days
    
    # Cost basis tracking
    total_cost = Column(String(20))
    fees_paid = Column(String(20), default="0")
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<Position(symbol={self.symbol}, quantity={self.quantity}, status={self.status})>"


# Pydantic Models
class PositionBase(BaseModel):
    """Base position schema"""
    symbol: str = Field(..., min_length=1, max_length=20)
    position_type: PositionType = PositionType.LONG
    quantity: Decimal = Field(..., gt=0)
    average_price: Decimal = Field(..., gt=0)
    current_price: Optional[Decimal] = Field(None, gt=0)


class PositionCreate(PositionBase):
    """Position creation schema"""
    portfolio_id: uuid.UUID


class PositionUpdate(BaseModel):
    """Position update schema"""
    quantity: Optional[Decimal] = Field(None, gt=0)
    current_price: Optional[Decimal] = Field(None, gt=0)
    status: Optional[PositionStatus] = None


class PositionResponse(PositionBase):
    """Position response schema"""
    id: uuid.UUID
    portfolio_id: uuid.UUID
    status: PositionStatus
    market_value: Decimal
    unrealized_pnl: Decimal
    unrealized_pnl_pct: float
    entry_date: datetime
    exit_date: Optional[datetime] = None
    holding_period: Optional[int] = None
    total_cost: Decimal
    fees_paid: Decimal
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class PortfolioBase(BaseModel):
    """Base portfolio schema"""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    rebalance_frequency: Optional[str] = "monthly"
    auto_rebalancing: bool = False
    risk_target: Optional[float] = Field(None, gt=0, le=1.0)
    max_position_size: Optional[float] = Field(None, gt=0, le=1.0)


class PortfolioCreate(PortfolioBase):
    """Portfolio creation schema"""
    user_id: uuid.UUID


class PortfolioUpdate(BaseModel):
    """Portfolio update schema"""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = None
    rebalance_frequency: Optional[str] = None
    auto_rebalancing: Optional[bool] = None
    risk_target: Optional[float] = Field(None, gt=0, le=1.0)
    max_position_size: Optional[float] = Field(None, gt=0, le=1.0)


class PortfolioResponse(PortfolioBase):
    """Portfolio response schema"""
    id: uuid.UUID
    user_id: uuid.UUID
    total_value: Decimal
    cash_balance: Decimal
    invested_amount: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    beta: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    volatility: Optional[float] = None
    var_95: Optional[float] = None
    positions: List[PositionResponse] = []
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class PortfolioAllocation(BaseModel):
    """Portfolio allocation schema"""
    portfolio_id: uuid.UUID
    allocations: Dict[str, float]  # symbol -> weight
    total_allocation: float = Field(..., ge=0.99, le=1.01)
    rebalancing_needed: bool = False
    rebalancing_threshold: float = 0.05
    expected_return: Optional[float] = None
    expected_volatility: Optional[float] = None
    
    
class PortfolioPerformance(BaseModel):
    """Portfolio performance metrics"""
    portfolio_id: uuid.UUID
    period: str  # e.g., "1D", "1W", "1M", "1Y"
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    beta: Optional[float] = None
    alpha: Optional[float] = None
    information_ratio: Optional[float] = None
    calmar_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    
    # Benchmark comparison
    benchmark_return: Optional[float] = None
    excess_return: Optional[float] = None
    tracking_error: Optional[float] = None
    
    # Risk metrics
    var_95: Optional[float] = None
    cvar_95: Optional[float] = None
    downside_deviation: Optional[float] = None
    upside_capture: Optional[float] = None
    downside_capture: Optional[float] = None


class PortfolioSummary(BaseModel):
    """Portfolio summary information"""
    portfolio_id: uuid.UUID
    name: str
    total_value: Decimal
    day_change: Decimal
    day_change_pct: float
    total_return: Decimal
    total_return_pct: float
    position_count: int
    cash_percentage: float
    top_holdings: List[Dict[str, Any]]
    sector_allocation: Dict[str, float]
    asset_class_allocation: Dict[str, float]
    last_updated: datetime


class RebalancingRecommendation(BaseModel):
    """Portfolio rebalancing recommendation"""
    portfolio_id: uuid.UUID
    current_allocation: Dict[str, float]
    target_allocation: Dict[str, float]
    recommended_trades: List[Dict[str, Any]]
    total_turnover: float
    estimated_cost: Decimal
    expected_benefit: Optional[str] = None
    risk_reduction: Optional[float] = None
    
    
class PortfolioOptimizationRequest(BaseModel):
    """Portfolio optimization request"""
    portfolio_id: uuid.UUID
    optimization_method: str  # "mean_variance", "risk_parity", etc.
    constraints: Optional[Dict[str, Any]] = None
    target_return: Optional[float] = None
    risk_tolerance: Optional[str] = None
    universe: Optional[List[str]] = None  # Symbol universe
    
    
class PortfolioAnalytics(BaseModel):
    """Portfolio analytics and insights"""
    portfolio_id: uuid.UUID
    analysis_date: datetime
    
    # Performance attribution
    sector_contribution: Dict[str, float]
    security_contribution: Dict[str, float]
    
    # Risk analysis
    risk_factors: Dict[str, float]
    correlation_matrix: Optional[Dict[str, Dict[str, float]]] = None
    
    # Recommendations
    rebalancing_score: float
    diversification_score: float
    risk_adjusted_score: float
    
    # Alerts
    risk_alerts: List[str]
    performance_alerts: List[str]
    rebalancing_alerts: List[str]