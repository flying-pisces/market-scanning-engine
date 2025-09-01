"""
Market Scanning Engine - Portfolio and Execution Pydantic Models
Author: Claude Code (System Architect)
Version: 1.0

Models for portfolio management, trade execution, and performance tracking.
"""

from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import List, Dict, Optional, Union, Any
from uuid import UUID, uuid4

from pydantic import (
    BaseModel, 
    Field, 
    validator, 
    root_validator,
    conint,
    condecimal,
    conlist,
    constr
)

from .core_models import (
    BaseModelWithDefaults, 
    RiskScore, 
    PositionStatus
)


# ============================================================================
# MATCHING AND NOTIFICATION MODELS
# ============================================================================

class SignalMatch(BaseModelWithDefaults):
    """Signal to user matching results"""
    id: Optional[UUID] = Field(default_factory=uuid4)
    signal_id: UUID = Field(..., description="Signal ID reference")
    user_id: UUID = Field(..., description="User ID reference")

    # Matching timestamp
    matched_at: datetime = Field(default_factory=datetime.utcnow)

    # Match quality scores (0-100)
    overall_match_score: conint(ge=0, le=100) = Field(..., description="Overall match quality score")
    risk_tolerance_match: conint(ge=0, le=100) = Field(..., description="Risk tolerance match score")
    asset_preference_match: conint(ge=0, le=100) = Field(..., description="Asset preference match score")
    time_horizon_match: conint(ge=0, le=100) = Field(..., description="Time horizon match score")
    position_size_match: conint(ge=0, le=100) = Field(..., description="Position size match score")

    # Recommended position parameters
    recommended_position_size_pct: Optional[condecimal(ge=0, le=100, decimal_places=2)] = Field(
        None, description="Recommended position size %"
    )
    recommended_position_size_dollars_cents: Optional[int] = Field(
        None, description="Recommended position size in cents"
    )
    adjusted_stop_loss_cents: Optional[int] = Field(None, description="User-adjusted stop loss in cents")
    adjusted_target_price_cents: Optional[int] = Field(None, description="User-adjusted target price in cents")

    # Match reasoning
    match_factors: Dict[str, Any] = Field(default_factory=dict, description="Detailed matching factors")
    exclusion_reasons: List[str] = Field(default_factory=list, description="Reasons for non-match")

    # Notification status
    notification_status: str = Field(
        default="PENDING", 
        description="Notification status",
        regex="^(PENDING|SENT|READ|DISMISSED|ACTED_ON)$"
    )
    notification_sent_at: Optional[datetime] = None
    user_response: Optional[str] = Field(
        None, 
        description="User response to signal",
        regex="^(INTERESTED|NOT_INTERESTED|MAYBE|ALREADY_HAVE)$"
    )
    user_response_at: Optional[datetime] = None

    # Match expiration
    expires_at: Optional[datetime] = None

    # Metadata
    matching_algorithm_version: Optional[constr(max_length=20)] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @validator('recommended_position_size_dollars_cents')
    def validate_position_size_dollars(cls, v):
        if v is not None and v <= 0:
            raise ValueError('Position size in dollars must be positive')
        return v

    @validator('adjusted_stop_loss_cents', 'adjusted_target_price_cents')
    def validate_adjusted_prices(cls, v):
        if v is not None and v <= 0:
            raise ValueError('Adjusted prices must be positive')
        return v

    @property
    def is_expired(self) -> bool:
        """Check if the match has expired"""
        return self.expires_at is not None and self.expires_at < datetime.utcnow()


class UserSignalInteraction(BaseModelWithDefaults):
    """Track user interactions with signals"""
    id: Optional[UUID] = Field(default_factory=uuid4)
    signal_match_id: UUID = Field(..., description="Signal match ID reference")
    user_id: UUID = Field(..., description="User ID reference")
    signal_id: UUID = Field(..., description="Signal ID reference")

    # Interaction details
    interaction_type: constr(max_length=50) = Field(..., description="Type of interaction")
    interaction_timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Interaction context
    interaction_source: Optional[constr(max_length=50)] = Field(None, description="Source of interaction")
    session_id: Optional[constr(max_length=100)] = None

    # User modifications (if any)
    modified_position_size_pct: Optional[condecimal(ge=0, le=100, decimal_places=2)] = None
    modified_stop_loss_cents: Optional[int] = None
    modified_target_price_cents: Optional[int] = None
    modification_reason: Optional[str] = None

    # Interaction metadata
    device_type: Optional[constr(max_length=20)] = None
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None

    created_at: Optional[datetime] = None

    @validator('modified_stop_loss_cents', 'modified_target_price_cents')
    def validate_modified_prices(cls, v):
        if v is not None and v <= 0:
            raise ValueError('Modified prices must be positive')
        return v


# ============================================================================
# TRADE EXECUTION MODELS
# ============================================================================

class TradeExecution(BaseModelWithDefaults):
    """Trade execution record"""
    id: Optional[UUID] = Field(default_factory=uuid4)
    signal_match_id: Optional[UUID] = Field(None, description="Associated signal match ID")
    user_id: UUID = Field(..., description="User ID reference")
    signal_id: Optional[UUID] = Field(None, description="Associated signal ID")
    asset_id: UUID = Field(..., description="Asset ID reference")

    # Trade identification
    external_trade_id: Optional[constr(max_length=100)] = Field(None, description="Broker's trade ID")
    order_id: Optional[constr(max_length=100)] = Field(None, description="Internal order ID")

    # Trade details
    trade_type: str = Field(..., description="Trade type", regex="^(BUY|SELL|SELL_SHORT|BUY_TO_COVER)$")
    quantity: int = Field(..., description="Number of shares/contracts")
    execution_price_cents: int = Field(..., description="Execution price in cents")

    # Timing
    order_submitted_at: Optional[datetime] = None
    executed_at: datetime = Field(default_factory=datetime.utcnow)

    # Position sizing and risk management
    position_size_dollars_cents: int = Field(..., description="Position size in cents")
    position_size_pct_of_portfolio: Optional[condecimal(decimal_places=2)] = None
    stop_loss_price_cents: Optional[int] = None
    take_profit_price_cents: Optional[int] = None

    # Execution quality metrics
    slippage_bps: Optional[int] = Field(None, description="Slippage in basis points")
    execution_time_ms: Optional[int] = Field(None, description="Execution time in milliseconds")

    # Costs and fees (in cents)
    commission_cents: int = Field(default=0, description="Commission in cents")
    sec_fees_cents: int = Field(default=0, description="SEC fees in cents")
    other_fees_cents: int = Field(default=0, description="Other fees in cents")

    # Trade status
    status: str = Field(
        default="EXECUTED",
        description="Trade status",
        regex="^(PENDING|PARTIAL|EXECUTED|CANCELLED|REJECTED)$"
    )

    # Execution metadata
    broker_name: Optional[constr(max_length=50)] = None
    execution_venue: Optional[constr(max_length=50)] = None

    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @validator('quantity')
    def validate_quantity(cls, v):
        if v == 0:
            raise ValueError('Quantity cannot be zero')
        return v

    @validator('execution_price_cents')
    def validate_execution_price(cls, v):
        if v <= 0:
            raise ValueError('Execution price must be positive')
        return v

    @validator('position_size_dollars_cents')
    def validate_position_size(cls, v):
        if v <= 0:
            raise ValueError('Position size must be positive')
        return v

    @validator('slippage_bps')
    def validate_slippage(cls, v):
        if v is not None and v < -10000:  # Allow negative slippage (price improvement)
            raise ValueError('Slippage cannot be less than -10000 bps')
        return v

    @validator('execution_time_ms')
    def validate_execution_time(cls, v):
        if v is not None and v < 0:
            raise ValueError('Execution time must be non-negative')
        return v

    @property
    def total_cost_cents(self) -> int:
        """Calculate total transaction costs"""
        return self.commission_cents + self.sec_fees_cents + self.other_fees_cents

    @property
    def net_amount_cents(self) -> int:
        """Calculate net amount (position value + costs for buy, position value - costs for sell)"""
        if self.trade_type in ['BUY', 'BUY_TO_COVER']:
            return self.position_size_dollars_cents + self.total_cost_cents
        else:
            return self.position_size_dollars_cents - self.total_cost_cents


# ============================================================================
# POSITION MODELS
# ============================================================================

class Position(BaseModelWithDefaults):
    """Current portfolio position"""
    id: Optional[UUID] = Field(default_factory=uuid4)
    user_id: UUID = Field(..., description="User ID reference")
    asset_id: UUID = Field(..., description="Asset ID reference")

    # Position details
    quantity: int = Field(..., description="Position quantity (positive=long, negative=short)")
    average_cost_cents: int = Field(..., description="Average cost basis in cents")
    current_price_cents: Optional[int] = Field(None, description="Current market price in cents")

    # Position dates
    opened_at: datetime = Field(..., description="Position opening timestamp")
    last_updated_at: datetime = Field(default_factory=datetime.utcnow)
    closed_at: Optional[datetime] = Field(None, description="Position closing timestamp")

    # Risk management
    stop_loss_price_cents: Optional[int] = None
    take_profit_price_cents: Optional[int] = None
    trailing_stop_pct: Optional[condecimal(decimal_places=2)] = None

    # Realized P&L (set when position is closed)
    realized_pnl_cents: Optional[int] = None

    # Associated signals
    originating_signal_ids: List[UUID] = Field(default_factory=list)

    # Position status
    status: PositionStatus = Field(default=PositionStatus.OPEN)

    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @validator('quantity')
    def validate_quantity(cls, v):
        if v == 0:
            raise ValueError('Position quantity cannot be zero')
        return v

    @validator('average_cost_cents')
    def validate_average_cost(cls, v):
        if v <= 0:
            raise ValueError('Average cost must be positive')
        return v

    @validator('current_price_cents')
    def validate_current_price(cls, v):
        if v is not None and v <= 0:
            raise ValueError('Current price must be positive')
        return v

    @validator('stop_loss_price_cents', 'take_profit_price_cents')
    def validate_risk_prices(cls, v):
        if v is not None and v <= 0:
            raise ValueError('Risk management prices must be positive')
        return v

    @property
    def unrealized_pnl_cents(self) -> Optional[int]:
        """Calculate unrealized P&L"""
        if self.status == PositionStatus.OPEN and self.current_price_cents:
            return (self.current_price_cents - self.average_cost_cents) * self.quantity
        return None

    @property
    def position_value_cents(self) -> Optional[int]:
        """Calculate current position value"""
        if self.status == PositionStatus.OPEN and self.current_price_cents:
            return self.current_price_cents * abs(self.quantity)
        return None

    @property
    def is_long(self) -> bool:
        """Check if position is long"""
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        """Check if position is short"""
        return self.quantity < 0

    @property
    def unrealized_pnl_pct(self) -> Optional[float]:
        """Calculate unrealized P&L percentage"""
        if self.unrealized_pnl_cents and self.average_cost_cents:
            cost_basis = self.average_cost_cents * abs(self.quantity)
            return float(self.unrealized_pnl_cents) / float(cost_basis) * 100
        return None


class PositionHistory(BaseModelWithDefaults):
    """Position modification history"""
    id: Optional[UUID] = Field(default_factory=uuid4)
    position_id: UUID = Field(..., description="Position ID reference")

    # Historical snapshot
    snapshot_timestamp: datetime = Field(default_factory=datetime.utcnow)
    quantity: int = Field(..., description="Position quantity at snapshot")
    price_cents: int = Field(..., description="Price at snapshot in cents")

    # Change details
    change_type: constr(max_length=30) = Field(..., description="Type of change")
    quantity_change: int = Field(default=0, description="Change in quantity")

    # Associated trade (if applicable)
    trade_execution_id: Optional[UUID] = None

    # Reason for change
    change_reason: Optional[constr(max_length=255)] = None

    created_at: Optional[datetime] = None

    @validator('quantity')
    def validate_quantity(cls, v):
        if v == 0:
            raise ValueError('Quantity in history cannot be zero')
        return v

    @validator('price_cents')
    def validate_price(cls, v):
        if v <= 0:
            raise ValueError('Price must be positive')
        return v


# ============================================================================
# PORTFOLIO ANALYTICS MODELS
# ============================================================================

class PortfolioAllocation(BaseModelWithDefaults):
    """Portfolio asset allocation at a point in time"""
    id: Optional[UUID] = Field(default_factory=uuid4)
    portfolio_snapshot_id: UUID = Field(..., description="Portfolio snapshot ID reference")
    asset_id: UUID = Field(..., description="Asset ID reference")
    asset_class_id: Optional[int] = Field(None, description="Asset class ID reference")

    # Allocation details
    position_value_cents: int = Field(..., description="Position value in cents")
    weight_pct: condecimal(ge=0, le=100, decimal_places=2) = Field(..., description="Portfolio weight %")
    quantity: int = Field(..., description="Position quantity")
    average_cost_cents: int = Field(..., description="Average cost in cents")
    current_price_cents: int = Field(..., description="Current price in cents")

    # Position performance
    unrealized_pnl_cents: Optional[int] = None
    unrealized_pnl_pct: Optional[condecimal(decimal_places=4)] = None

    created_at: Optional[datetime] = None

    @validator('position_value_cents')
    def validate_position_value(cls, v):
        if v < 0:
            raise ValueError('Position value must be non-negative')
        return v

    @validator('quantity')
    def validate_quantity(cls, v):
        if v == 0:
            raise ValueError('Allocation quantity cannot be zero')
        return v

    @validator('average_cost_cents', 'current_price_cents')
    def validate_prices(cls, v):
        if v <= 0:
            raise ValueError('Prices must be positive')
        return v


class PortfolioSnapshot(BaseModelWithDefaults):
    """Daily portfolio performance snapshot"""
    id: Optional[UUID] = Field(default_factory=uuid4)
    user_id: UUID = Field(..., description="User ID reference")
    snapshot_date: date = Field(..., description="Snapshot date")

    # Portfolio values (in cents)
    total_value_cents: int = Field(..., description="Total portfolio value in cents")
    cash_balance_cents: int = Field(..., description="Cash balance in cents")
    equity_value_cents: int = Field(..., description="Equity value in cents")
    unrealized_pnl_cents: int = Field(default=0, description="Unrealized P&L in cents")
    realized_pnl_cents: int = Field(default=0, description="Realized P&L in cents")

    # Portfolio composition
    number_of_positions: int = Field(default=0, description="Number of positions")
    largest_position_pct: Optional[condecimal(decimal_places=2)] = Field(None, description="Largest position %")
    portfolio_beta: Optional[condecimal(decimal_places=4)] = Field(None, description="Portfolio beta")
    portfolio_var_1d_pct: Optional[condecimal(decimal_places=3)] = Field(None, description="1-day portfolio VaR %")

    # Risk metrics (0-100 scale)
    overall_risk_score: Optional[conint(ge=0, le=100)] = None
    concentration_risk_score: Optional[conint(ge=0, le=100)] = None
    liquidity_risk_score: Optional[conint(ge=0, le=100)] = None

    # Performance metrics
    daily_return_pct: Optional[condecimal(decimal_places=4)] = Field(None, description="Daily return %")
    mtd_return_pct: Optional[condecimal(decimal_places=4)] = Field(None, description="Month-to-date return %")
    ytd_return_pct: Optional[condecimal(decimal_places=4)] = Field(None, description="Year-to-date return %")
    total_return_pct: Optional[condecimal(decimal_places=4)] = Field(None, description="Total return %")

    # Drawdown tracking
    peak_value_cents: Optional[int] = Field(None, description="Peak portfolio value in cents")
    current_drawdown_pct: Optional[condecimal(decimal_places=3)] = Field(None, description="Current drawdown %")
    max_drawdown_pct: Optional[condecimal(decimal_places=3)] = Field(None, description="Maximum drawdown %")

    created_at: Optional[datetime] = None

    # Related allocations (populated from separate query)
    allocations: List[PortfolioAllocation] = Field(default_factory=list)

    @validator('total_value_cents', 'cash_balance_cents', 'equity_value_cents')
    def validate_portfolio_values(cls, v):
        if v < 0:
            raise ValueError('Portfolio values must be non-negative')
        return v

    @validator('number_of_positions')
    def validate_number_of_positions(cls, v):
        if v < 0:
            raise ValueError('Number of positions must be non-negative')
        return v

    @validator('peak_value_cents')
    def validate_peak_value(cls, v):
        if v is not None and v < 0:
            raise ValueError('Peak value must be non-negative')
        return v

    @model_validator(mode="before")
    def validate_portfolio_composition(cls, values):
        """Validate portfolio value consistency"""
        total = values.get('total_value_cents', 0)
        cash = values.get('cash_balance_cents', 0)
        equity = values.get('equity_value_cents', 0)
        
        if abs((cash + equity) - total) > 100:  # Allow 1 dollar tolerance for rounding
            raise ValueError('Cash + Equity must equal Total portfolio value')
        
        return values

    @property
    def net_pnl_cents(self) -> int:
        """Calculate total P&L"""
        return self.unrealized_pnl_cents + self.realized_pnl_cents

    @property
    def net_pnl_pct(self) -> Optional[float]:
        """Calculate total P&L percentage"""
        if self.total_value_cents > 0:
            cost_basis = self.total_value_cents - self.net_pnl_cents
            if cost_basis > 0:
                return float(self.net_pnl_cents) / float(cost_basis) * 100
        return None