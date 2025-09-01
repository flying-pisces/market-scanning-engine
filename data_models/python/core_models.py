"""
Market Scanning Engine - Core Pydantic Models
Author: Claude Code (System Architect)
Version: 1.0

Core data models for API validation and serialization.
All monetary values are stored in cents to avoid floating point precision issues.
All scores use 0-100 integer scale for consistency.
"""

from datetime import datetime, date, timedelta
from decimal import Decimal
from enum import Enum
from typing import List, Dict, Optional, Union, Any
from uuid import UUID, uuid4

from pydantic import (
    BaseModel, 
    Field, 
    validator,
    model_validator, 
    root_validator,
    # EmailStr,  # Optional - requires email-validator
    AnyHttpUrl,
    conint,
    condecimal,
    conlist,
    constr
)


# ============================================================================
# BASE CLASSES AND ENUMS
# ============================================================================

class BaseModelWithDefaults(BaseModel):
    """Base model with common configuration"""
    class Config:
        use_enum_values = True
        validate_assignment = True
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            date: lambda v: v.isoformat(),
            Decimal: lambda v: float(v),
            UUID: lambda v: str(v)
        }


class RiskScore(BaseModel):
    """0-100 risk score with validation"""
    score: conint(ge=0, le=100) = Field(..., description="Risk score from 0 (lowest risk) to 100 (highest risk)")
    
    @validator('score')
    def validate_score_range(cls, v):
        if not 0 <= v <= 100:
            raise ValueError('Score must be between 0 and 100')
        return v


# Enums for type safety
class AssetCategory(str, Enum):
    EQUITY = "equity"
    FIXED_INCOME = "fixed_income"
    DERIVATIVES = "derivatives"
    COMMODITY = "commodity"
    CRYPTO = "crypto"


class SignalDirection(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class SignalCategory(str, Enum):
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    OPTIONS = "options"
    NEWS = "news"
    MACRO = "macro"


class SignalStrength(str, Enum):
    WEAK = "WEAK"
    MODERATE = "MODERATE"
    STRONG = "STRONG"
    VERY_STRONG = "VERY_STRONG"


class SignalStatus(str, Enum):
    ACTIVE = "ACTIVE"
    EXPIRED = "EXPIRED"
    TRIGGERED = "TRIGGERED"
    CANCELLED = "CANCELLED"


class OptionType(str, Enum):
    CALL = "CALL"
    PUT = "PUT"


class MarketRegime(str, Enum):
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    SIDEWAYS = "sideways"
    HIGH_VOL = "high_vol"
    LOW_VOL = "low_vol"


class PositionStatus(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    CLOSING = "CLOSING"


# ============================================================================
# ASSET MODELS
# ============================================================================

class AssetClass(BaseModelWithDefaults):
    """Asset class definition"""
    id: Optional[int] = None
    name: constr(max_length=50) = Field(..., description="Asset class name")
    category: AssetCategory = Field(..., description="Asset class category")
    description: Optional[str] = None
    is_active: bool = True
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class Asset(BaseModelWithDefaults):
    """Financial asset definition"""
    id: Optional[UUID] = Field(default_factory=uuid4)
    symbol: constr(max_length=20) = Field(..., description="Asset symbol (e.g., AAPL)")
    name: constr(max_length=255) = Field(..., description="Full asset name")
    asset_class_id: Optional[int] = None
    exchange: constr(max_length=20) = Field(..., description="Exchange code")
    currency: constr(max_length=3, min_length=3) = Field(default="USD", description="Currency code")
    sector: Optional[constr(max_length=50)] = None
    industry: Optional[constr(max_length=100)] = None
    market_cap: Optional[int] = Field(None, description="Market cap in cents")
    avg_volume_30d: Optional[int] = Field(None, description="30-day average volume")
    is_active: bool = True
    listing_date: Optional[date] = None
    delisting_date: Optional[date] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @validator('market_cap')
    def validate_market_cap(cls, v):
        if v is not None and v < 0:
            raise ValueError('Market cap must be non-negative')
        return v

    @validator('avg_volume_30d')
    def validate_volume(cls, v):
        if v is not None and v < 0:
            raise ValueError('Volume must be non-negative')
        return v


# ============================================================================
# USER MODELS
# ============================================================================

class UserAssetPreference(BaseModelWithDefaults):
    """User preferences for specific asset classes"""
    id: Optional[UUID] = Field(default_factory=uuid4)
    user_id: UUID = Field(..., description="User profile ID")
    asset_class_id: int = Field(..., description="Asset class ID")
    preference_weight: condecimal(ge=0, le=1, decimal_places=2) = Field(..., description="Preference weight 0-1")
    is_enabled: bool = True
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class UserProfile(BaseModelWithDefaults):
    """Comprehensive user profile with risk preferences"""
    id: Optional[UUID] = Field(default_factory=uuid4)
    user_id: constr(max_length=255) = Field(..., description="External auth system ID")
    email: str = Field(..., description="User email address")
    display_name: Optional[constr(max_length=100)] = None

    # Risk preferences (0-100 scale)
    risk_tolerance: conint(ge=0, le=100) = Field(..., description="Risk tolerance score 0-100")
    max_position_size_pct: condecimal(ge=0, le=100, decimal_places=2) = Field(
        ..., description="Maximum position size as percentage of portfolio"
    )
    max_daily_loss_pct: condecimal(ge=0, le=100, decimal_places=2) = Field(
        ..., description="Maximum daily loss as percentage"
    )

    # Time horizon preferences
    min_holding_period_hours: int = Field(default=1, description="Minimum holding period in hours")
    max_holding_period_hours: int = Field(default=8760, description="Maximum holding period in hours")

    # Portfolio constraints
    max_open_positions: int = Field(default=10, description="Maximum number of open positions")
    min_trade_amount_cents: int = Field(default=100000, description="Minimum trade amount in cents ($1000)")
    max_trade_amount_cents: int = Field(default=10000000, description="Maximum trade amount in cents ($100k)")

    # Notification settings
    notification_preferences: Dict[str, Any] = Field(default_factory=dict)

    # Status
    is_active: bool = True
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    # Asset preferences (populated from separate table)
    asset_preferences: List[UserAssetPreference] = Field(default_factory=list)

    @validator('min_holding_period_hours', 'max_holding_period_hours')
    def validate_holding_periods(cls, v):
        if v < 0:
            raise ValueError('Holding period must be non-negative')
        return v

    @model_validator(mode="before")
    def validate_holding_period_range(cls, values):
        min_hours = values.get('min_holding_period_hours', 1)
        max_hours = values.get('max_holding_period_hours', 8760)
        if min_hours > max_hours:
            raise ValueError('Minimum holding period cannot exceed maximum')
        return values

    @validator('max_open_positions')
    def validate_max_positions(cls, v):
        if v < 1:
            raise ValueError('Must allow at least 1 open position')
        if v > 100:
            raise ValueError('Maximum 100 open positions allowed')
        return v

    @model_validator(mode="before")
    def validate_trade_amounts(cls, values):
        min_amount = values.get('min_trade_amount_cents', 0)
        max_amount = values.get('max_trade_amount_cents', 0)
        if min_amount > max_amount:
            raise ValueError('Minimum trade amount cannot exceed maximum')
        return values


# ============================================================================
# MARKET DATA MODELS
# ============================================================================

class MarketData(BaseModelWithDefaults):
    """Real-time and historical market data"""
    id: Optional[UUID] = Field(default_factory=uuid4)
    asset_id: UUID = Field(..., description="Asset ID reference")
    timestamp: datetime = Field(..., description="Data timestamp")

    # OHLCV data (in cents)
    open_price_cents: int = Field(..., description="Opening price in cents")
    high_price_cents: int = Field(..., description="High price in cents")
    low_price_cents: int = Field(..., description="Low price in cents")
    close_price_cents: int = Field(..., description="Closing price in cents")
    volume: int = Field(..., description="Trading volume")

    # Bid/Ask data
    bid_price_cents: Optional[int] = Field(None, description="Bid price in cents")
    ask_price_cents: Optional[int] = Field(None, description="Ask price in cents")
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None

    # Metadata
    data_source: constr(max_length=50) = Field(..., description="Data source identifier")
    data_quality_score: conint(ge=0, le=100) = Field(..., description="Data quality score 0-100")
    created_at: Optional[datetime] = None

    @validator('open_price_cents', 'high_price_cents', 'low_price_cents', 'close_price_cents')
    def validate_prices(cls, v):
        if v <= 0:
            raise ValueError('Prices must be positive')
        return v

    @validator('volume')
    def validate_volume(cls, v):
        if v < 0:
            raise ValueError('Volume must be non-negative')
        return v

    @model_validator(mode="before")
    def validate_ohlc_consistency(cls, values):
        """Validate OHLC price consistency"""
        open_price = values.get('open_price_cents')
        high_price = values.get('high_price_cents')
        low_price = values.get('low_price_cents')
        close_price = values.get('close_price_cents')

        if all(p is not None for p in [open_price, high_price, low_price, close_price]):
            if not (low_price <= open_price <= high_price and low_price <= close_price <= high_price):
                raise ValueError('OHLC prices are not consistent (High >= Open,Close >= Low)')
        
        return values


class TechnicalIndicators(BaseModelWithDefaults):
    """Technical analysis indicators"""
    id: Optional[UUID] = Field(default_factory=uuid4)
    asset_id: UUID = Field(..., description="Asset ID reference")
    timestamp: datetime = Field(..., description="Calculation timestamp")
    timeframe: constr(max_length=10) = Field(..., description="Timeframe (1m, 5m, 15m, 1h, 4h, 1d)")

    # Moving averages (in cents)
    sma_20: Optional[int] = Field(None, description="20-period SMA in cents")
    sma_50: Optional[int] = Field(None, description="50-period SMA in cents")
    sma_200: Optional[int] = Field(None, description="200-period SMA in cents")
    ema_12: Optional[int] = Field(None, description="12-period EMA in cents")
    ema_26: Optional[int] = Field(None, description="26-period EMA in cents")

    # Momentum indicators
    rsi_14: Optional[condecimal(ge=0, le=100, decimal_places=2)] = Field(None, description="14-period RSI")
    macd_line: Optional[int] = Field(None, description="MACD line in cents")
    macd_signal: Optional[int] = Field(None, description="MACD signal line in cents")
    macd_histogram: Optional[int] = Field(None, description="MACD histogram in cents")

    # Volatility indicators
    bollinger_upper: Optional[int] = Field(None, description="Bollinger upper band in cents")
    bollinger_middle: Optional[int] = Field(None, description="Bollinger middle band in cents")
    bollinger_lower: Optional[int] = Field(None, description="Bollinger lower band in cents")
    atr_14: Optional[int] = Field(None, description="14-period ATR in cents")

    # Volume indicators
    volume_sma_20: Optional[int] = Field(None, description="20-period volume SMA")
    on_balance_volume: Optional[int] = Field(None, description="On-balance volume")

    # Support/Resistance
    support_level: Optional[int] = Field(None, description="Support level in cents")
    resistance_level: Optional[int] = Field(None, description="Resistance level in cents")

    created_at: Optional[datetime] = None


class OptionsData(BaseModelWithDefaults):
    """Options chain data with Greeks"""
    id: Optional[UUID] = Field(default_factory=uuid4)
    underlying_asset_id: UUID = Field(..., description="Underlying asset ID")
    option_symbol: constr(max_length=30) = Field(..., description="Option symbol")

    # Contract specifications
    expiration_date: date = Field(..., description="Option expiration date")
    strike_price_cents: int = Field(..., description="Strike price in cents")
    option_type: OptionType = Field(..., description="Option type (CALL/PUT)")
    contract_size: int = Field(default=100, description="Contract size")

    # Market data
    timestamp: datetime = Field(..., description="Data timestamp")
    bid_price_cents: Optional[int] = Field(None, description="Bid price in cents")
    ask_price_cents: Optional[int] = Field(None, description="Ask price in cents")
    last_price_cents: Optional[int] = Field(None, description="Last traded price in cents")

    # Volume and open interest
    volume: int = Field(default=0, description="Daily volume")
    open_interest: int = Field(default=0, description="Open interest")

    # Greeks (scaled by 10000 for precision)
    delta: Optional[int] = Field(None, description="Delta scaled by 10000 (-10000 to 10000)")
    gamma: Optional[int] = Field(None, description="Gamma scaled by 10000 (0 to 100000)")
    theta: Optional[int] = Field(None, description="Theta scaled by 10000 (-10000 to 0)")
    vega: Optional[int] = Field(None, description="Vega scaled by 10000 (0 to 100000)")
    rho: Optional[int] = Field(None, description="Rho scaled by 10000 (-10000 to 10000)")

    # Implied volatility (scaled by 10000)
    implied_volatility: Optional[int] = Field(None, description="Implied volatility scaled by 10000")

    # Metadata
    data_source: constr(max_length=50) = Field(..., description="Data source")
    created_at: Optional[datetime] = None

    @validator('strike_price_cents')
    def validate_strike_price(cls, v):
        if v <= 0:
            raise ValueError('Strike price must be positive')
        return v

    @validator('volume', 'open_interest')
    def validate_volume_oi(cls, v):
        if v < 0:
            raise ValueError('Volume and open interest must be non-negative')
        return v

    @validator('delta')
    def validate_delta(cls, v):
        if v is not None and not -10000 <= v <= 10000:
            raise ValueError('Delta must be between -1.0 and 1.0 (scaled: -10000 to 10000)')
        return v

    @validator('gamma', 'vega')
    def validate_positive_greeks(cls, v):
        if v is not None and not 0 <= v <= 100000:
            raise ValueError('Gamma and Vega must be between 0.0 and 10.0 (scaled: 0 to 100000)')
        return v

    @validator('theta')
    def validate_theta(cls, v):
        if v is not None and not -10000 <= v <= 0:
            raise ValueError('Theta must be between -1.0 and 0.0 (scaled: -10000 to 0)')
        return v

    @validator('rho')
    def validate_rho(cls, v):
        if v is not None and not -10000 <= v <= 10000:
            raise ValueError('Rho must be between -1.0 and 1.0 (scaled: -10000 to 10000)')
        return v