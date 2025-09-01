"""
Market Scanning Engine - Signal and Risk Pydantic Models
Author: Claude Code (System Architect)
Version: 1.0

Models for signal generation, risk assessment, and portfolio management.
"""

from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import List, Dict, Optional, Union, Any
from uuid import UUID, uuid4

from pydantic import (
    BaseModel, 
    Field, 
    validator, 
    model_validator,
    root_validator,
    conint,
    condecimal,
    conlist,
    constr
)

from .core_models import (
    BaseModelWithDefaults, 
    RiskScore, 
    SignalDirection, 
    SignalCategory, 
    SignalStrength, 
    SignalStatus,
    MarketRegime,
    PositionStatus
)


# ============================================================================
# SIGNAL MODELS
# ============================================================================

class SignalType(BaseModelWithDefaults):
    """Signal type configuration"""
    id: Optional[int] = None
    name: constr(max_length=50) = Field(..., description="Signal type name")
    category: SignalCategory = Field(..., description="Signal category")
    description: Optional[str] = None
    is_active: bool = True
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class SignalFactor(BaseModelWithDefaults):
    """Individual factor contributing to a signal"""
    id: Optional[UUID] = Field(default_factory=uuid4)
    signal_id: UUID = Field(..., description="Parent signal ID")
    factor_name: constr(max_length=100) = Field(..., description="Factor name")
    factor_category: Optional[constr(max_length=50)] = Field(None, description="Factor category")

    # Factor contribution (0-100 scale)
    contribution_score: conint(ge=0, le=100) = Field(..., description="Factor contribution score 0-100")
    weight: condecimal(ge=0, le=1, decimal_places=4) = Field(..., description="Factor weight in signal")

    # Factor-specific data
    factor_value: Optional[Decimal] = Field(None, description="Raw factor value")
    factor_percentile: Optional[conint(ge=0, le=100)] = Field(None, description="Factor percentile rank")
    factor_z_score: Optional[condecimal(decimal_places=3)] = Field(None, description="Factor z-score")

    # Metadata
    calculation_method: Optional[constr(max_length=100)] = None
    data_source: Optional[constr(max_length=50)] = None
    lookback_periods: Optional[int] = None
    created_at: Optional[datetime] = None

    @validator('lookback_periods')
    def validate_lookback_periods(cls, v):
        if v is not None and v <= 0:
            raise ValueError('Lookback periods must be positive')
        return v


class SignalBacktest(BaseModelWithDefaults):
    """Backtesting results for a signal"""
    backtest_return_pct: Optional[condecimal(decimal_places=4)] = Field(None, description="Backtest return percentage")
    backtest_volatility_pct: Optional[condecimal(decimal_places=3)] = Field(None, description="Backtest volatility percentage")
    backtest_sample_size: Optional[int] = Field(None, description="Number of backtest samples")
    backtest_period_start: Optional[date] = Field(None, description="Backtest period start date")
    backtest_period_end: Optional[date] = Field(None, description="Backtest period end date")

    @validator('backtest_sample_size')
    def validate_sample_size(cls, v):
        if v is not None and v <= 0:
            raise ValueError('Backtest sample size must be positive')
        return v

    @model_validator(mode="before")
    def validate_backtest_period(cls, values):
        start_date = values.get('backtest_period_start')
        end_date = values.get('backtest_period_end')
        if start_date and end_date and start_date > end_date:
            raise ValueError('Backtest start date must be before end date')
        return values


class Signal(BaseModelWithDefaults):
    """Core signal with comprehensive scoring and metadata"""
    id: Optional[UUID] = Field(default_factory=uuid4)
    asset_id: UUID = Field(..., description="Target asset ID")
    signal_type_id: int = Field(..., description="Signal type ID")

    # Signal identification
    signal_name: constr(max_length=100) = Field(..., description="Signal display name")
    direction: SignalDirection = Field(..., description="Signal direction")

    # Timing information
    generated_at: datetime = Field(default_factory=datetime.utcnow, description="Signal generation timestamp")
    valid_until: Optional[datetime] = Field(None, description="Signal expiration time")
    target_entry_time: Optional[datetime] = Field(None, description="Recommended entry time")
    recommended_holding_period_hours: Optional[int] = Field(None, description="Recommended holding period")

    # Core scoring (0-100 scale)
    risk_score: conint(ge=0, le=100) = Field(..., description="Risk score 0-100")
    profit_potential_score: conint(ge=0, le=100) = Field(..., description="Profit potential score 0-100")
    confidence_score: conint(ge=0, le=100) = Field(..., description="Confidence score 0-100")

    # Risk-adjusted metrics
    sharpe_ratio: Optional[condecimal(decimal_places=3)] = Field(None, description="Expected Sharpe ratio")
    max_drawdown_pct: Optional[condecimal(decimal_places=2)] = Field(None, description="Expected max drawdown %")
    win_rate_pct: Optional[condecimal(decimal_places=2)] = Field(None, description="Historical win rate %")

    # Price targets (in cents)
    entry_price_cents: Optional[int] = Field(None, description="Recommended entry price in cents")
    target_price_cents: Optional[int] = Field(None, description="Target price in cents")
    stop_loss_price_cents: Optional[int] = Field(None, description="Stop loss price in cents")

    # Position sizing
    recommended_position_size_pct: Optional[condecimal(ge=0, le=100, decimal_places=2)] = Field(
        None, description="Recommended position size %"
    )
    max_position_size_pct: Optional[condecimal(ge=0, le=100, decimal_places=2)] = Field(
        None, description="Maximum position size %"
    )

    # Signal metadata
    signal_source: constr(max_length=100) = Field(..., description="Signal source/algorithm")
    methodology_version: Optional[constr(max_length=20)] = None
    signal_strength: Optional[SignalStrength] = None

    # Backtesting (embedded)
    backtesting: Optional[SignalBacktest] = None

    # Asset-specific data
    asset_specific_data: Dict[str, Any] = Field(default_factory=dict)

    # Additional metadata
    tags: List[str] = Field(default_factory=list)
    notes: Optional[str] = None

    # Status
    status: SignalStatus = Field(default=SignalStatus.ACTIVE)
    is_paper_trading: bool = False

    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    # Related objects (populated from other queries)
    factors: List[SignalFactor] = Field(default_factory=list)

    @validator('recommended_holding_period_hours')
    def validate_holding_period(cls, v):
        if v is not None and v <= 0:
            raise ValueError('Holding period must be positive')
        return v

    @validator('entry_price_cents', 'target_price_cents', 'stop_loss_price_cents')
    def validate_prices(cls, v):
        if v is not None and v <= 0:
            raise ValueError('Prices must be positive')
        return v

    @model_validator(mode="before")
    def validate_price_consistency(cls, values):
        """Validate price target consistency based on direction"""
        direction = values.get('direction')
        entry = values.get('entry_price_cents')
        target = values.get('target_price_cents')
        stop_loss = values.get('stop_loss_price_cents')

        if all(p is not None for p in [entry, target, stop_loss]):
            if direction == SignalDirection.BUY:
                if not (stop_loss < entry < target):
                    raise ValueError('For BUY signals: stop_loss < entry < target')
            elif direction == SignalDirection.SELL:
                if not (target < entry < stop_loss):
                    raise ValueError('For SELL signals: target < entry < stop_loss')

        return values

    @property
    def composite_score(self) -> int:
        """Calculate composite score from individual components"""
        return (self.risk_score + self.profit_potential_score + self.confidence_score) // 3

    @property
    def risk_adjusted_score(self) -> float:
        """Calculate risk-adjusted score using Sharpe ratio if available"""
        if self.sharpe_ratio:
            return float(self.profit_potential_score * self.confidence_score * min(self.sharpe_ratio, 3)) / 100
        return float(self.profit_potential_score * self.confidence_score) / 100


class SignalPerformance(BaseModelWithDefaults):
    """Track signal performance over time"""
    id: Optional[UUID] = Field(default_factory=uuid4)
    signal_id: UUID = Field(..., description="Signal ID reference")

    # Performance timestamp
    evaluation_date: date = Field(..., description="Performance evaluation date")
    days_since_signal: int = Field(..., description="Days since signal generation")

    # Price performance (in cents)
    entry_price_cents: Optional[int] = Field(None, description="Actual entry price in cents")
    current_price_cents: Optional[int] = Field(None, description="Current price in cents")
    high_since_signal_cents: Optional[int] = Field(None, description="Highest price since signal")
    low_since_signal_cents: Optional[int] = Field(None, description="Lowest price since signal")

    # Performance metrics
    unrealized_pnl_pct: Optional[condecimal(decimal_places=4)] = Field(None, description="Unrealized P&L %")
    realized_pnl_pct: Optional[condecimal(decimal_places=4)] = Field(None, description="Realized P&L %")
    max_favorable_excursion_pct: Optional[condecimal(decimal_places=4)] = Field(None, description="Max favorable move %")
    max_adverse_excursion_pct: Optional[condecimal(decimal_places=4)] = Field(None, description="Max adverse move %")

    # Risk metrics
    current_risk_score: Optional[conint(ge=0, le=100)] = Field(None, description="Current risk score")
    volatility_realized_pct: Optional[condecimal(decimal_places=3)] = Field(None, description="Realized volatility %")

    # Status
    position_status: Optional[str] = Field(None, description="Position status")
    created_at: Optional[datetime] = None

    @validator('days_since_signal')
    def validate_days_since_signal(cls, v):
        if v < 0:
            raise ValueError('Days since signal must be non-negative')
        return v


# ============================================================================
# RISK ASSESSMENT MODELS
# ============================================================================

class RiskFactor(BaseModelWithDefaults):
    """Risk factor configuration"""
    id: Optional[int] = None
    factor_name: constr(max_length=100) = Field(..., description="Risk factor name")
    factor_category: constr(max_length=50) = Field(..., description="Risk factor category")
    description: Optional[str] = None
    weight: condecimal(ge=0, le=1, decimal_places=4) = Field(..., description="Factor weight in risk model")
    is_active: bool = True
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class RiskFactorContribution(BaseModelWithDefaults):
    """Individual risk factor contribution to overall risk"""
    id: Optional[UUID] = Field(default_factory=uuid4)
    risk_assessment_id: UUID = Field(..., description="Parent risk assessment ID")
    risk_factor_id: int = Field(..., description="Risk factor ID")

    # Factor contribution
    factor_score: conint(ge=0, le=100) = Field(..., description="Factor risk score 0-100")
    contribution_weight: condecimal(ge=0, le=1, decimal_places=4) = Field(..., description="Factor weight")
    marginal_contribution: Optional[condecimal(decimal_places=3)] = Field(None, description="Marginal risk contribution")

    # Factor-specific metrics
    factor_value: Optional[Decimal] = Field(None, description="Raw factor value")
    factor_percentile: Optional[conint(ge=0, le=100)] = Field(None, description="Factor percentile")
    factor_z_score: Optional[condecimal(decimal_places=3)] = Field(None, description="Factor z-score")

    created_at: Optional[datetime] = None


class RiskAssessment(BaseModelWithDefaults):
    """Comprehensive multi-factor risk assessment"""
    id: Optional[UUID] = Field(default_factory=uuid4)
    asset_id: UUID = Field(..., description="Asset ID reference")
    assessment_timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Overall risk score (0-100)
    overall_risk_score: conint(ge=0, le=100) = Field(..., description="Overall risk score")

    # Component risk scores (0-100 each)
    market_risk_score: Optional[conint(ge=0, le=100)] = Field(None, description="Market risk component")
    liquidity_risk_score: Optional[conint(ge=0, le=100)] = Field(None, description="Liquidity risk component")
    credit_risk_score: Optional[conint(ge=0, le=100)] = Field(None, description="Credit risk component")
    volatility_risk_score: Optional[conint(ge=0, le=100)] = Field(None, description="Volatility risk component")
    concentration_risk_score: Optional[conint(ge=0, le=100)] = Field(None, description="Concentration risk component")

    # Quantitative risk metrics
    beta: Optional[condecimal(decimal_places=4)] = Field(None, description="Beta coefficient")
    var_1d_pct: Optional[condecimal(decimal_places=3)] = Field(None, description="1-day VaR (95%)")
    var_5d_pct: Optional[condecimal(decimal_places=3)] = Field(None, description="5-day VaR (95%)")
    expected_shortfall_pct: Optional[condecimal(decimal_places=3)] = Field(None, description="Expected shortfall")

    # Volatility measures
    realized_volatility_30d_pct: Optional[condecimal(decimal_places=3)] = Field(None, description="30d realized vol %")
    implied_volatility_pct: Optional[condecimal(decimal_places=3)] = Field(None, description="Implied volatility %")
    volatility_skew: Optional[condecimal(decimal_places=3)] = Field(None, description="Volatility skew")

    # Liquidity measures
    bid_ask_spread_bps: Optional[int] = Field(None, description="Bid-ask spread in basis points")
    average_daily_volume_20d: Optional[int] = Field(None, description="20-day average daily volume")
    turnover_ratio: Optional[condecimal(decimal_places=4)] = Field(None, description="Turnover ratio")
    market_impact_score: Optional[conint(ge=0, le=100)] = Field(None, description="Market impact score")

    # Market regime indicators
    market_regime: Optional[MarketRegime] = Field(None, description="Current market regime")
    regime_confidence_pct: Optional[conint(ge=0, le=100)] = Field(None, description="Regime confidence")

    # Model metadata
    model_version: Optional[constr(max_length=20)] = None
    calculation_method: Optional[constr(max_length=100)] = None
    data_quality_score: Optional[conint(ge=0, le=100)] = Field(None, description="Data quality score")

    created_at: Optional[datetime] = None

    # Related objects
    factor_contributions: List[RiskFactorContribution] = Field(default_factory=list)

    @validator('bid_ask_spread_bps')
    def validate_bid_ask_spread(cls, v):
        if v is not None and v < 0:
            raise ValueError('Bid-ask spread must be non-negative')
        return v

    @validator('average_daily_volume_20d')
    def validate_volume(cls, v):
        if v is not None and v < 0:
            raise ValueError('Volume must be non-negative')
        return v


class MarketRegimeData(BaseModelWithDefaults):
    """Market regime classification and indicators"""
    id: Optional[UUID] = Field(default_factory=uuid4)
    regime_date: date = Field(..., description="Regime date")

    # Regime classification
    primary_regime: constr(max_length=30) = Field(..., description="Primary market regime")
    volatility_regime: constr(max_length=20) = Field(..., description="Volatility regime")
    liquidity_regime: constr(max_length=20) = Field(..., description="Liquidity regime")

    # Regime confidence scores (0-100)
    regime_confidence: conint(ge=0, le=100) = Field(..., description="Regime classification confidence")
    regime_stability: conint(ge=0, le=100) = Field(..., description="Regime stability score")

    # Market-wide risk metrics
    market_stress_index: conint(ge=0, le=100) = Field(..., description="Market stress index")
    correlation_regime: Optional[constr(max_length=20)] = Field(None, description="Correlation regime")

    # Supporting indicators
    vix_level: Optional[condecimal(decimal_places=2)] = Field(None, description="VIX level")
    credit_spreads_bps: Optional[int] = Field(None, description="Credit spreads in bps")
    yield_curve_slope_bps: Optional[int] = Field(None, description="Yield curve slope in bps")
    dollar_strength_index: Optional[condecimal(decimal_places=3)] = Field(None, description="Dollar strength index")

    # Model metadata
    model_version: Optional[constr(max_length=20)] = None

    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @validator('vix_level')
    def validate_vix_level(cls, v):
        if v is not None and v < 0:
            raise ValueError('VIX level must be non-negative')
        return v