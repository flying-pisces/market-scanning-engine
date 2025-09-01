"""
Pydantic schemas for signal-related endpoints
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from uuid import UUID
from decimal import Decimal

from pydantic import BaseModel, Field, validator


class SignalBase(BaseModel):
    """Base signal schema"""
    signal_name: str = Field(..., max_length=255, description="Signal name")
    signal_type: str = Field(..., max_length=50, description="Signal type (technical, options_flow, etc.)")
    asset_class: str = Field(..., max_length=50, description="Asset class (stocks, daily_options, etc.)")
    symbol: str = Field(..., max_length=20, description="Asset symbol")
    asset_name: Optional[str] = Field(None, max_length=255, description="Full asset name")
    
    # Core scoring (0-100 scale)
    risk_score: float = Field(..., ge=0, le=100, description="Risk score (0=safe, 100=risky)")
    confidence_score: float = Field(..., ge=0, le=100, description="Confidence in signal")
    profit_potential_score: float = Field(..., ge=0, le=100, description="Expected profit potential")
    
    # Trading details
    direction: Optional[str] = Field(None, description="Signal direction (bullish/bearish/neutral)")
    entry_price_cents: Optional[int] = Field(None, ge=0, description="Entry price in cents")
    target_price_cents: Optional[int] = Field(None, ge=0, description="Target price in cents")
    stop_loss_price_cents: Optional[int] = Field(None, ge=0, description="Stop loss price in cents")
    
    # Position sizing
    min_position_size_cents: Optional[int] = Field(10000, ge=1000, description="Minimum position size")
    max_position_size_cents: Optional[int] = Field(100000, ge=1000, description="Maximum position size")
    
    # Timing
    expires_at: Optional[datetime] = Field(None, description="Signal expiration time")
    
    # Additional data
    generation_method: Optional[str] = Field(None, max_length=100, description="How signal was generated")
    market_context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Market context data")


class SignalCreate(SignalBase):
    """Schema for creating a new signal"""
    
    @validator('min_position_size_cents', 'max_position_size_cents')
    def validate_position_sizes(cls, v):
        if v is not None and v < 1000:  # $10 minimum
            raise ValueError("Position size must be at least $10 (1000 cents)")
        return v
    
    @validator('max_position_size_cents')
    def validate_position_size_range(cls, v, values):
        min_size = values.get('min_position_size_cents')
        if min_size and v and v < min_size:
            raise ValueError("Maximum position size must be >= minimum position size")
        return v
    
    @validator('target_price_cents')
    def validate_target_price(cls, v, values):
        entry_price = values.get('entry_price_cents')
        direction = values.get('direction')
        
        if v and entry_price and direction:
            if direction.lower() == 'bullish' and v <= entry_price:
                raise ValueError("Target price must be higher than entry price for bullish signals")
            elif direction.lower() == 'bearish' and v >= entry_price:
                raise ValueError("Target price must be lower than entry price for bearish signals")
        
        return v


class SignalUpdate(BaseModel):
    """Schema for updating a signal"""
    signal_name: Optional[str] = Field(None, max_length=255)
    confidence_score: Optional[float] = Field(None, ge=0, le=100)
    profit_potential_score: Optional[float] = Field(None, ge=0, le=100)
    target_price_cents: Optional[int] = Field(None, ge=0)
    stop_loss_price_cents: Optional[int] = Field(None, ge=0)
    expires_at: Optional[datetime] = None
    is_active: Optional[bool] = None
    market_context: Optional[Dict[str, Any]] = None


class SignalResponse(SignalBase):
    """Schema for signal response"""
    id: UUID
    created_at: datetime
    status: str
    is_active: bool
    
    class Config:
        from_attributes = True


class SignalGenerationRequest(BaseModel):
    """Request to generate signals for specific criteria"""
    asset_classes: Optional[List[str]] = Field(None, description="Asset classes to scan")
    symbols: Optional[List[str]] = Field(None, description="Specific symbols to analyze")
    risk_score_min: Optional[float] = Field(None, ge=0, le=100, description="Minimum risk score")
    risk_score_max: Optional[float] = Field(None, ge=0, le=100, description="Maximum risk score")
    min_confidence: Optional[float] = Field(0.6, ge=0, le=1, description="Minimum confidence threshold")
    max_signals: Optional[int] = Field(50, ge=1, le=500, description="Maximum signals to generate")
    signal_types: Optional[List[str]] = Field(None, description="Types of signals to generate")
    
    @validator('risk_score_max')
    def validate_risk_range(cls, v, values):
        min_risk = values.get('risk_score_min')
        if min_risk is not None and v is not None and v < min_risk:
            raise ValueError("Maximum risk score must be >= minimum risk score")
        return v


class SignalGenerationResponse(BaseModel):
    """Response from signal generation"""
    request_id: UUID
    signals_generated: int
    signals: List[SignalResponse]
    generation_time_ms: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RiskBreakdown(BaseModel):
    """Detailed risk score breakdown"""
    total_risk_score: float
    base_risk: float
    volatility_factor: float
    liquidity_factor: float
    time_factor: float
    market_conditions_factor: float
    position_size_factor: float
    confidence: float
    methodology: str


class SignalWithRisk(SignalResponse):
    """Signal response with detailed risk breakdown"""
    risk_breakdown: RiskBreakdown