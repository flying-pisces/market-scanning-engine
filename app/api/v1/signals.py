"""
Signal generation and management endpoints
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, status, Query, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_

from app.core.database import get_db
from app.core.exceptions import NotFoundError, ValidationError, SignalGenerationError
from app.core.config import get_settings
from app.models.database import Signal
from app.schemas.signal import (
    SignalCreate, SignalUpdate, SignalResponse, SignalGenerationRequest,
    SignalGenerationResponse, SignalWithRisk, RiskBreakdown
)
from app.services.risk_scoring import risk_scorer, AssetClass

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/signals", tags=["Signals"])


@router.post("/", response_model=SignalResponse, status_code=status.HTTP_201_CREATED)
async def create_signal(
    signal_data: SignalCreate,
    db: AsyncSession = Depends(get_db)
) -> SignalResponse:
    """Create a new trading signal"""
    
    try:
        signal = Signal(
            signal_name=signal_data.signal_name,
            signal_type=signal_data.signal_type,
            asset_class=signal_data.asset_class,
            symbol=signal_data.symbol.upper(),
            asset_name=signal_data.asset_name,
            risk_score=signal_data.risk_score,
            confidence_score=signal_data.confidence_score,
            profit_potential_score=signal_data.profit_potential_score,
            direction=signal_data.direction,
            entry_price_cents=signal_data.entry_price_cents,
            target_price_cents=signal_data.target_price_cents,
            stop_loss_price_cents=signal_data.stop_loss_price_cents,
            min_position_size_cents=signal_data.min_position_size_cents or 10000,
            max_position_size_cents=signal_data.max_position_size_cents or 100000,
            expires_at=signal_data.expires_at or (datetime.utcnow() + timedelta(hours=24)),
            generation_method=signal_data.generation_method or "manual",
            market_context=signal_data.market_context or {},
        )
        
        db.add(signal)
        await db.flush()
        await db.refresh(signal)
        
        logger.info(f"Created signal {signal.id} for {signal.symbol}")
        return SignalResponse.from_orm(signal)
        
    except Exception as e:
        logger.error(f"Failed to create signal: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create signal"
        )


@router.get("/", response_model=List[SignalResponse])
async def list_signals(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    asset_class: Optional[str] = Query(None),
    symbol: Optional[str] = Query(None),
    risk_score_min: Optional[float] = Query(None, ge=0, le=100),
    risk_score_max: Optional[float] = Query(None, ge=0, le=100),
    min_confidence: Optional[float] = Query(None, ge=0, le=100),
    active_only: bool = Query(True),
    db: AsyncSession = Depends(get_db)
) -> List[SignalResponse]:
    """List signals with filtering options"""
    
    try:
        query = select(Signal)
        
        # Apply filters
        filters = []
        
        if active_only:
            filters.append(Signal.is_active == True)
            filters.append(Signal.expires_at > datetime.utcnow())
        
        if asset_class:
            filters.append(Signal.asset_class == asset_class)
        
        if symbol:
            filters.append(Signal.symbol == symbol.upper())
        
        if risk_score_min is not None:
            filters.append(Signal.risk_score >= risk_score_min)
        
        if risk_score_max is not None:
            filters.append(Signal.risk_score <= risk_score_max)
        
        if min_confidence is not None:
            filters.append(Signal.confidence_score >= min_confidence)
        
        if filters:
            query = query.where(and_(*filters))
        
        # Add pagination and ordering
        query = query.order_by(Signal.created_at.desc()).offset(skip).limit(limit)
        
        result = await db.execute(query)
        signals = result.scalars().all()
        
        logger.info(f"Retrieved {len(signals)} signals")
        return [SignalResponse.from_orm(signal) for signal in signals]
        
    except Exception as e:
        logger.error(f"Failed to list signals: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve signals"
        )


@router.get("/{signal_id}", response_model=SignalResponse)
async def get_signal(
    signal_id: UUID,
    db: AsyncSession = Depends(get_db)
) -> SignalResponse:
    """Get signal by ID"""
    
    try:
        result = await db.execute(
            select(Signal).where(Signal.id == signal_id)
        )
        signal = result.scalar_one_or_none()
        
        if not signal:
            raise NotFoundError("Signal", str(signal_id))
        
        return SignalResponse.from_orm(signal)
        
    except NotFoundError:
        raise
    except Exception as e:
        logger.error(f"Failed to get signal {signal_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve signal"
        )


@router.put("/{signal_id}", response_model=SignalResponse)
async def update_signal(
    signal_id: UUID,
    signal_data: SignalUpdate,
    db: AsyncSession = Depends(get_db)
) -> SignalResponse:
    """Update signal"""
    
    try:
        result = await db.execute(
            select(Signal).where(Signal.id == signal_id)
        )
        signal = result.scalar_one_or_none()
        
        if not signal:
            raise NotFoundError("Signal", str(signal_id))
        
        # Update fields that were provided
        update_data = signal_data.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(signal, field, value)
        
        await db.flush()
        await db.refresh(signal)
        
        logger.info(f"Updated signal {signal_id}")
        return SignalResponse.from_orm(signal)
        
    except NotFoundError:
        raise
    except Exception as e:
        logger.error(f"Failed to update signal {signal_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update signal"
        )


@router.delete("/{signal_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_signal(
    signal_id: UUID,
    db: AsyncSession = Depends(get_db)
) -> None:
    """Deactivate signal"""
    
    try:
        result = await db.execute(
            select(Signal).where(Signal.id == signal_id)
        )
        signal = result.scalar_one_or_none()
        
        if not signal:
            raise NotFoundError("Signal", str(signal_id))
        
        signal.is_active = False
        await db.flush()
        
        logger.info(f"Deactivated signal {signal_id}")
        
    except NotFoundError:
        raise
    except Exception as e:
        logger.error(f"Failed to delete signal {signal_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete signal"
        )


@router.post("/generate", response_model=SignalGenerationResponse)
async def generate_signals(
    request: SignalGenerationRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
) -> SignalGenerationResponse:
    """Generate signals based on criteria"""
    
    start_time = datetime.utcnow()
    request_id = uuid4()
    
    try:
        logger.info(f"Starting signal generation {request_id}")
        
        # Default symbols for MVP
        default_symbols = {
            AssetClass.DAILY_OPTIONS: ["SPY", "QQQ", "SPX", "XSP", "NDX"],
            AssetClass.STOCKS: ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
            AssetClass.BONDS: ["TLT", "HYG", "LQD"],
            AssetClass.SAFE_ASSETS: ["SCHO", "BIL", "VTEB"],
        }
        
        generated_signals = []
        
        # Determine what to generate
        asset_classes_to_scan = request.asset_classes or [cls.value for cls in AssetClass]
        
        for asset_class_name in asset_classes_to_scan:
            try:
                asset_class = AssetClass(asset_class_name)
                symbols_to_scan = request.symbols or default_symbols.get(asset_class, [asset_class_name])
                
                for symbol in symbols_to_scan:
                    # Generate basic signal using our risk scorer
                    signal = await _generate_basic_signal(
                        asset_class, symbol, request, db
                    )
                    
                    if signal:
                        generated_signals.append(signal)
                        
                        # Stop if we've hit the max
                        if len(generated_signals) >= (request.max_signals or 50):
                            break
                
                if len(generated_signals) >= (request.max_signals or 50):
                    break
                    
            except ValueError:
                logger.warning(f"Invalid asset class: {asset_class_name}")
                continue
        
        # Calculate generation time
        end_time = datetime.utcnow()
        generation_time_ms = (end_time - start_time).total_seconds() * 1000
        
        logger.info(f"Generated {len(generated_signals)} signals in {generation_time_ms:.1f}ms")
        
        return SignalGenerationResponse(
            request_id=request_id,
            signals_generated=len(generated_signals),
            signals=[SignalResponse.from_orm(signal) for signal in generated_signals],
            generation_time_ms=generation_time_ms,
            metadata={
                "asset_classes_scanned": asset_classes_to_scan,
                "symbols_scanned": sum(len(default_symbols.get(AssetClass(ac), [ac])) 
                                     for ac in asset_classes_to_scan 
                                     if ac in [e.value for e in AssetClass]),
            }
        )
        
    except Exception as e:
        logger.error(f"Signal generation failed: {e}")
        raise SignalGenerationError(f"Failed to generate signals: {str(e)}")


@router.get("/{signal_id}/risk-breakdown", response_model=RiskBreakdown)
async def get_signal_risk_breakdown(
    signal_id: UUID,
    db: AsyncSession = Depends(get_db)
) -> RiskBreakdown:
    """Get detailed risk breakdown for a signal"""
    
    try:
        result = await db.execute(
            select(Signal).where(Signal.id == signal_id)
        )
        signal = result.scalar_one_or_none()
        
        if not signal:
            raise NotFoundError("Signal", str(signal_id))
        
        # Recalculate risk breakdown using current scorer
        asset_class = AssetClass(signal.asset_class)
        risk_data = risk_scorer.calculate_basic_risk_score(
            asset_class=asset_class,
            symbol=signal.symbol,
            market_data=signal.market_context,
        )
        
        return RiskBreakdown(
            total_risk_score=risk_data["risk_score"],
            base_risk=risk_data["base_risk"],
            volatility_factor=risk_data["risk_factors"]["volatility"],
            liquidity_factor=risk_data["risk_factors"]["liquidity"],
            time_factor=risk_data["risk_factors"]["time_horizon"],
            market_conditions_factor=risk_data["risk_factors"]["market_conditions"],
            position_size_factor=risk_data["risk_factors"]["position_size"],
            confidence=risk_data["confidence"],
            methodology=risk_data["methodology"]
        )
        
    except NotFoundError:
        raise
    except Exception as e:
        logger.error(f"Failed to get risk breakdown for signal {signal_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to calculate risk breakdown"
        )


async def _generate_basic_signal(
    asset_class: AssetClass,
    symbol: str,
    request: SignalGenerationRequest,
    db: AsyncSession
) -> Optional[Signal]:
    """Generate a basic signal for testing (MVP implementation)"""
    
    try:
        # Calculate risk score
        risk_data = risk_scorer.calculate_basic_risk_score(
            asset_class=asset_class,
            symbol=symbol,
            market_data=None,  # No real market data for MVP
        )
        
        risk_score = risk_data["risk_score"]
        
        # Check if risk score meets criteria
        if request.risk_score_min is not None and risk_score < request.risk_score_min:
            return None
        if request.risk_score_max is not None and risk_score > request.risk_score_max:
            return None
        
        # Mock confidence score for MVP
        confidence_score = risk_data["confidence"] * 100
        
        if confidence_score < (request.min_confidence or 0.6) * 100:
            return None
        
        # Generate mock signal data
        signal_types = ["technical_momentum", "options_flow", "value_opportunity"]
        signal_type = signal_types[hash(symbol) % len(signal_types)]
        
        # Create mock prices (simplified)
        base_price = _get_mock_price(symbol)
        direction = "bullish" if hash(symbol + str(datetime.utcnow().day)) % 2 else "bearish"
        
        if direction == "bullish":
            target_price = int(base_price * 1.05)  # 5% upside
            stop_loss = int(base_price * 0.98)     # 2% stop loss
        else:
            target_price = int(base_price * 0.95)  # 5% downside
            stop_loss = int(base_price * 1.02)     # 2% stop loss
        
        # Create signal
        signal = Signal(
            signal_name=f"{symbol} {signal_type} signal",
            signal_type=signal_type,
            asset_class=asset_class.value,
            symbol=symbol,
            asset_name=f"Mock {symbol} Asset",
            risk_score=risk_score,
            confidence_score=confidence_score,
            profit_potential_score=min(100, max(10, 100 - risk_score + 20)),  # Inverse relationship with some randomness
            direction=direction,
            entry_price_cents=base_price,
            target_price_cents=target_price,
            stop_loss_price_cents=stop_loss,
            min_position_size_cents=10000,
            max_position_size_cents=50000,
            expires_at=datetime.utcnow() + timedelta(hours=24),
            generation_method="mvp_basic_generator",
            market_context=risk_data,
        )
        
        db.add(signal)
        await db.flush()
        await db.refresh(signal)
        
        return signal
        
    except Exception as e:
        logger.error(f"Failed to generate signal for {symbol}: {e}")
        return None


def _get_mock_price(symbol: str) -> int:
    """Get mock price in cents for a symbol"""
    mock_prices = {
        "SPY": 45000,    # $450.00
        "QQQ": 38000,    # $380.00
        "SPX": 45000,    # $450.00 (actually an index)
        "XSP": 450,      # $4.50 (mini)
        "NDX": 15000,    # $150.00 (scaled)
        "AAPL": 17500,   # $175.00
        "MSFT": 40000,   # $400.00
        "GOOGL": 14000,  # $140.00
        "AMZN": 15000,   # $150.00
        "TSLA": 25000,   # $250.00
        "TLT": 9500,     # $95.00
        "HYG": 8200,     # $82.00
        "LQD": 11500,    # $115.00
        "SCHO": 5050,    # $50.50
        "BIL": 9150,     # $91.50
        "VTEB": 5250,    # $52.50
    }
    
    return mock_prices.get(symbol, 10000)  # Default $100.00