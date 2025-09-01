"""
Base Signal Generator - Abstract Interface for All Signal Generators
Author: Claude Code (System Architect)
Version: 1.0

Defines the base interface and common functionality for all signal generators.
All concrete signal generators must inherit from BaseSignalGenerator.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Union, Tuple
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator
from decimal import Decimal

from data_models.python.core_models import Asset, MarketData, TechnicalIndicators, OptionsData
from data_models.python.signal_models import (
    Signal, SignalFactor, SignalCategory, SignalDirection, 
    SignalStrength, SignalStatus, SignalBacktest
)


class GeneratorConfig(BaseModel):
    """Base configuration for signal generators"""
    generator_name: str = Field(..., description="Unique generator name")
    category: SignalCategory = Field(..., description="Signal category")
    enabled: bool = Field(default=True, description="Whether generator is enabled")
    
    # Risk and performance parameters
    base_risk_score: int = Field(..., ge=0, le=100, description="Base risk score for this generator")
    min_confidence_threshold: float = Field(default=0.5, ge=0, le=1, description="Minimum confidence to generate signal")
    max_signals_per_hour: int = Field(default=100, ge=1, description="Rate limiting per hour")
    
    # Timing parameters
    signal_ttl_hours: int = Field(default=24, ge=1, description="Signal time-to-live in hours")
    lookback_periods: int = Field(default=20, ge=1, description="Number of periods for calculations")
    
    # Asset filtering
    supported_asset_classes: List[str] = Field(default_factory=list, description="Supported asset classes")
    min_market_cap_cents: Optional[int] = Field(None, ge=0, description="Minimum market cap filter")
    min_avg_volume: Optional[int] = Field(None, ge=0, description="Minimum average volume filter")
    
    # Strategy-specific parameters
    strategy_params: Dict[str, Any] = Field(default_factory=dict, description="Strategy-specific parameters")


class SignalGenerationResult(BaseModel):
    """Result container for signal generation process"""
    signals: List[Signal] = Field(default_factory=list, description="Generated signals")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Generation metadata")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered")
    execution_time_ms: float = Field(..., ge=0, description="Execution time in milliseconds")
    assets_processed: int = Field(..., ge=0, description="Number of assets processed")


class BaseSignalGenerator(ABC):
    """
    Abstract base class for all signal generators.
    
    Provides common functionality for:
    - Configuration management
    - Asset filtering
    - Risk scoring
    - Performance tracking
    - Error handling
    """
    
    def __init__(self, config: GeneratorConfig):
        self.config = config
        self.logger = logging.getLogger(f"signal_generator.{config.generator_name}")
        self._last_run: Optional[datetime] = None
        self._signal_count_hour: int = 0
        self._hour_start: datetime = datetime.utcnow()
        
        # Performance metrics
        self._total_signals_generated: int = 0
        self._total_execution_time_ms: float = 0.0
        self._error_count: int = 0
        
    @property
    def name(self) -> str:
        """Generator name"""
        return self.config.generator_name
    
    @property
    def category(self) -> SignalCategory:
        """Signal category"""
        return self.config.category
    
    @property
    def is_enabled(self) -> bool:
        """Whether generator is enabled"""
        return self.config.enabled
    
    def enable(self) -> None:
        """Enable the generator"""
        self.config.enabled = True
        self.logger.info(f"Generator {self.name} enabled")
    
    def disable(self) -> None:
        """Disable the generator"""
        self.config.enabled = False
        self.logger.info(f"Generator {self.name} disabled")
    
    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits"""
        now = datetime.utcnow()
        
        # Reset hourly counter if needed
        if (now - self._hour_start).total_seconds() >= 3600:
            self._signal_count_hour = 0
            self._hour_start = now
        
        return self._signal_count_hour < self.config.max_signals_per_hour
    
    def _filter_assets(self, assets: List[Asset]) -> List[Asset]:
        """Filter assets based on generator configuration"""
        filtered = []
        
        for asset in assets:
            # Check asset class support
            if (self.config.supported_asset_classes and 
                asset.category not in self.config.supported_asset_classes):
                continue
            
            # Check market cap filter
            if (self.config.min_market_cap_cents and 
                asset.market_cap and 
                asset.market_cap < self.config.min_market_cap_cents):
                continue
            
            # Check volume filter
            if (self.config.min_avg_volume and 
                asset.avg_volume_30d and 
                asset.avg_volume_30d < self.config.min_avg_volume):
                continue
            
            # Check if asset is active
            if not asset.is_active:
                continue
            
            filtered.append(asset)
        
        return filtered
    
    def _calculate_base_risk_score(self, asset: Asset, market_data: Optional[MarketData] = None) -> int:
        """
        Calculate base risk score for an asset based on asset class and market conditions.
        
        Risk Score Ranges by Asset Class:
        - Daily Options: 70-95 (high risk, high reward)
        - Stocks: 30-80 (moderate to high risk)
        - ETFs: 20-70 (low to moderate risk)  
        - Bonds: 5-40 (low risk)
        - Safe Assets: 0-20 (very low risk)
        """
        base_score = self.config.base_risk_score
        
        # Adjust based on asset characteristics
        if asset.market_cap:
            # Smaller market cap = higher risk
            if asset.market_cap < 100_000_000_000:  # < $1B
                base_score = min(100, base_score + 15)
            elif asset.market_cap < 1_000_000_000_000:  # < $10B
                base_score = min(100, base_score + 5)
        
        # Adjust based on volume (liquidity risk)
        if asset.avg_volume_30d:
            if asset.avg_volume_30d < 100_000:  # Low volume
                base_score = min(100, base_score + 10)
        
        # Adjust based on current volatility if market data available
        if market_data and hasattr(market_data, 'volume'):
            # High volume relative to average suggests higher volatility
            if asset.avg_volume_30d and market_data.volume > asset.avg_volume_30d * 2:
                base_score = min(100, base_score + 5)
        
        return max(0, min(100, base_score))
    
    def _create_signal_factor(self, 
                            signal_id: UUID,
                            factor_name: str,
                            contribution_score: int,
                            weight: float,
                            factor_value: Optional[Decimal] = None,
                            factor_percentile: Optional[int] = None,
                            calculation_method: Optional[str] = None) -> SignalFactor:
        """Create a signal factor with proper validation"""
        return SignalFactor(
            signal_id=signal_id,
            factor_name=factor_name,
            factor_category=self.category.value,
            contribution_score=max(0, min(100, contribution_score)),
            weight=max(0.0, min(1.0, weight)),
            factor_value=factor_value,
            factor_percentile=max(0, min(100, factor_percentile)) if factor_percentile else None,
            calculation_method=calculation_method or self.name,
            data_source=self.name,
            lookback_periods=self.config.lookback_periods,
            created_at=datetime.utcnow()
        )
    
    @abstractmethod
    async def generate_signals(self, 
                              assets: List[Asset],
                              market_data: Dict[UUID, MarketData],
                              technical_indicators: Dict[UUID, TechnicalIndicators],
                              options_data: Dict[UUID, List[OptionsData]] = None,
                              additional_data: Dict[str, Any] = None) -> SignalGenerationResult:
        """
        Generate signals for given assets.
        
        Args:
            assets: List of assets to analyze
            market_data: Market data by asset_id
            technical_indicators: Technical indicators by asset_id
            options_data: Options data by asset_id (if applicable)
            additional_data: Additional strategy-specific data
            
        Returns:
            SignalGenerationResult containing generated signals and metadata
        """
        pass
    
    @abstractmethod
    def validate_configuration(self) -> Tuple[bool, List[str]]:
        """
        Validate generator configuration.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        pass
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get generator performance metrics"""
        avg_execution_time = (self._total_execution_time_ms / max(1, self._total_signals_generated))
        
        return {
            "generator_name": self.name,
            "total_signals_generated": self._total_signals_generated,
            "average_execution_time_ms": avg_execution_time,
            "total_execution_time_ms": self._total_execution_time_ms,
            "error_count": self._error_count,
            "last_run": self._last_run.isoformat() if self._last_run else None,
            "signals_this_hour": self._signal_count_hour,
            "rate_limit": self.config.max_signals_per_hour
        }
    
    def reset_performance_metrics(self) -> None:
        """Reset performance counters"""
        self._total_signals_generated = 0
        self._total_execution_time_ms = 0.0
        self._error_count = 0
        self._signal_count_hour = 0
        self._hour_start = datetime.utcnow()
        self.logger.info(f"Performance metrics reset for {self.name}")
    
    async def run_generation(self, 
                           assets: List[Asset],
                           market_data: Dict[UUID, MarketData],
                           technical_indicators: Dict[UUID, TechnicalIndicators],
                           options_data: Dict[UUID, List[OptionsData]] = None,
                           additional_data: Dict[str, Any] = None) -> SignalGenerationResult:
        """
        Main entry point for signal generation with error handling and metrics.
        """
        if not self.is_enabled:
            return SignalGenerationResult(
                signals=[],
                metadata={"generator": self.name, "status": "disabled"},
                errors=["Generator is disabled"],
                execution_time_ms=0.0,
                assets_processed=0
            )
        
        if not self._check_rate_limit():
            return SignalGenerationResult(
                signals=[],
                metadata={"generator": self.name, "status": "rate_limited"},
                errors=["Rate limit exceeded"],
                execution_time_ms=0.0,
                assets_processed=0
            )
        
        start_time = datetime.utcnow()
        filtered_assets = self._filter_assets(assets)
        
        try:
            result = await self.generate_signals(
                filtered_assets, 
                market_data, 
                technical_indicators, 
                options_data, 
                additional_data
            )
            
            # Update metrics
            execution_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            result.execution_time_ms = execution_time_ms
            result.assets_processed = len(filtered_assets)
            
            self._total_signals_generated += len(result.signals)
            self._total_execution_time_ms += execution_time_ms
            self._signal_count_hour += len(result.signals)
            self._last_run = start_time
            
            # Add metadata
            result.metadata.update({
                "generator": self.name,
                "category": self.category.value,
                "assets_filtered": len(assets) - len(filtered_assets),
                "timestamp": start_time.isoformat()
            })
            
            self.logger.info(
                f"Generated {len(result.signals)} signals for {len(filtered_assets)} assets "
                f"in {execution_time_ms:.2f}ms"
            )
            
            return result
            
        except Exception as e:
            self._error_count += 1
            error_msg = f"Error in {self.name}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            execution_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return SignalGenerationResult(
                signals=[],
                metadata={
                    "generator": self.name,
                    "status": "error",
                    "timestamp": start_time.isoformat()
                },
                errors=[error_msg],
                execution_time_ms=execution_time_ms,
                assets_processed=len(filtered_assets)
            )
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.name}, category={self.category.value})>"