"""
Technical Analysis Signal Generators
Author: Claude Code (System Architect)  
Version: 1.0

Implementation of various technical analysis based signal generators including:
- Moving Average Crossovers
- RSI Momentum Signals
- MACD Signals  
- Bollinger Bands Mean Reversion
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
from decimal import Decimal
from uuid import UUID, uuid4

from signal_generation.core.base_generator import BaseSignalGenerator, GeneratorConfig, SignalGenerationResult
from data_models.python.core_models import Asset, MarketData, TechnicalIndicators
from data_models.python.signal_models import (
    Signal, SignalFactor, SignalDirection, SignalStrength, 
    SignalStatus, SignalCategory
)


class MovingAverageSignalGenerator(BaseSignalGenerator):
    """
    Moving Average Crossover Signal Generator
    
    Generates signals based on moving average crossovers and trend analysis.
    Supports multiple MA types and timeframes with configurable parameters.
    """
    
    def __init__(self, config: GeneratorConfig):
        super().__init__(config)
        
        # Default strategy parameters
        self.short_ma_period = config.strategy_params.get("short_ma_period", 20)
        self.long_ma_period = config.strategy_params.get("long_ma_period", 50) 
        self.ma_type = config.strategy_params.get("ma_type", "sma")  # sma, ema
        self.trend_confirmation_period = config.strategy_params.get("trend_confirmation_period", 3)
        self.volume_confirmation = config.strategy_params.get("volume_confirmation", True)
    
    async def generate_signals(self,
                              assets: List[Asset],
                              market_data: Dict[UUID, MarketData],
                              technical_indicators: Dict[UUID, TechnicalIndicators],
                              options_data: Dict[UUID, List] = None,
                              additional_data: Dict[str, Any] = None) -> SignalGenerationResult:
        
        signals = []
        
        for asset in assets:
            if asset.id not in technical_indicators or asset.id not in market_data:
                continue
                
            tech_data = technical_indicators[asset.id]
            mkt_data = market_data[asset.id]
            
            signal = await self._analyze_moving_averages(asset, tech_data, mkt_data)
            if signal:
                signals.append(signal)
        
        return SignalGenerationResult(
            signals=signals,
            metadata={
                "strategy": "moving_average_crossover",
                "parameters": {
                    "short_ma": self.short_ma_period,
                    "long_ma": self.long_ma_period,
                    "ma_type": self.ma_type
                }
            },
            errors=[],
            execution_time_ms=0.0,  # Will be set by base class
            assets_processed=len(assets)
        )
    
    async def _analyze_moving_averages(self,
                                     asset: Asset,
                                     tech_data: TechnicalIndicators,
                                     mkt_data: MarketData) -> Optional[Signal]:
        
        # Get moving averages based on type
        if self.ma_type == "sma":
            short_ma = tech_data.sma_20 if self.short_ma_period == 20 else tech_data.sma_50
            long_ma = tech_data.sma_50 if self.long_ma_period == 50 else tech_data.sma_200
        else:  # ema
            short_ma = tech_data.ema_12 if self.short_ma_period <= 12 else tech_data.ema_26
            long_ma = tech_data.ema_26 if self.long_ma_period <= 26 else tech_data.sma_50
        
        if not short_ma or not long_ma:
            return None
        
        current_price = mkt_data.close_price_cents
        
        # Determine signal direction
        direction = None
        signal_strength_score = 0
        
        # Bullish crossover: short MA > long MA and price > short MA
        if short_ma > long_ma and current_price > short_ma:
            direction = SignalDirection.BUY
            # Strength based on separation between MAs and price position
            ma_separation_pct = ((short_ma - long_ma) / long_ma) * 100
            price_above_ma_pct = ((current_price - short_ma) / short_ma) * 100
            signal_strength_score = min(100, int(ma_separation_pct * 20 + price_above_ma_pct * 10))
            
        # Bearish crossover: short MA < long MA and price < short MA  
        elif short_ma < long_ma and current_price < short_ma:
            direction = SignalDirection.SELL
            ma_separation_pct = ((long_ma - short_ma) / long_ma) * 100
            price_below_ma_pct = ((short_ma - current_price) / short_ma) * 100
            signal_strength_score = min(100, int(ma_separation_pct * 20 + price_below_ma_pct * 10))
        
        if not direction:
            return None
        
        # Volume confirmation
        volume_factor_score = 50  # Default neutral score
        if self.volume_confirmation and asset.avg_volume_30d:
            volume_ratio = mkt_data.volume / asset.avg_volume_30d
            if volume_ratio > 1.5:  # Above average volume
                volume_factor_score = min(100, int(volume_ratio * 40))
            elif volume_ratio < 0.5:  # Below average volume
                volume_factor_score = max(20, int(volume_ratio * 60))
        
        # Calculate confidence score
        confidence_score = int((signal_strength_score + volume_factor_score) / 2)
        
        if confidence_score < self.config.min_confidence_threshold * 100:
            return None
        
        # Calculate risk score  
        base_risk = self._calculate_base_risk_score(asset, mkt_data)
        
        # Adjust risk based on signal strength (stronger signals = slightly lower risk)
        risk_adjustment = -min(10, signal_strength_score // 10)
        final_risk_score = max(0, min(100, base_risk + risk_adjustment))
        
        # Create signal
        signal_id = uuid4()
        signal = Signal(
            id=signal_id,
            asset_id=asset.id,
            signal_type_id=1,  # Technical signal type
            signal_name=f"{self.ma_type.upper()} MA Crossover ({self.short_ma_period}/{self.long_ma_period})",
            direction=direction,
            risk_score=final_risk_score,
            profit_potential_score=signal_strength_score,
            confidence_score=confidence_score,
            entry_price_cents=current_price,
            signal_source=self.name,
            methodology_version="1.0",
            signal_strength=self._determine_signal_strength(confidence_score),
            status=SignalStatus.ACTIVE,
            valid_until=datetime.utcnow() + timedelta(hours=self.config.signal_ttl_hours),
            recommended_holding_period_hours=24,
            asset_specific_data={
                "short_ma_value": short_ma,
                "long_ma_value": long_ma, 
                "ma_separation_pct": ((short_ma - long_ma) / long_ma) * 100,
                "volume_ratio": mkt_data.volume / asset.avg_volume_30d if asset.avg_volume_30d else None
            }
        )
        
        # Add signal factors
        factors = [
            self._create_signal_factor(
                signal_id, "MA Crossover", signal_strength_score, 0.6,
                factor_value=Decimal(str(abs(short_ma - long_ma))),
                calculation_method=f"{self.ma_type.upper()} {self.short_ma_period}/{self.long_ma_period}"
            ),
            self._create_signal_factor(
                signal_id, "Volume Confirmation", volume_factor_score, 0.4,
                factor_value=Decimal(str(mkt_data.volume)),
                factor_percentile=min(100, int(volume_ratio * 50)) if asset.avg_volume_30d else None
            )
        ]
        
        signal.factors = factors
        
        return signal
    
    def _determine_signal_strength(self, confidence_score: int) -> SignalStrength:
        """Determine signal strength based on confidence score"""
        if confidence_score >= 85:
            return SignalStrength.VERY_STRONG
        elif confidence_score >= 70:
            return SignalStrength.STRONG  
        elif confidence_score >= 55:
            return SignalStrength.MODERATE
        else:
            return SignalStrength.WEAK
    
    def validate_configuration(self) -> Tuple[bool, List[str]]:
        """Validate generator configuration"""
        errors = []
        
        if self.short_ma_period <= 0:
            errors.append("short_ma_period must be positive")
        
        if self.long_ma_period <= 0:
            errors.append("long_ma_period must be positive")
            
        if self.short_ma_period >= self.long_ma_period:
            errors.append("short_ma_period must be less than long_ma_period")
        
        if self.ma_type not in ["sma", "ema"]:
            errors.append("ma_type must be 'sma' or 'ema'")
            
        return len(errors) == 0, errors


class RSISignalGenerator(BaseSignalGenerator):
    """
    RSI Momentum Signal Generator
    
    Generates signals based on RSI momentum oscillator with support for:
    - Overbought/oversold conditions
    - RSI divergence detection  
    - Multiple timeframe analysis
    """
    
    def __init__(self, config: GeneratorConfig):
        super().__init__(config)
        
        self.rsi_period = config.strategy_params.get("rsi_period", 14)
        self.oversold_threshold = config.strategy_params.get("oversold_threshold", 30)
        self.overbought_threshold = config.strategy_params.get("overbought_threshold", 70)
        self.extreme_oversold = config.strategy_params.get("extreme_oversold", 20)
        self.extreme_overbought = config.strategy_params.get("extreme_overbought", 80)
        self.divergence_detection = config.strategy_params.get("divergence_detection", False)
    
    async def generate_signals(self,
                              assets: List[Asset],
                              market_data: Dict[UUID, MarketData],
                              technical_indicators: Dict[UUID, TechnicalIndicators],
                              options_data: Dict[UUID, List] = None,
                              additional_data: Dict[str, Any] = None) -> SignalGenerationResult:
        
        signals = []
        
        for asset in assets:
            if asset.id not in technical_indicators or asset.id not in market_data:
                continue
                
            tech_data = technical_indicators[asset.id] 
            mkt_data = market_data[asset.id]
            
            signal = await self._analyze_rsi(asset, tech_data, mkt_data)
            if signal:
                signals.append(signal)
        
        return SignalGenerationResult(
            signals=signals,
            metadata={
                "strategy": "rsi_momentum",
                "parameters": {
                    "rsi_period": self.rsi_period,
                    "oversold_threshold": self.oversold_threshold,
                    "overbought_threshold": self.overbought_threshold
                }
            },
            errors=[],
            execution_time_ms=0.0,
            assets_processed=len(assets)
        )
    
    async def _analyze_rsi(self,
                          asset: Asset,
                          tech_data: TechnicalIndicators,
                          mkt_data: MarketData) -> Optional[Signal]:
        
        rsi = tech_data.rsi_14
        if not rsi:
            return None
        
        rsi_value = float(rsi)
        current_price = mkt_data.close_price_cents
        
        direction = None
        signal_strength_score = 0
        rsi_factor_score = 0
        
        # Oversold condition - potential buy signal
        if rsi_value <= self.oversold_threshold:
            direction = SignalDirection.BUY
            
            if rsi_value <= self.extreme_oversold:
                signal_strength_score = 90
                rsi_factor_score = 95
            else:
                # Linear scale between oversold and extreme oversold
                intensity = (self.oversold_threshold - rsi_value) / (self.oversold_threshold - self.extreme_oversold)
                signal_strength_score = int(60 + intensity * 30)
                rsi_factor_score = int(70 + intensity * 25)
        
        # Overbought condition - potential sell signal
        elif rsi_value >= self.overbought_threshold:
            direction = SignalDirection.SELL
            
            if rsi_value >= self.extreme_overbought:
                signal_strength_score = 90
                rsi_factor_score = 95
            else:
                # Linear scale between overbought and extreme overbought  
                intensity = (rsi_value - self.overbought_threshold) / (self.extreme_overbought - self.overbought_threshold)
                signal_strength_score = int(60 + intensity * 30)
                rsi_factor_score = int(70 + intensity * 25)
        
        if not direction:
            return None
        
        # Additional confirmation factors
        trend_factor_score = 50  # Default neutral
        
        # Check if price is in alignment with RSI signal
        if tech_data.sma_20:
            price_vs_sma = ((current_price - tech_data.sma_20) / tech_data.sma_20) * 100
            
            if direction == SignalDirection.BUY and price_vs_sma < 0:
                # Price below SMA supports oversold buy signal
                trend_factor_score = min(80, 60 + abs(price_vs_sma))
            elif direction == SignalDirection.SELL and price_vs_sma > 0:
                # Price above SMA supports overbought sell signal
                trend_factor_score = min(80, 60 + price_vs_sma)
        
        # Calculate overall confidence
        confidence_score = int((rsi_factor_score * 0.7 + trend_factor_score * 0.3))
        
        if confidence_score < self.config.min_confidence_threshold * 100:
            return None
        
        # Risk scoring - RSI signals can be high risk due to potential false signals
        base_risk = self._calculate_base_risk_score(asset, mkt_data)
        
        # Extreme RSI values suggest higher volatility = higher risk
        if rsi_value <= self.extreme_oversold or rsi_value >= self.extreme_overbought:
            risk_adjustment = 10
        else:
            risk_adjustment = 5
        
        final_risk_score = min(100, base_risk + risk_adjustment)
        
        # Set price targets based on historical RSI reversal patterns
        target_price = None
        stop_loss_price = None
        
        if direction == SignalDirection.BUY:
            # Target: price at RSI 50-60 level (neutral to slightly bullish)
            target_price = int(current_price * 1.05)  # Conservative 5% target
            stop_loss_price = int(current_price * 0.95)  # 5% stop loss
        else:  # SELL
            target_price = int(current_price * 0.95)  # Conservative 5% target  
            stop_loss_price = int(current_price * 1.05)  # 5% stop loss
        
        # Create signal
        signal_id = uuid4()
        signal = Signal(
            id=signal_id,
            asset_id=asset.id,
            signal_type_id=1,  # Technical signal type
            signal_name=f"RSI {direction.value} ({rsi_value:.1f})",
            direction=direction,
            risk_score=final_risk_score,
            profit_potential_score=signal_strength_score,
            confidence_score=confidence_score,
            entry_price_cents=current_price,
            target_price_cents=target_price,
            stop_loss_price_cents=stop_loss_price,
            signal_source=self.name,
            methodology_version="1.0",
            signal_strength=self._determine_signal_strength(confidence_score),
            status=SignalStatus.ACTIVE,
            valid_until=datetime.utcnow() + timedelta(hours=self.config.signal_ttl_hours),
            recommended_holding_period_hours=48,  # RSI signals may take time to develop
            asset_specific_data={
                "rsi_value": rsi_value,
                "rsi_condition": "extreme_oversold" if rsi_value <= self.extreme_oversold else
                               "oversold" if rsi_value <= self.oversold_threshold else
                               "extreme_overbought" if rsi_value >= self.extreme_overbought else
                               "overbought",
                "price_vs_sma_pct": price_vs_sma if tech_data.sma_20 else None
            }
        )
        
        # Add signal factors
        factors = [
            self._create_signal_factor(
                signal_id, "RSI Level", rsi_factor_score, 0.7,
                factor_value=Decimal(str(rsi_value)),
                factor_percentile=int(rsi_value),
                calculation_method=f"RSI-{self.rsi_period}"
            ),
            self._create_signal_factor(
                signal_id, "Trend Alignment", trend_factor_score, 0.3,
                factor_value=Decimal(str(price_vs_sma)) if tech_data.sma_20 else None,
                calculation_method="Price vs SMA-20"
            )
        ]
        
        signal.factors = factors
        
        return signal
    
    def _determine_signal_strength(self, confidence_score: int) -> SignalStrength:
        """Determine signal strength based on confidence score"""
        if confidence_score >= 85:
            return SignalStrength.VERY_STRONG
        elif confidence_score >= 70:
            return SignalStrength.STRONG
        elif confidence_score >= 55:
            return SignalStrength.MODERATE
        else:
            return SignalStrength.WEAK
    
    def validate_configuration(self) -> Tuple[bool, List[str]]:
        """Validate generator configuration"""
        errors = []
        
        if self.rsi_period <= 0:
            errors.append("rsi_period must be positive")
        
        if not 0 <= self.oversold_threshold <= 50:
            errors.append("oversold_threshold must be between 0 and 50")
            
        if not 50 <= self.overbought_threshold <= 100:
            errors.append("overbought_threshold must be between 50 and 100")
        
        if self.extreme_oversold >= self.oversold_threshold:
            errors.append("extreme_oversold must be less than oversold_threshold")
            
        if self.extreme_overbought <= self.overbought_threshold:
            errors.append("extreme_overbought must be greater than overbought_threshold")
            
        return len(errors) == 0, errors


class MACDSignalGenerator(BaseSignalGenerator):
    """
    MACD Signal Generator
    
    Generates signals based on MACD (Moving Average Convergence Divergence) including:
    - MACD line and signal line crossovers
    - Zero line crossovers  
    - Histogram divergence analysis
    """
    
    def __init__(self, config: GeneratorConfig):
        super().__init__(config)
        
        self.fast_period = config.strategy_params.get("fast_period", 12)
        self.slow_period = config.strategy_params.get("slow_period", 26)
        self.signal_period = config.strategy_params.get("signal_period", 9)
        self.histogram_threshold = config.strategy_params.get("histogram_threshold", 0)
        self.zero_line_crossover = config.strategy_params.get("zero_line_crossover", True)
    
    async def generate_signals(self,
                              assets: List[Asset],
                              market_data: Dict[UUID, MarketData],
                              technical_indicators: Dict[UUID, TechnicalIndicators],
                              options_data: Dict[UUID, List] = None,
                              additional_data: Dict[str, Any] = None) -> SignalGenerationResult:
        
        signals = []
        
        for asset in assets:
            if asset.id not in technical_indicators or asset.id not in market_data:
                continue
                
            tech_data = technical_indicators[asset.id]
            mkt_data = market_data[asset.id]
            
            signal = await self._analyze_macd(asset, tech_data, mkt_data)
            if signal:
                signals.append(signal)
        
        return SignalGenerationResult(
            signals=signals,
            metadata={
                "strategy": "macd_crossover",
                "parameters": {
                    "fast_period": self.fast_period,
                    "slow_period": self.slow_period,
                    "signal_period": self.signal_period
                }
            },
            errors=[],
            execution_time_ms=0.0,
            assets_processed=len(assets)
        )
    
    async def _analyze_macd(self,
                           asset: Asset,
                           tech_data: TechnicalIndicators,
                           mkt_data: MarketData) -> Optional[Signal]:
        
        macd_line = tech_data.macd_line
        macd_signal = tech_data.macd_signal  
        macd_histogram = tech_data.macd_histogram
        
        if not all([macd_line, macd_signal, macd_histogram]):
            return None
        
        current_price = mkt_data.close_price_cents
        
        direction = None
        signal_strength_score = 0
        signal_type = ""
        
        # MACD line above signal line = bullish
        if macd_line > macd_signal:
            direction = SignalDirection.BUY
            signal_type = "bullish_crossover"
            
            # Strength based on separation and histogram
            separation = abs(macd_line - macd_signal)
            histogram_strength = max(0, macd_histogram) if macd_histogram > 0 else 0
            
            signal_strength_score = min(100, int(separation / 100 * 50 + histogram_strength / 100 * 50))
            
        # MACD line below signal line = bearish  
        elif macd_line < macd_signal:
            direction = SignalDirection.SELL
            signal_type = "bearish_crossover"
            
            separation = abs(macd_line - macd_signal)
            histogram_strength = abs(min(0, macd_histogram)) if macd_histogram < 0 else 0
            
            signal_strength_score = min(100, int(separation / 100 * 50 + histogram_strength / 100 * 50))
        
        if not direction:
            return None
        
        # Zero line crossover confirmation
        zero_line_factor_score = 50  # Neutral default
        
        if self.zero_line_crossover:
            if direction == SignalDirection.BUY and macd_line > 0:
                # MACD above zero confirms bullish momentum
                zero_line_factor_score = min(80, 60 + abs(macd_line) / 100 * 20)
                signal_type += "_above_zero"
            elif direction == SignalDirection.SELL and macd_line < 0:
                # MACD below zero confirms bearish momentum
                zero_line_factor_score = min(80, 60 + abs(macd_line) / 100 * 20)
                signal_type += "_below_zero"
        
        # Trend confirmation using price vs moving averages
        trend_factor_score = 50
        if tech_data.ema_12 and tech_data.ema_26:
            if direction == SignalDirection.BUY and current_price > tech_data.ema_12:
                trend_factor_score = 70
            elif direction == SignalDirection.SELL and current_price < tech_data.ema_12:
                trend_factor_score = 70
        
        # Calculate confidence score
        confidence_score = int((signal_strength_score * 0.5 + zero_line_factor_score * 0.3 + trend_factor_score * 0.2))
        
        if confidence_score < self.config.min_confidence_threshold * 100:
            return None
        
        # Risk scoring - MACD signals can be early, so moderate risk
        base_risk = self._calculate_base_risk_score(asset, mkt_data)
        
        # Higher risk if signal is against zero line (counter-trend)
        if ((direction == SignalDirection.BUY and macd_line < 0) or 
            (direction == SignalDirection.SELL and macd_line > 0)):
            risk_adjustment = 10
        else:
            risk_adjustment = 0
            
        final_risk_score = min(100, base_risk + risk_adjustment)
        
        # Price targets based on MACD momentum
        target_price = None
        stop_loss_price = None
        
        momentum_strength = abs(macd_histogram) / max(abs(macd_line), 1)
        target_pct = min(0.10, 0.03 + momentum_strength * 0.02)  # 3-10% target based on momentum
        
        if direction == SignalDirection.BUY:
            target_price = int(current_price * (1 + target_pct))
            stop_loss_price = int(current_price * 0.96)  # 4% stop loss
        else:
            target_price = int(current_price * (1 - target_pct))
            stop_loss_price = int(current_price * 1.04)  # 4% stop loss
        
        # Create signal
        signal_id = uuid4()
        signal = Signal(
            id=signal_id,
            asset_id=asset.id,
            signal_type_id=1,  # Technical signal type
            signal_name=f"MACD {signal_type.replace('_', ' ').title()}",
            direction=direction,
            risk_score=final_risk_score,
            profit_potential_score=signal_strength_score,
            confidence_score=confidence_score,
            entry_price_cents=current_price,
            target_price_cents=target_price,
            stop_loss_price_cents=stop_loss_price,
            signal_source=self.name,
            methodology_version="1.0",
            signal_strength=self._determine_signal_strength(confidence_score),
            status=SignalStatus.ACTIVE,
            valid_until=datetime.utcnow() + timedelta(hours=self.config.signal_ttl_hours),
            recommended_holding_period_hours=72,  # MACD signals may take several days
            asset_specific_data={
                "macd_line": macd_line,
                "macd_signal": macd_signal,
                "macd_histogram": macd_histogram,
                "signal_type": signal_type,
                "zero_line_position": "above" if macd_line > 0 else "below",
                "momentum_strength": momentum_strength
            }
        )
        
        # Add signal factors
        factors = [
            self._create_signal_factor(
                signal_id, "MACD Crossover", signal_strength_score, 0.5,
                factor_value=Decimal(str(abs(macd_line - macd_signal))),
                calculation_method=f"MACD({self.fast_period},{self.slow_period},{self.signal_period})"
            ),
            self._create_signal_factor(
                signal_id, "Zero Line Position", zero_line_factor_score, 0.3,
                factor_value=Decimal(str(macd_line)),
                calculation_method="MACD vs Zero Line"
            ),
            self._create_signal_factor(
                signal_id, "Trend Confirmation", trend_factor_score, 0.2,
                calculation_method="Price vs EMA Alignment"
            )
        ]
        
        signal.factors = factors
        
        return signal
    
    def _determine_signal_strength(self, confidence_score: int) -> SignalStrength:
        """Determine signal strength based on confidence score"""
        if confidence_score >= 85:
            return SignalStrength.VERY_STRONG
        elif confidence_score >= 70:
            return SignalStrength.STRONG
        elif confidence_score >= 55:
            return SignalStrength.MODERATE
        else:
            return SignalStrength.WEAK
    
    def validate_configuration(self) -> Tuple[bool, List[str]]:
        """Validate generator configuration"""
        errors = []
        
        if self.fast_period <= 0:
            errors.append("fast_period must be positive")
            
        if self.slow_period <= 0:
            errors.append("slow_period must be positive")
            
        if self.signal_period <= 0:
            errors.append("signal_period must be positive")
            
        if self.fast_period >= self.slow_period:
            errors.append("fast_period must be less than slow_period")
            
        return len(errors) == 0, errors


class BollingerBandsSignalGenerator(BaseSignalGenerator):
    """
    Bollinger Bands Mean Reversion Signal Generator
    
    Generates signals based on Bollinger Bands analysis including:
    - Band squeeze and expansion
    - Price touching or crossing bands
    - Mean reversion opportunities
    """
    
    def __init__(self, config: GeneratorConfig):
        super().__init__(config)
        
        self.bb_period = config.strategy_params.get("bb_period", 20)
        self.bb_std_dev = config.strategy_params.get("bb_std_dev", 2.0)
        self.squeeze_threshold = config.strategy_params.get("squeeze_threshold", 0.1)  # Band width as % of middle
        self.band_touch_sensitivity = config.strategy_params.get("band_touch_sensitivity", 0.02)  # % from band to trigger
    
    async def generate_signals(self,
                              assets: List[Asset],
                              market_data: Dict[UUID, MarketData],
                              technical_indicators: Dict[UUID, TechnicalIndicators],
                              options_data: Dict[UUID, List] = None,
                              additional_data: Dict[str, Any] = None) -> SignalGenerationResult:
        
        signals = []
        
        for asset in assets:
            if asset.id not in technical_indicators or asset.id not in market_data:
                continue
                
            tech_data = technical_indicators[asset.id]
            mkt_data = market_data[asset.id]
            
            signal = await self._analyze_bollinger_bands(asset, tech_data, mkt_data)
            if signal:
                signals.append(signal)
        
        return SignalGenerationResult(
            signals=signals,
            metadata={
                "strategy": "bollinger_bands_mean_reversion",
                "parameters": {
                    "bb_period": self.bb_period,
                    "bb_std_dev": self.bb_std_dev,
                    "squeeze_threshold": self.squeeze_threshold
                }
            },
            errors=[],
            execution_time_ms=0.0,
            assets_processed=len(assets)
        )
    
    async def _analyze_bollinger_bands(self,
                                      asset: Asset,
                                      tech_data: TechnicalIndicators,
                                      mkt_data: MarketData) -> Optional[Signal]:
        
        bb_upper = tech_data.bollinger_upper
        bb_middle = tech_data.bollinger_middle 
        bb_lower = tech_data.bollinger_lower
        
        if not all([bb_upper, bb_middle, bb_lower]):
            return None
        
        current_price = mkt_data.close_price_cents
        
        # Calculate band position (0 = lower band, 0.5 = middle, 1 = upper band)
        if bb_upper == bb_lower:  # Avoid division by zero
            return None
            
        band_position = (current_price - bb_lower) / (bb_upper - bb_lower)
        
        # Calculate band width as percentage of middle band
        band_width_pct = ((bb_upper - bb_lower) / bb_middle) * 100
        
        direction = None
        signal_strength_score = 0
        signal_type = ""
        
        # Price near lower band - potential buy (mean reversion)
        if band_position <= 0.1:  # Within 10% of lower band
            direction = SignalDirection.BUY
            signal_type = "lower_band_bounce"
            
            # Stronger signal the closer to lower band
            signal_strength_score = int((0.1 - band_position) * 1000)  # 0-100 range
            signal_strength_score = max(50, min(100, signal_strength_score))
            
        # Price near upper band - potential sell (mean reversion)
        elif band_position >= 0.9:  # Within 10% of upper band
            direction = SignalDirection.SELL
            signal_type = "upper_band_rejection"
            
            # Stronger signal the closer to upper band  
            signal_strength_score = int((band_position - 0.9) * 1000)  # 0-100 range
            signal_strength_score = max(50, min(100, signal_strength_score))
        
        # Band squeeze breakout - momentum signal
        elif band_width_pct < self.squeeze_threshold:
            # Determine breakout direction based on price position
            if band_position > 0.6:
                direction = SignalDirection.BUY
                signal_type = "squeeze_breakout_bullish"
            elif band_position < 0.4:
                direction = SignalDirection.SELL
                signal_type = "squeeze_breakout_bearish"
            else:
                return None  # Too uncertain
                
            # Signal strength based on how tight the squeeze is
            squeeze_intensity = self.squeeze_threshold / max(band_width_pct, 0.01)
            signal_strength_score = min(100, int(squeeze_intensity * 30))
        
        if not direction:
            return None
        
        # Volume confirmation
        volume_factor_score = 50
        if asset.avg_volume_30d and mkt_data.volume:
            volume_ratio = mkt_data.volume / asset.avg_volume_30d
            
            if "breakout" in signal_type:
                # Breakouts should have higher volume
                if volume_ratio > 1.5:
                    volume_factor_score = min(90, int(volume_ratio * 40))
                else:
                    volume_factor_score = max(30, int(volume_ratio * 50))
            else:
                # Mean reversion signals are less volume dependent
                volume_factor_score = min(70, max(40, int(volume_ratio * 35 + 35)))
        
        # RSI confirmation for mean reversion signals
        rsi_factor_score = 50
        if tech_data.rsi_14 and ("bounce" in signal_type or "rejection" in signal_type):
            rsi_value = float(tech_data.rsi_14)
            
            if direction == SignalDirection.BUY and rsi_value < 40:
                # Oversold RSI supports lower band bounce
                rsi_factor_score = int(80 - rsi_value)  # Lower RSI = higher score
            elif direction == SignalDirection.SELL and rsi_value > 60:
                # Overbought RSI supports upper band rejection
                rsi_factor_score = int(rsi_value - 20)  # Higher RSI = higher score
        
        # Calculate confidence score
        if "breakout" in signal_type:
            confidence_score = int((signal_strength_score * 0.4 + volume_factor_score * 0.6))
        else:
            confidence_score = int((signal_strength_score * 0.5 + volume_factor_score * 0.2 + rsi_factor_score * 0.3))
        
        if confidence_score < self.config.min_confidence_threshold * 100:
            return None
        
        # Risk scoring
        base_risk = self._calculate_base_risk_score(asset, mkt_data)
        
        # Mean reversion signals are generally lower risk, breakouts higher risk
        if "breakout" in signal_type:
            risk_adjustment = 10  # Breakouts can fail
        else:
            risk_adjustment = -5   # Mean reversion has statistical backing
            
        final_risk_score = max(0, min(100, base_risk + risk_adjustment))
        
        # Price targets
        if direction == SignalDirection.BUY:
            if "bounce" in signal_type:
                # Target middle band for mean reversion
                target_price = bb_middle
                stop_loss_price = int(bb_lower * 0.98)  # 2% below lower band
            else:
                # Breakout target above upper band
                target_price = int(bb_upper * 1.02)
                stop_loss_price = int(current_price * 0.96)
        else:
            if "rejection" in signal_type:
                # Target middle band for mean reversion
                target_price = bb_middle
                stop_loss_price = int(bb_upper * 1.02)  # 2% above upper band
            else:
                # Breakout target below lower band
                target_price = int(bb_lower * 0.98)
                stop_loss_price = int(current_price * 1.04)
        
        # Create signal
        signal_id = uuid4()
        signal = Signal(
            id=signal_id,
            asset_id=asset.id,
            signal_type_id=1,  # Technical signal type
            signal_name=f"BB {signal_type.replace('_', ' ').title()}",
            direction=direction,
            risk_score=final_risk_score,
            profit_potential_score=signal_strength_score,
            confidence_score=confidence_score,
            entry_price_cents=current_price,
            target_price_cents=target_price,
            stop_loss_price_cents=stop_loss_price,
            signal_source=self.name,
            methodology_version="1.0",
            signal_strength=self._determine_signal_strength(confidence_score),
            status=SignalStatus.ACTIVE,
            valid_until=datetime.utcnow() + timedelta(hours=self.config.signal_ttl_hours),
            recommended_holding_period_hours=24 if "breakout" in signal_type else 48,
            asset_specific_data={
                "band_position": band_position,
                "band_width_pct": band_width_pct,
                "signal_type": signal_type,
                "bb_upper": bb_upper,
                "bb_middle": bb_middle,
                "bb_lower": bb_lower,
                "squeeze_active": band_width_pct < self.squeeze_threshold
            }
        )
        
        # Add signal factors
        factors = [
            self._create_signal_factor(
                signal_id, "Band Position", signal_strength_score, 0.5,
                factor_value=Decimal(str(band_position)),
                factor_percentile=int(band_position * 100),
                calculation_method=f"BB({self.bb_period},{self.bb_std_dev})"
            ),
            self._create_signal_factor(
                signal_id, "Volume Confirmation", volume_factor_score, 0.3,
                factor_value=Decimal(str(mkt_data.volume)),
                calculation_method="Volume Ratio vs 30d Avg"
            )
        ]
        
        if "bounce" in signal_type or "rejection" in signal_type:
            factors.append(
                self._create_signal_factor(
                    signal_id, "RSI Confirmation", rsi_factor_score, 0.2,
                    factor_value=Decimal(str(tech_data.rsi_14)) if tech_data.rsi_14 else None,
                    calculation_method="RSI-14 Alignment"
                )
            )
        
        signal.factors = factors
        
        return signal
    
    def _determine_signal_strength(self, confidence_score: int) -> SignalStrength:
        """Determine signal strength based on confidence score"""
        if confidence_score >= 85:
            return SignalStrength.VERY_STRONG
        elif confidence_score >= 70:
            return SignalStrength.STRONG
        elif confidence_score >= 55:
            return SignalStrength.MODERATE
        else:
            return SignalStrength.WEAK
    
    def validate_configuration(self) -> Tuple[bool, List[str]]:
        """Validate generator configuration"""
        errors = []
        
        if self.bb_period <= 0:
            errors.append("bb_period must be positive")
            
        if self.bb_std_dev <= 0:
            errors.append("bb_std_dev must be positive")
            
        if not 0 < self.squeeze_threshold < 1:
            errors.append("squeeze_threshold must be between 0 and 1")
            
        if not 0 < self.band_touch_sensitivity < 0.1:
            errors.append("band_touch_sensitivity must be between 0 and 0.1")
            
        return len(errors) == 0, errors