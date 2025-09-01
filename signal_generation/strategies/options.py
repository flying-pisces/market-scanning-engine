"""
Options Flow Analysis Signal Generators
Author: Claude Code (System Architect)
Version: 1.0

Implementation of options-specific signal generators including:
- Options Flow Analysis (unusual volume, large orders)
- Gamma Exposure Signals (dealer positioning)
- Put/Call Ratio Analysis
- Unusual Activity Detection
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
from decimal import Decimal
from uuid import UUID, uuid4
from statistics import mean, stdev
from collections import defaultdict

from signal_generation.core.base_generator import BaseSignalGenerator, GeneratorConfig, SignalGenerationResult
from data_models.python.core_models import Asset, MarketData, TechnicalIndicators, OptionsData, OptionType
from data_models.python.signal_models import (
    Signal, SignalFactor, SignalDirection, SignalStrength,
    SignalStatus, SignalCategory
)


class OptionsFlowSignalGenerator(BaseSignalGenerator):
    """
    Options Flow Signal Generator
    
    Analyzes options flow patterns to identify:
    - Unusual options volume
    - Large block trades  
    - Smart money flows
    - Directional bias from options activity
    """
    
    def __init__(self, config: GeneratorConfig):
        super().__init__(config)
        
        # Flow analysis parameters
        self.volume_threshold_multiplier = config.strategy_params.get("volume_threshold_multiplier", 3.0)
        self.min_trade_size = config.strategy_params.get("min_trade_size", 50)  # contracts
        self.premium_threshold = config.strategy_params.get("premium_threshold", 100000)  # $1000 in cents
        self.time_decay_days = config.strategy_params.get("time_decay_days", 30)  # DTE filter
        self.moneyness_range = config.strategy_params.get("moneyness_range", 0.2)  # ATM +/- 20%
        
        # Smart money indicators
        self.sweep_detection = config.strategy_params.get("sweep_detection", True)
        self.block_trade_threshold = config.strategy_params.get("block_trade_threshold", 100)  # contracts
        self.dark_pool_premium_threshold = config.strategy_params.get("dark_pool_premium_threshold", 500000)  # $5000
    
    async def generate_signals(self,
                              assets: List[Asset],
                              market_data: Dict[UUID, MarketData],
                              technical_indicators: Dict[UUID, TechnicalIndicators],
                              options_data: Dict[UUID, List[OptionsData]] = None,
                              additional_data: Dict[str, Any] = None) -> SignalGenerationResult:
        
        if not options_data:
            return SignalGenerationResult(
                signals=[],
                metadata={"strategy": "options_flow", "error": "No options data provided"},
                errors=["Options data required for options flow analysis"],
                execution_time_ms=0.0,
                assets_processed=0
            )
        
        signals = []
        
        for asset in assets:
            if asset.id not in options_data or asset.id not in market_data:
                continue
                
            option_chains = options_data[asset.id]
            mkt_data = market_data[asset.id]
            
            signal = await self._analyze_options_flow(asset, option_chains, mkt_data)
            if signal:
                signals.append(signal)
        
        return SignalGenerationResult(
            signals=signals,
            metadata={
                "strategy": "options_flow",
                "parameters": {
                    "volume_threshold_multiplier": self.volume_threshold_multiplier,
                    "min_trade_size": self.min_trade_size,
                    "premium_threshold": self.premium_threshold
                }
            },
            errors=[],
            execution_time_ms=0.0,
            assets_processed=len(assets)
        )
    
    async def _analyze_options_flow(self,
                                   asset: Asset,
                                   option_chains: List[OptionsData],
                                   mkt_data: MarketData) -> Optional[Signal]:
        
        current_price = mkt_data.close_price_cents
        current_time = datetime.utcnow()
        
        # Filter options for analysis
        relevant_options = self._filter_relevant_options(option_chains, current_price, current_time)
        
        if len(relevant_options) < 5:  # Need sufficient data
            return None
        
        # Calculate flow metrics
        flow_analysis = self._calculate_flow_metrics(relevant_options, current_price)
        
        if not flow_analysis["unusual_activity"]:
            return None
        
        # Determine signal direction and strength
        direction, signal_strength = self._determine_flow_direction(flow_analysis)
        
        if not direction:
            return None
        
        # Smart money detection
        smart_money_score = self._detect_smart_money_patterns(relevant_options, flow_analysis)
        
        # Calculate confidence based on multiple factors
        confidence_factors = {
            "volume_anomaly": flow_analysis["volume_score"],
            "premium_flow": flow_analysis["premium_score"],
            "smart_money": smart_money_score,
            "time_clustering": flow_analysis["time_clustering_score"],
            "strike_concentration": flow_analysis["strike_concentration_score"]
        }
        
        confidence_score = int(sum(score * weight for score, weight in [
            (confidence_factors["volume_anomaly"], 0.25),
            (confidence_factors["premium_flow"], 0.25),
            (confidence_factors["smart_money"], 0.25),
            (confidence_factors["time_clustering"], 0.15),
            (confidence_factors["strike_concentration"], 0.10)
        ]))
        
        if confidence_score < self.config.min_confidence_threshold * 100:
            return None
        
        # Risk scoring for options flow
        base_risk = self._calculate_base_risk_score(asset, mkt_data)
        
        # Options flow signals are inherently high risk
        risk_adjustments = 0
        risk_adjustments += 15  # Base options risk premium
        
        # Higher risk for extreme unusual activity (could be hedging)
        if flow_analysis["volume_score"] > 90:
            risk_adjustments += 10
        
        # Lower risk if smart money is clearly involved
        if smart_money_score > 80:
            risk_adjustments -= 5
        
        final_risk_score = min(100, base_risk + risk_adjustments)
        
        # Price targets based on flow concentration
        target_strikes = flow_analysis["primary_strikes"]
        entry_price = current_price
        
        if direction == SignalDirection.BUY:
            # Target highest call strike with significant activity
            call_strikes = [s for s in target_strikes if s > current_price]
            target_price = int(min(call_strikes) * 1.02) if call_strikes else int(current_price * 1.05)
            stop_loss_price = int(current_price * 0.95)
        else:
            # Target lowest put strike with significant activity
            put_strikes = [s for s in target_strikes if s < current_price]
            target_price = int(max(put_strikes) * 0.98) if put_strikes else int(current_price * 0.95)
            stop_loss_price = int(current_price * 1.05)
        
        # Create signal
        signal_id = uuid4()
        signal = Signal(
            id=signal_id,
            asset_id=asset.id,
            signal_type_id=2,  # Options signal type
            signal_name=f"Options Flow {direction.value} - {flow_analysis['flow_type']}",
            direction=direction,
            risk_score=final_risk_score,
            profit_potential_score=signal_strength,
            confidence_score=confidence_score,
            entry_price_cents=entry_price,
            target_price_cents=target_price,
            stop_loss_price_cents=stop_loss_price,
            signal_source=self.name,
            methodology_version="1.0",
            signal_strength=self._determine_signal_strength(confidence_score),
            status=SignalStatus.ACTIVE,
            valid_until=datetime.utcnow() + timedelta(hours=8),  # Options flow is short-term
            recommended_holding_period_hours=4,  # Very short-term signal
            asset_specific_data={
                "flow_type": flow_analysis["flow_type"],
                "total_unusual_volume": flow_analysis["total_unusual_volume"],
                "total_premium": flow_analysis["total_premium"],
                "smart_money_indicators": flow_analysis["smart_money_indicators"],
                "primary_strikes": target_strikes,
                "call_put_ratio": flow_analysis["call_put_ratio"],
                "avg_time_to_expiry_days": flow_analysis["avg_dte"]
            }
        )
        
        # Add signal factors
        factors = [
            self._create_signal_factor(
                signal_id, "Volume Anomaly", confidence_factors["volume_anomaly"], 0.25,
                factor_value=Decimal(str(flow_analysis["total_unusual_volume"])),
                calculation_method="Volume vs Historical Average"
            ),
            self._create_signal_factor(
                signal_id, "Premium Flow", confidence_factors["premium_flow"], 0.25,
                factor_value=Decimal(str(flow_analysis["total_premium"])),
                calculation_method="Premium Weighted Flow Analysis"
            ),
            self._create_signal_factor(
                signal_id, "Smart Money Detection", confidence_factors["smart_money"], 0.25,
                factor_value=Decimal(str(smart_money_score)),
                calculation_method="Block Trade and Sweep Detection"
            ),
            self._create_signal_factor(
                signal_id, "Time Clustering", confidence_factors["time_clustering"], 0.15,
                factor_value=Decimal(str(flow_analysis["time_clustering_score"])),
                calculation_method="Trade Time Distribution Analysis"
            ),
            self._create_signal_factor(
                signal_id, "Strike Concentration", confidence_factors["strike_concentration"], 0.10,
                factor_value=Decimal(str(len(target_strikes))),
                calculation_method="Strike Price Activity Concentration"
            )
        ]
        
        signal.factors = factors
        
        return signal
    
    def _filter_relevant_options(self,
                               option_chains: List[OptionsData],
                               current_price: int,
                               current_time: datetime) -> List[OptionsData]:
        """Filter options for relevant analysis"""
        filtered = []
        
        for option in option_chains:
            # Time to expiry filter
            days_to_expiry = (option.expiration_date - current_time.date()).days
            if days_to_expiry < 1 or days_to_expiry > self.time_decay_days:
                continue
            
            # Moneyness filter (ATM +/- range)
            strike = option.strike_price_cents
            moneyness = abs(strike - current_price) / current_price
            if moneyness > self.moneyness_range:
                continue
            
            # Volume filter
            if option.volume < self.min_trade_size:
                continue
            
            # Premium filter
            if option.last_price_cents and option.last_price_cents * option.volume < self.premium_threshold:
                continue
            
            filtered.append(option)
        
        return filtered
    
    def _calculate_flow_metrics(self,
                              options: List[OptionsData],
                              current_price: int) -> Dict[str, Any]:
        """Calculate comprehensive flow metrics"""
        
        # Separate calls and puts
        calls = [opt for opt in options if opt.option_type == OptionType.CALL]
        puts = [opt for opt in options if opt.option_type == OptionType.PUT]
        
        # Volume analysis
        total_call_volume = sum(opt.volume for opt in calls)
        total_put_volume = sum(opt.volume for opt in puts)
        total_volume = total_call_volume + total_put_volume
        
        # Premium analysis
        total_call_premium = sum(opt.last_price_cents * opt.volume for opt in calls if opt.last_price_cents)
        total_put_premium = sum(opt.last_price_cents * opt.volume for opt in puts if opt.last_price_cents)
        total_premium = total_call_premium + total_put_premium
        
        # Historical comparison (simplified - would use actual historical data)
        avg_daily_volume = sum(opt.open_interest for opt in options) / max(len(options), 1)
        volume_ratio = total_volume / max(avg_daily_volume, 1)
        
        # Unusual activity detection
        volume_score = min(100, int(volume_ratio * 20)) if volume_ratio > self.volume_threshold_multiplier else 0
        unusual_activity = volume_score > 60
        
        # Premium score based on size and concentration
        premium_score = min(100, int(total_premium / 10000000 * 50))  # Scale by $100k
        
        # Time clustering (simplified)
        time_clustering_score = 70 if len(options) > 10 else 40
        
        # Strike concentration
        strike_distribution = defaultdict(int)
        for opt in options:
            strike_distribution[opt.strike_price_cents] += opt.volume
        
        # Find strikes with significant activity
        significant_strikes = [
            strike for strike, volume in strike_distribution.items()
            if volume > total_volume * 0.1  # At least 10% of total volume
        ]
        
        strike_concentration_score = min(100, len(significant_strikes) * 15)
        
        # Call/Put ratio
        call_put_ratio = total_call_volume / max(total_put_volume, 1)
        
        # Flow type determination
        if call_put_ratio > 2.0:
            flow_type = "bullish_call_heavy"
        elif call_put_ratio < 0.5:
            flow_type = "bearish_put_heavy"
        elif total_call_premium > total_put_premium * 1.5:
            flow_type = "bullish_premium_weighted"
        elif total_put_premium > total_call_premium * 1.5:
            flow_type = "bearish_premium_weighted"
        else:
            flow_type = "mixed_flow"
        
        # Average days to expiry
        avg_dte = mean([(opt.expiration_date - datetime.utcnow().date()).days for opt in options])
        
        return {
            "unusual_activity": unusual_activity,
            "total_volume": total_volume,
            "total_unusual_volume": max(0, total_volume - avg_daily_volume),
            "total_premium": total_premium,
            "volume_score": volume_score,
            "premium_score": premium_score,
            "time_clustering_score": time_clustering_score,
            "strike_concentration_score": strike_concentration_score,
            "call_put_ratio": call_put_ratio,
            "flow_type": flow_type,
            "primary_strikes": significant_strikes,
            "avg_dte": avg_dte,
            "smart_money_indicators": []
        }
    
    def _determine_flow_direction(self, flow_analysis: Dict[str, Any]) -> Tuple[Optional[SignalDirection], int]:
        """Determine signal direction and strength from flow analysis"""
        
        flow_type = flow_analysis["flow_type"]
        call_put_ratio = flow_analysis["call_put_ratio"]
        
        # Base signal strength from volume and premium scores
        base_strength = int((flow_analysis["volume_score"] + flow_analysis["premium_score"]) / 2)
        
        if "bullish" in flow_type:
            return SignalDirection.BUY, min(100, base_strength + 10)
        elif "bearish" in flow_type:
            return SignalDirection.SELL, min(100, base_strength + 10)
        elif call_put_ratio > 1.5:
            return SignalDirection.BUY, base_strength
        elif call_put_ratio < 0.67:
            return SignalDirection.SELL, base_strength
        else:
            return None, 0
    
    def _detect_smart_money_patterns(self,
                                   options: List[OptionsData],
                                   flow_analysis: Dict[str, Any]) -> int:
        """Detect smart money patterns in options flow"""
        
        smart_money_score = 0
        indicators = []
        
        # Large block trades
        large_trades = [opt for opt in options if opt.volume >= self.block_trade_threshold]
        if large_trades:
            smart_money_score += min(30, len(large_trades) * 10)
            indicators.append(f"Large blocks: {len(large_trades)}")
        
        # High premium trades (likely institutional)
        high_premium_trades = [
            opt for opt in options
            if opt.last_price_cents and opt.last_price_cents * opt.volume >= self.dark_pool_premium_threshold
        ]
        if high_premium_trades:
            smart_money_score += min(40, len(high_premium_trades) * 15)
            indicators.append(f"High premium trades: {len(high_premium_trades)}")
        
        # Sweep detection (simplified - multiple strikes hit simultaneously)
        strike_times = defaultdict(list)
        for opt in options:
            strike_times[opt.strike_price_cents].append(opt.timestamp)
        
        # Look for simultaneous activity across multiple strikes
        simultaneous_strikes = sum(1 for strikes in strike_times.values() if len(strikes) > 2)
        if simultaneous_strikes >= 3:
            smart_money_score += 30
            indicators.append(f"Multi-strike sweep detected")
        
        flow_analysis["smart_money_indicators"] = indicators
        return min(100, smart_money_score)
    
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
        
        if self.volume_threshold_multiplier <= 1.0:
            errors.append("volume_threshold_multiplier must be greater than 1.0")
        
        if self.min_trade_size <= 0:
            errors.append("min_trade_size must be positive")
        
        if self.premium_threshold <= 0:
            errors.append("premium_threshold must be positive")
        
        if self.time_decay_days <= 0:
            errors.append("time_decay_days must be positive")
        
        if not 0 < self.moneyness_range <= 1:
            errors.append("moneyness_range must be between 0 and 1")
        
        return len(errors) == 0, errors


class GammaExposureSignalGenerator(BaseSignalGenerator):
    """
    Gamma Exposure Signal Generator
    
    Analyzes dealer gamma exposure to identify:
    - Support and resistance levels from gamma walls
    - Squeeze potential from negative gamma
    - Market maker positioning effects
    """
    
    def __init__(self, config: GeneratorConfig):
        super().__init__(config)
        
        self.gamma_threshold = config.strategy_params.get("gamma_threshold", 1000000)  # $10M gamma
        self.strike_spacing = config.strategy_params.get("strike_spacing", 500)  # 5.00 strike spacing
        self.max_dte = config.strategy_params.get("max_dte", 45)  # Maximum days to expiry
        self.min_open_interest = config.strategy_params.get("min_open_interest", 100)
    
    async def generate_signals(self,
                              assets: List[Asset],
                              market_data: Dict[UUID, MarketData],
                              technical_indicators: Dict[UUID, TechnicalIndicators],
                              options_data: Dict[UUID, List[OptionsData]] = None,
                              additional_data: Dict[str, Any] = None) -> SignalGenerationResult:
        
        if not options_data:
            return SignalGenerationResult(
                signals=[],
                metadata={"strategy": "gamma_exposure", "error": "No options data provided"},
                errors=["Options data required for gamma analysis"],
                execution_time_ms=0.0,
                assets_processed=0
            )
        
        signals = []
        
        for asset in assets:
            if asset.id not in options_data or asset.id not in market_data:
                continue
                
            option_chains = options_data[asset.id]
            mkt_data = market_data[asset.id]
            
            signal = await self._analyze_gamma_exposure(asset, option_chains, mkt_data)
            if signal:
                signals.append(signal)
        
        return SignalGenerationResult(
            signals=signals,
            metadata={
                "strategy": "gamma_exposure",
                "parameters": {
                    "gamma_threshold": self.gamma_threshold,
                    "strike_spacing": self.strike_spacing
                }
            },
            errors=[],
            execution_time_ms=0.0,
            assets_processed=len(assets)
        )
    
    async def _analyze_gamma_exposure(self,
                                    asset: Asset,
                                    option_chains: List[OptionsData],
                                    mkt_data: MarketData) -> Optional[Signal]:
        
        current_price = mkt_data.close_price_cents
        current_time = datetime.utcnow()
        
        # Filter relevant options
        relevant_options = [
            opt for opt in option_chains
            if opt.gamma and opt.open_interest >= self.min_open_interest
            and (opt.expiration_date - current_time.date()).days <= self.max_dte
        ]
        
        if len(relevant_options) < 10:
            return None
        
        # Calculate gamma exposure by strike
        gamma_profile = self._calculate_gamma_profile(relevant_options, current_price)
        
        if not gamma_profile["significant_levels"]:
            return None
        
        # Analyze gamma implications
        analysis = self._analyze_gamma_implications(gamma_profile, current_price, mkt_data)
        
        if not analysis["signal_direction"]:
            return None
        
        direction = analysis["signal_direction"]
        signal_strength = analysis["signal_strength"]
        confidence_score = analysis["confidence_score"]
        
        if confidence_score < self.config.min_confidence_threshold * 100:
            return None
        
        # Risk scoring for gamma-based signals
        base_risk = self._calculate_base_risk_score(asset, mkt_data)
        
        # Gamma signals can be very directional but timing-sensitive
        risk_adjustments = 20  # Base gamma risk premium
        
        # Higher risk in negative gamma environment (squeeze potential)
        if gamma_profile["net_gamma"] < 0:
            risk_adjustments += 15
        else:
            risk_adjustments += 5
        
        final_risk_score = min(100, base_risk + risk_adjustments)
        
        # Price targets based on gamma levels
        target_price = analysis["target_level"]
        stop_loss_price = analysis["stop_loss_level"]
        
        # Create signal
        signal_id = uuid4()
        signal = Signal(
            id=signal_id,
            asset_id=asset.id,
            signal_type_id=2,  # Options signal type
            signal_name=f"Gamma {analysis['scenario']} - {direction.value}",
            direction=direction,
            risk_score=final_risk_score,
            profit_potential_score=signal_strength,
            confidence_score=confidence_score,
            entry_price_cents=current_price,
            target_price_cents=target_price,
            stop_loss_price_cents=stop_loss_price,
            signal_source=self.name,
            methodology_version="1.0",
            signal_strength=self._determine_signal_strength(confidence_score),
            status=SignalStatus.ACTIVE,
            valid_until=datetime.utcnow() + timedelta(hours=12),
            recommended_holding_period_hours=6,  # Gamma effects are intraday
            asset_specific_data={
                "scenario": analysis["scenario"],
                "net_gamma": gamma_profile["net_gamma"],
                "gamma_levels": gamma_profile["significant_levels"],
                "nearest_gamma_wall": analysis["nearest_gamma_wall"],
                "squeeze_potential": gamma_profile["net_gamma"] < 0,
                "dealer_positioning": analysis["dealer_positioning"]
            }
        )
        
        # Add signal factors
        factors = [
            self._create_signal_factor(
                signal_id, "Gamma Magnitude", signal_strength, 0.4,
                factor_value=Decimal(str(abs(gamma_profile["net_gamma"]))),
                calculation_method="Net Dealer Gamma Exposure"
            ),
            self._create_signal_factor(
                signal_id, "Level Proximity", analysis["proximity_score"], 0.3,
                factor_value=Decimal(str(analysis["distance_to_level"])),
                calculation_method="Distance to Significant Gamma Level"
            ),
            self._create_signal_factor(
                signal_id, "Open Interest Support", analysis["oi_score"], 0.3,
                factor_value=Decimal(str(gamma_profile["total_oi"])),
                calculation_method="Open Interest Weighted Gamma"
            )
        ]
        
        signal.factors = factors
        
        return signal
    
    def _calculate_gamma_profile(self,
                               options: List[OptionsData],
                               current_price: int) -> Dict[str, Any]:
        """Calculate gamma exposure profile across strikes"""
        
        gamma_by_strike = defaultdict(float)
        oi_by_strike = defaultdict(int)
        
        for opt in options:
            strike = opt.strike_price_cents
            gamma = float(opt.gamma) / 10000 if opt.gamma else 0  # Convert from scaled
            oi = opt.open_interest
            
            # Dealer gamma = -customer gamma
            # Calls: dealer is short gamma (negative), customers long gamma
            # Puts: dealer is long gamma (positive), customers short gamma
            if opt.option_type == OptionType.CALL:
                dealer_gamma = -gamma * oi * 100  # 100 shares per contract
            else:
                dealer_gamma = gamma * oi * 100
            
            gamma_by_strike[strike] += dealer_gamma
            oi_by_strike[strike] += oi
        
        # Find significant gamma levels
        significant_levels = []
        for strike, gamma_exposure in gamma_by_strike.items():
            if abs(gamma_exposure) > self.gamma_threshold:
                significant_levels.append({
                    "strike": strike,
                    "gamma_exposure": gamma_exposure,
                    "open_interest": oi_by_strike[strike],
                    "distance_pct": abs(strike - current_price) / current_price * 100
                })
        
        # Sort by gamma magnitude
        significant_levels.sort(key=lambda x: abs(x["gamma_exposure"]), reverse=True)
        
        # Calculate net gamma
        net_gamma = sum(gamma_by_strike.values())
        total_oi = sum(oi_by_strike.values())
        
        return {
            "gamma_by_strike": gamma_by_strike,
            "significant_levels": significant_levels[:10],  # Top 10 levels
            "net_gamma": net_gamma,
            "total_oi": total_oi
        }
    
    def _analyze_gamma_implications(self,
                                  gamma_profile: Dict[str, Any],
                                  current_price: int,
                                  mkt_data: MarketData) -> Dict[str, Any]:
        """Analyze gamma profile for trading implications"""
        
        significant_levels = gamma_profile["significant_levels"]
        net_gamma = gamma_profile["net_gamma"]
        
        if not significant_levels:
            return {"signal_direction": None}
        
        # Find nearest significant level
        nearest_level = min(significant_levels, 
                          key=lambda x: abs(x["strike"] - current_price))
        
        distance_to_level = abs(nearest_level["strike"] - current_price)
        distance_pct = distance_to_level / current_price * 100
        
        # Determine scenario
        scenario = ""
        signal_direction = None
        signal_strength = 0
        
        if net_gamma < -50000000:  # Significant negative gamma
            scenario = "negative_gamma_squeeze"
            # In negative gamma, dealers amplify moves
            # Look for breakout direction
            if current_price > nearest_level["strike"]:
                signal_direction = SignalDirection.BUY  # Upside squeeze
            else:
                signal_direction = SignalDirection.SELL  # Downside squeeze
            signal_strength = min(100, int(abs(net_gamma) / 1000000))
            
        elif distance_pct < 2:  # Very close to gamma wall
            if nearest_level["gamma_exposure"] > 0:
                # Positive gamma acts as support/resistance
                scenario = "gamma_wall_rejection"
                if current_price < nearest_level["strike"]:
                    signal_direction = SignalDirection.BUY  # Support bounce
                else:
                    signal_direction = SignalDirection.SELL  # Resistance rejection
            else:
                # Negative gamma accelerates moves
                scenario = "gamma_wall_break"
                if current_price > nearest_level["strike"]:
                    signal_direction = SignalDirection.BUY  # Break higher
                else:
                    signal_direction = SignalDirection.SELL  # Break lower
            
            signal_strength = min(100, int(abs(nearest_level["gamma_exposure"]) / 10000000))
        
        else:
            # Not close enough to any level
            return {"signal_direction": None}
        
        # Calculate confidence factors
        proximity_score = max(0, 100 - distance_pct * 20)  # Closer = higher score
        gamma_magnitude_score = min(100, abs(nearest_level["gamma_exposure"]) / 50000000 * 100)
        oi_score = min(100, nearest_level["open_interest"] / 1000 * 50)
        
        confidence_score = int((proximity_score * 0.4 + gamma_magnitude_score * 0.4 + oi_score * 0.2))
        
        # Price targets
        if signal_direction == SignalDirection.BUY:
            # Target next resistance level or 2% move
            higher_levels = [l for l in significant_levels if l["strike"] > current_price]
            target_level = higher_levels[0]["strike"] if higher_levels else int(current_price * 1.02)
            stop_loss_level = int(current_price * 0.98)
        else:
            # Target next support level or 2% move
            lower_levels = [l for l in significant_levels if l["strike"] < current_price]
            target_level = lower_levels[0]["strike"] if lower_levels else int(current_price * 0.98)
            stop_loss_level = int(current_price * 1.02)
        
        return {
            "signal_direction": signal_direction,
            "signal_strength": signal_strength,
            "confidence_score": confidence_score,
            "scenario": scenario,
            "nearest_gamma_wall": nearest_level["strike"],
            "target_level": target_level,
            "stop_loss_level": stop_loss_level,
            "proximity_score": proximity_score,
            "distance_to_level": distance_to_level,
            "oi_score": oi_score,
            "dealer_positioning": "net_short_gamma" if net_gamma < 0 else "net_long_gamma"
        }
    
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
        
        if self.gamma_threshold <= 0:
            errors.append("gamma_threshold must be positive")
        
        if self.strike_spacing <= 0:
            errors.append("strike_spacing must be positive")
        
        if self.max_dte <= 0:
            errors.append("max_dte must be positive")
        
        if self.min_open_interest <= 0:
            errors.append("min_open_interest must be positive")
        
        return len(errors) == 0, errors


class PutCallRatioSignalGenerator(BaseSignalGenerator):
    """
    Put/Call Ratio Signal Generator
    
    Analyzes put/call ratios to identify:
    - Extreme sentiment readings
    - Contrarian opportunities
    - Hedging activity patterns
    """
    
    def __init__(self, config: GeneratorConfig):
        super().__init__(config)
        
        # Ratio thresholds for contrarian signals
        self.extreme_bearish_threshold = config.strategy_params.get("extreme_bearish_threshold", 1.5)
        self.extreme_bullish_threshold = config.strategy_params.get("extreme_bullish_threshold", 0.4)
        self.moderate_bearish_threshold = config.strategy_params.get("moderate_bearish_threshold", 1.1)
        self.moderate_bullish_threshold = config.strategy_params.get("moderate_bullish_threshold", 0.7)
        
        # Analysis parameters
        self.volume_weighted = config.strategy_params.get("volume_weighted", True)
        self.oi_weighted = config.strategy_params.get("oi_weighted", False)
        self.max_dte = config.strategy_params.get("max_dte", 60)
        self.min_volume = config.strategy_params.get("min_volume", 10)
    
    async def generate_signals(self,
                              assets: List[Asset],
                              market_data: Dict[UUID, MarketData],
                              technical_indicators: Dict[UUID, TechnicalIndicators],
                              options_data: Dict[UUID, List[OptionsData]] = None,
                              additional_data: Dict[str, Any] = None) -> SignalGenerationResult:
        
        if not options_data:
            return SignalGenerationResult(
                signals=[],
                metadata={"strategy": "put_call_ratio", "error": "No options data provided"},
                errors=["Options data required for put/call ratio analysis"],
                execution_time_ms=0.0,
                assets_processed=0
            )
        
        signals = []
        
        for asset in assets:
            if asset.id not in options_data or asset.id not in market_data:
                continue
                
            option_chains = options_data[asset.id]
            mkt_data = market_data[asset.id]
            tech_data = technical_indicators.get(asset.id)
            
            signal = await self._analyze_put_call_ratio(asset, option_chains, mkt_data, tech_data)
            if signal:
                signals.append(signal)
        
        return SignalGenerationResult(
            signals=signals,
            metadata={
                "strategy": "put_call_ratio",
                "parameters": {
                    "extreme_bearish_threshold": self.extreme_bearish_threshold,
                    "extreme_bullish_threshold": self.extreme_bullish_threshold,
                    "volume_weighted": self.volume_weighted
                }
            },
            errors=[],
            execution_time_ms=0.0,
            assets_processed=len(assets)
        )
    
    async def _analyze_put_call_ratio(self,
                                    asset: Asset,
                                    option_chains: List[OptionsData],
                                    mkt_data: MarketData,
                                    tech_data: Optional[TechnicalIndicators] = None) -> Optional[Signal]:
        
        current_time = datetime.utcnow()
        
        # Filter relevant options
        relevant_options = [
            opt for opt in option_chains
            if opt.volume >= self.min_volume
            and (opt.expiration_date - current_time.date()).days <= self.max_dte
        ]
        
        if len(relevant_options) < 5:
            return None
        
        # Calculate ratios
        ratio_analysis = self._calculate_put_call_ratios(relevant_options)
        
        # Determine signal based on extreme readings
        signal_analysis = self._analyze_ratio_extremes(ratio_analysis, mkt_data, tech_data)
        
        if not signal_analysis["signal_direction"]:
            return None
        
        direction = signal_analysis["signal_direction"]
        confidence_score = signal_analysis["confidence_score"]
        
        if confidence_score < self.config.min_confidence_threshold * 100:
            return None
        
        # Risk scoring
        base_risk = self._calculate_base_risk_score(asset, mkt_data)
        
        # Put/call ratio signals are contrarian and can be early
        risk_adjustments = 10  # Base contrarian risk premium
        
        # Higher risk for extreme readings (could persist longer)
        if ratio_analysis["primary_ratio"] > self.extreme_bearish_threshold:
            risk_adjustments += 15
        elif ratio_analysis["primary_ratio"] < self.extreme_bullish_threshold:
            risk_adjustments += 15
        else:
            risk_adjustments += 5
        
        final_risk_score = min(100, base_risk + risk_adjustments)
        
        signal_strength = signal_analysis["signal_strength"]
        
        # Price targets
        current_price = mkt_data.close_price_cents
        
        if direction == SignalDirection.BUY:
            # Contrarian buy after extreme bearishness
            target_price = int(current_price * 1.05)  # Conservative 5% target
            stop_loss_price = int(current_price * 0.95)  # 5% stop
        else:
            # Contrarian sell after extreme bullishness
            target_price = int(current_price * 0.95)  # Conservative 5% target
            stop_loss_price = int(current_price * 1.05)  # 5% stop
        
        # Create signal
        signal_id = uuid4()
        signal = Signal(
            id=signal_id,
            asset_id=asset.id,
            signal_type_id=2,  # Options signal type
            signal_name=f"P/C Ratio {signal_analysis['signal_type']} - {direction.value}",
            direction=direction,
            risk_score=final_risk_score,
            profit_potential_score=signal_strength,
            confidence_score=confidence_score,
            entry_price_cents=current_price,
            target_price_cents=target_price,
            stop_loss_price_cents=stop_loss_price,
            signal_source=self.name,
            methodology_version="1.0",
            signal_strength=self._determine_signal_strength(confidence_score),
            status=SignalStatus.ACTIVE,
            valid_until=datetime.utcnow() + timedelta(hours=24),
            recommended_holding_period_hours=48,  # Contrarian signals take time
            asset_specific_data={
                "signal_type": signal_analysis["signal_type"],
                "put_call_ratio": ratio_analysis["primary_ratio"],
                "volume_ratio": ratio_analysis["volume_ratio"],
                "oi_ratio": ratio_analysis["oi_ratio"],
                "total_put_volume": ratio_analysis["total_put_volume"],
                "total_call_volume": ratio_analysis["total_call_volume"],
                "sentiment_extreme": signal_analysis["sentiment_extreme"]
            }
        )
        
        # Add signal factors
        factors = [
            self._create_signal_factor(
                signal_id, "Ratio Extreme", signal_strength, 0.5,
                factor_value=Decimal(str(ratio_analysis["primary_ratio"])),
                factor_percentile=signal_analysis["ratio_percentile"],
                calculation_method="Put/Call Volume Ratio"
            ),
            self._create_signal_factor(
                signal_id, "Technical Confirmation", signal_analysis["technical_score"], 0.3,
                calculation_method="RSI and Trend Alignment"
            ),
            self._create_signal_factor(
                signal_id, "Volume Support", signal_analysis["volume_score"], 0.2,
                factor_value=Decimal(str(ratio_analysis["total_volume"])),
                calculation_method="Total Options Volume"
            )
        ]
        
        signal.factors = factors
        
        return signal
    
    def _calculate_put_call_ratios(self, options: List[OptionsData]) -> Dict[str, Any]:
        """Calculate various put/call ratios"""
        
        calls = [opt for opt in options if opt.option_type == OptionType.CALL]
        puts = [opt for opt in options if opt.option_type == OptionType.PUT]
        
        # Volume-based ratio
        total_call_volume = sum(opt.volume for opt in calls)
        total_put_volume = sum(opt.volume for opt in puts)
        volume_ratio = total_put_volume / max(total_call_volume, 1)
        
        # Open interest-based ratio
        total_call_oi = sum(opt.open_interest for opt in calls)
        total_put_oi = sum(opt.open_interest for opt in puts)
        oi_ratio = total_put_oi / max(total_call_oi, 1)
        
        # Premium-weighted ratio
        call_premium = sum(opt.last_price_cents * opt.volume for opt in calls if opt.last_price_cents)
        put_premium = sum(opt.last_price_cents * opt.volume for opt in puts if opt.last_price_cents)
        premium_ratio = put_premium / max(call_premium, 1) if call_premium > 0 else 0
        
        # Choose primary ratio based on configuration
        if self.volume_weighted:
            primary_ratio = volume_ratio
        elif self.oi_weighted:
            primary_ratio = oi_ratio
        else:
            primary_ratio = (volume_ratio + oi_ratio) / 2
        
        return {
            "primary_ratio": primary_ratio,
            "volume_ratio": volume_ratio,
            "oi_ratio": oi_ratio,
            "premium_ratio": premium_ratio,
            "total_call_volume": total_call_volume,
            "total_put_volume": total_put_volume,
            "total_volume": total_call_volume + total_put_volume,
            "total_call_oi": total_call_oi,
            "total_put_oi": total_put_oi
        }
    
    def _analyze_ratio_extremes(self,
                               ratio_analysis: Dict[str, Any],
                               mkt_data: MarketData,
                               tech_data: Optional[TechnicalIndicators]) -> Dict[str, Any]:
        """Analyze ratio extremes for contrarian signals"""
        
        primary_ratio = ratio_analysis["primary_ratio"]
        
        signal_direction = None
        signal_type = ""
        signal_strength = 0
        sentiment_extreme = ""
        
        # Extreme bearish sentiment -> Contrarian BUY
        if primary_ratio >= self.extreme_bearish_threshold:
            signal_direction = SignalDirection.BUY
            signal_type = "extreme_bearish_contrarian"
            sentiment_extreme = "extreme_bearish"
            # Strength based on how extreme the reading is
            signal_strength = min(100, int((primary_ratio - 1.0) * 50))
            
        # Extreme bullish sentiment -> Contrarian SELL
        elif primary_ratio <= self.extreme_bullish_threshold:
            signal_direction = SignalDirection.SELL
            signal_type = "extreme_bullish_contrarian"
            sentiment_extreme = "extreme_bullish"
            # Strength based on how extreme the reading is
            signal_strength = min(100, int((1.0 - primary_ratio) * 100))
            
        # Moderate readings - less conviction
        elif primary_ratio >= self.moderate_bearish_threshold:
            signal_direction = SignalDirection.BUY
            signal_type = "moderate_bearish_contrarian"
            sentiment_extreme = "moderate_bearish"
            signal_strength = int((primary_ratio - 0.8) * 100)
            
        elif primary_ratio <= self.moderate_bullish_threshold:
            signal_direction = SignalDirection.SELL
            signal_type = "moderate_bullish_contrarian"
            sentiment_extreme = "moderate_bullish"
            signal_strength = int((1.0 - primary_ratio) * 60)
        
        if not signal_direction:
            return {"signal_direction": None}
        
        # Technical confirmation
        technical_score = 50  # Neutral default
        if tech_data and tech_data.rsi_14:
            rsi_value = float(tech_data.rsi_14)
            
            if signal_direction == SignalDirection.BUY:
                # Buy signal stronger if RSI also oversold
                if rsi_value < 30:
                    technical_score = 90
                elif rsi_value < 40:
                    technical_score = 70
                else:
                    technical_score = 50
            else:
                # Sell signal stronger if RSI also overbought
                if rsi_value > 70:
                    technical_score = 90
                elif rsi_value > 60:
                    technical_score = 70
                else:
                    technical_score = 50
        
        # Volume score
        total_volume = ratio_analysis["total_volume"]
        volume_score = min(100, int(total_volume / 1000 * 20))  # Scale by volume
        
        # Calculate confidence
        confidence_score = int((signal_strength * 0.5 + technical_score * 0.3 + volume_score * 0.2))
        
        # Ratio percentile (simplified)
        if primary_ratio > 1.5:
            ratio_percentile = 95
        elif primary_ratio > 1.2:
            ratio_percentile = 85
        elif primary_ratio > 1.0:
            ratio_percentile = 70
        elif primary_ratio > 0.8:
            ratio_percentile = 50
        elif primary_ratio > 0.6:
            ratio_percentile = 30
        elif primary_ratio > 0.4:
            ratio_percentile = 15
        else:
            ratio_percentile = 5
        
        return {
            "signal_direction": signal_direction,
            "signal_type": signal_type,
            "signal_strength": signal_strength,
            "confidence_score": confidence_score,
            "sentiment_extreme": sentiment_extreme,
            "technical_score": technical_score,
            "volume_score": volume_score,
            "ratio_percentile": ratio_percentile
        }
    
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
        
        if self.extreme_bearish_threshold <= 1.0:
            errors.append("extreme_bearish_threshold must be greater than 1.0")
        
        if self.extreme_bullish_threshold >= 1.0:
            errors.append("extreme_bullish_threshold must be less than 1.0")
        
        if self.moderate_bearish_threshold <= self.extreme_bullish_threshold:
            errors.append("moderate_bearish_threshold must be greater than extreme_bullish_threshold")
        
        if self.moderate_bullish_threshold >= self.extreme_bearish_threshold:
            errors.append("moderate_bullish_threshold must be less than extreme_bearish_threshold")
        
        if self.max_dte <= 0:
            errors.append("max_dte must be positive")
        
        if self.min_volume <= 0:
            errors.append("min_volume must be positive")
        
        return len(errors) == 0, errors


class UnusualVolumeSignalGenerator(BaseSignalGenerator):
    """
    Unusual Options Volume Signal Generator
    
    Identifies unusual options volume patterns that may indicate:
    - Insider activity
    - Institutional positioning
    - Event anticipation
    - Smart money flows
    """
    
    def __init__(self, config: GeneratorConfig):
        super().__init__(config)
        
        # Volume anomaly detection parameters
        self.volume_multiple_threshold = config.strategy_params.get("volume_multiple_threshold", 5.0)
        self.min_absolute_volume = config.strategy_params.get("min_absolute_volume", 100)
        self.historical_periods = config.strategy_params.get("historical_periods", 20)  # Days to compare
        self.dte_range = config.strategy_params.get("dte_range", [7, 60])  # Days to expiry range
        
        # Pattern recognition
        self.block_size_threshold = config.strategy_params.get("block_size_threshold", 50)
        self.time_concentration_window = config.strategy_params.get("time_concentration_window", 3600)  # 1 hour in seconds
    
    async def generate_signals(self,
                              assets: List[Asset],
                              market_data: Dict[UUID, MarketData],
                              technical_indicators: Dict[UUID, TechnicalIndicators],
                              options_data: Dict[UUID, List[OptionsData]] = None,
                              additional_data: Dict[str, Any] = None) -> SignalGenerationResult:
        
        if not options_data:
            return SignalGenerationResult(
                signals=[],
                metadata={"strategy": "unusual_volume", "error": "No options data provided"},
                errors=["Options data required for unusual volume analysis"],
                execution_time_ms=0.0,
                assets_processed=0
            )
        
        signals = []
        
        for asset in assets:
            if asset.id not in options_data or asset.id not in market_data:
                continue
                
            option_chains = options_data[asset.id]
            mkt_data = market_data[asset.id]
            tech_data = technical_indicators.get(asset.id)
            
            signal = await self._analyze_unusual_volume(asset, option_chains, mkt_data, tech_data)
            if signal:
                signals.append(signal)
        
        return SignalGenerationResult(
            signals=signals,
            metadata={
                "strategy": "unusual_volume",
                "parameters": {
                    "volume_multiple_threshold": self.volume_multiple_threshold,
                    "min_absolute_volume": self.min_absolute_volume,
                    "dte_range": self.dte_range
                }
            },
            errors=[],
            execution_time_ms=0.0,
            assets_processed=len(assets)
        )
    
    async def _analyze_unusual_volume(self,
                                    asset: Asset,
                                    option_chains: List[OptionsData],
                                    mkt_data: MarketData,
                                    tech_data: Optional[TechnicalIndicators]) -> Optional[Signal]:
        
        current_time = datetime.utcnow()
        current_price = mkt_data.close_price_cents
        
        # Filter options in DTE range
        relevant_options = [
            opt for opt in option_chains
            if self.dte_range[0] <= (opt.expiration_date - current_time.date()).days <= self.dte_range[1]
            and opt.volume >= self.min_absolute_volume
        ]
        
        if len(relevant_options) < 3:
            return None
        
        # Detect unusual volume patterns
        unusual_patterns = self._detect_unusual_patterns(relevant_options, current_price)
        
        if not unusual_patterns["has_unusual_activity"]:
            return None
        
        # Analyze patterns for directional bias
        analysis = self._analyze_pattern_implications(unusual_patterns, current_price, tech_data)
        
        if not analysis["signal_direction"]:
            return None
        
        direction = analysis["signal_direction"]
        confidence_score = analysis["confidence_score"]
        
        if confidence_score < self.config.min_confidence_threshold * 100:
            return None
        
        # Risk scoring
        base_risk = self._calculate_base_risk_score(asset, mkt_data)
        
        # Unusual volume signals are high conviction but high risk
        risk_adjustments = 25  # Base unusual activity risk premium
        
        # Adjust based on pattern type
        if "sweep" in unusual_patterns["pattern_type"]:
            risk_adjustments += 10  # Sweeps are aggressive
        elif "block" in unusual_patterns["pattern_type"]:
            risk_adjustments += 5   # Blocks are more institutional
        
        final_risk_score = min(100, base_risk + risk_adjustments)
        
        signal_strength = analysis["signal_strength"]
        
        # Price targets based on unusual activity concentration
        target_price = analysis.get("target_price", int(current_price * 1.03))
        stop_loss_price = analysis.get("stop_loss_price", int(current_price * 0.97))
        
        # Create signal
        signal_id = uuid4()
        signal = Signal(
            id=signal_id,
            asset_id=asset.id,
            signal_type_id=2,  # Options signal type
            signal_name=f"Unusual Volume {unusual_patterns['pattern_type']} - {direction.value}",
            direction=direction,
            risk_score=final_risk_score,
            profit_potential_score=signal_strength,
            confidence_score=confidence_score,
            entry_price_cents=current_price,
            target_price_cents=target_price,
            stop_loss_price_cents=stop_loss_price,
            signal_source=self.name,
            methodology_version="1.0",
            signal_strength=self._determine_signal_strength(confidence_score),
            status=SignalStatus.ACTIVE,
            valid_until=datetime.utcnow() + timedelta(hours=6),  # Short-term signal
            recommended_holding_period_hours=4,
            asset_specific_data={
                "pattern_type": unusual_patterns["pattern_type"],
                "total_unusual_volume": unusual_patterns["total_unusual_volume"],
                "volume_multiple": unusual_patterns["max_volume_multiple"],
                "unusual_strikes": unusual_patterns["unusual_strikes"],
                "time_concentration": unusual_patterns["time_concentration"],
                "call_put_split": unusual_patterns["call_put_split"],
                "institutional_indicators": analysis["institutional_indicators"]
            }
        )
        
        # Add signal factors
        factors = [
            self._create_signal_factor(
                signal_id, "Volume Anomaly", signal_strength, 0.4,
                factor_value=Decimal(str(unusual_patterns["max_volume_multiple"])),
                calculation_method="Volume vs Historical Average"
            ),
            self._create_signal_factor(
                signal_id, "Pattern Recognition", analysis["pattern_score"], 0.3,
                calculation_method=f"Pattern Type: {unusual_patterns['pattern_type']}"
            ),
            self._create_signal_factor(
                signal_id, "Directional Bias", analysis["directional_score"], 0.3,
                factor_value=Decimal(str(unusual_patterns["call_put_split"])),
                calculation_method="Call/Put Volume Analysis"
            )
        ]
        
        signal.factors = factors
        
        return signal
    
    def _detect_unusual_patterns(self,
                               options: List[OptionsData],
                               current_price: int) -> Dict[str, Any]:
        """Detect unusual volume patterns in options data"""
        
        unusual_options = []
        pattern_types = []
        
        for opt in options:
            # Calculate volume multiple (simplified - would use historical data)
            historical_avg_volume = max(opt.open_interest / 20, 10)  # Rough approximation
            volume_multiple = opt.volume / historical_avg_volume
            
            if volume_multiple >= self.volume_multiple_threshold:
                unusual_options.append({
                    "option": opt,
                    "volume_multiple": volume_multiple,
                    "moneyness": abs(opt.strike_price_cents - current_price) / current_price
                })
        
        if not unusual_options:
            return {"has_unusual_activity": False}
        
        # Analyze patterns
        total_unusual_volume = sum(item["option"].volume for item in unusual_options)
        max_volume_multiple = max(item["volume_multiple"] for item in unusual_options)
        
        # Pattern recognition
        pattern_type = "unusual_volume"
        
        # Check for block trades
        large_trades = [item for item in unusual_options 
                       if item["option"].volume >= self.block_size_threshold]
        if large_trades:
            pattern_type += "_block"
            pattern_types.append("block_trades")
        
        # Check for sweeps (multiple strikes hit)
        strikes_hit = len(set(item["option"].strike_price_cents for item in unusual_options))
        if strikes_hit >= 3:
            pattern_type += "_sweep"
            pattern_types.append("multi_strike_sweep")
        
        # Time concentration analysis (simplified)
        time_concentration = len(unusual_options) / max(1, len(set(item["option"].timestamp.hour 
                                                                  for item in unusual_options)))
        
        # Call/Put split
        calls = [item for item in unusual_options if item["option"].option_type == OptionType.CALL]
        puts = [item for item in unusual_options if item["option"].option_type == OptionType.PUT]
        
        call_volume = sum(item["option"].volume for item in calls)
        put_volume = sum(item["option"].volume for item in puts)
        call_put_split = call_volume / max(put_volume, 1)
        
        return {
            "has_unusual_activity": True,
            "unusual_options": unusual_options,
            "pattern_type": pattern_type,
            "pattern_types": pattern_types,
            "total_unusual_volume": total_unusual_volume,
            "max_volume_multiple": max_volume_multiple,
            "unusual_strikes": [item["option"].strike_price_cents for item in unusual_options],
            "time_concentration": time_concentration,
            "call_put_split": call_put_split,
            "strikes_hit": strikes_hit
        }
    
    def _analyze_pattern_implications(self,
                                    unusual_patterns: Dict[str, Any],
                                    current_price: int,
                                    tech_data: Optional[TechnicalIndicators]) -> Dict[str, Any]:
        """Analyze unusual patterns for trading implications"""
        
        call_put_split = unusual_patterns["call_put_split"]
        unusual_options = unusual_patterns["unusual_options"]
        pattern_types = unusual_patterns["pattern_types"]
        
        # Determine directional bias
        signal_direction = None
        directional_score = 0
        
        if call_put_split > 2.0:
            signal_direction = SignalDirection.BUY
            directional_score = min(100, int(call_put_split * 30))
        elif call_put_split < 0.5:
            signal_direction = SignalDirection.SELL
            directional_score = min(100, int((1/call_put_split) * 30))
        
        if not signal_direction:
            return {"signal_direction": None}
        
        # Pattern scoring
        pattern_score = 50  # Base score
        
        if "block_trades" in pattern_types:
            pattern_score += 25  # Institutional activity
        
        if "multi_strike_sweep" in pattern_types:
            pattern_score += 20  # Aggressive positioning
        
        pattern_score = min(100, pattern_score)
        
        # Signal strength based on volume anomaly
        signal_strength = min(100, int(unusual_patterns["max_volume_multiple"] * 15))
        
        # Technical confirmation
        technical_score = 50
        if tech_data and tech_data.rsi_14:
            rsi = float(tech_data.rsi_14)
            if signal_direction == SignalDirection.BUY and rsi < 50:
                technical_score = 70
            elif signal_direction == SignalDirection.SELL and rsi > 50:
                technical_score = 70
        
        # Confidence calculation
        confidence_score = int((
            signal_strength * 0.4 +
            pattern_score * 0.3 +
            directional_score * 0.2 +
            technical_score * 0.1
        ))
        
        # Price targets based on strike concentration
        target_strikes = unusual_patterns["unusual_strikes"]
        
        if signal_direction == SignalDirection.BUY:
            call_strikes = [s for s in target_strikes if s > current_price]
            target_price = min(call_strikes) if call_strikes else int(current_price * 1.03)
            stop_loss_price = int(current_price * 0.97)
        else:
            put_strikes = [s for s in target_strikes if s < current_price]
            target_price = max(put_strikes) if put_strikes else int(current_price * 0.97)
            stop_loss_price = int(current_price * 1.03)
        
        # Institutional indicators
        institutional_indicators = []
        if "block_trades" in pattern_types:
            institutional_indicators.append("Large block trades detected")
        if unusual_patterns["time_concentration"] > 2:
            institutional_indicators.append("Time-concentrated activity")
        if unusual_patterns["max_volume_multiple"] > 10:
            institutional_indicators.append("Extreme volume anomaly")
        
        return {
            "signal_direction": signal_direction,
            "signal_strength": signal_strength,
            "confidence_score": confidence_score,
            "pattern_score": pattern_score,
            "directional_score": directional_score,
            "target_price": target_price,
            "stop_loss_price": stop_loss_price,
            "institutional_indicators": institutional_indicators
        }
    
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
        
        if self.volume_multiple_threshold <= 1.0:
            errors.append("volume_multiple_threshold must be greater than 1.0")
        
        if self.min_absolute_volume <= 0:
            errors.append("min_absolute_volume must be positive")
        
        if self.historical_periods <= 0:
            errors.append("historical_periods must be positive")
        
        if len(self.dte_range) != 2 or self.dte_range[0] >= self.dte_range[1]:
            errors.append("dte_range must be [min, max] with min < max")
        
        if self.block_size_threshold <= 0:
            errors.append("block_size_threshold must be positive")
        
        return len(errors) == 0, errors