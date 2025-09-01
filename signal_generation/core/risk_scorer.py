"""
Risk Scoring System
Author: Claude Code (System Architect)
Version: 1.0

Advanced risk scoring system that calibrates and refines risk scores based on:
- Market conditions and volatility
- Asset-specific risk factors
- Cross-asset correlations
- Historical performance data
- Real-time market stress indicators
"""

import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
from decimal import Decimal
from enum import Enum
from dataclasses import dataclass
from uuid import UUID
from collections import defaultdict

from data_models.python.core_models import Asset, MarketData, TechnicalIndicators
from data_models.python.signal_models import Signal, SignalCategory, MarketRegime


class RiskFactorType(str, Enum):
    """Types of risk factors"""
    MARKET_RISK = "market_risk"
    LIQUIDITY_RISK = "liquidity_risk"  
    VOLATILITY_RISK = "volatility_risk"
    CONCENTRATION_RISK = "concentration_risk"
    CORRELATION_RISK = "correlation_risk"
    TIME_DECAY_RISK = "time_decay_risk"
    EVENT_RISK = "event_risk"


@dataclass
class RiskFactor:
    """Individual risk factor assessment"""
    factor_type: RiskFactorType
    score: int  # 0-100
    weight: float  # 0-1
    description: str
    data_source: str
    confidence: float = 0.8  # 0-1


@dataclass
class RiskAssessment:
    """Comprehensive risk assessment result"""
    signal_id: UUID
    original_risk_score: int
    calibrated_risk_score: int
    risk_factors: List[RiskFactor]
    market_regime: Optional[str]
    volatility_percentile: Optional[int]
    liquidity_score: Optional[int]
    correlation_risk_score: Optional[int]
    assessment_timestamp: datetime
    confidence_level: float  # 0-1


class RiskScorer:
    """
    Advanced risk scoring and calibration system
    
    Provides multi-factor risk assessment that considers:
    1. Base asset class risk
    2. Current market conditions  
    3. Volatility environment
    4. Liquidity conditions
    5. Cross-asset correlations
    6. Time-sensitive factors
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger("risk_scorer")
        
        # Risk model parameters
        self.volatility_lookback_days = self.config.get("volatility_lookback_days", 20)
        self.correlation_lookback_days = self.config.get("correlation_lookback_days", 30)
        self.min_volume_threshold = self.config.get("min_volume_threshold", 10000)
        
        # Asset class base risk mappings
        self.asset_class_base_risk = {
            "daily_options": {"min": 70, "max": 95, "base": 80},
            "weekly_options": {"min": 60, "max": 85, "base": 70},
            "monthly_options": {"min": 50, "max": 75, "base": 60},
            "large_cap_stocks": {"min": 25, "max": 60, "base": 40},
            "mid_cap_stocks": {"min": 35, "max": 70, "base": 50},
            "small_cap_stocks": {"min": 45, "max": 80, "base": 60},
            "growth_stocks": {"min": 40, "max": 75, "base": 55},
            "value_stocks": {"min": 25, "max": 55, "base": 35},
            "dividend_stocks": {"min": 20, "max": 50, "base": 30},
            "sector_etfs": {"min": 30, "max": 65, "base": 45},
            "broad_market_etfs": {"min": 20, "max": 50, "base": 30},
            "commodity_etfs": {"min": 40, "max": 75, "base": 55},
            "international_etfs": {"min": 35, "max": 70, "base": 50},
            "treasury_bonds": {"min": 5, "max": 25, "base": 15},
            "corporate_bonds": {"min": 15, "max": 40, "base": 25},
            "high_yield_bonds": {"min": 25, "max": 50, "base": 35},
            "municipal_bonds": {"min": 10, "max": 30, "base": 20},
            "tbills": {"min": 0, "max": 10, "base": 5},
            "cds": {"min": 2, "max": 15, "base": 8},
            "money_market": {"min": 0, "max": 8, "base": 3}
        }
        
        # Market regime risk multipliers
        self.regime_multipliers = {
            "bull_market": 0.9,
            "bear_market": 1.3,
            "sideways": 1.0,
            "high_vol": 1.4,
            "low_vol": 0.8,
            "crisis": 1.8,
            "recovery": 1.1
        }
        
        # Performance tracking
        self._assessments_count = 0
        self._calibration_adjustments = []
    
    async def assess_risk(self,
                         signal: Signal,
                         asset: Asset,
                         market_data: MarketData,
                         technical_indicators: Optional[TechnicalIndicators] = None,
                         market_regime_data: Optional[Dict[str, Any]] = None,
                         historical_data: Optional[List[MarketData]] = None) -> RiskAssessment:
        """
        Perform comprehensive risk assessment and calibration
        
        Args:
            signal: Signal to assess
            asset: Asset information
            market_data: Current market data
            technical_indicators: Technical indicators
            market_regime_data: Market regime information
            historical_data: Historical price data for volatility calculation
            
        Returns:
            RiskAssessment with calibrated risk score and detailed factors
        """
        
        try:
            original_risk_score = signal.risk_score
            
            # Calculate individual risk factors
            risk_factors = []
            
            # 1. Market risk assessment
            market_risk = await self._assess_market_risk(asset, market_data, market_regime_data)
            risk_factors.append(market_risk)
            
            # 2. Liquidity risk assessment
            liquidity_risk = await self._assess_liquidity_risk(asset, market_data)
            risk_factors.append(liquidity_risk)
            
            # 3. Volatility risk assessment
            volatility_risk = await self._assess_volatility_risk(
                asset, market_data, technical_indicators, historical_data
            )
            risk_factors.append(volatility_risk)
            
            # 4. Time decay risk (for options)
            if "option" in asset.name.lower() or signal.category == SignalCategory.OPTIONS:
                time_decay_risk = await self._assess_time_decay_risk(signal, asset)
                risk_factors.append(time_decay_risk)
            
            # 5. Concentration risk
            concentration_risk = await self._assess_concentration_risk(signal, asset)
            risk_factors.append(concentration_risk)
            
            # Calculate calibrated risk score
            calibrated_score = self._calculate_calibrated_risk_score(
                original_risk_score, risk_factors, market_regime_data
            )
            
            # Extract key metrics
            volatility_percentile = self._calculate_volatility_percentile(volatility_risk)
            liquidity_score = int(100 - liquidity_risk.score)  # Invert for liquidity score
            
            # Calculate overall confidence
            confidence_level = self._calculate_confidence_level(risk_factors)
            
            assessment = RiskAssessment(
                signal_id=signal.id,
                original_risk_score=original_risk_score,
                calibrated_risk_score=calibrated_score,
                risk_factors=risk_factors,
                market_regime=market_regime_data.get("primary_regime") if market_regime_data else None,
                volatility_percentile=volatility_percentile,
                liquidity_score=liquidity_score,
                correlation_risk_score=None,  # TODO: Implement correlation analysis
                assessment_timestamp=datetime.utcnow(),
                confidence_level=confidence_level
            )
            
            # Track performance
            self._assessments_count += 1
            adjustment = calibrated_score - original_risk_score
            self._calibration_adjustments.append(adjustment)
            
            self.logger.info(
                f"Risk assessment for signal {signal.id}: "
                f"{original_risk_score} -> {calibrated_score} (adj: {adjustment:+d})"
            )
            
            return assessment
            
        except Exception as e:
            self.logger.error(f"Risk assessment failed for signal {signal.id}: {str(e)}", exc_info=True)
            
            # Return minimal assessment on failure
            return RiskAssessment(
                signal_id=signal.id,
                original_risk_score=signal.risk_score,
                calibrated_risk_score=min(100, signal.risk_score + 20),  # Conservative increase
                risk_factors=[],
                market_regime=None,
                volatility_percentile=None,
                liquidity_score=None,
                correlation_risk_score=None,
                assessment_timestamp=datetime.utcnow(),
                confidence_level=0.3
            )
    
    async def _assess_market_risk(self,
                                 asset: Asset,
                                 market_data: MarketData,
                                 market_regime_data: Optional[Dict[str, Any]]) -> RiskFactor:
        """Assess overall market risk conditions"""
        
        base_score = 50  # Neutral starting point
        
        # Market regime impact
        if market_regime_data:
            regime = market_regime_data.get("primary_regime", "sideways")
            stress_index = market_regime_data.get("market_stress_index", 50)
            
            # Adjust based on regime
            regime_adjustment = self.regime_multipliers.get(regime, 1.0)
            regime_impact = int((regime_adjustment - 1.0) * 50)
            
            # Stress index impact
            stress_impact = max(0, stress_index - 50) // 2  # Scale stress above 50
            
            base_score = min(100, base_score + regime_impact + stress_impact)
        
        # Market hours adjustment
        market_time = market_data.timestamp if hasattr(market_data, 'timestamp') else datetime.utcnow()
        if market_time.hour < 10 or market_time.hour > 15:
            base_score += 10  # Higher risk during low liquidity hours
        
        return RiskFactor(
            factor_type=RiskFactorType.MARKET_RISK,
            score=min(100, max(0, base_score)),
            weight=0.25,
            description=f"Market conditions and regime assessment",
            data_source="market_regime_analysis",
            confidence=0.8
        )
    
    async def _assess_liquidity_risk(self, asset: Asset, market_data: MarketData) -> RiskFactor:
        """Assess liquidity risk based on volume and spreads"""
        
        base_score = 30  # Start with moderate liquidity risk
        
        # Volume analysis
        current_volume = getattr(market_data, 'volume', 0)
        avg_volume = asset.avg_volume_30d or 100000  # Default if not available
        
        volume_ratio = current_volume / max(avg_volume, 1)
        
        if volume_ratio < 0.3:
            base_score += 40  # Very low volume = high liquidity risk
        elif volume_ratio < 0.7:
            base_score += 20  # Below average volume
        elif volume_ratio > 3.0:
            base_score += 10  # Very high volume can indicate instability
        
        # Market cap adjustment
        if asset.market_cap:
            if asset.market_cap < 50_000_000_000:  # <$500M = small cap
                base_score += 30
            elif asset.market_cap < 200_000_000_000:  # <$2B = mid cap
                base_score += 15
            elif asset.market_cap < 1_000_000_000_000:  # <$10B = large cap
                base_score += 5
            # Mega cap (>$10B) gets no adjustment
        
        # Bid-ask spread impact (if available)
        if (hasattr(market_data, 'bid_price_cents') and hasattr(market_data, 'ask_price_cents') and
            market_data.bid_price_cents and market_data.ask_price_cents):
            
            spread_pct = (market_data.ask_price_cents - market_data.bid_price_cents) / market_data.ask_price_cents
            
            if spread_pct > 0.02:  # >2% spread
                base_score += 25
            elif spread_pct > 0.01:  # >1% spread
                base_score += 15
            elif spread_pct > 0.005:  # >0.5% spread
                base_score += 5
        
        return RiskFactor(
            factor_type=RiskFactorType.LIQUIDITY_RISK,
            score=min(100, max(0, base_score)),
            weight=0.20,
            description=f"Liquidity assessment based on volume ratio {volume_ratio:.2f}x",
            data_source="volume_and_spread_analysis",
            confidence=0.75
        )
    
    async def _assess_volatility_risk(self,
                                    asset: Asset,
                                    market_data: MarketData,
                                    technical_indicators: Optional[TechnicalIndicators],
                                    historical_data: Optional[List[MarketData]]) -> RiskFactor:
        """Assess volatility risk using multiple measures"""
        
        base_score = 40  # Start with moderate volatility risk
        
        # ATR-based volatility (if available)
        if technical_indicators and technical_indicators.atr_14:
            atr = technical_indicators.atr_14
            current_price = market_data.close_price_cents
            
            atr_pct = (atr / current_price) * 100
            
            if atr_pct > 5.0:  # >5% daily range
                base_score += 40
            elif atr_pct > 3.0:  # >3% daily range  
                base_score += 25
            elif atr_pct > 2.0:  # >2% daily range
                base_score += 15
            elif atr_pct > 1.0:  # >1% daily range
                base_score += 5
        
        # Historical volatility (if available)
        if historical_data and len(historical_data) >= 20:
            returns = []
            for i in range(1, len(historical_data)):
                prev_price = historical_data[i-1].close_price_cents
                curr_price = historical_data[i].close_price_cents
                daily_return = (curr_price - prev_price) / prev_price
                returns.append(daily_return)
            
            if returns:
                volatility = np.std(returns) * np.sqrt(252) * 100  # Annualized volatility %
                
                if volatility > 40:  # >40% annualized
                    base_score += 35
                elif volatility > 30:  # >30% annualized
                    base_score += 25  
                elif volatility > 20:  # >20% annualized
                    base_score += 15
                elif volatility > 15:  # >15% annualized
                    base_score += 5
        
        # Bollinger Band squeeze/expansion (if available)
        if (technical_indicators and technical_indicators.bollinger_upper and 
            technical_indicators.bollinger_lower and technical_indicators.bollinger_middle):
            
            bb_upper = technical_indicators.bollinger_upper
            bb_lower = technical_indicators.bollinger_lower  
            bb_middle = technical_indicators.bollinger_middle
            
            if bb_middle > 0:
                bb_width_pct = ((bb_upper - bb_lower) / bb_middle) * 100
                
                if bb_width_pct > 20:  # Very wide bands = high volatility
                    base_score += 20
                elif bb_width_pct < 5:  # Very narrow bands = compression (potential expansion)
                    base_score += 15
        
        return RiskFactor(
            factor_type=RiskFactorType.VOLATILITY_RISK,
            score=min(100, max(0, base_score)),
            weight=0.25,
            description="Multi-factor volatility risk assessment",
            data_source="atr_historical_bollinger_analysis",
            confidence=0.85 if historical_data else 0.65
        )
    
    async def _assess_time_decay_risk(self, signal: Signal, asset: Asset) -> RiskFactor:
        """Assess time decay risk for options signals"""
        
        base_score = 20  # Options have inherent time decay
        
        # Holding period impact
        if signal.recommended_holding_period_hours:
            hours = signal.recommended_holding_period_hours
            
            if hours <= 4:  # Very short term
                base_score += 40  # High theta risk
            elif hours <= 24:  # Intraday
                base_score += 25
            elif hours <= 168:  # One week
                base_score += 15
            elif hours <= 720:  # One month
                base_score += 5
        
        # Signal expiration vs holding period
        if signal.valid_until and signal.recommended_holding_period_hours:
            time_to_expiry = (signal.valid_until - datetime.utcnow()).total_seconds() / 3600
            holding_period = signal.recommended_holding_period_hours
            
            if time_to_expiry < holding_period:
                base_score += 30  # Signal may expire before position closes
        
        # Options-specific risk (simplified - would analyze actual Greeks)
        if "option" in asset.name.lower():
            base_score += 15  # Base options time decay risk
        
        return RiskFactor(
            factor_type=RiskFactorType.TIME_DECAY_RISK,
            score=min(100, max(0, base_score)),
            weight=0.15,
            description="Time decay risk for options positions",
            data_source="time_decay_analysis",
            confidence=0.7
        )
    
    async def _assess_concentration_risk(self, signal: Signal, asset: Asset) -> RiskFactor:
        """Assess concentration risk based on position sizing"""
        
        base_score = 20  # Start with low concentration risk
        
        # Position size impact
        if signal.recommended_position_size_pct:
            position_size = float(signal.recommended_position_size_pct)
            
            if position_size > 25:  # >25% of portfolio
                base_score += 60
            elif position_size > 15:  # >15% of portfolio
                base_score += 40
            elif position_size > 10:  # >10% of portfolio
                base_score += 25
            elif position_size > 5:  # >5% of portfolio
                base_score += 10
        
        # Max position size check
        if signal.max_position_size_pct:
            max_size = float(signal.max_position_size_pct)
            if max_size > 30:
                base_score += 30  # High maximum concentration
        
        # Asset-specific concentration risk
        if asset.market_cap and asset.market_cap < 100_000_000_000:  # <$1B
            base_score += 15  # Small cap concentration risk
        
        return RiskFactor(
            factor_type=RiskFactorType.CONCENTRATION_RISK,
            score=min(100, max(0, base_score)),
            weight=0.15,
            description="Portfolio concentration risk assessment",
            data_source="position_sizing_analysis",
            confidence=0.8
        )
    
    def _calculate_calibrated_risk_score(self,
                                       original_score: int,
                                       risk_factors: List[RiskFactor],
                                       market_regime_data: Optional[Dict[str, Any]]) -> int:
        """Calculate calibrated risk score using weighted risk factors"""
        
        # Calculate weighted risk adjustment
        total_adjustment = 0.0
        total_weight = 0.0
        
        for factor in risk_factors:
            # Calculate adjustment from neutral (50)
            factor_adjustment = (factor.score - 50) * factor.weight * factor.confidence
            total_adjustment += factor_adjustment
            total_weight += factor.weight * factor.confidence
        
        # Normalize adjustment
        if total_weight > 0:
            normalized_adjustment = total_adjustment / total_weight
        else:
            normalized_adjustment = 0
        
        # Apply adjustment to original score
        calibrated_score = original_score + int(normalized_adjustment)
        
        # Market regime final adjustment
        if market_regime_data:
            regime = market_regime_data.get("primary_regime", "sideways")
            regime_multiplier = self.regime_multipliers.get(regime, 1.0)
            
            if regime_multiplier != 1.0:
                adjustment = (calibrated_score - 50) * (regime_multiplier - 1.0)
                calibrated_score += int(adjustment)
        
        # Ensure score stays within bounds
        calibrated_score = max(0, min(100, calibrated_score))
        
        return calibrated_score
    
    def _calculate_volatility_percentile(self, volatility_risk: RiskFactor) -> int:
        """Convert volatility risk score to percentile"""
        # Simple mapping - would use historical distributions in production
        score = volatility_risk.score
        
        if score >= 90:
            return 95  # Very high volatility
        elif score >= 80:
            return 85  # High volatility
        elif score >= 70:
            return 75  # Above average volatility
        elif score >= 60:
            return 65  # Slightly above average
        elif score >= 40:
            return 50  # Average volatility
        elif score >= 30:
            return 35  # Below average volatility
        elif score >= 20:
            return 25  # Low volatility
        else:
            return 15  # Very low volatility
    
    def _calculate_confidence_level(self, risk_factors: List[RiskFactor]) -> float:
        """Calculate overall confidence in risk assessment"""
        
        if not risk_factors:
            return 0.3  # Low confidence without factors
        
        # Weight average of individual factor confidences
        total_confidence = sum(factor.confidence * factor.weight for factor in risk_factors)
        total_weight = sum(factor.weight for factor in risk_factors)
        
        if total_weight > 0:
            avg_confidence = total_confidence / total_weight
        else:
            avg_confidence = 0.5
        
        # Bonus for having multiple factors
        factor_bonus = min(0.2, len(risk_factors) * 0.05)
        
        return min(1.0, avg_confidence + factor_bonus)
    
    async def assess_portfolio_risk(self,
                                   signals: List[Signal],
                                   assets: Dict[UUID, Asset],
                                   market_data: Dict[UUID, MarketData],
                                   max_portfolio_risk: int = 80) -> Dict[str, Any]:
        """Assess portfolio-level risk from multiple signals"""
        
        if not signals:
            return {
                "portfolio_risk_score": 0,
                "risk_concentration": 0,
                "correlation_risk": 0,
                "diversification_score": 100,
                "recommendations": []
            }
        
        # Assess individual signals
        individual_assessments = []
        for signal in signals:
            asset = assets.get(signal.asset_id)
            mkt_data = market_data.get(signal.asset_id)
            
            if asset and mkt_data:
                assessment = await self.assess_risk(signal, asset, mkt_data)
                individual_assessments.append(assessment)
        
        if not individual_assessments:
            return {
                "portfolio_risk_score": max_portfolio_risk,
                "risk_concentration": 100,
                "correlation_risk": 100,
                "diversification_score": 0,
                "recommendations": ["No valid risk assessments available"]
            }
        
        # Calculate portfolio metrics
        avg_risk = sum(assessment.calibrated_risk_score for assessment in individual_assessments) / len(individual_assessments)
        max_risk = max(assessment.calibrated_risk_score for assessment in individual_assessments)
        
        # Risk concentration (how much risk is in top positions)
        risk_scores = [assessment.calibrated_risk_score for assessment in individual_assessments]
        risk_scores.sort(reverse=True)
        
        top_3_risk = sum(risk_scores[:3]) if len(risk_scores) >= 3 else sum(risk_scores)
        total_risk = sum(risk_scores)
        risk_concentration = (top_3_risk / max(total_risk, 1)) * 100
        
        # Diversification score (inverse of concentration)
        diversification_score = max(0, 100 - risk_concentration)
        
        # Portfolio risk score (weighted average with concentration penalty)
        portfolio_risk_score = int(avg_risk * 0.7 + risk_concentration * 0.3)
        
        # Generate recommendations
        recommendations = []
        
        if portfolio_risk_score > max_portfolio_risk:
            recommendations.append(f"Portfolio risk {portfolio_risk_score} exceeds maximum {max_portfolio_risk}")
        
        if risk_concentration > 60:
            recommendations.append("High risk concentration detected - consider diversification")
        
        if max_risk > 90:
            recommendations.append("Some positions have very high risk scores - review individual positions")
        
        if diversification_score < 40:
            recommendations.append("Low diversification - consider spreading risk across more assets")
        
        return {
            "portfolio_risk_score": portfolio_risk_score,
            "average_position_risk": int(avg_risk),
            "maximum_position_risk": max_risk,
            "risk_concentration": int(risk_concentration),
            "correlation_risk": 50,  # TODO: Implement correlation analysis
            "diversification_score": int(diversification_score),
            "total_positions": len(individual_assessments),
            "high_risk_positions": len([a for a in individual_assessments if a.calibrated_risk_score > 75]),
            "recommendations": recommendations
        }
    
    def get_risk_scorer_statistics(self) -> Dict[str, Any]:
        """Get risk scorer performance statistics"""
        
        avg_adjustment = (sum(self._calibration_adjustments) / 
                         max(len(self._calibration_adjustments), 1))
        
        positive_adjustments = len([adj for adj in self._calibration_adjustments if adj > 0])
        negative_adjustments = len([adj for adj in self._calibration_adjustments if adj < 0])
        
        return {
            "total_assessments": self._assessments_count,
            "average_risk_adjustment": avg_adjustment,
            "positive_adjustments": positive_adjustments,
            "negative_adjustments": negative_adjustments,
            "no_adjustments": len(self._calibration_adjustments) - positive_adjustments - negative_adjustments,
            "adjustment_distribution": {
                "mean": avg_adjustment,
                "std": np.std(self._calibration_adjustments) if self._calibration_adjustments else 0,
                "min": min(self._calibration_adjustments) if self._calibration_adjustments else 0,
                "max": max(self._calibration_adjustments) if self._calibration_adjustments else 0
            }
        }
    
    def reset_statistics(self) -> None:
        """Reset risk scorer statistics"""
        self._assessments_count = 0
        self._calibration_adjustments = []
        
        self.logger.info("Risk scorer statistics reset")