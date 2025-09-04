"""
Enhanced signal validation and quality scoring
Validates trading signals for accuracy, reliability, and risk assessment
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from enum import Enum

from app.models.signal import SignalCreate, Signal, AssetClass, SignalType, TimeFrame
from app.models.market import MarketDataPoint
from app.core.cache import get_market_cache, get_signal_cache
from app.services.risk_scoring import RiskScoringService

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Signal validation strictness levels"""
    BASIC = "basic"        # Basic checks only
    STANDARD = "standard"  # Standard validation
    STRICT = "strict"      # Strict validation
    PREMIUM = "premium"    # Premium validation with ML scoring


class ValidationResult(Enum):
    """Validation result types"""
    APPROVED = "approved"
    REJECTED = "rejected" 
    WARNING = "warning"
    REVIEW_REQUIRED = "review_required"


@dataclass
class ValidationIssue:
    """Signal validation issue"""
    severity: str  # "error", "warning", "info"
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None


@dataclass
class QualityMetrics:
    """Signal quality metrics"""
    technical_score: float      # 0-100 based on technical analysis
    risk_consistency: float     # 0-100 how well risk aligns with asset class
    market_context_score: float # 0-100 considering market conditions
    historical_accuracy: float  # 0-100 based on similar signals
    confidence_calibration: float # 0-100 how well calibrated confidence is
    overall_quality: float      # 0-100 weighted average
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "technical_score": self.technical_score,
            "risk_consistency": self.risk_consistency,
            "market_context_score": self.market_context_score,
            "historical_accuracy": self.historical_accuracy,
            "confidence_calibration": self.confidence_calibration,
            "overall_quality": self.overall_quality
        }


@dataclass
class ValidationSummary:
    """Complete validation summary"""
    result: ValidationResult
    quality_metrics: QualityMetrics
    issues: List[ValidationIssue]
    recommendations: List[str]
    adjusted_confidence: Optional[float] = None
    adjusted_risk_score: Optional[int] = None


class SignalValidator:
    """Enhanced signal validation system"""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.validation_level = validation_level
        self.risk_scorer = RiskScoringService()
        
        # Quality score weights
        self.quality_weights = {
            "technical_score": 0.30,
            "risk_consistency": 0.25,
            "market_context_score": 0.20,
            "historical_accuracy": 0.15,
            "confidence_calibration": 0.10
        }
        
        # Validation thresholds by level
        self.thresholds = {
            ValidationLevel.BASIC: {
                "min_confidence": 0.10,
                "min_quality": 20.0,
                "max_risk_deviation": 40
            },
            ValidationLevel.STANDARD: {
                "min_confidence": 0.25,
                "min_quality": 40.0,
                "max_risk_deviation": 25
            },
            ValidationLevel.STRICT: {
                "min_confidence": 0.50,
                "min_quality": 60.0,
                "max_risk_deviation": 15
            },
            ValidationLevel.PREMIUM: {
                "min_confidence": 0.65,
                "min_quality": 75.0,
                "max_risk_deviation": 10
            }
        }
    
    async def validate_signal(
        self,
        signal: SignalCreate,
        market_data: Optional[MarketDataPoint] = None,
        historical_signals: Optional[List[Signal]] = None
    ) -> ValidationSummary:
        """Comprehensive signal validation"""
        
        issues = []
        recommendations = []
        
        # Run validation checks
        basic_issues = await self._basic_validation(signal)
        issues.extend(basic_issues)
        
        if self.validation_level in [ValidationLevel.STANDARD, ValidationLevel.STRICT, ValidationLevel.PREMIUM]:
            advanced_issues = await self._advanced_validation(signal, market_data)
            issues.extend(advanced_issues)
        
        if self.validation_level in [ValidationLevel.STRICT, ValidationLevel.PREMIUM]:
            context_issues = await self._context_validation(signal, market_data)
            issues.extend(context_issues)
        
        # Calculate quality metrics
        quality_metrics = await self._calculate_quality_metrics(
            signal, market_data, historical_signals
        )
        
        # Determine validation result
        result = self._determine_validation_result(signal, quality_metrics, issues)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(signal, quality_metrics, issues)
        
        # Adjust confidence and risk if needed
        adjusted_confidence, adjusted_risk = self._calculate_adjustments(
            signal, quality_metrics, issues
        )
        
        return ValidationSummary(
            result=result,
            quality_metrics=quality_metrics,
            issues=issues,
            recommendations=recommendations,
            adjusted_confidence=adjusted_confidence,
            adjusted_risk_score=adjusted_risk
        )
    
    async def _basic_validation(self, signal: SignalCreate) -> List[ValidationIssue]:
        """Basic signal validation checks"""
        issues = []
        
        # Required fields validation
        if not signal.symbol:
            issues.append(ValidationIssue(
                severity="error",
                code="MISSING_SYMBOL",
                message="Signal symbol is required"
            ))
        
        if signal.entry_price <= 0:
            issues.append(ValidationIssue(
                severity="error", 
                code="INVALID_PRICE",
                message="Entry price must be positive"
            ))
        
        if not (0 <= signal.confidence <= 1):
            issues.append(ValidationIssue(
                severity="error",
                code="INVALID_CONFIDENCE",
                message="Confidence must be between 0 and 1"
            ))
        
        if not (0 <= signal.risk_score <= 100):
            issues.append(ValidationIssue(
                severity="error",
                code="INVALID_RISK_SCORE",
                message="Risk score must be between 0 and 100"
            ))
        
        # Price relationship validation
        if signal.signal_type == SignalType.BUY:
            if signal.target_price and signal.target_price <= signal.entry_price:
                issues.append(ValidationIssue(
                    severity="error",
                    code="INVALID_TARGET_BUY",
                    message="Target price must be above entry price for buy signals"
                ))
            
            if signal.stop_loss and signal.stop_loss >= signal.entry_price:
                issues.append(ValidationIssue(
                    severity="error",
                    code="INVALID_STOP_BUY",
                    message="Stop loss must be below entry price for buy signals"
                ))
        
        elif signal.signal_type == SignalType.SELL:
            if signal.target_price and signal.target_price >= signal.entry_price:
                issues.append(ValidationIssue(
                    severity="error",
                    code="INVALID_TARGET_SELL",
                    message="Target price must be below entry price for sell signals"
                ))
            
            if signal.stop_loss and signal.stop_loss <= signal.entry_price:
                issues.append(ValidationIssue(
                    severity="error",
                    code="INVALID_STOP_SELL",
                    message="Stop loss must be above entry price for sell signals"
                ))
        
        # Expiration validation
        if signal.expires_at and signal.expires_at <= datetime.now(timezone.utc):
            issues.append(ValidationIssue(
                severity="error",
                code="EXPIRED_SIGNAL",
                message="Signal expiration date is in the past"
            ))
        
        # Risk-reward validation
        if signal.target_price and signal.stop_loss:
            reward = abs(signal.target_price - signal.entry_price)
            risk = abs(signal.entry_price - signal.stop_loss)
            
            if risk > 0:
                risk_reward_ratio = reward / risk
                if risk_reward_ratio < 0.5:  # Less than 1:2 risk/reward
                    issues.append(ValidationIssue(
                        severity="warning",
                        code="POOR_RISK_REWARD",
                        message=f"Poor risk/reward ratio: {risk_reward_ratio:.2f}",
                        details={"risk_reward_ratio": risk_reward_ratio}
                    ))
        
        return issues
    
    async def _advanced_validation(
        self,
        signal: SignalCreate,
        market_data: Optional[MarketDataPoint]
    ) -> List[ValidationIssue]:
        """Advanced validation with market context"""
        issues = []
        
        if not market_data:
            issues.append(ValidationIssue(
                severity="warning",
                code="NO_MARKET_DATA",
                message="No current market data available for validation"
            ))
            return issues
        
        # Price deviation check
        price_deviation = abs(signal.entry_price - market_data.price) / market_data.price
        if price_deviation > 0.05:  # More than 5% deviation
            issues.append(ValidationIssue(
                severity="warning",
                code="PRICE_DEVIATION",
                message=f"Signal price deviates {price_deviation:.1%} from current market price",
                details={"deviation_pct": price_deviation}
            ))
        
        # Volume validation for liquidity
        if market_data.volume and market_data.volume < 10000:  # Low volume threshold
            issues.append(ValidationIssue(
                severity="warning",
                code="LOW_VOLUME",
                message="Low trading volume may impact signal execution",
                details={"volume": market_data.volume}
            ))
        
        # Volatility check
        if market_data.high and market_data.low:
            daily_range = (market_data.high - market_data.low) / market_data.price
            
            if daily_range > 0.10:  # More than 10% daily range
                issues.append(ValidationIssue(
                    severity="info",
                    code="HIGH_VOLATILITY",
                    message=f"High volatility detected ({daily_range:.1%})",
                    details={"daily_range_pct": daily_range}
                ))
        
        return issues
    
    async def _context_validation(
        self,
        signal: SignalCreate,
        market_data: Optional[MarketDataPoint]
    ) -> List[ValidationIssue]:
        """Context-aware validation"""
        issues = []
        
        # Market hours validation
        current_time = datetime.now(timezone.utc)
        is_market_hours = self._is_market_hours(current_time)
        
        if not is_market_hours and signal.timeframe == TimeFrame.INTRADAY:
            issues.append(ValidationIssue(
                severity="warning",
                code="OFF_MARKET_HOURS",
                message="Intraday signal generated outside market hours"
            ))
        
        # Asset class specific validation
        asset_class_issues = self._validate_asset_class_specific(signal)
        issues.extend(asset_class_issues)
        
        return issues
    
    def _validate_asset_class_specific(self, signal: SignalCreate) -> List[ValidationIssue]:
        """Asset class specific validation rules"""
        issues = []
        
        if signal.asset_class == AssetClass.DAILY_OPTIONS:
            # Options-specific validation
            if signal.expires_at:
                time_to_expiry = (signal.expires_at - datetime.now(timezone.utc)).total_seconds() / 3600
                
                if time_to_expiry < 1:  # Less than 1 hour to expiry
                    issues.append(ValidationIssue(
                        severity="error",
                        code="OPTION_NEAR_EXPIRY",
                        message=f"Option expires in {time_to_expiry:.1f} hours - too risky"
                    ))
                elif time_to_expiry < 4:  # Less than 4 hours
                    issues.append(ValidationIssue(
                        severity="warning",
                        code="OPTION_SHORT_EXPIRY",
                        message=f"Option expires in {time_to_expiry:.1f} hours"
                    ))
            
            # High risk score validation for options
            if signal.risk_score < 70:
                issues.append(ValidationIssue(
                    severity="warning",
                    code="LOW_OPTIONS_RISK",
                    message="Options signals should typically have risk scores >= 70"
                ))
        
        elif signal.asset_class == AssetClass.SAFE_ASSETS:
            # Safe assets should have low risk scores
            if signal.risk_score > 30:
                issues.append(ValidationIssue(
                    severity="warning",
                    code="HIGH_SAFE_ASSET_RISK",
                    message="Safe assets should typically have risk scores <= 30"
                ))
            
            # Conservative profit targets for safe assets
            if signal.target_price:
                expected_return = abs(signal.target_price - signal.entry_price) / signal.entry_price
                if expected_return > 0.05:  # More than 5% return
                    issues.append(ValidationIssue(
                        severity="warning",
                        code="HIGH_SAFE_ASSET_RETURN",
                        message=f"High expected return ({expected_return:.1%}) for safe asset"
                    ))
        
        return issues
    
    async def _calculate_quality_metrics(
        self,
        signal: SignalCreate,
        market_data: Optional[MarketDataPoint],
        historical_signals: Optional[List[Signal]]
    ) -> QualityMetrics:
        """Calculate comprehensive quality metrics"""
        
        # Technical score based on signal completeness and validity
        technical_score = self._calculate_technical_score(signal)
        
        # Risk consistency with asset class
        risk_consistency = self._calculate_risk_consistency(signal)
        
        # Market context score
        market_context_score = self._calculate_market_context_score(signal, market_data)
        
        # Historical accuracy based on similar signals
        historical_accuracy = self._calculate_historical_accuracy(signal, historical_signals)
        
        # Confidence calibration
        confidence_calibration = self._calculate_confidence_calibration(signal)
        
        # Overall quality as weighted average
        overall_quality = (
            technical_score * self.quality_weights["technical_score"] +
            risk_consistency * self.quality_weights["risk_consistency"] +
            market_context_score * self.quality_weights["market_context_score"] +
            historical_accuracy * self.quality_weights["historical_accuracy"] +
            confidence_calibration * self.quality_weights["confidence_calibration"]
        )
        
        return QualityMetrics(
            technical_score=technical_score,
            risk_consistency=risk_consistency,
            market_context_score=market_context_score,
            historical_accuracy=historical_accuracy,
            confidence_calibration=confidence_calibration,
            overall_quality=overall_quality
        )
    
    def _calculate_technical_score(self, signal: SignalCreate) -> float:
        """Calculate technical analysis quality score"""
        score = 50.0  # Base score
        
        # Bonus for having target price
        if signal.target_price:
            score += 15.0
        
        # Bonus for having stop loss
        if signal.stop_loss:
            score += 15.0
        
        # Bonus for reasonable risk/reward if both present
        if signal.target_price and signal.stop_loss:
            reward = abs(signal.target_price - signal.entry_price)
            risk = abs(signal.entry_price - signal.stop_loss)
            
            if risk > 0:
                risk_reward_ratio = reward / risk
                if risk_reward_ratio >= 2.0:  # Good risk/reward
                    score += 20.0
                elif risk_reward_ratio >= 1.0:
                    score += 10.0
        
        # Penalty for very short or very long timeframes without justification
        if signal.timeframe == TimeFrame.INTRADAY and not signal.description:
            score -= 10.0
        
        return min(100.0, max(0.0, score))
    
    def _calculate_risk_consistency(self, signal: SignalCreate) -> float:
        """Calculate how well risk score aligns with asset class"""
        
        # Expected risk ranges by asset class
        expected_ranges = {
            AssetClass.DAILY_OPTIONS: (70, 95),
            AssetClass.STOCKS: (30, 80),
            AssetClass.ETFS: (20, 70),
            AssetClass.BONDS: (5, 40),
            AssetClass.SAFE_ASSETS: (0, 20)
        }
        
        expected_min, expected_max = expected_ranges.get(
            signal.asset_class, (0, 100)
        )
        
        if expected_min <= signal.risk_score <= expected_max:
            return 100.0
        
        # Calculate penalty for being outside expected range
        if signal.risk_score < expected_min:
            deviation = expected_min - signal.risk_score
        else:
            deviation = signal.risk_score - expected_max
        
        # Penalty increases with deviation
        penalty = min(100.0, deviation * 2)  # 2 points penalty per point of deviation
        return max(0.0, 100.0 - penalty)
    
    def _calculate_market_context_score(
        self,
        signal: SignalCreate,
        market_data: Optional[MarketDataPoint]
    ) -> float:
        """Calculate market context appropriateness score"""
        
        if not market_data:
            return 50.0  # Neutral score without market data
        
        score = 70.0  # Base score with market data
        
        # Price deviation penalty
        price_deviation = abs(signal.entry_price - market_data.price) / market_data.price
        if price_deviation > 0.05:
            score -= min(30.0, price_deviation * 100)  # Penalty for price deviation
        
        # Volume bonus/penalty
        if market_data.volume:
            if market_data.volume >= 100000:  # Good volume
                score += 15.0
            elif market_data.volume < 10000:  # Poor volume
                score -= 20.0
        
        # Volatility consideration
        if market_data.high and market_data.low and market_data.price:
            daily_range = (market_data.high - market_data.low) / market_data.price
            
            if signal.asset_class == AssetClass.DAILY_OPTIONS:
                # Options benefit from higher volatility
                if daily_range > 0.05:
                    score += 10.0
            else:
                # Other assets penalized for excessive volatility
                if daily_range > 0.10:
                    score -= 15.0
        
        return min(100.0, max(0.0, score))
    
    def _calculate_historical_accuracy(
        self,
        signal: SignalCreate,
        historical_signals: Optional[List[Signal]]
    ) -> float:
        """Calculate historical accuracy based on similar signals"""
        
        if not historical_signals:
            return 50.0  # Neutral score without historical data
        
        # Filter similar signals (same symbol, asset class, signal type)
        similar_signals = [
            s for s in historical_signals
            if (s.symbol == signal.symbol and
                s.asset_class == signal.asset_class and
                s.signal_type == signal.signal_type)
        ]
        
        if not similar_signals:
            return 50.0
        
        # Calculate success rate of similar signals
        # This would need actual tracking of signal outcomes
        # For now, use confidence as a proxy
        avg_confidence = np.mean([s.confidence for s in similar_signals])
        
        # Convert average confidence to accuracy score
        return min(100.0, avg_confidence * 120)  # Boost confidence scores slightly
    
    def _calculate_confidence_calibration(self, signal: SignalCreate) -> float:
        """Calculate how well-calibrated the confidence appears"""
        
        score = 50.0  # Base score
        
        # Reasonable confidence ranges by asset class
        reasonable_ranges = {
            AssetClass.DAILY_OPTIONS: (0.4, 0.9),  # Options can be very confident or uncertain
            AssetClass.STOCKS: (0.5, 0.85),
            AssetClass.ETFS: (0.6, 0.80),
            AssetClass.BONDS: (0.65, 0.80),
            AssetClass.SAFE_ASSETS: (0.70, 0.85)
        }
        
        reasonable_min, reasonable_max = reasonable_ranges.get(
            signal.asset_class, (0.3, 0.9)
        )
        
        if reasonable_min <= signal.confidence <= reasonable_max:
            score += 30.0
        else:
            # Penalty for unreasonable confidence
            if signal.confidence < reasonable_min:
                penalty = (reasonable_min - signal.confidence) * 100
            else:
                penalty = (signal.confidence - reasonable_max) * 100
            
            score -= min(40.0, penalty)
        
        # Bonus for modest confidence (shows awareness of uncertainty)
        if 0.6 <= signal.confidence <= 0.75:
            score += 20.0
        
        return min(100.0, max(0.0, score))
    
    def _determine_validation_result(
        self,
        signal: SignalCreate,
        quality_metrics: QualityMetrics,
        issues: List[ValidationIssue]
    ) -> ValidationResult:
        """Determine overall validation result"""
        
        # Check for blocking errors
        error_issues = [issue for issue in issues if issue.severity == "error"]
        if error_issues:
            return ValidationResult.REJECTED
        
        # Get validation thresholds for current level
        thresholds = self.thresholds[self.validation_level]
        
        # Check minimum requirements
        if signal.confidence < thresholds["min_confidence"]:
            return ValidationResult.REJECTED
        
        if quality_metrics.overall_quality < thresholds["min_quality"]:
            return ValidationResult.REJECTED
        
        # Check for warnings that might require review
        warning_issues = [issue for issue in issues if issue.severity == "warning"]
        
        if self.validation_level in [ValidationLevel.STRICT, ValidationLevel.PREMIUM]:
            if len(warning_issues) > 2:
                return ValidationResult.REVIEW_REQUIRED
            
            if quality_metrics.overall_quality < thresholds["min_quality"] + 10:
                return ValidationResult.WARNING
        
        # Signal passes validation
        return ValidationResult.APPROVED
    
    def _generate_recommendations(
        self,
        signal: SignalCreate,
        quality_metrics: QualityMetrics,
        issues: List[ValidationIssue]
    ) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        # Quality-based recommendations
        if quality_metrics.technical_score < 70:
            if not signal.target_price:
                recommendations.append("Consider adding a target price for better risk management")
            if not signal.stop_loss:
                recommendations.append("Add a stop loss to limit potential losses")
        
        if quality_metrics.risk_consistency < 70:
            recommendations.append(
                f"Risk score ({signal.risk_score}) may not be appropriate for {signal.asset_class.value}"
            )
        
        if quality_metrics.confidence_calibration < 60:
            if signal.confidence > 0.9:
                recommendations.append("Consider if confidence level is realistic")
            elif signal.confidence < 0.3:
                recommendations.append("Low confidence signals may not be actionable")
        
        # Issue-based recommendations
        for issue in issues:
            if issue.code == "POOR_RISK_REWARD":
                recommendations.append("Improve risk/reward ratio by adjusting target or stop loss")
            elif issue.code == "PRICE_DEVIATION":
                recommendations.append("Verify entry price against current market price")
            elif issue.code == "LOW_VOLUME":
                recommendations.append("Consider liquidity constraints for execution")
        
        return recommendations
    
    def _calculate_adjustments(
        self,
        signal: SignalCreate,
        quality_metrics: QualityMetrics,
        issues: List[ValidationIssue]
    ) -> Tuple[Optional[float], Optional[int]]:
        """Calculate confidence and risk adjustments"""
        
        adjusted_confidence = None
        adjusted_risk = None
        
        # Adjust confidence based on quality
        if quality_metrics.overall_quality < 60:
            confidence_adjustment = (60 - quality_metrics.overall_quality) / 100
            adjusted_confidence = max(0.1, signal.confidence - confidence_adjustment)
        
        # Adjust risk score for consistency
        if quality_metrics.risk_consistency < 70:
            expected_ranges = {
                AssetClass.DAILY_OPTIONS: (70, 95),
                AssetClass.STOCKS: (30, 80),
                AssetClass.ETFS: (20, 70),
                AssetClass.BONDS: (5, 40),
                AssetClass.SAFE_ASSETS: (0, 20)
            }
            
            expected_min, expected_max = expected_ranges.get(
                signal.asset_class, (signal.risk_score, signal.risk_score)
            )
            
            # Adjust towards expected range
            if signal.risk_score < expected_min:
                adjusted_risk = expected_min
            elif signal.risk_score > expected_max:
                adjusted_risk = expected_max
        
        return adjusted_confidence, adjusted_risk
    
    def _is_market_hours(self, dt: datetime) -> bool:
        """Check if datetime is during market hours (simplified)"""
        # US market: 9:30 AM - 4:00 PM ET, Mon-Fri
        weekday = dt.weekday()
        if weekday >= 5:  # Weekend
            return False
        
        # Simplified hour check (would need proper timezone handling in production)
        hour = dt.hour
        return 14 <= hour < 21  # Approximate ET hours in UTC
    
    async def batch_validate_signals(
        self,
        signals: List[SignalCreate],
        market_data_map: Optional[Dict[str, MarketDataPoint]] = None
    ) -> Dict[str, ValidationSummary]:
        """Validate multiple signals in batch"""
        
        results = {}
        
        # Process signals concurrently
        tasks = []
        for signal in signals:
            market_data = market_data_map.get(signal.symbol) if market_data_map else None
            task = self.validate_signal(signal, market_data)
            tasks.append((signal.symbol, task))
        
        # Execute all validations
        for symbol, task in tasks:
            try:
                validation_result = await task
                results[symbol] = validation_result
            except Exception as e:
                logger.error(f"Error validating signal for {symbol}: {e}")
                results[symbol] = ValidationSummary(
                    result=ValidationResult.REJECTED,
                    quality_metrics=QualityMetrics(0, 0, 0, 0, 0, 0),
                    issues=[ValidationIssue("error", "VALIDATION_ERROR", str(e))],
                    recommendations=["Manual review required due to validation error"]
                )
        
        return results


# Global validator instances
_validators: Dict[ValidationLevel, SignalValidator] = {}


def get_signal_validator(level: ValidationLevel = ValidationLevel.STANDARD) -> SignalValidator:
    """Get signal validator instance for specified level"""
    if level not in _validators:
        _validators[level] = SignalValidator(level)
    return _validators[level]


async def validate_signal_with_caching(
    signal: SignalCreate,
    validation_level: ValidationLevel = ValidationLevel.STANDARD,
    use_cache: bool = True
) -> ValidationSummary:
    """Validate signal with optional caching"""
    
    # Generate cache key
    cache_key = f"{signal.symbol}_{signal.signal_type.value}_{hash(str(signal.model_dump()))}"
    
    # Try cache first if enabled
    if use_cache:
        signal_cache = get_signal_cache()
        if signal_cache:
            cached_result = await signal_cache.cache.get(
                "validation_results", 
                cache_key
            )
            if cached_result:
                logger.info(f"Using cached validation result for {signal.symbol}")
                return ValidationSummary(**cached_result)
    
    # Perform validation
    validator = get_signal_validator(validation_level)
    
    # Get market data if available
    market_data = None
    market_cache = get_market_cache()
    if market_cache:
        market_data = await market_cache.get_quote(signal.symbol)
    
    # Validate signal
    result = await validator.validate_signal(signal, market_data)
    
    # Cache result if enabled
    if use_cache and result.result != ValidationResult.REJECTED:
        if signal_cache:
            await signal_cache.cache.set(
                "validation_results",
                cache_key,
                result.__dict__,
                ttl=300  # 5 minutes
            )
    
    return result