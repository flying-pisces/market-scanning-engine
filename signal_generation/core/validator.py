"""
Signal Validation System
Author: Claude Code (System Architect)
Version: 1.0

Comprehensive signal validation and quality control system that ensures:
- Signal data integrity and consistency
- Risk score validation and calibration
- Cross-validation with market conditions
- Quality scoring and filtering
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple, Set
from decimal import Decimal
from enum import Enum
from dataclasses import dataclass
from uuid import UUID

from data_models.python.core_models import Asset, MarketData, TechnicalIndicators
from data_models.python.signal_models import (
    Signal, SignalFactor, SignalDirection, SignalStrength,
    SignalStatus, SignalCategory, MarketRegime
)


class ValidationSeverity(str, Enum):
    """Validation issue severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Individual validation issue"""
    severity: ValidationSeverity
    category: str
    message: str
    field: Optional[str] = None
    suggested_fix: Optional[str] = None


@dataclass
class ValidationResult:
    """Complete validation result for a signal"""
    signal_id: UUID
    is_valid: bool
    quality_score: int  # 0-100
    issues: List[ValidationIssue]
    validated_at: datetime
    validator_version: str
    
    @property
    def has_critical_issues(self) -> bool:
        return any(issue.severity == ValidationSeverity.CRITICAL for issue in self.issues)
    
    @property
    def has_errors(self) -> bool:
        return any(issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] 
                  for issue in self.issues)
    
    @property
    def warning_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == ValidationSeverity.WARNING)
    
    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues 
                  if issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL])


class SignalValidator:
    """
    Comprehensive signal validation system
    
    Validates signals across multiple dimensions:
    1. Data integrity and consistency
    2. Risk score calibration
    3. Market condition alignment
    4. Factor analysis validation
    5. Price target reasonableness
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger("signal_validator")
        self.validator_version = "1.0"
        
        # Validation thresholds
        self.min_quality_score = self.config.get("min_quality_score", 60)
        self.max_risk_score = self.config.get("max_risk_score", 100)
        self.min_confidence_score = self.config.get("min_confidence_score", 40)
        
        # Price target validation
        self.max_price_move_pct = self.config.get("max_price_move_pct", 0.20)  # 20%
        self.max_stop_loss_pct = self.config.get("max_stop_loss_pct", 0.15)   # 15%
        
        # Factor validation
        self.min_factor_count = self.config.get("min_factor_count", 2)
        self.max_factor_count = self.config.get("max_factor_count", 10)
        self.min_factor_weight_sum = self.config.get("min_factor_weight_sum", 0.8)
        
        # Market condition filters
        self.enable_market_regime_check = self.config.get("enable_market_regime_check", True)
        self.enable_volatility_check = self.config.get("enable_volatility_check", True)
        
        # Performance tracking
        self._validation_count = 0
        self._rejection_count = 0
        self._quality_scores = []
    
    async def validate_signal(self,
                             signal: Signal,
                             asset: Asset,
                             market_data: Optional[MarketData] = None,
                             technical_indicators: Optional[TechnicalIndicators] = None,
                             market_regime_data: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Validate a complete signal with comprehensive checks
        
        Args:
            signal: Signal to validate
            asset: Related asset information
            market_data: Current market data for context
            technical_indicators: Technical indicators for confirmation
            market_regime_data: Market regime information
            
        Returns:
            ValidationResult with validation outcome and issues
        """
        issues = []
        
        try:
            # Core data validation
            core_issues = self._validate_core_data(signal, asset)
            issues.extend(core_issues)
            
            # Risk score validation
            risk_issues = self._validate_risk_scores(signal, asset, market_data)
            issues.extend(risk_issues)
            
            # Price target validation
            price_issues = self._validate_price_targets(signal, market_data)
            issues.extend(price_issues)
            
            # Factor analysis validation
            factor_issues = self._validate_signal_factors(signal)
            issues.extend(factor_issues)
            
            # Market condition validation
            if market_data and self.enable_market_regime_check:
                market_issues = self._validate_market_conditions(signal, market_data, market_regime_data)
                issues.extend(market_issues)
            
            # Technical confirmation validation
            if technical_indicators:
                tech_issues = self._validate_technical_confirmation(signal, technical_indicators)
                issues.extend(tech_issues)
            
            # Time-based validation
            time_issues = self._validate_timing(signal)
            issues.extend(time_issues)
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(signal, issues)
            
            # Determine overall validity
            is_valid = (quality_score >= self.min_quality_score and
                       not any(issue.severity == ValidationSeverity.CRITICAL for issue in issues))
            
            # Update performance metrics
            self._validation_count += 1
            if not is_valid:
                self._rejection_count += 1
            self._quality_scores.append(quality_score)
            
            validation_result = ValidationResult(
                signal_id=signal.id,
                is_valid=is_valid,
                quality_score=quality_score,
                issues=issues,
                validated_at=datetime.utcnow(),
                validator_version=self.validator_version
            )
            
            self.logger.info(
                f"Validated signal {signal.id}: "
                f"Valid={is_valid}, Quality={quality_score}, Issues={len(issues)}"
            )
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Validation failed for signal {signal.id}: {str(e)}", exc_info=True)
            
            # Return failed validation result
            return ValidationResult(
                signal_id=signal.id,
                is_valid=False,
                quality_score=0,
                issues=[ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    category="validation_error",
                    message=f"Validation process failed: {str(e)}"
                )],
                validated_at=datetime.utcnow(),
                validator_version=self.validator_version
            )
    
    def _validate_core_data(self, signal: Signal, asset: Asset) -> List[ValidationIssue]:
        """Validate core signal data integrity"""
        issues = []
        
        # Required field validation
        if not signal.signal_name or len(signal.signal_name.strip()) == 0:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="data_integrity",
                field="signal_name",
                message="Signal name is required and cannot be empty"
            ))
        
        if not signal.direction:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                category="data_integrity",
                field="direction",
                message="Signal direction is required"
            ))
        
        if not signal.signal_source:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="data_integrity",
                field="signal_source",
                message="Signal source is required for tracking"
            ))
        
        # Score validation
        for score_field, score_value in [
            ("risk_score", signal.risk_score),
            ("profit_potential_score", signal.profit_potential_score),
            ("confidence_score", signal.confidence_score)
        ]:
            if not 0 <= score_value <= 100:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="score_validation",
                    field=score_field,
                    message=f"{score_field} must be between 0 and 100, got {score_value}",
                    suggested_fix=f"Clamp {score_field} to valid range [0, 100]"
                ))
        
        # Asset alignment validation
        if signal.asset_id != asset.id:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                category="data_integrity",
                field="asset_id",
                message="Signal asset_id does not match provided asset"
            ))
        
        # Time validation
        now = datetime.utcnow()
        if signal.generated_at and signal.generated_at > now + timedelta(minutes=5):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="time_validation",
                field="generated_at",
                message="Signal generated_at is in the future"
            ))
        
        if signal.valid_until and signal.valid_until <= signal.generated_at:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="time_validation",
                field="valid_until",
                message="Signal valid_until must be after generated_at"
            ))
        
        return issues
    
    def _validate_risk_scores(self,
                            signal: Signal,
                            asset: Asset,
                            market_data: Optional[MarketData]) -> List[ValidationIssue]:
        """Validate risk score calibration and reasonableness"""
        issues = []
        
        risk_score = signal.risk_score
        
        # Basic range validation
        if risk_score > self.max_risk_score:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="risk_validation",
                field="risk_score",
                message=f"Risk score {risk_score} exceeds maximum {self.max_risk_score}"
            ))
        
        # Asset class specific risk validation
        expected_risk_ranges = {
            "daily_options": (70, 95),
            "stocks": (30, 80),
            "etfs": (20, 70),
            "bonds": (5, 40),
            "safe_assets": (0, 20)
        }
        
        asset_category = getattr(asset, 'category', None)
        if asset_category and asset_category in expected_risk_ranges:
            min_risk, max_risk = expected_risk_ranges[asset_category]
            
            if risk_score < min_risk:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="risk_calibration",
                    field="risk_score",
                    message=f"Risk score {risk_score} below expected range [{min_risk}, {max_risk}] for {asset_category}",
                    suggested_fix=f"Consider risk score >= {min_risk} for {asset_category}"
                ))
            
            elif risk_score > max_risk:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="risk_calibration",
                    field="risk_score",
                    message=f"Risk score {risk_score} above expected range [{min_risk}, {max_risk}] for {asset_category}",
                    suggested_fix=f"Consider risk score <= {max_risk} for {asset_category}"
                ))
        
        # Risk-reward consistency
        risk_reward_ratio = signal.profit_potential_score / max(risk_score, 1)
        if risk_reward_ratio > 3.0:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="risk_validation",
                message=f"High risk-reward ratio {risk_reward_ratio:.2f} may be unrealistic",
                suggested_fix="Review profit potential or increase risk score"
            ))
        
        # Volatility-adjusted risk validation
        if market_data and hasattr(market_data, 'volume') and asset.avg_volume_30d:
            volume_ratio = market_data.volume / asset.avg_volume_30d
            
            if volume_ratio > 3.0 and risk_score < 60:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="risk_calibration",
                    message=f"High volume ({volume_ratio:.1f}x avg) suggests higher risk than {risk_score}",
                    suggested_fix="Consider increasing risk score for high volume conditions"
                ))
        
        return issues
    
    def _validate_price_targets(self,
                              signal: Signal,
                              market_data: Optional[MarketData]) -> List[ValidationIssue]:
        """Validate price targets for reasonableness"""
        issues = []
        
        entry_price = signal.entry_price_cents
        target_price = signal.target_price_cents
        stop_loss_price = signal.stop_loss_price_cents
        
        if not entry_price:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="price_targets",
                field="entry_price_cents",
                message="Entry price not specified"
            ))
            return issues  # Can't validate targets without entry price
        
        # Validate target price
        if target_price:
            target_move_pct = abs(target_price - entry_price) / entry_price
            
            if target_move_pct > self.max_price_move_pct:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="price_targets",
                    field="target_price_cents",
                    message=f"Target move {target_move_pct:.1%} exceeds maximum {self.max_price_move_pct:.1%}",
                    suggested_fix=f"Consider target closer to entry price"
                ))
            
            # Direction consistency
            if signal.direction == SignalDirection.BUY and target_price <= entry_price:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="price_targets",
                    field="target_price_cents",
                    message="BUY signal target price must be above entry price"
                ))
            
            elif signal.direction == SignalDirection.SELL and target_price >= entry_price:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="price_targets",
                    field="target_price_cents",
                    message="SELL signal target price must be below entry price"
                ))
        
        # Validate stop loss
        if stop_loss_price:
            stop_loss_move_pct = abs(stop_loss_price - entry_price) / entry_price
            
            if stop_loss_move_pct > self.max_stop_loss_pct:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="price_targets",
                    field="stop_loss_price_cents",
                    message=f"Stop loss {stop_loss_move_pct:.1%} exceeds maximum {self.max_stop_loss_pct:.1%}",
                    suggested_fix=f"Consider stop loss closer to entry price"
                ))
            
            # Direction consistency  
            if signal.direction == SignalDirection.BUY and stop_loss_price >= entry_price:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="price_targets",
                    field="stop_loss_price_cents",
                    message="BUY signal stop loss must be below entry price"
                ))
            
            elif signal.direction == SignalDirection.SELL and stop_loss_price <= entry_price:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="price_targets",
                    field="stop_loss_price_cents",
                    message="SELL signal stop loss must be above entry price"
                ))
        
        # Risk-reward ratio validation
        if target_price and stop_loss_price:
            potential_gain = abs(target_price - entry_price)
            potential_loss = abs(stop_loss_price - entry_price)
            
            if potential_loss > 0:
                risk_reward_ratio = potential_gain / potential_loss
                
                if risk_reward_ratio < 0.5:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category="price_targets",
                        message=f"Poor risk-reward ratio {risk_reward_ratio:.2f} (< 0.5)",
                        suggested_fix="Adjust target or stop loss for better risk-reward"
                    ))
        
        return issues
    
    def _validate_signal_factors(self, signal: Signal) -> List[ValidationIssue]:
        """Validate signal factors and their contributions"""
        issues = []
        
        factors = signal.factors
        
        if len(factors) < self.min_factor_count:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="factor_validation",
                field="factors",
                message=f"Signal has {len(factors)} factors, minimum recommended is {self.min_factor_count}",
                suggested_fix="Add more supporting factors for signal strength"
            ))
        
        if len(factors) > self.max_factor_count:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="factor_validation",
                field="factors",
                message=f"Signal has {len(factors)} factors, maximum recommended is {self.max_factor_count}",
                suggested_fix="Consolidate or remove less important factors"
            ))
        
        if factors:
            # Weight validation
            total_weight = sum(factor.weight for factor in factors)
            
            if total_weight < self.min_factor_weight_sum:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="factor_validation",
                    field="factors",
                    message=f"Factor weights sum to {total_weight:.3f}, should be >= {self.min_factor_weight_sum}",
                    suggested_fix="Increase factor weights or add more factors"
                ))
            
            # Individual factor validation
            for i, factor in enumerate(factors):
                if factor.contribution_score < 0 or factor.contribution_score > 100:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        category="factor_validation",
                        field=f"factors[{i}].contribution_score",
                        message=f"Factor '{factor.factor_name}' score {factor.contribution_score} outside valid range [0, 100]"
                    ))
                
                if factor.weight < 0 or factor.weight > 1:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        category="factor_validation",
                        field=f"factors[{i}].weight",
                        message=f"Factor '{factor.factor_name}' weight {factor.weight} outside valid range [0, 1]"
                    ))
                
                if not factor.factor_name or len(factor.factor_name.strip()) == 0:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        category="factor_validation",
                        field=f"factors[{i}].factor_name",
                        message="Factor name cannot be empty"
                    ))
        
        return issues
    
    def _validate_market_conditions(self,
                                   signal: Signal,
                                   market_data: MarketData,
                                   market_regime_data: Optional[Dict[str, Any]]) -> List[ValidationIssue]:
        """Validate signal against current market conditions"""
        issues = []
        
        # Volume validation
        if hasattr(market_data, 'volume') and market_data.volume == 0:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="market_conditions",
                message="Zero trading volume may indicate stale or invalid market data"
            ))
        
        # Market hours validation (simplified)
        market_time = market_data.timestamp if hasattr(market_data, 'timestamp') else datetime.utcnow()
        market_hour = market_time.hour
        
        # US market hours (Eastern Time): 9:30 AM - 4:00 PM
        if market_hour < 9 or market_hour > 16:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                category="market_conditions",
                message=f"Signal generated outside regular market hours ({market_hour}:00)",
                suggested_fix="Consider after-hours risk factors"
            ))
        
        # Market regime validation
        if market_regime_data:
            regime = market_regime_data.get("primary_regime")
            stress_index = market_regime_data.get("market_stress_index", 50)
            
            if stress_index > 80:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="market_conditions",
                    message=f"High market stress ({stress_index}) may affect signal reliability",
                    suggested_fix="Consider increasing risk score or shortening holding period"
                ))
            
            # Signal type vs market regime consistency
            if regime == "high_vol" and signal.risk_score < 60:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="market_conditions",
                    message=f"Low risk score in high volatility regime may be underestimated",
                    suggested_fix="Consider higher risk score for high volatility conditions"
                ))
        
        return issues
    
    def _validate_technical_confirmation(self,
                                       signal: Signal,
                                       technical_indicators: TechnicalIndicators) -> List[ValidationIssue]:
        """Validate signal against technical indicators"""
        issues = []
        
        # RSI validation for momentum signals
        if technical_indicators.rsi_14 and signal.direction:
            rsi = float(technical_indicators.rsi_14)
            
            if signal.direction == SignalDirection.BUY and rsi > 80:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="technical_confirmation",
                    message=f"BUY signal with overbought RSI ({rsi:.1f}) may face resistance",
                    suggested_fix="Consider waiting for RSI pullback or adjust risk"
                ))
            
            elif signal.direction == SignalDirection.SELL and rsi < 20:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="technical_confirmation",
                    message=f"SELL signal with oversold RSI ({rsi:.1f}) may face support",
                    suggested_fix="Consider waiting for RSI bounce or adjust risk"
                ))
        
        # Moving average trend validation
        if (technical_indicators.sma_20 and technical_indicators.sma_50 and
            signal.entry_price_cents):
            
            sma_20 = technical_indicators.sma_20
            sma_50 = technical_indicators.sma_50
            entry_price = signal.entry_price_cents
            
            # Trend direction
            trend_direction = "bullish" if sma_20 > sma_50 else "bearish"
            
            if (signal.direction == SignalDirection.BUY and trend_direction == "bearish"):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    category="technical_confirmation",
                    message="BUY signal against bearish trend (SMA 20 < SMA 50)",
                    suggested_fix="Consider counter-trend risk factors"
                ))
            
            elif (signal.direction == SignalDirection.SELL and trend_direction == "bullish"):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    category="technical_confirmation",
                    message="SELL signal against bullish trend (SMA 20 > SMA 50)",
                    suggested_fix="Consider counter-trend risk factors"
                ))
        
        return issues
    
    def _validate_timing(self, signal: Signal) -> List[ValidationIssue]:
        """Validate signal timing aspects"""
        issues = []
        
        now = datetime.utcnow()
        
        # Signal freshness
        if signal.generated_at:
            age_hours = (now - signal.generated_at).total_seconds() / 3600
            
            if age_hours > 24:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="timing",
                    message=f"Signal is {age_hours:.1f} hours old, may be stale",
                    suggested_fix="Consider signal expiration or refresh"
                ))
        
        # Signal expiration
        if signal.valid_until:
            time_to_expiry_hours = (signal.valid_until - now).total_seconds() / 3600
            
            if time_to_expiry_hours <= 0:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="timing",
                    field="valid_until",
                    message="Signal has expired",
                    suggested_fix="Update expiration time or mark as expired"
                ))
            
            elif time_to_expiry_hours < 1:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="timing",
                    field="valid_until",
                    message=f"Signal expires in {time_to_expiry_hours:.1f} hours",
                    suggested_fix="Consider extending expiration if still valid"
                ))
        
        # Holding period validation
        if signal.recommended_holding_period_hours:
            if signal.recommended_holding_period_hours < 1:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="timing",
                    field="recommended_holding_period_hours",
                    message="Very short holding period may indicate high-frequency trading",
                    suggested_fix="Ensure sufficient time for signal to develop"
                ))
            
            elif signal.recommended_holding_period_hours > 8760:  # 1 year
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="timing",
                    field="recommended_holding_period_hours",
                    message="Very long holding period may indicate investment rather than trading signal",
                    suggested_fix="Consider shorter time horizon for active trading"
                ))
        
        return issues
    
    def _calculate_quality_score(self, signal: Signal, issues: List[ValidationIssue]) -> int:
        """Calculate overall quality score for the signal"""
        
        # Start with base score
        base_score = 100
        
        # Deduct points for issues
        for issue in issues:
            if issue.severity == ValidationSeverity.CRITICAL:
                base_score -= 50
            elif issue.severity == ValidationSeverity.ERROR:
                base_score -= 20
            elif issue.severity == ValidationSeverity.WARNING:
                base_score -= 10
            elif issue.severity == ValidationSeverity.INFO:
                base_score -= 5
        
        # Bonus points for good practices
        if signal.factors and len(signal.factors) >= 3:
            base_score += 5  # Good factor coverage
        
        if signal.target_price_cents and signal.stop_loss_price_cents:
            base_score += 5  # Complete risk management
        
        if signal.backtesting and signal.backtesting.backtest_sample_size and signal.backtesting.backtest_sample_size > 50:
            base_score += 10  # Historical validation
        
        # Confidence alignment bonus
        if 70 <= signal.confidence_score <= 85:
            base_score += 5  # Good confidence range
        
        # Ensure score is in valid range
        return max(0, min(100, base_score))
    
    async def validate_signal_batch(self,
                                   signals: List[Signal],
                                   assets: Dict[UUID, Asset],
                                   market_data: Dict[UUID, MarketData] = None,
                                   technical_indicators: Dict[UUID, TechnicalIndicators] = None,
                                   market_regime_data: Optional[Dict[str, Any]] = None) -> List[ValidationResult]:
        """Validate a batch of signals efficiently"""
        
        tasks = []
        for signal in signals:
            asset = assets.get(signal.asset_id)
            if not asset:
                self.logger.warning(f"Asset not found for signal {signal.id}")
                continue
            
            mkt_data = market_data.get(signal.asset_id) if market_data else None
            tech_data = technical_indicators.get(signal.asset_id) if technical_indicators else None
            
            task = self.validate_signal(signal, asset, mkt_data, tech_data, market_regime_data)
            tasks.append(task)
        
        if not tasks:
            return []
        
        # Execute validations concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and log them
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Validation failed for signal {signals[i].id}: {result}")
            else:
                valid_results.append(result)
        
        return valid_results
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validator performance statistics"""
        
        rejection_rate = (self._rejection_count / max(self._validation_count, 1)) * 100
        avg_quality_score = sum(self._quality_scores) / max(len(self._quality_scores), 1)
        
        return {
            "total_validations": self._validation_count,
            "total_rejections": self._rejection_count,
            "rejection_rate_pct": rejection_rate,
            "average_quality_score": avg_quality_score,
            "validator_version": self.validator_version,
            "config": self.config
        }
    
    def reset_statistics(self) -> None:
        """Reset validation statistics"""
        self._validation_count = 0
        self._rejection_count = 0
        self._quality_scores = []
        
        self.logger.info("Validation statistics reset")
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update validator configuration"""
        self.config.update(new_config)
        
        # Update threshold values
        self.min_quality_score = self.config.get("min_quality_score", 60)
        self.max_risk_score = self.config.get("max_risk_score", 100)
        self.min_confidence_score = self.config.get("min_confidence_score", 40)
        self.max_price_move_pct = self.config.get("max_price_move_pct", 0.20)
        self.max_stop_loss_pct = self.config.get("max_stop_loss_pct", 0.15)
        
        self.logger.info("Validator configuration updated")