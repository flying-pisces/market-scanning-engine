"""
Basic Risk Scoring Service
Implements the core 0-100 risk scoring algorithm for MVP
"""

import logging
from typing import Dict, Any, Optional
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum

logger = logging.getLogger(__name__)


class AssetClass(str, Enum):
    """Asset classes with base risk profiles"""
    DAILY_OPTIONS = "daily_options"
    STOCKS = "stocks"
    ETFS = "etfs"
    BONDS = "bonds"
    SAFE_ASSETS = "safe_assets"


class RiskScorer:
    """Basic risk scoring implementation for MVP"""
    
    # Base risk scores for different asset classes (0-100 scale)
    ASSET_CLASS_BASE_RISKS = {
        AssetClass.DAILY_OPTIONS: 85,  # High risk: 70-95 range
        AssetClass.STOCKS: 55,         # Medium risk: 30-80 range
        AssetClass.ETFS: 45,           # Medium-low risk: 20-70 range
        AssetClass.BONDS: 25,          # Low risk: 5-40 range
        AssetClass.SAFE_ASSETS: 10,    # Ultra-low risk: 0-20 range
    }
    
    # Risk factor weights (sum to 1.0)
    RISK_FACTOR_WEIGHTS = {
        "volatility": 0.30,     # Historical volatility impact
        "liquidity": 0.25,      # Bid/ask spreads, volume
        "time_horizon": 0.20,   # Time to expiration/holding period
        "market_conditions": 0.15,  # VIX, market regime
        "position_size": 0.10,  # Portfolio impact
    }
    
    def calculate_basic_risk_score(
        self,
        asset_class: AssetClass,
        symbol: str,
        market_data: Optional[Dict[str, Any]] = None,
        additional_factors: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Calculate basic risk score for an asset
        
        Args:
            asset_class: Type of asset (options, stocks, etc.)
            symbol: Asset symbol (SPY, AAPL, etc.)
            market_data: Optional market data for more accurate scoring
            additional_factors: Optional additional risk factors
            
        Returns:
            Dict with risk score and breakdown
        """
        try:
            # Start with base risk for asset class
            base_risk = self.ASSET_CLASS_BASE_RISKS[asset_class]
            
            # Calculate individual risk factors
            risk_factors = self._calculate_risk_factors(
                asset_class, symbol, market_data, additional_factors
            )
            
            # Weighted risk calculation
            weighted_risk = self._calculate_weighted_risk(base_risk, risk_factors)
            
            # Ensure risk score is within 0-100 bounds
            final_risk_score = max(0, min(100, weighted_risk))
            
            # Round to configured precision
            risk_score = float(
                Decimal(str(final_risk_score)).quantize(
                    Decimal('0.01'), 
                    rounding=ROUND_HALF_UP
                )
            )
            
            logger.info(f"Calculated risk score {risk_score} for {symbol} ({asset_class})")
            
            return {
                "risk_score": risk_score,
                "asset_class": asset_class,
                "base_risk": base_risk,
                "risk_factors": risk_factors,
                "methodology": "basic_mvp_scoring",
                "confidence": self._calculate_confidence(risk_factors),
            }
            
        except Exception as e:
            logger.error(f"Risk scoring failed for {symbol}: {e}")
            raise
    
    def _calculate_risk_factors(
        self,
        asset_class: AssetClass,
        symbol: str,
        market_data: Optional[Dict[str, Any]],
        additional_factors: Optional[Dict[str, float]]
    ) -> Dict[str, float]:
        """Calculate individual risk factors"""
        
        risk_factors = {}
        
        # Volatility factor (0-100 adjustment from base)
        risk_factors["volatility"] = self._calculate_volatility_factor(
            asset_class, symbol, market_data
        )
        
        # Liquidity factor
        risk_factors["liquidity"] = self._calculate_liquidity_factor(
            asset_class, symbol, market_data
        )
        
        # Time horizon factor
        risk_factors["time_horizon"] = self._calculate_time_factor(
            asset_class, additional_factors
        )
        
        # Market conditions factor
        risk_factors["market_conditions"] = self._calculate_market_conditions_factor()
        
        # Position size factor
        risk_factors["position_size"] = self._calculate_position_size_factor(
            additional_factors
        )
        
        return risk_factors
    
    def _calculate_volatility_factor(
        self,
        asset_class: AssetClass,
        symbol: str,
        market_data: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate volatility risk factor adjustment"""
        
        # Default volatility adjustments by symbol (simplified for MVP)
        symbol_volatility_adjustments = {
            # Daily options - higher volatility increases risk
            "SPY": 0,    # Baseline
            "QQQ": 5,    # Tech heavy, more volatile
            "SPX": 0,    # Similar to SPY
            "XSP": 0,    # Mini SPX
            "NDX": 10,   # Pure tech, most volatile
            
            # Individual stocks - vary by company
            "AAPL": -5,  # Large cap, relatively stable
            "MSFT": -5,  # Large cap, stable
            "TSLA": 25,  # High volatility stock
            "GME": 40,   # Meme stock volatility
            
            # Bonds - generally low volatility
            "TLT": -10,  # Treasury bonds, low volatility
            "HYG": 5,    # High yield, more volatile
            
            # Safe assets - minimal volatility
            "SCHO": -15, # Short term treasuries
            "BIL": -20,  # Treasury bills
        }
        
        base_adjustment = symbol_volatility_adjustments.get(symbol, 0)
        
        # If we have market data, use actual volatility (simplified)
        if market_data and "implied_volatility" in market_data:
            iv = market_data["implied_volatility"]
            if iv > 30:  # High IV
                base_adjustment += 10
            elif iv < 15:  # Low IV  
                base_adjustment -= 10
        
        return base_adjustment
    
    def _calculate_liquidity_factor(
        self,
        asset_class: AssetClass,
        symbol: str,
        market_data: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate liquidity risk factor adjustment"""
        
        # Liquidity adjustments based on typical trading volumes
        if asset_class == AssetClass.DAILY_OPTIONS:
            # Major ETF options are highly liquid
            major_etf_options = ["SPY", "QQQ", "SPX", "XSP", "NDX"]
            return -5 if symbol in major_etf_options else 10
        
        elif asset_class == AssetClass.STOCKS:
            # Large cap stocks generally more liquid
            large_cap_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]
            return -5 if symbol in large_cap_symbols else 5
        
        elif asset_class == AssetClass.SAFE_ASSETS:
            # Government securities are highly liquid
            return -10
        
        # Default: no adjustment
        return 0
    
    def _calculate_time_factor(
        self,
        asset_class: AssetClass,
        additional_factors: Optional[Dict[str, float]]
    ) -> float:
        """Calculate time horizon risk factor adjustment"""
        
        if additional_factors and "days_to_expiration" in additional_factors:
            dte = additional_factors["days_to_expiration"]
            
            if asset_class == AssetClass.DAILY_OPTIONS:
                # Options with very short time to expiration are riskier
                if dte <= 1:
                    return 20  # Same day expiration - very risky
                elif dte <= 7:
                    return 10  # Weekly options - risky
                elif dte <= 30:
                    return 0   # Monthly options - baseline
                else:
                    return -5  # LEAPS - slightly less risky
        
        # For non-options, shorter holding periods generally riskier
        if additional_factors and "expected_holding_days" in additional_factors:
            holding_days = additional_factors["expected_holding_days"]
            if holding_days <= 1:
                return 15  # Day trading
            elif holding_days <= 7:
                return 5   # Short term
            else:
                return -5  # Long term holding
        
        return 0  # No time factor adjustment
    
    def _calculate_market_conditions_factor(self) -> float:
        """Calculate market conditions risk factor adjustment"""
        
        # For MVP, use simplified market conditions
        # In production, this would pull real VIX data, market regime indicators
        
        # Assume normal market conditions for MVP
        # VIX around 20 = normal volatility
        # This would be replaced with real-time VIX data
        mock_vix = 20
        
        if mock_vix > 30:
            return 15  # High volatility environment
        elif mock_vix > 25:
            return 5   # Elevated volatility
        elif mock_vix < 15:
            return -5  # Low volatility environment
        else:
            return 0   # Normal conditions
    
    def _calculate_position_size_factor(
        self,
        additional_factors: Optional[Dict[str, float]]
    ) -> float:
        """Calculate position size risk factor adjustment"""
        
        if additional_factors and "position_size_percent" in additional_factors:
            position_pct = additional_factors["position_size_percent"]
            
            # Larger positions relative to portfolio are riskier
            if position_pct > 20:
                return 15  # Very large position
            elif position_pct > 10:
                return 8   # Large position
            elif position_pct > 5:
                return 3   # Medium position
            else:
                return 0   # Small position
        
        return 0  # No position size information
    
    def _calculate_weighted_risk(
        self,
        base_risk: float,
        risk_factors: Dict[str, float]
    ) -> float:
        """Calculate weighted risk score from base risk and factors"""
        
        total_adjustment = 0
        
        for factor, adjustment in risk_factors.items():
            weight = self.RISK_FACTOR_WEIGHTS.get(factor, 0)
            total_adjustment += adjustment * weight
        
        return base_risk + total_adjustment
    
    def _calculate_confidence(self, risk_factors: Dict[str, float]) -> float:
        """Calculate confidence in the risk score (0-1 scale)"""
        
        # For MVP, use simplified confidence calculation
        # More data available = higher confidence
        
        base_confidence = 0.7
        
        # Reduce confidence if we're making large adjustments
        # (indicates we might be missing important data)
        max_adjustment = max(abs(factor) for factor in risk_factors.values())
        
        if max_adjustment > 20:
            base_confidence -= 0.2
        elif max_adjustment > 10:
            base_confidence -= 0.1
        
        return max(0.1, min(1.0, base_confidence))


# Global risk scorer instance
risk_scorer = RiskScorer()