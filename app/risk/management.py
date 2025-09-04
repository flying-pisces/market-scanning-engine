"""
Risk Management System
Advanced risk calculation and monitoring for portfolio management
"""

from enum import Enum
from typing import Optional, Dict, Any, List, Union
from datetime import datetime, timezone
from decimal import Decimal
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from ..models.portfolio import Portfolio, Position


class VaRMethod(str, Enum):
    """Value at Risk calculation methods"""
    HISTORICAL = "historical"
    PARAMETRIC = "parametric" 
    MONTE_CARLO = "monte_carlo"


class StressTestType(str, Enum):
    """Stress test scenario types"""
    HISTORICAL = "historical"
    HYPOTHETICAL = "hypothetical"
    FACTOR_BASED = "factor_based"


# Risk Models
class VaRResult(BaseModel):
    """Value at Risk calculation result"""
    var_value: float
    confidence_level: float
    time_horizon: int
    method: VaRMethod
    currency: str = "USD"
    expected_shortfall: Optional[float] = None
    portfolio_value: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    simulations: Optional[int] = None


class ConditionalVaRResult(BaseModel):
    """Conditional VaR (Expected Shortfall) result"""
    cvar_value: float
    var_value: float
    confidence_level: float
    method: VaRMethod
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class StressTest(BaseModel):
    """Stress test scenario definition"""
    name: str
    scenario_type: str
    shocks: Dict[str, float]  # symbol -> shock percentage
    description: Optional[str] = None


class StressTestResult(BaseModel):
    """Stress test result"""
    scenario_name: str
    portfolio_loss: float
    portfolio_value_before: float
    portfolio_value_after: float
    position_impacts: Dict[str, float]  # symbol -> impact
    worst_position: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RiskMetrics(BaseModel):
    """Comprehensive risk metrics"""
    var_95: Optional[float] = None
    var_99: Optional[float] = None
    expected_shortfall: Optional[float] = None
    current_drawdown: Optional[float] = None
    max_drawdown: Optional[float] = None
    volatility: Optional[float] = None
    beta: Optional[float] = None
    correlation_with_market: Optional[float] = None
    max_position_concentration: Optional[float] = None
    liquidity_score: Optional[float] = None
    portfolio_value: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RiskManager:
    """Advanced risk management system"""
    
    def __init__(self):
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
        
    def calculate_var(
        self,
        portfolio: Portfolio,
        returns_data: pd.DataFrame,
        confidence_level: float = 0.95,
        method: VaRMethod = VaRMethod.HISTORICAL,
        time_horizon: int = 1,
        simulations: int = 10000
    ) -> Optional[VaRResult]:
        """
        Calculate Value at Risk for portfolio
        
        Args:
            portfolio: Portfolio object
            returns_data: Historical returns data
            confidence_level: Confidence level (0.95 for 95% VaR)
            method: VaR calculation method
            time_horizon: Time horizon in days
            simulations: Number of Monte Carlo simulations
            
        Returns:
            VaRResult object or None if calculation fails
        """
        try:
            if method == VaRMethod.HISTORICAL:
                return self._calculate_historical_var(
                    portfolio, returns_data, confidence_level, time_horizon
                )
            elif method == VaRMethod.PARAMETRIC:
                return self._calculate_parametric_var(
                    portfolio, returns_data, confidence_level, time_horizon
                )
            elif method == VaRMethod.MONTE_CARLO:
                return self._calculate_monte_carlo_var(
                    portfolio, returns_data, confidence_level, time_horizon, simulations
                )
            else:
                return None
                
        except Exception:
            return None
    
    def _calculate_historical_var(
        self,
        portfolio: Portfolio,
        returns_data: pd.DataFrame,
        confidence_level: float,
        time_horizon: int
    ) -> VaRResult:
        """Calculate Historical VaR"""
        # Get portfolio weights
        weights = self._get_portfolio_weights(portfolio)
        
        # Filter returns data for portfolio symbols
        portfolio_symbols = list(weights.keys())
        available_symbols = [s for s in portfolio_symbols if s in returns_data.columns]
        
        if not available_symbols:
            raise ValueError("No return data available for portfolio symbols")
        
        # Calculate portfolio returns
        portfolio_returns = (returns_data[available_symbols] * 
                           pd.Series({s: weights.get(s, 0) for s in available_symbols})).sum(axis=1)
        
        # Adjust for time horizon
        if time_horizon > 1:
            portfolio_returns = portfolio_returns * np.sqrt(time_horizon)
        
        # Calculate VaR as percentile
        var_percentile = (1 - confidence_level) * 100
        var_return = np.percentile(portfolio_returns, var_percentile)
        
        portfolio_value = float(portfolio.total_value) if hasattr(portfolio, 'total_value') else 100000.0
        var_value = abs(var_return * portfolio_value)
        
        # Calculate Expected Shortfall
        tail_returns = portfolio_returns[portfolio_returns <= var_return]
        expected_shortfall = abs(tail_returns.mean() * portfolio_value) if len(tail_returns) > 0 else var_value
        
        return VaRResult(
            var_value=var_value,
            confidence_level=confidence_level,
            time_horizon=time_horizon,
            method=VaRMethod.HISTORICAL,
            expected_shortfall=expected_shortfall,
            portfolio_value=portfolio_value
        )
    
    def _calculate_parametric_var(
        self,
        portfolio: Portfolio,
        returns_data: pd.DataFrame,
        confidence_level: float,
        time_horizon: int
    ) -> VaRResult:
        """Calculate Parametric VaR assuming normal distribution"""
        from scipy import stats
        
        weights = self._get_portfolio_weights(portfolio)
        portfolio_symbols = list(weights.keys())
        available_symbols = [s for s in portfolio_symbols if s in returns_data.columns]
        
        if not available_symbols:
            raise ValueError("No return data available for portfolio symbols")
        
        # Calculate portfolio returns
        portfolio_returns = (returns_data[available_symbols] * 
                           pd.Series({s: weights.get(s, 0) for s in available_symbols})).sum(axis=1)
        
        # Calculate mean and standard deviation
        mean_return = portfolio_returns.mean()
        std_return = portfolio_returns.std()
        
        # Adjust for time horizon
        if time_horizon > 1:
            mean_return = mean_return * time_horizon
            std_return = std_return * np.sqrt(time_horizon)
        
        # Calculate VaR using normal distribution
        z_score = stats.norm.ppf(1 - confidence_level)
        var_return = mean_return + z_score * std_return
        
        portfolio_value = float(portfolio.total_value) if hasattr(portfolio, 'total_value') else 100000.0
        var_value = abs(var_return * portfolio_value)
        
        # Expected Shortfall for normal distribution
        expected_shortfall_return = mean_return - (std_return * stats.norm.pdf(z_score) / (1 - confidence_level))
        expected_shortfall = abs(expected_shortfall_return * portfolio_value)
        
        return VaRResult(
            var_value=var_value,
            confidence_level=confidence_level,
            time_horizon=time_horizon,
            method=VaRMethod.PARAMETRIC,
            expected_shortfall=expected_shortfall,
            portfolio_value=portfolio_value
        )
    
    def _calculate_monte_carlo_var(
        self,
        portfolio: Portfolio,
        returns_data: pd.DataFrame,
        confidence_level: float,
        time_horizon: int,
        simulations: int
    ) -> VaRResult:
        """Calculate Monte Carlo VaR"""
        weights = self._get_portfolio_weights(portfolio)
        portfolio_symbols = list(weights.keys())
        available_symbols = [s for s in portfolio_symbols if s in returns_data.columns]
        
        if not available_symbols:
            raise ValueError("No return data available for portfolio symbols")
        
        # Calculate covariance matrix
        returns_subset = returns_data[available_symbols]
        mean_returns = returns_subset.mean()
        cov_matrix = returns_subset.cov()
        
        # Generate random scenarios
        np.random.seed(42)  # For reproducible results
        scenarios = np.random.multivariate_normal(
            mean_returns.values, 
            cov_matrix.values, 
            simulations
        )
        
        # Calculate portfolio returns for each scenario
        weight_array = np.array([weights.get(s, 0) for s in available_symbols])
        portfolio_returns = scenarios.dot(weight_array)
        
        # Adjust for time horizon
        if time_horizon > 1:
            portfolio_returns = portfolio_returns * np.sqrt(time_horizon)
        
        # Calculate VaR
        var_percentile = (1 - confidence_level) * 100
        var_return = np.percentile(portfolio_returns, var_percentile)
        
        portfolio_value = float(portfolio.total_value) if hasattr(portfolio, 'total_value') else 100000.0
        var_value = abs(var_return * portfolio_value)
        
        # Expected Shortfall
        tail_returns = portfolio_returns[portfolio_returns <= var_return]
        expected_shortfall = abs(tail_returns.mean() * portfolio_value) if len(tail_returns) > 0 else var_value
        
        return VaRResult(
            var_value=var_value,
            confidence_level=confidence_level,
            time_horizon=time_horizon,
            method=VaRMethod.MONTE_CARLO,
            expected_shortfall=expected_shortfall,
            portfolio_value=portfolio_value,
            simulations=simulations
        )
    
    def calculate_conditional_var(
        self,
        portfolio: Portfolio,
        returns_data: pd.DataFrame,
        confidence_level: float = 0.95,
        method: VaRMethod = VaRMethod.HISTORICAL
    ) -> Optional[ConditionalVaRResult]:
        """Calculate Conditional VaR (Expected Shortfall)"""
        try:
            var_result = self.calculate_var(portfolio, returns_data, confidence_level, method)
            if var_result and var_result.expected_shortfall:
                return ConditionalVaRResult(
                    cvar_value=var_result.expected_shortfall,
                    var_value=var_result.var_value,
                    confidence_level=confidence_level,
                    method=method
                )
            return None
        except Exception:
            return None
    
    def run_stress_tests(
        self,
        portfolio: Portfolio,
        stress_scenarios: List[StressTest]
    ) -> List[StressTestResult]:
        """Run stress tests on portfolio"""
        results = []
        
        for scenario in stress_scenarios:
            try:
                result = self._run_single_stress_test(portfolio, scenario)
                if result:
                    results.append(result)
            except Exception:
                continue
                
        return results
    
    def _run_single_stress_test(
        self,
        portfolio: Portfolio,
        scenario: StressTest
    ) -> Optional[StressTestResult]:
        """Run a single stress test scenario"""
        try:
            weights = self._get_portfolio_weights(portfolio)
            portfolio_value = float(portfolio.total_value) if hasattr(portfolio, 'total_value') else 100000.0
            
            position_impacts = {}
            total_loss = 0.0
            
            for symbol, shock in scenario.shocks.items():
                if symbol in weights:
                    position_value = weights[symbol] * portfolio_value
                    position_loss = position_value * shock
                    position_impacts[symbol] = position_loss
                    total_loss += position_loss
            
            # Find worst impacted position
            worst_position = None
            if position_impacts:
                worst_position = min(position_impacts.keys(), key=lambda s: position_impacts[s])
            
            return StressTestResult(
                scenario_name=scenario.name,
                portfolio_loss=total_loss,
                portfolio_value_before=portfolio_value,
                portfolio_value_after=portfolio_value + total_loss,
                position_impacts=position_impacts,
                worst_position=worst_position
            )
            
        except Exception:
            return None
    
    def calculate_correlation_matrix(self, returns_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation matrix for assets"""
        return returns_data.corr()
    
    def calculate_portfolio_beta(
        self,
        portfolio_returns: pd.DataFrame,
        market_returns: pd.Series,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Calculate portfolio and individual asset betas"""
        results = {"individual_betas": {}, "portfolio_beta": 0.0}
        
        try:
            # Calculate individual betas
            for symbol in portfolio_returns.columns:
                if symbol in portfolio_returns.columns:
                    asset_returns = portfolio_returns[symbol].dropna()
                    market_aligned = market_returns.reindex(asset_returns.index).dropna()
                    
                    if len(market_aligned) > 10:  # Need sufficient data
                        covariance = np.cov(asset_returns, market_aligned)[0, 1]
                        market_variance = np.var(market_aligned)
                        beta = covariance / market_variance if market_variance > 0 else 0.0
                        results["individual_betas"][symbol] = beta
            
            # Calculate portfolio beta
            if weights and results["individual_betas"]:
                portfolio_beta = sum(
                    weights.get(symbol, 0) * beta 
                    for symbol, beta in results["individual_betas"].items()
                )
                results["portfolio_beta"] = portfolio_beta
                
        except Exception:
            pass
            
        return results
    
    def calculate_drawdown_metrics(self, portfolio_values: pd.Series) -> Dict[str, Any]:
        """Calculate drawdown metrics"""
        try:
            # Calculate running maximum
            peak = portfolio_values.cummax()
            
            # Calculate drawdown
            drawdown = (portfolio_values - peak) / peak
            
            # Maximum drawdown
            max_drawdown = drawdown.min()
            
            # Find drawdown periods
            drawdown_periods = []
            in_drawdown = False
            start_idx = None
            
            for idx, dd in drawdown.items():
                if dd < -0.001 and not in_drawdown:  # Start of drawdown (>0.1%)
                    in_drawdown = True
                    start_idx = idx
                elif dd >= -0.001 and in_drawdown:  # End of drawdown
                    in_drawdown = False
                    if start_idx:
                        drawdown_periods.append((start_idx, idx))
            
            # Calculate drawdown duration
            max_drawdown_duration = 0
            if drawdown_periods:
                durations = [(end - start).days for start, end in drawdown_periods]
                max_drawdown_duration = max(durations) if durations else 0
            
            return {
                "max_drawdown": max_drawdown,
                "max_drawdown_duration": max_drawdown_duration,
                "recovery_time": max_drawdown_duration,  # Simplified
                "drawdown_series": drawdown.to_dict()
            }
            
        except Exception:
            return {
                "max_drawdown": 0.0,
                "max_drawdown_duration": 0,
                "recovery_time": 0,
                "drawdown_series": {}
            }
    
    def assess_liquidity_risk(
        self,
        portfolio: Portfolio,
        liquidity_data: Dict[str, Dict[str, float]],
        liquidation_timeframe: int = 5
    ) -> Dict[str, Any]:
        """Assess portfolio liquidity risk"""
        try:
            weights = self._get_portfolio_weights(portfolio)
            
            position_scores = {}
            total_score = 0.0
            total_cost = 0.0
            
            for symbol, weight in weights.items():
                if symbol in liquidity_data:
                    data = liquidity_data[symbol]
                    
                    # Calculate liquidity score (0-1, higher is better)
                    volume_score = min(data.get('avg_daily_volume', 0) / 1000000, 10) / 10  # Normalize to 10M volume
                    spread_score = max(0, 1 - data.get('bid_ask_spread', 0.01) * 100)  # Penalize wide spreads
                    
                    liquidity_score = (volume_score + spread_score) / 2
                    position_scores[symbol] = liquidity_score
                    
                    total_score += weight * liquidity_score
                    total_cost += weight * data.get('bid_ask_spread', 0.01) / 2  # Half spread cost
            
            return {
                "overall_liquidity_score": total_score,
                "position_liquidity_scores": position_scores,
                "estimated_liquidation_cost": total_cost,
                "time_to_liquidate": liquidation_timeframe
            }
            
        except Exception:
            return {
                "overall_liquidity_score": 0.5,
                "position_liquidity_scores": {},
                "estimated_liquidation_cost": 0.01,
                "time_to_liquidate": liquidation_timeframe
            }
    
    def _get_portfolio_weights(self, portfolio: Portfolio) -> Dict[str, float]:
        """Extract portfolio weights from portfolio object"""
        weights = {}
        
        if hasattr(portfolio, 'positions'):
            total_value = float(portfolio.total_value) if hasattr(portfolio, 'total_value') else 0.0
            
            if total_value > 0:
                for position in portfolio.positions:
                    if hasattr(position, 'symbol') and hasattr(position, 'market_value'):
                        symbol = position.symbol
                        market_value = float(position.market_value)
                        weight = market_value / total_value
                        weights[symbol] = weight
        
        return weights