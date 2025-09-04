"""
Portfolio optimization and position sizing algorithms
Implements Modern Portfolio Theory, Black-Litterman, and advanced optimization techniques
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from enum import Enum

# Optimization libraries
import cvxpy as cp
from scipy.optimize import minimize
from scipy.stats import norm

# Portfolio optimization libraries
try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns
    from pypfopt.black_litterman import BlackLittermanModel
    from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
except ImportError:
    # Fallback implementations if pypfopt not available
    EfficientFrontier = None
    BlackLittermanModel = None

from app.models.user import User
from app.models.signal import Signal, AssetClass
from app.core.cache import get_cache

logger = logging.getLogger(__name__)


class OptimizationMethod(Enum):
    """Portfolio optimization methods"""
    MEAN_VARIANCE = "mean_variance"          # Classic Markowitz
    BLACK_LITTERMAN = "black_litterman"      # Black-Litterman with views
    RISK_PARITY = "risk_parity"             # Risk parity allocation
    KELLY_CRITERION = "kelly_criterion"      # Kelly optimal betting
    EQUAL_WEIGHT = "equal_weight"           # Equal weight allocation
    MINIMUM_VARIANCE = "minimum_variance"    # Minimum variance portfolio
    MAXIMUM_SHARPE = "maximum_sharpe"       # Maximum Sharpe ratio
    HIERARCHICAL_RISK = "hierarchical_risk" # Hierarchical risk parity


class RiskModel(Enum):
    """Risk estimation methods"""
    SAMPLE_COV = "sample_cov"               # Sample covariance
    LEDOIT_WOLF = "ledoit_wolf"             # Ledoit-Wolf shrinkage
    OAS = "oas"                             # Oracle Approximating Shrinkage
    CONSTANT_CORRELATION = "constant_corr"   # Constant correlation model


@dataclass
class OptimizationConstraints:
    """Portfolio optimization constraints"""
    max_weight: float = 0.3              # Maximum weight per asset
    min_weight: float = 0.0              # Minimum weight per asset
    max_positions: int = 20              # Maximum number of positions
    sector_limits: Dict[str, float] = None # Sector concentration limits
    turnover_limit: float = None         # Portfolio turnover limit
    risk_budget: float = None            # Risk budget constraint
    
    def __post_init__(self):
        if self.sector_limits is None:
            self.sector_limits = {}


@dataclass
class PortfolioMetrics:
    """Portfolio performance and risk metrics"""
    expected_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    var_95: float                        # 95% Value at Risk
    cvar_95: float                       # 95% Conditional VaR
    calmar_ratio: float                  # Calmar ratio (return/max_drawdown)
    sortino_ratio: float                 # Sortino ratio (downside risk)
    tracking_error: float
    information_ratio: float
    beta: float
    alpha: float


@dataclass
class OptimizationResult:
    """Portfolio optimization result"""
    weights: Dict[str, float]            # Asset weights
    metrics: PortfolioMetrics            # Portfolio metrics
    method: OptimizationMethod           # Optimization method used
    constraints: OptimizationConstraints # Constraints applied
    rebalance_needed: bool               # Whether rebalancing is needed
    timestamp: datetime


class PositionSizer:
    """Advanced position sizing algorithms"""
    
    def __init__(self):
        self.default_risk_per_trade = 0.02  # 2% risk per trade
    
    def kelly_criterion(
        self,
        win_probability: float,
        avg_win: float,
        avg_loss: float,
        current_capital: float
    ) -> float:
        """Calculate optimal position size using Kelly Criterion"""
        if avg_loss == 0:
            return 0
        
        # Kelly fraction: f* = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_prob, q = loss_prob
        b = abs(avg_win / avg_loss)
        p = win_probability
        q = 1 - p
        
        kelly_fraction = (b * p - q) / b
        
        # Apply fractional Kelly to reduce volatility
        fractional_kelly = kelly_fraction * 0.25  # Use 25% of full Kelly
        
        # Cap at reasonable maximum
        position_fraction = min(fractional_kelly, 0.1)  # Max 10% per position
        
        return max(0, position_fraction * current_capital)
    
    def volatility_targeting(
        self,
        target_volatility: float,
        asset_volatility: float,
        current_capital: float,
        correlation: float = 1.0
    ) -> float:
        """Size position to achieve target portfolio volatility"""
        if asset_volatility == 0:
            return 0
        
        # Position size = (target_vol / asset_vol) * correlation_adjustment
        position_fraction = (target_volatility / asset_volatility) * correlation
        
        # Cap position size
        position_fraction = min(position_fraction, 0.2)  # Max 20%
        
        return max(0, position_fraction * current_capital)
    
    def risk_parity_size(
        self,
        asset_volatility: float,
        portfolio_volatilities: List[float],
        current_capital: float,
        target_risk_contribution: float = None
    ) -> float:
        """Calculate position size for risk parity allocation"""
        if not portfolio_volatilities or asset_volatility == 0:
            return 0
        
        # Equal risk contribution by default
        if target_risk_contribution is None:
            target_risk_contribution = 1.0 / len(portfolio_volatilities)
        
        # Inverse volatility weighting
        inverse_vol = 1.0 / asset_volatility
        sum_inverse_vol = sum(1.0 / vol for vol in portfolio_volatilities)
        
        position_fraction = (inverse_vol / sum_inverse_vol) * target_risk_contribution
        
        return position_fraction * current_capital
    
    def fixed_risk_size(
        self,
        entry_price: float,
        stop_loss: float,
        risk_per_trade: float,
        current_capital: float
    ) -> float:
        """Calculate position size based on fixed risk per trade"""
        if entry_price <= 0 or stop_loss <= 0:
            return 0
        
        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share == 0:
            return 0
        
        # Calculate position size
        risk_amount = current_capital * risk_per_trade
        position_size = risk_amount / risk_per_share
        
        return position_size
    
    def optimal_f(
        self,
        historical_returns: List[float],
        current_capital: float
    ) -> float:
        """Calculate optimal fraction using Optimal F method"""
        if not historical_returns:
            return 0
        
        # Find the largest loss
        largest_loss = min(historical_returns)
        
        if largest_loss >= 0:
            return 0  # No losses in history
        
        # Test different fractions to find optimal
        best_geomean = -np.inf
        best_fraction = 0
        
        for f in np.arange(0.01, 0.5, 0.01):  # Test 1% to 50%
            geomean = self._calculate_geometric_mean(historical_returns, f, largest_loss)
            
            if geomean > best_geomean:
                best_geomean = geomean
                best_fraction = f
        
        return best_fraction * current_capital
    
    def _calculate_geometric_mean(self, returns: List[float], fraction: float, largest_loss: float) -> float:
        """Calculate geometric mean for Optimal F"""
        hprs = []  # Holding Period Returns
        
        for ret in returns:
            # HPR = 1 + (fraction * return / largest_loss)
            hpr = 1 + (fraction * ret / abs(largest_loss))
            hprs.append(max(hpr, 0.01))  # Avoid negative HPRs
        
        # Geometric mean
        if len(hprs) == 0:
            return 0
        
        product = np.prod(hprs)
        return product ** (1.0 / len(hprs)) - 1


class PortfolioOptimizer:
    """Advanced portfolio optimization engine"""
    
    def __init__(self):
        self.position_sizer = PositionSizer()
        self.risk_free_rate = 0.02  # 2% risk-free rate
    
    async def optimize_portfolio(
        self,
        user: User,
        signals: List[Signal],
        method: OptimizationMethod = OptimizationMethod.MEAN_VARIANCE,
        constraints: OptimizationConstraints = None
    ) -> OptimizationResult:
        """Optimize portfolio allocation"""
        
        if constraints is None:
            constraints = OptimizationConstraints()
        
        try:
            # Get historical data for assets
            asset_data = await self._get_asset_data([s.symbol for s in signals])
            
            if not asset_data or len(asset_data) < 2:
                logger.warning("Insufficient asset data for optimization")
                return self._create_equal_weight_portfolio(signals, constraints)
            
            # Calculate expected returns and covariance matrix
            expected_returns_vec = self._calculate_expected_returns(asset_data, signals)
            cov_matrix = self._calculate_covariance_matrix(asset_data)
            
            # Apply optimization method
            if method == OptimizationMethod.MEAN_VARIANCE:
                weights = self._optimize_mean_variance(expected_returns_vec, cov_matrix, constraints)
            elif method == OptimizationMethod.BLACK_LITTERMAN:
                weights = self._optimize_black_litterman(asset_data, signals, constraints)
            elif method == OptimizationMethod.RISK_PARITY:
                weights = self._optimize_risk_parity(cov_matrix, constraints)
            elif method == OptimizationMethod.KELLY_CRITERION:
                weights = self._optimize_kelly(signals, user.portfolio_value, constraints)
            elif method == OptimizationMethod.MINIMUM_VARIANCE:
                weights = self._optimize_minimum_variance(cov_matrix, constraints)
            elif method == OptimizationMethod.MAXIMUM_SHARPE:
                weights = self._optimize_maximum_sharpe(expected_returns_vec, cov_matrix, constraints)
            else:
                weights = self._create_equal_weights([s.symbol for s in signals])
            
            # Calculate portfolio metrics
            metrics = self._calculate_portfolio_metrics(weights, expected_returns_vec, cov_matrix, asset_data)
            
            # Check if rebalancing is needed
            rebalance_needed = await self._check_rebalance_needed(user, weights)
            
            return OptimizationResult(
                weights=weights,
                metrics=metrics,
                method=method,
                constraints=constraints,
                rebalance_needed=rebalance_needed,
                timestamp=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            logger.error(f"Portfolio optimization failed: {e}")
            return self._create_equal_weight_portfolio(signals, constraints)
    
    async def _get_asset_data(self, symbols: List[str], days: int = 252) -> Optional[pd.DataFrame]:
        """Get historical asset data"""
        try:
            # Try cache first
            cache = get_cache()
            if cache:
                cached_data = await cache.get("portfolio", f"asset_data_{'_'.join(sorted(symbols))}")
                if cached_data:
                    return pd.DataFrame(cached_data)
            
            # Generate synthetic data for demonstration
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            dates = pd.date_range(start_date, end_date, freq='D')
            
            data = {}
            np.random.seed(42)  # For reproducible results
            
            for symbol in symbols:
                # Generate synthetic returns
                returns = np.random.normal(0.0008, 0.02, len(dates))  # ~20% annual vol
                prices = 100 * np.cumprod(1 + returns)
                data[symbol] = prices
            
            df = pd.DataFrame(data, index=dates)
            
            # Cache the data
            if cache:
                await cache.set("portfolio", f"asset_data_{'_'.join(sorted(symbols))}", 
                              df.to_dict(), ttl=3600)
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting asset data: {e}")
            return None
    
    def _calculate_expected_returns(self, asset_data: pd.DataFrame, signals: List[Signal]) -> np.ndarray:
        """Calculate expected returns incorporating signal views"""
        # Calculate historical returns
        returns = asset_data.pct_change().dropna()
        historical_means = returns.mean() * 252  # Annualize
        
        # Incorporate signal predictions
        expected_returns = historical_means.copy()
        
        for signal in signals:
            if signal.symbol in expected_returns.index:
                # Use signal confidence to adjust expected returns
                signal_return = (signal.target_price - signal.entry_price) / signal.entry_price
                confidence_weight = signal.confidence * 0.5  # Weight by confidence
                
                # Blend historical and signal-based returns
                expected_returns[signal.symbol] = (
                    historical_means[signal.symbol] * (1 - confidence_weight) +
                    signal_return * confidence_weight
                )
        
        return expected_returns.values
    
    def _calculate_covariance_matrix(self, asset_data: pd.DataFrame, method: RiskModel = RiskModel.LEDOIT_WOLF) -> np.ndarray:
        """Calculate covariance matrix with shrinkage"""
        returns = asset_data.pct_change().dropna()
        
        if method == RiskModel.SAMPLE_COV:
            return returns.cov().values * 252  # Annualize
        
        elif method == RiskModel.LEDOIT_WOLF:
            # Ledoit-Wolf shrinkage estimator
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf()
            cov_lw = lw.fit(returns.fillna(0)).covariance_
            return cov_lw * 252
        
        elif method == RiskModel.CONSTANT_CORRELATION:
            # Constant correlation model
            corr = returns.corr().fillna(0)
            avg_corr = corr.values[np.triu_indices_from(corr.values, k=1)].mean()
            
            # Create constant correlation matrix
            const_corr = np.full_like(corr.values, avg_corr)
            np.fill_diagonal(const_corr, 1.0)
            
            # Scale by volatilities
            vol = returns.std() * np.sqrt(252)
            cov_matrix = np.outer(vol, vol) * const_corr
            return cov_matrix
        
        else:
            return returns.cov().values * 252
    
    def _optimize_mean_variance(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        constraints: OptimizationConstraints
    ) -> Dict[str, float]:
        """Mean-variance optimization using CVXPY"""
        
        n_assets = len(expected_returns)
        weights = cp.Variable(n_assets)
        
        # Objective: maximize utility = expected_return - 0.5 * gamma * variance
        gamma = 2.0  # Risk aversion parameter
        portfolio_return = expected_returns.T @ weights
        portfolio_variance = cp.quad_form(weights, cov_matrix)
        utility = portfolio_return - 0.5 * gamma * portfolio_variance
        
        # Constraints
        constraints_list = [
            cp.sum(weights) == 1,  # Weights sum to 1
            weights >= constraints.min_weight,  # Minimum weight
            weights <= constraints.max_weight,  # Maximum weight
        ]
        
        # Solve optimization
        problem = cp.Problem(cp.Maximize(utility), constraints_list)
        problem.solve()
        
        if weights.value is None:
            logger.warning("Mean-variance optimization failed, using equal weights")
            return self._create_equal_weights([f"asset_{i}" for i in range(n_assets)])
        
        # Convert to dictionary
        weight_dict = {}
        for i, weight in enumerate(weights.value):
            weight_dict[f"asset_{i}"] = max(0, weight)
        
        return weight_dict
    
    def _optimize_risk_parity(self, cov_matrix: np.ndarray, constraints: OptimizationConstraints) -> Dict[str, float]:
        """Risk parity optimization"""
        n_assets = len(cov_matrix)
        
        # Initial guess - inverse volatility
        inv_vol = 1 / np.sqrt(np.diag(cov_matrix))
        initial_weights = inv_vol / np.sum(inv_vol)
        
        # Objective function: minimize sum of squared risk contributions
        def risk_parity_objective(weights):
            weights = np.array(weights)
            portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
            
            # Risk contributions
            risk_contrib = (weights * (cov_matrix @ weights)) / portfolio_vol
            target_contrib = portfolio_vol / n_assets
            
            # Sum of squared deviations from equal risk contribution
            return np.sum((risk_contrib - target_contrib) ** 2)
        
        # Constraints
        constraints_opt = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
        ]
        
        # Bounds
        bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n_assets)]
        
        # Optimize
        result = minimize(
            risk_parity_objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_opt
        )
        
        if result.success:
            weights = result.x
        else:
            logger.warning("Risk parity optimization failed, using equal weights")
            weights = np.ones(n_assets) / n_assets
        
        # Convert to dictionary
        weight_dict = {}
        for i, weight in enumerate(weights):
            weight_dict[f"asset_{i}"] = max(0, weight)
        
        return weight_dict
    
    def _optimize_kelly(
        self,
        signals: List[Signal],
        portfolio_value: float,
        constraints: OptimizationConstraints
    ) -> Dict[str, float]:
        """Kelly criterion optimization"""
        
        weights = {}
        total_allocation = 0
        
        for signal in signals:
            if signal.target_price and signal.entry_price and signal.stop_loss:
                # Calculate win probability from confidence
                win_prob = signal.confidence
                
                # Calculate average win/loss
                avg_win = (signal.target_price - signal.entry_price) / signal.entry_price
                avg_loss = (signal.entry_price - signal.stop_loss) / signal.entry_price
                
                # Kelly position size
                kelly_size = self.position_sizer.kelly_criterion(
                    win_prob, avg_win, avg_loss, portfolio_value
                )
                
                # Convert to weight
                weight = kelly_size / portfolio_value
                weight = min(weight, constraints.max_weight)  # Apply constraints
                
                weights[signal.symbol] = weight
                total_allocation += weight
        
        # Normalize if total allocation exceeds 100%
        if total_allocation > 1.0:
            for symbol in weights:
                weights[symbol] /= total_allocation
        
        return weights
    
    def _optimize_minimum_variance(self, cov_matrix: np.ndarray, constraints: OptimizationConstraints) -> Dict[str, float]:
        """Minimum variance optimization"""
        n_assets = len(cov_matrix)
        weights = cp.Variable(n_assets)
        
        # Objective: minimize portfolio variance
        portfolio_variance = cp.quad_form(weights, cov_matrix)
        
        # Constraints
        constraints_list = [
            cp.sum(weights) == 1,
            weights >= constraints.min_weight,
            weights <= constraints.max_weight,
        ]
        
        # Solve
        problem = cp.Problem(cp.Minimize(portfolio_variance), constraints_list)
        problem.solve()
        
        if weights.value is None:
            return self._create_equal_weights([f"asset_{i}" for i in range(n_assets)])
        
        # Convert to dictionary
        weight_dict = {}
        for i, weight in enumerate(weights.value):
            weight_dict[f"asset_{i}"] = max(0, weight)
        
        return weight_dict
    
    def _optimize_maximum_sharpe(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        constraints: OptimizationConstraints
    ) -> Dict[str, float]:
        """Maximum Sharpe ratio optimization"""
        
        n_assets = len(expected_returns)
        
        # Initial guess
        initial_weights = np.ones(n_assets) / n_assets
        
        # Objective function: negative Sharpe ratio
        def negative_sharpe(weights):
            weights = np.array(weights)
            portfolio_return = np.sum(expected_returns * weights)
            portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
            
            if portfolio_vol == 0:
                return -np.inf
            
            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol
            return -sharpe  # Minimize negative Sharpe
        
        # Constraints
        constraints_opt = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        ]
        
        # Bounds
        bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n_assets)]
        
        # Optimize
        result = minimize(
            negative_sharpe,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_opt
        )
        
        if result.success:
            weights = result.x
        else:
            weights = initial_weights
        
        # Convert to dictionary
        weight_dict = {}
        for i, weight in enumerate(weights):
            weight_dict[f"asset_{i}"] = max(0, weight)
        
        return weight_dict
    
    def _optimize_black_litterman(
        self,
        asset_data: pd.DataFrame,
        signals: List[Signal],
        constraints: OptimizationConstraints
    ) -> Dict[str, float]:
        """Black-Litterman optimization with signal views"""
        
        # This would implement Black-Litterman model
        # For now, fall back to mean-variance
        returns = asset_data.pct_change().dropna()
        expected_returns = returns.mean().values * 252
        cov_matrix = returns.cov().values * 252
        
        return self._optimize_mean_variance(expected_returns, cov_matrix, constraints)
    
    def _calculate_portfolio_metrics(
        self,
        weights: Dict[str, float],
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        asset_data: pd.DataFrame
    ) -> PortfolioMetrics:
        """Calculate comprehensive portfolio metrics"""
        
        weight_values = np.array(list(weights.values()))
        
        # Basic metrics
        portfolio_return = np.sum(expected_returns * weight_values)
        portfolio_variance = weight_values.T @ cov_matrix @ weight_values
        portfolio_vol = np.sqrt(portfolio_variance)
        
        # Sharpe ratio
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
        
        # Calculate historical portfolio returns for advanced metrics
        returns = asset_data.pct_change().dropna()
        portfolio_returns = returns.dot(weight_values)
        
        # VaR and CVaR
        var_95 = np.percentile(portfolio_returns, 5)
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
        
        # Maximum drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Sortino ratio (downside deviation)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_deviation = np.sqrt(np.mean(downside_returns ** 2)) if len(downside_returns) > 0 else 0
        sortino_ratio = (portfolio_return - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        
        # Calmar ratio
        calmar_ratio = portfolio_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return PortfolioMetrics(
            expected_return=portfolio_return,
            volatility=portfolio_vol,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=abs(max_drawdown),
            var_95=var_95,
            cvar_95=cvar_95,
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio,
            tracking_error=0,  # Would calculate vs benchmark
            information_ratio=0,  # Would calculate vs benchmark
            beta=1.0,  # Would calculate vs benchmark
            alpha=0.0  # Would calculate vs benchmark
        )
    
    async def _check_rebalance_needed(self, user: User, target_weights: Dict[str, float]) -> bool:
        """Check if portfolio rebalancing is needed"""
        # This would compare current portfolio weights with target weights
        # For now, assume rebalancing is always beneficial
        return True
    
    def _create_equal_weights(self, symbols: List[str]) -> Dict[str, float]:
        """Create equal weight allocation"""
        if not symbols:
            return {}
        
        weight = 1.0 / len(symbols)
        return {symbol: weight for symbol in symbols}
    
    def _create_equal_weight_portfolio(self, signals: List[Signal], constraints: OptimizationConstraints) -> OptimizationResult:
        """Create equal weight portfolio as fallback"""
        symbols = [s.symbol for s in signals]
        weights = self._create_equal_weights(symbols)
        
        # Mock metrics
        metrics = PortfolioMetrics(
            expected_return=0.08,
            volatility=0.15,
            sharpe_ratio=0.4,
            max_drawdown=0.1,
            var_95=-0.02,
            cvar_95=-0.03,
            calmar_ratio=0.8,
            sortino_ratio=0.5,
            tracking_error=0.05,
            information_ratio=0.2,
            beta=1.0,
            alpha=0.0
        )
        
        return OptimizationResult(
            weights=weights,
            metrics=metrics,
            method=OptimizationMethod.EQUAL_WEIGHT,
            constraints=constraints,
            rebalance_needed=True,
            timestamp=datetime.now(timezone.utc)
        )