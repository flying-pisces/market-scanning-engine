"""
Comprehensive backtesting framework
Tests trading strategies against historical data with realistic execution modeling
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json

from app.models.signal import Signal, SignalType, AssetClass, TimeFrame
from app.portfolio.optimization import PortfolioOptimizer, OptimizationMethod
from app.core.cache import get_cache

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order execution types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class ExecutionModel(Enum):
    """Trade execution models"""
    PERFECT = "perfect"           # Perfect execution at signal price
    REALISTIC = "realistic"       # Account for slippage and delays
    PESSIMISTIC = "pessimistic"   # Conservative execution assumptions


@dataclass
class Position:
    """Trading position"""
    symbol: str
    shares: float
    entry_price: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    target_price: Optional[float] = None
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    
    def update_price(self, price: float, timestamp: datetime):
        """Update current price and P&L"""
        self.current_price = price
        self.unrealized_pnl = (price - self.entry_price) * self.shares


@dataclass
class Trade:
    """Completed trade record"""
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    shares: float
    pnl: float
    pnl_pct: float
    holding_period: float  # in days
    signal_id: str
    exit_reason: str  # "target", "stop", "time", "manual"


@dataclass
class BacktestMetrics:
    """Backtest performance metrics"""
    # Returns
    total_return: float
    annualized_return: float
    excess_return: float
    
    # Risk metrics
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    avg_trade: float
    profit_factor: float
    
    # Advanced metrics
    var_95: float
    cvar_95: float
    tail_ratio: float
    common_sense_ratio: float
    gain_to_pain_ratio: float
    
    # Time-based
    start_date: datetime
    end_date: datetime
    days: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "total_trades": self.total_trades,
            "profit_factor": self.profit_factor
        }


@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    start_date: datetime
    end_date: datetime
    initial_capital: float = 100000
    commission: float = 5.0         # Per trade
    slippage: float = 0.001         # 0.1% slippage
    execution_model: ExecutionModel = ExecutionModel.REALISTIC
    max_positions: int = 20
    position_sizing: str = "equal"   # "equal", "kelly", "risk_parity"
    rebalance_frequency: str = "monthly"  # "daily", "weekly", "monthly"
    benchmark: str = "SPY"
    risk_free_rate: float = 0.02


class MarketDataProvider:
    """Provides historical market data for backtesting"""
    
    def __init__(self):
        self.data_cache: Dict[str, pd.DataFrame] = {}
    
    async def get_price_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        frequency: str = "daily"
    ) -> Optional[pd.DataFrame]:
        """Get historical price data"""
        
        cache_key = f"{symbol}_{start_date.date()}_{end_date.date()}_{frequency}"
        
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        
        # Generate synthetic data for demonstration
        dates = pd.date_range(start_date, end_date, freq='D')
        n_points = len(dates)
        
        # Random walk with trend
        np.random.seed(hash(symbol) % (2**32))
        returns = np.random.normal(0.0005, 0.02, n_points)  # ~12% annual return, 30% vol
        prices = 100 * np.cumprod(1 + returns)
        
        # Generate OHLCV data
        high_factor = np.random.uniform(1.0, 1.03, n_points)
        low_factor = np.random.uniform(0.97, 1.0, n_points)
        
        data = pd.DataFrame({
            'open': prices * np.random.uniform(0.99, 1.01, n_points),
            'high': prices * high_factor,
            'low': prices * low_factor,
            'close': prices,
            'volume': np.random.randint(100000, 10000000, n_points),
            'adj_close': prices
        }, index=dates)
        
        self.data_cache[cache_key] = data
        return data
    
    async def get_multiple_symbols(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, pd.DataFrame]:
        """Get data for multiple symbols"""
        data = {}
        
        for symbol in symbols:
            symbol_data = await self.get_price_data(symbol, start_date, end_date)
            if symbol_data is not None:
                data[symbol] = symbol_data
        
        return data


class ExecutionEngine:
    """Handles trade execution with realistic modeling"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
    
    def execute_order(
        self,
        symbol: str,
        shares: float,
        order_type: OrderType,
        timestamp: datetime,
        market_data: pd.Series,
        limit_price: Optional[float] = None
    ) -> Tuple[float, float]:  # (executed_price, executed_shares)
        """Execute trade order with execution modeling"""
        
        if self.config.execution_model == ExecutionModel.PERFECT:
            # Perfect execution
            if order_type == OrderType.MARKET:
                executed_price = market_data['close']
            else:
                executed_price = limit_price or market_data['close']
            
            return executed_price, shares
        
        elif self.config.execution_model == ExecutionModel.REALISTIC:
            # Add slippage and partial fills
            base_price = market_data['close']
            
            if order_type == OrderType.MARKET:
                # Market order with slippage
                slippage_factor = 1 + (self.config.slippage if shares > 0 else -self.config.slippage)
                executed_price = base_price * slippage_factor
                
                # Partial fill based on volume
                available_volume = market_data.get('volume', 1000000)
                max_shares = available_volume * 0.1  # Can trade up to 10% of volume
                executed_shares = min(abs(shares), max_shares) * (1 if shares > 0 else -1)
                
                return executed_price, executed_shares
            
            else:
                # Limit order execution
                if limit_price is None:
                    return base_price, shares
                
                # Check if limit price would be hit
                high_price = market_data['high']
                low_price = market_data['low']
                
                if shares > 0:  # Buy order
                    if limit_price >= low_price:
                        executed_price = min(limit_price, high_price)
                        return executed_price, shares
                else:  # Sell order
                    if limit_price <= high_price:
                        executed_price = max(limit_price, low_price)
                        return executed_price, shares
                
                # Order not filled
                return 0.0, 0.0
        
        else:  # PESSIMISTIC
            # Worst-case execution
            base_price = market_data['close']
            slippage_factor = 1 + (self.config.slippage * 2 if shares > 0 else -self.config.slippage * 2)
            executed_price = base_price * slippage_factor
            
            # Reduce executed shares
            executed_shares = shares * 0.9  # Only 90% filled
            
            return executed_price, executed_shares


class BacktestEngine:
    """Main backtesting engine"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.data_provider = MarketDataProvider()
        self.execution_engine = ExecutionEngine(config)
        self.portfolio_optimizer = PortfolioOptimizer()
        
        # Portfolio state
        self.cash = config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.portfolio_value = config.initial_capital
        self.trades: List[Trade] = []
        
        # Performance tracking
        self.daily_returns: List[float] = []
        self.daily_portfolio_values: List[float] = []
        self.daily_dates: List[datetime] = []
        
        # Benchmark data
        self.benchmark_data: Optional[pd.DataFrame] = None
    
    async def run_backtest(
        self,
        signals: List[Signal],
        strategy_name: str = "Signal Strategy"
    ) -> Dict[str, Any]:
        """Run complete backtest"""
        
        try:
            logger.info(f"Starting backtest: {strategy_name}")
            logger.info(f"Period: {self.config.start_date.date()} to {self.config.end_date.date()}")
            logger.info(f"Initial Capital: ${self.config.initial_capital:,.0f}")
            
            # Prepare data
            await self._prepare_data(signals)
            
            # Run simulation
            await self._run_simulation(signals)
            
            # Calculate metrics
            metrics = self._calculate_metrics()
            
            # Create result summary
            result = {
                "strategy_name": strategy_name,
                "config": self.config.__dict__,
                "metrics": metrics,
                "trades": [trade.__dict__ for trade in self.trades],
                "portfolio_timeline": {
                    "dates": [d.isoformat() for d in self.daily_dates],
                    "values": self.daily_portfolio_values,
                    "returns": self.daily_returns
                },
                "final_portfolio_value": self.portfolio_value,
                "total_pnl": self.portfolio_value - self.config.initial_capital,
                "positions": {symbol: pos.__dict__ for symbol, pos in self.positions.items()}
            }
            
            logger.info(f"Backtest completed: {metrics.total_return:.1%} return, "
                       f"{metrics.sharpe_ratio:.2f} Sharpe, {metrics.max_drawdown:.1%} max drawdown")
            
            return result
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            raise
    
    async def _prepare_data(self, signals: List[Signal]):
        """Prepare market data for backtesting"""
        
        # Get unique symbols
        symbols = list(set([signal.symbol for signal in signals]))
        
        # Add benchmark
        if self.config.benchmark not in symbols:
            symbols.append(self.config.benchmark)
        
        logger.info(f"Loading data for {len(symbols)} symbols")
        
        # Load market data
        market_data = await self.data_provider.get_multiple_symbols(
            symbols, self.config.start_date, self.config.end_date
        )
        
        self.market_data = market_data
        
        # Load benchmark data
        if self.config.benchmark in market_data:
            self.benchmark_data = market_data[self.config.benchmark]
    
    async def _run_simulation(self, signals: List[Signal]):
        """Run the main simulation loop"""
        
        # Get all trading days
        if not self.market_data:
            raise ValueError("No market data available")
        
        # Use the first symbol's data for trading days
        first_symbol = list(self.market_data.keys())[0]
        trading_days = self.market_data[first_symbol].index
        
        # Sort signals by time
        signals_by_time = {}
        for signal in signals:
            if hasattr(signal, 'created_at') and signal.created_at:
                signal_date = signal.created_at.date()
            else:
                # Use start date for demo signals
                signal_date = self.config.start_date.date()
            
            if signal_date not in signals_by_time:
                signals_by_time[signal_date] = []
            signals_by_time[signal_date].append(signal)
        
        # Simulate each trading day
        for current_date in trading_days:
            await self._process_trading_day(current_date, signals_by_time.get(current_date.date(), []))
    
    async def _process_trading_day(self, current_date: pd.Timestamp, new_signals: List[Signal]):
        """Process a single trading day"""
        
        current_datetime = current_date.to_pydatetime().replace(tzinfo=timezone.utc)
        
        # Update existing positions
        await self._update_positions(current_date)
        
        # Check for position exits
        await self._check_exits(current_date)
        
        # Process new signals
        if new_signals:
            await self._process_new_signals(current_date, new_signals)
        
        # Rebalancing (if configured)
        await self._check_rebalancing(current_date)
        
        # Update portfolio value and record performance
        self._update_portfolio_value(current_date)
        
        # Record daily performance
        self.daily_dates.append(current_datetime)
        self.daily_portfolio_values.append(self.portfolio_value)
        
        # Calculate daily return
        if len(self.daily_portfolio_values) > 1:
            prev_value = self.daily_portfolio_values[-2]
            daily_return = (self.portfolio_value - prev_value) / prev_value
            self.daily_returns.append(daily_return)
        else:
            self.daily_returns.append(0.0)
    
    async def _update_positions(self, current_date: pd.Timestamp):
        """Update existing positions with current prices"""
        
        for symbol, position in self.positions.items():
            if symbol in self.market_data:
                current_price = self.market_data[symbol].loc[current_date, 'close']
                position.update_price(current_price, current_date.to_pydatetime())
    
    async def _check_exits(self, current_date: pd.Timestamp):
        """Check for position exits (stops, targets, time-based)"""
        
        positions_to_close = []
        
        for symbol, position in self.positions.items():
            if symbol not in self.market_data:
                continue
            
            market_data = self.market_data[symbol].loc[current_date]
            
            exit_reason = None
            exit_price = None
            
            # Check stop loss
            if position.stop_loss:
                if position.shares > 0 and market_data['low'] <= position.stop_loss:
                    exit_reason = "stop_loss"
                    exit_price = position.stop_loss
                elif position.shares < 0 and market_data['high'] >= position.stop_loss:
                    exit_reason = "stop_loss"
                    exit_price = position.stop_loss
            
            # Check target price
            if not exit_reason and position.target_price:
                if position.shares > 0 and market_data['high'] >= position.target_price:
                    exit_reason = "target"
                    exit_price = position.target_price
                elif position.shares < 0 and market_data['low'] <= position.target_price:
                    exit_reason = "target"
                    exit_price = position.target_price
            
            # Time-based exit (30 days max holding period)
            if not exit_reason:
                holding_days = (current_date.to_pydatetime() - position.entry_time).days
                if holding_days > 30:
                    exit_reason = "time_limit"
                    exit_price = market_data['close']
            
            if exit_reason:
                positions_to_close.append((symbol, exit_reason, exit_price, current_date))
        
        # Close positions
        for symbol, exit_reason, exit_price, exit_date in positions_to_close:
            await self._close_position(symbol, exit_reason, exit_price, exit_date)
    
    async def _close_position(
        self,
        symbol: str,
        exit_reason: str,
        exit_price: float,
        exit_date: pd.Timestamp
    ):
        """Close a position and record the trade"""
        
        position = self.positions[symbol]
        
        # Execute exit order
        executed_price, executed_shares = self.execution_engine.execute_order(
            symbol, -position.shares, OrderType.MARKET,
            exit_date.to_pydatetime(), self.market_data[symbol].loc[exit_date]
        )
        
        # Calculate P&L
        pnl = (executed_price - position.entry_price) * position.shares
        pnl_pct = (executed_price / position.entry_price - 1) * (1 if position.shares > 0 else -1)
        
        # Account for commission
        pnl -= self.config.commission
        
        # Update cash
        self.cash += executed_price * position.shares + pnl
        
        # Record trade
        trade = Trade(
            symbol=symbol,
            entry_time=position.entry_time,
            exit_time=exit_date.to_pydatetime(),
            entry_price=position.entry_price,
            exit_price=executed_price,
            shares=position.shares,
            pnl=pnl,
            pnl_pct=pnl_pct,
            holding_period=(exit_date.to_pydatetime() - position.entry_time).days,
            signal_id=f"{symbol}_{position.entry_time.strftime('%Y%m%d')}",
            exit_reason=exit_reason
        )
        
        self.trades.append(trade)
        
        # Remove position
        del self.positions[symbol]
        
        logger.debug(f"Closed {symbol}: {pnl:.2f} P&L ({pnl_pct:.1%}) - {exit_reason}")
    
    async def _process_new_signals(self, current_date: pd.Timestamp, signals: List[Signal]):
        """Process new trading signals"""
        
        for signal in signals:
            # Skip if already have position in this symbol
            if signal.symbol in self.positions:
                continue
            
            # Skip if no market data
            if signal.symbol not in self.market_data:
                continue
            
            # Calculate position size
            position_size = self._calculate_position_size(signal, current_date)
            
            if abs(position_size) < self.config.initial_capital * 0.01:  # Skip very small positions
                continue
            
            # Execute entry order
            shares = position_size / signal.entry_price
            executed_price, executed_shares = self.execution_engine.execute_order(
                signal.symbol, shares, OrderType.MARKET,
                current_date.to_pydatetime(), self.market_data[signal.symbol].loc[current_date]
            )
            
            if executed_shares != 0:
                # Create position
                position = Position(
                    symbol=signal.symbol,
                    shares=executed_shares,
                    entry_price=executed_price,
                    entry_time=current_date.to_pydatetime(),
                    stop_loss=signal.stop_loss,
                    target_price=signal.target_price
                )
                
                self.positions[signal.symbol] = position
                
                # Update cash
                cost = executed_price * executed_shares + self.config.commission
                self.cash -= cost
                
                logger.debug(f"Opened {signal.symbol}: {executed_shares:.0f} shares @ ${executed_price:.2f}")
    
    def _calculate_position_size(self, signal: Signal, current_date: pd.Timestamp) -> float:
        """Calculate position size for new signal"""
        
        if self.config.position_sizing == "equal":
            # Equal weight allocation
            target_positions = min(len(self.positions) + 1, self.config.max_positions)
            return self.portfolio_value / target_positions
        
        elif self.config.position_sizing == "kelly":
            # Kelly criterion sizing
            win_prob = signal.confidence
            avg_win = (signal.target_price - signal.entry_price) / signal.entry_price if signal.target_price else 0.1
            avg_loss = (signal.entry_price - signal.stop_loss) / signal.entry_price if signal.stop_loss else 0.05
            
            # Kelly fraction
            kelly_fraction = (win_prob * avg_win - (1 - win_prob) * avg_loss) / avg_win if avg_win > 0 else 0
            kelly_fraction = max(0, min(kelly_fraction * 0.25, 0.1))  # Fractional Kelly, max 10%
            
            return self.portfolio_value * kelly_fraction
        
        else:  # Default to equal weight
            return self.portfolio_value / self.config.max_positions
    
    async def _check_rebalancing(self, current_date: pd.Timestamp):
        """Check if portfolio rebalancing is needed"""
        
        # Simple monthly rebalancing
        if self.config.rebalance_frequency == "monthly":
            if current_date.day == 1:  # First trading day of month
                await self._rebalance_portfolio(current_date)
    
    async def _rebalance_portfolio(self, current_date: pd.Timestamp):
        """Rebalance portfolio to target weights"""
        
        # This would implement portfolio rebalancing logic
        # For now, just log the event
        logger.debug(f"Rebalancing portfolio on {current_date.date()}")
    
    def _update_portfolio_value(self, current_date: pd.Timestamp):
        """Update total portfolio value"""
        
        positions_value = 0
        for symbol, position in self.positions.items():
            if symbol in self.market_data:
                current_price = self.market_data[symbol].loc[current_date, 'close']
                positions_value += position.shares * current_price
        
        self.portfolio_value = self.cash + positions_value
    
    def _calculate_metrics(self) -> BacktestMetrics:
        """Calculate comprehensive backtest metrics"""
        
        if len(self.daily_returns) == 0:
            return self._create_empty_metrics()
        
        # Convert to numpy arrays
        returns = np.array(self.daily_returns)
        portfolio_values = np.array(self.daily_portfolio_values)
        
        # Basic return metrics
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        days = len(returns)
        annualized_return = (1 + total_return) ** (252 / days) - 1
        
        # Risk metrics
        volatility = np.std(returns) * np.sqrt(252)
        sharpe_ratio = (annualized_return - self.config.risk_free_rate) / volatility if volatility > 0 else 0
        
        # Drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_deviation = np.sqrt(np.mean(downside_returns**2)) * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (annualized_return - self.config.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Trade statistics
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]
        
        total_trades = len(self.trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # VaR and CVaR
        var_95 = np.percentile(returns, 5)
        cvar_95 = np.mean(returns[returns <= var_95]) if np.sum(returns <= var_95) > 0 else var_95
        
        return BacktestMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            excess_return=annualized_return - self.config.risk_free_rate,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=abs(max_drawdown),
            max_drawdown_duration=0,  # Would calculate actual duration
            total_trades=total_trades,
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_trade=np.mean([t.pnl for t in self.trades]) if self.trades else 0,
            profit_factor=profit_factor,
            var_95=var_95,
            cvar_95=cvar_95,
            tail_ratio=0,  # Would calculate tail ratio
            common_sense_ratio=0,  # Would calculate common sense ratio
            gain_to_pain_ratio=0,  # Would calculate gain to pain ratio
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            days=days
        )
    
    def _create_empty_metrics(self) -> BacktestMetrics:
        """Create empty metrics for failed backtests"""
        return BacktestMetrics(
            total_return=0, annualized_return=0, excess_return=0,
            volatility=0, sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0,
            max_drawdown=0, max_drawdown_duration=0,
            total_trades=0, winning_trades=0, losing_trades=0,
            win_rate=0, avg_win=0, avg_loss=0, avg_trade=0, profit_factor=0,
            var_95=0, cvar_95=0, tail_ratio=0, common_sense_ratio=0, gain_to_pain_ratio=0,
            start_date=self.config.start_date, end_date=self.config.end_date, days=0
        )