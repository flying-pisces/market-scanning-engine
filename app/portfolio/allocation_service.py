"""
Portfolio Allocation Service
Manages user portfolio allocations and rebalancing
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone, timedelta

from app.portfolio.optimization import PortfolioOptimizer, OptimizationMethod, OptimizationConstraints, PositionSizer
from app.core.cache import get_cache
from app.core.kafka_client import get_producer, KafkaTopics
from app.database import get_db
from app.models.user import User
from app.models.signal import Signal, SignalStatus
from app.services.matching import MatchingService
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class PortfolioAllocationService:
    """Manages portfolio allocations and rebalancing for users"""
    
    def __init__(self):
        self.optimizer = PortfolioOptimizer()
        self.position_sizer = PositionSizer()
        self.matching_service = MatchingService()
        self.is_running = False
        
        # Service configuration
        self.rebalance_threshold = 0.05  # 5% deviation threshold for rebalancing
        self.max_positions = 20          # Maximum positions per portfolio
        self.min_position_size = 0.01    # Minimum 1% position size
    
    async def start(self):
        """Start the allocation service"""
        if self.is_running:
            return
        
        self.is_running = True
        logger.info("Portfolio Allocation Service started")
        
        # Start background tasks
        tasks = [
            self._monitor_portfolios(),
            self._process_rebalancing_queue(),
            self._update_position_sizes()
        ]
        
        await asyncio.gather(*tasks)
    
    async def stop(self):
        """Stop the allocation service"""
        self.is_running = False
        logger.info("Portfolio Allocation Service stopped")
    
    async def optimize_user_portfolio(
        self,
        user_id: int,
        method: OptimizationMethod = None,
        force_rebalance: bool = False
    ) -> Dict[str, Any]:
        """Optimize portfolio allocation for a specific user"""
        try:
            async with get_db() as db:
                # Get user
                user_result = await db.execute(select(User).where(User.id == user_id))
                user = user_result.scalar_one_or_none()
                
                if not user:
                    return {"error": "User not found"}
                
                # Get user's active signals
                signals = await self._get_user_signals(db, user_id)
                
                if not signals:
                    return {"message": "No active signals for optimization"}
                
                # Determine optimization method based on user risk tolerance
                if method is None:
                    method = self._select_optimization_method(user)
                
                # Create constraints based on user preferences
                constraints = self._create_user_constraints(user)
                
                # Perform optimization
                optimization_result = await self.optimizer.optimize_portfolio(
                    user, signals, method, constraints
                )
                
                # Store optimization result
                await self._store_optimization_result(user_id, optimization_result)
                
                # Check if rebalancing is needed
                if optimization_result.rebalance_needed or force_rebalance:
                    rebalance_plan = await self._create_rebalance_plan(user, optimization_result)
                    
                    # Publish rebalancing event
                    await self._publish_rebalance_event(user_id, rebalance_plan)
                    
                    return {
                        "optimization": optimization_result,
                        "rebalance_plan": rebalance_plan,
                        "status": "rebalance_recommended"
                    }
                else:
                    return {
                        "optimization": optimization_result,
                        "status": "portfolio_optimal"
                    }
                    
        except Exception as e:
            logger.error(f"Error optimizing portfolio for user {user_id}: {e}")
            return {"error": str(e)}
    
    async def calculate_position_sizes(
        self,
        user_id: int,
        signals: List[Signal],
        method: str = "kelly"
    ) -> Dict[str, Dict[str, float]]:
        """Calculate optimal position sizes for new signals"""
        try:
            async with get_db() as db:
                user_result = await db.execute(select(User).where(User.id == user_id))
                user = user_result.scalar_one_or_none()
                
                if not user:
                    return {"error": "User not found"}
                
                position_sizes = {}
                
                for signal in signals:
                    size_info = await self._calculate_signal_position_size(
                        user, signal, method
                    )
                    position_sizes[signal.symbol] = size_info
                
                return position_sizes
                
        except Exception as e:
            logger.error(f"Error calculating position sizes for user {user_id}: {e}")
            return {"error": str(e)}
    
    async def _get_user_signals(self, db: AsyncSession, user_id: int) -> List[Signal]:
        """Get active signals for a user"""
        # Get user's matched signals from the last 30 days
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=30)
        
        # This would normally query a user_signals table
        # For now, get recent active signals and filter by risk compatibility
        result = await db.execute(
            select(Signal)
            .where(Signal.status == SignalStatus.ACTIVE)
            .where(Signal.created_at >= cutoff_date)
            .limit(50)
        )
        
        all_signals = result.scalars().all()
        
        # Get user for risk filtering
        user_result = await db.execute(select(User).where(User.id == user_id))
        user = user_result.scalar_one_or_none()
        
        if not user:
            return []
        
        # Filter signals based on risk compatibility
        compatible_signals = []
        for signal in all_signals:
            # Use matching service to check compatibility
            matches = await self.matching_service.find_matches_for_signal(signal)
            user_matches = [m for m in matches if m.user_id == user_id]
            
            if user_matches and user_matches[0].match_score > 70:
                compatible_signals.append(signal)
        
        return compatible_signals[:20]  # Limit to top 20 signals
    
    def _select_optimization_method(self, user: User) -> OptimizationMethod:
        """Select optimization method based on user profile"""
        risk_tolerance = user.risk_tolerance
        
        if risk_tolerance <= 20:
            return OptimizationMethod.MINIMUM_VARIANCE
        elif risk_tolerance <= 40:
            return OptimizationMethod.RISK_PARITY
        elif risk_tolerance <= 60:
            return OptimizationMethod.MEAN_VARIANCE
        elif risk_tolerance <= 80:
            return OptimizationMethod.MAXIMUM_SHARPE
        else:
            return OptimizationMethod.KELLY_CRITERION
    
    def _create_user_constraints(self, user: User) -> OptimizationConstraints:
        """Create optimization constraints based on user preferences"""
        risk_tolerance = user.risk_tolerance
        
        # Conservative users get stricter constraints
        if risk_tolerance <= 30:
            max_weight = 0.15  # Max 15% per position
            max_positions = 15
        elif risk_tolerance <= 60:
            max_weight = 0.25  # Max 25% per position
            max_positions = 20
        else:
            max_weight = 0.35  # Max 35% per position
            max_positions = 25
        
        return OptimizationConstraints(
            max_weight=max_weight,
            min_weight=0.01,  # Min 1%
            max_positions=max_positions,
            sector_limits={},  # Could add sector diversification
            turnover_limit=0.3,  # Max 30% turnover
        )
    
    async def _calculate_signal_position_size(
        self,
        user: User,
        signal: Signal,
        method: str
    ) -> Dict[str, float]:
        """Calculate position size for a specific signal"""
        
        portfolio_value = user.portfolio_value or 10000  # Default $10k
        
        if method == "kelly":
            # Kelly criterion sizing
            win_prob = signal.confidence
            avg_win = (signal.target_price - signal.entry_price) / signal.entry_price if signal.target_price else 0.05
            avg_loss = (signal.entry_price - signal.stop_loss) / signal.entry_price if signal.stop_loss else 0.03
            
            position_value = self.position_sizer.kelly_criterion(
                win_prob, avg_win, avg_loss, portfolio_value
            )
            
        elif method == "fixed_risk":
            # Fixed risk per trade
            risk_per_trade = 0.02  # 2% risk
            position_value = self.position_sizer.fixed_risk_size(
                signal.entry_price, signal.stop_loss, risk_per_trade, portfolio_value
            )
            
        elif method == "volatility_target":
            # Target volatility sizing
            target_vol = 0.15  # 15% target volatility
            asset_vol = 0.25    # Estimated asset volatility (would calculate from data)
            position_value = self.position_sizer.volatility_targeting(
                target_vol, asset_vol, portfolio_value
            )
            
        else:
            # Default fixed percentage
            position_value = portfolio_value * 0.05  # 5% default
        
        # Calculate position details
        shares = position_value / signal.entry_price if signal.entry_price > 0 else 0
        weight = position_value / portfolio_value
        
        return {
            "position_value": position_value,
            "shares": shares,
            "weight": weight,
            "entry_price": signal.entry_price,
            "stop_loss": signal.stop_loss,
            "target_price": signal.target_price,
            "risk_amount": position_value * 0.02,  # 2% risk
            "method": method
        }
    
    async def _store_optimization_result(self, user_id: int, result):
        """Store optimization result in cache"""
        cache = get_cache()
        if not cache:
            return
        
        optimization_data = {
            "user_id": user_id,
            "weights": result.weights,
            "metrics": {
                "expected_return": result.metrics.expected_return,
                "volatility": result.metrics.volatility,
                "sharpe_ratio": result.metrics.sharpe_ratio,
                "max_drawdown": result.metrics.max_drawdown,
            },
            "method": result.method.value,
            "rebalance_needed": result.rebalance_needed,
            "timestamp": result.timestamp.isoformat()
        }
        
        await cache.set(
            "portfolio_optimization",
            str(user_id),
            optimization_data,
            ttl=86400  # 24 hours
        )
    
    async def _create_rebalance_plan(self, user: User, optimization_result) -> Dict[str, Any]:
        """Create detailed rebalancing plan"""
        
        # Get current portfolio weights (would query from user's positions)
        current_weights = await self._get_current_portfolio_weights(user.id)
        target_weights = optimization_result.weights
        
        trades = []
        total_turnover = 0
        
        for symbol, target_weight in target_weights.items():
            current_weight = current_weights.get(symbol, 0)
            weight_diff = target_weight - current_weight
            
            if abs(weight_diff) > self.min_position_size:  # Only rebalance if difference is significant
                trade_value = weight_diff * user.portfolio_value
                
                trade = {
                    "symbol": symbol,
                    "action": "buy" if weight_diff > 0 else "sell",
                    "current_weight": current_weight,
                    "target_weight": target_weight,
                    "weight_change": weight_diff,
                    "trade_value": abs(trade_value),
                    "shares": abs(trade_value) / 100,  # Assuming $100 per share for demo
                }
                
                trades.append(trade)
                total_turnover += abs(weight_diff)
        
        # Handle positions to close (not in target weights)
        for symbol, current_weight in current_weights.items():
            if symbol not in target_weights and current_weight > self.min_position_size:
                trade_value = current_weight * user.portfolio_value
                
                trade = {
                    "symbol": symbol,
                    "action": "sell",
                    "current_weight": current_weight,
                    "target_weight": 0,
                    "weight_change": -current_weight,
                    "trade_value": trade_value,
                    "shares": trade_value / 100,
                }
                
                trades.append(trade)
                total_turnover += current_weight
        
        return {
            "trades": trades,
            "total_turnover": total_turnover,
            "estimated_cost": len(trades) * 5.0,  # $5 per trade estimate
            "execution_priority": "high" if total_turnover > 0.2 else "normal",
            "expected_improvement": {
                "return_increase": 0.005,  # Mock improvement
                "risk_reduction": 0.002,
                "sharpe_improvement": 0.1
            }
        }
    
    async def _get_current_portfolio_weights(self, user_id: int) -> Dict[str, float]:
        """Get current portfolio weights for user"""
        # This would query the user's current positions
        # For now, return mock current weights
        return {
            "AAPL": 0.15,
            "GOOGL": 0.12,
            "MSFT": 0.10,
            "TSLA": 0.08,
            "SPY": 0.20
        }
    
    async def _publish_rebalance_event(self, user_id: int, rebalance_plan: Dict[str, Any]):
        """Publish rebalancing event"""
        producer = get_producer()
        if not producer:
            return
        
        event = {
            "type": "portfolio_rebalance_plan",
            "user_id": user_id,
            "trades_count": len(rebalance_plan["trades"]),
            "total_turnover": rebalance_plan["total_turnover"],
            "estimated_cost": rebalance_plan["estimated_cost"],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        await producer.send_message(
            KafkaTopics.USER_ACTIONS,
            event,
            key=f"rebalance_{user_id}"
        )
    
    async def _monitor_portfolios(self):
        """Monitor portfolios for rebalancing needs"""
        while self.is_running:
            try:
                async with get_db() as db:
                    # Get users with significant portfolio values
                    result = await db.execute(
                        select(User)
                        .where(User.portfolio_value > 1000)
                        .limit(100)
                    )
                    
                    users = result.scalars().all()
                    
                    for user in users:
                        try:
                            # Check if rebalancing is needed
                            needs_rebalance = await self._check_rebalance_needed(user)
                            
                            if needs_rebalance:
                                logger.info(f"Scheduling rebalancing for user {user.id}")
                                # Could add to rebalancing queue or trigger immediately
                                await self.optimize_user_portfolio(user.id)
                            
                        except Exception as e:
                            logger.error(f"Error checking rebalance for user {user.id}: {e}")
                
                # Check every 4 hours
                await asyncio.sleep(4 * 3600)
                
            except Exception as e:
                logger.error(f"Error in portfolio monitoring: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour on error
    
    async def _check_rebalance_needed(self, user: User) -> bool:
        """Check if user portfolio needs rebalancing"""
        try:
            # Get last optimization
            cache = get_cache()
            if not cache:
                return True
            
            last_optimization = await cache.get("portfolio_optimization", str(user.id))
            if not last_optimization:
                return True
            
            # Check if optimization is stale (older than 7 days)
            last_timestamp = datetime.fromisoformat(last_optimization["timestamp"])
            if (datetime.now(timezone.utc) - last_timestamp).days > 7:
                return True
            
            # Check if portfolio has drifted significantly
            current_weights = await self._get_current_portfolio_weights(user.id)
            target_weights = last_optimization["weights"]
            
            total_drift = 0
            for symbol, target_weight in target_weights.items():
                current_weight = current_weights.get(symbol, 0)
                total_drift += abs(target_weight - current_weight)
            
            return total_drift > self.rebalance_threshold
            
        except Exception as e:
            logger.error(f"Error checking rebalance need for user {user.id}: {e}")
            return False
    
    async def _process_rebalancing_queue(self):
        """Process rebalancing queue"""
        while self.is_running:
            try:
                # This would process queued rebalancing requests
                # For now, just a placeholder
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error processing rebalancing queue: {e}")
                await asyncio.sleep(300)
    
    async def _update_position_sizes(self):
        """Update position sizes based on new market data"""
        while self.is_running:
            try:
                # This would recalculate position sizes as market conditions change
                # For now, just a placeholder
                await asyncio.sleep(1800)  # Update every 30 minutes
                
            except Exception as e:
                logger.error(f"Error updating position sizes: {e}")
                await asyncio.sleep(1800)
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        return {
            "is_running": self.is_running,
            "rebalance_threshold": self.rebalance_threshold,
            "max_positions": self.max_positions,
            "min_position_size": self.min_position_size
        }


# Global allocation service instance
portfolio_allocation_service: Optional[PortfolioAllocationService] = None


async def init_portfolio_allocation_service() -> PortfolioAllocationService:
    """Initialize portfolio allocation service"""
    global portfolio_allocation_service
    
    portfolio_allocation_service = PortfolioAllocationService()
    logger.info("Portfolio Allocation Service initialized")
    return portfolio_allocation_service


def get_portfolio_allocation_service() -> Optional[PortfolioAllocationService]:
    """Get global portfolio allocation service instance"""
    return portfolio_allocation_service