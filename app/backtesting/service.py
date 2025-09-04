"""
Backtesting Service
Manages backtesting operations, results storage, and analysis
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone, timedelta
from dataclasses import asdict

from app.backtesting.engine import BacktestEngine, BacktestConfig, ExecutionModel
from app.core.cache import get_cache
from app.core.kafka_client import get_producer, KafkaTopics
from app.database import get_db
from app.models.signal import Signal, SignalStatus
from app.models.user import User
from sqlalchemy import select, and_

logger = logging.getLogger(__name__)


class BacktestingService:
    """Manages backtesting operations and results"""
    
    def __init__(self):
        self.is_running = False
        self.active_backtests: Dict[str, BacktestEngine] = {}
        self.backtest_queue: List[Dict[str, Any]] = []
        
    async def start(self):
        """Start the backtesting service"""
        if self.is_running:
            return
        
        self.is_running = True
        logger.info("Backtesting Service started")
        
        # Start background tasks
        tasks = [
            self._process_backtest_queue(),
            self._cleanup_old_results()
        ]
        
        await asyncio.gather(*tasks)
    
    async def stop(self):
        """Stop the backtesting service"""
        self.is_running = False
        logger.info("Backtesting Service stopped")
    
    async def run_strategy_backtest(
        self,
        strategy_name: str,
        signals: List[Signal],
        config: BacktestConfig,
        user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Run backtest for a trading strategy"""
        
        try:
            backtest_id = f"{strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            logger.info(f"Starting backtest {backtest_id}")
            
            # Create backtest engine
            engine = BacktestEngine(config)
            self.active_backtests[backtest_id] = engine
            
            # Run backtest
            result = await engine.run_backtest(signals, strategy_name)
            
            # Add metadata
            result.update({
                "backtest_id": backtest_id,
                "user_id": user_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "signals_count": len(signals)
            })
            
            # Store results
            await self._store_backtest_result(backtest_id, result)
            
            # Publish completion event
            await self._publish_backtest_event(backtest_id, result, "completed")
            
            # Clean up
            del self.active_backtests[backtest_id]
            
            return result
            
        except Exception as e:
            logger.error(f"Backtest failed for {strategy_name}: {e}")
            
            # Publish failure event
            error_result = {
                "backtest_id": backtest_id,
                "strategy_name": strategy_name,
                "error": str(e),
                "status": "failed"
            }
            
            await self._publish_backtest_event(backtest_id, error_result, "failed")
            
            # Clean up
            if backtest_id in self.active_backtests:
                del self.active_backtests[backtest_id]
            
            raise
    
    async def run_signal_backtest(
        self,
        signals: List[Signal],
        start_date: datetime,
        end_date: datetime,
        initial_capital: float = 100000,
        user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Run backtest for specific signals"""
        
        config = BacktestConfig(
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            execution_model=ExecutionModel.REALISTIC
        )
        
        return await self.run_strategy_backtest(
            "Signal Analysis",
            signals,
            config,
            user_id
        )
    
    async def run_user_strategy_backtest(
        self,
        user_id: int,
        days_back: int = 365,
        initial_capital: Optional[float] = None
    ) -> Dict[str, Any]:
        """Run backtest for user's historical signal matches"""
        
        try:
            async with get_db() as db:
                # Get user
                user_result = await db.execute(select(User).where(User.id == user_id))
                user = user_result.scalar_one_or_none()
                
                if not user:
                    return {"error": "User not found"}
                
                # Get user's historical signals
                end_date = datetime.now(timezone.utc)
                start_date = end_date - timedelta(days=days_back)
                
                signals = await self._get_user_signals(db, user_id, start_date, end_date)
                
                if not signals:
                    return {"error": "No historical signals found for user"}
                
                # Use user's portfolio value or default
                capital = initial_capital or user.portfolio_value or 100000
                
                # Create config based on user preferences
                config = BacktestConfig(
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=capital,
                    execution_model=ExecutionModel.REALISTIC,
                    max_positions=min(20, 30 - user.risk_tolerance // 10),  # Conservative users = fewer positions
                    commission=5.0,
                    slippage=0.001
                )
                
                return await self.run_strategy_backtest(
                    f"User_{user_id}_Strategy",
                    signals,
                    config,
                    user_id
                )
                
        except Exception as e:
            logger.error(f"Error running user backtest for {user_id}: {e}")
            return {"error": str(e)}
    
    async def queue_backtest(
        self,
        strategy_name: str,
        signal_ids: List[int],
        config_dict: Dict[str, Any],
        user_id: Optional[int] = None,
        priority: int = 1
    ) -> str:
        """Queue a backtest for background processing"""
        
        backtest_id = f"queued_{strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        backtest_job = {
            "id": backtest_id,
            "strategy_name": strategy_name,
            "signal_ids": signal_ids,
            "config": config_dict,
            "user_id": user_id,
            "priority": priority,
            "queued_at": datetime.now(timezone.utc),
            "status": "queued"
        }
        
        self.backtest_queue.append(backtest_job)
        self.backtest_queue.sort(key=lambda x: x["priority"], reverse=True)
        
        logger.info(f"Queued backtest {backtest_id}")
        
        return backtest_id
    
    async def get_backtest_result(self, backtest_id: str) -> Optional[Dict[str, Any]]:
        """Get stored backtest result"""
        
        cache = get_cache()
        if not cache:
            return None
        
        result = await cache.get("backtest_results", backtest_id)
        return result
    
    async def list_user_backtests(self, user_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """List backtests for a user"""
        
        cache = get_cache()
        if not cache:
            return []
        
        # This would normally query a database
        # For now, return mock recent backtests
        return [
            {
                "backtest_id": f"user_{user_id}_backtest_1",
                "strategy_name": "User Portfolio Strategy",
                "created_at": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
                "total_return": 0.15,
                "sharpe_ratio": 1.2,
                "max_drawdown": -0.08,
                "status": "completed"
            },
            {
                "backtest_id": f"user_{user_id}_backtest_2",
                "strategy_name": "Conservative Signals",
                "created_at": (datetime.now(timezone.utc) - timedelta(days=7)).isoformat(),
                "total_return": 0.08,
                "sharpe_ratio": 0.9,
                "max_drawdown": -0.05,
                "status": "completed"
            }
        ]
    
    async def compare_strategies(
        self,
        strategy_configs: List[Dict[str, Any]],
        user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Compare multiple strategies via backtesting"""
        
        comparison_results = {}
        
        for i, strategy_config in enumerate(strategy_configs):
            strategy_name = strategy_config.get("name", f"Strategy_{i+1}")
            
            # Extract signals and config
            signal_ids = strategy_config.get("signal_ids", [])
            config_dict = strategy_config.get("config", {})
            
            # Get signals from database
            signals = await self._get_signals_by_ids(signal_ids)
            
            if signals:
                # Create backtest config
                config = BacktestConfig(
                    start_date=datetime.fromisoformat(config_dict.get("start_date", "2023-01-01T00:00:00+00:00")),
                    end_date=datetime.fromisoformat(config_dict.get("end_date", "2024-01-01T00:00:00+00:00")),
                    initial_capital=config_dict.get("initial_capital", 100000),
                    execution_model=ExecutionModel(config_dict.get("execution_model", "realistic"))
                )
                
                # Run backtest
                result = await self.run_strategy_backtest(strategy_name, signals, config, user_id)
                comparison_results[strategy_name] = result
        
        # Create comparison summary
        summary = self._create_comparison_summary(comparison_results)
        
        return {
            "comparison_id": f"comp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "strategies": comparison_results,
            "summary": summary,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
    
    async def _get_user_signals(
        self,
        db,
        user_id: int,
        start_date: datetime,
        end_date: datetime
    ) -> List[Signal]:
        """Get signals that would have matched the user in the given period"""
        
        # This would query actual user signal matches
        # For now, get some recent signals and filter by risk compatibility
        
        result = await db.execute(
            select(Signal)
            .where(Signal.status == SignalStatus.ACTIVE)
            .where(Signal.created_at >= start_date)
            .where(Signal.created_at <= end_date)
            .limit(50)
        )
        
        all_signals = result.scalars().all()
        
        # Get user for risk filtering
        user_result = await db.execute(select(User).where(User.id == user_id))
        user = user_result.scalar_one_or_none()
        
        if not user:
            return []
        
        # Simple risk-based filtering
        user_risk = user.risk_tolerance
        compatible_signals = []
        
        for signal in all_signals:
            # Check if signal risk is compatible with user risk tolerance
            risk_diff = abs(signal.risk_score - user_risk)
            if risk_diff <= 20:  # Within 20 points
                compatible_signals.append(signal)
        
        return compatible_signals[:30]  # Limit to 30 signals
    
    async def _get_signals_by_ids(self, signal_ids: List[int]) -> List[Signal]:
        """Get signals by their IDs"""
        
        if not signal_ids:
            return []
        
        try:
            async with get_db() as db:
                result = await db.execute(
                    select(Signal).where(Signal.id.in_(signal_ids))
                )
                return result.scalars().all()
        except Exception as e:
            logger.error(f"Error getting signals by IDs: {e}")
            return []
    
    async def _store_backtest_result(self, backtest_id: str, result: Dict[str, Any]):
        """Store backtest result in cache"""
        
        cache = get_cache()
        if not cache:
            return
        
        # Store full result
        await cache.set(
            "backtest_results",
            backtest_id,
            result,
            ttl=86400 * 7  # 7 days
        )
        
        # Store summary for quick access
        summary = {
            "backtest_id": backtest_id,
            "strategy_name": result.get("strategy_name"),
            "total_return": result.get("metrics", {}).get("total_return"),
            "sharpe_ratio": result.get("metrics", {}).get("sharpe_ratio"),
            "max_drawdown": result.get("metrics", {}).get("max_drawdown"),
            "created_at": result.get("created_at"),
            "status": "completed"
        }
        
        await cache.set(
            "backtest_summaries",
            backtest_id,
            summary,
            ttl=86400 * 30  # 30 days
        )
    
    async def _publish_backtest_event(
        self,
        backtest_id: str,
        result: Dict[str, Any],
        event_type: str
    ):
        """Publish backtest event"""
        
        producer = get_producer()
        if not producer:
            return
        
        event = {
            "type": f"backtest_{event_type}",
            "backtest_id": backtest_id,
            "strategy_name": result.get("strategy_name"),
            "user_id": result.get("user_id"),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        if event_type == "completed":
            metrics = result.get("metrics", {})
            event.update({
                "total_return": metrics.get("total_return"),
                "sharpe_ratio": metrics.get("sharpe_ratio"),
                "max_drawdown": metrics.get("max_drawdown"),
                "total_trades": metrics.get("total_trades")
            })
        elif event_type == "failed":
            event["error"] = result.get("error")
        
        await producer.send_message(
            KafkaTopics.SYSTEM_METRICS,
            event,
            key=f"backtest_{backtest_id}"
        )
    
    def _create_comparison_summary(self, comparison_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Create comparison summary of multiple strategies"""
        
        if not comparison_results:
            return {}
        
        summaries = []
        for strategy_name, result in comparison_results.items():
            metrics = result.get("metrics", {})
            summaries.append({
                "strategy": strategy_name,
                "total_return": metrics.get("total_return", 0),
                "sharpe_ratio": metrics.get("sharpe_ratio", 0),
                "max_drawdown": metrics.get("max_drawdown", 0),
                "volatility": metrics.get("volatility", 0),
                "win_rate": metrics.get("win_rate", 0)
            })
        
        # Find best performing strategies
        best_return = max(summaries, key=lambda x: x["total_return"])
        best_sharpe = max(summaries, key=lambda x: x["sharpe_ratio"])
        lowest_drawdown = min(summaries, key=lambda x: x["max_drawdown"])
        
        return {
            "strategies_compared": len(summaries),
            "best_return": {
                "strategy": best_return["strategy"],
                "return": best_return["total_return"]
            },
            "best_sharpe": {
                "strategy": best_sharpe["strategy"],
                "sharpe_ratio": best_sharpe["sharpe_ratio"]
            },
            "lowest_drawdown": {
                "strategy": lowest_drawdown["strategy"],
                "max_drawdown": lowest_drawdown["max_drawdown"]
            },
            "average_return": sum(s["total_return"] for s in summaries) / len(summaries),
            "average_sharpe": sum(s["sharpe_ratio"] for s in summaries) / len(summaries)
        }
    
    async def _process_backtest_queue(self):
        """Process queued backtests"""
        
        while self.is_running:
            try:
                if self.backtest_queue and len(self.active_backtests) < 3:  # Max 3 concurrent backtests
                    job = self.backtest_queue.pop(0)
                    
                    # Update status
                    job["status"] = "running"
                    job["started_at"] = datetime.now(timezone.utc)
                    
                    # Get signals
                    signals = await self._get_signals_by_ids(job["signal_ids"])
                    
                    if signals:
                        # Create config
                        config = BacktestConfig(
                            **job["config"]
                        )
                        
                        # Run backtest in background
                        asyncio.create_task(
                            self._run_queued_backtest(job, signals, config)
                        )
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error processing backtest queue: {e}")
                await asyncio.sleep(30)
    
    async def _run_queued_backtest(
        self,
        job: Dict[str, Any],
        signals: List[Signal],
        config: BacktestConfig
    ):
        """Run a queued backtest"""
        
        try:
            result = await self.run_strategy_backtest(
                job["strategy_name"],
                signals,
                config,
                job["user_id"]
            )
            
            job["status"] = "completed"
            job["completed_at"] = datetime.now(timezone.utc)
            job["result_id"] = result["backtest_id"]
            
            logger.info(f"Completed queued backtest {job['id']}")
            
        except Exception as e:
            job["status"] = "failed"
            job["error"] = str(e)
            job["completed_at"] = datetime.now(timezone.utc)
            
            logger.error(f"Queued backtest {job['id']} failed: {e}")
    
    async def _cleanup_old_results(self):
        """Clean up old backtest results"""
        
        while self.is_running:
            try:
                # This would clean up old cached results
                # For now, just log
                logger.debug("Cleaning up old backtest results")
                
                await asyncio.sleep(86400)  # Daily cleanup
                
            except Exception as e:
                logger.error(f"Error in backtest cleanup: {e}")
                await asyncio.sleep(86400)
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        
        return {
            "is_running": self.is_running,
            "active_backtests": len(self.active_backtests),
            "queued_backtests": len(self.backtest_queue),
            "queue_priorities": [job["priority"] for job in self.backtest_queue]
        }


# Global backtesting service
backtesting_service: Optional[BacktestingService] = None


async def init_backtesting_service() -> BacktestingService:
    """Initialize backtesting service"""
    global backtesting_service
    
    backtesting_service = BacktestingService()
    logger.info("Backtesting Service initialized")
    return backtesting_service


def get_backtesting_service() -> Optional[BacktestingService]:
    """Get global backtesting service instance"""
    return backtesting_service