"""
Background job processing system
Handles batch operations, cleanup tasks, and scheduled processes
"""

import asyncio
import logging
import json
from typing import Dict, List, Any, Callable, Optional
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from enum import Enum
import uuid

from app.core.kafka_client import get_producer, KafkaTopics
from app.core.cache import get_cache, get_signal_cache
from app.database import get_db
from app.models.signal import Signal, SignalStatus
from app.models.user import User
from app.services.matching import MatchingService
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete, update

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Job execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


class JobPriority(Enum):
    """Job priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class JobResult:
    """Job execution result"""
    job_id: str
    status: JobStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    duration: Optional[float] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class JobDefinition:
    """Background job definition"""
    job_id: str
    name: str
    handler: Callable
    args: List[Any]
    kwargs: Dict[str, Any]
    priority: JobPriority = JobPriority.NORMAL
    max_retries: int = 3
    retry_delay: float = 60.0  # seconds
    timeout: Optional[float] = None
    scheduled_at: Optional[datetime] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)


class BackgroundJobProcessor:
    """Background job processing system"""
    
    def __init__(self, max_concurrent_jobs: int = 5, poll_interval: float = 1.0):
        self.max_concurrent_jobs = max_concurrent_jobs
        self.poll_interval = poll_interval
        self.is_running = False
        
        # Job queues by priority
        self.job_queues: Dict[JobPriority, List[JobDefinition]] = {
            priority: [] for priority in JobPriority
        }
        
        # Active jobs
        self.active_jobs: Dict[str, asyncio.Task] = {}
        
        # Job results cache
        self.job_results: Dict[str, JobResult] = {}
        
        # Job handlers registry
        self.handlers: Dict[str, Callable] = {}
        
        # Register default handlers
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """Register built-in job handlers"""
        self.handlers.update({
            "cleanup_expired_signals": self._cleanup_expired_signals,
            "update_signal_matches": self._update_signal_matches,
            "calculate_user_stats": self._calculate_user_stats,
            "process_market_close": self._process_market_close,
            "generate_daily_summary": self._generate_daily_summary,
            "backup_data": self._backup_data,
            "health_check": self._health_check,
            "cache_warmup": self._cache_warmup
        })
    
    async def start(self):
        """Start the job processor"""
        if self.is_running:
            return
        
        self.is_running = True
        logger.info("Background job processor started")
        
        # Start main processing loop
        await self._process_jobs()
    
    async def stop(self):
        """Stop the job processor"""
        self.is_running = False
        
        # Cancel active jobs
        for job_id, task in self.active_jobs.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                logger.info(f"Cancelled job {job_id}")
        
        self.active_jobs.clear()
        logger.info("Background job processor stopped")
    
    def add_job(
        self,
        name: str,
        handler_name: str,
        args: List[Any] = None,
        kwargs: Dict[str, Any] = None,
        priority: JobPriority = JobPriority.NORMAL,
        max_retries: int = 3,
        timeout: Optional[float] = None,
        scheduled_at: Optional[datetime] = None
    ) -> str:
        """Add a job to the queue"""
        
        if handler_name not in self.handlers:
            raise ValueError(f"Unknown job handler: {handler_name}")
        
        job_id = str(uuid.uuid4())
        job = JobDefinition(
            job_id=job_id,
            name=name,
            handler=self.handlers[handler_name],
            args=args or [],
            kwargs=kwargs or {},
            priority=priority,
            max_retries=max_retries,
            timeout=timeout,
            scheduled_at=scheduled_at
        )
        
        # Add to appropriate queue
        self.job_queues[priority].append(job)
        
        logger.info(f"Added job {name} ({job_id}) with priority {priority.name}")
        return job_id
    
    def schedule_job(
        self,
        name: str,
        handler_name: str,
        run_at: datetime,
        args: List[Any] = None,
        kwargs: Dict[str, Any] = None,
        priority: JobPriority = JobPriority.NORMAL
    ) -> str:
        """Schedule a job to run at specific time"""
        
        return self.add_job(
            name=name,
            handler_name=handler_name,
            args=args,
            kwargs=kwargs,
            priority=priority,
            scheduled_at=run_at
        )
    
    def get_job_result(self, job_id: str) -> Optional[JobResult]:
        """Get job execution result"""
        return self.job_results.get(job_id)
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get job queue status"""
        return {
            "active_jobs": len(self.active_jobs),
            "pending_jobs": {
                priority.name: len(jobs) 
                for priority, jobs in self.job_queues.items()
            },
            "total_pending": sum(len(jobs) for jobs in self.job_queues.values()),
            "max_concurrent": self.max_concurrent_jobs,
            "is_running": self.is_running
        }
    
    async def _process_jobs(self):
        """Main job processing loop"""
        while self.is_running:
            try:
                # Check for scheduled jobs that are ready to run
                await self._process_scheduled_jobs()
                
                # Process pending jobs if we have capacity
                if len(self.active_jobs) < self.max_concurrent_jobs:
                    job = self._get_next_job()
                    if job:
                        await self._execute_job(job)
                
                # Clean up completed jobs
                await self._cleanup_completed_jobs()
                
                await asyncio.sleep(self.poll_interval)
                
            except Exception as e:
                logger.error(f"Error in job processing loop: {e}")
                await asyncio.sleep(self.poll_interval * 2)  # Back off on error
    
    async def _process_scheduled_jobs(self):
        """Move scheduled jobs to ready queue when their time comes"""
        current_time = datetime.now(timezone.utc)
        
        for priority, jobs in self.job_queues.items():
            ready_jobs = []
            remaining_jobs = []
            
            for job in jobs:
                if job.scheduled_at and job.scheduled_at <= current_time:
                    job.scheduled_at = None  # Clear schedule time
                    ready_jobs.append(job)
                elif job.scheduled_at is None:
                    ready_jobs.append(job)
                else:
                    remaining_jobs.append(job)
            
            # Update queue with ready jobs first (higher priority)
            self.job_queues[priority] = ready_jobs + remaining_jobs
    
    def _get_next_job(self) -> Optional[JobDefinition]:
        """Get the next job to execute (highest priority first)"""
        for priority in sorted(JobPriority, key=lambda x: x.value, reverse=True):
            jobs = self.job_queues[priority]
            if jobs:
                return jobs.pop(0)
        return None
    
    async def _execute_job(self, job: JobDefinition):
        """Execute a job"""
        job_result = JobResult(
            job_id=job.job_id,
            status=JobStatus.RUNNING,
            started_at=datetime.now(timezone.utc)
        )
        
        self.job_results[job.job_id] = job_result
        
        # Create and start job task
        task = asyncio.create_task(
            self._run_job_handler(job, job_result)
        )
        self.active_jobs[job.job_id] = task
        
        logger.info(f"Started job {job.name} ({job.job_id})")
    
    async def _run_job_handler(self, job: JobDefinition, result: JobResult):
        """Run job handler with timeout and error handling"""
        try:
            # Apply timeout if specified
            if job.timeout:
                task_result = await asyncio.wait_for(
                    job.handler(*job.args, **job.kwargs),
                    timeout=job.timeout
                )
            else:
                task_result = await job.handler(*job.args, **job.kwargs)
            
            # Job completed successfully
            result.status = JobStatus.COMPLETED
            result.result = task_result
            result.completed_at = datetime.now(timezone.utc)
            result.duration = (result.completed_at - result.started_at).total_seconds()
            
            logger.info(f"Completed job {job.name} ({job.job_id}) in {result.duration:.2f}s")
            
        except asyncio.TimeoutError:
            result.status = JobStatus.FAILED
            result.error = f"Job timed out after {job.timeout}s"
            result.completed_at = datetime.now(timezone.utc)
            logger.error(f"Job {job.name} ({job.job_id}) timed out")
            
        except Exception as e:
            result.status = JobStatus.FAILED
            result.error = str(e)
            result.completed_at = datetime.now(timezone.utc)
            logger.error(f"Job {job.name} ({job.job_id}) failed: {e}")
            
            # Retry if possible
            if hasattr(job, 'retry_count') and job.retry_count < job.max_retries:
                await self._retry_job(job)
    
    async def _retry_job(self, job: JobDefinition):
        """Retry a failed job"""
        if not hasattr(job, 'retry_count'):
            job.retry_count = 0
        
        job.retry_count += 1
        job.scheduled_at = datetime.now(timezone.utc) + timedelta(seconds=job.retry_delay)
        
        # Re-queue the job
        self.job_queues[job.priority].append(job)
        
        logger.info(f"Scheduled retry {job.retry_count}/{job.max_retries} for job {job.name}")
    
    async def _cleanup_completed_jobs(self):
        """Remove completed job tasks"""
        completed_jobs = []
        
        for job_id, task in self.active_jobs.items():
            if task.done():
                completed_jobs.append(job_id)
        
        for job_id in completed_jobs:
            del self.active_jobs[job_id]
    
    # Built-in job handlers
    
    async def _cleanup_expired_signals(self) -> Dict[str, int]:
        """Clean up expired signals"""
        try:
            async with get_db() as db:
                # Delete expired signals
                now = datetime.now(timezone.utc)
                result = await db.execute(
                    delete(Signal)
                    .where(Signal.expires_at < now)
                    .where(Signal.status != SignalStatus.EXECUTED)
                )
                
                await db.commit()
                deleted_count = result.rowcount
                
                # Clear cache for expired signals
                cache = get_signal_cache()
                if cache:
                    # This would need to iterate through cached signals
                    # For now, just invalidate some common patterns
                    await cache.cache.delete_pattern("signals", "*")
                
                logger.info(f"Cleaned up {deleted_count} expired signals")
                return {"deleted_signals": deleted_count}
                
        except Exception as e:
            logger.error(f"Error cleaning up expired signals: {e}")
            raise
    
    async def _update_signal_matches(self, limit: int = 100) -> Dict[str, int]:
        """Update signal matches for active signals"""
        try:
            matching_service = MatchingService()
            updated_count = 0
            
            async with get_db() as db:
                # Get recent active signals
                result = await db.execute(
                    select(Signal)
                    .where(Signal.status == SignalStatus.ACTIVE)
                    .where(Signal.expires_at > datetime.now(timezone.utc))
                    .limit(limit)
                )
                
                signals = result.scalars().all()
                
                for signal in signals:
                    # Re-run matching for this signal
                    matches = await matching_service.find_matches_for_signal(signal)
                    if matches:
                        updated_count += 1
                        
                        # Publish updated matches
                        producer = get_producer()
                        if producer:
                            await producer.send_message(
                                KafkaTopics.SIGNALS_MATCHED,
                                {
                                    "signal_id": signal.id,
                                    "matches": [match.model_dump() for match in matches],
                                    "updated_at": datetime.now(timezone.utc).isoformat()
                                }
                            )
            
            logger.info(f"Updated matches for {updated_count} signals")
            return {"updated_signals": updated_count}
            
        except Exception as e:
            logger.error(f"Error updating signal matches: {e}")
            raise
    
    async def _calculate_user_stats(self, days: int = 30) -> Dict[str, Any]:
        """Calculate user performance statistics"""
        try:
            stats = {"users_processed": 0, "total_signals": 0}
            
            async with get_db() as db:
                # This would calculate various user statistics
                # For now, just count active users
                result = await db.execute(select(User))
                users = result.scalars().all()
                stats["users_processed"] = len(users)
                
                # Count signals in date range
                cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
                result = await db.execute(
                    select(Signal).where(Signal.created_at >= cutoff_date)
                )
                signals = result.scalars().all()
                stats["total_signals"] = len(signals)
            
            logger.info(f"Calculated stats for {stats['users_processed']} users")
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating user stats: {e}")
            raise
    
    async def _process_market_close(self) -> Dict[str, Any]:
        """Process end-of-market tasks"""
        try:
            result = {"tasks_completed": 0}
            
            # Expire day-trading signals
            async with get_db() as db:
                await db.execute(
                    update(Signal)
                    .where(Signal.timeframe == "INTRADAY")
                    .where(Signal.status == SignalStatus.ACTIVE)
                    .values(status=SignalStatus.EXPIRED)
                )
                await db.commit()
                result["tasks_completed"] += 1
            
            # Generate end-of-day summary
            producer = get_producer()
            if producer:
                summary = {
                    "type": "market_close_summary",
                    "date": datetime.now(timezone.utc).date().isoformat(),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
                await producer.send_message(
                    KafkaTopics.SYSTEM_METRICS,
                    summary
                )
                result["tasks_completed"] += 1
            
            logger.info("Processed market close tasks")
            return result
            
        except Exception as e:
            logger.error(f"Error processing market close: {e}")
            raise
    
    async def _generate_daily_summary(self) -> Dict[str, Any]:
        """Generate daily performance summary"""
        try:
            # This would generate comprehensive daily summaries
            summary = {
                "date": datetime.now(timezone.utc).date().isoformat(),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "signals_generated": 0,
                "users_matched": 0
            }
            
            logger.info("Generated daily summary")
            return summary
            
        except Exception as e:
            logger.error(f"Error generating daily summary: {e}")
            raise
    
    async def _backup_data(self, backup_type: str = "incremental") -> Dict[str, Any]:
        """Backup critical data"""
        try:
            # Mock backup operation
            result = {
                "backup_type": backup_type,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "completed"
            }
            
            logger.info(f"Completed {backup_type} backup")
            return result
            
        except Exception as e:
            logger.error(f"Error during backup: {e}")
            raise
    
    async def _health_check(self) -> Dict[str, Any]:
        """System health check"""
        try:
            health_status = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "database": "healthy",
                "cache": "healthy",
                "kafka": "healthy"
            }
            
            # Test database connection
            try:
                async with get_db() as db:
                    await db.execute(select(1))
            except Exception:
                health_status["database"] = "unhealthy"
            
            # Test cache connection
            cache = get_cache()
            if not cache or not await cache.health_check():
                health_status["cache"] = "unhealthy"
            
            # Test Kafka producer
            producer = get_producer()
            if not producer:
                health_status["kafka"] = "unhealthy"
            
            logger.info("Completed health check")
            return health_status
            
        except Exception as e:
            logger.error(f"Error during health check: {e}")
            raise
    
    async def _cache_warmup(self) -> Dict[str, int]:
        """Warm up cache with frequently accessed data"""
        try:
            warmed_items = 0
            
            # This would preload frequently accessed data into cache
            # For now, just log the operation
            
            logger.info(f"Cache warmup completed - warmed {warmed_items} items")
            return {"warmed_items": warmed_items}
            
        except Exception as e:
            logger.error(f"Error during cache warmup: {e}")
            raise


# Global job processor instance
job_processor: Optional[BackgroundJobProcessor] = None


async def init_background_jobs() -> BackgroundJobProcessor:
    """Initialize background job processor"""
    global job_processor
    
    job_processor = BackgroundJobProcessor(
        max_concurrent_jobs=5,
        poll_interval=1.0
    )
    
    # Schedule recurring tasks
    _schedule_recurring_tasks(job_processor)
    
    logger.info("Background job processor initialized")
    return job_processor


def _schedule_recurring_tasks(processor: BackgroundJobProcessor):
    """Schedule recurring maintenance tasks"""
    now = datetime.now(timezone.utc)
    
    # Schedule cleanup every hour
    processor.schedule_job(
        name="Hourly Cleanup",
        handler_name="cleanup_expired_signals",
        run_at=now + timedelta(hours=1)
    )
    
    # Schedule match updates every 15 minutes
    processor.schedule_job(
        name="Update Matches",
        handler_name="update_signal_matches",
        run_at=now + timedelta(minutes=15)
    )
    
    # Schedule daily stats at 6 AM
    tomorrow_6am = (now + timedelta(days=1)).replace(hour=6, minute=0, second=0, microsecond=0)
    processor.schedule_job(
        name="Daily Stats",
        handler_name="calculate_user_stats",
        run_at=tomorrow_6am,
        priority=JobPriority.LOW
    )
    
    # Schedule market close processing at 4:30 PM EST (assuming UTC)
    market_close_time = now.replace(hour=21, minute=30, second=0, microsecond=0)
    if market_close_time <= now:
        market_close_time += timedelta(days=1)
    
    processor.schedule_job(
        name="Market Close Processing",
        handler_name="process_market_close",
        run_at=market_close_time,
        priority=JobPriority.HIGH
    )


def get_job_processor() -> Optional[BackgroundJobProcessor]:
    """Get global job processor instance"""
    return job_processor