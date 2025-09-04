"""
Performance monitoring and metrics collection system
Tracks system performance, user metrics, and business KPIs
"""

import asyncio
import logging
import time
import psutil
import json
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from enum import Enum

from app.core.kafka_client import get_producer, KafkaTopics
from app.core.cache import get_cache
from app.database import get_db
from app.models.signal import Signal, SignalStatus
from app.models.user import User
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics collected"""
    COUNTER = "counter"      # Monotonically increasing values
    GAUGE = "gauge"         # Point-in-time values
    HISTOGRAM = "histogram"  # Distribution of values
    TIMER = "timer"         # Duration measurements


@dataclass
class MetricPoint:
    """Individual metric data point"""
    name: str
    metric_type: MetricType
    value: float
    timestamp: datetime
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


@dataclass
class SystemStats:
    """System performance statistics"""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    disk_free_gb: float
    active_connections: int
    uptime_seconds: float
    timestamp: datetime


@dataclass
class BusinessMetrics:
    """Business performance metrics"""
    total_users: int
    active_users_24h: int
    signals_generated_24h: int
    signals_matched_24h: int
    avg_user_risk_score: float
    top_asset_classes: List[str]
    market_data_points_24h: int
    system_alerts: int
    timestamp: datetime


class MetricsCollector:
    """Centralized metrics collection system"""
    
    def __init__(self, collection_interval: float = 60.0):
        self.collection_interval = collection_interval
        self.is_running = False
        
        # In-memory metric storage
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.timers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # System stats history
        self.system_stats_history: deque = deque(maxlen=1440)  # 24 hours at 1-minute intervals
        
        # Alert thresholds
        self.alert_thresholds = {
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "disk_usage_percent": 90.0,
            "response_time_p95": 2.0,  # 2 seconds
            "error_rate": 5.0,  # 5%
            "queue_depth": 1000
        }
        
        # Alert state tracking
        self.active_alerts: Dict[str, datetime] = {}
        self.alert_cooldown = timedelta(minutes=5)
    
    async def start(self):
        """Start metrics collection"""
        if self.is_running:
            return
        
        self.is_running = True
        logger.info("Metrics collector started")
        
        # Start collection tasks
        tasks = [
            self._collect_system_metrics(),
            self._collect_business_metrics(),
            self._process_alerts(),
            self._publish_metrics()
        ]
        
        await asyncio.gather(*tasks)
    
    async def stop(self):
        """Stop metrics collection"""
        self.is_running = False
        logger.info("Metrics collector stopped")
    
    # Core metric recording methods
    
    def increment_counter(self, name: str, value: float = 1.0, tags: Dict[str, str] = None):
        """Increment a counter metric"""
        key = self._make_metric_key(name, tags)
        self.counters[key] += value
    
    def set_gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """Set a gauge metric value"""
        key = self._make_metric_key(name, tags)
        self.gauges[key] = value
    
    def record_histogram(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a value in a histogram"""
        key = self._make_metric_key(name, tags)
        self.histograms[key].append(value)
    
    def start_timer(self, name: str, tags: Dict[str, str] = None) -> 'TimerContext':
        """Start a timer (returns context manager)"""
        return TimerContext(self, name, tags)
    
    def record_timer(self, name: str, duration: float, tags: Dict[str, str] = None):
        """Record a timer value"""
        key = self._make_metric_key(name, tags)
        self.timers[key].append(duration)
    
    def _make_metric_key(self, name: str, tags: Dict[str, str] = None) -> str:
        """Create a unique key for metric with tags"""
        if not tags:
            return name
        
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}|{tag_str}"
    
    # System metrics collection
    
    async def _collect_system_metrics(self):
        """Collect system performance metrics"""
        while self.is_running:
            try:
                stats = await self._get_system_stats()
                self.system_stats_history.append(stats)
                
                # Record as individual metrics
                self.set_gauge("system.cpu_percent", stats.cpu_percent)
                self.set_gauge("system.memory_percent", stats.memory_percent)
                self.set_gauge("system.disk_usage_percent", stats.disk_usage_percent)
                self.set_gauge("system.active_connections", stats.active_connections)
                self.set_gauge("system.uptime_seconds", stats.uptime_seconds)
                
                # Check for alerts
                await self._check_system_alerts(stats)
                
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def _get_system_stats(self) -> SystemStats:
        """Get current system statistics"""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_used_mb = (memory.total - memory.available) / 1024 / 1024
        memory_available_mb = memory.available / 1024 / 1024
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_usage_percent = (disk.used / disk.total) * 100
        disk_free_gb = disk.free / 1024 / 1024 / 1024
        
        # Network connections (approximate active connections)
        connections = len(psutil.net_connections())
        
        # System uptime
        boot_time = psutil.boot_time()
        uptime_seconds = time.time() - boot_time
        
        return SystemStats(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory_used_mb,
            memory_available_mb=memory_available_mb,
            disk_usage_percent=disk_usage_percent,
            disk_free_gb=disk_free_gb,
            active_connections=connections,
            uptime_seconds=uptime_seconds,
            timestamp=datetime.now(timezone.utc)
        )
    
    # Business metrics collection
    
    async def _collect_business_metrics(self):
        """Collect business and application metrics"""
        while self.is_running:
            try:
                metrics = await self._get_business_metrics()
                
                # Record business metrics
                self.set_gauge("business.total_users", metrics.total_users)
                self.set_gauge("business.active_users_24h", metrics.active_users_24h)
                self.set_gauge("business.signals_generated_24h", metrics.signals_generated_24h)
                self.set_gauge("business.signals_matched_24h", metrics.signals_matched_24h)
                self.set_gauge("business.avg_user_risk_score", metrics.avg_user_risk_score)
                
                await asyncio.sleep(self.collection_interval * 5)  # Every 5 minutes
                
            except Exception as e:
                logger.error(f"Error collecting business metrics: {e}")
                await asyncio.sleep(self.collection_interval * 5)
    
    async def _get_business_metrics(self) -> BusinessMetrics:
        """Get current business metrics"""
        try:
            async with get_db() as db:
                # Total users
                result = await db.execute(select(func.count(User.id)))
                total_users = result.scalar() or 0
                
                # Active users (last 24h)
                yesterday = datetime.now(timezone.utc) - timedelta(days=1)
                result = await db.execute(
                    select(func.count(User.id.distinct()))
                    .where(User.last_login >= yesterday)
                )
                active_users_24h = result.scalar() or 0
                
                # Signals generated (last 24h)
                result = await db.execute(
                    select(func.count(Signal.id))
                    .where(Signal.created_at >= yesterday)
                )
                signals_generated_24h = result.scalar() or 0
                
                # Average user risk score
                result = await db.execute(
                    select(func.avg(User.risk_tolerance))
                )
                avg_risk_score = float(result.scalar() or 50.0)
                
                # Top asset classes (mock for now)
                top_asset_classes = ["STOCKS", "ETFS", "BONDS"]
                
                return BusinessMetrics(
                    total_users=total_users,
                    active_users_24h=active_users_24h,
                    signals_generated_24h=signals_generated_24h,
                    signals_matched_24h=0,  # Would need to track this
                    avg_user_risk_score=avg_risk_score,
                    top_asset_classes=top_asset_classes,
                    market_data_points_24h=0,  # Would need to track this
                    system_alerts=len(self.active_alerts),
                    timestamp=datetime.now(timezone.utc)
                )
                
        except Exception as e:
            logger.error(f"Error getting business metrics: {e}")
            return BusinessMetrics(
                total_users=0,
                active_users_24h=0,
                signals_generated_24h=0,
                signals_matched_24h=0,
                avg_user_risk_score=50.0,
                top_asset_classes=[],
                market_data_points_24h=0,
                system_alerts=len(self.active_alerts),
                timestamp=datetime.now(timezone.utc)
            )
    
    # Alert processing
    
    async def _check_system_alerts(self, stats: SystemStats):
        """Check system stats against alert thresholds"""
        alerts_to_fire = []
        
        # CPU alert
        if stats.cpu_percent > self.alert_thresholds["cpu_percent"]:
            alerts_to_fire.append({
                "type": "HIGH_CPU",
                "message": f"CPU usage at {stats.cpu_percent:.1f}%",
                "severity": "warning" if stats.cpu_percent < 90 else "critical",
                "value": stats.cpu_percent,
                "threshold": self.alert_thresholds["cpu_percent"]
            })
        
        # Memory alert
        if stats.memory_percent > self.alert_thresholds["memory_percent"]:
            alerts_to_fire.append({
                "type": "HIGH_MEMORY",
                "message": f"Memory usage at {stats.memory_percent:.1f}%",
                "severity": "warning" if stats.memory_percent < 95 else "critical",
                "value": stats.memory_percent,
                "threshold": self.alert_thresholds["memory_percent"]
            })
        
        # Disk alert
        if stats.disk_usage_percent > self.alert_thresholds["disk_usage_percent"]:
            alerts_to_fire.append({
                "type": "HIGH_DISK_USAGE",
                "message": f"Disk usage at {stats.disk_usage_percent:.1f}%",
                "severity": "critical",
                "value": stats.disk_usage_percent,
                "threshold": self.alert_thresholds["disk_usage_percent"]
            })
        
        # Fire alerts
        for alert in alerts_to_fire:
            await self._fire_alert(alert)
    
    async def _fire_alert(self, alert: Dict[str, Any]):
        """Fire system alert"""
        alert_key = alert["type"]
        now = datetime.now(timezone.utc)
        
        # Check cooldown period
        if alert_key in self.active_alerts:
            last_alert = self.active_alerts[alert_key]
            if now - last_alert < self.alert_cooldown:
                return  # Still in cooldown
        
        # Record alert
        self.active_alerts[alert_key] = now
        self.increment_counter("alerts.fired", tags={"type": alert["type"]})
        
        # Log alert
        logger.warning(f"ALERT: {alert['message']}")
        
        # Publish alert
        producer = get_producer()
        if producer:
            alert_message = {
                "type": "system_alert",
                "alert_type": alert["type"],
                "message": alert["message"],
                "severity": alert["severity"],
                "value": alert["value"],
                "threshold": alert["threshold"],
                "timestamp": now.isoformat(),
                "hostname": psutil.uname().node
            }
            
            await producer.send_message(
                KafkaTopics.SYSTEM_METRICS,
                alert_message,
                key="alert"
            )
    
    async def _process_alerts(self):
        """Process and clean up alerts"""
        while self.is_running:
            try:
                now = datetime.now(timezone.utc)
                
                # Clear resolved alerts (older than cooldown period)
                resolved_alerts = []
                for alert_key, alert_time in self.active_alerts.items():
                    if now - alert_time > self.alert_cooldown * 3:  # 3x cooldown for resolution
                        resolved_alerts.append(alert_key)
                
                for alert_key in resolved_alerts:
                    del self.active_alerts[alert_key]
                    logger.info(f"Alert resolved: {alert_key}")
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error processing alerts: {e}")
                await asyncio.sleep(60)
    
    # Metric publishing
    
    async def _publish_metrics(self):
        """Publish metrics to external systems"""
        while self.is_running:
            try:
                # Publish to Kafka
                await self._publish_to_kafka()
                
                # Cache key metrics
                await self._cache_metrics()
                
                await asyncio.sleep(self.collection_interval * 2)  # Every 2 minutes
                
            except Exception as e:
                logger.error(f"Error publishing metrics: {e}")
                await asyncio.sleep(self.collection_interval * 2)
    
    async def _publish_to_kafka(self):
        """Publish metrics to Kafka"""
        producer = get_producer()
        if not producer:
            return
        
        now = datetime.now(timezone.utc)
        
        # Prepare metrics payload
        metrics_payload = {
            "timestamp": now.isoformat(),
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "system_stats": asdict(self.system_stats_history[-1]) if self.system_stats_history else None,
            "active_alerts": len(self.active_alerts)
        }
        
        # Add histogram summaries
        histogram_summaries = {}
        for name, values in self.histograms.items():
            if values:
                histogram_summaries[name] = {
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values),
                    "p50": self._percentile(list(values), 50),
                    "p95": self._percentile(list(values), 95),
                    "p99": self._percentile(list(values), 99)
                }
        
        metrics_payload["histograms"] = histogram_summaries
        
        # Add timer summaries
        timer_summaries = {}
        for name, values in self.timers.items():
            if values:
                timer_summaries[name] = {
                    "count": len(values),
                    "min_ms": min(values) * 1000,
                    "max_ms": max(values) * 1000,
                    "avg_ms": (sum(values) / len(values)) * 1000,
                    "p50_ms": self._percentile(list(values), 50) * 1000,
                    "p95_ms": self._percentile(list(values), 95) * 1000,
                    "p99_ms": self._percentile(list(values), 99) * 1000
                }
        
        metrics_payload["timers"] = timer_summaries
        
        # Publish
        await producer.send_message(
            KafkaTopics.SYSTEM_METRICS,
            metrics_payload,
            key="metrics"
        )
    
    async def _cache_metrics(self):
        """Cache key metrics for API access"""
        cache = get_cache()
        if not cache:
            return
        
        # Cache system health summary
        health_summary = {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cpu_percent": self.gauges.get("system.cpu_percent", 0),
            "memory_percent": self.gauges.get("system.memory_percent", 0),
            "active_alerts": len(self.active_alerts)
        }
        
        # Determine overall health
        if (health_summary["cpu_percent"] > 80 or 
            health_summary["memory_percent"] > 85 or 
            len(self.active_alerts) > 0):
            health_summary["status"] = "degraded" if len(self.active_alerts) < 3 else "unhealthy"
        
        await cache.set("system", "health", health_summary, ttl=120)
        
        # Cache business metrics summary
        business_summary = {
            "total_users": self.gauges.get("business.total_users", 0),
            "active_users_24h": self.gauges.get("business.active_users_24h", 0),
            "signals_generated_24h": self.gauges.get("business.signals_generated_24h", 0),
            "avg_user_risk_score": self.gauges.get("business.avg_user_risk_score", 50)
        }
        
        await cache.set("business", "summary", business_summary, ttl=300)
    
    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile of values"""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int(len(sorted_values) * (percentile / 100))
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    # Public API methods
    
    def get_metric_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics"""
        return {
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "histogram_count": {name: len(values) for name, values in self.histograms.items()},
            "timer_count": {name: len(values) for name, values in self.timers.items()},
            "active_alerts": len(self.active_alerts),
            "collection_interval": self.collection_interval,
            "uptime": time.time() - (self.system_stats_history[0].timestamp.timestamp() 
                                   if self.system_stats_history else time.time())
        }
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get current system health status"""
        if not self.system_stats_history:
            return {"status": "unknown", "message": "No data available"}
        
        latest_stats = self.system_stats_history[-1]
        
        health_status = "healthy"
        issues = []
        
        if latest_stats.cpu_percent > 80:
            health_status = "degraded"
            issues.append(f"High CPU usage: {latest_stats.cpu_percent:.1f}%")
        
        if latest_stats.memory_percent > 85:
            health_status = "degraded"
            issues.append(f"High memory usage: {latest_stats.memory_percent:.1f}%")
        
        if latest_stats.disk_usage_percent > 90:
            health_status = "critical"
            issues.append(f"High disk usage: {latest_stats.disk_usage_percent:.1f}%")
        
        if self.active_alerts:
            health_status = "degraded" if health_status == "healthy" else health_status
            issues.extend([f"Active alert: {alert}" for alert in self.active_alerts.keys()])
        
        return {
            "status": health_status,
            "timestamp": latest_stats.timestamp.isoformat(),
            "issues": issues,
            "stats": asdict(latest_stats)
        }


class TimerContext:
    """Context manager for timing operations"""
    
    def __init__(self, collector: MetricsCollector, name: str, tags: Dict[str, str] = None):
        self.collector = collector
        self.name = name
        self.tags = tags
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.collector.record_timer(self.name, duration, self.tags)


# Global metrics collector instance
metrics_collector: Optional[MetricsCollector] = None


async def init_monitoring(collection_interval: float = 60.0) -> MetricsCollector:
    """Initialize monitoring system"""
    global metrics_collector
    
    metrics_collector = MetricsCollector(collection_interval)
    logger.info("Monitoring system initialized")
    return metrics_collector


def get_metrics_collector() -> Optional[MetricsCollector]:
    """Get global metrics collector instance"""
    return metrics_collector


# Convenience functions for common metrics

def increment(name: str, value: float = 1.0, tags: Dict[str, str] = None):
    """Increment a counter metric"""
    if metrics_collector:
        metrics_collector.increment_counter(name, value, tags)


def gauge(name: str, value: float, tags: Dict[str, str] = None):
    """Set a gauge metric"""
    if metrics_collector:
        metrics_collector.set_gauge(name, value, tags)


def histogram(name: str, value: float, tags: Dict[str, str] = None):
    """Record a histogram value"""
    if metrics_collector:
        metrics_collector.record_histogram(name, value, tags)


def timer(name: str, tags: Dict[str, str] = None) -> TimerContext:
    """Start a timer (returns context manager)"""
    if metrics_collector:
        return metrics_collector.start_timer(name, tags)
    else:
        # Return a no-op context manager
        class NoOpTimer:
            def __enter__(self): return self
            def __exit__(self, *args): pass
        return NoOpTimer()