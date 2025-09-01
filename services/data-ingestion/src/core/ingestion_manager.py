"""
Ingestion Manager - Core orchestrator for all data ingestion operations.
Handles multiple data sources with failover and load balancing capabilities.
"""

import asyncio
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from ..sources.market_data_sources import MarketDataSourceManager
from ..sources.news_sources import NewsSourceManager
from ..sources.economic_data_sources import EconomicDataSourceManager
from ..processors.data_processor import DataProcessor
from ..publishers.kafka_publisher import KafkaPublisher
from ..storage.influxdb_writer import InfluxDBWriter
from ..storage.redis_cache import RedisCache
from ..config.settings import Settings


logger = structlog.get_logger(__name__)


class IngestionManager:
    """
    Central manager for all data ingestion operations.
    Orchestrates data sources, processing, and publishing with fault tolerance.
    """
    
    def __init__(self, settings: Settings, metrics_collector):
        self.settings = settings
        self.metrics_collector = metrics_collector
        self.is_running = False
        self.start_time = None
        
        # Source managers
        self.market_data_manager = MarketDataSourceManager(settings, metrics_collector)
        self.news_manager = NewsSourceManager(settings, metrics_collector)
        self.economic_data_manager = EconomicDataSourceManager(settings, metrics_collector)
        
        # Processing and publishing
        self.data_processor = DataProcessor(settings, metrics_collector)
        self.kafka_publisher = KafkaPublisher(settings, metrics_collector)
        self.influxdb_writer = InfluxDBWriter(settings, metrics_collector)
        self.redis_cache = RedisCache(settings, metrics_collector)
        
        # Task management
        self.tasks: List[asyncio.Task] = []
        self.executor = ThreadPoolExecutor(max_workers=settings.ingestion_worker_threads)
        
        # Status tracking
        self.source_status: Dict[str, Dict[str, Any]] = {}
        self.processing_stats = {
            "messages_processed": 0,
            "messages_failed": 0,
            "bytes_processed": 0,
            "last_message_time": None,
            "processing_rate": 0.0
        }
    
    async def start(self):
        """Start the ingestion manager and all data sources."""
        logger.info("Starting ingestion manager")
        
        try:
            self.start_time = datetime.now(timezone.utc)
            
            # Initialize storage and messaging
            await self._initialize_dependencies()
            
            # Start source managers
            await self._start_source_managers()
            
            # Start processing tasks
            await self._start_processing_tasks()
            
            # Start monitoring tasks
            await self._start_monitoring_tasks()
            
            self.is_running = True
            logger.info("Ingestion manager started successfully")
            
        except Exception as e:
            logger.error("Failed to start ingestion manager", error=str(e))
            await self.stop()
            raise
    
    async def stop(self):
        """Stop the ingestion manager and all data sources."""
        logger.info("Stopping ingestion manager")
        
        self.is_running = False
        
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
            self.tasks.clear()
        
        # Stop source managers
        await self._stop_source_managers()
        
        # Close connections
        await self._cleanup_dependencies()
        
        # Shutdown thread pool
        self.executor.shutdown(wait=True)
        
        logger.info("Ingestion manager stopped")
    
    def is_ready(self) -> bool:
        """Check if the ingestion manager is ready."""
        return (
            self.is_running and
            self.kafka_publisher.is_connected() and
            self.influxdb_writer.is_connected() and
            self.redis_cache.is_connected()
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status and statistics."""
        uptime = None
        if self.start_time:
            uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        
        return {
            "is_running": self.is_running,
            "is_ready": self.is_ready(),
            "uptime_seconds": uptime,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sources": self.source_status,
            "processing_stats": self.processing_stats,
            "dependencies": {
                "kafka_connected": self.kafka_publisher.is_connected(),
                "influxdb_connected": self.influxdb_writer.is_connected(),
                "redis_connected": self.redis_cache.is_connected()
            }
        }
    
    def get_sources_info(self) -> Dict[str, Any]:
        """Get information about all data sources."""
        return {
            "market_data": self.market_data_manager.get_sources_info(),
            "news": self.news_manager.get_sources_info(),
            "economic_data": self.economic_data_manager.get_sources_info()
        }
    
    async def restart_source(self, source_name: str):
        """Restart a specific data source."""
        logger.info("Restarting data source", source=source_name)
        
        # Try each manager
        managers = [
            self.market_data_manager,
            self.news_manager,
            self.economic_data_manager
        ]
        
        for manager in managers:
            try:
                await manager.restart_source(source_name)
                logger.info("Source restarted successfully", source=source_name)
                return
            except ValueError:
                continue  # Source not found in this manager
        
        raise ValueError(f"Source not found: {source_name}")
    
    async def pause_source(self, source_name: str):
        """Pause a specific data source."""
        logger.info("Pausing data source", source=source_name)
        
        managers = [
            self.market_data_manager,
            self.news_manager,
            self.economic_data_manager
        ]
        
        for manager in managers:
            try:
                await manager.pause_source(source_name)
                logger.info("Source paused successfully", source=source_name)
                return
            except ValueError:
                continue
        
        raise ValueError(f"Source not found: {source_name}")
    
    async def resume_source(self, source_name: str):
        """Resume a paused data source."""
        logger.info("Resuming data source", source=source_name)
        
        managers = [
            self.market_data_manager,
            self.news_manager,
            self.economic_data_manager
        ]
        
        for manager in managers:
            try:
                await manager.resume_source(source_name)
                logger.info("Source resumed successfully", source=source_name)
                return
            except ValueError:
                continue
        
        raise ValueError(f"Source not found: {source_name}")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _initialize_dependencies(self):
        """Initialize external dependencies with retry logic."""
        logger.info("Initializing dependencies")
        
        # Initialize in order of dependency
        await self.redis_cache.connect()
        await self.influxdb_writer.connect()
        await self.kafka_publisher.connect()
        await self.data_processor.initialize()
        
        logger.info("Dependencies initialized successfully")
    
    async def _start_source_managers(self):
        """Start all source managers."""
        logger.info("Starting source managers")
        
        # Start managers concurrently
        await asyncio.gather(
            self.market_data_manager.start(),
            self.news_manager.start(),
            self.economic_data_manager.start()
        )
        
        logger.info("Source managers started")
    
    async def _stop_source_managers(self):
        """Stop all source managers."""
        logger.info("Stopping source managers")
        
        await asyncio.gather(
            self.market_data_manager.stop(),
            self.news_manager.stop(),
            self.economic_data_manager.stop(),
            return_exceptions=True
        )
        
        logger.info("Source managers stopped")
    
    async def _start_processing_tasks(self):
        """Start data processing tasks."""
        logger.info("Starting processing tasks")
        
        # Start data processing pipeline
        self.tasks.append(
            asyncio.create_task(self._data_processing_loop())
        )
        
        # Start batch processing for time-series data
        self.tasks.append(
            asyncio.create_task(self._batch_processing_loop())
        )
        
        logger.info("Processing tasks started")
    
    async def _start_monitoring_tasks(self):
        """Start monitoring and health check tasks."""
        logger.info("Starting monitoring tasks")
        
        # Status update task
        self.tasks.append(
            asyncio.create_task(self._status_update_loop())
        )
        
        # Metrics collection task
        self.tasks.append(
            asyncio.create_task(self._metrics_collection_loop())
        )
        
        # Health check task
        self.tasks.append(
            asyncio.create_task(self._health_check_loop())
        )
        
        logger.info("Monitoring tasks started")
    
    async def _cleanup_dependencies(self):
        """Cleanup external dependencies."""
        logger.info("Cleaning up dependencies")
        
        await asyncio.gather(
            self.data_processor.cleanup(),
            self.kafka_publisher.disconnect(),
            self.influxdb_writer.disconnect(),
            self.redis_cache.disconnect(),
            return_exceptions=True
        )
        
        logger.info("Dependencies cleaned up")
    
    async def _data_processing_loop(self):
        """Main data processing loop."""
        logger.info("Starting data processing loop")
        
        try:
            while self.is_running:
                # Get data from all source managers
                data_batches = await asyncio.gather(
                    self.market_data_manager.get_processed_data(),
                    self.news_manager.get_processed_data(),
                    self.economic_data_manager.get_processed_data(),
                    return_exceptions=True
                )
                
                # Process each batch
                for batch in data_batches:
                    if isinstance(batch, Exception):
                        logger.warning("Failed to get data batch", error=str(batch))
                        continue
                    
                    if batch:
                        await self._process_data_batch(batch)
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.01)  # 10ms
                
        except asyncio.CancelledError:
            logger.info("Data processing loop cancelled")
        except Exception as e:
            logger.error("Error in data processing loop", error=str(e))
            # Continue processing on error
    
    async def _process_data_batch(self, batch: Dict[str, Any]):
        """Process a batch of data."""
        try:
            # Validate and enrich data
            processed_batch = await self.data_processor.process_batch(batch)
            
            # Publish to Kafka
            await self.kafka_publisher.publish_batch(processed_batch)
            
            # Store in InfluxDB for time-series data
            if processed_batch.get("time_series_data"):
                await self.influxdb_writer.write_batch(processed_batch["time_series_data"])
            
            # Cache frequently accessed data
            if processed_batch.get("cache_data"):
                await self.redis_cache.cache_batch(processed_batch["cache_data"])
            
            # Update processing stats
            self._update_processing_stats(processed_batch)
            
        except Exception as e:
            logger.error("Failed to process data batch", error=str(e))
            self.processing_stats["messages_failed"] += len(batch.get("messages", []))
            self.metrics_collector.increment_counter("processing_errors")
    
    async def _batch_processing_loop(self):
        """Batch processing loop for time-series aggregation."""
        logger.info("Starting batch processing loop")
        
        try:
            while self.is_running:
                # Process batched data every 5 seconds
                await asyncio.sleep(5)
                
                # Get accumulated data for batch processing
                batch_data = await self.data_processor.get_batch_data()
                
                if batch_data:
                    # Process aggregations and analytics
                    await self._process_analytics_batch(batch_data)
                
        except asyncio.CancelledError:
            logger.info("Batch processing loop cancelled")
        except Exception as e:
            logger.error("Error in batch processing loop", error=str(e))
    
    async def _process_analytics_batch(self, batch_data: Dict[str, Any]):
        """Process analytics and aggregations."""
        try:
            # Calculate moving averages, volume metrics, etc.
            analytics = await self.data_processor.calculate_analytics(batch_data)
            
            # Store analytics in InfluxDB
            if analytics:
                await self.influxdb_writer.write_analytics(analytics)
            
        except Exception as e:
            logger.error("Failed to process analytics batch", error=str(e))
    
    async def _status_update_loop(self):
        """Update source status periodically."""
        try:
            while self.is_running:
                # Update source status
                self.source_status = {
                    "market_data": self.market_data_manager.get_status(),
                    "news": self.news_manager.get_status(),
                    "economic_data": self.economic_data_manager.get_status()
                }
                
                await asyncio.sleep(10)  # Update every 10 seconds
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("Error in status update loop", error=str(e))
    
    async def _metrics_collection_loop(self):
        """Collect and update metrics."""
        try:
            while self.is_running:
                # Collect metrics from all components
                metrics = {
                    "processing_rate": self.processing_stats["processing_rate"],
                    "messages_processed": self.processing_stats["messages_processed"],
                    "messages_failed": self.processing_stats["messages_failed"],
                    "kafka_lag": await self.kafka_publisher.get_lag_metrics(),
                    "influxdb_write_rate": self.influxdb_writer.get_write_rate(),
                    "redis_cache_hit_rate": self.redis_cache.get_hit_rate()
                }
                
                self.metrics_collector.update_metrics(metrics)
                
                await asyncio.sleep(5)  # Update every 5 seconds
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("Error in metrics collection loop", error=str(e))
    
    async def _health_check_loop(self):
        """Perform health checks and auto-recovery."""
        try:
            while self.is_running:
                # Check health of all components
                if not await self._perform_health_checks():
                    logger.warning("Health check failed, attempting recovery")
                    await self._attempt_recovery()
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("Error in health check loop", error=str(e))
    
    async def _perform_health_checks(self) -> bool:
        """Perform comprehensive health checks."""
        try:
            checks = await asyncio.gather(
                self.kafka_publisher.health_check(),
                self.influxdb_writer.health_check(),
                self.redis_cache.health_check(),
                return_exceptions=True
            )
            
            return all(
                check is True or (not isinstance(check, Exception))
                for check in checks
            )
        except Exception:
            return False
    
    async def _attempt_recovery(self):
        """Attempt to recover from failures."""
        logger.info("Attempting system recovery")
        
        try:
            # Try to reconnect failed components
            if not self.kafka_publisher.is_connected():
                await self.kafka_publisher.reconnect()
            
            if not self.influxdb_writer.is_connected():
                await self.influxdb_writer.reconnect()
            
            if not self.redis_cache.is_connected():
                await self.redis_cache.reconnect()
            
            logger.info("Recovery attempt completed")
            
        except Exception as e:
            logger.error("Recovery attempt failed", error=str(e))
    
    def _update_processing_stats(self, batch: Dict[str, Any]):
        """Update processing statistics."""
        message_count = len(batch.get("messages", []))
        bytes_count = batch.get("total_bytes", 0)
        
        self.processing_stats["messages_processed"] += message_count
        self.processing_stats["bytes_processed"] += bytes_count
        self.processing_stats["last_message_time"] = time.time()
        
        # Calculate processing rate (messages per second)
        if hasattr(self, '_last_rate_calculation'):
            time_diff = time.time() - self._last_rate_calculation
            if time_diff > 0:
                msg_diff = message_count
                self.processing_stats["processing_rate"] = msg_diff / time_diff
        
        self._last_rate_calculation = time.time()