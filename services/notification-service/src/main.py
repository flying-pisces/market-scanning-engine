#!/usr/bin/env python3
"""
Notification Service for Market Scanning Engine
Multi-channel notification delivery with high throughput and reliability.
"""

import asyncio
import logging
import signal
from contextlib import asynccontextmanager
from typing import Dict, Any

import structlog
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from .core.notification_manager import NotificationManager
from .core.health_checker import HealthChecker
from .core.metrics_collector import MetricsCollector
from .api.routes import notification_routes, webhook_routes, admin_routes
from .config.settings import Settings
from .utils.logging_config import setup_logging


# Global instances
notification_manager: NotificationManager = None
health_checker: HealthChecker = None
metrics_collector: MetricsCollector = None
settings = Settings()

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global notification_manager, health_checker, metrics_collector
    
    # Startup
    logger.info("Starting Notification Service", version="1.0.0")
    
    try:
        # Initialize components
        metrics_collector = MetricsCollector()
        health_checker = HealthChecker(settings)
        notification_manager = NotificationManager(settings, metrics_collector)
        
        # Start background tasks
        await notification_manager.start()
        await health_checker.start()
        
        logger.info("Notification Service started successfully")
        
        yield
        
    except Exception as e:
        logger.error("Failed to start Notification Service", error=str(e))
        raise
    finally:
        # Shutdown
        logger.info("Shutting down Notification Service")
        
        if notification_manager:
            await notification_manager.stop()
        if health_checker:
            await health_checker.stop()
        
        logger.info("Notification Service stopped")


# FastAPI application
app = FastAPI(
    title="Market Scanning Notification Service",
    description="High-performance multi-channel notification delivery service",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(notification_routes, prefix="/api/v1", tags=["notifications"])
app.include_router(webhook_routes, prefix="/webhooks", tags=["webhooks"])
app.include_router(admin_routes, prefix="/admin", tags=["admin"])


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if not health_checker:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    health_status = await health_checker.get_health_status()
    
    if health_status["status"] != "healthy":
        raise HTTPException(
            status_code=503,
            detail=health_status
        )
    
    return health_status


@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint."""
    if not notification_manager or not notification_manager.is_ready():
        raise HTTPException(status_code=503, detail="Service not ready")
    
    return {
        "status": "ready", 
        "timestamp": notification_manager.get_status()["timestamp"],
        "channels_available": await notification_manager.get_available_channels()
    }


@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint."""
    if not metrics_collector:
        raise HTTPException(status_code=503, detail="Metrics not available")
    
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/status")
async def get_status():
    """Get service status and statistics."""
    if not notification_manager:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    status = notification_manager.get_status()
    health_status = await health_checker.get_health_status()
    
    return {
        "service": "notification-service",
        "status": status,
        "health": health_status,
        "metrics": metrics_collector.get_summary() if metrics_collector else {},
        "channels": await notification_manager.get_channel_status()
    }


@app.post("/api/v1/notifications")
async def send_notification(
    notification_request: Dict[str, Any],
    background_tasks: BackgroundTasks
):
    """Send a notification through specified channels."""
    if not notification_manager:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        # Validate request
        if not notification_request.get("message"):
            raise HTTPException(status_code=400, detail="Message is required")
        
        if not notification_request.get("channels"):
            raise HTTPException(status_code=400, detail="At least one channel is required")
        
        # Queue notification for processing
        notification_id = await notification_manager.queue_notification(notification_request)
        
        return {
            "notification_id": notification_id,
            "status": "queued",
            "message": "Notification queued for delivery"
        }
        
    except Exception as e:
        logger.error("Failed to queue notification", error=str(e), request=notification_request)
        raise HTTPException(status_code=500, detail=f"Failed to queue notification: {e}")


@app.post("/api/v1/notifications/bulk")
async def send_bulk_notifications(
    bulk_request: Dict[str, Any],
    background_tasks: BackgroundTasks
):
    """Send multiple notifications in bulk."""
    if not notification_manager:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        notifications = bulk_request.get("notifications", [])
        if not notifications:
            raise HTTPException(status_code=400, detail="No notifications provided")
        
        if len(notifications) > settings.max_bulk_size:
            raise HTTPException(
                status_code=400, 
                detail=f"Bulk size exceeds limit of {settings.max_bulk_size}"
            )
        
        # Queue all notifications
        notification_ids = await notification_manager.queue_bulk_notifications(notifications)
        
        return {
            "notification_ids": notification_ids,
            "status": "queued",
            "count": len(notification_ids),
            "message": f"{len(notification_ids)} notifications queued for delivery"
        }
        
    except Exception as e:
        logger.error("Failed to queue bulk notifications", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to queue bulk notifications: {e}")


@app.get("/api/v1/notifications/{notification_id}/status")
async def get_notification_status(notification_id: str):
    """Get the status of a specific notification."""
    if not notification_manager:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        status = await notification_manager.get_notification_status(notification_id)
        
        if not status:
            raise HTTPException(status_code=404, detail="Notification not found")
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get notification status", 
                    notification_id=notification_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get notification status: {e}")


@app.post("/api/v1/notifications/{notification_id}/retry")
async def retry_notification(notification_id: str, background_tasks: BackgroundTasks):
    """Retry a failed notification."""
    if not notification_manager:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        success = await notification_manager.retry_notification(notification_id)
        
        if not success:
            raise HTTPException(
                status_code=404, 
                detail="Notification not found or cannot be retried"
            )
        
        return {
            "notification_id": notification_id,
            "status": "retry_queued",
            "message": "Notification queued for retry"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to retry notification", 
                    notification_id=notification_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to retry notification: {e}")


@app.get("/api/v1/channels")
async def get_channels():
    """Get information about available notification channels."""
    if not notification_manager:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        channels = await notification_manager.get_channel_info()
        return {"channels": channels}
        
    except Exception as e:
        logger.error("Failed to get channel information", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get channel information: {e}")


@app.post("/api/v1/channels/{channel_name}/test")
async def test_channel(channel_name: str, test_request: Dict[str, Any]):
    """Test a notification channel."""
    if not notification_manager:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        result = await notification_manager.test_channel(channel_name, test_request)
        
        return {
            "channel": channel_name,
            "test_result": result,
            "status": "success" if result.get("success", False) else "failed"
        }
        
    except Exception as e:
        logger.error("Failed to test channel", channel=channel_name, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to test channel: {e}")


@app.get("/api/v1/stats")
async def get_delivery_stats():
    """Get notification delivery statistics."""
    if not notification_manager:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        stats = await notification_manager.get_delivery_stats()
        return {"statistics": stats}
        
    except Exception as e:
        logger.error("Failed to get delivery statistics", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get delivery statistics: {e}")


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info("Received shutdown signal", signal=signum)
    asyncio.create_task(shutdown_handler())


async def shutdown_handler():
    """Graceful shutdown handler."""
    logger.info("Initiating graceful shutdown")
    # The lifespan context manager will handle cleanup


def main():
    """Main entry point."""
    # Setup logging
    setup_logging(settings.log_level)
    
    # Setup signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Run the application
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        log_config=None,  # Use our custom logging
        access_log=False,  # Disable uvicorn access logs
        workers=1,  # Single worker for async application
        loop="asyncio"
    )


if __name__ == "__main__":
    main()