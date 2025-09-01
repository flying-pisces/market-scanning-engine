#!/usr/bin/env python3
"""
Data Ingestion Service for Market Scanning Engine
High-throughput, fault-tolerant data ingestion with real-time processing capabilities.
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

from .core.ingestion_manager import IngestionManager
from .core.health_checker import HealthChecker
from .core.metrics_collector import MetricsCollector
from .config.settings import Settings
from .utils.logging_config import setup_logging


# Global instances
ingestion_manager: IngestionManager = None
health_checker: HealthChecker = None
metrics_collector: MetricsCollector = None
settings = Settings()

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global ingestion_manager, health_checker, metrics_collector
    
    # Startup
    logger.info("Starting Data Ingestion Service", version="1.0.0")
    
    try:
        # Initialize components
        metrics_collector = MetricsCollector()
        health_checker = HealthChecker(settings)
        ingestion_manager = IngestionManager(settings, metrics_collector)
        
        # Start background tasks
        await ingestion_manager.start()
        await health_checker.start()
        
        logger.info("Data Ingestion Service started successfully")
        
        yield
        
    except Exception as e:
        logger.error("Failed to start Data Ingestion Service", error=str(e))
        raise
    finally:
        # Shutdown
        logger.info("Shutting down Data Ingestion Service")
        
        if ingestion_manager:
            await ingestion_manager.stop()
        if health_checker:
            await health_checker.stop()
        
        logger.info("Data Ingestion Service stopped")


# FastAPI application
app = FastAPI(
    title="Market Data Ingestion Service",
    description="High-performance data ingestion service for real-time market data",
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
    if not ingestion_manager or not ingestion_manager.is_ready():
        raise HTTPException(status_code=503, detail="Service not ready")
    
    return {"status": "ready", "timestamp": ingestion_manager.get_status()["timestamp"]}


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
    if not ingestion_manager:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    status = ingestion_manager.get_status()
    health_status = await health_checker.get_health_status()
    
    return {
        "service": "data-ingestion",
        "status": status,
        "health": health_status,
        "metrics": metrics_collector.get_summary() if metrics_collector else {}
    }


@app.get("/sources")
async def get_data_sources():
    """Get information about configured data sources."""
    if not ingestion_manager:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    return ingestion_manager.get_sources_info()


@app.post("/sources/{source_name}/restart")
async def restart_data_source(source_name: str, background_tasks: BackgroundTasks):
    """Restart a specific data source."""
    if not ingestion_manager:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        background_tasks.add_task(ingestion_manager.restart_source, source_name)
        return {"message": f"Restart initiated for source: {source_name}"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Failed to restart source", source=source_name, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to restart source: {e}")


@app.post("/sources/{source_name}/pause")
async def pause_data_source(source_name: str):
    """Pause a specific data source."""
    if not ingestion_manager:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        await ingestion_manager.pause_source(source_name)
        return {"message": f"Source paused: {source_name}"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Failed to pause source", source=source_name, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to pause source: {e}")


@app.post("/sources/{source_name}/resume")
async def resume_data_source(source_name: str):
    """Resume a paused data source."""
    if not ingestion_manager:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        await ingestion_manager.resume_source(source_name)
        return {"message": f"Source resumed: {source_name}"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Failed to resume source", source=source_name, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to resume source: {e}")


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