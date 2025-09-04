"""
Market Scanning Engine - FastAPI Application
Main entry point for the risk-based market scanning API.
"""

import logging
from contextlib import asynccontextmanager
from typing import Dict, Any
import asyncio

from fastapi import FastAPI, HTTPException, Depends, status, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from app.core.config import get_settings
from app.core.database import init_db, close_db
from app.core.kafka_client import init_kafka, close_kafka
from app.core.cache import init_cache, close_cache
from app.services.background_jobs import init_background_jobs, get_job_processor
from app.core.monitoring import init_monitoring, get_metrics_collector
from app.ml.training_service import init_ml_training_service
from app.ml.prediction_service import init_ml_prediction_service
from app.portfolio.allocation_service import init_portfolio_allocation_service
from app.backtesting.service import init_backtesting_service
from app.api.v1.router import api_router
from app.api.websocket import websocket_endpoint, websocket_manager
from app.core.exceptions import APIException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("Starting Market Scanning Engine...")
    settings = get_settings()
    
    # Initialize database
    await init_db(settings.database_url)
    logger.info("Database initialized")
    
    # Initialize Redis cache
    redis_url = getattr(settings, 'redis_url', 'redis://localhost:6379')
    if await init_cache(redis_url):
        logger.info("Redis cache initialized")
    else:
        logger.warning("Failed to initialize Redis cache - performance may be impacted")
    
    # Initialize Kafka infrastructure
    kafka_servers = getattr(settings, 'kafka_bootstrap_servers', 'localhost:9092')
    if await init_kafka(kafka_servers):
        logger.info("Kafka infrastructure initialized")
    else:
        logger.warning("Failed to initialize Kafka - some features may be limited")
    
    # Initialize monitoring system
    metrics_collector = await init_monitoring(collection_interval=60.0)
    monitoring_task = asyncio.create_task(metrics_collector.start())
    logger.info("Monitoring system started")
    
    # Initialize ML services
    ml_training_service = await init_ml_training_service()
    ml_training_task = asyncio.create_task(ml_training_service.start())
    logger.info("ML Training Service started")
    
    ml_prediction_service = await init_ml_prediction_service()
    ml_prediction_task = asyncio.create_task(ml_prediction_service.start())
    logger.info("ML Prediction Service started")
    
    # Initialize portfolio allocation service
    portfolio_service = await init_portfolio_allocation_service()
    portfolio_task = asyncio.create_task(portfolio_service.start())
    logger.info("Portfolio Allocation Service started")
    
    # Initialize backtesting service
    backtest_service = await init_backtesting_service()
    backtest_task = asyncio.create_task(backtest_service.start())
    logger.info("Backtesting Service started")
    
    # Initialize background job processor
    job_processor = await init_background_jobs()
    background_task = asyncio.create_task(job_processor.start())
    logger.info("Background job processor started")
    
    # Start WebSocket Kafka consumers
    websocket_task = asyncio.create_task(websocket_manager.start_kafka_consumers())
    logger.info("WebSocket service started")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Market Scanning Engine...")
    
    # Stop ML services
    if 'ml_training_service' in locals():
        await ml_training_service.stop()
    if 'ml_training_task' in locals():
        ml_training_task.cancel()
        try:
            await ml_training_task
        except asyncio.CancelledError:
            pass
    
    if 'ml_prediction_service' in locals():
        await ml_prediction_service.stop()
    if 'ml_prediction_task' in locals():
        ml_prediction_task.cancel()
        try:
            await ml_prediction_task
        except asyncio.CancelledError:
            pass
    
    if 'portfolio_service' in locals():
        await portfolio_service.stop()
    if 'portfolio_task' in locals():
        portfolio_task.cancel()
        try:
            await portfolio_task
        except asyncio.CancelledError:
            pass
    
    if 'backtest_service' in locals():
        await backtest_service.stop()
    if 'backtest_task' in locals():
        backtest_task.cancel()
        try:
            await backtest_task
        except asyncio.CancelledError:
            pass
    
    # Stop monitoring system
    if 'metrics_collector' in locals():
        await metrics_collector.stop()
    if 'monitoring_task' in locals():
        monitoring_task.cancel()
        try:
            await monitoring_task
        except asyncio.CancelledError:
            pass
    
    # Stop background job processor
    if 'job_processor' in locals():
        await job_processor.stop()
    if 'background_task' in locals():
        background_task.cancel()
        try:
            await background_task
        except asyncio.CancelledError:
            pass
    
    # Stop WebSocket consumers
    websocket_manager.is_running = False
    websocket_task.cancel()
    try:
        await websocket_task
    except asyncio.CancelledError:
        pass
    
    # Close cache connections
    await close_cache()
    logger.info("Cache connections closed")
    
    # Close Kafka connections
    await close_kafka()
    logger.info("Kafka connections closed")
    
    # Close database connections
    await close_db()
    logger.info("Database connections closed")

# Create FastAPI app
def create_app() -> FastAPI:
    settings = get_settings()
    
    app = FastAPI(
        title="Market Scanning Engine",
        description="Risk-based trading signal generation and matching system",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.environment != "production" else None,
        redoc_url="/redoc" if settings.environment != "production" else None,
    )
    
    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_hosts,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.allowed_hosts
    )
    
    # Add exception handler
    @app.exception_handler(APIException)
    async def api_exception_handler(request, exc: APIException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.message,
                "details": exc.details,
                "timestamp": exc.timestamp.isoformat()
            }
        )
    
    # Health check endpoint
    @app.get("/health", tags=["System"])
    async def health_check() -> Dict[str, Any]:
        """Health check endpoint"""
        metrics_collector = get_metrics_collector()
        if metrics_collector:
            health_status = metrics_collector.get_system_health()
            return {
                "service": "market-scanning-engine",
                "version": "0.1.0",
                **health_status
            }
        else:
            return {
                "status": "healthy",
                "service": "market-scanning-engine",
                "version": "0.1.0"
            }
    
    # Metrics endpoint
    @app.get("/metrics", tags=["System"])
    async def get_metrics() -> Dict[str, Any]:
        """Get system metrics"""
        metrics_collector = get_metrics_collector()
        if metrics_collector:
            return metrics_collector.get_metric_summary()
        else:
            return {"error": "Metrics collector not available"}
    
    # WebSocket endpoint
    @app.websocket("/ws/{user_id}")
    async def websocket_handler(websocket: WebSocket, user_id: int):
        """WebSocket endpoint for real-time updates"""
        await websocket_endpoint(websocket, user_id)
    
    @app.websocket("/ws/{user_id}/{connection_type}")
    async def typed_websocket_handler(websocket: WebSocket, user_id: int, connection_type: str):
        """WebSocket endpoint with connection type"""
        await websocket_endpoint(websocket, user_id, connection_type)
    
    # Include API router
    app.include_router(api_router, prefix="/api/v1")
    
    return app

# Create app instance
app = create_app()

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.environment == "development",
        log_level="info"
    )