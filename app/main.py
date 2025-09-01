"""
Market Scanning Engine - FastAPI Application
Main entry point for the risk-based market scanning API.
"""

import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from app.core.config import get_settings
from app.core.database import init_db, close_db
from app.api.v1.router import api_router
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
    
    yield
    
    # Shutdown
    logger.info("Shutting down Market Scanning Engine...")
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
        return {
            "status": "healthy",
            "service": "market-scanning-engine",
            "version": "0.1.0"
        }
    
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