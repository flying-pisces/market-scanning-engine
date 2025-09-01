"""
Main API router for v1 endpoints
"""

from fastapi import APIRouter

from app.api.v1 import users, signals, matching

# Create main API router
api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(users.router)
api_router.include_router(signals.router)
api_router.include_router(matching.router)

# Version info endpoint
@api_router.get("/", tags=["API Info"])
async def api_info():
    """API version information"""
    return {
        "api_version": "1.0",
        "service": "market-scanning-engine",
        "description": "Risk-based trading signal generation and matching",
        "endpoints": {
            "users": "/users - User profile management",
            "signals": "/signals - Signal generation and management", 
            "matching": "/matching - User-signal matching",
            "health": "/health - System health check"
        },
        "features": [
            "0-100 risk scoring system",
            "Multi-asset class support (options, stocks, bonds, safe assets)",
            "Real-time signal generation",
            "Intelligent user-signal matching",
            "RESTful API with async database operations"
        ]
    }