"""
Configuration management for Market Scanning Engine
"""

import os
from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # Basic app config
    environment: str = "development"
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    
    # Security
    allowed_hosts: List[str] = ["*"]
    secret_key: str = "your-secret-key-change-in-production"
    
    # Database
    database_url: str = "postgresql+asyncpg://postgres:password@localhost/market_scanner"
    
    # Risk scoring parameters
    risk_score_precision: int = 2  # decimal places
    default_confidence_threshold: float = 0.6
    max_signals_per_request: int = 100
    
    # Signal generation limits
    max_processing_time_seconds: int = 30
    enable_signal_caching: bool = True
    cache_ttl_seconds: int = 300  # 5 minutes
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()