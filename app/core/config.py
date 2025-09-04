"""
Configuration management for Market Scanning Engine
Comprehensive settings for all services and environments
"""

import os
from functools import lru_cache
from typing import List, Optional, Dict, Any
from enum import Enum

from pydantic_settings import BaseSettings
from pydantic import Field


class Environment(str, Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Settings(BaseSettings):
    """Comprehensive application settings"""
    
    # === BASIC APPLICATION CONFIG ===
    environment: Environment = Environment.DEVELOPMENT
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    log_level: LogLevel = LogLevel.INFO
    
    # Security
    allowed_hosts: List[str] = ["*"]
    secret_key: str = Field(default="your-secret-key-change-in-production", env="SECRET_KEY")
    
    # === DATABASE CONFIGURATION ===
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:password@localhost/market_scanner",
        env="DATABASE_URL"
    )
    database_pool_size: int = 10
    database_max_overflow: int = 20
    database_pool_timeout: int = 30
    
    # === KAFKA CONFIGURATION ===
    kafka_bootstrap_servers: str = Field(default="localhost:9092", env="KAFKA_BOOTSTRAP_SERVERS")
    kafka_client_id: str = "market-scanner"
    kafka_acks: str = "1"
    kafka_retries: int = 3
    kafka_batch_size: int = 16384
    kafka_compression_type: str = "snappy"
    
    # === REDIS CONFIGURATION ===
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    redis_max_connections: int = 10
    redis_retry_attempts: int = 3
    redis_default_ttl: int = 3600
    
    # === INFLUXDB CONFIGURATION ===
    influxdb_url: str = Field(default="http://localhost:8086", env="INFLUXDB_URL")
    influxdb_token: str = Field(default="your-token-here", env="INFLUXDB_TOKEN")
    influxdb_org: str = Field(default="market-scanner", env="INFLUXDB_ORG")
    influxdb_bucket: str = Field(default="market-data", env="INFLUXDB_BUCKET")
    
    # === MARKET DATA API KEYS ===
    alpha_vantage_api_key: str = Field(default="demo", env="ALPHA_VANTAGE_API_KEY")
    finnhub_api_key: str = Field(default="demo", env="FINNHUB_API_KEY")
    polygon_api_key: str = Field(default="demo", env="POLYGON_API_KEY")
    iex_api_token: str = Field(default="demo", env="IEX_API_TOKEN")
    
    # === ML CONFIGURATION ===
    ml_model_cache_dir: str = "models"
    ml_max_training_time: int = 3600  # 1 hour
    ml_retraining_interval_hours: int = 24
    ml_ensemble_enabled: bool = True
    ml_prediction_confidence_threshold: float = 0.6
    ml_feature_count: int = 20
    
    # === PORTFOLIO OPTIMIZATION ===
    portfolio_max_positions: int = 20
    portfolio_rebalance_threshold: float = 0.05
    portfolio_min_position_size: float = 0.01
    portfolio_default_risk_free_rate: float = 0.02
    portfolio_optimization_timeout: int = 300  # 5 minutes
    
    # === BACKTESTING CONFIGURATION ===
    backtest_max_concurrent: int = 3
    backtest_default_commission: float = 5.0
    backtest_default_slippage: float = 0.001
    backtest_max_duration_days: int = 1825  # 5 years
    backtest_result_retention_days: int = 30
    
    # === RISK MANAGEMENT ===
    risk_var_confidence_level: float = 0.95
    risk_max_portfolio_drawdown: float = 0.20
    risk_position_size_limit: float = 0.10
    risk_sector_concentration_limit: float = 0.30
    risk_correlation_limit: float = 0.80
    risk_stress_test_scenarios: int = 1000
    
    # === SIGNAL PROCESSING ===
    signal_max_processing_time: int = 30
    signal_cache_enabled: bool = True
    signal_cache_ttl: int = 300
    signal_validation_level: str = "standard"  # basic, standard, strict, premium
    signal_min_confidence: float = 0.25
    signal_max_per_request: int = 100
    
    # === API RATE LIMITING ===
    api_rate_limit_per_minute: int = 1000
    api_burst_limit: int = 100
    api_rate_limit_storage: str = "redis"  # redis, memory
    api_enable_throttling: bool = True
    
    # === MONITORING & METRICS ===
    metrics_collection_interval: float = 60.0
    metrics_retention_days: int = 90
    monitoring_enabled: bool = True
    alert_cpu_threshold: float = 80.0
    alert_memory_threshold: float = 85.0
    alert_disk_threshold: float = 90.0
    
    # === WEBSOCKET CONFIGURATION ===
    websocket_max_connections: int = 1000
    websocket_ping_interval: int = 30
    websocket_ping_timeout: int = 10
    
    # === BACKGROUND JOBS ===
    job_max_concurrent: int = 5
    job_queue_max_size: int = 1000
    job_retry_attempts: int = 3
    job_retry_delay: float = 60.0
    
    # === OPTIONS PRICING ===
    options_risk_free_rate: float = 0.02
    options_dividend_yield: float = 0.02
    options_volatility_surface_points: int = 100
    options_greeks_calculation_enabled: bool = True
    
    # === EXECUTION & ROUTING ===
    execution_enabled: bool = False  # Disabled by default for safety
    execution_max_order_size: float = 100000.0
    execution_slippage_tolerance: float = 0.002
    execution_timeout_seconds: int = 30
    smart_routing_enabled: bool = True
    
    # === USER ANALYTICS ===
    analytics_enabled: bool = True
    analytics_batch_size: int = 100
    analytics_retention_days: int = 365
    user_behavior_tracking: bool = True
    
    # === DEVELOPMENT & TESTING ===
    enable_debug_endpoints: bool = False
    mock_market_data: bool = False
    test_mode: bool = False
    performance_profiling: bool = False
    
    # === KUBERNETES & DEPLOYMENT ===
    k8s_namespace: str = "market-scanner"
    k8s_replica_count: int = 2
    k8s_max_replicas: int = 10
    k8s_min_replicas: int = 2
    k8s_cpu_request: str = "100m"
    k8s_cpu_limit: str = "500m"
    k8s_memory_request: str = "256Mi"
    k8s_memory_limit: str = "1Gi"
    
    # === HEALTH CHECKS ===
    health_check_interval: int = 30
    health_check_timeout: int = 10
    health_check_retries: int = 3
    
    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment == Environment.PRODUCTION
    
    @property
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.environment == Environment.DEVELOPMENT
    
    @property
    def is_testing(self) -> bool:
        """Check if running in testing"""
        return self.environment == Environment.TESTING
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration dictionary"""
        return {
            "url": self.database_url,
            "pool_size": self.database_pool_size,
            "max_overflow": self.database_max_overflow,
            "pool_timeout": self.database_pool_timeout
        }
    
    def get_kafka_config(self) -> Dict[str, Any]:
        """Get Kafka configuration dictionary"""
        return {
            "bootstrap_servers": self.kafka_bootstrap_servers,
            "client_id": self.kafka_client_id,
            "acks": self.kafka_acks,
            "retries": self.kafka_retries,
            "batch_size": self.kafka_batch_size,
            "compression_type": self.kafka_compression_type
        }
    
    def get_redis_config(self) -> Dict[str, Any]:
        """Get Redis configuration dictionary"""
        return {
            "url": self.redis_url,
            "max_connections": self.redis_max_connections,
            "retry_attempts": self.redis_retry_attempts,
            "default_ttl": self.redis_default_ttl
        }
    
    def get_ml_config(self) -> Dict[str, Any]:
        """Get ML configuration dictionary"""
        return {
            "model_cache_dir": self.ml_model_cache_dir,
            "max_training_time": self.ml_max_training_time,
            "retraining_interval_hours": self.ml_retraining_interval_hours,
            "ensemble_enabled": self.ml_ensemble_enabled,
            "prediction_confidence_threshold": self.ml_prediction_confidence_threshold,
            "feature_count": self.ml_feature_count
        }
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"


# Environment-specific settings
class DevelopmentSettings(Settings):
    """Development environment settings"""
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = True
    log_level: LogLevel = LogLevel.DEBUG
    enable_debug_endpoints: bool = True
    mock_market_data: bool = True
    
    # Relaxed limits for development
    api_rate_limit_per_minute: int = 10000
    ml_max_training_time: int = 600  # 10 minutes
    

class ProductionSettings(Settings):
    """Production environment settings"""
    environment: Environment = Environment.PRODUCTION
    debug: bool = False
    log_level: LogLevel = LogLevel.INFO
    enable_debug_endpoints: bool = False
    mock_market_data: bool = False
    
    # Stricter security in production
    allowed_hosts: List[str] = ["yourdomain.com", "api.yourdomain.com"]
    
    # Production performance settings
    database_pool_size: int = 20
    redis_max_connections: int = 20
    job_max_concurrent: int = 10


class TestingSettings(Settings):
    """Testing environment settings"""
    environment: Environment = Environment.TESTING
    debug: bool = True
    log_level: LogLevel = LogLevel.DEBUG
    test_mode: bool = True
    mock_market_data: bool = True
    
    # In-memory databases for testing
    database_url: str = "sqlite+aiosqlite:///:memory:"
    redis_url: str = "redis://localhost:6379/15"  # Test DB
    
    # Faster execution for tests
    ml_max_training_time: int = 60
    backtest_max_concurrent: int = 1


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance based on environment"""
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    if env == "production":
        return ProductionSettings()
    elif env == "testing":
        return TestingSettings()
    else:
        return DevelopmentSettings()


def get_settings_for_environment(env: str) -> Settings:
    """Get settings for specific environment"""
    if env.lower() == "production":
        return ProductionSettings()
    elif env.lower() == "testing":
        return TestingSettings()
    else:
        return DevelopmentSettings()