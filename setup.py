#!/usr/bin/env python3
"""
Market Scanning Engine - Setup Configuration
Setup script for the market scanning engine package.
"""

import os
from setuptools import setup, find_packages

# Read requirements from files
def read_requirements(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="market-scanning-engine",
    version="0.1.0",
    description="Risk-based market scanning engine for trading signals",
    long_description=open("README.md", "r").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    
    # Package structure
    packages=find_packages(include=['data_models', 'data_models.*', 'signal_generation', 'signal_generation.*']),
    python_requires=">=3.9",
    
    # Dependencies
    install_requires=[
        # Core dependencies
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.5.0",
        "pydantic-settings>=2.1.0",
        
        # Database
        "asyncpg>=0.29.0",
        "sqlalchemy[asyncio]>=2.0.23",
        "alembic>=1.12.1",
        
        # Data processing
        "pandas>=2.1.4",
        "numpy>=1.25.2",
        
        # Kafka
        "confluent-kafka>=2.3.0",
        
        # Redis
        "redis[hiredis]>=5.0.1",
        "aioredis>=2.0.1",
        
        # HTTP clients
        "aiohttp>=3.9.1",
        "httpx>=0.25.2",
        
        # Utils
        "python-multipart>=0.0.6",
        "structlog>=23.2.0",
        "tenacity>=8.2.3",
    ],
    
    # Optional dependencies
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-asyncio>=0.21.1",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.7.0",
        ],
        "monitoring": [
            "prometheus-client>=0.19.0",
        ],
        "influx": [
            "influxdb-client[async]>=1.39.0",
        ]
    },
    
    # Entry points
    entry_points={
        "console_scripts": [
            "market-scanner=signal_generation.cli:main",
        ],
    },
    
    # Metadata
    author="Market Scanning Engine Team",
    author_email="dev@market-scanner.com",
    url="https://github.com/market-scanning-engine/core",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
)