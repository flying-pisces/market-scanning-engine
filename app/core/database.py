"""
Database connection and session management
"""

import logging
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncSession, 
    create_async_engine, 
    async_sessionmaker
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import text

logger = logging.getLogger(__name__)

# Database engine and session factory
engine: Optional[object] = None
SessionFactory: Optional[async_sessionmaker] = None


class Base(DeclarativeBase):
    """Base class for all database models"""
    pass


async def init_db(database_url: str) -> None:
    """Initialize database connection"""
    global engine, SessionFactory
    
    try:
        engine = create_async_engine(
            database_url,
            echo=True,  # Set to False in production
            pool_pre_ping=True,
            pool_recycle=300,
        )
        
        SessionFactory = async_sessionmaker(
            bind=engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        # Test connection
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        
        logger.info("Database connection established successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


async def close_db() -> None:
    """Close database connection"""
    global engine
    
    if engine:
        await engine.dispose()
        logger.info("Database connection closed")


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Get database session context manager"""
    if not SessionFactory:
        raise RuntimeError("Database not initialized")
    
    async with SessionFactory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for FastAPI to get database session"""
    async with get_db_session() as session:
        yield session