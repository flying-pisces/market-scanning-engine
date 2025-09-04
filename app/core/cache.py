"""
Redis caching layer for improved performance
Handles caching of market data, signals, and user preferences
"""

import json
import logging
import asyncio
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
import pickle
import hashlib

import aioredis
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)


class CacheConfig:
    """Redis cache configuration"""
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        default_ttl: int = 3600,  # 1 hour
        max_connections: int = 10,
        retry_attempts: int = 3,
        retry_delay: float = 1.0
    ):
        self.redis_url = redis_url
        self.default_ttl = default_ttl
        self.max_connections = max_connections
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay


class CacheNamespaces:
    """Cache key namespaces for different data types"""
    
    MARKET_DATA = "market_data"
    SIGNALS = "signals"
    USER_PREFERENCES = "user_prefs"
    TECHNICAL_INDICATORS = "tech_indicators"
    PRICE_HISTORY = "price_history"
    RISK_SCORES = "risk_scores"
    MATCHING_RESULTS = "matches"
    SYSTEM_METRICS = "system"
    SESSION_DATA = "sessions"


class RedisCache:
    """Redis-based caching service"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.redis_client: Optional[aioredis.Redis] = None
        self.is_connected = False
    
    async def connect(self) -> bool:
        """Connect to Redis server"""
        try:
            self.redis_client = aioredis.from_url(
                self.config.redis_url,
                max_connections=self.config.max_connections,
                retry_on_timeout=True,
                decode_responses=False  # Handle bytes for pickle serialization
            )
            
            # Test connection
            await self.redis_client.ping()
            self.is_connected = True
            logger.info("Redis cache connected successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.is_connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from Redis server"""
        if self.redis_client:
            await self.redis_client.close()
            self.is_connected = False
            logger.info("Redis cache disconnected")
    
    async def health_check(self) -> bool:
        """Check Redis connection health"""
        if not self.redis_client or not self.is_connected:
            return False
        
        try:
            await self.redis_client.ping()
            return True
        except Exception:
            self.is_connected = False
            return False
    
    def _make_key(self, namespace: str, identifier: str, suffix: str = None) -> str:
        """Create Redis key with namespace"""
        key = f"{namespace}:{identifier}"
        if suffix:
            key += f":{suffix}"
        return key
    
    def _hash_key(self, data: Union[str, Dict, List]) -> str:
        """Create hash for complex keys"""
        if isinstance(data, str):
            return data
        
        serialized = json.dumps(data, sort_keys=True)
        return hashlib.md5(serialized.encode()).hexdigest()[:16]
    
    async def set(
        self,
        namespace: str,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        serialize: str = "json"
    ) -> bool:
        """Set value in cache"""
        if not await self.health_check():
            if not await self.connect():
                return False
        
        try:
            redis_key = self._make_key(namespace, key)
            cache_ttl = ttl or self.config.default_ttl
            
            # Serialize value
            if serialize == "json":
                serialized_value = json.dumps(value, default=str)
            elif serialize == "pickle":
                serialized_value = pickle.dumps(value)
            else:
                serialized_value = str(value)
            
            # Set with TTL
            await self.redis_client.setex(redis_key, cache_ttl, serialized_value)
            return True
            
        except Exception as e:
            logger.error(f"Failed to set cache key {key}: {e}")
            return False
    
    async def get(
        self,
        namespace: str,
        key: str,
        deserialize: str = "json"
    ) -> Optional[Any]:
        """Get value from cache"""
        if not await self.health_check():
            return None
        
        try:
            redis_key = self._make_key(namespace, key)
            cached_value = await self.redis_client.get(redis_key)
            
            if cached_value is None:
                return None
            
            # Deserialize value
            if deserialize == "json":
                return json.loads(cached_value)
            elif deserialize == "pickle":
                return pickle.loads(cached_value)
            else:
                return cached_value.decode('utf-8') if isinstance(cached_value, bytes) else cached_value
            
        except Exception as e:
            logger.error(f"Failed to get cache key {key}: {e}")
            return None
    
    async def delete(self, namespace: str, key: str) -> bool:
        """Delete key from cache"""
        if not await self.health_check():
            return False
        
        try:
            redis_key = self._make_key(namespace, key)
            result = await self.redis_client.delete(redis_key)
            return result > 0
            
        except Exception as e:
            logger.error(f"Failed to delete cache key {key}: {e}")
            return False
    
    async def exists(self, namespace: str, key: str) -> bool:
        """Check if key exists in cache"""
        if not await self.health_check():
            return False
        
        try:
            redis_key = self._make_key(namespace, key)
            result = await self.redis_client.exists(redis_key)
            return result > 0
            
        except Exception as e:
            logger.error(f"Failed to check cache key {key}: {e}")
            return False
    
    async def expire(self, namespace: str, key: str, ttl: int) -> bool:
        """Set TTL for existing key"""
        if not await self.health_check():
            return False
        
        try:
            redis_key = self._make_key(namespace, key)
            result = await self.redis_client.expire(redis_key, ttl)
            return result
            
        except Exception as e:
            logger.error(f"Failed to set TTL for cache key {key}: {e}")
            return False
    
    async def get_ttl(self, namespace: str, key: str) -> Optional[int]:
        """Get TTL for key"""
        if not await self.health_check():
            return None
        
        try:
            redis_key = self._make_key(namespace, key)
            ttl = await self.redis_client.ttl(redis_key)
            return ttl if ttl > 0 else None
            
        except Exception as e:
            logger.error(f"Failed to get TTL for cache key {key}: {e}")
            return None
    
    async def mget(self, namespace: str, keys: List[str], deserialize: str = "json") -> Dict[str, Any]:
        """Get multiple values from cache"""
        if not await self.health_check():
            return {}
        
        try:
            redis_keys = [self._make_key(namespace, key) for key in keys]
            cached_values = await self.redis_client.mget(redis_keys)
            
            result = {}
            for i, (original_key, cached_value) in enumerate(zip(keys, cached_values)):
                if cached_value is not None:
                    try:
                        if deserialize == "json":
                            result[original_key] = json.loads(cached_value)
                        elif deserialize == "pickle":
                            result[original_key] = pickle.loads(cached_value)
                        else:
                            result[original_key] = cached_value.decode('utf-8') if isinstance(cached_value, bytes) else cached_value
                    except Exception as e:
                        logger.warning(f"Failed to deserialize cached value for {original_key}: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get multiple cache keys: {e}")
            return {}
    
    async def mset(self, namespace: str, data: Dict[str, Any], ttl: Optional[int] = None, serialize: str = "json") -> bool:
        """Set multiple values in cache"""
        if not await self.health_check():
            if not await self.connect():
                return False
        
        try:
            cache_ttl = ttl or self.config.default_ttl
            pipe = self.redis_client.pipeline()
            
            for key, value in data.items():
                redis_key = self._make_key(namespace, key)
                
                # Serialize value
                if serialize == "json":
                    serialized_value = json.dumps(value, default=str)
                elif serialize == "pickle":
                    serialized_value = pickle.dumps(value)
                else:
                    serialized_value = str(value)
                
                pipe.setex(redis_key, cache_ttl, serialized_value)
            
            await pipe.execute()
            return True
            
        except Exception as e:
            logger.error(f"Failed to set multiple cache keys: {e}")
            return False
    
    async def delete_pattern(self, namespace: str, pattern: str) -> int:
        """Delete keys matching pattern"""
        if not await self.health_check():
            return 0
        
        try:
            search_pattern = self._make_key(namespace, pattern)
            keys = []
            
            async for key in self.redis_client.scan_iter(match=search_pattern):
                keys.append(key)
            
            if keys:
                return await self.redis_client.delete(*keys)
            return 0
            
        except Exception as e:
            logger.error(f"Failed to delete pattern {pattern}: {e}")
            return 0
    
    async def increment(self, namespace: str, key: str, amount: int = 1, ttl: Optional[int] = None) -> Optional[int]:
        """Increment counter"""
        if not await self.health_check():
            if not await self.connect():
                return None
        
        try:
            redis_key = self._make_key(namespace, key)
            result = await self.redis_client.incrby(redis_key, amount)
            
            # Set TTL if specified and key is new
            if ttl and result == amount:
                await self.redis_client.expire(redis_key, ttl)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to increment cache key {key}: {e}")
            return None
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get Redis stats"""
        if not await self.health_check():
            return {"connected": False}
        
        try:
            info = await self.redis_client.info()
            return {
                "connected": True,
                "used_memory": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "total_commands_processed": info.get("total_commands_processed"),
                "keyspace_hits": info.get("keyspace_hits"),
                "keyspace_misses": info.get("keyspace_misses"),
                "uptime_in_seconds": info.get("uptime_in_seconds")
            }
            
        except Exception as e:
            logger.error(f"Failed to get Redis stats: {e}")
            return {"connected": False, "error": str(e)}


class MarketDataCache:
    """Specialized cache for market data"""
    
    def __init__(self, redis_cache: RedisCache):
        self.cache = redis_cache
    
    async def cache_quote(self, symbol: str, quote_data: Dict[str, Any], ttl: int = 300):
        """Cache real-time quote data (5 min default TTL)"""
        return await self.cache.set(
            CacheNamespaces.MARKET_DATA,
            f"quote:{symbol}",
            quote_data,
            ttl=ttl
        )
    
    async def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get cached quote data"""
        return await self.cache.get(CacheNamespaces.MARKET_DATA, f"quote:{symbol}")
    
    async def cache_price_history(self, symbol: str, timeframe: str, prices: List[float], ttl: int = 1800):
        """Cache price history (30 min default TTL)"""
        return await self.cache.set(
            CacheNamespaces.PRICE_HISTORY,
            f"{symbol}:{timeframe}",
            prices,
            ttl=ttl,
            serialize="pickle"
        )
    
    async def get_price_history(self, symbol: str, timeframe: str) -> Optional[List[float]]:
        """Get cached price history"""
        return await self.cache.get(
            CacheNamespaces.PRICE_HISTORY,
            f"{symbol}:{timeframe}",
            deserialize="pickle"
        )
    
    async def cache_technical_indicators(self, symbol: str, indicators: Dict[str, Any], ttl: int = 600):
        """Cache technical indicators (10 min default TTL)"""
        return await self.cache.set(
            CacheNamespaces.TECHNICAL_INDICATORS,
            symbol,
            indicators,
            ttl=ttl
        )
    
    async def get_technical_indicators(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get cached technical indicators"""
        return await self.cache.get(CacheNamespaces.TECHNICAL_INDICATORS, symbol)


class SignalCache:
    """Specialized cache for trading signals"""
    
    def __init__(self, redis_cache: RedisCache):
        self.cache = redis_cache
    
    async def cache_signal(self, signal_id: str, signal_data: Dict[str, Any], ttl: int = 3600):
        """Cache signal data (1 hour default TTL)"""
        return await self.cache.set(
            CacheNamespaces.SIGNALS,
            signal_id,
            signal_data,
            ttl=ttl
        )
    
    async def get_signal(self, signal_id: str) -> Optional[Dict[str, Any]]:
        """Get cached signal data"""
        return await self.cache.get(CacheNamespaces.SIGNALS, signal_id)
    
    async def cache_user_matches(self, user_id: int, matches: List[Dict[str, Any]], ttl: int = 1800):
        """Cache user signal matches (30 min default TTL)"""
        return await self.cache.set(
            CacheNamespaces.MATCHING_RESULTS,
            f"user:{user_id}",
            matches,
            ttl=ttl
        )
    
    async def get_user_matches(self, user_id: int) -> Optional[List[Dict[str, Any]]]:
        """Get cached user matches"""
        return await self.cache.get(CacheNamespaces.MATCHING_RESULTS, f"user:{user_id}")
    
    async def invalidate_user_matches(self, user_id: int) -> bool:
        """Invalidate cached user matches"""
        return await self.cache.delete(CacheNamespaces.MATCHING_RESULTS, f"user:{user_id}")


class UserCache:
    """Specialized cache for user data"""
    
    def __init__(self, redis_cache: RedisCache):
        self.cache = redis_cache
    
    async def cache_user_preferences(self, user_id: int, preferences: Dict[str, Any], ttl: int = 7200):
        """Cache user preferences (2 hours default TTL)"""
        return await self.cache.set(
            CacheNamespaces.USER_PREFERENCES,
            str(user_id),
            preferences,
            ttl=ttl
        )
    
    async def get_user_preferences(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get cached user preferences"""
        return await self.cache.get(CacheNamespaces.USER_PREFERENCES, str(user_id))
    
    async def cache_risk_score(self, user_id: int, asset_class: str, risk_score: int, ttl: int = 3600):
        """Cache calculated risk score (1 hour default TTL)"""
        return await self.cache.set(
            CacheNamespaces.RISK_SCORES,
            f"{user_id}:{asset_class}",
            risk_score,
            ttl=ttl
        )
    
    async def get_risk_score(self, user_id: int, asset_class: str) -> Optional[int]:
        """Get cached risk score"""
        return await self.cache.get(CacheNamespaces.RISK_SCORES, f"{user_id}:{asset_class}")


# Global cache instances (initialized at startup)
redis_cache: Optional[RedisCache] = None
market_data_cache: Optional[MarketDataCache] = None
signal_cache: Optional[SignalCache] = None
user_cache: Optional[UserCache] = None


async def init_cache(redis_url: str = "redis://localhost:6379") -> bool:
    """Initialize Redis cache system"""
    global redis_cache, market_data_cache, signal_cache, user_cache
    
    try:
        config = CacheConfig(redis_url=redis_url)
        redis_cache = RedisCache(config)
        
        if await redis_cache.connect():
            # Initialize specialized caches
            market_data_cache = MarketDataCache(redis_cache)
            signal_cache = SignalCache(redis_cache)
            user_cache = UserCache(redis_cache)
            
            logger.info("Cache system initialized successfully")
            return True
        else:
            logger.error("Failed to initialize cache system")
            return False
            
    except Exception as e:
        logger.error(f"Cache initialization failed: {e}")
        return False


async def close_cache():
    """Close cache connections"""
    global redis_cache
    
    if redis_cache:
        await redis_cache.disconnect()
        redis_cache = None
    
    logger.info("Cache connections closed")


def get_cache() -> Optional[RedisCache]:
    """Get global cache instance"""
    return redis_cache


def get_market_cache() -> Optional[MarketDataCache]:
    """Get market data cache instance"""
    return market_data_cache


def get_signal_cache() -> Optional[SignalCache]:
    """Get signal cache instance"""
    return signal_cache


def get_user_cache() -> Optional[UserCache]:
    """Get user cache instance"""
    return user_cache