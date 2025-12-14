"""Simple caching layer for TRAFFIX queries."""

from typing import Any, Optional, Callable
from datetime import datetime, timedelta
import hashlib
import json
import asyncio
from functools import wraps


class SimpleCache:
    """Thread-safe in-memory cache with TTL."""
    
    def __init__(self, default_ttl: int = 3600):
        """
        Initialize cache.
        
        Args:
            default_ttl: Default time-to-live in seconds
        """
        self._cache = {}
        self._timestamps = {}
        self.default_ttl = default_ttl
        self._lock = asyncio.Lock()
        
    def _make_key(self, *args, **kwargs) -> str:
        """Create cache key from arguments."""
        key_data = {
            "args": args,
            "kwargs": kwargs
        }
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_string.encode()).hexdigest()
        
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        async with self._lock:
            if key not in self._cache:
                return None
                
            # Check if expired
            timestamp = self._timestamps.get(key)
            if timestamp and (datetime.now() - timestamp).total_seconds() > self.default_ttl:
                # Expired, remove it
                del self._cache[key]
                del self._timestamps[key]
                return None
                
            return self._cache[key]
            
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache."""
        async with self._lock:
            self._cache[key] = value
            self._timestamps[key] = datetime.now()
            
    async def delete(self, key: str):
        """Delete value from cache."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
            if key in self._timestamps:
                del self._timestamps[key]
                
    async def clear(self):
        """Clear all cache."""
        async with self._lock:
            self._cache.clear()
            self._timestamps.clear()
            
    def cache_async(self, ttl: Optional[int] = None):
        """
        Decorator for caching async function results.
        
        Args:
            ttl: Time-to-live in seconds (uses default if not provided)
        """
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Create cache key
                cache_key = f"{func.__name__}_{self._make_key(*args, **kwargs)}"
                
                # Try to get from cache
                cached_value = await self.get(cache_key)
                if cached_value is not None:
                    return cached_value
                    
                # Execute function
                result = await func(*args, **kwargs)
                
                # Store in cache
                await self.set(cache_key, result, ttl or self.default_ttl)
                
                return result
            return wrapper
        return decorator


# Global cache instance
_global_cache = None


def get_cache() -> SimpleCache:
    """Get or create global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = SimpleCache(default_ttl=3600)  # 1 hour default
    return _global_cache

