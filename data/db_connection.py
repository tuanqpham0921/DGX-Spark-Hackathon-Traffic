"""Database connection module for PostgreSQL."""

import asyncpg
import logging
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class DatabaseConnection:
    """Manages PostgreSQL database connections using asyncpg."""
    
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ):
        """Initialize database connection parameters."""
        self.host = host or os.getenv("DB_HOST", "localhost")
        self.port = port or int(os.getenv("DB_PORT", "5432"))
        self.database = database or os.getenv("DB_NAME", "traffix")
        self.user = user or os.getenv("DB_USER", "postgres")
        self.password = password or os.getenv("DB_PASSWORD", "")
        self.pool: Optional[asyncpg.Pool] = None
        
    async def create_pool(self, min_size: int = 10, max_size: int = 20) -> None:
        """Create connection pool."""
        try:
            self.pool = await asyncpg.create_pool(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                min_size=min_size,
                max_size=max_size,
            )
            logger.info("Database connection pool created successfully")
        except Exception as e:
            logger.error(f"Failed to create database pool: {e}")
            raise
            
    async def close_pool(self) -> None:
        """Close connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("Database connection pool closed")
            
    @asynccontextmanager
    async def acquire_connection(self):
        """Context manager for acquiring database connection."""
        if not self.pool:
            await self.create_pool()
            
        async with self.pool.acquire() as connection:
            yield connection
            
    async def execute_query(
        self,
        query: str,
        *args,
        timeout: float = 30.0
    ) -> List[asyncpg.Record]:
        """Execute a SELECT query and return results."""
        async with self.acquire_connection() as conn:
            try:
                results = await conn.fetch(query, *args, timeout=timeout)
                logger.debug(f"Query executed successfully: {len(results)} rows")
                return results
            except Exception as e:
                logger.error(f"Query execution failed: {e}")
                raise
                
    async def execute_query_one(
        self,
        query: str,
        *args,
        timeout: float = 30.0
    ) -> Optional[asyncpg.Record]:
        """Execute a SELECT query and return one result."""
        async with self.acquire_connection() as conn:
            try:
                result = await conn.fetchrow(query, *args, timeout=timeout)
                return result
            except Exception as e:
                logger.error(f"Query execution failed: {e}")
                raise
                
    async def execute_many(
        self,
        query: str,
        args_list: List[tuple],
        timeout: float = 30.0
    ) -> None:
        """Execute a query with multiple parameter sets."""
        async with self.acquire_connection() as conn:
            try:
                await conn.executemany(query, args_list, timeout=timeout)
                logger.debug(f"Batch query executed: {len(args_list)} operations")
            except Exception as e:
                logger.error(f"Batch execution failed: {e}")
                raise
                
    def records_to_dicts(self, records: List[asyncpg.Record]) -> List[Dict[str, Any]]:
        """Convert asyncpg records to list of dictionaries."""
        return [dict(record) for record in records]


# Global database connection instance
_db_connection: Optional[DatabaseConnection] = None


def get_db_connection() -> DatabaseConnection:
    """Get or create global database connection instance."""
    global _db_connection
    if _db_connection is None:
        _db_connection = DatabaseConnection()
    return _db_connection


async def initialize_database():
    """Initialize database connection pool."""
    db = get_db_connection()
    await db.create_pool()
    

async def close_database():
    """Close database connection pool."""
    global _db_connection
    if _db_connection:
        await _db_connection.close_pool()
        _db_connection = None

