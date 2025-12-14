"""Data module for database connections and queries."""

from .db_connection import DatabaseConnection, get_db_connection
# from .queries import TrafficQueries

__all__ = ["DatabaseConnection", "get_db_connection", "TrafficQueries"]

