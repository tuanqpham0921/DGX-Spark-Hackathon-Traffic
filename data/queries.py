"""SQL queries for traffic data retrieval."""

from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import logging

from .db_connection import DatabaseConnection, get_db_connection

logger = logging.getLogger(__name__)

# NOTE (TQP): HERER
# some of these are not needed...
class TrafficQueries:
    """SQL query builder and executor for traffic data."""
    
    def __init__(self, db: Optional[DatabaseConnection] = None):
        """Initialize with database connection."""
        self.db = db or get_db_connection()
        
    async def get_events_by_region(
        self,
        region: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_types: Optional[List[str]] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Retrieve traffic events filtered by region and time.
        
        Args:
            region: "District of Columbia", "Virginia", or "All"
            start_time: Filter events after this time
            end_time: Filter events before this time
            event_types: Filter by event_details_level_1
            limit: Maximum number of records to return
        """
        query = """
            SELECT 
                office_event_id_tag,
                system,
                event_details_level_1,
                event_details_level_2,
                event_details_level_3,
                granular_incident_place,
                road,
                direction,
                county,
                state,
                "Start Time" as start_time,
                "End Time" as end_time,
                roadway_clearance_time,
                overall_event_time,
                latitude,
                longitude
            FROM public.events
            WHERE ($1 = 'All' OR state = $1)
        """
        
        params = [region]
        param_idx = 2
        
        if start_time:
            query += f' AND TO_TIMESTAMP("Start Time", \'MM/DD/YY HH24:MI\') >= ${param_idx}'
            params.append(start_time)
            param_idx += 1
            
        if end_time:
            query += f' AND TO_TIMESTAMP("End Time", \'MM/DD/YY HH24:MI\') <= ${param_idx}'
            params.append(end_time)
            param_idx += 1
            
        if event_types:
            query += f" AND event_details_level_1 = ANY(${param_idx})"
            params.append(event_types)
            param_idx += 1
            
        query += f" ORDER BY TO_TIMESTAMP(\"Start Time\", 'MM/DD/YY HH24:MI') DESC LIMIT ${param_idx}"
        params.append(limit)
        
        records = await self.db.execute_query(query, *params)
        return self.db.records_to_dicts(records)
        
    async def get_trips_by_region(
        self,
        region: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        hour_start: Optional[int] = None,
        hour_end: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve trip data filtered by region and time.
        
        Args:
            region: "District of Columbia", "Virginia", or "All"
            start_date: Filter trips after this date
            end_date: Filter trips before this date
            hour_start: Filter by hour of day (0-23)
            hour_end: Filter by hour of day (0-23)
        """
        query = """
            SELECT 
                date,
                hour,
                region,
                total_car_travel_forecast,
                total_car_travel_actual,
                unique_drivers,
                network_reliability,
                network_efficiency,
                events_in_hour
            FROM public.trips
            WHERE ($1 = 'All' OR region = $1)
        """
        
        params = [region]
        param_idx = 2
        
        if start_date:
            query += f" AND date >= ${param_idx}"
            params.append(start_date.date())
            param_idx += 1
            
        if end_date:
            query += f" AND date <= ${param_idx}"
            params.append(end_date.date())
            param_idx += 1
            
        if hour_start is not None:
            query += f" AND hour >= ${param_idx}"
            params.append(hour_start)
            param_idx += 1
            
        if hour_end is not None:
            query += f" AND hour <= ${param_idx}"
            params.append(hour_end)
            param_idx += 1
            
        query += " ORDER BY date DESC, hour DESC"
        
        records = await self.db.execute_query(query, *params)
        return self.db.records_to_dicts(records)
        
    async def get_weather_by_timerange(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict[str, Any]]:
        """
        Retrieve weather data for a time range.
        
        Args:
            start_time: Start of time range
            end_time: End of time range
        """
        query = """
            SELECT 
                datetime,
                temp,
                feelslike,
                dew,
                humidity,
                precip,
                precipprob,
                preciptype,
                windgust,
                windspeed,
                winddir,
                cloudcover,
                visibility,
                solarradiation,
                solarenergy,
                uvindex,
                severerisk,
                conditions,
                stations
            FROM public.weather
            WHERE datetime >= $1 AND datetime <= $2
            ORDER BY datetime ASC
        """
        
        records = await self.db.execute_query(query, start_time, end_time)
        return self.db.records_to_dicts(records)
        
    async def get_integrated_data(
        self,
        region: str,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """
        Retrieve integrated data combining events, trips, and weather.
        
        Args:
            region: "District of Columbia" or "Virginia"
            start_time: Start of analysis period
            end_time: End of analysis period
            
        Returns:
            Dictionary with events, trips, and weather data
        """
        # Get events
        events = await self.get_events_by_region(
            region=region,
            start_time=start_time,
            end_time=end_time
        )
        
        # Get trips
        trips = await self.get_trips_by_region(
            region=region,
            start_date=start_time,
            end_date=end_time
        )
        
        # Get weather
        weather = await self.get_weather_by_timerange(
            start_time=start_time,
            end_time=end_time
        )
        
        return {
            "events": events,
            "trips": trips,
            "weather": weather,
            "metadata": {
                "region": region,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "event_count": len(events),
                "trip_records": len(trips),
                "weather_records": len(weather)
            }
        }
        
    async def get_event_statistics(
        self,
        region: str,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """
        Get aggregated event statistics for a region and time period.
        
        Args:
            region: "District of Columbia" or "Virginia"
            start_time: Start of analysis period
            end_time: End of analysis period
        """
        query = """
            SELECT 
                event_details_level_1,
                COUNT(*) as event_count,
                AVG(roadway_clearance_time) as avg_clearance_time,
                AVG(overall_event_time) as avg_event_duration
            FROM public.events
            WHERE state = $1
                AND TO_TIMESTAMP("Start Time", 'MM/DD/YY HH24:MI') >= $2
                AND TO_TIMESTAMP("End Time", 'MM/DD/YY HH24:MI') <= $3
            GROUP BY event_details_level_1
            ORDER BY event_count DESC
        """
        
        records = await self.db.execute_query(query, region, start_time, end_time)
        return self.db.records_to_dicts(records)
        
    async def get_congestion_hotspots(
        self,
        region: str,
        start_time: datetime,
        end_time: datetime,
        min_events: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Identify congestion hotspots by road/location.
        
        Args:
            region: "District of Columbia" or "Virginia"
            start_time: Start of analysis period
            end_time: End of analysis period
            min_events: Minimum events to be considered a hotspot
        """
        query = """
            SELECT 
                road,
                county,
                COUNT(*) as event_count,
                AVG(latitude) as avg_latitude,
                AVG(longitude) as avg_longitude,
                ARRAY_AGG(DISTINCT event_details_level_1) as event_types
            FROM public.events
            WHERE state = $1
                AND TO_TIMESTAMP("Start Time", 'MM/DD/YY HH24:MI') >= $2
                AND TO_TIMESTAMP("End Time", 'MM/DD/YY HH24:MI') <= $3
                AND road IS NOT NULL
            GROUP BY road, county
            HAVING COUNT(*) >= $4
            ORDER BY event_count DESC
            LIMIT 20
        """
        
        records = await self.db.execute_query(
            query, region, start_time, end_time, min_events
        )
        return self.db.records_to_dicts(records)
        
    async def get_hourly_metrics(
        self,
        region: str,
        date: datetime
    ) -> List[Dict[str, Any]]:
        """
        Get hourly traffic metrics for a specific date.
        
        Args:
            region: "District of Columbia" or "Virginia"
            date: Date to analyze
        """
        query = """
            SELECT 
                hour,
                total_car_travel_actual,
                total_car_travel_forecast,
                unique_drivers,
                network_reliability,
                network_efficiency,
                events_in_hour,
                (total_car_travel_actual - total_car_travel_forecast) as forecast_error
            FROM public.trips
            WHERE region = $1 AND date = $2
            ORDER BY hour ASC
        """
        
        records = await self.db.execute_query(query, region, date.date())
        return self.db.records_to_dicts(records)
        
    async def get_dashboard_metrics(
        self,
        region: str,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """
        Get aggregated metrics for dashboard cards.
        
        Args:
            region: "District of Columbia", "Virginia", or "All"
            start_time: Start of analysis period
            end_time: End of analysis period
            
        Returns:
            Dictionary with metrics: total_trips, avg_reliability, avg_efficiency, active_events
        """
        # Get trip metrics
        trip_query = """
            SELECT 
                SUM(total_car_travel_actual) as total_trips,
                SUM(total_car_travel_forecast) as total_forecast,
                AVG(network_reliability) as avg_reliability,
                AVG(network_efficiency) as avg_efficiency,
                SUM(events_in_hour) as total_events
            FROM public.trips
            WHERE ($1 = 'All' OR region = $1)
                AND date >= $2
                AND date <= $3
        """
        
        trip_result = await self.db.execute_query_one(
            trip_query,
            region,
            start_time.date(),
            end_time.date()
        )
        
        # Get active events count
        event_query = """
            SELECT COUNT(*) as active_events
            FROM public.events
            WHERE ($1 = 'All' OR state = $1)
                AND TO_TIMESTAMP("Start Time", 'MM/DD/YY HH24:MI') >= $2
                AND TO_TIMESTAMP("End Time", 'MM/DD/YY HH24:MI') <= $3
        """
        
        event_result = await self.db.execute_query_one(
            event_query,
            region,
            start_time,
            end_time
        )
        
        total_actual = float(trip_result["total_trips"] or 0)
        total_forecast = float(trip_result["total_forecast"] or 0)
        return {
            "total_trips": total_actual,
            "total_trips_forecast": total_forecast,
            "forecast_gap": total_actual - total_forecast,
            "avg_reliability": float(trip_result["avg_reliability"] or 0),
            "avg_efficiency": float(trip_result["avg_efficiency"] or 0),
            "active_events": int(event_result["active_events"] or 0)
        }
        
    async def get_correlated_data(
        self,
        region: str,
        start_time: datetime,
        end_time: datetime,
        road: Optional[str] = None,
        county: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get correlated events, weather, and trip data with joins.
        
        Args:
            region: "District of Columbia", "Virginia", or "All"
            start_time: Start of analysis period
            end_time: End of analysis period
            road: Optional road filter
            county: Optional county filter
            
        Returns:
            List of correlated records
        """
        query = """
            SELECT 
                e.office_event_id_tag,
                e.event_details_level_1,
                e.event_details_level_2,
                e.road,
                e.county,
                e.state,
                e.start_time,
                e.end_time,
                e.overall_event_time,
                e.latitude,
                e.longitude,
                w.temp,
                w.precip,
                w.preciptype,
                w.windspeed,
                w.visibility,
                w.conditions,
                t.total_car_travel_actual,
                t.network_efficiency,
                t.network_reliability,
                t.events_in_hour
            FROM public.events e
            LEFT JOIN public.weather w 
                ON DATE_TRUNC('hour', TO_TIMESTAMP(e."Start Time", 'MM/DD/YY HH24:MI')) = w.datetime
            LEFT JOIN public.trips t 
                ON DATE(TO_TIMESTAMP(e."Start Time", 'MM/DD/YY HH24:MI')) = t.date 
                AND EXTRACT(HOUR FROM TO_TIMESTAMP(e."Start Time", 'MM/DD/YY HH24:MI')) = t.hour
                AND (e.state = t.region OR $1 = 'All')
            WHERE ($1 = 'All' OR e.state = $1)
                AND TO_TIMESTAMP(e."Start Time", 'MM/DD/YY HH24:MI') >= $2
                AND TO_TIMESTAMP(e."End Time", 'MM/DD/YY HH24:MI') <= $3
        """
        
        params = [region, start_time, end_time]
        param_idx = 4
        
        if road:
            query += f" AND e.road ILIKE ${param_idx}"
            params.append(f"%{road}%")
            param_idx += 1
            
        if county:
            query += f" AND e.county ILIKE ${param_idx}"
            params.append(f"%{county}%")
            param_idx += 1
            
        query += ' ORDER BY TO_TIMESTAMP(e."Start Time", \'MM/DD/YY HH24:MI\') DESC LIMIT 500'
        
        records = await self.db.execute_query(query, *params)
        return self.db.records_to_dicts(records)
        
    async def find_event_by_location(
        self,
        road: str = None,
        county: str = None,
        start_time: datetime = None,
        end_time: datetime = None
    ) -> List[Dict[str, Any]]:
        """
        Find events by location (road/county) regardless of region filter.
        Used when user asks about specific locations in "All" region mode.
        
        Args:
            road: Road name to search for
            county: County name to search for
            start_time: Optional start time
            end_time: Optional end time
            
        Returns:
            List of matching events with their state/region
        """
        query = """
            SELECT 
                office_event_id_tag,
                system,
                event_details_level_1,
                event_details_level_2,
                road,
                direction,
                county,
                state,
                "Start Time" as start_time,
                "End Time" as end_time,
                overall_event_time,
                latitude,
                longitude
            FROM public.events
            WHERE 1=1
        """
        
        params = []
        param_idx = 1
        
        if road:
            query += f" AND road ILIKE ${param_idx}"
            params.append(f"%{road}%")
            param_idx += 1
            
        if county:
            query += f" AND county ILIKE ${param_idx}"
            params.append(f"%{county}%")
            param_idx += 1
            
        if start_time:
            query += f' AND TO_TIMESTAMP("Start Time", \'MM/DD/YY HH24:MI\') >= ${param_idx}'
            params.append(start_time)
            param_idx += 1
            
        if end_time:
            query += f' AND TO_TIMESTAMP("End Time", \'MM/DD/YY HH24:MI\') <= ${param_idx}'
            params.append(end_time)
            param_idx += 1
            
        query += " ORDER BY TO_TIMESTAMP(\"Start Time\", 'MM/DD/YY HH24:MI') DESC LIMIT 100"
        
        records = await self.db.execute_query(query, *params)
        return self.db.records_to_dicts(records)
    
    # ===== NEWS CACHE METHODS =====
    
    async def cache_news_article(
        self,
        title: str,
        url: str,
        summary: str,
        published_date: Optional[datetime],
        source: str,
        category: str,
        keywords: List[str],
        content: Optional[str] = None,
        region: str = "All"
    ) -> bool:
        """
        Cache a news article in PostgreSQL.
        Returns True if successfully cached, False if already exists.
        """
        try:
            query = """
                INSERT INTO public.news_cache 
                (title, url, summary, published_date, source, category, keywords, content, region)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (url) DO UPDATE SET
                    cached_at = NOW()
                RETURNING id
            """
            
            result = await self.db.execute_query(
                query, title, url, summary, published_date, source, 
                category, keywords, content, region
            )
            return len(result) > 0
        except Exception as e:
            logger.error(f"Error caching news article: {e}")
            return False
    
    async def get_cached_news(
        self,
        category: Optional[str] = None,
        region: Optional[str] = None,
        max_age_hours: int = 24,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get cached news articles.
        
        Args:
            category: 'traffic' or 'political_economy', None for all
            region: Region filter
            max_age_hours: Maximum age of cached articles in hours
            limit: Maximum number of articles to return
        """
        query = """
            SELECT 
                id, title, url, summary, published_date, source, 
                category, keywords, content, cached_at, region
            FROM public.news_cache
            WHERE cached_at > NOW() - INTERVAL '1 hour' * $1
        """
        
        params = [max_age_hours]
        param_idx = 2
        
        if category:
            query += f" AND category = ${param_idx}"
            params.append(category)
            param_idx += 1
        
        if region and region != "All":
            query += f" AND (region = ${param_idx} OR region = 'All')"
            params.append(region)
            param_idx += 1
        
        query += f" ORDER BY published_date DESC NULLS LAST, cached_at DESC LIMIT ${param_idx}"
        params.append(limit)
        
        try:
            records = await self.db.execute_query(query, *params)
            return self.db.records_to_dicts(records)
        except Exception as e:
            logger.error(f"Error fetching cached news: {e}")
            return []
    
    async def clear_old_news_cache(self, days: int = 7) -> int:
        """
        Clear news articles older than specified days.
        Returns number of deleted articles.
        """
        query = """
            DELETE FROM public.news_cache
            WHERE cached_at < NOW() - INTERVAL '1 day' * $1
            RETURNING id
        """
        
        try:
            result = await self.db.execute_query(query, days)
            return len(result)
        except Exception as e:
            logger.error(f"Error clearing old news cache: {e}")
            return 0
