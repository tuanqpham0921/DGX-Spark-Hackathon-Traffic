"""
SQL Queries for Austin Traffic Hotspot Prediction

Provides query methods for:
- Grid sector management and lookup
- Historical pattern analysis
- Risk score persistence and retrieval
- Live incident queries
- Weather data access
- Deployment recommendations

All queries use parameterized statements to prevent SQL injection.
"""

from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.db_connection import DatabaseConnection, get_db_connection

logger = logging.getLogger(__name__)


class AustinQueries:
    """
    Query builder for Austin traffic hotspot data.

    This class provides async methods for all database operations
    related to the Austin hotspot prediction system.

    Usage:
        queries = AustinQueries()
        sectors = await queries.get_all_sectors()
        patterns = await queries.get_historical_patterns(hour=14, day_of_week=2)
    """

    def __init__(self, db: Optional[DatabaseConnection] = None):
        """
        Initialize query builder.

        Args:
            db: Optional DatabaseConnection instance. Uses global if not provided.
        """
        self.db = db or get_db_connection()

    # =========================================================================
    # GRID SECTOR QUERIES
    # =========================================================================

    async def get_road_segments(
        self,
        *,
        limit: int = 1000,
        priority_network: Optional[str] = None,
        council_district: Optional[str] = None,
        street_level: Optional[int] = None,
        min_street_level: Optional[int] = None,
        min_lanes: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch roadway segments from the austin_road_segments table.

        Args:
            limit: Maximum number of segments to return.
            priority_network: Filter by priority network (e.g., "Bicycle Priority").
            council_district: Filter by council district (string match).
            street_level: Filter by ASMP street level (1-5).
            min_lanes: Filter segments where assum_lanes_fut or exist_lanes
                       are at least this number.
        """
        conditions = []
        params: List[Any] = []

        if priority_network:
            conditions.append("priority_network ILIKE $%d" % (len(params) + 1))
            params.append(f"%{priority_network}%")

        if council_district:
            conditions.append("council_district ILIKE $%d" % (len(params) + 1))
            params.append(f"%{council_district}%")

        if street_level is not None:
            conditions.append("street_level = $%d" % (len(params) + 1))
            params.append(int(street_level))

        if min_street_level is not None:
            conditions.append("street_level >= $%d" % (len(params) + 1))
            params.append(int(min_street_level))

        if min_lanes is not None:
            conditions.append("""
                (
                    COALESCE(NULLIF(existing_lanes, ''), '0')::NUMERIC >= $%d
                    OR COALESCE(NULLIF(future_lanes, ''), '0')::NUMERIC >= $%d
                )
            """ % (len(params) + 1, len(params) + 1))
            params.append(float(min_lanes))

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)

        query = f"""
            SELECT
                id,
                segment_uid,
                segment_id,
                asmp_street_network_id,
                street_name,
                street_level,
                improvement,
                existing_lanes,
                future_lanes,
                priority_network,
                council_district,
                shape_length,
                properties,
                geometry
            FROM public.austin_road_segments
            {where_clause}
            ORDER BY asmp_street_network_id NULLS LAST
            LIMIT ${len(params)}
        """

        records = await self.db.execute_query(query, *params)
        return self._records_to_dicts(records)

    async def get_all_sectors(self) -> List[Dict[str, Any]]:
        """
        Get all 100 grid sectors with their coordinates.

        Returns:
            List of sector dicts with:
            - id: Sector ID (1-100)
            - sector_code: Code like 'A1', 'E5', 'J10'
            - row_idx, col_idx: Grid position
            - center_lat, center_lon: Center coordinates
            - north_lat, south_lat, east_lon, west_lon: Bounds
            - neighborhood_name: Optional neighborhood label
        """
        query = """
            SELECT
                id, sector_code, row_idx, col_idx,
                north_lat, south_lat, east_lon, west_lon,
                center_lat, center_lon,
                area_sq_miles, neighborhood_name
            FROM public.austin_grid_sectors
            ORDER BY id
        """

        records = await self.db.execute_query(query)
        return self._records_to_dicts(records)

    async def get_sector_by_id(self, sector_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a specific sector by ID.

        Args:
            sector_id: Sector ID (1-100)

        Returns:
            Sector dict or None if not found
        """
        query = """
            SELECT * FROM public.austin_grid_sectors
            WHERE id = $1
        """

        record = await self.db.execute_query_one(query, sector_id)
        return dict(record) if record else None

    async def get_sector_by_code(self, sector_code: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific sector by code (e.g., 'A1', 'E5').

        Args:
            sector_code: Sector code string

        Returns:
            Sector dict or None if not found
        """
        query = """
            SELECT * FROM public.austin_grid_sectors
            WHERE sector_code = $1
        """

        record = await self.db.execute_query_one(query, sector_code.upper())
        return dict(record) if record else None

    async def get_sector_for_coords(
        self,
        lat: float,
        lon: float
    ) -> Optional[Dict[str, Any]]:
        """
        Get the sector containing given coordinates.

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            Sector dict or None if outside grid
        """
        query = """
            SELECT * FROM public.austin_grid_sectors
            WHERE $1 BETWEEN south_lat AND north_lat
              AND $2 BETWEEN west_lon AND east_lon
            LIMIT 1
        """

        record = await self.db.execute_query_one(query, lat, lon)
        return dict(record) if record else None

    async def update_sector_neighborhood(
        self,
        sector_id: int,
        neighborhood_name: str
    ) -> bool:
        """
        Update the neighborhood name for a sector.

        Args:
            sector_id: Sector ID
            neighborhood_name: Neighborhood label

        Returns:
            True if updated successfully
        """
        query = """
            UPDATE public.austin_grid_sectors
            SET neighborhood_name = $2
            WHERE id = $1
        """

        try:
            await self.db.execute_query(query, sector_id, neighborhood_name)
            return True
        except Exception as e:
            logger.error(f"Failed to update sector neighborhood: {e}")
            return False

    # =========================================================================
    # HISTORICAL PATTERN QUERIES
    # =========================================================================

    async def get_historical_patterns(
        self,
        hour: int,
        day_of_week: int
    ) -> Dict[str, Any]:
        """
        Get pre-computed historical patterns for a specific time slot.

        Used by the Hotspot Agent to determine base risk levels
        for each sector based on historical incident data.

        Args:
            hour: Hour of day (0-23)
            day_of_week: Day of week (0=Monday, 6=Sunday)

        Returns:
            Dict mapping sector_id (as string) to pattern data:
            {
                "1": {"avg_incidents": 2.5, "incident_count": 50, ...},
                "2": {"avg_incidents": 1.2, ...},
                ...
            }
        """
        query = """
            SELECT
                sector_id,
                avg_incidents,
                incident_count,
                avg_severity,
                incident_type_distribution,
                sample_size,
                first_incident_date,
                last_incident_date
            FROM public.austin_historical_patterns
            WHERE day_of_week = $1 AND hour_of_day = $2
        """

        records = await self.db.execute_query(query, day_of_week, hour)

        patterns = {}
        for record in records or []:
            sector_id = str(record["sector_id"])
            patterns[sector_id] = {
                "avg_incidents": float(record["avg_incidents"] or 0),
                "incident_count": record["incident_count"] or 0,
                "avg_severity": float(record["avg_severity"] or 0),
                "sample_size": record["sample_size"] or 0,
                "incident_types": record["incident_type_distribution"] or {},
                "first_date": record["first_incident_date"],
                "last_date": record["last_incident_date"]
            }

        logger.debug(f"Retrieved patterns for {len(patterns)} sectors "
                    f"(hour={hour}, dow={day_of_week})")
        return patterns

    async def get_all_historical_patterns(self) -> List[Dict[str, Any]]:
        """
        Get all historical patterns (for analysis/export).

        Returns:
            List of all pattern records
        """
        query = """
            SELECT
                hp.*,
                gs.sector_code
            FROM public.austin_historical_patterns hp
            JOIN public.austin_grid_sectors gs ON hp.sector_id = gs.id
            ORDER BY hp.sector_id, hp.day_of_week, hp.hour_of_day
        """

        records = await self.db.execute_query(query)
        return self._records_to_dicts(records)

    async def compute_historical_patterns(self) -> int:
        """
        Compute and cache historical patterns from incident data.

        This aggregates all incidents by sector, day of week, and hour
        to create the pattern cache used for predictions.

        Should be run periodically (e.g., daily) to update patterns
        with new incident data.

        Returns:
            Number of pattern records created/updated
        """
        query = """
            INSERT INTO public.austin_historical_patterns
            (sector_id, day_of_week, hour_of_day, avg_incidents, incident_count,
             incident_type_distribution, sample_size, first_incident_date, last_incident_date)
            SELECT
                grid_sector_id as sector_id,
                day_of_week,
                hour_of_day,
                COUNT(*)::float / GREATEST(COUNT(DISTINCT DATE(published_date)), 1) as avg_incidents,
                COUNT(*) as incident_count,
                jsonb_object_agg(
                    COALESCE(issue_reported, 'UNKNOWN'),
                    type_count
                ) as incident_type_distribution,
                COUNT(DISTINCT DATE(published_date)) as sample_size,
                MIN(DATE(published_date)) as first_incident_date,
                MAX(DATE(published_date)) as last_incident_date
            FROM (
                SELECT
                    grid_sector_id,
                    day_of_week,
                    hour_of_day,
                    published_date,
                    issue_reported,
                    COUNT(*) as type_count
                FROM public.austin_incidents
                WHERE grid_sector_id IS NOT NULL
                  AND grid_sector_id > 0
                GROUP BY grid_sector_id, day_of_week, hour_of_day, published_date, issue_reported
            ) sub
            GROUP BY grid_sector_id, day_of_week, hour_of_day
            ON CONFLICT (sector_id, day_of_week, hour_of_day) DO UPDATE SET
                avg_incidents = EXCLUDED.avg_incidents,
                incident_count = EXCLUDED.incident_count,
                incident_type_distribution = EXCLUDED.incident_type_distribution,
                sample_size = EXCLUDED.sample_size,
                first_incident_date = EXCLUDED.first_incident_date,
                last_incident_date = EXCLUDED.last_incident_date,
                last_updated = NOW()
            RETURNING id
        """

        try:
            result = await self.db.execute_query(query)
            count = len(result) if result else 0
            logger.info(f"Computed {count} historical patterns")
            return count
        except Exception as e:
            logger.error(f"Failed to compute historical patterns: {e}")
            return 0

    # =========================================================================
    # LIVE INCIDENT QUERIES
    # =========================================================================

    async def get_live_incidents(
        self,
        hours_back: int = 1,
        sector_id: Optional[int] = None,
        issue_type: Optional[str] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Get recent live incidents.

        Args:
            hours_back: How many hours of history to query
            sector_id: Optional filter by specific sector
            issue_type: Optional filter by issue type (case-insensitive partial match)
            limit: Maximum records to return

        Returns:
            List of incident dicts with sector information
        """
        query = """
            SELECT
                i.id,
                i.traffic_report_id,
                i.published_date,
                i.issue_reported,
                i.address,
                i.latitude,
                i.longitude,
                i.status,
                i.grid_sector_id,
                i.hour_of_day,
                i.day_of_week,
                s.sector_code
            FROM public.austin_incidents i
            LEFT JOIN public.austin_grid_sectors s ON i.grid_sector_id = s.id
            WHERE i.published_date >= NOW() - INTERVAL '1 hour' * $1
        """

        params: List[Any] = [hours_back]
        param_count = 1

        if sector_id:
            param_count += 1
            query += f" AND i.grid_sector_id = ${param_count}"
            params.append(sector_id)

        if issue_type:
            param_count += 1
            query += f" AND i.issue_reported ILIKE ${param_count}"
            params.append(f"%{issue_type}%")

        param_count += 1
        query += f" ORDER BY i.published_date DESC LIMIT ${param_count}"
        params.append(limit)

        records = await self.db.execute_query(query, *params)
        return self._records_to_dicts(records)

    async def get_incidents_within_hours(
        self,
        hours_back: int = 24,
        limit: int = 5000
    ) -> List[Dict[str, Any]]:
        """
        Fetch incidents within the last N hours with coordinates.
        """
        query = """
            SELECT
                id,
                traffic_report_id,
                issue_reported,
                status,
                latitude,
                longitude,
                published_date
            FROM public.austin_incidents
            WHERE published_date >= NOW() - INTERVAL '1 hour' * $1
              AND latitude IS NOT NULL
              AND longitude IS NOT NULL
            ORDER BY published_date DESC
            LIMIT $2
        """

        records = await self.db.execute_query(query, hours_back, limit)
        return self._records_to_dicts(records)

    async def get_incidents_since_days(
        self,
        days_back: int = 365,
        limit: int = 100000
    ) -> List[Dict[str, Any]]:
        """
        Fetch incidents within the last N days with coordinates.
        """
        query = """
            SELECT
                id,
                traffic_report_id,
                issue_reported,
                status,
                latitude,
                longitude,
                published_date
            FROM public.austin_incidents
            WHERE published_date >= NOW() - INTERVAL '1 day' * $1
              AND latitude IS NOT NULL
              AND longitude IS NOT NULL
            ORDER BY published_date DESC
            LIMIT $2
        """

        records = await self.db.execute_query(query, days_back, limit)
        return self._records_to_dicts(records)

    async def get_incident_counts_by_sector(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[int, int]:
        """
        Get incident counts per sector for a time range.

        Useful for calculating actual vs predicted comparison.

        Args:
            start_time: Start of time range
            end_time: End of time range

        Returns:
            Dict mapping sector_id to incident count
        """
        query = """
            SELECT
                grid_sector_id,
                COUNT(*) as incident_count
            FROM public.austin_incidents
            WHERE published_date >= $1
              AND published_date < $2
              AND grid_sector_id IS NOT NULL
              AND grid_sector_id > 0
            GROUP BY grid_sector_id
        """

        records = await self.db.execute_query(query, start_time, end_time)

        return {
            record["grid_sector_id"]: record["incident_count"]
            for record in records or []
        }

    async def get_sector_hotspots(
        self,
        start_time: datetime,
        end_time: datetime,
        min_incidents: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Get sectors with high incident counts (hotspots).

        Args:
            start_time: Start of analysis period
            end_time: End of analysis period
            min_incidents: Minimum incidents to qualify as hotspot

        Returns:
            List of hotspot sectors with counts and types
        """
        query = """
            SELECT
                s.id as sector_id,
                s.sector_code,
                s.center_lat,
                s.center_lon,
                s.neighborhood_name,
                COUNT(i.id) as incident_count,
                ARRAY_AGG(DISTINCT i.issue_reported) as issue_types,
                AVG(i.hour_of_day) as avg_hour
            FROM public.austin_grid_sectors s
            JOIN public.austin_incidents i ON s.id = i.grid_sector_id
            WHERE i.published_date >= $1 AND i.published_date < $2
            GROUP BY s.id, s.sector_code, s.center_lat, s.center_lon, s.neighborhood_name
            HAVING COUNT(i.id) >= $3
            ORDER BY incident_count DESC
        """

        records = await self.db.execute_query(query, start_time, end_time, min_incidents)
        return self._records_to_dicts(records)

    async def get_incident_type_distribution(
        self,
        hours_back: int = 24
    ) -> Dict[str, int]:
        """
        Get distribution of incident types.

        Args:
            hours_back: Hours of history to analyze

        Returns:
            Dict mapping issue_reported to count
        """
        query = """
            SELECT
                COALESCE(issue_reported, 'UNKNOWN') as issue_type,
                COUNT(*) as count
            FROM public.austin_incidents
            WHERE published_date >= NOW() - INTERVAL '1 hour' * $1
            GROUP BY issue_reported
            ORDER BY count DESC
        """

        records = await self.db.execute_query(query, hours_back)

        return {
            record["issue_type"]: record["count"]
            for record in records or []
        }

    # =========================================================================
    # RISK SCORE QUERIES
    # =========================================================================

    async def save_risk_predictions(
        self,
        predictions: List[Dict[str, Any]],
        target_hour: datetime,
        model_version: str = "v1.0"
    ) -> int:
        """
        Save computed risk predictions to database.

        Args:
            predictions: List of sector predictions with:
                - sector_id: int
                - risk_score: float (0-1)
                - risk_level: str
                - confidence: float (0-1)
                - weather_factor: float
                - historical_factor: float
                - time_factor: float (optional)
            target_hour: The hour being predicted for
            model_version: Version identifier for tracking

        Returns:
            Number of predictions saved
        """
        query = """
            INSERT INTO public.austin_risk_scores
            (sector_id, prediction_timestamp, target_hour, risk_score, risk_level,
             confidence, weather_factor, historical_factor, time_factor, model_version)
            VALUES ($1, NOW(), $2, $3, $4, $5, $6, $7, $8, $9)
        """

        count = 0
        for pred in predictions:
            try:
                await self.db.execute_query(
                    query,
                    pred["sector_id"],
                    target_hour,
                    pred["risk_score"],
                    pred.get("risk_level", "UNKNOWN"),
                    pred.get("confidence", 0.5),
                    pred.get("weather_factor", 1.0),
                    pred.get("historical_factor", 0.5),
                    pred.get("time_factor", 1.0),
                    model_version
                )
                count += 1
            except Exception as e:
                logger.error(f"Failed to save prediction for sector {pred.get('sector_id')}: {e}")

        logger.info(f"Saved {count} risk predictions for {target_hour}")
        return count

    async def get_latest_predictions(
        self,
        target_hour: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Get most recent predictions for all sectors.

        Args:
            target_hour: Optional specific hour to query.
                        If None, returns most recent predictions.

        Returns:
            List of prediction dicts with sector info
        """
        if target_hour:
            query = """
                SELECT
                    r.id,
                    r.sector_id,
                    r.target_hour,
                    r.risk_score,
                    r.risk_level,
                    r.confidence,
                    r.weather_factor,
                    r.historical_factor,
                    r.time_factor,
                    r.prediction_timestamp,
                    s.sector_code,
                    s.center_lat,
                    s.center_lon,
                    s.neighborhood_name
                FROM public.austin_risk_scores r
                JOIN public.austin_grid_sectors s ON r.sector_id = s.id
                WHERE r.target_hour = $1
                ORDER BY r.risk_score DESC
            """
            records = await self.db.execute_query(query, target_hour)
        else:
            # Get most recent prediction for each sector
            query = """
                SELECT DISTINCT ON (r.sector_id)
                    r.id,
                    r.sector_id,
                    r.target_hour,
                    r.risk_score,
                    r.risk_level,
                    r.confidence,
                    r.weather_factor,
                    r.historical_factor,
                    r.time_factor,
                    r.prediction_timestamp,
                    s.sector_code,
                    s.center_lat,
                    s.center_lon,
                    s.neighborhood_name
                FROM public.austin_risk_scores r
                JOIN public.austin_grid_sectors s ON r.sector_id = s.id
                ORDER BY r.sector_id, r.prediction_timestamp DESC
            """
            records = await self.db.execute_query(query)

        return self._records_to_dicts(records)

    async def get_high_risk_sectors(
        self,
        min_risk: float = 0.6,
        target_hour: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Get sectors above a risk threshold.

        Args:
            min_risk: Minimum risk score (default 0.6 = MEDIUM)
            target_hour: Optional specific hour

        Returns:
            List of high-risk sector predictions
        """
        if target_hour:
            query = """
                SELECT
                    r.*, s.sector_code, s.center_lat, s.center_lon
                FROM public.austin_risk_scores r
                JOIN public.austin_grid_sectors s ON r.sector_id = s.id
                WHERE r.risk_score >= $1 AND r.target_hour = $2
                ORDER BY r.risk_score DESC
            """
            records = await self.db.execute_query(query, min_risk, target_hour)
        else:
            query = """
                SELECT DISTINCT ON (r.sector_id)
                    r.*, s.sector_code, s.center_lat, s.center_lon
                FROM public.austin_risk_scores r
                JOIN public.austin_grid_sectors s ON r.sector_id = s.id
                WHERE r.risk_score >= $1
                ORDER BY r.sector_id, r.prediction_timestamp DESC
            """
            records = await self.db.execute_query(query, min_risk)

        return self._records_to_dicts(records)

    # =========================================================================
    # WEATHER QUERIES
    # =========================================================================

    async def get_weather_forecast(
        self,
        hours_ahead: int = 24
    ) -> List[Dict[str, Any]]:
        """
        Get weather forecast for upcoming hours.

        Args:
            hours_ahead: How many hours of forecast to retrieve

        Returns:
            List of weather forecast records ordered by datetime
        """
        query = """
            SELECT *
            FROM public.austin_weather
            WHERE datetime >= NOW()
              AND datetime <= NOW() + INTERVAL '1 hour' * $1
              AND forecast_or_actual = 'forecast'
            ORDER BY datetime
        """

        records = await self.db.execute_query(query, hours_ahead)
        return self._records_to_dicts(records)

    async def get_weather_for_hour(
        self,
        target_hour: datetime
    ) -> Optional[Dict[str, Any]]:
        """
        Get weather data for a specific hour.

        Prefers actual data over forecast if available.

        Args:
            target_hour: The hour to get weather for

        Returns:
            Weather dict or None
        """
        query = """
            SELECT *
            FROM public.austin_weather
            WHERE datetime >= $1
              AND datetime < $1 + INTERVAL '1 hour'
            ORDER BY
                CASE WHEN forecast_or_actual = 'actual' THEN 0 ELSE 1 END,
                datetime
            LIMIT 1
        """

        record = await self.db.execute_query_one(query, target_hour)
        return dict(record) if record else None

    async def get_current_weather(self) -> Optional[Dict[str, Any]]:
        """
        Get current weather (most recent forecast for current hour).

        Returns:
            Current weather dict or None
        """
        now = datetime.now().replace(minute=0, second=0, microsecond=0)
        return await self.get_weather_for_hour(now)

    # =========================================================================
    # DEPLOYMENT RECOMMENDATION QUERIES
    # =========================================================================

    async def save_deployment_recommendations(
        self,
        recommendations: List[Dict[str, Any]],
        target_start: datetime,
        target_end: datetime
    ) -> int:
        """
        Save deployment recommendations.

        Args:
            recommendations: List of recommendation dicts
            target_start: Start of target period
            target_end: End of target period

        Returns:
            Number of recommendations saved
        """
        # First, deactivate old recommendations for this period
        deactivate_query = """
            UPDATE public.austin_deployment_recommendations
            SET is_active = FALSE
            WHERE target_period_start = $1
        """
        await self.db.execute_query(deactivate_query, target_start)

        # Insert new recommendations
        insert_query = """
            INSERT INTO public.austin_deployment_recommendations
            (recommendation_timestamp, target_period_start, target_period_end,
             sector_id, priority_rank, recommended_units, risk_score, risk_level,
             rationale, weather_conditions)
            VALUES (NOW(), $1, $2, $3, $4, $5, $6, $7, $8, $9)
        """

        count = 0
        for rec in recommendations:
            try:
                await self.db.execute_query(
                    insert_query,
                    target_start,
                    target_end,
                    rec["sector_id"],
                    rec["priority_rank"],
                    rec["recommended_units"],
                    rec["risk_score"],
                    rec.get("risk_level", "HIGH"),
                    rec.get("rationale", ""),
                    rec.get("weather_conditions")
                )
                count += 1
            except Exception as e:
                logger.error(f"Failed to save recommendation: {e}")

        logger.info(f"Saved {count} deployment recommendations")
        return count

    async def get_active_recommendations(
        self,
        target_hour: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Get active deployment recommendations.

        Args:
            target_hour: Optional specific hour (defaults to current)

        Returns:
            List of active recommendations
        """
        if target_hour:
            query = """
                SELECT
                    r.*,
                    s.sector_code,
                    s.center_lat,
                    s.center_lon,
                    s.neighborhood_name
                FROM public.austin_deployment_recommendations r
                JOIN public.austin_grid_sectors s ON r.sector_id = s.id
                WHERE r.is_active = TRUE
                  AND r.target_period_start <= $1
                  AND r.target_period_end > $1
                ORDER BY r.priority_rank
            """
            records = await self.db.execute_query(query, target_hour)
        else:
            query = """
                SELECT
                    r.*,
                    s.sector_code,
                    s.center_lat,
                    s.center_lon,
                    s.neighborhood_name
                FROM public.austin_deployment_recommendations r
                JOIN public.austin_grid_sectors s ON r.sector_id = s.id
                WHERE r.is_active = TRUE
                  AND r.target_period_start <= NOW()
                  AND r.target_period_end > NOW()
                ORDER BY r.priority_rank
            """
            records = await self.db.execute_query(query)

        return self._records_to_dicts(records)

    # =========================================================================
    # ANALYTICS QUERIES
    # =========================================================================

    async def get_hourly_incident_trend(
        self,
        days_back: int = 7
    ) -> List[Dict[str, Any]]:
        """
        Get hourly incident counts for trend analysis.

        Args:
            days_back: Number of days to analyze

        Returns:
            List of hourly aggregates with counts
        """
        query = """
            SELECT
                DATE(published_date) as date,
                hour_of_day,
                COUNT(*) as incident_count,
                ARRAY_AGG(DISTINCT issue_reported) as issue_types
            FROM public.austin_incidents
            WHERE published_date >= NOW() - INTERVAL '1 day' * $1
            GROUP BY DATE(published_date), hour_of_day
            ORDER BY date, hour_of_day
        """

        records = await self.db.execute_query(query, days_back)
        return self._records_to_dicts(records)

    async def get_prediction_accuracy(
        self,
        days_back: int = 7
    ) -> Dict[str, Any]:
        """
        Calculate prediction accuracy metrics.

        Compares predictions to actual incidents.

        Args:
            days_back: Number of days to analyze

        Returns:
            Dict with accuracy metrics
        """
        query = """
            SELECT
                COUNT(*) as total_predictions,
                AVG(CASE WHEN pa.was_accurate THEN 1 ELSE 0 END) as accuracy_rate,
                AVG(pa.error_magnitude) as avg_error,
                COUNT(CASE WHEN pa.was_accurate THEN 1 END) as correct_predictions
            FROM public.austin_prediction_accuracy pa
            WHERE pa.evaluated_at >= NOW() - INTERVAL '1 day' * $1
        """

        record = await self.db.execute_query_one(query, days_back)

        if record:
            return {
                "total_predictions": record["total_predictions"] or 0,
                "accuracy_rate": float(record["accuracy_rate"] or 0),
                "avg_error": float(record["avg_error"] or 0),
                "correct_predictions": record["correct_predictions"] or 0
            }
        return {
            "total_predictions": 0,
            "accuracy_rate": 0,
            "avg_error": 0,
            "correct_predictions": 0
        }

    async def get_dashboard_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics for dashboard display.

        Returns:
            Dict with key metrics
        """
        # Get incident stats
        incident_query = """
            SELECT
                COUNT(*) as total_incidents,
                COUNT(CASE WHEN published_date >= NOW() - INTERVAL '1 hour' THEN 1 END) as last_hour,
                COUNT(CASE WHEN published_date >= NOW() - INTERVAL '24 hours' THEN 1 END) as last_24h,
                COUNT(DISTINCT grid_sector_id) as sectors_with_incidents
            FROM public.austin_incidents
            WHERE published_date >= NOW() - INTERVAL '7 days'
        """

        incident_record = await self.db.execute_query_one(incident_query)

        # Get prediction stats
        prediction_query = """
            SELECT
                COUNT(CASE WHEN risk_level IN ('CRITICAL', 'HIGH') THEN 1 END) as high_risk_sectors,
                AVG(risk_score) as avg_risk_score
            FROM (
                SELECT DISTINCT ON (sector_id) risk_level, risk_score
                FROM public.austin_risk_scores
                WHERE target_hour >= NOW()
                ORDER BY sector_id, prediction_timestamp DESC
            ) latest
        """

        prediction_record = await self.db.execute_query_one(prediction_query)

        return {
            "incidents": {
                "total_7d": incident_record["total_incidents"] if incident_record else 0,
                "last_hour": incident_record["last_hour"] if incident_record else 0,
                "last_24h": incident_record["last_24h"] if incident_record else 0,
                "active_sectors": incident_record["sectors_with_incidents"] if incident_record else 0
            },
            "predictions": {
                "high_risk_sectors": prediction_record["high_risk_sectors"] if prediction_record else 0,
                "avg_risk_score": float(prediction_record["avg_risk_score"] or 0) if prediction_record else 0
            }
        }

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _records_to_dicts(self, records) -> List[Dict[str, Any]]:
        """
        Convert asyncpg records to list of dictionaries.

        Handles Decimal and datetime conversion for JSON serialization.
        """
        if not records:
            return []

        result = []
        for record in records:
            row = {}
            for key, value in dict(record).items():
                if isinstance(value, Decimal):
                    row[key] = float(value)
                elif isinstance(value, datetime):
                    row[key] = value.isoformat()
                else:
                    row[key] = value
            result.append(row)

        return result
