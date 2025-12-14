"""
Austin Traffic Data Ingestion Pipeline

Handles fetching and storing data from:
1. Austin Open Data Portal (Real-Time Traffic Incidents)
2. NOAA Weather API (Forecasts and Current Conditions)

API Sources:
- Traffic: https://data.austintexas.gov/resource/dx9v-zd7x.json
- Weather: https://api.weather.gov/gridpoints/EWX/143,91/forecast/hourly
"""

import aiohttp
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.db_connection import DatabaseConnection, get_db_connection
from utils.config_loader import get_config

logger = logging.getLogger(__name__)


class AustinDataIngestion:
    """
    Ingest traffic and weather data for Austin hotspot prediction.

    This class handles:
    - Fetching live traffic incidents from Austin Open Data API
    - Fetching weather forecasts from NOAA Weather API
    - Transforming data to match our database schema
    - Calculating grid sector assignments
    - Upserting data to PostgreSQL

    Usage:
        ingestion = AustinDataIngestion()
        results = await ingestion.run_full_ingestion()
        print(f"Ingested {results['incidents']} incidents, {results['weather']} weather records")
    """

    def __init__(self, db: Optional[DatabaseConnection] = None):
        """
        Initialize the ingestion pipeline.

        Args:
            db: Optional DatabaseConnection instance. If not provided,
                uses the global connection from get_db_connection().
        """
        self.db = db or get_db_connection()
        self.config = get_config()
        self.austin_config = self.config.get("austin", {})
        self.grid_config = self.austin_config.get("grid", {})

    # =========================================================================
    # AUSTIN TRAFFIC INCIDENTS
    # =========================================================================

    async def fetch_live_incidents(
        self,
        limit: int = 1000,
        offset: int = 0,
        since_hours: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch live incidents from Austin Open Data API.

        API Endpoint: https://data.austintexas.gov/resource/dx9v-zd7x.json

        The API uses Socrata Open Data API (SODA) which supports:
        - $limit: Max records per request (default 1000, max 50000)
        - $offset: Pagination offset
        - $order: Sort order
        - $where: SoQL filter clause

        Args:
            limit: Maximum records to fetch (default 1000)
            offset: Pagination offset for fetching more records
            since_hours: Only fetch incidents from last N hours (optional)

        Returns:
            List of transformed incident records ready for database insertion
        """
        base_url = self.austin_config.get("data_sources", {}).get(
            "traffic_incidents", {}
        ).get("url", "https://data.austintexas.gov/resource/dx9v-zd7x.json")

        params = {
            "$limit": limit,
            "$offset": offset,
            "$order": "published_date DESC"
        }

        # Add time filter if specified
        if since_hours:
            since_time = datetime.utcnow() - timedelta(hours=since_hours)
            params["$where"] = f"published_date >= '{since_time.isoformat()}'"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"Fetched {len(data)} incidents from Austin API")
                        return self._transform_incidents(data)
                    else:
                        error_text = await response.text()
                        logger.error(f"Austin API error {response.status}: {error_text[:200]}")
                        return []
        except aiohttp.ClientError as e:
            logger.error(f"Network error fetching Austin incidents: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error fetching Austin incidents: {e}")
            return []

    async def fetch_all_incidents(
        self,
        max_records: int = 10000,
        batch_size: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Fetch all available incidents using pagination.

        Args:
            max_records: Maximum total records to fetch
            batch_size: Records per API request

        Returns:
            Combined list of all fetched incidents
        """
        all_incidents = []
        offset = 0

        while len(all_incidents) < max_records:
            batch = await self.fetch_live_incidents(
                limit=batch_size,
                offset=offset
            )

            if not batch:
                break  # No more records

            all_incidents.extend(batch)
            offset += batch_size

            logger.info(f"Fetched {len(all_incidents)} total incidents so far...")

            # Small delay to be nice to the API
            await asyncio.sleep(0.5)

        logger.info(f"Completed fetching {len(all_incidents)} total incidents")
        return all_incidents

    def _transform_incidents(
        self,
        raw_incidents: List[Dict]
    ) -> List[Dict[str, Any]]:
        """
        Transform raw API data to our database schema.

        Performs the following transformations:
        - Parses and validates coordinates
        - Parses datetime strings
        - Calculates grid sector ID from coordinates
        - Extracts hour_of_day and day_of_week for pattern analysis

        Args:
            raw_incidents: List of raw incident records from API

        Returns:
            List of transformed records matching our schema
        """
        transformed = []

        for inc in raw_incidents:
            try:
                # Parse coordinates
                lat = self._parse_float(inc.get("latitude"))
                lon = self._parse_float(inc.get("longitude"))

                # Skip incidents without valid coordinates
                if lat is None or lon is None:
                    continue

                # Skip incidents outside Austin bounds
                if not self._is_valid_austin_coords(lat, lon):
                    logger.debug(f"Skipping incident outside Austin: ({lat}, {lon})")
                    continue

                # Parse timestamp
                published_str = inc.get("published_date", "")
                if not published_str:
                    continue

                published = self._parse_datetime(published_str)
                if not published:
                    logger.debug(f"Could not parse date: {published_str}")
                    continue

                # Build transformed record
                transformed.append({
                    "traffic_report_id": inc.get("traffic_report_id"),
                    "published_date": published,
                    "issue_reported": inc.get("issue_reported"),
                    "address": inc.get("address"),
                    "latitude": lat,
                    "longitude": lon,
                    "status": inc.get("status", "ACTIVE"),
                    "hour_of_day": published.hour,
                    "day_of_week": published.weekday(),
                    "grid_sector_id": self._calculate_grid_sector(lat, lon)
                })

            except Exception as e:
                logger.warning(f"Skipping invalid incident: {e}")
                continue

        logger.info(f"Transformed {len(transformed)}/{len(raw_incidents)} valid incidents")
        return transformed

    def _parse_float(self, value: Any) -> Optional[float]:
        """Safely parse a float value."""
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    def _is_valid_austin_coords(self, lat: float, lon: float) -> bool:
        """
        Check if coordinates are within Austin bounds.

        Uses the grid bounds from configuration with a small buffer.
        """
        bounds = self.grid_config.get("bounds", {})
        buffer = 0.05  # ~3 mile buffer

        north = bounds.get("north", 30.5167) + buffer
        south = bounds.get("south", 30.0833) - buffer
        east = bounds.get("east", -97.5833) + buffer
        west = bounds.get("west", -97.9167) - buffer

        return south <= lat <= north and west <= lon <= east

    def _parse_datetime(self, dt_str: str) -> Optional[datetime]:
        """
        Parse datetime from various API formats.

        The Austin API may return dates in several formats:
        - ISO 8601: 2025-01-15T14:30:00.000
        - ISO 8601 with timezone: 2025-01-15T14:30:00.000Z
        - Simple: 2025-01-15 14:30:00
        """
        if not dt_str:
            return None

        # Clean the string
        dt_str = dt_str.replace("Z", "").replace("T", " ")

        # Remove timezone offset if present
        if "+" in dt_str:
            dt_str = dt_str.split("+")[0]
        if "-" in dt_str and dt_str.count("-") > 2:
            # Has timezone like -06:00
            dt_str = dt_str.rsplit("-", 1)[0]

        # Remove microseconds for cleaner parsing
        if "." in dt_str:
            dt_str = dt_str.split(".")[0]

        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d",
        ]

        for fmt in formats:
            try:
                return datetime.strptime(dt_str.strip(), fmt)
            except ValueError:
                continue

        logger.warning(f"Could not parse datetime: {dt_str}")
        return None

    def _calculate_grid_sector(self, lat: float, lon: float) -> int:
        """
        Calculate which grid sector a coordinate falls into.

        The grid is 10x10 (100 sectors) covering Austin.
        Sectors are numbered 1-100, with codes A1-J10.

        Grid layout (row letter + column number):
            1    2    3    4    5    6    7    8    9    10
        A   1    2    3    4    5    6    7    8    9    10
        B   11   12   13   14   15   16   17   18   19   20
        ...
        J   91   92   93   94   95   96   97   98   99   100

        Args:
            lat: Latitude coordinate
            lon: Longitude coordinate

        Returns:
            Sector ID (1-100), or 0 if outside bounds
        """
        bounds = self.grid_config.get("bounds", {})
        rows = self.grid_config.get("rows", 10)
        cols = self.grid_config.get("cols", 10)

        north = bounds.get("north", 30.5167)
        south = bounds.get("south", 30.0833)
        east = bounds.get("east", -97.5833)
        west = bounds.get("west", -97.9167)

        # Check if within bounds
        if lat < south or lat > north or lon < west or lon > east:
            return 0  # Outside grid

        # Calculate row (0-9) from latitude
        # Row 0 is northernmost (A), row 9 is southernmost (J)
        lat_range = north - south
        row = int((north - lat) / lat_range * rows)

        # Calculate column (0-9) from longitude
        # Column 0 is westernmost (1), column 9 is easternmost (10)
        lon_range = east - west
        col = int((lon - west) / lon_range * cols)

        # Clamp to valid range (handles edge cases)
        row = max(0, min(rows - 1, row))
        col = max(0, min(cols - 1, col))

        # Return 1-indexed sector ID
        return row * cols + col + 1

    def get_sector_code(self, sector_id: int) -> str:
        """
        Convert sector ID to code (e.g., 1 -> 'A1', 55 -> 'F5').

        Args:
            sector_id: Sector ID (1-100)

        Returns:
            Sector code string (e.g., 'A1', 'J10')
        """
        if sector_id < 1 or sector_id > 100:
            return "XX"

        row = (sector_id - 1) // 10
        col = (sector_id - 1) % 10 + 1

        return f"{chr(65 + row)}{col}"

    async def ingest_incidents(self, incidents: List[Dict[str, Any]]) -> int:
        """
        Insert or update incidents in database.

        Uses upsert (INSERT ... ON CONFLICT UPDATE) to handle
        duplicate traffic_report_ids gracefully.

        Args:
            incidents: List of transformed incident records

        Returns:
            Number of records successfully processed
        """
        if not incidents:
            return 0

        query = """
            INSERT INTO public.austin_incidents
            (traffic_report_id, published_date, issue_reported, address,
             latitude, longitude, status, hour_of_day, day_of_week, grid_sector_id)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            ON CONFLICT (traffic_report_id) DO UPDATE SET
                status = EXCLUDED.status,
                created_at = NOW()
        """

        count = 0
        errors = 0

        for inc in incidents:
            try:
                await self.db.execute_query(
                    query,
                    inc["traffic_report_id"],
                    inc["published_date"],
                    inc["issue_reported"],
                    inc["address"],
                    inc["latitude"],
                    inc["longitude"],
                    inc["status"],
                    inc["hour_of_day"],
                    inc["day_of_week"],
                    inc["grid_sector_id"]
                )
                count += 1
            except Exception as e:
                errors += 1
                if errors <= 5:  # Log first 5 errors
                    logger.error(f"Failed to insert incident {inc.get('traffic_report_id')}: {e}")

        if errors > 5:
            logger.error(f"... and {errors - 5} more insertion errors")

        logger.info(f"Ingested {count} Austin incidents ({errors} errors)")
        return count

    # =========================================================================
    # NOAA WEATHER DATA
    # =========================================================================

    async def fetch_noaa_weather(self) -> Dict[str, Any]:
        """
        Fetch weather forecast from NOAA Weather API.

        Uses the Austin grid point (EWX/143,91) for localized forecasts.
        NOAA requires a User-Agent header identifying the application.

        Returns:
            Dict with 'periods' list containing hourly forecasts
        """
        base_url = self.austin_config.get("data_sources", {}).get(
            "noaa_weather", {}
        ).get("base_url", "https://api.weather.gov")

        grid_point = self.austin_config.get("data_sources", {}).get(
            "noaa_weather", {}
        ).get("grid_point", "EWX/143,91")

        # NOAA requires User-Agent header
        headers = {
            "User-Agent": "TRAFFIX-Austin/1.0 (traffic-hotspot-prediction; contact@example.com)",
            "Accept": "application/geo+json"
        }

        try:
            async with aiohttp.ClientSession(headers=headers) as session:
                # Get hourly forecast
                forecast_url = f"{base_url}/gridpoints/{grid_point}/forecast/hourly"

                async with session.get(forecast_url) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info("Fetched NOAA weather forecast successfully")
                        return self._transform_weather(data)
                    else:
                        error_text = await response.text()
                        logger.error(f"NOAA API error {response.status}: {error_text[:200]}")
                        return {"periods": []}

        except aiohttp.ClientError as e:
            logger.error(f"Network error fetching NOAA weather: {e}")
            return {"periods": []}
        except Exception as e:
            logger.error(f"Unexpected error fetching NOAA weather: {e}")
            return {"periods": []}

    def _transform_weather(self, raw_data: Dict) -> Dict[str, Any]:
        """
        Transform NOAA weather data to our schema.

        NOAA returns GeoJSON with properties containing forecast periods.
        Each period has temperature, wind, and conditions information.

        Args:
            raw_data: Raw NOAA API response

        Returns:
            Dict with 'periods' list of transformed weather records
        """
        periods = raw_data.get("properties", {}).get("periods", [])

        weather_records = []

        for period in periods[:72]:  # Next 72 hours (3 days)
            try:
                # Parse start time
                start_time = period.get("startTime", "")
                if not start_time:
                    continue

                dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))

                # Parse wind speed (format: "10 mph" or "5 to 15 mph")
                wind_speed = self._parse_wind_speed(period.get("windSpeed", "0 mph"))

                # Parse wind direction
                wind_dir = self._parse_wind_direction(period.get("windDirection", "N"))

                # Get precipitation probability
                precip_prob = period.get("probabilityOfPrecipitation", {})
                if isinstance(precip_prob, dict):
                    precip_prob = precip_prob.get("value", 0) or 0
                else:
                    precip_prob = 0

                # Get humidity
                humidity = period.get("relativeHumidity", {})
                if isinstance(humidity, dict):
                    humidity = humidity.get("value", 50) or 50
                else:
                    humidity = 50

                weather_records.append({
                    "datetime": dt,
                    "temp": period.get("temperature"),
                    "wind_speed": wind_speed,
                    "wind_direction": wind_dir,
                    "conditions": period.get("shortForecast"),
                    "precip_probability": precip_prob,
                    "humidity": humidity,
                    "forecast_or_actual": "forecast"
                })

            except Exception as e:
                logger.warning(f"Skipping weather period: {e}")
                continue

        logger.info(f"Transformed {len(weather_records)} weather periods")
        return {"periods": weather_records}

    def _parse_wind_speed(self, wind_str: str) -> float:
        """
        Parse wind speed from NOAA format.

        Handles formats like:
        - "10 mph"
        - "5 to 15 mph" (returns higher value)
        - "10 to 20 mph"
        """
        if not wind_str:
            return 0.0

        try:
            # Handle range format "5 to 15 mph"
            if " to " in wind_str:
                parts = wind_str.split(" to ")
                return float(parts[1].split()[0])  # Take higher value

            # Handle simple format "10 mph"
            return float(wind_str.split()[0])
        except (ValueError, IndexError):
            return 0.0

    def _parse_wind_direction(self, direction: str) -> int:
        """
        Convert cardinal/intercardinal direction to degrees.

        Args:
            direction: Direction string (e.g., "N", "NE", "SSW")

        Returns:
            Degrees (0-359), with 0 = North
        """
        directions = {
            "N": 0, "NNE": 22, "NE": 45, "ENE": 67,
            "E": 90, "ESE": 112, "SE": 135, "SSE": 157,
            "S": 180, "SSW": 202, "SW": 225, "WSW": 247,
            "W": 270, "WNW": 292, "NW": 315, "NNW": 337
        }
        return directions.get(direction.upper(), 0)

    async def ingest_weather(self, weather_records: List[Dict[str, Any]]) -> int:
        """
        Insert weather records into database.

        Uses upsert to update existing forecasts for the same datetime.

        Args:
            weather_records: List of transformed weather records

        Returns:
            Number of records successfully processed
        """
        if not weather_records:
            return 0

        query = """
            INSERT INTO public.austin_weather
            (datetime, temp, wind_speed, wind_direction, conditions,
             precip_probability, humidity, forecast_or_actual)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (datetime, forecast_or_actual) DO UPDATE SET
                temp = EXCLUDED.temp,
                wind_speed = EXCLUDED.wind_speed,
                wind_direction = EXCLUDED.wind_direction,
                conditions = EXCLUDED.conditions,
                precip_probability = EXCLUDED.precip_probability,
                humidity = EXCLUDED.humidity
        """

        count = 0
        errors = 0

        for record in weather_records:
            try:
                await self.db.execute_query(
                    query,
                    record["datetime"],
                    record["temp"],
                    record["wind_speed"],
                    record.get("wind_direction", 0),
                    record["conditions"],
                    record["precip_probability"],
                    record.get("humidity", 50),
                    record["forecast_or_actual"]
                )
                count += 1
            except Exception as e:
                errors += 1
                if errors <= 3:
                    logger.error(f"Failed to insert weather record: {e}")

        logger.info(f"Ingested {count} weather records ({errors} errors)")
        return count

    # =========================================================================
    # FULL INGESTION CYCLE
    # =========================================================================

    async def run_full_ingestion(
        self,
        incident_limit: int = 5000,
        include_weather: bool = True
    ) -> Dict[str, int]:
        """
        Run complete data ingestion cycle.

        Fetches and stores:
        1. Live traffic incidents from Austin API
        2. Weather forecast from NOAA (optional)

        Args:
            incident_limit: Maximum incidents to fetch
            include_weather: Whether to fetch weather data

        Returns:
            Dict with counts of ingested records:
            {
                "incidents": int,
                "weather": int
            }
        """
        results = {
            "incidents": 0,
            "weather": 0
        }

        # Ingest live incidents
        logger.info("Starting Austin incident ingestion...")
        incidents = await self.fetch_live_incidents(limit=incident_limit)
        results["incidents"] = await self.ingest_incidents(incidents)

        # Ingest weather forecast
        if include_weather:
            logger.info("Starting NOAA weather ingestion...")
            weather = await self.fetch_noaa_weather()
            results["weather"] = await self.ingest_weather(weather.get("periods", []))

        logger.info(f"Ingestion complete: {results}")
        return results

    async def run_historical_ingestion(
        self,
        max_records: int = 50000
    ) -> Dict[str, int]:
        """
        Run full historical data ingestion.

        Fetches all available historical incidents using pagination.
        Use this for initial data loading.

        Args:
            max_records: Maximum total records to fetch

        Returns:
            Dict with ingestion counts
        """
        logger.info(f"Starting historical ingestion (max {max_records} records)...")

        incidents = await self.fetch_all_incidents(max_records=max_records)
        ingested = await self.ingest_incidents(incidents)

        results = {"incidents": ingested, "weather": 0}

        # Also fetch weather
        weather = await self.fetch_noaa_weather()
        results["weather"] = await self.ingest_weather(weather.get("periods", []))

        logger.info(f"Historical ingestion complete: {results}")
        return results


# ============================================================================
# CLI INTERFACE
# ============================================================================

async def run_ingestion():
    """Run data ingestion from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Austin Traffic Data Ingestion")
    parser.add_argument(
        "--historical",
        action="store_true",
        help="Run full historical ingestion"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5000,
        help="Maximum incidents to fetch"
    )
    parser.add_argument(
        "--no-weather",
        action="store_true",
        help="Skip weather data ingestion"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    ingestion = AustinDataIngestion()

    if args.historical:
        results = await ingestion.run_historical_ingestion(max_records=args.limit)
    else:
        results = await ingestion.run_full_ingestion(
            incident_limit=args.limit,
            include_weather=not args.no_weather
        )

    print(f"\nIngestion Results:")
    print(f"  Incidents: {results['incidents']}")
    print(f"  Weather:   {results['weather']}")

    return results


if __name__ == "__main__":
    asyncio.run(run_ingestion())
