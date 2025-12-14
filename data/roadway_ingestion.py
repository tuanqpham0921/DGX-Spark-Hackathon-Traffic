"""
Austin roadway network ingestion.

Loads the Austin Strategic Mobility Plan street network GeoJSON into
the database so we can score risk per roadway segment instead of grid cells.
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from db_connection import (
    get_db_connection,
    initialize_database,
    close_database,
    DatabaseConnection,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

GEOJSON_PATH = Path(__file__).resolve().parent / "raw" / "austin_street_segments.geojson"

UPSERT_SQL = """
INSERT INTO public.austin_road_segments (
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
)
VALUES (
    $1, $2, $3, $4, $5, $6,
    $7, $8, $9, $10, $11,
    $12::jsonb, $13::jsonb
)
ON CONFLICT (segment_uid) DO UPDATE SET
    segment_id = EXCLUDED.segment_id,
    asmp_street_network_id = EXCLUDED.asmp_street_network_id,
    street_name = EXCLUDED.street_name,
    street_level = EXCLUDED.street_level,
    improvement = EXCLUDED.improvement,
    existing_lanes = EXCLUDED.existing_lanes,
    future_lanes = EXCLUDED.future_lanes,
    priority_network = EXCLUDED.priority_network,
    council_district = EXCLUDED.council_district,
    shape_length = EXCLUDED.shape_length,
    properties = EXCLUDED.properties,
    geometry = EXCLUDED.geometry,
    updated_at = NOW();
"""


class AustinRoadwayIngestion:
    """Loads Austin roadway segments into the database."""

    def __init__(self, geojson_path: Path = GEOJSON_PATH):
        self.geojson_path = geojson_path
        self.db: DatabaseConnection = get_db_connection()

    async def ingest(self, limit: Optional[int] = None, batch_size: int = 500) -> None:
        """Ingest roadway segments from GeoJSON into Postgres."""
        if not self.geojson_path.exists():
            raise FileNotFoundError(f"GeoJSON file not found: {self.geojson_path}")

        logger.info("Loading roadway GeoJSON from %s", self.geojson_path)
        with self.geojson_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        features = data.get("features", [])
        if limit:
            features = features[:limit]

        logger.info("Preparing %d roadway features for ingestion", len(features))
        rows = [
            row for row in (self._prepare_row(feature) for feature in features) if row is not None
        ]

        if not rows:
            logger.warning("No valid roadway segments found to ingest.")
            return

        await initialize_database()
        inserted = 0

        try:
            for chunk in _chunked(rows, batch_size):
                await self.db.execute_many(UPSERT_SQL, chunk)
                inserted += len(chunk)
                if inserted % (batch_size * 2) == 0:
                    logger.info("Upserted %d roadway segments...", inserted)
        finally:
            await close_database()

        logger.info("Roadway ingestion complete (%d segments upserted).", inserted)

    def _prepare_row(self, feature: Dict[str, Any]) -> Optional[Tuple[Any, ...]]:
        """Convert a GeoJSON feature into a tuple for insertion."""
        geometry = feature.get("geometry")
        properties = feature.get("properties") or {}

        if not geometry or not geometry.get("coordinates"):
            return None

        segment_uid = (
            properties.get("asmp_street_network_id")
            or properties.get("segment_id")
            or properties.get("objectid")
        )

        if segment_uid is None:
            return None

        asmp_id = _to_int(properties.get("asmp_street_network_id"))

        return (
            str(segment_uid),
            properties.get("segment_id"),
            asmp_id,
            properties.get("name"),
            _to_int(properties.get("street_level")),
            properties.get("improvement"),
            properties.get("exist_lanes"),
            properties.get("assum_lanes_fut"),
            properties.get("priority_network"),
            properties.get("council_district"),
            _to_float(properties.get("shape_length")),
            json.dumps(properties),
            json.dumps(geometry),
        )


def _to_int(value: Any) -> Optional[int]:
    """Best-effort conversion to integer."""
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return None


def _to_float(value: Any) -> Optional[float]:
    """Best-effort conversion to float."""
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _chunked(seq: Sequence[Tuple[Any, ...]], size: int) -> Iterable[List[Tuple[Any, ...]]]:
    """Yield fixed-size chunks from a sequence."""
    chunk: List[Tuple[Any, ...]] = []
    for item in seq:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


async def _run(limit: Optional[int]) -> None:
    ingestion = AustinRoadwayIngestion()
    await ingestion.ingest(limit=limit)


def main():
    parser = argparse.ArgumentParser(description="Ingest Austin roadway network GeoJSON.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of features ingested (for testing).",
    )
    args = parser.parse_args()

    asyncio.run(_run(args.limit))


if __name__ == "__main__":
    main()
