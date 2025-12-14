-- Migration: 002_austin_road_segments.sql
-- Creates roadway segment storage for Austin Strategic Mobility Plan network

BEGIN;

CREATE TABLE IF NOT EXISTS public.austin_road_segments (
    id SERIAL PRIMARY KEY,
    segment_uid TEXT NOT NULL UNIQUE,
    segment_id TEXT,
    asmp_street_network_id BIGINT,
    street_name TEXT,
    street_level INTEGER,
    improvement TEXT,
    existing_lanes TEXT,
    future_lanes TEXT,
    priority_network TEXT,
    council_district TEXT,
    shape_length DOUBLE PRECISION,
    properties JSONB,
    geometry JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_austin_road_segments_asmp
    ON public.austin_road_segments (asmp_street_network_id);

CREATE INDEX IF NOT EXISTS idx_austin_road_segments_council
    ON public.austin_road_segments (council_district);

CREATE INDEX IF NOT EXISTS idx_austin_road_segments_priority
    ON public.austin_road_segments (priority_network);

CREATE INDEX IF NOT EXISTS idx_austin_road_segments_geometry_gin
    ON public.austin_road_segments
    USING gin ((geometry -> 'coordinates'));

COMMIT;
