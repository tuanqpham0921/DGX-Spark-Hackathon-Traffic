-- TRAFFIX Database Initialization Script
-- Creates the required tables for the traffic analysis system

-- Events table (traffic incidents)
CREATE TABLE IF NOT EXISTS public.events (
    id SERIAL PRIMARY KEY,
    event_id VARCHAR(100) UNIQUE,
    event_type VARCHAR(50),
    description TEXT,
    severity VARCHAR(20),
    state VARCHAR(50),
    county VARCHAR(100),
    city VARCHAR(100),
    street VARCHAR(255),
    latitude DECIMAL(10, 7),
    longitude DECIMAL(10, 7),
    start_time TIMESTAMP WITH TIME ZONE,
    end_time TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Weather table
CREATE TABLE IF NOT EXISTS public.weather (
    id SERIAL PRIMARY KEY,
    state VARCHAR(50),
    city VARCHAR(100),
    observation_time TIMESTAMP WITH TIME ZONE,
    temperature_f DECIMAL(5, 2),
    humidity_pct DECIMAL(5, 2),
    wind_speed_mph DECIMAL(5, 2),
    wind_direction VARCHAR(10),
    precipitation_in DECIMAL(5, 2),
    visibility_miles DECIMAL(5, 2),
    conditions VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Trips table (hourly traffic metrics)
CREATE TABLE IF NOT EXISTS public.trips (
    id SERIAL PRIMARY KEY,
    state VARCHAR(50),
    region VARCHAR(100),
    route_id VARCHAR(100),
    hour TIMESTAMP WITH TIME ZONE,
    trip_count INTEGER,
    avg_speed_mph DECIMAL(6, 2),
    avg_travel_time_min DECIMAL(8, 2),
    reliability_index DECIMAL(5, 3),
    efficiency_score DECIMAL(5, 3),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_events_state ON public.events(state);
CREATE INDEX IF NOT EXISTS idx_events_start_time ON public.events(start_time);
CREATE INDEX IF NOT EXISTS idx_events_event_type ON public.events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_state_time ON public.events(state, start_time);

CREATE INDEX IF NOT EXISTS idx_weather_state ON public.weather(state);
CREATE INDEX IF NOT EXISTS idx_weather_observation_time ON public.weather(observation_time);

CREATE INDEX IF NOT EXISTS idx_trips_state ON public.trips(state);
CREATE INDEX IF NOT EXISTS idx_trips_hour ON public.trips(hour);
CREATE INDEX IF NOT EXISTS idx_trips_state_hour ON public.trips(state, hour);

-- ============================================================================
-- AUSTIN HOTSPOT PREDICTION TABLES
-- ============================================================================

-- Austin Grid Sectors (10x10 = 100 sectors)
CREATE TABLE IF NOT EXISTS public.austin_grid_sectors (
    id INTEGER PRIMARY KEY,
    sector_code VARCHAR(3) NOT NULL,
    row_idx INTEGER NOT NULL,
    col_idx INTEGER NOT NULL,
    north_lat DECIMAL(10, 7) NOT NULL,
    south_lat DECIMAL(10, 7) NOT NULL,
    east_lon DECIMAL(10, 7) NOT NULL,
    west_lon DECIMAL(10, 7) NOT NULL,
    center_lat DECIMAL(10, 7) NOT NULL,
    center_lon DECIMAL(10, 7) NOT NULL,
    area_sq_miles DECIMAL(6, 2),
    neighborhood_name VARCHAR(100)
);

-- Austin Traffic Incidents
CREATE TABLE IF NOT EXISTS public.austin_incidents (
    id SERIAL PRIMARY KEY,
    traffic_report_id VARCHAR(100) UNIQUE,
    published_date TIMESTAMP WITH TIME ZONE,
    issue_reported VARCHAR(255),
    address TEXT,
    latitude DECIMAL(10, 7),
    longitude DECIMAL(10, 7),
    status VARCHAR(50),
    hour_of_day INTEGER,
    day_of_week INTEGER,
    grid_sector_id INTEGER REFERENCES public.austin_grid_sectors(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Austin Weather Data
CREATE TABLE IF NOT EXISTS public.austin_weather (
    id SERIAL PRIMARY KEY,
    datetime TIMESTAMP WITH TIME ZONE NOT NULL,
    temp INTEGER,
    wind_speed DECIMAL(5, 2),
    wind_direction INTEGER,
    conditions VARCHAR(255),
    precip_probability INTEGER,
    humidity INTEGER,
    forecast_or_actual VARCHAR(20) DEFAULT 'forecast',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(datetime, forecast_or_actual)
);

-- Austin Historical Patterns (pre-computed aggregates)
CREATE TABLE IF NOT EXISTS public.austin_historical_patterns (
    id SERIAL PRIMARY KEY,
    sector_id INTEGER REFERENCES public.austin_grid_sectors(id),
    day_of_week INTEGER NOT NULL,
    hour_of_day INTEGER NOT NULL,
    avg_incidents DECIMAL(6, 3),
    incident_count INTEGER,
    avg_severity DECIMAL(4, 2),
    incident_type_distribution JSONB,
    sample_size INTEGER,
    first_incident_date DATE,
    last_incident_date DATE,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(sector_id, day_of_week, hour_of_day)
);

-- Austin Risk Scores (predictions)
CREATE TABLE IF NOT EXISTS public.austin_risk_scores (
    id SERIAL PRIMARY KEY,
    sector_id INTEGER REFERENCES public.austin_grid_sectors(id),
    prediction_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    target_hour TIMESTAMP WITH TIME ZONE NOT NULL,
    risk_score DECIMAL(4, 3),
    risk_level VARCHAR(20),
    confidence DECIMAL(4, 3),
    weather_factor DECIMAL(4, 3),
    historical_factor DECIMAL(4, 3),
    time_factor DECIMAL(4, 3),
    model_version VARCHAR(20)
);

-- Austin Deployment Recommendations
CREATE TABLE IF NOT EXISTS public.austin_deployment_recommendations (
    id SERIAL PRIMARY KEY,
    recommendation_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    target_period_start TIMESTAMP WITH TIME ZONE,
    target_period_end TIMESTAMP WITH TIME ZONE,
    sector_id INTEGER REFERENCES public.austin_grid_sectors(id),
    priority_rank INTEGER,
    recommended_units INTEGER,
    risk_score DECIMAL(4, 3),
    risk_level VARCHAR(20),
    rationale TEXT,
    weather_conditions JSONB,
    is_active BOOLEAN DEFAULT TRUE
);

-- Austin Prediction Accuracy (for model evaluation)
CREATE TABLE IF NOT EXISTS public.austin_prediction_accuracy (
    id SERIAL PRIMARY KEY,
    prediction_id INTEGER,
    sector_id INTEGER,
    target_hour TIMESTAMP WITH TIME ZONE,
    predicted_risk DECIMAL(4, 3),
    actual_incidents INTEGER,
    was_accurate BOOLEAN,
    error_magnitude DECIMAL(4, 3),
    evaluated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for Austin tables
CREATE INDEX IF NOT EXISTS idx_austin_incidents_published ON public.austin_incidents(published_date);
CREATE INDEX IF NOT EXISTS idx_austin_incidents_sector ON public.austin_incidents(grid_sector_id);
CREATE INDEX IF NOT EXISTS idx_austin_incidents_hour ON public.austin_incidents(hour_of_day);
CREATE INDEX IF NOT EXISTS idx_austin_weather_datetime ON public.austin_weather(datetime);
CREATE INDEX IF NOT EXISTS idx_austin_risk_target ON public.austin_risk_scores(target_hour);
CREATE INDEX IF NOT EXISTS idx_austin_risk_sector ON public.austin_risk_scores(sector_id);

-- Populate grid sectors (10x10 grid)
INSERT INTO public.austin_grid_sectors (id, sector_code, row_idx, col_idx, north_lat, south_lat, east_lon, west_lon, center_lat, center_lon, area_sq_miles)
SELECT
    row_num * 10 + col_num + 1 as id,
    chr(65 + row_num) || (col_num + 1)::text as sector_code,
    row_num as row_idx,
    col_num as col_idx,
    30.5167 - (row_num * 0.0433) as north_lat,
    30.5167 - ((row_num + 1) * 0.0433) as south_lat,
    -97.9167 + ((col_num + 1) * 0.0333) as east_lon,
    -97.9167 + (col_num * 0.0333) as west_lon,
    30.5167 - (row_num * 0.0433) - 0.02165 as center_lat,
    -97.9167 + (col_num * 0.0333) + 0.01665 as center_lon,
    3.0 as area_sq_miles
FROM generate_series(0, 9) as row_num,
     generate_series(0, 9) as col_num
ON CONFLICT (id) DO NOTHING;

INSERT INTO austin_incident_segments (incident_id, segment_id, distance_m)
SELECT i.id,
       s.id,
       ST_DistanceSphere(i.point_geom, ST_ClosestPoint(s.geometry, i.point_geom)) AS distance_m
FROM austin_incidents i
JOIN austin_road_segments s
  ON ST_DWithin(i.point_geom, s.geometry, 0.0005)   -- ~55m
QUALIFY ROW_NUMBER() OVER (PARTITION BY i.id ORDER BY ST_Distance(i.point_geom, s.geometry)) = 1
ON CONFLICT (incident_id) DO UPDATE
SET segment_id = EXCLUDED.segment_id,
    distance_m = EXCLUDED.distance_m;