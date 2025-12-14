"""
Austin Hotspot Dashboard - Streamlit UI for Risk Predictions

A real-time dashboard for visualizing Austin traffic incident hotspots
and deployment recommendations.

Features:
- Interactive risk heatmap with color-coded sectors
- Real-time predictions and dispatch briefings
- Deployment recommendations panel
- Weather conditions display
- Recent incidents table
- Auto-refresh capability

Usage:
    streamlit run ui/austin_hotspot_dashboard.py

Or from the project root:
    python -m streamlit run ui/austin_hotspot_dashboard.py
"""

import streamlit as st
import asyncio
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import folium
from streamlit_folium import st_folium
import requests
import sys
from pathlib import Path
from typing import Optional
import json

from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="TRAFFIX Austin - Hotspot Predictor",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgba(20, 120, 110, 0.5), transparent 40%),
                    radial-gradient(circle at 80% 25%, rgba(90, 60, 160, 0.45), transparent 45%),
                    radial-gradient(circle at 30% 80%, rgba(35, 160, 130, 0.4), transparent 50%),
                    radial-gradient(circle at 80% 75%, rgba(80, 50, 150, 0.35), transparent 45%),
                    #0f172a;
        background-size: 200% 200%;
        animation: shellGlow 25s ease-in-out infinite alternate;
    }
    @keyframes shellGlow {
        0% {
            background-position: 0% 0%, 100% 0%, 0% 100%, 100% 100%;
        }
        50% {
            background-position: 15% 10%, 85% 5%, 5% 95%, 95% 80%;
        }
        100% {
            background-position: 0% 0%, 100% 0%, 0% 100%, 100% 100%;
        }
    }
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2.5rem;
    }
    .map-card {
        background: #ffffff;
        padding: 1.2rem 1.2rem 1.5rem;
        border-radius: 22px;
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.18);
        border: 1px solid rgba(226, 232, 240, 0.8);
    }
    .info-card {
        background: #ffffff;
        padding: 1rem 1.2rem;
        border-radius: 18px;
        box-shadow: 0 10px 35px rgba(15, 23, 42, 0.12);
        border: 1px solid rgba(226, 232, 240, 0.7);
        margin-bottom: 1rem;
    }
    .info-card h4 {
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 0.2rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .info-card p {
        font-size: 1.5rem;
        font-weight: 600;
        color: #0f172a;
        margin: 0;
    }
    .top-sector-list li {
        list-style: none;
        padding: 0.35rem 0;
        border-bottom: 1px dashed rgba(148, 163, 184, 0.4);
    }
    .top-sector-list li:last-child {
        border-bottom: none;
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        color: #0f172a;
        text-transform: uppercase;
        margin-bottom: 0.1rem;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: #64748b;
        letter-spacing: 0.2em;
        text-transform: uppercase;
    }
    .hero-wrapper {
        text-align: center;
        margin-bottom: 0.85rem;
    }
    .map-meta {
        text-align: center;
        font-size: 0.95rem;
        color: #475569;
        margin-top: -0.35rem;
        margin-bottom: 0.75rem;
    }
    .high-risk-card {
        margin: 0 auto 1.4rem;
        width: 95%;
        max-width: 1400px;
        padding: 1.5rem 1.8rem;
        background: #020617;
        border: 1px solid rgba(16, 185, 129, 0.35);
        color: #10b981;
    }
    .high-risk-card header {
        display: flex;
        flex-direction: column;
        gap: 0.25rem;
        margin-bottom: 1rem;
    }
    .high-risk-card header h4 {
        font-size: 1.1rem;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #22d3ee;
    }
    .high-risk-card header span {
        color: #34d399;
        font-size: 0.9rem;
    }
    .high-risk-card ul {
        list-style: none;
        margin: 0;
        padding: 0;
        display: flex;
        flex-direction: column;
        gap: 0.75rem;
        max-height: 420px;
        overflow-y: auto;
    }
    .high-risk-card li {
        display: flex;
        flex-direction: column;
        gap: 0.15rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px dashed rgba(148, 163, 184, 0.4);
    }
    .high-risk-card li:last-child {
        border-bottom: none;
    }
    .high-risk-card .segment-meta {
        color: #a7f3d0;
        font-size: 0.85rem;
    }
    .high-risk-metrics {
        font-size: 0.8rem;
        color: #99f6e4;
        margin-top: 1rem;
    }
    .snapshot-column, .detail-column {
        min-height: 60vh;
        display: flex;
        flex-direction: column;
        gap: 1rem;
    }
    .dispatch-card {
        background: #020617 !important;
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #10b981;
    }
    .dispatch-card h4 {
        color: #22d3ee;
    }
    .dispatch-card p {
        color: #10b981 !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# API configuration
API_URL = "http://localhost:8000"

# Risk level colors
RISK_COLORS = {
    "CRITICAL": "#dc2626",   # red-600
    "HIGH": "#f97316",       # orange-500
    "MEDIUM": "#eab308",     # yellow-500
    "LOW": "#22c55e",        # green-500
    "MINIMAL": "#94a3b8"     # slate-400
}

# Risk level icons
RISK_ICONS = {
    "CRITICAL": "🔴",
    "HIGH": "🟠",
    "MEDIUM": "🟡",
    "LOW": "🟢",
    "MINIMAL": "⚪"
}

STREET_LEVEL_COLORS = {
    1: "#10b981",
    2: "#0ea5e9",
    3: "#f59e0b",
    4: "#ef4444",
    5: "#7f1d1d"
}

MAP_TILES = {
    "Dark": {
        "tiles": "CartoDB dark_matter",
        "attr": "© OpenStreetMap contributors © CartoDB"
    },
    "Light": {
        "tiles": "CartoDB positron",
        "attr": "© OpenStreetMap contributors © CartoDB"
    },
    "Topographic": {
        "tiles": "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        "attr": "© OpenStreetMap contributors © OpenTopoMap"
    }
}


def get_predictions():
    """Fetch predictions from API."""
    try:
        response = requests.post(
            f"{API_URL}/api/austin/predict",
            json={"include_weather": True},
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API server. Make sure the backend is running on localhost:8000")
        return None
    except Exception as e:
        st.error(f"Error fetching predictions: {e}")
        return None

# NOTE (TQP): This function is not currently used in the dashboard
def get_grid():
    """Fetch grid sectors from API."""
    try:
        response = requests.get(f"{API_URL}/api/austin/grid", timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def get_live_incidents(hours_back: int = 1):
    """Fetch live incidents from API."""
    try:
        response = requests.get(
            f"{API_URL}/api/austin/live-incidents",
            params={"hours_back": hours_back},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def get_incident_types(hours_back: int = 24):
    """Fetch incident type distribution from API."""
    try:
        response = requests.get(
            f"{API_URL}/api/austin/incident-types",
            params={"hours_back": hours_back},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def get_road_segments(
    limit: int = 500,
    priority_network: Optional[str] = None,
    street_level: Optional[int] = None,
    min_street_level: Optional[int] = None
):
    """Fetch roadway segments from API."""
    try:
        params = {"limit": limit}
        if priority_network:
            params["priority_network"] = priority_network
        if street_level is not None:
            params["street_level"] = street_level
        if min_street_level is not None:
            params["min_street_level"] = min_street_level

        response = requests.get(
            f"{API_URL}/api/austin/road-segments",
            params=params,
            timeout=15
        )
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def get_risk_color(level: str) -> str:
    """Get color for risk level."""
    return RISK_COLORS.get(level, RISK_COLORS["MINIMAL"])


def get_risk_radius(score: float) -> int:
    """Calculate marker radius based on risk score."""
    # Scale from 15 (min) to 35 (max) based on risk score
    return int(15 + score * 20)


def create_risk_heatmap(
    predictions: list,
    grid_metadata: Optional[dict] = None,
    segments: Optional[list] = None,
    map_style: str = "Dark"
):
    """Create Folium map with risk heatmap overlay."""
    # Austin center coordinates
    bounds = (grid_metadata or {}).get("bounds", {})
    center_lat = bounds.get("center_lat")
    center_lon = bounds.get("center_lon")

    if center_lat is None or center_lon is None:
        north = bounds.get("north")
        south = bounds.get("south")
        east = bounds.get("east")
        west = bounds.get("west")

        if north is not None and south is not None:
            center_lat = (north + south) / 2
        else:
            center_lat = 30.2672

        if east is not None and west is not None:
            center_lon = (east + west) / 2
        else:
            center_lon = -97.7431

    # Create base map
    tile_cfg = MAP_TILES.get(map_style, MAP_TILES["Dark"])
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles=tile_cfg["tiles"],
        attr=tile_cfg.get("attr")
    )

    # Add sector risk rectangles
    # (Grid overlay removed for roadway-centric view)

    # Overlay roadway segments
    if segments:
        for seg in segments:
            geometry = seg.get("geometry") or {}
            if isinstance(geometry, str):
                try:
                    geometry = json.loads(geometry)
                except json.JSONDecodeError:
                    geometry = {}
            coords = geometry.get("coordinates")
            if not coords:
                continue

            geo_type = geometry.get("type", "LineString")
            if geo_type == "LineString":
                line_strings = [coords]
            elif geo_type == "MultiLineString":
                line_strings = coords
            else:
                continue

            street_level = seg.get("street_level")
            risk_level = seg.get("risk_level", "MINIMAL")
            risk_score = seg.get("risk_score", 0.0)
            color = RISK_COLORS.get(risk_level, STREET_LEVEL_COLORS.get(street_level, "#6366f1"))
            name = seg.get("street_name") or seg.get("segment_uid") or "Unnamed Segment"

            popup_html = f"""
            <div style="font-family: Arial, sans-serif; min-width: 180px;">
                <b>{name}</b><br>
                <b>Street Level:</b> {street_level or 'N/A'}<br>
                <b>Priority:</b> {seg.get('priority_network', 'N/A')}<br>
                <b>Risk:</b> {risk_score:.0%} ({risk_level})<br>
                <b>Incidents (24h):</b> {seg.get('recent_incident_count', 0)}<br>
                <b>Lanes:</b> {seg.get('existing_lanes') or '-'} ➜ {seg.get('future_lanes') or '-'}
            </div>
            """

            for line in line_strings:
                latlon = [(pt[1], pt[0]) for pt in line if len(pt) >= 2]
                if len(latlon) < 2:
                    continue

                folium.PolyLine(
                    latlon,
                    color=color,
                    weight=3 if (street_level and street_level >= 3) else 2,
                    opacity=0.85,
                    tooltip=f"{name} (Level {street_level or '?'})",
                    popup=folium.Popup(popup_html, max_width=240)
                ).add_to(m)

    # Add legend for risk levels
    legend_items = "".join([
        f"<div><span style='display:inline-block;width:14px;height:4px;background:{color};margin-right:6px;'></span>{level.title()}</div>"
        for level, color in RISK_COLORS.items()
    ])
    legend_html = f"""
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000;
                background-color: white; padding: 12px; border-radius: 8px;
                border: 1px solid rgba(148,163,184,0.6); font-family: Arial; color: #000;">
        <b>Street Levels</b><br>
        {legend_items}
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    return m


def create_risk_distribution_chart(predictions: list):
    """Create a pie chart of risk level distribution."""
    level_counts = {}
    for pred in predictions:
        level = pred.get("risk_level", "MINIMAL")
        level_counts[level] = level_counts.get(level, 0) + 1

    # Order levels
    ordered_levels = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "MINIMAL"]
    labels = []
    values = []
    colors = []

    for level in ordered_levels:
        if level in level_counts:
            labels.append(level)
            values.append(level_counts[level])
            colors.append(RISK_COLORS[level])

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker_colors=colors,
        hole=0.4,
        textinfo='label+value',
        textposition='outside'
    )])

    fig.update_layout(
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20),
        height=250
    )

    return fig


def create_incident_type_chart(distribution: dict):
    """Create a bar chart of incident types."""
    if not distribution:
        return None

    # Sort by count
    sorted_items = sorted(distribution.items(), key=lambda x: x[1], reverse=True)[:10]

    types = [item[0] for item in sorted_items]
    counts = [item[1] for item in sorted_items]

    fig = go.Figure(data=[go.Bar(
        x=counts,
        y=types,
        orientation='h',
        marker_color='#3b82f6'
    )])

    fig.update_layout(
        xaxis_title="Count",
        yaxis_title="",
        margin=dict(l=20, r=20, t=20, b=40),
        height=300,
        yaxis=dict(autorange="reversed")
    )

    return fig


def main():
    """Main dashboard application."""
    # Sidebar configuration
    with st.sidebar:
        st.markdown("## Configuration")

        # Auto-refresh toggle
        auto_refresh = st.checkbox("Auto-refresh (60s)", value=False)

        # Historical incidents slider
        hours_back = st.slider(
            "Incident History (hours)",
            min_value=1,
            max_value=48,
            value=24
        )

        map_options = list(MAP_TILES.keys())
        default_index = map_options.index("Dark") if "Dark" in map_options else 0
        map_style = st.selectbox(
            "Map Style",
            map_options,
            index=default_index
        )

        # Manual refresh button
        if st.button("🔄 Refresh Predictions", type="primary", use_container_width=True):
            st.rerun()

        st.markdown("---")

        # About section
        st.markdown("### About")
        st.markdown("""
        This dashboard predicts traffic incident hotspots
        across Austin using:
        - Historical incident patterns
        - NOAA weather data
        - Time-of-day factors

        **Risk Levels:**
        - 🔴 CRITICAL: Deploy 2 units
        - 🟠 HIGH: Deploy 1 unit
        - 🟡 MEDIUM: Monitor
        - 🟢 LOW: Routine
        - ⚪ MINIMAL: Clear
        """)

    # Load predictions
    with st.spinner("Loading predictions..."):
        predictions_data = get_predictions()

    road_segments_data = get_road_segments(limit=600)
    segment_list = []
    segment_filters = {}
    segment_analysis = {}
    if road_segments_data and road_segments_data.get("success"):
        segment_list = road_segments_data.get("segments", [])
        segment_filters = road_segments_data.get("filters", {})
        segment_analysis = road_segments_data.get("analysis", {})

    if predictions_data and predictions_data.get("success"):
        predictions = predictions_data.get("predictions", [])
        recommendations = predictions_data.get("recommendations", [])
        narrative = predictions_data.get("narrative", "")
        weather = predictions_data.get("weather", {})
        summary = predictions_data.get("summary", {})
        target_hour = predictions_data.get("target_hour", "")
        grid_info = predictions_data.get("grid", {})

        map_segments = sorted(
            segment_list,
            key=lambda seg: seg.get("risk_score", 0),
            reverse=True
        )[:100] if segment_list else segment_list

        # Centered map section
        map_left, map_center, map_right = st.columns([0.02, 0.96, 0.02])
        with map_center:
            st.markdown('<div class="map-card">', unsafe_allow_html=True)
            st.markdown("""
            <div class="hero-wrapper">
                <div class="hero-title">TRAFFIX</div>
                <div class="hero-subtitle">Austin Roadway Intelligence</div>
            </div>
            """, unsafe_allow_html=True)
            if target_hour:
                try:
                    target_dt = datetime.fromisoformat(target_hour)
                    target_label = target_dt.strftime('%A, %B %d at %I:%M %p')
                except Exception:
                    target_label = target_hour
                st.markdown(
                    f"<div class='map-meta'>Forecast window: {target_label}</div>",
                    unsafe_allow_html=True
                )

            # Create and display map
            risk_map = create_risk_heatmap(predictions, grid_info, map_segments, map_style=map_style)
            st_folium(risk_map, width=None, height=720, returned_objects=[])

            # Removed legacy grid caption

            st.markdown('</div>', unsafe_allow_html=True)

            if narrative:
                st.markdown(f"""
                <div class="info-card dispatch-card">
                    <h4>📋 Dispatch Briefing</h4>
                    <p style="font-size: 0.95rem; font-weight: 500; line-height: 1.5;">
                        {narrative}
                    </p>
                </div>
                """, unsafe_allow_html=True)

        # Roadway segments card above summary section
        if segment_list:
            ranked_segments = sorted(
                segment_list,
                key=lambda seg: seg.get("risk_score", 0),
                reverse=True
            )

            def format_segment(seg: Dict[str, Any]) -> str:
                name = seg.get("street_name") or seg.get("segment_uid", "Segment")
                lvl = seg.get("street_level") or "?"
                priority = seg.get("priority_network") or "General"
                risk_level = seg.get("risk_level", "MINIMAL")
                risk_score = seg.get("risk_score", 0)
                incident_count = seg.get("recent_incident_count", 0)
                hist_avg = seg.get("historical_avg_daily_incidents", 0)
                icon = RISK_ICONS.get(risk_level, "⚪")
                return (
                    f"<li><strong>{icon} {name}</strong>"
                    f"<span class='segment-meta'>Level {lvl} · {priority} · Risk {risk_score:.0%} ({risk_level}) "
                    f"· {incident_count} recent · {hist_avg:.2f} avg/day</span></li>"
                )

            items = "".join(format_segment(seg) for seg in ranked_segments)
            limit_used = segment_filters.get("limit")
            hours_back = segment_analysis.get("hours_back")
            incident_sample = segment_analysis.get("incident_sample")
            hist_days = segment_analysis.get("historical_days")
            hist_sample = segment_analysis.get("historical_sample")

            st.markdown(f"""
            <div class="info-card high-risk-card">
                <header>
                    <h4>🛣️ High-Risk Roadway Segments</h4>
                    <span>Live priority corridors pulled from {len(segment_list)} tracked segments.</span>
                </header>
                <ul>{items}</ul>
                <div class="high-risk-metrics">
                    Recent window: {hours_back or 24}h · Incidents: {incident_sample or 0} ·
                    Historical window: {hist_days or 365}d · Incidents: {hist_sample or 0} ·
                    API limit: {limit_used or 'n/a'}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.caption("No roadway segments loaded.")

        # Summary and weather columns
        summary_col, detail_col = st.columns([1.5, 1])

        with summary_col:
            st.markdown("<div class='snapshot-column'>", unsafe_allow_html=True)
            st.markdown("### 📊 Snapshot")
            avg_risk = summary.get('avg_risk_score', 0)
            cards = [
                ("High Risk Sectors", f"{summary.get('high_risk_count', 0)}", "Needs immediate coverage"),
                ("Units to Deploy", f"{summary.get('total_units_recommended', 0)}", "Recommended tow units"),
                ("Avg Risk Score", f"{avg_risk:.1%}", "Citywide weighted average"),
                ("Medium Risk", f"{summary.get('medium_risk_count', 0)}", "Monitor closely")
            ]

            card_rows = [cards[:2], cards[2:]]
            for row in card_rows:
                row_cols = st.columns(len(row))
                for (title, value, helper), col in zip(row, row_cols):
                    with col:
                        st.markdown(f"""
                        <div class="info-card">
                            <h4>{title}</h4>
                            <p>{value}</p>
                            <span style="font-size: 0.85rem; color: #94a3b8;">{helper}</span>
                        </div>
                        """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with detail_col:
            st.markdown("<div class='detail-column'>", unsafe_allow_html=True)
            st.markdown("### 🌤️ Weather")
            if weather:
                weather_factor = predictions_data.get("weather_factor", 1.0)
                st.markdown(f"""
                <div class="info-card">
                    <div style="display: flex; gap: 1rem; flex-wrap: wrap;">
                        <div>
                            <h4>Temperature</h4>
                            <p>{weather.get('temp', '--')}°F</p>
                        </div>
                        <div>
                            <h4>Wind</h4>
                            <p>{weather.get('wind_speed', '--')} mph</p>
                        </div>
                        <div>
                            <h4>Precip Probability</h4>
                            <p>{weather.get('precip_probability', 0)}%</p>
                        </div>
                    </div>
                    <span style="font-size: 0.9rem; color: #475569;">
                        Conditions: {weather.get('conditions', '--')}
                    </span><br/>
                    <span style="font-size: 0.85rem; color: #f97316; font-weight: 600;">
                        Weather Factor: {weather_factor:.1f}x
                    </span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.caption("Weather data unavailable")
            st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.error("Failed to load predictions. Make sure the API server is running.")
        st.markdown("""
        **Troubleshooting:**
        1. Start the API server: `cd api && python main.py`
        2. Ensure PostgreSQL is running with Austin schema
        3. Run data ingestion: `POST /api/austin/ingest`
        """)

    # Deployment Recommendations section
    st.markdown("---")
    st.markdown("### 🚛 Deployment Recommendations")

    if predictions_data and predictions_data.get("success"):
        recommendations = predictions_data.get("recommendations", [])

        if recommendations:
            rec_cols = st.columns(min(len(recommendations), 5))

            for i, rec in enumerate(recommendations[:5]):
                with rec_cols[i]:
                    risk_level = rec.get("risk_level", "HIGH")
                    bg_color = "#fee2e2" if rec.get("priority_rank") == 1 else "#f3f4f6"

                    st.markdown(f"""
                    <div style="background-color: {bg_color}; padding: 15px; border-radius: 10px; text-align: center;">
                        <div style="font-size: 12px; color: #6b7280;">Priority #{rec.get('priority_rank', i+1)}</div>
                        <div style="font-size: 24px; font-weight: bold;">{rec.get('sector_code', '??')}</div>
                        <div style="font-size: 14px;">Deploy {rec.get('recommended_units', 1)} unit(s)</div>
                        <div style="font-size: 12px; color: {get_risk_color(risk_level)};">{rec.get('risk_score', 0):.1%} risk</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Show rationales
            with st.expander("📝 Deployment Rationales"):
                for rec in recommendations[:10]:
                    st.markdown(f"**{rec.get('sector_code')}**: {rec.get('rationale', 'No rationale')}")
        else:
            st.info("No high-risk sectors requiring deployment at this time.")

    # Risk Distribution and Incident Types
    st.markdown("---")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("### 📈 Risk Distribution")
        if predictions_data and predictions_data.get("success"):
            predictions = predictions_data.get("predictions", [])
            if predictions:
                fig = create_risk_distribution_chart(predictions)
                st.plotly_chart(fig, use_container_width=True)

    with chart_col2:
        st.markdown("### 📊 Incident Types (24h)")
        incident_data = get_incident_types(hours_back=24)
        if incident_data and incident_data.get("success"):
            distribution = incident_data.get("distribution", {})
            if distribution:
                fig = create_incident_type_chart(distribution)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No incident data available")
        else:
            st.info("Could not load incident types")

    # Recent Incidents table
    st.markdown("---")
    st.markdown(f"### 📋 Recent Incidents (Last {hours_back} hours)")

    incidents_data = get_live_incidents(hours_back=hours_back)
    if incidents_data and incidents_data.get("success"):
        incidents = incidents_data.get("incidents", [])

        if incidents:
            # Convert to DataFrame
            df = pd.DataFrame(incidents)

            # Select and rename columns
            display_cols = {
                "traffic_report_id": "ID",
                "issue_reported": "Issue",
                "sector_code": "Sector",
                "address": "Address",
                "published_date": "Time"
            }

            available_cols = [col for col in display_cols.keys() if col in df.columns]
            df_display = df[available_cols].rename(columns=display_cols)

            # Format time
            if "Time" in df_display.columns:
                df_display["Time"] = pd.to_datetime(df_display["Time"]).dt.strftime("%m/%d %H:%M")

            st.dataframe(
                df_display.head(20),
                use_container_width=True,
                hide_index=True
            )

            st.caption(f"Showing {min(20, len(incidents))} of {len(incidents)} incidents")
        else:
            st.info(f"No incidents in the last {hours_back} hour(s)")
    else:
        st.warning("Could not load incident data")

    # Auto-refresh logic
    if auto_refresh:
        import time
        time.sleep(60)
        st.rerun()

    # Footer
    st.markdown("---")
    st.caption(
        f"TRAFFIX Austin Hotspot Predictor | "
        f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
        f"Data sources: Austin Open Data, NOAA Weather API"
    )


if __name__ == "__main__":
    main()