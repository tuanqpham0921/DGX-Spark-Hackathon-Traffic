"""
TRAFFIX API Server
FastAPI backend serving the Next.js frontend with real-time data from Postgres, Tavily, and LangGraph agents.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path so we can import from data, agents, utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import asyncio
import json
import math

# Import TRAFFIX components
from data.queries import TrafficQueries
from data.db_connection import get_db_connection, initialize_database
from utils.config_loader import get_config
from utils.date_ranges import get_date_range_by_type, get_poc_constraints
from utils.logger import get_logger

logger = get_logger(__name__)

# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database connection on startup."""
    logger.info("Starting TRAFFIX API server...")
    await initialize_database()
    logger.info("Database initialized successfully")
    yield
    logger.info("Shutting down TRAFFIX API server...")

# Initialize FastAPI with lifespan
app = FastAPI(
    title="TRAFFIX API",
    description="AI-powered traffic intelligence for DC & Virginia",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "https://*.vercel.app",
        "*"  # For development; restrict in production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    persona: str = "manager"  # executive, manager, analyst
    region: str = "Virginia"
    time_period: str = "MTD"
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class MetricsRequest(BaseModel):
    region: str = "Virginia"
    time_period: str = "MTD"
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class EventsRequest(BaseModel):
    region: str = "Virginia"
    time_period: str = "MTD"
    limit: int = 20
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class NewsRequest(BaseModel):
    region: str = "Virginia"
    query: Optional[str] = None
    days: int = 7

class WeatherRequest(BaseModel):
    start_date: str
    end_date: str

class RecommendationsRequest(BaseModel):
    region: str = "Virginia"
    time_period: str = "MTD"
    start_date: Optional[str] = None
    end_date: Optional[str] = None


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "TRAFFIX API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/health")
async def health_check():
    """Detailed health check."""
    try:
        db = get_db_connection()
        # Simple query to test DB connection
        await db.execute_query_one("SELECT 1 as test")
        db_status = "healthy"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_status = "unhealthy"
    
    return {
        "status": "healthy" if db_status == "healthy" else "degraded",
        "components": {
            "api": "healthy",
            "database": db_status,
            "timestamp": datetime.now().isoformat()
        }
    }


@app.post("/api/metrics")
async def get_metrics(request: MetricsRequest):
    """
    Get dashboard metrics (BAN numbers).
    Returns: total_trips, avg_reliability, avg_efficiency, active_events
    """
    try:
        queries = TrafficQueries(get_db_connection())
        
        # Parse date range
        if request.start_date and request.end_date:
            start_time = datetime.fromisoformat(request.start_date)
            end_time = datetime.fromisoformat(request.end_date).replace(hour=23, minute=59, second=59)
        else:
            date_range = get_date_range_by_type(request.time_period)
            start_time = date_range["start"]
            end_time = date_range["end"]
        
        # Get metrics from database
        metrics = await queries.get_dashboard_metrics(
            region=request.region,
            start_time=start_time,
            end_time=end_time
        )
        total_trips = float(metrics["total_trips"])
        total_forecast = float(metrics.get("total_trips_forecast") or 0)
        forecast_gap = float(metrics.get("forecast_gap") or (total_trips - total_forecast))
        gap_percent = (forecast_gap / total_forecast * 100) if total_forecast else 0.0

        def format_large_number(value: float) -> str:
            abs_val = abs(value)
            if abs_val >= 1_000_000:
                return f"{value / 1_000_000:.1f}M"
            if abs_val >= 1_000:
                return f"{value / 1_000:.1f}K"
            return f"{value:.0f}"
        
        # Format for frontend
        return {
            "success": True,
            "data": {
                "total_trips": int(total_trips),
                "total_trips_formatted": format_large_number(total_trips),
                "forecast_trips": int(total_forecast),
                "forecast_trips_formatted": format_large_number(total_forecast),
                "forecast_gap": forecast_gap,
                "forecast_gap_formatted": f"{forecast_gap:+,.0f}",
                "forecast_gap_percent": round(gap_percent, 1),
                "avg_reliability": round(metrics["avg_reliability"] * 100, 1),  # Convert to percentage (0-100)
                "avg_reliability_formatted": f"{metrics['avg_reliability'] * 100:.1f}%",
                "avg_efficiency": round(metrics["avg_efficiency"] * 100, 1),  # Convert to percentage (0-100)
                "active_events": int(metrics["active_events"]),
                "region": request.region,
                "time_period": request.time_period,
                "start_date": start_time.isoformat(),
                "end_date": end_time.isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Error fetching metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/events")
async def get_events(request: EventsRequest):
    """
    Get recent road events from Postgres.
    Returns events with type, severity, location, time.
    """
    try:
        queries = TrafficQueries(get_db_connection())
        
        # Parse date range
        if request.start_date and request.end_date:
            # Use custom date range
            start_time = datetime.fromisoformat(request.start_date)
            end_time = datetime.fromisoformat(request.end_date)
            # Set end time to end of day
            end_time = end_time.replace(hour=23, minute=59, second=59)
        else:
            # Use predefined time period
            date_range = get_date_range_by_type(request.time_period)
            start_time = date_range["start"]
            end_time = date_range["end"]
        
        # Get events from database
        events = await queries.get_events_by_region(
            region=request.region,
            start_time=start_time,
            end_time=end_time,
            limit=request.limit
        )
        
        # Format events for frontend
        formatted_events = []
        for event in events:
            # Calculate time ago
            try:
                event_time = datetime.strptime(event.get("start_time", ""), "%m/%d/%y %H:%M")
                time_diff = datetime.now() - event_time
                if time_diff.total_seconds() < 3600:
                    time_ago = f"{int(time_diff.total_seconds() / 60)} min ago"
                elif time_diff.total_seconds() < 86400:
                    time_ago = f"{int(time_diff.total_seconds() / 3600)} hours ago"
                else:
                    time_ago = f"{int(time_diff.days)} days ago"
            except:
                time_ago = "Recently"
            
            formatted_events.append({
                "id": event.get("office_event_id_tag", ""),
                "name": event.get("road", "Unknown Road"),
                "type": event.get("event_details_level_1", "Incident"),
                "description": event.get("event_details_level_2", "Traffic event"),
                "time": time_ago,
                "severity": "high" if "accident" in str(event.get("event_details_level_1", "")).lower() else "medium",
                "location": {
                    "road": event.get("road", ""),
                    "county": event.get("county", ""),
                    "state": event.get("state", ""),
                    "latitude": event.get("latitude"),
                    "longitude": event.get("longitude")
                },
                "duration": event.get("overall_event_time", 0)
            })
        
        return {
            "success": True,
            "data": formatted_events,
            "count": len(formatted_events),
            "region": request.region,
            "time_period": request.time_period
        }
    except Exception as e:
        logger.error(f"Error fetching events: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/worst-accidents")
async def get_worst_accidents(request: EventsRequest):
    """
    Get top 10 worst accidents based on overall roadway clearance time.
    Returns accidents with detailed summaries including location, clearance time, and event details.
    """
    try:
        queries = TrafficQueries(get_db_connection())
        
        # Parse date range
        if request.start_date and request.end_date:
            start_time = datetime.fromisoformat(request.start_date)
            end_time = datetime.fromisoformat(request.end_date)
            end_time = end_time.replace(hour=23, minute=59, second=59)
        else:
            date_range = get_date_range_by_type(request.time_period)
            start_time = date_range["start"]
            end_time = date_range["end"]
        
        # Get all accidents from database
        db = get_db_connection()
        
        query = """
            SELECT 
                office_event_id_tag,
                event_details_level_1,
                event_details_level_2,
                event_details_level_3,
                road,
                county,
                state,
                "Start Time" as start_time,
                "End Time" as end_time,
                overall_event_time,
                roadway_clearance_time,
                granular_incident_place,
                latitude,
                longitude
            FROM public.events
            WHERE ($1 = 'All' OR state = $1)
                AND TO_TIMESTAMP("Start Time", 'MM/DD/YY HH24:MI') >= $2
                AND TO_TIMESTAMP("End Time", 'MM/DD/YY HH24:MI') <= $3
                AND LOWER(event_details_level_1) LIKE '%accident%'
                AND overall_event_time IS NOT NULL
                AND overall_event_time > 0
                AND road IS NOT NULL
                AND LOWER(road) NOT LIKE '%unknown%'
                AND (granular_incident_place IS NOT NULL OR (road IS NOT NULL AND county IS NOT NULL))
            ORDER BY overall_event_time DESC
            LIMIT 10
        """
        
        async with db.pool.acquire() as conn:
            rows = await conn.fetch(query, request.region, start_time, end_time)
        
        # Format accidents for frontend
        worst_accidents = []
        for row in rows:
            # Parse times
            overall_time = float(row["overall_event_time"]) if row["overall_event_time"] else 0
            clearance_time = float(row["roadway_clearance_time"]) if row["roadway_clearance_time"] else 0
            
            # Format time strings
            overall_hours = overall_time / 60.0
            clearance_hours = clearance_time / 60.0
            
            if overall_hours < 1:
                overall_str = f"{int(overall_time)} minutes"
            else:
                overall_str = f"{overall_hours:.1f} hours"
            
            if clearance_hours < 1:
                clearance_str = f"{int(clearance_time)} minutes"
            else:
                clearance_str = f"{clearance_hours:.1f} hours"
            
            # Build simple summary: location + event details + total time
            location = row["granular_incident_place"] or f"{row['road']}, {row['county']}"
            event_level_2 = row["event_details_level_2"] or ""
            event_level_3 = row["event_details_level_3"] or ""
            
            # Build non-repetitive details from event_details_2 and event_details_3
            details_parts = []
            if event_level_2 and event_level_2.strip():
                details_parts.append(event_level_2.strip())
            if event_level_3 and event_level_3.strip() and event_level_3.strip().lower() != event_level_2.strip().lower():
                details_parts.append(event_level_3.strip())
            
            # Create concise summary: location + details + total time
            if details_parts:
                details_text = " ".join(details_parts)
                summary = f"Located at {location}. {details_text} Total time: {overall_str}."
            else:
                summary = f"Located at {location}. Total time: {overall_str}."
            
            worst_accidents.append({
                "id": row["office_event_id_tag"],
                "road": row["road"],
                "county": row["county"],
                "state": row["state"],
                "location": location,
                "overall_time": overall_time,
                "overall_time_formatted": overall_str,
                "clearance_time": clearance_time,
                "clearance_time_formatted": clearance_str,
                "event_type": row["event_details_level_1"],
                "event_details_2": event_level_2,
                "event_details_3": event_level_3,
                "summary": summary,
                "start_time": row["start_time"],
                "end_time": row["end_time"],
                "latitude": float(row["latitude"]) if row["latitude"] else None,
                "longitude": float(row["longitude"]) if row["longitude"] else None
            })
        
        return {
            "success": True,
            "data": worst_accidents,
            "count": len(worst_accidents),
            "region": request.region,
            "time_period": request.time_period if not request.start_date else "custom"
        }
        
    except Exception as e:
        logger.error(f"Error fetching worst accidents: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analytics/hourly-traffic")
async def get_hourly_traffic(request: EventsRequest):
    """
    Get hourly traffic data grouped by date within the selected range.
    Returns traffic volume per hour (0-23) for each day.
    """
    try:
        # Parse date range
        if request.start_date and request.end_date:
            start_time = datetime.fromisoformat(request.start_date)
            end_time = datetime.fromisoformat(request.end_date)
            end_time = end_time.replace(hour=23, minute=59, second=59)
        else:
            date_range = get_date_range_by_type(request.time_period)
            start_time = date_range["start"]
            end_time = date_range["end"]
        
        db = get_db_connection()
        
        query = """
            SELECT 
                date,
                hour,
                total_car_travel_actual as traffic,
                network_reliability as reliability
            FROM public.trips
            WHERE ($1 = 'All' OR region = $1)
                AND date >= $2::date
                AND date <= $3::date
            ORDER BY date, hour
        """
        
        async with db.pool.acquire() as conn:
            rows = await conn.fetch(query, request.region, start_time.date(), end_time.date())
        
        # Group by hour and date
        hourly_data = {}
        for row in rows:
            hour = int(row["hour"])
            date_str = row["date"].strftime("%Y-%m-%d")
            day_name = row["date"].strftime("%b %d")
            
            if hour not in hourly_data:
                hourly_data[hour] = []
            
            hourly_data[hour].append({
                "date": date_str,
                "day": day_name,
                "traffic": float(row["traffic"]) if row["traffic"] else 0,
                "reliability": float(row["reliability"]) * 100 if row["reliability"] else 0
            })
        
        # Format for frontend
        formatted_data = []
        for hour in range(24):
            hour_label = f"{hour:02d}:00"
            if hour in hourly_data:
                formatted_data.append({
                    "hour": hour,
                    "hour_label": hour_label,
                    "days": hourly_data[hour]
                })
            else:
                formatted_data.append({
                    "hour": hour,
                    "hour_label": hour_label,
                    "days": []
                })
        
        return {
            "success": True,
            "data": formatted_data,
            "count": len(formatted_data),
            "date_range": {
                "start": start_time.strftime("%Y-%m-%d"),
                "end": end_time.strftime("%Y-%m-%d")
            }
        }
        
    except Exception as e:
        logger.error(f"Error fetching hourly traffic: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/news")
async def get_news(request: NewsRequest):
    """
    Get top 10 latest news articles:
    - 5 traffic-related (congestion, accidents, delays, VDOT)
    - 5 political/economic (government, metro, economy affecting traffic)
    
    Uses PostgreSQL caching for speed. Provides citations and summaries.
    This data enriches the AI chatbot's context.
    """
    try:
        queries = TrafficQueries(get_db_connection())
        
        # Try to get cached news first (cached for 24 hours)
        cached_traffic = await queries.get_cached_news(
            category="traffic",
            region=request.region,
            max_age_hours=24,
            limit=5
        )
        
        cached_political = await queries.get_cached_news(
            category="political_economy",
            region=request.region,
            max_age_hours=24,
            limit=5
        )
        
        logger.info(f"Cached news: {len(cached_traffic)} traffic, {len(cached_political)} political/economy")
        
        # If we have enough cached articles, return them
        if len(cached_traffic) >= 5 and len(cached_political) >= 5:
            logger.info("Returning cached news articles")
            all_articles = cached_traffic[:5] + cached_political[:5]
            
            # Sort by published_date (newest first)
            all_articles.sort(
                key=lambda x: x.get("published_date") or datetime.min, 
                reverse=True
            )
            
            # Format for frontend
            news_items = []
            for article in all_articles:
                news_items.append({
                    "id": article.get("id"),
                    "title": article.get("title", ""),
                    "source": article.get("source", "News"),
                    "url": article.get("url", ""),
                    "content": article.get("summary", "")[:200] + "...",
                    "summary": article.get("summary", ""),
                    "published": article.get("published_date", "").isoformat() if article.get("published_date") else "",
                    "time": "Recently",
                    "category": "Traffic" if article.get("category") == "traffic" else "Political/Economy",
                    "keywords": article.get("keywords", []),
                    "impact": "High" if "accident" in str(article.get("keywords", [])).lower() else "Medium"
                })
            
            return {
                "success": True,
                "data": news_items,
                "count": len(news_items),
                "cached": True
            }
        
        # If not enough cached, fetch fresh from Tavily
        logger.info("Fetching fresh news from Tavily API")
        tavily = TavilyClient()
        
        # Build region-specific query
        region_query = f"{request.region}" if request.region != "All" else "Washington DC Virginia"
        
        # ===== SEARCH 1: TRAFFIC ARTICLES =====
        traffic_keywords = [
            "traffic congestion", "traffic delays", "road accident", 
            "VDOT", "highway accident", "traffic jam", "road closure",
            "highway construction", "traffic incident"
        ]
        
        traffic_query = f"{region_query} {' OR '.join(traffic_keywords[:5])}"  # Limit query length
        logger.info(f"Traffic query: {traffic_query}")
        
        traffic_results = await tavily.search(
            query=traffic_query,
            max_results=10,
            search_depth="basic"
        )
        
        # ===== SEARCH 2: POLITICAL/ECONOMIC ARTICLES =====
        political_keywords = [
            "government shutdown", "airplane travel", "metro shutdown", 
            "metro delays", "new governor", "economic downturn", 
            "consumer spending", "return to work", "transit funding",
            "infrastructure bill", "commuter"
        ]
        
        political_query = f"{region_query} {' OR '.join(political_keywords[:5])}"
        logger.info(f"Political/Economic query: {political_query}")
        
        political_results = await tavily.search(
            query=political_query,
            max_results=10,
            search_depth="basic"
        )
        
        # Process and cache traffic articles
        traffic_articles = []
        for result in traffic_results.get("results", [])[:5]:
            title = result.get("title", "")
            url = result.get("url", "")
            content = result.get("content", "")
            summary = content[:300] if content else title
            published_str = result.get("published_date", "")
            
            # Parse published date
            published_date = None
            if published_str:
                try:
                    published_date = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
                except:
                    published_date = datetime.now()
            else:
                published_date = datetime.now()
            
            # Determine which keywords matched
            matched_keywords = [kw for kw in traffic_keywords if kw.lower() in (title + content).lower()]
            
            # Cache to PostgreSQL
            await queries.cache_news_article(
                title=title,
                url=url,
                summary=summary,
                published_date=published_date,
                source=result.get("source", "News"),
                category="traffic",
                keywords=matched_keywords,
                content=content,
                region=request.region
            )
            
            traffic_articles.append({
                "title": title,
                "source": result.get("source", "News"),
                "url": url,
                "content": summary[:200] + "...",
                "summary": summary,
                "published": published_date.isoformat(),
                "time": "Recently",
                "category": "Traffic",
                "keywords": matched_keywords,
                "impact": "High" if "accident" in title.lower() else "Medium"
            })
        
        # Process and cache political/economic articles
        political_articles = []
        for result in political_results.get("results", [])[:5]:
            title = result.get("title", "")
            url = result.get("url", "")
            content = result.get("content", "")
            summary = content[:300] if content else title
            published_str = result.get("published_date", "")
            
            # Parse published date
            published_date = None
            if published_str:
                try:
                    published_date = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
                except:
                    published_date = datetime.now()
            else:
                published_date = datetime.now()
            
            # Determine which keywords matched
            matched_keywords = [kw for kw in political_keywords if kw.lower() in (title + content).lower()]
            
            # Cache to PostgreSQL
            await queries.cache_news_article(
                title=title,
                url=url,
                summary=summary,
                published_date=published_date,
                source=result.get("source", "News"),
                category="political_economy",
                keywords=matched_keywords,
                content=content,
                region=request.region
            )
            
            political_articles.append({
                "title": title,
                "source": result.get("source", "News"),
                "url": url,
                "content": summary[:200] + "...",
                "summary": summary,
                "published": published_date.isoformat(),
                "time": "Recently",
                "category": "Political/Economy",
                "keywords": matched_keywords,
                "impact": "Medium"
            })
        
        # Combine and sort by published date (newest first)
        all_articles = traffic_articles + political_articles
        all_articles.sort(
            key=lambda x: datetime.fromisoformat(x["published"]) if x.get("published") else datetime.min,
            reverse=True
        )
        
        return {
            "success": True,
            "data": all_articles[:10],  # Top 10
            "count": len(all_articles[:10]),
            "cached": False,
            "metadata": {
                "traffic_articles": len(traffic_articles),
                "political_articles": len(political_articles)
            }
        }
        
    except Exception as e:
        logger.error(f"Error fetching news: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Try to return cached news as fallback
        try:
            queries = TrafficQueries(get_db_connection())
            cached_news = await queries.get_cached_news(
                region=request.region,
                max_age_hours=168,  # 1 week fallback
                limit=10
            )
            
            if cached_news:
                news_items = []
                for article in cached_news:
                    news_items.append({
                        "title": article.get("title", ""),
                        "source": article.get("source", "News"),
                        "url": article.get("url", ""),
                        "content": article.get("summary", "")[:200] + "...",
                        "summary": article.get("summary", ""),
                        "published": article.get("published_date", "").isoformat() if article.get("published_date") else "",
                        "time": "Recently",
                        "category": "Traffic" if article.get("category") == "traffic" else "Political/Economy",
                        "keywords": article.get("keywords", []),
                        "impact": "Medium"
                    })
                
                return {
                    "success": True,
                    "data": news_items,
                    "count": len(news_items),
                    "cached": True,
                    "fallback": True
                }
        except:
            pass
        
        # Last resort fallback
        return {
            "success": False,
            "error": "News service temporarily unavailable",
            "message": str(e),
            "data": []
        }


@app.post("/api/weather")
async def get_weather(request: WeatherRequest):
    """
    Get weather data from PostgreSQL weather table for the specified date range.
    Returns hourly weather records.
    """
    try:
        db = get_db_connection()
        
        # Parse date range
        start_date = datetime.fromisoformat(request.start_date)
        end_date = datetime.fromisoformat(request.end_date)
        # Set end time to end of day
        end_date = end_date.replace(hour=23, minute=59, second=59)
        
        # Query weather table
        query = """
            SELECT 
                datetime,
                temp,
                precip,
                preciptype,
                windspeed,
                visibility,
                conditions
            FROM public.weather
            WHERE datetime >= $1 AND datetime <= $2
            ORDER BY datetime DESC
        """
        
        async with db.pool.acquire() as conn:
            rows = await conn.fetch(query, start_date, end_date)
            
            weather_data = []
            for row in rows:
                weather_data.append({
                    "datetime": row["datetime"].isoformat() if row["datetime"] else None,
                    "temp": float(row["temp"]) if row["temp"] is not None else None,
                    "precip": float(row["precip"]) if row["precip"] is not None else None,
                    "preciptype": row["preciptype"],
                    "windspeed": float(row["windspeed"]) if row["windspeed"] is not None else None,
                    "visibility": float(row["visibility"]) if row["visibility"] is not None else None,
                    "conditions": row["conditions"]
                })
        
        return {
            "success": True,
            "data": weather_data,
            "count": len(weather_data)
        }
        
    except Exception as e:
        logger.error(f"Error fetching weather: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/recommendations")
async def get_recommendations(request: RecommendationsRequest):
    """
    Generate 3 specific AI-powered recommendations using multi-agent system:
    1. Accident hotspots - signage, infrastructure, behavior campaigns
    2. Construction impact - correlation with accidents
    3. Disabled vehicle patterns - prevention strategies
    """
    try:
        from utils.cache import get_cache
        import hashlib
        
        queries = TrafficQueries(get_db_connection())
        
        # Parse date range
        if request.start_date and request.end_date:
            start_time = datetime.fromisoformat(request.start_date)
            end_time = datetime.fromisoformat(request.end_date)
            end_time = end_time.replace(hour=23, minute=59, second=59)
        else:
            date_range = get_date_range_by_type(request.time_period)
            start_time = date_range["start"]
            end_time = date_range["end"]
        
        # Check cache first (cache by region + time period)
        cache = get_cache()
        cache_key = hashlib.md5(
            f"recommendations_{request.region}_{start_time.isoformat()}_{end_time.isoformat()}".encode()
        ).hexdigest()
        
        cached_result = await cache.get(cache_key)
        if cached_result:
            logger.info(f"Returning cached recommendations for {request.region}")
            return cached_result
        
        # Fetch comprehensive data
        logger.info(f"Fetching events for recommendations: {request.region}, {start_time} to {end_time}")
        events = await queries.get_events_by_region(
            region=request.region,
            start_time=start_time,
            end_time=end_time,
            limit=10000
        )
        
        logger.info(f"Fetched {len(events)} events for analysis")
        
        # ===== ANALYZE DATA BY CATEGORY =====
        
        # 1. ACCIDENTS - Find hotspots and patterns
        accidents = [e for e in events if "accident" in str(e.get("event_details_level_1", "")).lower()]
        accident_locations = {}
        accident_counties = {}
        for acc in accidents:
            road = acc.get("road", "Unknown")
            county = acc.get("county", "Unknown")
            if road not in accident_locations:
                accident_locations[road] = {"count": 0, "events": [], "county": county}
            accident_locations[road]["count"] += 1
            accident_locations[road]["events"].append(acc)
            accident_counties[county] = accident_counties.get(county, 0) + 1
        
        top_accident_roads = sorted(accident_locations.items(), key=lambda x: x[1]["count"], reverse=True)[:5]
        top_accident_counties = sorted(accident_counties.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # 2. CONSTRUCTION - Find construction zones and nearby accidents
        construction = [e for e in events if "construction" in str(e.get("event_details_level_1", "")).lower()]
        construction_accident_correlation = []
        for const in construction:
            const_road = const.get("road", "")
            const_county = const.get("county", "")
            # Find accidents near construction
            nearby_accidents = [a for a in accidents if 
                              a.get("road", "") == const_road or a.get("county", "") == const_county]
            if nearby_accidents:
                construction_accident_correlation.append({
                    "construction": const,
                    "nearby_accidents": len(nearby_accidents),
                    "road": const_road,
                    "county": const_county
                })
        
        # 3. DISABLED VEHICLES - Find patterns and frequent locations
        disabled = [e for e in events if "disabled" in str(e.get("event_details_level_1", "")).lower()]
        disabled_locations = {}
        disabled_times = {}  # Hour of day analysis
        for dis in disabled:
            road = dis.get("road", "Unknown")
            if road not in disabled_locations:
                disabled_locations[road] = {"count": 0, "events": []}
            disabled_locations[road]["count"] += 1
            disabled_locations[road]["events"].append(dis)
            
            # Time analysis
            start_time_str = dis.get("start_time", "")
            if start_time_str:
                try:
                    hour = int(start_time_str.split()[1].split(":")[0]) if " " in start_time_str else 0
                    disabled_times[hour] = disabled_times.get(hour, 0) + 1
                except:
                    pass
        
        top_disabled_roads = sorted(disabled_locations.items(), key=lambda x: x[1]["count"], reverse=True)[:5]
        peak_disabled_hours = sorted(disabled_times.items(), key=lambda x: x[1], reverse=True)[:3]
        
        logger.info(f"Analysis: {len(accidents)} accidents, {len(construction)} construction, {len(disabled)} disabled vehicles")
        
        # ===== USE MULTI-AGENT SYSTEM FOR DETAILED RECOMMENDATIONS =====
        from agents.research import ResearchAgent
        from agents.writer import WriterAgent
        
        config = get_config()
        
        # Initialize agents
        research_agent = ResearchAgent(config)
        writer_agent = WriterAgent(config)
        
        recommendations = []
        
        # ===== RECOMMENDATION 1: ACCIDENT HOTSPOTS =====
        if len(accidents) > 0:
            logger.info("Generating Recommendation 1: Accident Hotspots")
            
            # Prepare research query
            accident_query = f"""Analyze accident patterns in {request.region} from {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}.

Total Accidents: {len(accidents)}
Top Accident Locations:
{chr(10).join([f"- {road}: {data['count']} accidents (County: {data['county']})" for road, data in top_accident_roads[:3]])}

Top Accident Counties:
{chr(10).join([f"- {county}: {count} accidents" for county, count in top_accident_counties])}

Focus on: signage improvements, infrastructure problems, and travel behavior campaigns to reduce accidents."""
            
            # Use research agent to gather insights
            research_data = {
                "query": accident_query,
                "sql_data": {
                    "accidents": accidents[:10],  # Sample
                    "top_locations": top_accident_roads[:3],
                    "top_counties": top_accident_counties
                },
                "tavily_data": []
            }
            
            # Use writer agent to create detailed recommendation
            state = {
                "research_data": research_data,
                "query": accident_query,
                "mode": "quick",
                "detail_level": "analyst",
                "region": request.region
            }
            writer_response = await writer_agent.process(state)
            
            # Extract content from writer response
            content = writer_response.get("content", "")
            if isinstance(content, str):
                content_summary = content[:200]
            else:
                content_summary = str(content)[:200]
            
            # Structure the recommendation
            top_accident_location = top_accident_roads[0] if top_accident_roads else ("Unknown", {"county": request.region, "count": 0})
            recommendations.append({
                "id": "rec_1_accidents",
                "insight": f"{len(accidents)} accidents occurred in {request.region}, with {top_accident_location[0]} being the most problematic location ({top_accident_location[1]['count']} accidents). Immediate intervention needed to improve safety.",
                "confidence": "High",
                "location": f"{top_accident_location[1].get('county', request.region)} - {top_accident_location[0]}",
                "actions": [
                    "Install enhanced warning signage at high-accident zones",
                    "Conduct road surface and lighting infrastructure assessments",
                    "Launch targeted travel behavior campaign focusing on speed reduction",
                    "Deploy additional law enforcement during peak accident hours"
                ],
                "impact": f"Expected 25-35% reduction in accidents at target locations, potentially preventing {int(len(accidents) * 0.3)} incidents",
                "full_analysis": f"The concentration of accidents at specific locations suggests systemic issues requiring multi-pronged intervention. Infrastructure improvements should focus on road surface quality, lighting, and sight distance at {top_accident_location[0]}. Implement digital message boards with real-time safety alerts and speed warnings. Coordinate with local media for a public safety campaign emphasizing defensive driving in {top_accident_location[1].get('county', request.region)}. Consider traffic calming measures and enhanced enforcement during peak times.",
                "category": "Safety Improvement"
            })
        
        # ===== RECOMMENDATION 2: CONSTRUCTION-ACCIDENT CORRELATION =====
        if len(construction) > 0:
            logger.info("Generating Recommendation 2: Construction Impact")
            
            construction_query = f"""Analyze construction zones and accident correlation in {request.region} from {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}.

Total Construction Zones: {len(construction)}
Construction-Accident Correlations: {len(construction_accident_correlation)} zones with nearby accidents
{chr(10).join([f"- {item['road']} in {item['county']}: {item['nearby_accidents']} nearby accidents" for item in construction_accident_correlation[:3]])}

Focus on: improving construction zone safety, better signage, and traffic management around work zones."""
            
            research_data = {
                "query": construction_query,
                "sql_data": {
                    "construction": construction[:10],
                    "correlations": construction_accident_correlation[:5]
                },
                "tavily_data": []
            }
            
            state = {
                "research_data": research_data,
                "query": construction_query,
                "mode": "quick",
                "detail_level": "analyst",
                "region": request.region
            }
            writer_response = await writer_agent.process(state)
            
            top_construction_issue = construction_accident_correlation[0] if construction_accident_correlation else None
            
            if top_construction_issue:
                recommendations.append({
                    "id": "rec_2_construction",
                    "insight": f"{len(construction)} construction zones identified, with {len(construction_accident_correlation)} zones showing accident correlation. {top_construction_issue['road']} in {top_construction_issue['county']} has {top_construction_issue['nearby_accidents']} accidents near active construction.",
                    "confidence": "High",
                    "location": f"{top_construction_issue['county']} - {top_construction_issue['road']}",
                    "actions": [
                        "Implement enhanced construction zone warning systems (advance warning signs at 2 miles, 1 mile, and 0.5 miles)",
                        "Deploy variable speed limit signs around construction zones",
                        "Increase buffer zones and lane separation in active work areas",
                        "Require contractor compliance with MUTCD temporary traffic control standards"
                    ],
                    "impact": f"Estimated 40-50% reduction in construction zone incidents, improving safety for {len(construction_accident_correlation)} high-risk zones",
                    "full_analysis": f"Construction zones create hazardous conditions that require proactive traffic management. At {top_construction_issue['road']}, implement a comprehensive safety corridor: deploy early warning systems, reduce speed limits with enforcement, ensure adequate lane separation with physical barriers. Coordinate with contractors to minimize lane closures during peak hours. Launch a targeted media campaign about construction zone safety. Consider implementing automated speed enforcement in persistent problem areas.",
                    "category": "Infrastructure"
                })
            else:
                # If no correlation but construction exists
                recommendations.append({
                    "id": "rec_2_construction",
                    "insight": f"{len(construction)} construction zones active in {request.region}. Proactive safety measures recommended to prevent accident escalation.",
                    "confidence": "Medium",
                    "location": request.region,
                    "actions": [
                        "Establish standard construction zone safety protocols",
                        "Deploy advance warning signage for all construction areas",
                        "Implement regular safety audits of active work zones",
                        "Coordinate construction scheduling to minimize congestion"
                    ],
                    "impact": "Maintain low accident rates in construction zones, prevent future incidents",
                    "full_analysis": "While current construction zones show minimal accident correlation, maintaining rigorous safety standards is critical. Implement standardized traffic control plans across all construction sites, ensure contractor compliance with safety protocols, and deploy consistent warning signage. Regular safety audits will identify emerging risks before incidents occur.",
                    "category": "Infrastructure"
                })
        
        # ===== RECOMMENDATION 3: DISABLED VEHICLE PATTERNS =====
        if len(disabled) > 0:
            logger.info("Generating Recommendation 3: Disabled Vehicle Prevention")
            
            disabled_query = f"""Analyze disabled vehicle patterns in {request.region} from {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}.

Total Disabled Vehicles: {len(disabled)}
Top Disabled Vehicle Locations:
{chr(10).join([f"- {road}: {data['count']} incidents" for road, data in top_disabled_roads[:3]])}

Peak Times:
{chr(10).join([f"- Hour {hour}:00: {count} incidents" for hour, count in peak_disabled_hours])}

Focus on: prevention strategies, rapid response, and public awareness."""
            
            research_data = {
                "query": disabled_query,
                "sql_data": {
                    "disabled": disabled[:10],
                    "top_locations": top_disabled_roads[:3],
                    "peak_hours": peak_disabled_hours
                },
                "tavily_data": []
            }
            
            state = {
                "research_data": research_data,
                "query": disabled_query,
                "mode": "quick",
                "detail_level": "analyst",
                "region": request.region
            }
            writer_response = await writer_agent.process(state)
            
            top_disabled_location = top_disabled_roads[0] if top_disabled_roads else ("Unknown", {"count": 0})
            peak_hour = peak_disabled_hours[0] if peak_disabled_hours else (0, 0)
            
            recommendations.append({
                "id": "rec_3_disabled",
                "insight": f"{len(disabled)} disabled vehicle incidents recorded, with {top_disabled_location[0]} experiencing the highest frequency ({top_disabled_location[1]['count']} incidents). Peak time: {peak_hour[0]:02d}:00 hours with {peak_hour[1]} incidents.",
                "confidence": "High",
                "location": f"{request.region} - {top_disabled_location[0]}",
                "actions": [
                    "Deploy mobile mechanic 'Highway Assist' patrols during peak hours",
                    "Install emergency pull-off areas at high-frequency breakdown locations",
                    "Launch vehicle maintenance awareness campaign (check engine, tires, fuel before travel)",
                    "Implement QuickClear program with towing incentives for rapid vehicle removal"
                ],
                "impact": f"Expected 30% reduction in disabled vehicle duration, preventing {int(len(disabled) * 0.2)} secondary incidents caused by disabled vehicles",
                "full_analysis": f"Disabled vehicles create significant safety and congestion risks, particularly at {top_disabled_location[0]}. Establish a rapid-response Highway Assist program with mobile mechanics during peak hours ({peak_hour[0]:02d}:00-{(peak_hour[0]+2) % 24:02d}:00). Install clearly marked emergency pull-off zones every 2 miles on high-frequency corridors. Partner with local auto shops and AAA for a public awareness campaign on pre-trip vehicle checks. Implement a QuickClear towing program with subsidized removal for vehicles disabled over 30 minutes.",
                "category": "Congestion Management"
            })
        
        # Build response
        result = {
            "success": True,
            "data": recommendations,
            "count": len(recommendations),
            "metadata": {
                "region": request.region,
                "time_period": f"{start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}",
                "total_events_analyzed": len(events),
                "accidents": len(accidents),
                "construction": len(construction),
                "disabled_vehicles": len(disabled)
            }
        }
        
        # Cache the result for 1 hour (3600 seconds)
        await cache.set(cache_key, result, ttl=3600)
        logger.info(f"Cached recommendations for {request.region}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/map-data")
async def get_map_data(request: EventsRequest):
    """
    Get event data for map visualization.
    Returns events with coordinates.
    """
    try:
        queries = TrafficQueries(get_db_connection())
        
        # Parse date range
        date_range = get_date_range_by_type(request.time_period)
        start_time = date_range["start"]
        end_time = date_range["end"]
        
        # Get most recent events with coordinates (clustering handles visualization)
        events = await queries.get_events_by_region(
            region=request.region,
            start_time=start_time,
            end_time=end_time,
            limit=2000  # Most recent 2K events - good balance of coverage and performance
        )
        
        # Filter events with valid coordinates and include all details
        map_events = []
        for event in events:
            lat = event.get("latitude")
            lon = event.get("longitude")
            if lat and lon:
                map_events.append({
                    "id": event.get("office_event_id_tag", ""),
                    "lat": float(lat),
                    "lon": float(lon),
                    "type": event.get("event_details_level_1", "Event"),
                    "road": event.get("road", ""),
                    "county": event.get("county", ""),
                    "state": event.get("state", ""),
                    "severity": "high" if "accident" in str(event.get("event_details_level_1", "")).lower() else "medium",
                    "description": event.get("event_details_level_2", ""),
                    "details": event.get("event_details_level_3", ""),
                    "start_time": event.get("start_time", ""),
                    "end_time": event.get("end_time", ""),
                    "roadway_clearance_time": event.get("roadway_clearance_time", ""),
                    "overall_event_time": event.get("overall_event_time", ""),
                    "source": event.get("system", "Unknown"),
                    "location": event.get("granular_incident_place", "")
                })
        
        return {
            "success": True,
            "data": {
                "events": map_events,
                "center": {
                    "lat": 38.9072 if request.region == "District of Columbia" else 37.4316,
                    "lon": -77.0369 if request.region == "District of Columbia" else -78.6569
                },
                "zoom": 10
            },
            "count": len(map_events)
        }
    except Exception as e:
        logger.error(f"Error fetching map data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Enhanced chat endpoint with detail-level based analysis depth.
    - Executive: 2-4 sentence summaries in bullet format
    - Manager: Paragraph-style insights covering multiple aspects
    - Analyst: Deep analysis with exact details, trends, and recommendations using research agents
    """
    try:
        detail_level = request.persona
        
        # Parse date range
        if hasattr(request, 'start_date') and hasattr(request, 'end_date') and request.start_date and request.end_date:
            start_time = datetime.fromisoformat(request.start_date)
            end_time = datetime.fromisoformat(request.end_date)
            end_time = end_time.replace(hour=23, minute=59, second=59)
        else:
            date_range = get_date_range_by_type(request.time_period)
            start_time = date_range["start"]
            end_time = date_range["end"]
        
        # Utility helpers for normalization
        def normalize_text(value: Any, default: str = "Unknown") -> str:
            if value is None:
                return default
            text = str(value).strip()
            return text if text else default
        
        def parse_minutes(value: Any) -> Optional[float]:
            if value is None:
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                if isinstance(value, str):
                    parts = value.split(":")
                    try:
                        if len(parts) == 3:
                            hours, minutes, seconds = [float(p) for p in parts]
                            return hours * 60 + minutes + seconds / 60
                        if len(parts) == 2:
                            hours, minutes = [float(p) for p in parts]
                            return hours * 60 + minutes
                        if len(parts) == 1:
                            return float(parts[0])
                    except (TypeError, ValueError):
                        return None
            return None
        
        # Get comprehensive data from PostgreSQL for ALL levels
        queries = TrafficQueries(get_db_connection())
        db = get_db_connection()
        
        # Dashboard metrics (aggregates) - ALL LEVELS
        metrics = await queries.get_dashboard_metrics(
            region=request.region,
            start_time=start_time,
            end_time=end_time
        )
        
        # Events - ALL LEVELS get comprehensive event data (vary limit by level for speed)
        event_limit = 150 if detail_level == "analyst" else 100 if detail_level == "manager" else 75
        events_context = await queries.get_events_by_region(
            region=request.region,
            start_time=start_time,
            end_time=end_time,
            limit=event_limit
        ) or []
        
        # News - ALL LEVELS get recent cached news (essential for policy/economic questions)
        news_limit = 10 if detail_level == "analyst" else 8 if detail_level == "manager" else 6
        news_articles = await queries.get_cached_news(
            region=request.region,
            max_age_hours=240,  # Up to 10 days of context
            limit=news_limit
        ) or []
        
        # Weather - ALL LEVELS get weather data for correlation analysis
        weather_limit = 20 if detail_level == "analyst" else 15 if detail_level == "manager" else 10
        weather_query = """
            SELECT datetime, temp, precip, preciptype, conditions
            FROM public.weather
            WHERE datetime >= $1 AND datetime <= $2
            ORDER BY datetime DESC
            LIMIT $3
        """
        async with db.pool.acquire() as conn:
            weather_rows = await conn.fetch(weather_query, start_time, end_time, weather_limit)
        
        weather_data = [{
            "datetime": str(row["datetime"]),
            "temp": float(row["temp"]) if row["temp"] else None,
            "precip": float(row["precip"]) if row["precip"] else 0,
            "preciptype": row["preciptype"],
            "conditions": row["conditions"]
        } for row in weather_rows]
        
        # Build context tailored to persona
        context_parts = []
        context_parts.append(f"Region: {request.region}")
        context_parts.append(f"Time Period: {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}")
        
        total_trips = metrics.get("total_trips", 0)
        total_forecast = metrics.get("total_trips_forecast", 0)
        forecast_gap = metrics.get("forecast_gap", 0)
        avg_reliability = metrics.get("avg_reliability", 0)
        avg_efficiency = metrics.get("avg_efficiency", 0)
        active_events = int(metrics.get("active_events", 0))
        
        context_parts.append(f"Total Traffic: {int(total_trips):,} trips")
        if total_forecast:
            diff_pct = (forecast_gap / total_forecast) * 100 if total_forecast else 0
            context_parts.append(
                f"Trips vs Forecast: {int(total_trips):,} actual vs {int(total_forecast):,} forecast ({forecast_gap:+,.0f}, {diff_pct:+.1f}%)."
            )
        context_parts.append(f"Network Reliability: {avg_reliability * 100:.1f}%")
        context_parts.append(f"Network Efficiency: {avg_efficiency * 100:.1f}%")
        context_parts.append(f"Active Events: {active_events} incidents")
        
        # Add recent events summary
        if events_context:
            accident_count = sum(
                1 for e in events_context
                if 'accident' in str(e.get('event_details_level_1', '')).lower()
            )
            construction_count = sum(
                1 for e in events_context
                if 'construction' in str(e.get('event_details_level_1', '')).lower()
            )
            context_parts.append(
                f"\nRecent Events Sample: {len(events_context)} records analysed "
                f"({accident_count} accidents, {construction_count} construction). Showing top 5:"
            )
            for i, event in enumerate(events_context[:5], 1):
                event_type = normalize_text(event.get('event_details_level_1'))
                road = normalize_text(event.get('road'), "Unknown road")
                county = normalize_text(event.get('county'))
                duration = parse_minutes(event.get('overall_event_time'))
                duration_text = f" | Duration: {duration:.1f} min" if duration else ""
                context_parts.append(f"  {i}. {event_type} on {road}, {county}{duration_text}")
        
        # Add weather context (ALL LEVELS)
        if weather_data:
            rainy_periods = sum(1 for w in weather_data if w['precip'] and w['precip'] > 0)
            temps = [w['temp'] for w in weather_data if w['temp']]
            avg_temp = sum(temps) / len(temps) if temps else None
            context_parts.append(f"\nWeather Context: {len(weather_data)} periods analyzed")
            if avg_temp:
                context_parts.append(f"  - Average temperature: {avg_temp:.0f}°F")
            if rainy_periods:
                total_precip = sum(w['precip'] for w in weather_data if w['precip'])
                context_parts.append(f"  - Precipitation events: {rainy_periods} periods, total {total_precip:.2f} inches")
            else:
                context_parts.append(f"  - No significant precipitation during period")
        
        # Add news context (ALL LEVELS, but formatted differently for analyst)
        if news_articles and detail_level != "analyst":
            context_parts.append("\nNews Signals:")
            for article in news_articles:
                published = article.get("published_date")
                if isinstance(published, datetime):
                    date_str = published.strftime("%Y-%m-%d")
                else:
                    date_str = str(published)[:10] if published else "Unknown date"
                title = article.get("title", "Untitled")
                summary = article.get("summary", "")
                source = article.get("source", "")
                context_parts.append(f"  - {date_str} | {source}: {title} — {summary[:160]}")
        
        # Analyst-specific deep context
        if detail_level == "analyst" and events_context:
            from collections import Counter
            
            event_type_counter = Counter(
                normalize_text(e.get('event_details_level_1'))
                for e in events_context
            )
            context_parts.append("\nEvent Type Breakdown (top 5):")
            for event_type, count in event_type_counter.most_common(5):
                context_parts.append(f"  - {event_type}: {count} incidents")
            
            construction_events = [
                e for e in events_context
                if 'construction' in str(e.get('event_details_level_1', '')).lower()
            ]
            if construction_events:
                county_counter = Counter(
                    normalize_text(e.get('county'))
                    for e in construction_events
                )
                context_parts.append("\nConstruction Hotspots (by county):")
                for county, count in county_counter.most_common(5):
                    if county == "Unknown":
                        continue
                    roads_counter = Counter(
                        normalize_text(e.get('road'), "Unknown road")
                        for e in construction_events
                        if normalize_text(e.get('county')) == county
                    )
                    top_roads = ", ".join(
                        f"{road} ({road_count})"
                        for road, road_count in roads_counter.most_common(3)
                    )
                    avg_duration_values = [
                        parse_minutes(e.get('overall_event_time'))
                        for e in construction_events
                        if normalize_text(e.get('county')) == county
                    ]
                    avg_duration_values = [v for v in avg_duration_values if v]
                    duration_note = (
                        f" | Avg duration: {sum(avg_duration_values)/len(avg_duration_values):.1f} min"
                        if avg_duration_values else ""
                    )
                    context_parts.append(
                        f"  - {county}: {count} construction incidents | Roads: {top_roads}{duration_note}"
                    )
            
            # Highlight other major incident drivers
            accident_events = [
                e for e in events_context
                if 'accident' in str(e.get('event_details_level_1', '')).lower()
            ]
            disabled_vehicle_events = [
                e for e in events_context
                if 'disabled' in str(e.get('event_details_level_1', '')).lower()
            ]
            if accident_events or disabled_vehicle_events:
                context_parts.append("\nIncident Drivers:")
                if accident_events:
                    accident_county = Counter(
                        normalize_text(e.get('county'))
                        for e in accident_events
                    ).most_common(3)
                    context_parts.append(
                        "  - Accidents concentrated in: " +
                        ", ".join(f"{county} ({count})" for county, count in accident_county if county != "Unknown")
                        if accident_county else "  - No accident concentration identified."
                    )
                if disabled_vehicle_events:
                    disabled_county = Counter(
                        normalize_text(e.get('county'))
                        for e in disabled_vehicle_events
                    ).most_common(3)
                    context_parts.append(
                        "  - Disabled vehicles concentrated in: " +
                        ", ".join(f"{county} ({count})" for county, count in disabled_county if county != "Unknown")
                        if disabled_county else "  - No disabled vehicle hotspots identified."
                    )
            
            # Integrate news context when available
            if news_articles:
                context_parts.append("\nRelevant News Signals (recent):")
                shutdown_articles = 0
                economic_articles = 0
                for article in news_articles:
                    text_blob = " ".join([
                        str(article.get("title", "")),
                        str(article.get("summary", "")),
                        str(article.get("content", "")),
                        ",".join(article.get("keywords", []) or [])
                    ]).lower()
                    if any(keyword in text_blob for keyword in ["shutdown", "government", "federal"]):
                        shutdown_articles += 1
                    if any(keyword in text_blob for keyword in ["economy", "economic", "employment", "spending", "inflation", "consumer"]):
                        economic_articles += 1
                
                if shutdown_articles or economic_articles:
                    context_parts.append(
                        f"  - Articles referencing government/policy impacts: {shutdown_articles}, economic factors: {economic_articles}"
                    )
                
                for article in news_articles[:5]:
                    published = article.get("published_date")
                    if isinstance(published, datetime):
                        date_str = published.strftime("%Y-%m-%d")
                    else:
                        date_str = str(published)[:10] if published else "Unknown date"
                    title = article.get("title", "Untitled")
                    summary = article.get("summary", "")
                    source = article.get("source", "")
                    context_parts.append(
                        f"  - {date_str} | {source}: {title} — {summary[:180]}"
                    )
        
        context = "\n".join(context_parts)
        
        # ==== ANALYST LEVEL (DEEP RESEARCH): Deep analysis with caching ====
        if detail_level == "analyst":
            # Generate cache key based on question, region, and timeframe
            import hashlib
            cache_key_str = f"{request.message.lower()}_{request.region}_{start_time.date()}_{end_time.date()}"
            cache_key = f"chat_analyst_{hashlib.md5(cache_key_str.encode()).hexdigest()}"
            
            # Try to get from cache first
            from utils.cache import get_cache
            cache = get_cache()
            cached_response = await cache.get(cache_key)
            
            if cached_response:
                logger.info(f"Returning cached deep research response for: {request.message[:50]}...")
                return {
                    "success": True,
                    "response": cached_response,
                    "metadata": {
                        "persona": detail_level,
                        "region": request.region,
                        "time_period": f"{start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}",
                        "cached": True
                    }
                }
            
            # Not in cache, generate new response
            from langchain_openai import ChatOpenAI
            from langchain.schema import HumanMessage, SystemMessage
            
            config = get_config()
            llm = ChatOpenAI(
                model=config.get("llm", {}).get("model", "gpt-4"),
                temperature=0.6,
                max_tokens=1500  # Reasonable limit for faster responses
            )
            
            system_prompt = """You are a senior traffic analyst with full access to PostgreSQL tables: trips (volumes, forecast, reliability), events (all incidents with types/locations/durations), news_cache (recent traffic/policy/economic headlines), and weather (conditions/precipitation).

RULES:
- Begin with a 4-bullet executive summary using the format "• Heading: insight".
- Answer ANY question comprehensively using ALL available PostgreSQL data:
  • Forecast questions: Use trips actual vs forecast + events disruption + news signals (economy, shutdown, policy)
  • Construction questions: Analyze event_details_level_1 = Construction, group by county/road, recommend scheduling/routing/comms
  • Government/policy questions: Cite news_cache articles with government/shutdown/budget keywords + correlate with trip volumes
  • Behavior/demand questions: Connect trips trends + news economic indicators + weather impacts + event patterns
- Provide exact numbers, percentages, locations, durations, article titles/sources.
- Include a bullet section titled "• Recommendations:" listing 3-4 concrete actions with expected outcomes.
- Follow bullets with 3-6 paragraphs (400-700 words) diving into temporal trends, spatial clusters, weather/news context.
- Use only supplied data from PostgreSQL; no speculation or fabricated information.
- Clean formatting (plain text, no markdown headings)."""
            
            messages = [
                SystemMessage(content=system_prompt + "\n\nAvailable Data:\n" + context),
                HumanMessage(content=request.message)
            ]
            
            response = await llm.ainvoke(messages)
            response_text = response.content
            
            # Clean up formatting
            response_text = response_text.replace("**", "").replace("##", "").replace("###", "")
            response_text = response_text.replace("*", "•")
            
            # Cache the response for 2 hours (7200 seconds)
            await cache.set(cache_key, response_text, ttl=7200)
            logger.info(f"Cached deep research response for: {request.message[:50]}...")
        
        # ==== EXECUTIVE LEVEL (QUICK INSIGHTS): Fast, concise answers ====
        elif detail_level == "executive":
            from langchain_openai import ChatOpenAI
            from langchain.schema import HumanMessage, SystemMessage
            
            config = get_config()
            llm = ChatOpenAI(
                model=config.get("llm", {}).get("model", "gpt-4"),
                temperature=0.3,  # Lower for more focused responses
                max_tokens=400  # Keep it concise
            )
            
            system_prompt = """You are a traffic analyst for executives with access to PostgreSQL: trips (volumes, forecast, reliability), events (incidents/types/locations), news_cache (policy/economic headlines), weather (conditions/precip).

RULES:
- Answer ANY question using ALL available PostgreSQL data sources.
- If simple stat: state it directly in one sentence with context.
- Otherwise use 3-4 bullet points (format: "• Statement") highlighting business impact, key metrics, action.
- Answer forecast questions using trips actual/forecast gap + events + news (economy, shutdown, policy).
- Answer construction questions using events with Construction type, grouped by location.
- Answer policy questions (government shutdown, economy) citing news_cache articles + trip correlations.
- Prioritize clarity and direct answers grounded in supplied PostgreSQL data only.
- No markdown besides bullets (• ). No speculation.

Example:
• Ops alert: 45 accidents today (+15%) concentrated on I-495; reliability dropped 2.1pts to 84.3%.
• News signals government telework policy driving -8% volume vs forecast; 3 shutdown articles cached.
• Mitigate: Dispatch towing to Tysons; monitor disabled vehicle spikes near construction zones."""
            
            messages = [
                SystemMessage(content=system_prompt + "\n\nAvailable Data:\n" + context),
                HumanMessage(content=request.message)
            ]
            
            response = await llm.ainvoke(messages)
            response_text = response.content
            
            # Clean up formatting
            response_text = response_text.replace("**", "").replace("*", "•")
        
        # ==== MANAGER LEVEL (HIGH-LEVEL): Balanced overview with comprehensive thoughts ====
        else:  # manager
            from langchain_openai import ChatOpenAI
            from langchain.schema import HumanMessage, SystemMessage
            
            config = get_config()
            llm = ChatOpenAI(
                model=config.get("llm", {}).get("model", "gpt-4"),
                temperature=0.5,
                max_tokens=1200  # Allows up to ~8 paragraphs
            )
            
            system_prompt = """You are a traffic analyst for managers with access to PostgreSQL: trips (volumes, forecast, reliability), events (incidents/types/locations/durations), news_cache (traffic/policy/economic articles), weather (conditions/precip).

RULES:
- Answer ANY question using ALL available PostgreSQL data sources comprehensively.
- If simple stat: answer directly in 1-2 sentences with context.
- Otherwise structure as:
  • "• Snapshot:" bullet summarizing the headline metric shift.
  • "• Drivers:" bullet listing 2-3 causal factors (events, weather, news indicators, demand patterns).
  • "• Actions:" bullet with 2-3 operational recommendations.
  • Follow with 2-5 short paragraphs expanding on bullets (current state, causes, impacts, next steps).
- For forecast gaps: Use trips actual vs forecast + events disruption + news_cache (economy, shutdown, policy).
- For construction: Analyze events where event_details_level_1 = Construction, group by county/road, recommend mitigation.
- For policy/shutdown questions: Cite news_cache articles with specific titles/sources + correlate with trip volumes.
- Use only provided PostgreSQL data; no speculation.
- Tone: professional but conversational. Include precise numbers, roads, counties, timing, article citations.
- Maximum: 8 short paragraphs (400-600 words)."""
            
            messages = [
                SystemMessage(content=system_prompt + "\n\nAvailable Data:\n" + context),
                HumanMessage(content=request.message)
            ]
            
            response = await llm.ainvoke(messages)
            response_text = response.content
            
            # Clean up formatting
            response_text = response_text.replace("**", "").replace("##", "").replace("###", "")
        
        sources_count = 1 + len(events_context) + len(news_articles) + len(weather_data)  # Metrics + events + news + weather
        return {
            "success": True,
            "response": response_text,
            "metadata": {
                "persona": detail_level,
                "region": request.region,
                "time_period": f"{start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}",
                "sources": sources_count
            }
        }
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "response": f"I apologize, but I encountered an error processing your request. Please try rephrasing your question.",
            "error": str(e)
        }


# ============================================================================
# AUSTIN HOTSPOT PREDICTION ENDPOINTS
# ============================================================================

class AustinPredictionRequest(BaseModel):
    """Request model for Austin hotspot predictions."""
    target_hour: Optional[str] = None  # ISO datetime string
    include_weather: bool = True


class AustinSectorRequest(BaseModel):
    """Request model for sector queries."""
    sector_id: Optional[int] = None
    sector_code: Optional[str] = None


@app.post("/api/austin/predict")
async def get_austin_predictions(request: AustinPredictionRequest):
    """
    Get risk predictions for all Austin grid sectors.

    Returns risk scores (0-1) and deployment recommendations
    for the target hour (default: next hour).

    Response includes:
    - predictions: List of 100 sector predictions with risk_score, risk_level
    - recommendations: Top sectors needing tow truck staging
    - narrative: Dispatch briefing text
    - weather: Current weather conditions
    - summary: Aggregate statistics
    """
    try:
        from agents.hotspot import HotspotPredictionAgent
        from data.austin_ingestion import AustinDataIngestion
        from data.austin_queries import AustinQueries

        # Determine target hour
        if request.target_hour:
            target_hour = datetime.fromisoformat(request.target_hour)
        else:
            target_hour = datetime.now() + timedelta(hours=1)
            target_hour = target_hour.replace(minute=0, second=0, microsecond=0)

        # Fetch weather data if requested
        weather_data = {}
        if request.include_weather:
            try:
                ingestion = AustinDataIngestion()
                weather = await ingestion.fetch_noaa_weather()
                periods = weather.get("periods", [])
                if periods:
                    # Find the period closest to target hour
                    for period in periods:
                        period_dt = period.get("datetime")
                        if period_dt and period_dt >= target_hour:
                            weather_data = period
                            break
                    if not weather_data and periods:
                        weather_data = periods[0]
            except Exception as e:
                logger.warning(f"Could not fetch weather: {e}")

        # Run hotspot prediction
        # TODO (TQP): Integrate real LLM later 
        agent = HotspotPredictionAgent(llm=None)
        state = {
            "target_hour": target_hour,
            "weather_data": weather_data,
            "errors": []
        }

        # NOTE (TQO) - process is the bulk of the prediction logic
        result = await agent.process(state)

        # Build summary statistics
        predictions = result.get("risk_predictions", [])
        recommendations = result.get("deployment_recommendations", [])

        # Enrich predictions with grid bounds/indices for mapping
        grid_metadata = {}
        sector_lookup = {}
        try:
            queries = AustinQueries()
            sectors = await queries.get_all_sectors()
            sector_lookup = {sector["id"]: sector for sector in sectors}

            config = get_config()
            grid_config = config.get("austin", {}).get("grid", {})
            bounds = grid_config.get("bounds", {})
            grid_metadata = {
                "rows": grid_config.get("rows", 10),
                "cols": grid_config.get("cols", 10),
                "total_sectors": grid_config.get("total_sectors", len(sectors)),
                "bounds": {
                    "north": float(bounds.get("north", 30.5167)),
                    "south": float(bounds.get("south", 30.0833)),
                    "east": float(bounds.get("east", -97.5833)),
                    "west": float(bounds.get("west", -97.9167)),
                }
            }
        except Exception as e:
            logger.warning(f"Could not load sector metadata: {e}")

        if sector_lookup:
            for prediction in predictions:
                sector = sector_lookup.get(prediction.get("sector_id"))
                if not sector:
                    continue

                prediction["north_lat"] = float(sector.get("north_lat", 0))
                prediction["south_lat"] = float(sector.get("south_lat", 0))
                prediction["east_lon"] = float(sector.get("east_lon", 0))
                prediction["west_lon"] = float(sector.get("west_lon", 0))
                prediction["row_idx"] = int(sector.get("row_idx")) if sector.get("row_idx") is not None else None
                prediction["col_idx"] = int(sector.get("col_idx")) if sector.get("col_idx") is not None else None

        high_risk_count = len([p for p in predictions
                               if p.get("risk_level") in ["CRITICAL", "HIGH"]])
        medium_risk_count = len([p for p in predictions
                                 if p.get("risk_level") == "MEDIUM"])
        total_units = sum(r.get("recommended_units", 0) for r in recommendations)

        return {
            "success": True,
            "target_hour": target_hour.isoformat(),
            "predictions": predictions,
            "recommendations": recommendations,
            "narrative": result.get("prediction_narrative", ""),
            "weather": weather_data,
            "weather_factor": result.get("weather_factor", 1.0),
            "grid": grid_metadata,
            "summary": {
                "total_sectors": len(predictions),
                "high_risk_count": high_risk_count,
                "medium_risk_count": medium_risk_count,
                "low_risk_count": len(predictions) - high_risk_count - medium_risk_count,
                "total_units_recommended": total_units,
                "avg_risk_score": sum(p.get("risk_score", 0) for p in predictions) / len(predictions) if predictions else 0
            },
            "errors": result.get("errors", [])
        }

    except Exception as e:
        logger.error(f"Austin prediction error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/austin/grid")
async def get_austin_grid():
    """
    Get all Austin grid sectors with geometry.

    Returns the 100-sector grid covering Austin with
    bounding coordinates and center points for mapping.
    """
    try:
        from data.austin_queries import AustinQueries

        queries = AustinQueries()
        sectors = await queries.get_all_sectors()

        # Get config for grid metadata
        config = get_config()
        grid_config = config.get("austin", {}).get("grid", {})

        return {
            "success": True,
            "sectors": sectors,
            "count": len(sectors),
            "grid_config": {
                "rows": grid_config.get("rows", 10),
                "cols": grid_config.get("cols", 10),
                "bounds": grid_config.get("bounds", {}),
                "total_sectors": grid_config.get("total_sectors", 100)
            },
            "center": config.get("austin", {}).get("center", {
                "lat": 30.2672,
                "lon": -97.7431
            })
        }

    except Exception as e:
        logger.error(f"Austin grid error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/austin/live-incidents")
async def get_austin_live_incidents(
    hours_back: int = 1,
    sector_id: Optional[int] = None,
    issue_type: Optional[str] = None,
    limit: int = 100
):
    """
    Get recent live incidents from Austin.

    Query Parameters:
    - hours_back: How many hours of history (default: 1)
    - sector_id: Filter by specific sector ID
    - issue_type: Filter by issue type (partial match, e.g., "CRASH")
    - limit: Maximum records to return (default: 100)
    """
    try:
        from data.austin_queries import AustinQueries

        queries = AustinQueries()
        incidents = await queries.get_live_incidents(
            hours_back=hours_back,
            sector_id=sector_id,
            issue_type=issue_type,
            limit=limit
        )

        return {
            "success": True,
            "incidents": incidents,
            "count": len(incidents),
            "query": {
                "hours_back": hours_back,
                "sector_id": sector_id,
                "issue_type": issue_type,
                "limit": limit
            }
        }

    except Exception as e:
        logger.error(f"Austin incidents error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/austin/weather")
async def get_austin_weather():
    """
    Get current Austin weather forecast from NOAA.

    Returns the next 24 hours of hourly forecasts.
    """
    try:
        from data.austin_ingestion import AustinDataIngestion

        ingestion = AustinDataIngestion()
        weather = await ingestion.fetch_noaa_weather()

        return {
            "success": True,
            "forecast": weather.get("periods", [])[:24],
            "fetched_at": datetime.now().isoformat(),
            "source": "NOAA Weather API (api.weather.gov)"
        }

    except Exception as e:
        logger.error(f"Austin weather error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/austin/ingest")
async def trigger_austin_ingestion(
    incident_limit: int = 5000,
    include_weather: bool = True
):
    """
    Trigger manual data ingestion cycle.

    Fetches latest incidents from Austin Open Data API and
    weather from NOAA. Useful for testing and manual refreshes.

    Parameters:
    - incident_limit: Maximum incidents to fetch (default: 5000)
    - include_weather: Whether to fetch weather data (default: True)
    """
    try:
        from data.austin_ingestion import AustinDataIngestion

        ingestion = AustinDataIngestion()
        results = await ingestion.run_full_ingestion(
            incident_limit=incident_limit,
            include_weather=include_weather
        )

        return {
            "success": True,
            "ingested": results,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Austin ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/austin/compute-patterns")
async def compute_austin_patterns():
    """
    Compute/update historical patterns from incident data.

    Aggregates incident data by sector, hour, and day of week
    to create the pattern cache used for predictions.

    Should be run periodically (e.g., daily) to refresh patterns.
    """
    try:
        from data.austin_queries import AustinQueries

        queries = AustinQueries()
        count = await queries.compute_historical_patterns()

        return {
            "success": True,
            "patterns_updated": count,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Pattern computation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/austin/dashboard-stats")
async def get_austin_dashboard_stats():
    """
    Get summary statistics for Austin dashboard display.

    Returns:
    - Incident counts (total, last hour, last 24h)
    - Active sectors with incidents
    - Current prediction summary
    """
    try:
        from data.austin_queries import AustinQueries

        queries = AustinQueries()
        stats = await queries.get_dashboard_stats()

        return {
            "success": True,
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Dashboard stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/austin/hotspots")
async def get_austin_hotspots(
    hours_back: int = 24,
    min_incidents: int = 3
):
    """
    Get current hotspot sectors based on recent incidents.

    Parameters:
    - hours_back: Analysis window in hours (default: 24)
    - min_incidents: Minimum incidents to qualify as hotspot (default: 3)
    """
    try:
        from data.austin_queries import AustinQueries

        queries = AustinQueries()
        start_time = datetime.now() - timedelta(hours=hours_back)
        end_time = datetime.now()

        hotspots = await queries.get_sector_hotspots(
            start_time=start_time,
            end_time=end_time,
            min_incidents=min_incidents
        )

        return {
            "success": True,
            "hotspots": hotspots,
            "count": len(hotspots),
            "analysis_window": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "hours": hours_back
            },
            "threshold": min_incidents
        }

    except Exception as e:
        logger.error(f"Hotspots error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/austin/incident-types")
async def get_austin_incident_types(hours_back: int = 24):
    """
    Get distribution of incident types.

    Parameters:
    - hours_back: Analysis window in hours (default: 24)
    """
    try:
        from data.austin_queries import AustinQueries

        queries = AustinQueries()
        distribution = await queries.get_incident_type_distribution(hours_back=hours_back)

        return {
            "success": True,
            "distribution": distribution,
            "total_incidents": sum(distribution.values()),
            "hours_analyzed": hours_back
        }

    except Exception as e:
        logger.error(f"Incident types error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in meters between two lat/lon points."""
    if None in [lat1, lon1, lat2, lon2]:
        return math.inf
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def _classify_risk(score: float) -> str:
    if score >= 0.9:
        return "CRITICAL"
    if score >= 0.7:
        return "HIGH"
    if score >= 0.5:
        return "MEDIUM"
    if score >= 0.3:
        return "LOW"
    return "MINIMAL"


def _prepare_segment_geometry(segment: Dict[str, Any]) -> bool:
    geometry = segment.get("geometry")
    if isinstance(geometry, str):
        try:
            geometry = json.loads(geometry)
        except json.JSONDecodeError:
            return False
    if not isinstance(geometry, dict):
        return False

    coords = geometry.get("coordinates")
    if not coords:
        return False

    if geometry.get("type") == "LineString":
        lines = [coords]
    else:
        lines = coords if isinstance(coords, list) else []

    flattened = []
    for line in lines:
        if isinstance(line, list):
            flattened.extend([pt for pt in line if isinstance(pt, list) and len(pt) == 2])

    if not flattened:
        return False

    lats = [pt[1] for pt in flattened]
    lons = [pt[0] for pt in flattened]
    segment["_center_lat"] = sum(lats) / len(lats)
    segment["_center_lon"] = sum(lons) / len(lons)
    segment["_bbox"] = (
        min(lats),
        max(lats),
        min(lons),
        max(lons)
    )
    segment["_incidents"] = []
    segment["_hist_incidents"] = []
    segment["risk_score"] = 0.0
    segment["risk_level"] = "MINIMAL"
    segment["recent_incident_count"] = 0
    segment["recent_incidents"] = []
    return True


def _annotate_segment_risk(
    segments: List[Dict[str, Any]],
    recent_incidents: List[Dict[str, Any]],
    recent_hours: int,
    historical_incidents: Optional[List[Dict[str, Any]]] = None,
    historical_days: int = 365,
    distance_threshold_meters: float = 1609.0,
    max_assignments: int = 3
) -> None:
    if not segments:
        return

    prepared_segments = [seg for seg in segments if _prepare_segment_geometry(seg)]
    if not prepared_segments:
        return

    bbox_padding = max(distance_threshold_meters / 111000.0, 0.02)  # approx degrees (~2km)

    def assign_incident_to_segments(incident: Dict[str, Any], key: str) -> None:
        lat = incident.get("latitude")
        lon = incident.get("longitude")
        if lat is None or lon is None:
            return
        try:
            lat = float(lat)
            lon = float(lon)
        except (TypeError, ValueError):
            return

        candidates: List[Tuple[float, Dict[str, Any]]] = []
        for seg in prepared_segments:
            min_lat, max_lat, min_lon, max_lon = seg["_bbox"]
            if lat < min_lat - bbox_padding or lat > max_lat + bbox_padding:
                continue
            if lon < min_lon - bbox_padding or lon > max_lon + bbox_padding:
                continue
            dist = _haversine_distance_m(lat, lon, seg["_center_lat"], seg["_center_lon"])
            if dist <= distance_threshold_meters:
                candidates.append((dist, seg))

        if not candidates:
            # Fallback to closest segment overall
            closest = min(
                prepared_segments,
                key=lambda seg: _haversine_distance_m(lat, lon, seg["_center_lat"], seg["_center_lon"]),
            )
            candidates = [(_haversine_distance_m(lat, lon, closest["_center_lat"], closest["_center_lon"]), closest)]

        candidates.sort(key=lambda item: item[0])
        for dist, seg in candidates[:max_assignments]:
            if key == "_hist_incidents":
                seg[key].append(incident)
            else:
                seg[key].append({
                    "id": incident.get("id"),
                    "issue_reported": incident.get("issue_reported"),
                    "published_date": incident.get("published_date"),
                    "distance_m": round(dist, 1)
                })

    for incident in historical_incidents or []:
        assign_incident_to_segments(incident, "_hist_incidents")

    for incident in recent_incidents or []:
        assign_incident_to_segments(incident, "_incidents")

    max_recent_incidents = 5.0
    historical_days = max(1, historical_days)
    baseline_target = 0.5  # incidents per day representing high risk

    for seg in prepared_segments:
        incidents_for_segment = seg.pop("_incidents", [])
        hist_incidents = seg.pop("_hist_incidents", [])
        seg["recent_incident_count"] = len(incidents_for_segment)
        seg["recent_incidents"] = incidents_for_segment[:5]
        hist_count = len(hist_incidents)
        hist_avg_daily = hist_count / historical_days
        seg["historical_avg_daily_incidents"] = round(hist_avg_daily, 3)

        baseline = min(1.0, hist_avg_daily / baseline_target)
        recent_boost = min(0.5, len(incidents_for_segment) / max_recent_incidents) if incidents_for_segment else 0.0
        score = min(1.0, baseline + recent_boost)
        seg["risk_score"] = round(score, 3)
        seg["risk_level"] = _classify_risk(score)
        seg.pop("_center_lat", None)
        seg.pop("_center_lon", None)
        seg.pop("_bbox", None)


@app.get("/api/austin/road-segments")
async def get_austin_road_segments(
    limit: int = 500,
    priority_network: Optional[str] = None,
    council_district: Optional[str] = None,
    street_level: Optional[int] = None,
    min_lanes: Optional[int] = None,
    min_street_level: Optional[int] = None
):
    """
    Fetch roadway segments from the Austin Strategic Mobility Plan network.

    Query Parameters:
    - limit: Max number of segments to return (default: 500)
    - priority_network: Filter by priority network (e.g., "Bicycle Priority")
    - council_district: Filter by city council district (string match)
    - street_level: Filter by ASMP street level (1-5)
    - min_lanes: Filter for segments with at least this many lanes (existing or future)
    """
    try:
        from data.austin_queries import AustinQueries

        limit = max(1, min(limit, 50000))
        queries = AustinQueries()
        segments = await queries.get_road_segments(
            limit=limit,
            priority_network=priority_network,
            council_district=council_district,
            street_level=street_level,
            min_street_level=min_street_level,
            min_lanes=min_lanes
        )

        analysis_hours = 24
        analysis_days = 365
        recent_incidents = await queries.get_incidents_within_hours(hours_back=analysis_hours, limit=8000)
        historical_incidents = await queries.get_incidents_since_days(days_back=analysis_days, limit=50000)
        _annotate_segment_risk(
            segments,
            recent_incidents,
            analysis_hours,
            historical_incidents=historical_incidents,
            historical_days=analysis_days
        )

        return {
            "success": True,
            "segments": segments,
            "count": len(segments),
            "filters": {
                "limit": limit,
                "priority_network": priority_network,
                "council_district": council_district,
                "street_level": street_level,
                "min_street_level": min_street_level,
                "min_lanes": min_lanes
            },
            "analysis": {
                "hours_back": analysis_hours,
                "incident_sample": len(recent_incidents),
                "historical_days": analysis_days,
                "historical_sample": len(historical_incidents)
            }
        }

    except Exception as e:
        logger.error(f"Road segments error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )