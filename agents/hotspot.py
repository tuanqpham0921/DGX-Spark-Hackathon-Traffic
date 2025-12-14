"""
Hotspot Prediction Agent

The core intelligence of the Austin Traffic Hotspot system.
Learns the "Rhythm of the City" and predicts high-risk zones
with hourly risk scores for proactive tow truck staging.

Key Capabilities:
1. Historical pattern analysis (time-of-day, day-of-week)
2. Weather correlation (rain/ice/heat impact multipliers)
3. Grid-based risk scoring (100 sectors)
4. Deployment recommendations (where to stage units)
5. Narrative explanations (dispatch briefings)

Usage:
    agent = HotspotPredictionAgent()
    state = {
        "target_hour": datetime.now() + timedelta(hours=1),
        "weather_data": {"temp": 75, "precip_probability": 20, ...},
        "errors": []
    }
    result = await agent.process(state)
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.base import BaseAgent
from data.austin_queries import AustinQueries
from utils.config_loader import get_config

logger = logging.getLogger(__name__)


class HotspotPredictionAgent(BaseAgent):
    """
    Predicts high-risk traffic zones and generates deployment recommendations.

    The agent divides Austin into a 10x10 grid (100 sectors) and assigns
    a risk score (0.0-1.0) to each sector for a target hour.

    Risk calculation combines:
    - Historical incident patterns for that time slot
    - Current/forecast weather conditions
    - Time-of-day factors (rush hour, overnight, etc.)

    The agent outputs:
    - risk_predictions: List of all 100 sector predictions
    - deployment_recommendations: Top 10 sectors needing tow truck staging
    - prediction_narrative: Human-readable dispatch briefing
    """

    def __init__(self, llm: None):
        """
        Initialize the Hotspot Prediction Agent.

        Args:
            llm: Optional pre-configured LLM. If not provided,
                 creates one with low temperature (0.2) for consistent predictions.
        """
        # TODO (TQP): Pass LLM in later
        super().__init__(
            name="Hotspot Agent",
            role="Predictive Risk Analyst",
            llm=llm,
            temperature=0.2  # Low temperature for consistent predictions
        )

        # Load configuration
        self.config = get_config()
        self.austin_config = self.config.get("austin", {})
        self.grid_config = self.austin_config.get("grid", {})

        # Risk thresholds
        self.risk_thresholds = self.austin_config.get("risk_thresholds", {
            "minimal": 0.0,
            "low": 0.3,
            "medium": 0.6,
            "high": 0.8,
            "critical": 0.95
        })

        # Weather multipliers from config
        self.weather_multipliers = self.austin_config.get("weather_multipliers", {
            "rain_probability_high": 1.4,
            "active_precipitation": 1.3,
            "heavy_precipitation": 1.5,
            "freezing": 3.0,
            "extreme_heat": 1.5,
            "high_wind": 1.3,
            "low_visibility": 1.5,
            "max_multiplier": 5.0
        })

        # Time factors from config
        self.time_factors = self.austin_config.get("time_factors", {
            "rush_hour_peak": 1.5,
            "rush_hour_shoulder": 1.25,
            "overnight": 0.5,
            "late_night": 0.7,
            "midday": 1.0
        })

        # Deployment settings
        self.deployment_config = self.austin_config.get("deployment", {
            "max_recommendations": 10,
            "critical_units": 2,
            "high_units": 1
        })

        # Initialize query helper
        self.austin_queries = AustinQueries()

        logger.info("HotspotPredictionAgent initialized")


    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate risk predictions for the target hour.

        This is the main entry point for the agent. It orchestrates:
        1. Retrieving historical patterns
        2. Calculating weather impact
        3. Computing risk scores
        4. Generating recommendations
        5. Creating dispatch narrative

        Args:
            state: Workflow state containing:
                - target_hour: datetime for prediction (default: next hour)
                - weather_data: dict with temp, precip, conditions, etc.
                - errors: list to append any errors

        Returns:
            Updated state with:
                - risk_predictions: List of sector risk scores
                - deployment_recommendations: Top sectors needing units
                - prediction_narrative: Dispatch briefing text
                - target_hour: Normalized target hour
        """
        # Default to next hour if not specified
        target_hour = state.get("target_hour")
        if not target_hour:
            target_hour = datetime.now() + timedelta(hours=1)

        # Normalize to top of hour
        target_hour = target_hour.replace(minute=0, second=0, microsecond=0)

        weather_data = state.get("weather_data", {})

        logger.info(f"Hotspot Agent predicting for: {target_hour}")
        logger.info(f"Weather data available: {bool(weather_data)}")

        try:
            # Step 1: Get historical patterns for this time slot
            patterns = await self._get_historical_patterns(
                hour=target_hour.hour,
                day_of_week=target_hour.weekday()
            )
            logger.info(f"Retrieved patterns for {len(patterns)} sectors")

            # Step 2: Calculate weather risk factor
            weather_factor = self._calculate_weather_factor(weather_data)
            logger.info(f"Weather factor: {weather_factor:.2f}x")

            # Step 3: Compute risk scores for each sector
            risk_scores = await self._compute_sector_risks(
                patterns=patterns,
                weather_factor=weather_factor,
                target_hour=target_hour
            )
            logger.info(f"Computed risk scores for {len(risk_scores)} sectors")

            # Step 4: Generate deployment recommendations
            recommendations = self._generate_deployment_recommendations(
                risk_scores=risk_scores,
                target_hour=target_hour
            )
            logger.info(f"Generated {len(recommendations)} deployment recommendations")

            # Step 5: Generate narrative explanation
            # TODO (TQP): Mkae a narrative call
            # narrative = "Narrative generation not implemented yet."
            narrative = await self._generate_prediction_narrative(
                risk_scores=risk_scores,
                recommendations=recommendations,
                weather_data=weather_data,
                target_hour=target_hour
            )

            # Step 6: Save predictions to database
            try:
                await self.austin_queries.save_risk_predictions(
                    predictions=risk_scores,
                    target_hour=target_hour
                )
            except Exception as e:
                logger.warning(f"Could not save predictions to DB: {e}")

            # Update state with results
            state["risk_predictions"] = risk_scores
            state["deployment_recommendations"] = recommendations
            state["prediction_narrative"] = narrative
            state["target_hour"] = target_hour.isoformat()
            state["weather_factor"] = weather_factor
            state["next_agent"] = "complete"

            # Log summary
            high_risk_count = len([s for s in risk_scores
                                   if s["risk_level"] in ["CRITICAL", "HIGH"]])
            logger.info(f"Prediction complete: {high_risk_count} high-risk sectors")

        except Exception as e:
            logger.error(f"Hotspot prediction failed: {e}", exc_info=True)
            state["errors"] = state.get("errors", [])
            state["errors"].append(f"HotspotAgent: {str(e)}")
            state["next_agent"] = "complete"

        return state

    async def _get_historical_patterns(
        self,
        hour: int,
        day_of_week: int
    ) -> Dict[str, Any]:
        """
        Retrieve historical incident patterns for the time slot.

        Queries the austin_historical_patterns table for pre-computed
        averages by sector, hour, and day of week.

        Args:
            hour: Hour of day (0-23)
            day_of_week: Day of week (0=Monday, 6=Sunday)

        Returns:
            Dict mapping sector_id (as string) to pattern data including
            avg_incidents, incident_count, and incident_types.
        """
        try:
            patterns = await self.austin_queries.get_historical_patterns(
                hour=hour,
                day_of_week=day_of_week
            )
            return patterns
        except Exception as e:
            logger.error(f"Failed to get historical patterns: {e}")
            return {}

    def _calculate_weather_factor(self, weather_data: Dict) -> float:
        """
        Calculate weather-based risk multiplier.

        Analyzes multiple weather conditions and returns a combined
        multiplier from 1.0 (no impact) to max_multiplier (severe impact).

        Weather impacts (configurable via settings.yaml):
        - Precipitation probability >70%: 1.4x
        - Active precipitation >0.1": 1.3x
        - Heavy precipitation >0.5": 1.5x
        - Freezing (<32°F): 3.0x
        - Extreme heat (>100°F): 1.5x
        - High wind (>30 mph): 1.3x
        - Low visibility (<3 mi): 1.5x

        Args:
            weather_data: Dict with weather metrics

        Returns:
            Combined risk multiplier (1.0 - max_multiplier)
        """
        if not weather_data:
            return 1.0

        factor = 1.0

        # Get weather values with defaults
        precip = weather_data.get("precip", 0) or 0
        precip_prob = weather_data.get("precip_probability", 0) or 0
        temp = weather_data.get("temp", 70) or 70
        wind = weather_data.get("wind_speed", 0) or 0
        visibility = weather_data.get("visibility", 10) or 10

        # Also check conditions string for rain/storm keywords
        conditions = (weather_data.get("conditions", "") or "").lower()

        # Precipitation impact
        if precip_prob > 70:
            factor *= self.weather_multipliers.get("rain_probability_high", 1.4)
            logger.debug(f"High precip probability ({precip_prob}%): applying multiplier")

        if precip > 0.1:
            factor *= self.weather_multipliers.get("active_precipitation", 1.3)
            logger.debug(f"Active precipitation ({precip}\"): applying multiplier")

        if precip > 0.5:
            factor *= self.weather_multipliers.get("heavy_precipitation", 1.5)
            logger.debug(f"Heavy precipitation ({precip}\"): applying multiplier")

        # Check conditions for rain/storm keywords
        rain_keywords = ["rain", "storm", "thunder", "shower"]
        if any(keyword in conditions for keyword in rain_keywords):
            if precip_prob <= 70 and precip <= 0.1:  # Don't double-count
                factor *= 1.2
                logger.debug(f"Rain conditions detected in forecast: {conditions}")

        # Temperature impact
        if temp < 32:
            factor *= self.weather_multipliers.get("freezing", 3.0)
            logger.debug(f"Freezing temperature ({temp}°F): applying 3x multiplier")
        elif temp > 100:
            factor *= self.weather_multipliers.get("extreme_heat", 1.5)
            logger.debug(f"Extreme heat ({temp}°F): applying multiplier")

        # Wind impact
        if wind > 30:
            factor *= self.weather_multipliers.get("high_wind", 1.3)
            logger.debug(f"High wind ({wind} mph): applying multiplier")

        # Visibility impact
        if visibility < 3:
            factor *= self.weather_multipliers.get("low_visibility", 1.5)
            logger.debug(f"Low visibility ({visibility} mi): applying multiplier")

        # Cap at maximum multiplier
        max_mult = self.weather_multipliers.get("max_multiplier", 5.0)
        return min(factor, max_mult)

    async def _compute_sector_risks(
        self,
        patterns: Dict[str, Any],
        weather_factor: float,
        target_hour: datetime
    ) -> List[Dict[str, Any]]:
        """
        Compute risk score for each of the 100 grid sectors.

        Risk formula:
        1. Base risk from historical avg_incidents (normalized to 0-1)
        2. Multiply by weather factor
        3. Multiply by time-of-day factor
        4. Normalize final score to 0-1

        Args:
            patterns: Historical patterns by sector
            weather_factor: Weather risk multiplier
            target_hour: Target prediction hour

        Returns:
            List of risk prediction dicts, sorted by risk_score descending
        """
        # Get all sectors from database
        sectors = await self.austin_queries.get_all_sectors()

        if not sectors:
            logger.warning("No sectors found - using default grid")
            sectors = self._get_default_sectors()

        risk_scores = []

        for sector in sectors:
            sector_id = sector["id"]
            sector_code = sector.get("sector_code", f"S{sector_id}")

            # Get sector-specific historical data
            sector_pattern = patterns.get(str(sector_id), {})

            # Base risk from historical incidents
            # Normalize: assume >10 avg incidents/hour = max base risk
            avg_incidents = sector_pattern.get("avg_incidents", 0)
            base_risk = min(avg_incidents / 10.0, 1.0)

            # Apply weather factor
            adjusted_risk = base_risk * weather_factor

            # Apply time-of-day factor
            hour_factor = self._get_hour_factor(target_hour.hour)
            adjusted_risk *= hour_factor

            # If no historical data, use a small baseline risk
            if not sector_pattern:
                # Small baseline risk + weather effect
                adjusted_risk = 0.1 * weather_factor * hour_factor

            # Normalize to 0-1
            final_risk = min(adjusted_risk, 1.0)

            # Calculate confidence based on sample size
            confidence = self._calculate_confidence(sector_pattern)

            risk_scores.append({
                "sector_id": sector_id,
                "sector_code": sector_code,
                "center_lat": float(sector.get("center_lat", 0)),
                "center_lon": float(sector.get("center_lon", 0)),
                "risk_score": round(final_risk, 4),
                "risk_level": self._classify_risk(final_risk),
                "historical_factor": round(base_risk, 4),
                "weather_factor": round(weather_factor, 2),
                "time_factor": round(hour_factor, 2),
                "confidence": round(confidence, 2),
                "avg_incidents": round(avg_incidents, 2),
                "sample_size": sector_pattern.get("sample_size", 0),
                "incident_types": sector_pattern.get("incident_types", {})
            })

        # Sort by risk score descending
        risk_scores.sort(key=lambda x: x["risk_score"], reverse=True)

        return risk_scores

    def _get_hour_factor(self, hour: int) -> float:
        """
        Get time-of-day risk multiplier.

        Rush hours have highest risk, overnight has lowest.

        Hour ranges:
        - 7-8 AM, 5-6 PM: Rush hour peak (1.5x)
        - 6, 9 AM, 4, 7 PM: Rush hour shoulders (1.25x)
        - 12-5 AM: Overnight (0.5x)
        - 10 PM - midnight: Late night (0.7x)
        - Other hours: Midday (1.0x)

        Args:
            hour: Hour of day (0-23)

        Returns:
            Risk multiplier (0.5 - 1.5)
        """
        # Morning rush peak: 7-8 AM
        if hour in [7, 8]:
            return self.time_factors.get("rush_hour_peak", 1.5)
        # Evening rush peak: 5-6 PM (17-18)
        elif hour in [17, 18]:
            return self.time_factors.get("rush_hour_peak", 1.5)
        # Rush hour shoulders
        elif hour in [6, 9, 16, 19]:
            return self.time_factors.get("rush_hour_shoulder", 1.25)
        # Overnight (midnight - 5 AM)
        elif hour in [0, 1, 2, 3, 4, 5]:
            return self.time_factors.get("overnight", 0.5)
        # Late night (10 PM - midnight)
        elif hour in [22, 23]:
            return self.time_factors.get("late_night", 0.7)
        # Midday and early evening
        else:
            return self.time_factors.get("midday", 1.0)

    def _classify_risk(self, score: float) -> str:
        """
        Classify risk level from numeric score.

        Levels (configurable via risk_thresholds):
        - CRITICAL: >= 0.95
        - HIGH: 0.80 - 0.94
        - MEDIUM: 0.60 - 0.79
        - LOW: 0.30 - 0.59
        - MINIMAL: < 0.30

        Args:
            score: Risk score (0-1)

        Returns:
            Risk level string
        """
        if score >= self.risk_thresholds.get("critical", 0.95):
            return "CRITICAL"
        elif score >= self.risk_thresholds.get("high", 0.80):
            return "HIGH"
        elif score >= self.risk_thresholds.get("medium", 0.60):
            return "MEDIUM"
        elif score >= self.risk_thresholds.get("low", 0.30):
            return "LOW"
        else:
            return "MINIMAL"

    def _calculate_confidence(self, pattern: Dict) -> float:
        """
        Calculate prediction confidence based on historical sample size.

        More historical data = higher confidence in predictions.

        Confidence levels:
        - 100+ samples: 95%
        - 50-99 samples: 85%
        - 20-49 samples: 70%
        - 10-19 samples: 55%
        - 5-9 samples: 40%
        - <5 samples: 25%

        Args:
            pattern: Historical pattern dict with sample_size

        Returns:
            Confidence score (0-1)
        """
        sample_size = pattern.get("sample_size", 0)

        if sample_size >= 100:
            return 0.95
        elif sample_size >= 50:
            return 0.85
        elif sample_size >= 20:
            return 0.70
        elif sample_size >= 10:
            return 0.55
        elif sample_size >= 5:
            return 0.40
        else:
            return 0.25  # Low confidence with minimal data

    def _generate_deployment_recommendations(
        self,
        risk_scores: List[Dict],
        target_hour: datetime
    ) -> List[Dict[str, Any]]:
        """
        Generate tow truck deployment recommendations.

        Identifies top high-risk sectors and recommends unit staging
        based on risk level:
        - CRITICAL: 2 units
        - HIGH: 1 unit

        Args:
            risk_scores: List of sector risk predictions
            target_hour: Target hour for deployment

        Returns:
            List of deployment recommendation dicts
        """
        recommendations = []
        max_recs = self.deployment_config.get("max_recommendations", 10)
        critical_units = self.deployment_config.get("critical_units", 2)
        high_units = self.deployment_config.get("high_units", 1)

        # Get sectors with HIGH or CRITICAL risk
        high_risk_sectors = [
            s for s in risk_scores
            if s["risk_level"] in ["CRITICAL", "HIGH"]
        ]

        # Generate recommendations for top sectors
        for i, sector in enumerate(high_risk_sectors[:max_recs]):
            # More units for CRITICAL sectors
            units = critical_units if sector["risk_level"] == "CRITICAL" else high_units

            # Generate rationale
            rationale = self._generate_rationale(sector)

            recommendations.append({
                "priority_rank": i + 1,
                "sector_id": sector["sector_id"],
                "sector_code": sector["sector_code"],
                "center_lat": sector["center_lat"],
                "center_lon": sector["center_lon"],
                "recommended_units": units,
                "risk_score": sector["risk_score"],
                "risk_level": sector["risk_level"],
                "confidence": sector["confidence"],
                "target_period_start": target_hour.isoformat(),
                "target_period_end": (target_hour + timedelta(hours=1)).isoformat(),
                "rationale": rationale
            })

        return recommendations

    def _generate_rationale(self, sector: Dict) -> str:
        """
        Generate human-readable rationale for a deployment recommendation.

        Args:
            sector: Sector prediction dict

        Returns:
            Rationale string
        """
        parts = []

        if sector["avg_incidents"] > 5:
            parts.append(f"historically high incidents ({sector['avg_incidents']:.1f}/hr)")
        elif sector["avg_incidents"] > 2:
            parts.append(f"moderate incident history ({sector['avg_incidents']:.1f}/hr)")

        if sector["weather_factor"] > 2.0:
            parts.append(f"severe weather impact ({sector['weather_factor']:.1f}x)")
        elif sector["weather_factor"] > 1.5:
            parts.append(f"weather impact ({sector['weather_factor']:.1f}x)")

        if sector["time_factor"] > 1.2:
            parts.append("rush hour period")

        if sector["confidence"] < 0.5:
            parts.append("limited historical data")

        if not parts:
            parts.append("elevated baseline risk")

        return f"Sector {sector['sector_code']}: {', '.join(parts)}"

    def _get_default_sectors(self) -> List[Dict[str, Any]]:
        """
        Generate default sector list if database is unavailable.

        Creates a 10x10 grid based on Austin bounds from config.

        Returns:
            List of default sector dicts
        """
        # NOTE (TQP) - default Austin bounds
        bounds = self.grid_config.get("bounds", {})
        north = bounds.get("north", 30.5167)
        south = bounds.get("south", 30.0833)
        east = bounds.get("east", -97.5833)
        west = bounds.get("west", -97.9167)

        rows = self.grid_config.get("rows", 10)
        cols = self.grid_config.get("cols", 10)

        lat_step = (north - south) / rows
        lon_step = (east - west) / cols

        sectors = []
        for row in range(rows):
            for col in range(cols):
                sector_id = row * cols + col + 1
                sector_code = f"{chr(65 + row)}{col + 1}"

                center_lat = north - (row + 0.5) * lat_step
                center_lon = west + (col + 0.5) * lon_step

                sectors.append({
                    "id": sector_id,
                    "sector_code": sector_code,
                    "center_lat": center_lat,
                    "center_lon": center_lon
                })

        return sectors

    # TODO (TQP): Implement narrative generation
    # this is where the LLM is used
    async def _generate_prediction_narrative(
        self,
        risk_scores: List[Dict],
        recommendations: List[Dict],
        weather_data: Dict,
        target_hour: datetime
    ) -> str:
        """
        Generate a dispatch briefing narrative using LLM.

        Creates a 3-4 sentence summary explaining:
        - Overall risk picture
        - Top sectors to watch
        - Weather impacts
        - Recommended actions

        Args:
            risk_scores: All sector predictions
            recommendations: Deployment recommendations
            weather_data: Current weather
            target_hour: Target prediction hour

        Returns:
            Narrative string for dispatch briefing
        """
        # Calculate summary statistics
        high_risk_count = len([s for s in risk_scores
                               if s["risk_level"] in ["CRITICAL", "HIGH"]])
        medium_risk_count = len([s for s in risk_scores
                                 if s["risk_level"] == "MEDIUM"])
        top_sectors = risk_scores[:5]
        total_units = sum(r["recommended_units"] for r in recommendations)

        # Calculate average risk
        avg_risk = sum(s["risk_score"] for s in risk_scores) / len(risk_scores) \
                   if risk_scores else 0

        # Build prompt for narrative generation
        prompt = f"""Generate a brief dispatch briefing for tow truck operators.

TARGET HOUR: {target_hour.strftime('%A, %B %d at %I:%M %p')}

WEATHER CONDITIONS:
- Conditions: {weather_data.get('conditions', 'Clear')}
- Temperature: {weather_data.get('temp', 70)}°F
- Precipitation chance: {weather_data.get('precip_probability', 0)}%
- Wind: {weather_data.get('wind_speed', 0)} mph

RISK SUMMARY:
- High/Critical risk sectors: {high_risk_count} of 100
- Medium risk sectors: {medium_risk_count}
- Average risk score: {avg_risk:.1%}

TOP 5 RISK SECTORS:
{chr(10).join([f"- {s['sector_code']}: {s['risk_score']:.1%} ({s['risk_level']})" for s in top_sectors])}

DEPLOYMENT RECOMMENDATION:
- Total units to deploy: {total_units}
- Priority sectors: {', '.join([r['sector_code'] for r in recommendations[:5]])}

Write a concise 3-4 sentence dispatch briefing that:
1. States the overall risk level for this hour
2. Identifies the key areas to watch (reference specific sector codes)
3. Notes any weather factors affecting risk
4. Gives clear staging guidance

Be direct and actionable. This is for emergency responders.
"""
        import requests

        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": prompt}
        ]

        try:
            response = requests.post(
                # TODO (TQP): make configurable
                "http://localhost:5000/chat", 
                json={"messages": messages, "stream": False},
            )
            response = response.json()["answer"]

            return response
        except Exception as e:
            logger.error(f"Failed to generate narrative: {e}")
            # Fallback to template-based narrative
            return self._generate_fallback_narrative(
                high_risk_count, top_sectors, weather_data, total_units, recommendations
            )

# NOTE (TQP): System prompt for narrative generation
    def get_system_prompt(self) -> str:
        """Get the system prompt for narrative generation."""
        return """You are the Hotspot Prediction Agent for TRAFFIX Austin.

Your mission: Learn the "Rhythm of the City" and predict where traffic incidents
will likely occur in the next hour, enabling proactive tow truck staging.

Your responsibilities:

1. ANALYZE historical incident patterns by:
   - Time of day (rush hour peaks at 7-9 AM and 4-7 PM)
   - Day of week (weekday commute patterns vs weekend)
   - Grid sector (geographic hotspots like I-35, MoPac, downtown)

2. FACTOR IN weather impacts:
   - Rain increases crash risk by 30-50%
   - Ice/freezing conditions multiply risk 3-5x
   - Extreme heat (>100°F) causes vehicle breakdowns
   - Low visibility (<3 miles) increases risk
   - High winds (>30 mph) affect large vehicles

3. GENERATE risk scores (0.0 - 1.0) for each of 100 grid sectors:
   - MINIMAL: < 0.30 (routine monitoring)
   - LOW: 0.30 - 0.59 (standby awareness)
   - MEDIUM: 0.60 - 0.79 (increased readiness)
   - HIGH: 0.80 - 0.94 (active staging recommended)
   - CRITICAL: >= 0.95 (deploy units immediately)

4. PROVIDE deployment recommendations:
   - Which sectors need tow trucks staged
   - How many units per sector (2 for CRITICAL, 1 for HIGH)
   - Optimal positioning rationale

5. EXPLAIN the WHY behind predictions:
   - Reference specific data patterns
   - Cite weather factors
   - Note historical precedents

Always be data-driven and actionable. Emergency responders depend on your accuracy.
"""