# TRAFFIX — The Traffic Insight Agent

An AI-powered storytelling and analytics assistant for transportation analysts, planners, and operations managers in the **Austin Hotspot Prediction** for tow truck staging.

TRAFFIX helps users understand *why* congestion patterns change by combining structured data (Postgres), unstructured data (Tavily API, OpenAI), and agentic reasoning.

## Quick Start (TL;DR)

```bash
# 1. Clone and setup
git clone <repository-url>
cd traffix
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your OPENAI_API_KEY and TAVILY_API_KEY

# 3. Start infrastructure (PostgreSQL + Qdrant)
docker compose up -d postgres qdrant

# 4. Initialize database schema
docker compose exec -T postgres psql -U postgres -d traffix < scripts/init-db.sql

# need for migration
docker compose exec -T postgres psql -U postgres -d traffix < ./data/migrations/002_austin_road_segments.sql

# 5. Ollama Local LLM
ollama serve
# run the local model (auto-reload)
export FLASK_APP=llm/app.py
export FLASK_DEBUG=1
flask run --host 0.0.0.0 --port 5000

# 6. Ingest Austin data (no API keys needed - public APIs)
python data/austin_ingestion.py --limit 5000

# 7. Compute historical patterns
python -c "import asyncio; from data.austin_queries import AustinQueries; asyncio.run(AustinQueries().compute_historical_patterns())"

# 8. Start the API server
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 9. (New terminal) Start the Austin dashboard
streamlit run ui/austin_hotspot_dashboard.py

# Open http://localhost:8501 for dashboard, http://localhost:8000/docs for API
```

## Features

- **Multi-Agent RAG System**: Orchestrated workflow with specialized agents
- **Region-Aware Analysis**: Support for DC, Virginia, and Austin
- **Austin Hotspot Prediction**: Real-time risk scoring for 100 grid sectors
- **Three Operating Modes**:
  - Quick Mode: Daily digest with key insights
  - Deep Mode: Comprehensive multi-section reports
  - PDF Reader: Document analysis and Q&A
- **Data Integration**: PostgreSQL + Qdrant + Tavily API + NOAA Weather
- **Advanced Retrieval**: Multiple strategies (Naive, BM25, Multi-Query, Ensemble)
- **Quality Assurance**: RAGAS evaluation metrics
- **Interactive UI**: Streamlit dashboards + FastAPI backend

## Prerequisites

- Python 3.10+
- Docker & Docker Compose
- API Keys:
  - **OpenAI** (required for LLM)
  - **Tavily** (required for web search)
  - Cohere (optional, for reranking)
  - LangSmith (optional, for tracing)

**Note**: Austin data sources (Austin Open Data Portal, NOAA Weather) are free public APIs - no keys needed.

## Installation

### Step 1: Clone and Setup Python Environment

```bash
git clone <repository-url>
cd traffix

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
# Required
OPENAI_API_KEY=sk-your-key-here
TAVILY_API_KEY=tvly-your-key-here

# Database (defaults work with Docker Compose)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=traffix
DB_USER=postgres
DB_PASSWORD=password

# Optional
QDRANT_HOST=localhost
QDRANT_PORT=6333
LANGSMITH_API_KEY=your-key-here
```

### Step 3: Start Infrastructure with Docker Compose

```bash
# Start PostgreSQL and Qdrant
docker compose up -d postgres qdrant

# Verify containers are running
docker compose ps

# Expected output:
# traffix-postgres   Running (healthy)
# traffix-qdrant     Running (healthy)
```

### Step 4: Initialize Database Schema

```bash
# Apply database migrations
docker compose exec -T postgres psql -U postgres -d traffix < scripts/init-db.sql

# Verify tables created
docker compose exec postgres psql -U postgres -d traffix -c "\dt public.*"
```

### Step 5: Ollma Local LLM
```bash
# Download the local model
ollama serve
ollama pull gpt-oss:20b

# Run the LLM hosting (auto-reload)
export FLASK_APP=llm/app.py
export FLASK_DEBUG=1
flask run --host 0.0.0.0 --port 5000
```

to test out local llm
```bash
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is Python?"}'
```

### Step 6: Ingest Data

**Austin Traffic Data** (free, no API key):
```bash
# Fetch recent incidents (last ~1000)
python data/austin_ingestion.py --limit 1000

# Fetch historical data (recommended for better predictions)
python data/austin_ingestion.py --historical --limit 10000

# Compute historical patterns for predictions
python -c "
import asyncio
from data.austin_queries import AustinQueries
asyncio.run(AustinQueries().compute_historical_patterns())
"
```

## Running the Application

### Option 1: Full Stack (API + Dashboard)

```bash
# Terminal 1: Start FastAPI backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Start Austin Hotspot Dashboard
streamlit run ui/austin_hotspot_dashboard.py
```

- **Dashboard**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs
- **API Health**: http://localhost:8000/api/health

### Option 2: API Only

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Option 3: Streamlit Only (DC/Virginia)

```bash
streamlit run ui/streamlit_app.py
```

## API Endpoints

### Austin Hotspot Prediction

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/austin/predict` | POST | Get risk predictions for all 100 sectors |
| `/api/austin/grid` | GET | Get grid sector geometry |
| `/api/austin/live-incidents` | GET | Get recent incidents |
| `/api/austin/hotspots` | GET | Get current hotspot sectors |
| `/api/austin/weather` | GET | Get NOAA weather forecast |
| `/api/austin/ingest` | POST | Trigger data ingestion |
| `/api/austin/incident-types` | GET | Get incident type distribution |

### Example API Calls

```bash
# Get predictions
curl -X POST http://localhost:8000/api/austin/predict \
  -H "Content-Type: application/json" \
  -d '{"include_weather": true}'

# Get live incidents (last 24 hours)
curl "http://localhost:8000/api/austin/live-incidents?hours_back=24&limit=50"

# Get hotspots
curl "http://localhost:8000/api/austin/hotspots?hours_back=24&min_incidents=3"

# Trigger data refresh
curl -X POST "http://localhost:8000/api/austin/ingest?incident_limit=1000"
```

## Configuration

Risk thresholds and multipliers are in `config/settings.yaml`:

```yaml
austin:
  risk_thresholds:
    minimal: 0.0    # 0-29%
    low: 0.3        # 30-59%
    medium: 0.6     # 60-79%
    high: 0.8       # 80-94%
    critical: 0.95  # 95-100%

  weather_multipliers:
    rain_probability_high: 1.4
    freezing: 3.0
    heavy_precipitation: 1.5

  time_factors:
    rush_hour_peak: 1.5      # 7-8 AM, 5-6 PM
    overnight: 0.5           # 12-5 AM
```

## Docker Commands Reference

```bash
# Start all services
docker compose up -d

# Stop all services
docker compose down

# View logs
docker compose logs -f backend

# Reset database (warning: deletes data)
docker compose down -v
docker compose up -d postgres qdrant
docker compose exec -T postgres psql -U postgres -d traffix < scripts/init-db.sql

# Connect to database
docker compose exec postgres psql -U postgres -d traffix
```

## 📁 Project Structure

```
traffix/
├── agents/              # Multi-agent system
│   ├── supervisor.py    # Task router and orchestrator
│   ├── research.py      # Data analyst
│   ├── writer.py        # Report generator
│   ├── editor.py        # Quality checker
│   └── evaluator.py     # QA and metrics
├── data/                # Database integration
│   ├── db_connection.py # PostgreSQL connection
│   └── queries.py       # SQL queries
├── retrievers/          # RAG retrieval strategies
│   ├── naive.py
│   ├── bm25.py
│   ├── multi_query.py
│   └── ensemble.py
├── ui/                  # User interfaces
│   ├── streamlit_app.py # Streamlit POC
│   └── frontend_v0/     # V0/Vercel demo
├── evaluation/          # Quality metrics
│   └── ragas_eval.py    # RAGAS evaluation
├── config/              # Configuration
│   └── settings.yaml    # App settings
├── utils/               # Utilities
├── workflows/           # LangGraph workflows
└── tests/               # Test cases
```

## 🎯 Usage Examples

### Quick Mode
```python
from workflows.graph import create_workflow

workflow = create_workflow()
result = workflow.invoke({
    "mode": "quick",
    "region": "Virginia",
    "query": "Why was congestion higher yesterday evening?"
})
```

### Deep Mode
```python
result = workflow.invoke({
    "mode": "deep",
    "region": "District of Columbia",
    "query": "Analyze I-395 congestion patterns for the past week"
})
```

## 📊 Data Schema

### Events Table
- **office_event_id_tag**: Unique event ID
- **system**: Source system (VDOT, WAZE)
- **event_details_level_1/2/3**: Event classification
- **start_time**, **end_time**: Timestamps
- **latitude**, **longitude**: Location
- **state**: Region (DC/VA)

### Weather Table
- **datetime**: Timestamp
- **temp**, **humidity**, **precip**: Weather conditions
- **windspeed**, **visibility**: Environmental factors

### Trips Table
- **date**, **hour**: Temporal keys
- **region**: DC or Virginia
- **total_car_travel_actual/forecast**: Traffic volume
- **network_reliability**, **network_efficiency**: Performance metrics

## 🧪 Testing

```bash
pytest tests/
```

## 📈 Evaluation

Run RAGAS evaluation:
```bash
python evaluation/ragas_eval.py
```

## 🔍 Monitoring

View traces in LangSmith:
```bash
# Ensure LANGCHAIN_TRACING_V2=true in .env
# Visit https://smith.langchain.com
```

## 📝 License

MIT License

## 🤝 Contributing

Contributions welcome! Please read CONTRIBUTING.md for details.

## 📧 Support

For issues and questions, please open a GitHub issue.

