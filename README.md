# AI Support Ticket Assistant

An AI-powered support ticket analysis system built for the DOTMappers AI Engineer assessment.
The system uses a 500-row support ticket dataset to provide natural language querying, statistics, and anomaly detection.

## Features

- Support ticket data analysis
- Natural language queries using an LLM
- SLA breach detection
- Long resolution time detection
- Ticket statistics
- FastAPI REST API
- Streamlit UI
- Swagger API documentation

## Tech Stack

- Python
- Pandas
- FastAPI
- Streamlit
- Groq
- Pydantic
- Pytest

## Project Structure

```text
ai-support-system/
├── app/
│   ├── api.py
│   ├── data_layer.py
│   ├── anomaly.py
│   ├── llm_client.py
│   └── nl_query.py
├── data/
│   └── support_tickets.csv
├── ui/
│   └── streamlit_app.py
├── tests/
│   └── test_core.py
├── run.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## How to Run

### 1. Clone

```bash
git clone <your-github-repository-url>
cd ai-support-system
```

### 2. Create virtual environment

**Windows**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure LLM

Create `.env`:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-20b
```

Do not upload `.env` to GitHub.

### 5. Start

```bash
python run.py
```

- Streamlit: `http://localhost:8501`
- FastAPI: `http://localhost:8000`
- Swagger: `http://localhost:8000/docs`

## How It Works

**Data Layer**
- Loads and prepares the CSV using Pandas.

**Natural Language Querying**
- LLM converts the user's question into a structured JSON query plan.
- The application validates and executes the plan using Pandas.
- No generated Python or SQL code is executed.

**Anomaly Detection**
- Unresolved High/Critical tickets older than 24 hours
- Unusually long resolution times within a category

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | API, dataset and LLM status |
| POST | `/query` | Natural language queries |
| GET | `/anomalies` | Detected anomalies |
| GET | `/stats` | Dataset statistics |

## Dataset Time Reference

The dataset contains tickets from January–March 2024.

Relative time calculations use the latest dataset timestamp plus a small buffer as the reference time.

## Testing

```bash
pytest
```

Tests cover data processing, anomaly detection, and query execution.

## Limitations

- Uses the provided CSV dataset
- Data is kept in memory
- Query types are limited to supported operations
- No conversation memory
- Anomaly thresholds are fixed

## Future Improvements

- SQLite/PostgreSQL support
- More natural language query types
- Semantic search
- Conversation history
- Configurable anomaly thresholds
- Improved monitoring and logging

## Assessment

Built for the **DOTMappers AI Engineer — End-to-End AI System Sprint**.

Covers:
- Data ingestion
- LLM integration
- Natural language querying
- Anomaly detection
- REST API
- Streamlit UI
- Automated testing