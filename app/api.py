"""
REST API for the AI-powered support ticket system.

Endpoints:
  GET  /health          - service + LLM backend status
  POST /query           - natural language question -> answer
  GET  /anomalies       - flagged anomalous tickets
  GET  /stats           - basic dataset stats (bonus, not required)

Run directly with: uvicorn app.api:app --reload --port 8000
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.data_layer import load_tickets, dataset_now
from app.anomaly import detect_anomalies, anomaly_summary
from app.nl_query import answer_question
from app.llm_client import llm_status

app = FastAPI(
    title="AI Support Ticket Assistant",
    description="NL querying and anomaly detection over a customer support ticket dataset.",
    version="1.0.0",
)

# Permissive CORS so the Streamlit UI (a different port/origin) can call
# this API directly during local/dev use.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=["How many tickets are currently open?"])


@app.get("/health")
def health():
    try:
        df = load_tickets()
        data_ok = True
        row_count = len(df)
    except Exception as exc:  # dataset missing/corrupt
        data_ok = False
        row_count = 0
    return {
        "status": "ok" if data_ok else "degraded",
        "dataset_loaded": data_ok,
        "row_count": row_count,
        "dataset_reference_time": str(dataset_now()) if data_ok else None,
        "llm": llm_status(),
    }


@app.post("/query")
def query(req: QueryRequest):
    result = answer_question(req.question)
    if "error" in result:
        # Still return 200 with the error payload for LLM-side issues
        # (bad plan, model down) so the UI can show a friendly message;
        # reserve 4xx/5xx for actual API-contract problems.
        return result
    return result


@app.get("/anomalies")
def anomalies(
    since_days: Optional[int] = Query(
        None, description="Only consider tickets created in the last N days."
    )
):
    since = dataset_now() - pd.Timedelta(days=since_days) if since_days else None
    try:
        df = detect_anomalies(since=since)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    df = df.copy()
    if not df.empty:
        df["created_at"] = df["created_at"].astype(str)
        df = df.astype(object).where(pd.notnull(df), None)
    return {
        "summary": anomaly_summary(since=since),
        "tickets": df.to_dict(orient="records"),
    }


@app.get("/stats")
def stats():
    """Bonus endpoint: quick dataset overview, useful for the UI landing view."""
    df = load_tickets()
    return {
        "total_tickets": len(df),
        "by_status": df["status"].value_counts().to_dict(),
        "by_priority": df["priority"].value_counts().to_dict(),
        "by_category": df["category"].value_counts().to_dict(),
        "avg_response_time_hrs": round(float(df["response_time_hrs"].mean()), 2),
        "avg_resolution_time_hrs": round(float(df["resolution_time_hrs"].mean()), 2),
        "avg_customer_rating": round(float(df["customer_rating"].mean()), 2),
    }
