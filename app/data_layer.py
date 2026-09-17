"""
Data layer: loads the support-ticket CSV, normalizes types, and exposes
a small, well-documented schema description that is reused by both the
LLM prompt (so the model knows exactly what it can query) and the
anomaly-detection module.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd

DATA_PATH = os.environ.get(
    "TICKETS_CSV_PATH",
    os.path.join(os.path.dirname(__file__), "..", "data", "support_tickets.csv"),
)

# "Now" is pinned to the last timestamp seen in the dataset (plus a small
# buffer) rather than wall-clock time. The data is historical (Jan-Mar 2024),
# so using real wall-clock "now" would make every open ticket look
# absurdly overdue. This keeps "older than 24 hours" anomaly checks and
# "this week" queries meaningful relative to the dataset itself.
_NOW_BUFFER_HOURS = 6


@dataclass(frozen=True)
class ColumnDoc:
    name: str
    dtype: str
    description: str
    allowed_values: Optional[list] = None


SCHEMA: list[ColumnDoc] = [
    ColumnDoc("ticket_id", "string", "Unique ticket identifier, e.g. TKT-001."),
    ColumnDoc("created_at", "datetime", "Ticket creation timestamp (YYYY-MM-DD HH:MM)."),
    ColumnDoc("category", "string", "Issue category.", ["Billing", "Technical", "General"]),
    ColumnDoc("priority", "string", "Ticket urgency.", ["Low", "Medium", "High", "Critical"]),
    ColumnDoc("status", "string", "Current ticket status.", ["Open", "Resolved", "Escalated"]),
    ColumnDoc("response_time_hrs", "float", "Hours from creation to first agent response."),
    ColumnDoc(
        "resolution_time_hrs",
        "float (nullable)",
        "Hours from creation to resolution. Null if the ticket is not resolved.",
    ),
    ColumnDoc("agent_id", "string", "Assigned support agent identifier, e.g. AGT-04."),
    ColumnDoc(
        "customer_rating",
        "integer 1-5 (nullable)",
        "Post-resolution satisfaction rating. Null if unresolved.",
    ),
    ColumnDoc("issue_summary", "string", "Free-text description of the issue."),
]

_df_cache: Optional[pd.DataFrame] = None


def load_tickets(force_reload: bool = False) -> pd.DataFrame:
    """Load and normalize the ticket dataset. Cached after first call."""
    global _df_cache
    if _df_cache is not None and not force_reload:
        return _df_cache

    df = pd.read_csv(DATA_PATH)
    df["created_at"] = pd.to_datetime(df["created_at"])

    # Derived, read-only convenience columns used by anomaly detection
    # and by several NL-query aggregations. Keeping them out of the raw
    # CSV keeps the source data untouched and the derivation auditable.
    now = dataset_now(df)
    df["age_hours"] = (now - df["created_at"]).dt.total_seconds() / 3600.0
    df["is_unresolved"] = df["status"].isin(["Open", "Escalated"])

    _df_cache = df
    return df


def dataset_now(df: Optional[pd.DataFrame] = None) -> pd.Timestamp:
    """
    The reference 'now' for age/recency calculations: the latest
    created_at in the data plus a small buffer, not the real wall clock.
    """
    if df is None:
        df = load_tickets()
    return df["created_at"].max() + pd.Timedelta(hours=_NOW_BUFFER_HOURS)


def schema_prompt_block() -> str:
    """Render the schema as compact text for the LLM prompt."""
    lines = []
    for col in SCHEMA:
        allowed = f" Allowed values: {col.allowed_values}." if col.allowed_values else ""
        lines.append(f"- {col.name} ({col.dtype}): {col.description}{allowed}")
    lines.append(
        "- age_hours (float, derived): hours between created_at and the "
        "dataset's reference 'now' (the latest timestamp in the data)."
    )
    lines.append(
        "- is_unresolved (bool, derived): True if status is 'Open' or 'Escalated'."
    )
    return "\n".join(lines)
