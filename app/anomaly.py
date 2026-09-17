"""
Anomaly detection over the ticket dataset.

Deliberately rule-based rather than LLM-based: anomaly *detection* here
is a statistical/business-rule task (outlier resolution times, breached
SLAs), and rules are cheap, deterministic, auditable, and don't depend
on an external model being available. The LLM is reserved for natural
language understanding, per the assessment's intent.

Two anomaly classes are implemented, matching the brief's examples:
  1. Abnormally long resolution times (statistical outliers, per
     category since categories have very different typical durations).
  2. Unresolved high/critical priority tickets older than 24 hours
     (an SLA-breach style anomaly).
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from app.data_layer import load_tickets, dataset_now

# Number of standard deviations above the category mean that counts as
# an outlier. 2.0 is a conventional threshold that flags roughly the
# slowest ~2-5% of resolutions without being so strict it flags nothing
# on a 500-row dataset.
OUTLIER_Z_THRESHOLD = 2.0
SLA_BREACH_HOURS = 24
SLA_PRIORITIES = ("High", "Critical")


def _long_resolution_outliers(df: pd.DataFrame) -> pd.DataFrame:
    resolved = df[df["resolution_time_hrs"].notna()].copy()
    if resolved.empty:
        return resolved.assign(anomaly_reason=pd.Series(dtype=str), z_score=pd.Series(dtype=float))

    stats = resolved.groupby("category")["resolution_time_hrs"].agg(["mean", "std"])
    resolved = resolved.join(stats, on="category", rsuffix="_cat")
    # A category with a single sample (std = NaN/0) can't produce a
    # meaningful z-score; treat it as "not an outlier" rather than
    # dividing by zero.
    resolved["std"] = resolved["std"].replace(0, pd.NA)
    resolved["z_score"] = (resolved["resolution_time_hrs"] - resolved["mean"]) / resolved["std"]
    resolved["z_score"] = resolved["z_score"].astype(float)

    flagged = resolved[resolved["z_score"] >= OUTLIER_Z_THRESHOLD].copy()
    flagged["anomaly_reason"] = flagged.apply(
        lambda r: (
            f"Resolution took {r['resolution_time_hrs']:.1f}h, "
            f"{r['z_score']:.1f} std above the {r['category']} category "
            f"average ({r['mean']:.1f}h)."
        ),
        axis=1,
    )
    return flagged


def _sla_breaches(df: pd.DataFrame) -> pd.DataFrame:
    breached = df[
        df["is_unresolved"]
        & df["priority"].isin(SLA_PRIORITIES)
        & (df["age_hours"] > SLA_BREACH_HOURS)
    ].copy()
    breached["z_score"] = float("nan")
    breached["anomaly_reason"] = breached.apply(
        lambda r: (
            f"{r['priority']} priority ticket still '{r['status']}' after "
            f"{r['age_hours']:.1f}h (SLA threshold: {SLA_BREACH_HOURS}h)."
        ),
        axis=1,
    )
    return breached


def detect_anomalies(
    df: Optional[pd.DataFrame] = None,
    since: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Return a DataFrame of flagged tickets with an `anomaly_type` and
    human-readable `anomaly_reason` column. A ticket can be flagged by
    more than one rule; each match is a separate row.

    `since`: optionally restrict to tickets created on/after this
    timestamp (used for "anomalies this week" style questions).
    """
    if df is None:
        df = load_tickets()
    if since is not None:
        df = df[df["created_at"] >= since]

    long_res = _long_resolution_outliers(df)
    long_res = long_res.assign(anomaly_type="long_resolution_time")

    sla = _sla_breaches(df)
    sla = sla.assign(anomaly_type="unresolved_sla_breach")

    cols = [
        "ticket_id", "created_at", "category", "priority", "status",
        "response_time_hrs", "resolution_time_hrs", "agent_id",
        "customer_rating", "issue_summary", "anomaly_type", "anomaly_reason", "z_score",
    ]
    combined = pd.concat([long_res, sla], ignore_index=True)
    if combined.empty:
        return combined.reindex(columns=cols)
    return combined[cols].sort_values("created_at", ascending=False).reset_index(drop=True)


def anomaly_summary(since: Optional[pd.Timestamp] = None) -> dict:
    flagged = detect_anomalies(since=since)
    return {
        "total_flagged": int(len(flagged)),
        "by_type": flagged["anomaly_type"].value_counts().to_dict() if not flagged.empty else {},
        "reference_time": str(dataset_now()),
    }
