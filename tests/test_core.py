"""
Unit tests that don't require an LLM or network access — they cover the
deterministic parts of the system: data loading, anomaly detection, and
query-plan execution/validation. Run with: pytest
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.data_layer import load_tickets, dataset_now
from app.anomaly import detect_anomalies
from app.nl_query import QueryPlan, execute_plan, _validate_plan, PlanValidationError


def test_load_tickets_has_expected_columns():
    df = load_tickets()
    for col in ["ticket_id", "created_at", "category", "priority", "status", "agent_id"]:
        assert col in df.columns
    assert len(df) > 0


def test_dataset_now_is_after_last_created_at():
    df = load_tickets()
    assert dataset_now(df) > df["created_at"].max()


def test_anomaly_detection_returns_expected_columns():
    flagged = detect_anomalies()
    expected_cols = {"ticket_id", "anomaly_type", "anomaly_reason", "priority", "status"}
    if not flagged.empty:
        assert expected_cols.issubset(set(flagged.columns))
        assert set(flagged["anomaly_type"].unique()) <= {"long_resolution_time", "unresolved_sla_breach"}


def test_sla_breach_tickets_are_actually_breaching():
    flagged = detect_anomalies()
    sla = flagged[flagged["anomaly_type"] == "unresolved_sla_breach"]
    assert (sla["priority"].isin(["High", "Critical"])).all()
    assert (sla["status"].isin(["Open", "Escalated"])).all()


def test_execute_plan_count_open_tickets():
    plan = QueryPlan(
        intent="aggregate",
        filters=[{"column": "status", "op": "eq", "value": "Open"}],
    )
    _validate_plan(plan)
    result = execute_plan(plan)
    df = load_tickets()
    expected = int((df["status"] == "Open").sum())
    assert result["value"] == expected


def test_execute_plan_groupby_lowest_rating_agent():
    plan = QueryPlan(
        intent="groupby_agg",
        groupby="agent_id",
        metric={"column": "customer_rating", "agg": "mean"},
        sort_ascending=True,
        limit=1,
    )
    _validate_plan(plan)
    result = execute_plan(plan)
    assert result["intent"] == "groupby_agg"
    assert len(result["results"]) == 1

    df = load_tickets()
    means = df.groupby("agent_id")["customer_rating"].mean().sort_values()
    assert result["results"][0]["group"] == str(means.index[0])


def test_plan_validation_rejects_unknown_column():
    plan = QueryPlan(intent="aggregate", filters=[{"column": "ssn", "op": "eq", "value": "x"}])
    try:
        _validate_plan(plan)
        assert False, "expected PlanValidationError"
    except PlanValidationError:
        pass


def test_plan_validation_rejects_unknown_op():
    plan = QueryPlan(intent="aggregate", filters=[{"column": "priority", "op": "regex", "value": "x"}])
    try:
        _validate_plan(plan)
        assert False, "expected PlanValidationError"
    except PlanValidationError:
        pass
