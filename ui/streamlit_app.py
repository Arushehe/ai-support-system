"""
Minimal Streamlit UI for the AI Support Ticket Assistant.

Talks to the FastAPI backend over HTTP (API_BASE_URL, default
http://localhost:8000) rather than importing app/ code directly — this
keeps the UI a thin client and proves the REST API actually works
end-to-end, as the brief requires both.

Run with: streamlit run ui/streamlit_app.py
"""
import os

import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Support Ticket Assistant", page_icon="🎫", layout="wide")
st.title("🎫 AI Support Ticket Assistant")
st.caption(f"Backend: {API_BASE_URL}")


def api_get(path: str, params: dict | None = None):
    try:
        r = requests.get(f"{API_BASE_URL}{path}", params=params, timeout=30)
        r.raise_for_status()
        return r.json(), None
    except Exception as exc:
        return None, str(exc)


def api_post(path: str, json_body: dict):
    try:
        r = requests.post(f"{API_BASE_URL}{path}", json=json_body, timeout=30)
        r.raise_for_status()
        return r.json(), None
    except Exception as exc:
        return None, str(exc)


# --- Health / status strip -------------------------------------------------
health, err = api_get("/health")
cols = st.columns(4)
if err:
    st.error(f"Backend unreachable at {API_BASE_URL}: {err}. Start it with `uvicorn app.api:app --port 8000`.")
else:
    cols[0].metric("Tickets loaded", health["row_count"])
    cols[1].metric("Status", health["status"])
    llm = health["llm"]
    cols[2].metric("LLM provider", llm.get("provider", "-"))
    cols[3].metric("LLM configured", "yes" if llm.get("configured") else "no")
    if not llm.get("configured"):
        st.warning(
            "No LLM backend is configured, so natural-language queries will fail. "
            "Set GROQ_API_KEY (free tier: https://console.groq.com/keys), or set "
            "LLM_PROVIDER=ollama to use a local model."
        )

tab_query, tab_anomalies, tab_stats = st.tabs(["💬 Ask a question", "🚨 Anomalies", "📊 Overview"])

# --- NL Query tab ------------------------------------------------------------
with tab_query:
    st.subheader("Ask about the ticket data")
    example_qs = [
        "How many tickets are currently open?",
        "Which agent has the lowest average customer rating?",
        "Show me all Critical tickets that are still unresolved.",
        "What is the average customer rating for Technical category tickets?",
        "Which agent resolved the most tickets?",
    ]
    picked = st.selectbox("Example questions", ["(type your own below)"] + example_qs)
    default_text = "" if picked == "(type your own below)" else picked
    question = st.text_input("Your question", value=default_text, placeholder="e.g. How many billing tickets are open?")

    if st.button("Ask", type="primary", disabled=not question):
        with st.spinner("Thinking..."):
            resp, err = api_post("/query", {"question": question})
        if err:
            st.error(err)
        elif "error" in resp:
            st.error(resp["error"])
        else:
            st.success(resp["answer"])
            with st.expander("How the system interpreted this question"):
                st.json(resp.get("plan", {}))
            result = resp.get("result", {})
            if result.get("intent") == "list" and result.get("rows"):
                st.dataframe(pd.DataFrame(result["rows"]), use_container_width=True)
            elif result.get("intent") == "groupby_agg" and result.get("results"):
                rdf = pd.DataFrame(result["results"]).set_index("group")
                st.bar_chart(rdf)

# --- Anomalies tab -----------------------------------------------------------
with tab_anomalies:
    st.subheader("Flagged anomalies")
    st.caption(
        "Rule-based detection: statistical outliers in resolution time (per category), "
        "and unresolved High/Critical tickets older than 24 hours."
    )
    since_days = st.selectbox("Time window", ["All time", "Last 7 days", "Last 30 days"], index=0)
    days_map = {"All time": None, "Last 7 days": 7, "Last 30 days": 30}
    data, err = api_get("/anomalies", params={"since_days": days_map[since_days]})
    if err:
        st.error(err)
    else:
        summary = data["summary"]
        st.metric("Total flagged", summary["total_flagged"])
        if summary["by_type"]:
            st.write(summary["by_type"])
        tickets = data["tickets"]
        if tickets:
            st.dataframe(pd.DataFrame(tickets), use_container_width=True)
        else:
            st.info("No anomalies found for this window.")

# --- Overview tab -------------------------------------------------------------
with tab_stats:
    st.subheader("Dataset overview")
    data, err = api_get("/stats")
    if err:
        st.error(err)
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total tickets", data["total_tickets"])
        c2.metric("Avg response (hrs)", data["avg_response_time_hrs"])
        c3.metric("Avg resolution (hrs)", data["avg_resolution_time_hrs"])
        c1b, c2b, c3b = st.columns(3)
        c1b.bar_chart(pd.Series(data["by_status"], name="count"))
        c2b.bar_chart(pd.Series(data["by_priority"], name="count"))
        c3b.bar_chart(pd.Series(data["by_category"], name="count"))
