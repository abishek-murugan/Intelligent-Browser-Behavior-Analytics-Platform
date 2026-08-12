"""Interactive Streamlit dashboard for browser behavior analytics.

Run with ``uv run streamlit run app.py`` after the clustering, LSTM, and
recommendation pipelines have produced their persisted artifacts.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.utils.config_loader import get_paths

st.set_page_config(page_title="Browser Behavior Analytics", page_icon="📊", layout="wide")


@st.cache_data(ttl=30)
def load_data(path: str) -> pd.DataFrame:
    """Load one optional Parquet artifact, refreshing every 30 seconds."""
    file = Path(path)
    return pd.read_parquet(file) if file.is_file() else pd.DataFrame()


paths = get_paths()["paths"]
sessions = load_data(paths["session_segments"])
predictions = load_data(paths["lstm_predictions"])
recommendations = load_data(paths["recommendations"])

st.title("Browser Behavior Analytics")
st.caption("Behavior segments, next-session forecasts, and evidence-backed recommendations.")

if sessions.empty:
    st.warning(
        "No segment dataset found. Run the feature, clustering, LSTM, "
        "and recommendation pipelines first."
    )
    st.stop()

accuracy = (
    predictions["category_correct"].mean()
    if not predictions.empty and "category_correct" in predictions
    else None
)
kpis = st.columns(4)
kpis[0].metric("Sessions", f"{len(sessions):,}")
kpis[1].metric("Behavior segments", sessions["segment_id"].nunique())
kpis[2].metric("Observed categories", sessions["dominant_category"].nunique())
kpis[3].metric("LSTM category accuracy", f"{accuracy:.1%}" if accuracy is not None else "N/A")

overview, clusters, forecast, recommendation_tab = st.tabs(
    ["Overview", "Clusters", "LSTM forecast", "Recommendations"]
)

with overview:
    left, right = st.columns(2)
    with left:
        categories = sessions["dominant_category"].value_counts().rename_axis("category")
        st.plotly_chart(
            px.bar(
                categories.reset_index(name="sessions"),
                x="category",
                y="sessions",
                title="Sessions by category",
            ),
            width="stretch",
        )
    with right:
        by_hour = (
            sessions.groupby("hour", as_index=False).size().rename(columns={"size": "sessions"})
        )
        st.plotly_chart(
            px.line(by_hour, x="hour", y="sessions", markers=True, title="Session starts by hour"),
            width="stretch",
        )

with clusters:
    cluster_counts = sessions["segment_id"].value_counts().sort_index().rename_axis("segment")
    st.plotly_chart(
        px.bar(
            cluster_counts.reset_index(name="sessions"),
            x="segment",
            y="sessions",
            title="Segment sizes",
        ),
        width="stretch",
    )
    if {"avg_usage_percent", "session_duration_seconds"} <= set(sessions.columns):
        st.plotly_chart(
            px.scatter(
                sessions,
                x="session_duration_seconds",
                y="avg_usage_percent",
                color=sessions["segment_id"].astype(str),
                hover_data=["dominant_category", "event_count"],
                title="RAM usage and duration by segment",
                labels={"color": "Segment"},
            ),
            width="stretch",
        )

with forecast:
    if predictions.empty:
        st.info("Train the LSTM to view held-out next-session forecasts.")
    else:
        left, right = st.columns(2)
        with left:
            forecast_counts = (
                predictions["predicted_category"].value_counts().rename_axis("category")
            )
            st.plotly_chart(
                px.bar(
                    forecast_counts.reset_index(name="forecasts"),
                    x="category",
                    y="forecasts",
                    title="Predicted next categories",
                ),
                width="stretch",
            )
        with right:
            actual_counts = predictions["actual_category"].value_counts().rename_axis("category")
            st.plotly_chart(
                px.bar(
                    actual_counts.reset_index(name="sessions"),
                    x="category",
                    y="sessions",
                    title="Held-out actual categories",
                ),
                width="stretch",
            )
        st.dataframe(
            predictions[
                ["target_session_id", "actual_category", "predicted_category", "category_correct"]
            ],
            hide_index=True,
            width="stretch",
        )

with recommendation_tab:
    if recommendations.empty:
        st.info("Run the recommendation pipeline after LSTM training.")
    else:
        session_id = st.selectbox("Forecast session", recommendations["target_session_id"].unique())
        selected = recommendations[recommendations["target_session_id"] == session_id]
        st.subheader(f"Forecast: {selected.iloc[0]['predicted_category']}")
        st.dataframe(selected, hide_index=True, width="stretch")
