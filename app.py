"""Streamlit entrypoint for the Browser Behavior Analytics dashboard."""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.utils.config_loader import get_paths

st.set_page_config(page_title="Browser Behavior Analytics", page_icon="📊", layout="wide")


@st.cache_data(ttl=30)
def load_data(path: str) -> pd.DataFrame:
    file = Path(path)
    return pd.read_parquet(file) if file.is_file() else pd.DataFrame()


paths = get_paths()["paths"]
sessions = load_data(paths["session_segments"])
predictions = load_data(paths["lstm_predictions"])
recommendations = load_data(paths["recommendations"])

st.title("Browser Behavior Analytics")
st.caption("Session segmentation, next-session forecasting, and category recommendations.")

if sessions.empty:
    st.warning(
        "No segment dataset found. Run the feature, clustering, LSTM, "
        "and recommendation pipelines first."
    )
    st.stop()

first, second, third = st.columns(3)
first.metric("Sessions", len(sessions))
second.metric("Behavior segments", sessions["segment_id"].nunique())
third.metric(
    "LSTM test accuracy",
    f"{predictions['category_correct'].mean():.1%}"
    if not predictions.empty and "category_correct" in predictions
    else "Not available",
)

left, right = st.columns(2)
with left:
    st.subheader("Behavior segments")
    st.bar_chart(sessions["segment_id"].value_counts().sort_index())
with right:
    st.subheader("Next-session category forecasts")
    if predictions.empty:
        st.info("Train the LSTM to see forecasts.")
    else:
        st.bar_chart(predictions["predicted_category"].value_counts())

st.subheader("Recommendations")
if recommendations.empty:
    st.info("Run the recommendation pipeline after the LSTM.")
else:
    session_id = st.selectbox("Forecast session", recommendations["target_session_id"].unique())
    st.dataframe(
        recommendations[recommendations["target_session_id"] == session_id],
        use_container_width=True,
        hide_index=True,
    )
