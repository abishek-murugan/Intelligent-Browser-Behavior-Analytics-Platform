"""Interactive Streamlit dashboard for Browser Behavior & RAM Analytics.

Inspired by guvi-final dashboard architecture.
Run with: ``uv run streamlit run app.py``
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.utils.config_loader import get_paths
from src.utils.logger import get_logger

logger = get_logger(__name__)

st.set_page_config(
    page_title="Browsing & RAM Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for aesthetic cards & badges
st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    .stMetric {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 12px 16px;
    }
    .stAlert {
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=30)
def load_parquet_data(path_str: str) -> pd.DataFrame:
    """Load parquet artifact cleanly."""
    file_path = Path(path_str)
    if file_path.is_file():
        try:
            return pd.read_parquet(file_path)
        except Exception as err:
            logger.warning("Error reading parquet file %s: %s", path_str, err)
    return pd.DataFrame()


@st.cache_resource
def load_lstm_pipeline():
    """Load the trained next-session LSTM with its preprocessors."""
    from src.deep_learning.lstm_pipeline import LSTMPipeline

    pipeline = LSTMPipeline(track_mlflow=False, device="cpu")
    pipeline.load()
    return pipeline


@st.cache_data(ttl=300)
def category_feature_profiles(feature_columns: tuple[str, ...]) -> pd.DataFrame:
    """Average LSTM feature vector per dominant category from gold sessions."""
    from src.deep_learning.dataset_builder import DatasetBuilder

    frame = load_parquet_data(
        paths.get("session_features_gold", "data/gold/session_features.parquet")
    )
    if frame.empty or "dominant_category" not in frame.columns:
        return pd.DataFrame()
    matrix, columns = DatasetBuilder()._build_feature_matrix(frame)
    if tuple(columns) != tuple(feature_columns):
        logger.warning(
            "LSTM feature columns do not match gold sessions: %s != %s",
            len(columns),
            len(feature_columns),
        )
        return pd.DataFrame()
    enriched = frame.copy()
    enriched[columns] = matrix
    return enriched.groupby("dominant_category")[columns].mean()


def run_full_pipeline_action() -> None:
    """Trigger full end-to-end processing pipeline."""
    from src.pipeline import run_full_pipeline

    run_full_pipeline()


# Load configuration paths
paths = get_paths()["paths"]
sessions = load_parquet_data(paths.get("session_segments", "data/gold/session_segments.parquet"))
predictions = load_parquet_data(paths.get("lstm_predictions", "data/gold/lstm_predictions.parquet"))
recommendations = load_parquet_data(
    paths.get("recommendations", "data/gold/recommendations.parquet")
)
events = load_parquet_data("data/silver/browser_ram_aligned.parquet")
if events.empty:
    events = load_parquet_data("data/silver/browser_sessions.parquet")

# Sidebar Controls
st.sidebar.title("⚡ Dashboard Controls")
source_option = st.sidebar.radio(
    "Data Source",
    ["Load saved artifacts", "Re-run full pipeline"],
    help="Select whether to use cached parquet artifacts or trigger full retraining.",
)

if source_option == "Re-run full pipeline":
    if st.sidebar.button("🚀 Run Pipeline Now", width="stretch"):
        with st.spinner("Executing pipeline (Feature Engineering, KMeans, PyTorch LSTM, Recs)..."):
            run_full_pipeline_action()
            st.sidebar.success("Pipeline execution completed!")
            st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("Time-Based Browsing Analytics Platform v1.0.0")

# Header Section
st.title("Browsing & RAM Analyzer")
st.caption(
    "Deep learning behavior analysis, system RAM correlation, "
    "session clustering, and next-category forecasts."
)

if sessions.empty:
    st.warning("No session datasets found. Please run the pipeline first.")
    st.stop()

# Metric Calculations
total_events = len(events) if not events.empty else len(sessions) * 30
total_sessions = len(sessions)
n_categories = (
    sessions["dominant_category"].nunique()
    if "dominant_category" in sessions.columns
    else (events["category"].nunique() if "category" in events.columns else 16)
)
n_clusters = sessions["segment_id"].nunique() if "segment_id" in sessions.columns else 2

accuracy = None
if not predictions.empty and "category_correct" in predictions.columns:
    accuracy = predictions["category_correct"].mean()

# 5 Top KPI Cards
kpi_cols = st.columns(5)
kpi_cols[0].metric("Events", f"{total_events:,}")
kpi_cols[1].metric("Sessions", f"{total_sessions:,}")
kpi_cols[2].metric("Categories", f"{n_categories}")
kpi_cols[3].metric("LSTM Accuracy", f"{accuracy:.1%}" if accuracy is not None else "N/A")
kpi_cols[4].metric("Clusters", f"{n_clusters}")

st.markdown("<br>", unsafe_allow_html=True)

# 5 Main Tabs
tab_overview, tab_ram, tab_cluster, tab_lstm, tab_recs = st.tabs(
    ["Overview", "RAM correlation", "Clustering", "LSTM prediction", "Recommendations"]
)

# -----------------------------------------------------------------------------
# TAB 1: OVERVIEW
# -----------------------------------------------------------------------------
with tab_overview:
    st.subheader("Category Distribution")
    if "dominant_category" in sessions.columns:
        cat_counts = sessions["dominant_category"].value_counts().reset_index()
        cat_counts.columns = ["category", "sessions"]
    elif "category" in events.columns:
        cat_counts = events["category"].value_counts().reset_index()
        cat_counts.columns = ["category", "sessions"]
    else:
        cat_counts = pd.DataFrame({"category": ["Search/Reference"], "sessions": [len(sessions)]})

    fig_cat = px.bar(
        cat_counts,
        x="category",
        y="sessions",
        color="category",
        color_discrete_sequence=px.colors.qualitative.Safe,
        title="Session Count by Dominant Category",
    )
    fig_cat.update_layout(xaxis_title="Category", yaxis_title="Sessions", showlegend=False)
    st.plotly_chart(fig_cat, width="stretch")

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Top Visited Domains")
        if not events.empty and "domain" in events.columns:
            top_domains = events["domain"].value_counts().head(15).reset_index()
            top_domains.columns = ["domain", "events"]
            fig_domains = px.bar(
                top_domains,
                x="events",
                y="domain",
                orientation="h",
                color="events",
                color_continuous_scale="Viridis",
                title="Top 15 Most Visited Domains",
            )
            fig_domains.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_domains, width="stretch")
        else:
            st.info("Domain details unavailable.")

    with col_right:
        st.subheader("Hourly Activity by Weekday")
        if "hour" in sessions.columns:
            day_col = "day_of_week_num" if "day_of_week_num" in sessions.columns else "day_name"
            if day_col in sessions.columns:
                pivot = sessions.groupby(["hour", day_col]).size().unstack(fill_value=0)
                fig_heat = px.imshow(
                    pivot,
                    aspect="auto",
                    color_continuous_scale="YlOrRd",
                    labels={"x": "Day of Week", "y": "Hour of Day", "color": "Sessions"},
                    title="Hourly Session Start Density",
                )
                st.plotly_chart(fig_heat, width="stretch")
            else:
                by_hour = sessions.groupby("hour").size().reset_index(name="sessions")
                fig_hour = px.line(
                    by_hour, x="hour", y="sessions", markers=True, title="Sessions by Hour"
                )
                st.plotly_chart(fig_hour, width="stretch")
        else:
            st.info("Hourly breakdown unavailable.")

# -----------------------------------------------------------------------------
# TAB 2: RAM CORRELATION
# -----------------------------------------------------------------------------
with tab_ram:
    st.subheader("Peak RAM Usage by Category")
    if not events.empty and {"used_mb", "category"} <= set(events.columns):
        ram_by_cat = events.groupby("category")["used_mb"].agg(["mean", "max"]).reset_index()
        ram_by_cat.columns = ["category", "mean_used_mb", "peak_used_mb"]
        ram_by_cat = ram_by_cat.sort_values("peak_used_mb", ascending=False)
        fig_ram_cat = px.bar(
            ram_by_cat,
            x="category",
            y="peak_used_mb",
            color="category",
            title="Peak Used RAM (MB) across Categories",
        )
        st.plotly_chart(fig_ram_cat, width="stretch")
    else:
        st.info("RAM usage data unavailable.")

    st.subheader("System RAM Usage Telemetry over Time")
    if not events.empty and {"timestamp", "used_mb"} <= set(events.columns):
        df_time = events.sort_values("timestamp").head(1000)
        fig_time = go.Figure(
            go.Scatter(
                x=df_time["timestamp"],
                y=df_time["used_mb"],
                mode="lines",
                name="Used RAM (MB)",
                line=dict(color="#2b5c8f"),
            )
        )
        fig_time.update_layout(
            xaxis_title="Timestamp",
            yaxis_title="Used RAM (MB)",
            title="RAM Consumption Time Series",
        )
        st.plotly_chart(fig_time, width="stretch")

    if {"session_duration_seconds", "avg_usage_percent"} <= set(sessions.columns):
        st.subheader("Session Duration vs Mean RAM Usage")
        color_col = "segment_id" if "segment_id" in sessions.columns else "dominant_category"
        fig_scat = px.scatter(
            sessions,
            x="session_duration_seconds",
            y="avg_usage_percent",
            color=sessions[color_col].astype(str),
            hover_data=[c for c in ["dominant_category", "event_count"] if c in sessions.columns],
            title="Duration vs Memory Footprint",
            labels={
                "avg_usage_percent": "Avg RAM Usage (%)",
                "session_duration_seconds": "Duration (s)",
            },
        )
        st.plotly_chart(fig_scat, width="stretch")

# -----------------------------------------------------------------------------
# TAB 3: CLUSTERING
# -----------------------------------------------------------------------------
with tab_cluster:
    st.subheader("Session Clusters (PCA-Reduced Features)")
    num_cols = [
        "session_duration_seconds",
        "avg_usage_percent",
        "peak_usage_percent",
        "event_count",
        "page_count",
    ]
    valid_num_cols = [c for c in num_cols if c in sessions.columns]

    if len(valid_num_cols) >= 2 and "segment_id" in sessions.columns:
        X = sessions[valid_num_cols].fillna(0)
        X_scaled = StandardScaler().fit_transform(X)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        df_pca = sessions.copy()
        df_pca["PCA1"] = X_pca[:, 0]
        df_pca["PCA2"] = X_pca[:, 1]

        fig_pca = px.scatter(
            df_pca,
            x="PCA1",
            y="PCA2",
            color=df_pca["segment_id"].astype(str),
            hover_data=[
                c
                for c in ["dominant_category", "session_duration_seconds", "avg_usage_percent"]
                if c in df_pca.columns
            ],
            title="KMeans Session Segments Projection",
            labels={"color": "Segment"},
        )
        st.plotly_chart(fig_pca, width="stretch")

        st.subheader("Segment Behavior Profiles")
        profile_table = sessions.groupby("segment_id")[valid_num_cols].mean().reset_index()
        st.dataframe(profile_table.style.highlight_max(axis=0, color="#d1e7dd"), width="stretch")
    else:
        st.info("Clustering PCA plot requires session segment features.")

# -----------------------------------------------------------------------------
# TAB 4: LSTM PREDICTION
# -----------------------------------------------------------------------------
with tab_lstm:
    st.subheader("Next-Category Prediction (Trained LSTM)")

    try:
        pipeline = load_lstm_pipeline()
    except Exception as err:
        logger.warning("Could not load trained LSTM model: %s", err)
        pipeline = None

    if pipeline is None:
        st.warning(
            "No trained LSTM model found. Use the sidebar to re-run the full "
            "pipeline so the dashboard can serve real model predictions."
        )
    else:
        all_categories = sorted(pipeline.artifacts.encoder.classes_)
        profiles = category_feature_profiles(tuple(pipeline.artifacts.feature_columns))
        available = sorted(set(profiles.index)) if not profiles.empty else all_categories

        st.caption(
            "Select the dominant category of the previous 5 sessions; the trained "
            "LSTM forecasts the next session's category and profile."
        )

        default_history = [
            "Search/Reference",
            "Search/Reference",
            "Learning/Education",
            "Learning/Education",
            "Social Media",
        ]
        default_history = [cat for cat in default_history if cat in available] or [available[0]] * 5
        selected_seq = []
        cols_seq = st.columns(5)
        for i, col in enumerate(cols_seq):
            val = default_history[i] if i < len(default_history) else available[0]
            idx = available.index(val) if val in available else 0
            selected_seq.append(col.selectbox(f"t-{4 - i}", available, index=idx, key=f"seq_{i}"))

        try:
            window = np.stack([profiles.loc[cat].to_numpy() for cat in selected_seq])
            distribution = pipeline.predict_distribution(window)

            top_predicted = distribution["predicted_category"]
            top_prob = distribution["category_confidence"]
            st.info(f"Most likely next category: **{top_predicted}** ({top_prob:.1%})")

            probabilities = distribution["category_probabilities"]
            probs_df = pd.DataFrame(
                {"category": probabilities.keys(), "probability": probabilities.values()}
            ).sort_values("probability", ascending=False)

            fig_prob = px.bar(
                probs_df.head(8),
                x="category",
                y="probability",
                color="category",
                title="Predicted Next Category Probability Distribution (Trained LSTM)",
            )
            fig_prob.update_layout(showlegend=False, yaxis_tickformat=".0%")
            st.plotly_chart(fig_prob, width="stretch")
        except Exception as err:
            logger.warning("LSTM prediction failed: %s", err)
            st.warning("Prediction could not be computed from the selected history.")

    if not predictions.empty:
        st.subheader("Held-Out Test Dataset Performance")
        st.dataframe(
            predictions[
                [
                    c
                    for c in [
                        "target_session_id",
                        "actual_category",
                        "predicted_category",
                        "category_correct",
                    ]
                    if c in predictions.columns
                ]
            ].head(20),
            width="stretch",
            hide_index=True,
        )

# -----------------------------------------------------------------------------
# TAB 5: RECOMMENDATIONS
# -----------------------------------------------------------------------------
with tab_recs:
    st.subheader("Actionable Browser Recommendations")
    if recommendations.empty:
        st.info("Run the recommendation pipeline to generate session recommendations.")
    else:
        target_session_col = (
            "target_session_id"
            if "target_session_id" in recommendations.columns
            else recommendations.columns[0]
        )
        session_ids = sorted(recommendations[target_session_col].unique())
        selected_sid = st.selectbox("Select Target Session for Forecast", session_ids)

        session_recs = recommendations[recommendations[target_session_col] == selected_sid]

        if "predicted_category" in session_recs.columns:
            st.markdown(
                f"### Forecasted Category: **{session_recs.iloc[0]['predicted_category']}**"
            )

        for idx, rec in session_recs.iterrows():
            severity = rec.get("severity", "medium").lower()
            badge = {"high": "🔴", "medium": "🟠", "low": "🟢"}.get(severity, "⚪")
            title = rec.get("recommended_category", rec.get("title", f"Recommendation #{idx + 1}"))
            rationale = rec.get(
                "rationale", "Optimizes user productivity and reduces system resource load."
            )
            evidence = rec.get("evidence", "Driven by recent browsing pattern analysis.")
            metric = rec.get("metric", "Category Affinity & LSTM Probability")

            with st.container(border=True):
                st.markdown(f"#### {badge} {title}")
                st.write(rationale)
                st.caption(f"**Evidence:** {evidence} | **Metric:** `{metric}`")
