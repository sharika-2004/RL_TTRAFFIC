import streamlit as st
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import numpy as np

# --- PATHS ---
base_dir = os.path.dirname(os.path.abspath(__file__))
training_dir = os.path.join(base_dir, '..', 'Training')
metrics_path = os.path.join(training_dir, 'new_training_metrics.json')

summary_img_path = os.path.join(training_dir, 'summary_dashboard.png')
curves_img_path = os.path.join(training_dir, 'training_curves_all_metrics.png')

st.set_page_config(page_title="Traffic RL Analytics", page_icon="🚦", layout="wide")

# --- LOAD ---
@st.cache_data
def load_data():
    if not os.path.exists(metrics_path):
        st.error(f"Metrics not found at {metrics_path}")
        st.stop()
    with open(metrics_path, 'r') as f:
        return json.load(f)

data = load_data()
df = pd.DataFrame(data.get('evaluations', []))

# -----------------------------
# SIDEBAR CONTROLS
# -----------------------------
with st.sidebar:
    st.header("⚙️ Controls")

    window = st.slider("Smoothing Window", 1, 20, 3)
    normalize = st.checkbox("Normalize Metrics", False)

    st.write("---")
    if st.button("🔄 Refresh"):
        st.cache_data.clear()
        st.rerun()

# -----------------------------
# HEADER
# -----------------------------
st.title("🚦 Traffic RL Intelligence Dashboard")
st.caption(f"Source: {os.path.basename(metrics_path)}")

# -----------------------------
# KPI ROW
# -----------------------------
if not df.empty:
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Best Reward", f"{data['best_reward']:.2f}")
    col2.metric("Min Waiting", f"{df['waiting_time'].min():.3f}")
    col3.metric("Max Throughput", f"{df['throughput'].max():.2f}")
    col4.metric("Episodes", data['total_episodes'])

# -----------------------------
# PREPROCESSING
# -----------------------------
if not df.empty:
    df = df.sort_values("episode")

    # smoothing
    for col in ["reward", "waiting_time", "queue_length", "throughput"]:
        df[f"{col}_smooth"] = df[col].rolling(window=window).mean()

    # normalization
    if normalize:
        for col in ["reward", "waiting_time", "queue_length", "throughput"]:
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📉 Interactive Curves",
    "📊 Comparative Analysis",
    "🧠 Diagnostics",
    "🖼️ Static Summaries"
])

# =========================================================
# TAB 1 — INTERACTIVE CURVES
# =========================================================
with tab1:

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Reward", "Waiting Time", "Queue Length", "Throughput")
    )

    metrics = ["reward", "waiting_time", "queue_length", "throughput"]

    for i, metric in enumerate(metrics):
        row = i // 2 + 1
        col = i % 2 + 1

        fig.add_trace(
            go.Scatter(
                x=df['episode'],
                y=df[metric],
                name=f"{metric}",
                opacity=0.3
            ),
            row=row, col=col
        )

        fig.add_trace(
            go.Scatter(
                x=df['episode'],
                y=df[f"{metric}_smooth"],
                name=f"{metric}_trend",
                line=dict(width=3)
            ),
            row=row, col=col
        )

    fig.update_layout(height=700, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# TAB 2 — COMPARATIVE ANALYSIS
# =========================================================
with tab2:

    st.subheader("Cross-Metric Relationships")

    colA, colB = st.columns(2)

    # Scatter: Reward vs Waiting
    with colA:
        fig1 = px.scatter(df, x="waiting_time", y="reward",
                          title="Reward vs Waiting Time",
                          trendline="ols")
        st.plotly_chart(fig1, use_container_width=True)

    # Scatter: Queue vs Throughput
    with colB:
        fig2 = px.scatter(df, x="queue_length", y="throughput",
                          title="Queue vs Throughput",
                          trendline="ols")
        st.plotly_chart(fig2, use_container_width=True)

    st.write("### Correlation Matrix")

    corr = df[["reward", "waiting_time", "queue_length", "throughput"]].corr()
    fig_corr = px.imshow(corr, text_auto=True, title="Metric Correlations")
    st.plotly_chart(fig_corr, use_container_width=True)

# =========================================================
# TAB 3 — DIAGNOSTICS
# =========================================================
with tab3:

    st.subheader("Model Intelligence Diagnostics")

    # Stability
  

    # Trend slope
    slope = np.polyfit(df["episode"], df["reward"], 1)[0]
    st.write(f"Trend Slope: {slope:.4f}")

    if slope > 0:
        st.info("Learning improving")
    else:
        st.info("Learning plateau or degradation")

    # Trade-off insight
    

# =========================================================
# TAB 4 — STATIC IMAGES
# =========================================================
with tab4:

    col1, col2 = st.columns(2)

    # --- Updated paths (graph folder) ---
    graph_dir = os.path.join(base_dir, '..', 'graphs')

    summary_img = os.path.join(graph_dir, 'summary_dashboard.png')
    curves_img = os.path.join(graph_dir, 'training_curves_all_metrics.png')

    with col1:
        st.write("### Summary Dashboard")
        if os.path.exists(summary_img):
            st.image(summary_img, use_container_width=True)
        else:
            st.info("summary_dashboard.png not found in graph folder")

    with col2:
        st.write("### Training Curves")
        if os.path.exists(curves_img):
            st.image(curves_img, use_container_width=True)
        else:
            st.info("training_curves_all_metrics.png not found in graph folder")
