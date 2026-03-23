import streamlit as st
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="DQN Traffic Intelligence", page_icon="🧠", layout="wide")

# --- LOAD DATA ---
@st.cache_data
def load_dqn_data():
    path = "Training/dqn_training_metrics.json"
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)

data = load_dqn_data()

# -----------------------------
# UI START
# -----------------------------
if data:

    st.title("🚦 DQN Traffic Intelligence Dashboard")

    rewards = data.get("episode_rewards", [])
    eval_data = data.get("eval_rewards", [])

    df = pd.DataFrame({
        "Episode": range(1, len(rewards)+1),
        "Reward": rewards
    })

    # --- SIDEBAR CONTROLS ---
    with st.sidebar:
        st.header("⚙️ Controls")

        window = st.slider("Smoothing Window", 5, 100, 20)
        show_raw = st.checkbox("Show Raw Rewards", True)
        show_eval = st.checkbox("Show Evaluation Points", True)

        st.write("---")
        if st.button("🔄 Refresh"):
            st.cache_data.clear()
            st.rerun()

    # Moving average
    df["SMA"] = df["Reward"].rolling(window=window).mean()

    # -----------------------------
    # KPI ROW
    # -----------------------------
    col1, col2, col3, col4 = st.columns(4)

    best_reward = data.get("best_reward", 0)
    avg_100 = data.get("avg_reward_last_100", 0)
    start_reward = rewards[0] if rewards else 0
    improvement = ((avg_100 - start_reward)/abs(start_reward)*100) if start_reward != 0 else 0

    col1.metric("Peak Reward", f"{best_reward:.2f}")
    col2.metric("Avg Last 100", f"{avg_100:.2f}")
    col3.metric("Episodes", len(rewards))
    col4.metric("Learning Gain", f"{improvement:.1f}%")

    # -----------------------------
    # TABS
    # -----------------------------
    tab1, tab2, tab3 = st.tabs(["📈 Overview", "📊 Deep Analysis", "🧪 Diagnostics"])

    # =========================================================
    # TAB 1 — MAIN TRAINING CURVE
    # =========================================================
    with tab1:

        fig = go.Figure()

        if show_raw:
            fig.add_trace(go.Scatter(
                x=df["Episode"], y=df["Reward"],
                mode='lines',
                name="Raw Reward",
                opacity=0.3
            ))

        fig.add_trace(go.Scatter(
            x=df["Episode"], y=df["SMA"],
            mode='lines',
            name="Smoothed Trend",
            line=dict(width=3)
        ))

        # Evaluation overlay
        if show_eval and eval_data:
            eval_ep = [e["episode"] for e in eval_data]
            eval_r = [e["eval_reward"] for e in eval_data]

            fig.add_trace(go.Scatter(
                x=eval_ep, y=eval_r,
                mode='markers+lines',
                name="Evaluation",
                line=dict(dash='dash')
            ))

        fig.update_layout(
            title="Training Performance",
            hovermode="x unified"
        )

        st.plotly_chart(fig, use_container_width=True)

    # =========================================================
    # TAB 2 — DEEP ANALYSIS
    # =========================================================
    with tab2:

        colA, colB = st.columns(2)

        # Reward Distribution
        with colA:
            fig_hist = px.histogram(df, x="Reward", nbins=40, title="Reward Distribution")
            st.plotly_chart(fig_hist, use_container_width=True)

        # Rolling Std (volatility)
        df["STD"] = df["Reward"].rolling(window=window).std()

        with colB:
            fig_std = go.Figure()
            fig_std.add_trace(go.Scatter(
                x=df["Episode"], y=df["STD"],
                name="Volatility"
            ))
            fig_std.update_layout(title="Reward Volatility (Std Dev)")
            st.plotly_chart(fig_std, use_container_width=True)

    # =========================================================
    # TAB 3 — DIAGNOSTICS
    # =========================================================
    with tab3:

        st.subheader("Model Behavior Diagnostics")

        # Convergence detection
        

        # Reward trend slope
        recent = df["Reward"].tail(100)
        slope = np.polyfit(range(len(recent)), recent, 1)[0]

        st.write(f"**Recent Trend Slope:** {slope:.4f}")

        if slope > 0:
            st.info("Learning still improving")
        else:
            st.info("Learning plateau reached")

        st.write("---")

        st.write("### Raw Metadata")
        st.json({k: v for k, v in data.items() if not isinstance(v, list)})

else:
    st.error("No data found. Ensure training_metrics.json exists.")
