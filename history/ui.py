"""UI components for prediction history dashboard."""

import pandas as pd
import streamlit as st
import plotly.graph_objects as go  # type: ignore
import plotly.express as px  # type: ignore
from datetime import datetime
from typing import List, Dict, Any

from .history_utils import load_history, last_n_history


def history_dashboard(username: str) -> None:
    """
    Display prediction history dashboard for a user.
    
    Args:
        username: Username to display history for
    """
    st.title("📊 My Predictions History")
    st.markdown(f"Viewing prediction history for: **{username}**")
    
    # Back button
    if st.button("← Back to Dashboard"):
        st.session_state["show_history"] = False
        st.rerun()
    
    # Load history
    hist = last_n_history(username, n=10)
    
    if not hist:
        st.info("📝 No prediction history yet. Make some predictions to see your history here!")
        return
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(hist)
    
    # Parse timestamps
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
    
    st.markdown("---")
    
    # Charts section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Probability Trend")
        if len(df) > 0 and "probability" in df.columns and "timestamp" in df.columns:
            fig_line = px.line(
                df,
                x="timestamp",
                y="probability",
                title="Prediction Probability Over Time",
                labels={"probability": "Probability", "timestamp": "Date"},
                markers=True,
            )
            fig_line.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("No probability data available for chart.")
    
    with col2:
        st.subheader("📊 Risk Level Distribution")
        if "risk_level" in df.columns:
            risk_counts = df["risk_level"].value_counts()
            fig_bar = go.Figure(
                data=[
                    go.Bar(
                        x=risk_counts.index,
                        y=risk_counts.values,
                        marker_color=["#2ecc71" if "Low" in str(x) else "#e74c3c" for x in risk_counts.index],
                    )
                ]
            )
            fig_bar.update_layout(
                title="High Risk vs Low Risk Count",
                xaxis_title="Risk Level",
                yaxis_title="Count",
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No risk level data available for chart.")
    
    st.markdown("---")
    
    # History table
    st.subheader("📋 Recent Predictions")
    
    # Prepare display DataFrame
    display_cols = []
    if "timestamp" in df.columns:
        display_cols.append("timestamp")
    if "probability" in df.columns:
        display_cols.append("probability")
    if "risk_level" in df.columns:
        display_cols.append("risk_level")
    if "prediction" in df.columns:
        display_cols.append("prediction")
    
    if display_cols:
        display_df = df[display_cols].copy()
        
        # Format timestamp
        if "timestamp" in display_df.columns:
            display_df["timestamp"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        
        # Format probability
        if "probability" in display_df.columns:
            display_df["probability"] = display_df["probability"].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Download full history CSV
    st.markdown("---")
    st.subheader("💾 Export History")
    
    full_hist = load_history(username)
    if full_hist:
        # Convert full history to DataFrame
        full_df = pd.DataFrame(full_hist)
        
        # Flatten nested structures for CSV
        if "inputs" in full_df.columns:
            inputs_df = pd.json_normalize(full_df["inputs"])
            full_df = pd.concat([full_df.drop("inputs", axis=1), inputs_df], axis=1)
        
        if "guidance" in full_df.columns:
            guidance_df = pd.json_normalize(full_df["guidance"])
            guidance_df.columns = [f"guidance_{col}" for col in guidance_df.columns]
            full_df = pd.concat([full_df.drop("guidance", axis=1), guidance_df], axis=1)
        
        csv_data = full_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Full History CSV",
            data=csv_data,
            file_name=f"diabetes_history_{username}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )
    else:
        st.info("No history available to download.")

