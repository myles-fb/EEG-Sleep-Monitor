"""
Patient Dashboard — View study results, Q-score trends, MO counts, and alerts.
"""

import sys
from pathlib import Path

_src = Path(__file__).resolve().parent.parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import streamlit as st
import numpy as np
import requests

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from models import init_db
from services import patient_service, study_service

FASTAPI_URL = "http://localhost:8765"

init_db()

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")
st.title("Patient Dashboard")

# ---------------------------------------------------------------------------
# Patient & study selection
# ---------------------------------------------------------------------------

patients = patient_service.list_patients()
if not patients:
    st.info("No patients found. Create one on the Home page.")
    st.stop()

patient_names = {p.id: p.name for p in patients}
selected_pid = st.sidebar.selectbox(
    "Select Patient",
    options=[p.id for p in patients],
    format_func=lambda pid: patient_names[pid],
)

patient = patient_service.get_patient(selected_pid)
if patient is None:
    st.error("Patient not found.")
    st.stop()

studies = study_service.list_studies(patient.id)
if not studies:
    st.info(f"No studies for **{patient.name}**. Start one on the New Study page.")
    st.stop()

study_labels = {s.id: f"{s.source_file or s.source} — {s.status}" for s in studies}
selected_sid = st.sidebar.selectbox(
    "Select Study",
    options=[s.id for s in studies],
    format_func=lambda sid: study_labels[sid],
)

study = study_service.get_study(selected_sid)

# ---------------------------------------------------------------------------
# Study info
# ---------------------------------------------------------------------------

st.sidebar.divider()
st.sidebar.markdown(f"**Patient:** {patient.name}")
st.sidebar.markdown(f"**Study:** {study.source_file or study.source}")
st.sidebar.markdown(f"**Status:** {study.status}")
if study.duration_sec:
    hrs = study.duration_sec / 3600
    st.sidebar.markdown(f"**Duration:** {hrs:.1f} hours")

# Live Pi status indicator for Pi studies
if study.source == "pi" and study.device_id:
    st.sidebar.divider()
    try:
        resp = requests.get(f"{FASTAPI_URL}/api/devices", timeout=3)
        if resp.status_code == 200:
            live_devices = resp.json()
            device_info = next(
                (d for d in live_devices if d.get("db_id") == study.device_id),
                None,
            )
            if device_info:
                dev_status = device_info.get("status", "offline")
                if dev_status == "streaming":
                    st.sidebar.success(f"Pi: {device_info.get('name', 'device')} — streaming")
                elif dev_status == "connected":
                    st.sidebar.warning(f"Pi: {device_info.get('name', 'device')} — connected (idle)")
                else:
                    st.sidebar.error(f"Pi: {device_info.get('name', 'device')} — offline")
            else:
                st.sidebar.error("Pi device not found")
        else:
            st.sidebar.warning("Could not query device status")
    except requests.ConnectionError:
        st.sidebar.warning("WebSocket server unreachable")

# Auto-refresh for active live studies
if study.source == "pi" and study.status == "active":
    st.sidebar.divider()
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)
    if auto_refresh:
        import time as _time
        _time.sleep(30)
        st.rerun()

# ---------------------------------------------------------------------------
# Fetch data
# ---------------------------------------------------------------------------

MO_BANDS = ["0.5_3hz", "3_8hz", "8_15hz", "15_30hz"]
BAND_LABELS = {
    "0.5_3hz": "Delta (0.5-3 Hz)",
    "3_8hz": "Theta (3-8 Hz)",
    "8_15hz": "Alpha (8-15 Hz)",
    "15_30hz": "Beta (15-30 Hz)",
}
BAND_COLORS = {
    "0.5_3hz": "#1f77b4",
    "3_8hz": "#ff7f0e",
    "8_15hz": "#2ca02c",
    "15_30hz": "#d62728",
}

# Per-window (algorithm window level) data
q_data = {}
p_data = {}
for band in MO_BANDS:
    ts_q, vals_q = study_service.get_feature_timeseries(selected_sid, f"mo_q_{band}")
    ts_p, vals_p = study_service.get_feature_timeseries(selected_sid, f"mo_p_{band}")
    q_data[band] = (ts_q, vals_q)
    p_data[band] = (ts_p, vals_p)

ts_count, vals_count = study_service.get_feature_timeseries(selected_sid, "mo_count")

# Hourly summary
hourly = study_service.get_hourly_summary(selected_sid, patient.bucket_size_seconds)

# Alerts
alerts = study_service.get_alerts(selected_sid)

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

if not ts_count:
    st.warning("No feature data for this study yet.")
    st.stop()

# Convert to minutes for display
def to_minutes(ts_list):
    return [t / 60.0 for t in ts_list]

# --- Q-Score Timeline (per algorithm window) ---
st.subheader("MO Q-Score by Algorithm Window")
if HAS_PLOTLY:
    fig_q = go.Figure()
    for band in MO_BANDS:
        ts, vals = q_data[band]
        if ts:
            fig_q.add_trace(go.Scatter(
                x=to_minutes(ts),
                y=vals,
                mode="lines+markers",
                name=BAND_LABELS[band],
                line=dict(color=BAND_COLORS[band]),
                marker=dict(size=4),
            ))
    fig_q.update_layout(
        xaxis_title="Time (minutes)",
        yaxis_title="Q-Score",
        height=400,
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig_q, use_container_width=True)
else:
    st.line_chart({BAND_LABELS[b]: q_data[b][1] for b in MO_BANDS if q_data[b][0]})

# --- P-Value Timeline ---
st.subheader("MO P-Values by Algorithm Window")
if HAS_PLOTLY:
    fig_p = go.Figure()
    for band in MO_BANDS:
        ts, vals = p_data[band]
        if ts:
            fig_p.add_trace(go.Scatter(
                x=to_minutes(ts),
                y=vals,
                mode="lines+markers",
                name=BAND_LABELS[band],
                line=dict(color=BAND_COLORS[band]),
                marker=dict(size=4),
            ))
    fig_p.add_hline(y=0.05, line_dash="dash", line_color="gray",
                     annotation_text="p = 0.05")
    fig_p.update_layout(
        xaxis_title="Time (minutes)",
        yaxis_title="P-Value",
        height=350,
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig_p, use_container_width=True)

# --- MO Count per Algorithm Window ---
st.subheader("Significant MO Count per Window")
if HAS_PLOTLY:
    fig_count = go.Figure()
    fig_count.add_trace(go.Bar(
        x=to_minutes(ts_count),
        y=vals_count,
        marker_color="#636efa",
        name="Significant Bands",
    ))
    if patient.mo_count_threshold:
        fig_count.add_hline(
            y=patient.mo_count_threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Threshold = {patient.mo_count_threshold}",
        )
    fig_count.update_layout(
        xaxis_title="Time (minutes)",
        yaxis_title="# Significant Bands (p < 0.05)",
        height=350,
    )
    st.plotly_chart(fig_count, use_container_width=True)
else:
    st.bar_chart(dict(zip(to_minutes(ts_count), vals_count)))

# --- Hourly Summary Table ---
if hourly:
    st.subheader("Hourly Aggregation")
    table_data = []
    for h in hourly:
        row = {"Hour": h["label"]}
        row["MO Count (total)"] = h.get("mo_count_total", 0)
        for band in MO_BANDS:
            row[f"Q {BAND_LABELS[band]}"] = round(h.get(f"mo_q_{band}_mean", 0), 4)
        table_data.append(row)
    st.dataframe(table_data, use_container_width=True)

# --- Alerts ---
if alerts:
    st.subheader(f"Alerts ({len(alerts)})")
    for a in alerts:
        mins = a.timestamp / 60
        st.warning(
            f"**{a.alert_type.upper()}** at {mins:.0f} min "
            f"({a.bucket_time}) — value {a.actual_value:.0f} "
            f"exceeded threshold {a.threshold:.0f}"
        )
else:
    st.info("No alerts triggered for this study.")
