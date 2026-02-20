"""
Patient Dashboard — View study results, Q-score trends, MO counts, and alerts.

Supports multi-channel and single-channel studies with optional spectrogram
and diagnostic comparison tabs.
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
from services.spectrogram_service import list_available_channels, load_full_spectrogram
from app.viz_helpers import (
    create_spectrogram_heatmap,
    create_band_envelope_plot,
    create_q_heatmap,
    create_side_by_side,
    BAND_COLORS,
    BAND_DISPLAY,
)

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
# Study info sidebar
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
# Constants
# ---------------------------------------------------------------------------

MO_BANDS = ["0.5_3hz", "3_8hz", "8_15hz", "15_30hz"]
BAND_LABELS = BAND_DISPLAY

# ---------------------------------------------------------------------------
# Detect multi-channel vs single-channel
# ---------------------------------------------------------------------------

study_channels = study_service.get_study_channels(selected_sid)
is_multi_channel = len(study_channels) > 1
spec_channels = list_available_channels(selected_sid)
has_spectrograms = len(spec_channels) > 0

# ---------------------------------------------------------------------------
# Helper: fetch data for a specific channel (or all)
# ---------------------------------------------------------------------------

def to_minutes(ts_list):
    return [t / 60.0 for t in ts_list]


def fetch_channel_data(study_id, channel_index=None):
    """Fetch Q, P, count data for a specific channel (or all if None)."""
    q_data, p_data = {}, {}
    for band in MO_BANDS:
        ts_q, vals_q = study_service.get_feature_timeseries_by_channel(
            study_id, f"mo_q_{band}", channel_index)
        ts_p, vals_p = study_service.get_feature_timeseries_by_channel(
            study_id, f"mo_p_{band}", channel_index)
        q_data[band] = (ts_q, vals_q)
        p_data[band] = (ts_p, vals_p)
    ts_count, vals_count = study_service.get_feature_timeseries_by_channel(
        study_id, "mo_count", channel_index)
    return q_data, p_data, ts_count, vals_count


def render_standard_charts(q_data, p_data, ts_count, vals_count, hourly, alerts):
    """Render Q-score, P-value, MO count charts, hourly table, and alerts."""
    if not ts_count:
        st.warning("No feature data available.")
        return

    if not HAS_PLOTLY:
        st.warning("Install plotly for interactive charts.")
        return

    # Q-Score Timeline
    st.subheader("MO Q-Score by Algorithm Window")
    fig_q = go.Figure()
    for band in MO_BANDS:
        ts, vals = q_data[band]
        if ts:
            fig_q.add_trace(go.Scatter(
                x=to_minutes(ts), y=vals,
                mode="lines+markers",
                name=BAND_LABELS[band],
                line=dict(color=BAND_COLORS[band]),
                marker=dict(size=4),
            ))
    fig_q.update_layout(
        xaxis_title="Time (minutes)", yaxis_title="Q-Score",
        height=400, legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig_q, use_container_width=True)

    # P-Value Timeline
    st.subheader("MO P-Values by Algorithm Window")
    fig_p = go.Figure()
    for band in MO_BANDS:
        ts, vals = p_data[band]
        if ts:
            fig_p.add_trace(go.Scatter(
                x=to_minutes(ts), y=vals,
                mode="lines+markers",
                name=BAND_LABELS[band],
                line=dict(color=BAND_COLORS[band]),
                marker=dict(size=4),
            ))
    fig_p.add_hline(y=0.05, line_dash="dash", line_color="gray",
                    annotation_text="p = 0.05")
    fig_p.update_layout(
        xaxis_title="Time (minutes)", yaxis_title="P-Value",
        height=350, legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig_p, use_container_width=True)

    # MO Count
    st.subheader("Significant MO Count per Window")
    fig_count = go.Figure()
    fig_count.add_trace(go.Bar(
        x=to_minutes(ts_count), y=vals_count,
        marker_color="#636efa", name="Significant Bands",
    ))
    if patient.mo_count_threshold:
        fig_count.add_hline(
            y=patient.mo_count_threshold, line_dash="dash", line_color="red",
            annotation_text=f"Threshold = {patient.mo_count_threshold}",
        )
    fig_count.update_layout(
        xaxis_title="Time (minutes)",
        yaxis_title="# Significant Bands (p < 0.05)",
        height=350,
    )
    st.plotly_chart(fig_count, use_container_width=True)

    # Hourly Summary Table
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

    # Alerts
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

# ---------------------------------------------------------------------------
# Spectrogram tab content
# ---------------------------------------------------------------------------

def render_spectrograms_tab():
    """Render the Spectrograms tab with channel selector and stacked plots."""
    if not spec_channels:
        st.info("No spectrogram data available for this study.")
        return

    ch_options = []
    ch_label_map = {c["index"]: c.get("label") for c in study_channels} if study_channels else {}
    for ch in spec_channels:
        label = ch_label_map.get(ch, f"Channel {ch}")
        ch_options.append((ch, f"Ch {ch}: {label}"))

    selected_ch = st.selectbox(
        "Select Channel",
        options=[c[0] for c in ch_options],
        format_func=lambda ci: next(lbl for idx, lbl in ch_options if idx == ci),
        key="spec_ch_select",
    )

    spec_data = load_full_spectrogram(selected_sid, selected_ch)
    if spec_data is None:
        st.warning("Could not load spectrogram data.")
        return

    S = spec_data["S"]
    T = spec_data["T"]
    F = spec_data["F"]

    # Extract band envelopes
    band_envs = {k: v for k, v in spec_data.items() if k.startswith("env_")}

    # 1. Spectrogram heatmap
    fig_spec = create_spectrogram_heatmap(S, T, F, title="Power Spectrogram")
    st.plotly_chart(fig_spec, use_container_width=True)

    # 2. Band envelopes
    if band_envs:
        fig_env = create_band_envelope_plot(T, band_envs, title="Band Envelopes")
        st.plotly_chart(fig_env, use_container_width=True)

    # 3. Q-value heatmap
    q_data_ch, _, ts_count_ch, _ = fetch_channel_data(selected_sid, selected_ch)
    if ts_count_ch:
        fig_q = create_q_heatmap(ts_count_ch, q_data_ch, title="Q-Score Heatmap")
        st.plotly_chart(fig_q, use_container_width=True)

# ---------------------------------------------------------------------------
# Diagnostic comparison tab content
# ---------------------------------------------------------------------------

def render_diagnostic_tab():
    """Render the Diagnostic Comparison tab with side-by-side plots."""
    if not spec_channels:
        st.info("No spectrogram data available for diagnostic comparison.")
        return

    ch_label_map = {c["index"]: c.get("label") for c in study_channels} if study_channels else {}
    ch_options = []
    for ch in spec_channels:
        label = ch_label_map.get(ch, f"Channel {ch}")
        ch_options.append((ch, f"Ch {ch}: {label}"))

    selected_ch = st.selectbox(
        "Select Channel",
        options=[c[0] for c in ch_options],
        format_func=lambda ci: next(lbl for idx, lbl in ch_options if idx == ci),
        key="diag_ch_select",
    )

    # Get Q data for this channel
    q_data_ch, p_data_ch, ts_count_ch, _ = fetch_channel_data(selected_sid, selected_ch)
    if not ts_count_ch:
        st.warning("No feature data for this channel.")
        return

    # Time bucket selector
    bucket_options = ts_count_ch
    if not bucket_options:
        st.warning("No time windows available.")
        return

    selected_bucket = st.select_slider(
        "Select Time Window",
        options=bucket_options,
        format_func=lambda t: f"{t / 60:.1f} min",
        key="diag_bucket",
    )

    # Load spectrogram data
    spec_data = load_full_spectrogram(selected_sid, selected_ch)
    if spec_data is not None:
        T_spec = spec_data["T"]
        band_envs = {k: v for k, v in spec_data.items() if k.startswith("env_")}

        if band_envs:
            fig = create_side_by_side(
                T_spec, band_envs, ts_count_ch, q_data_ch,
                highlight_time=selected_bucket,
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No spectrogram data available for overlay.")

    # Detail panel: Q and P values per band for selected bucket
    st.subheader(f"Window Detail: {selected_bucket / 60:.1f} min")
    cols = st.columns(len(MO_BANDS))
    for col, band in zip(cols, MO_BANDS):
        with col:
            ts_q, vals_q = q_data_ch[band]
            ts_p, vals_p = p_data_ch[band]
            q_map = dict(zip(ts_q, vals_q))
            p_map = dict(zip(ts_p, vals_p))
            q_val = q_map.get(selected_bucket)
            p_val = p_map.get(selected_bucket)
            st.markdown(f"**{BAND_LABELS[band]}**")
            q_str = f"{q_val:.4f}" if q_val is not None else "N/A"
            p_str = f"{p_val:.4f}" if p_val is not None else "N/A"
            if p_val is not None and p_val < 0.05:
                st.markdown(f"Q: {q_str}  \nP: :red[**{p_str}**]")
            else:
                st.markdown(f"Q: {q_str}  \nP: {p_str}")

# ---------------------------------------------------------------------------
# Build tabs
# ---------------------------------------------------------------------------

# Get summary data (all channels combined for multi-channel)
q_data_all, p_data_all, ts_count_all, vals_count_all = fetch_channel_data(selected_sid, None)
hourly = study_service.get_hourly_summary(selected_sid, patient.bucket_size_seconds)
alerts = study_service.get_alerts(selected_sid)

if not ts_count_all:
    st.warning("No feature data for this study yet.")
    st.stop()

# Build tab list
tab_names = []
if is_multi_channel:
    tab_names.append("Summary")
    for ch_info in study_channels:
        idx = ch_info["index"]
        label = ch_info.get("label") or f"Ch {idx}"
        tab_names.append(f"Ch {idx}: {label}")
else:
    tab_names.append("Overview")

if has_spectrograms:
    tab_names.append("Spectrograms")
    tab_names.append("Diagnostic Comparison")

tabs = st.tabs(tab_names)
tab_idx = 0

# --- Summary / Overview tab ---
with tabs[tab_idx]:
    if is_multi_channel:
        st.subheader("Summary (All Channels)")
    render_standard_charts(q_data_all, p_data_all, ts_count_all, vals_count_all, hourly, alerts)
tab_idx += 1

# --- Per-channel tabs (multi-channel only) ---
if is_multi_channel:
    for ch_info in study_channels:
        with tabs[tab_idx]:
            ch_idx = ch_info["index"]
            ch_label = ch_info.get("label") or f"Channel {ch_idx}"
            st.subheader(f"Channel {ch_idx}: {ch_label}")
            ch_q, ch_p, ch_ts, ch_vals = fetch_channel_data(selected_sid, ch_idx)
            render_standard_charts(ch_q, ch_p, ch_ts, ch_vals, None, None)
        tab_idx += 1

# --- Spectrograms tab ---
if has_spectrograms:
    with tabs[tab_idx]:
        render_spectrograms_tab()
    tab_idx += 1

    # --- Diagnostic Comparison tab ---
    with tabs[tab_idx]:
        render_diagnostic_tab()
    tab_idx += 1
