"""
Summary Dashboard — Channel-averaged spectrogram, per-channel Q-heatmaps,
and dominant modulation frequency tracking.
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
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from models import init_db
from services import patient_service, study_service
from services.spectrogram_service import list_available_channels, load_full_spectrogram
from app.viz_helpers import (
    create_q_heatmap_with_significance,
    create_channel_averaged_spectrogram,
    create_envelope_spectrogram,
    create_dominant_freq_chart,
    BAND_DISPLAY,
    BAND_COLORS,
)

FASTAPI_URL = "http://localhost:8765"
MO_BANDS = ["0.5_3hz", "3_8hz", "8_15hz", "15_30hz"]

init_db()

st.set_page_config(page_title="Summary", page_icon="📊", layout="wide")
st.title("Summary")

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
# Study info & Pi status
# ---------------------------------------------------------------------------

st.sidebar.divider()
st.sidebar.markdown(f"**Patient:** {patient.name}")
st.sidebar.markdown(f"**Study:** {study.source_file or study.source}")
st.sidebar.markdown(f"**Status:** {study.status}")
if study.duration_sec:
    st.sidebar.markdown(f"**Duration:** {study.duration_sec / 3600:.1f} hours")

if study.source == "pi" and study.device_id:
    st.sidebar.divider()
    try:
        resp = requests.get(f"{FASTAPI_URL}/api/devices", timeout=3)
        if resp.status_code == 200:
            device_info = next(
                (d for d in resp.json() if d.get("db_id") == study.device_id), None
            )
            if device_info:
                dev_status = device_info.get("status", "offline")
                if dev_status == "streaming":
                    st.sidebar.success(f"Pi: {device_info.get('name', 'device')} — streaming")
                elif dev_status == "connected":
                    st.sidebar.warning(f"Pi: {device_info.get('name', 'device')} — connected (idle)")
                else:
                    st.sidebar.error(f"Pi: {device_info.get('name', 'device')} — offline")
    except requests.ConnectionError:
        st.sidebar.warning("WebSocket server unreachable")

if study.source == "pi" and study.status == "active":
    st.sidebar.divider()
    if st.sidebar.checkbox("Auto-refresh (30s)", value=False):
        import time as _time
        _time.sleep(30)
        st.rerun()

# ---------------------------------------------------------------------------
# Feature set menu
# ---------------------------------------------------------------------------

st.sidebar.divider()
st.sidebar.markdown("**Feature Sets**")
feature_sets = st.sidebar.multiselect(
    "Active feature sets",
    options=["MOs", "CAP (coming soon)"],
    default=["MOs"],
    help="Select which feature sets to display.",
)
show_mos = "MOs" in feature_sets

# ---------------------------------------------------------------------------
# Channel & band controls
# ---------------------------------------------------------------------------

study_channels = study_service.get_study_channels(selected_sid)
spec_channels = list_available_channels(selected_sid)

ch_label_map = {c["index"]: c.get("label") or f"Ch {c['index']}" for c in study_channels}

all_ch_indices = [c["index"] for c in study_channels] if study_channels else spec_channels
ch_options_fmt = {ci: ch_label_map.get(ci, f"Ch {ci}") for ci in all_ch_indices}

if all_ch_indices:
    st.sidebar.divider()
    selected_channels = st.sidebar.multiselect(
        "Channels (averaged spectrogram)",
        options=all_ch_indices,
        default=all_ch_indices,
        format_func=lambda ci: ch_options_fmt.get(ci, f"Ch {ci}"),
    )
else:
    selected_channels = []

st.sidebar.divider()
selected_bands = st.sidebar.multiselect(
    "Bands",
    options=MO_BANDS,
    default=MO_BANDS,
    format_func=lambda b: BAND_DISPLAY.get(b, b),
)

st.sidebar.divider()
p_cutoff = st.sidebar.slider(
    "P-value cutoff",
    min_value=0.001,
    max_value=0.5,
    value=0.05,
    step=0.001,
    format="%.3f",
    help="Cells where p >= cutoff are shown in gray.",
)

# ---------------------------------------------------------------------------
# Early exit if no feature sets selected
# ---------------------------------------------------------------------------

if not show_mos:
    st.info("Select a feature set in the sidebar to display data.")
    st.stop()

# ---------------------------------------------------------------------------
# Fetch MOs data
# ---------------------------------------------------------------------------

def fetch_channel_q_p(study_id, channel_index=None):
    """Return q_data and p_data dicts for selected bands."""
    q_data, p_data = {}, {}
    for band in selected_bands:
        ts_q, vals_q = study_service.get_feature_timeseries_by_channel(
            study_id, f"mo_q_{band}", channel_index)
        ts_p, vals_p = study_service.get_feature_timeseries_by_channel(
            study_id, f"mo_p_{band}", channel_index)
        q_data[band] = (ts_q, vals_q)
        p_data[band] = (ts_p, vals_p)
    return q_data, p_data


ref_ch = all_ch_indices[0] if all_ch_indices else None
ts_ref, _ = study_service.get_feature_timeseries_by_channel(
    selected_sid, "mo_count", ref_ch)

if not ts_ref:
    st.warning("No feature data for this study yet.")
    st.stop()

if not HAS_PLOTLY:
    st.warning("Install plotly for interactive charts.")
    st.stop()

# ---------------------------------------------------------------------------
# Section 1: Channel-Averaged Spectrogram
# ---------------------------------------------------------------------------

st.subheader("Channel-Averaged Spectrogram")

if not spec_channels:
    st.info("No spectrogram data available. Enable spectrogram saving when creating the study.")
elif not selected_channels:
    st.info("Select at least one channel in the sidebar.")
else:
    ch_to_load = [ci for ci in selected_channels if ci in spec_channels]
    if not ch_to_load:
        st.warning("No spectrogram data for the selected channels.")
    else:
        spec_data_list = []
        for ci in ch_to_load:
            d = load_full_spectrogram(selected_sid, ci)
            if d is not None:
                spec_data_list.append(d)
        if spec_data_list:
            ch_names_str = ", ".join(ch_label_map.get(ci, f"Ch {ci}") for ci in ch_to_load)
            fig_avg = create_channel_averaged_spectrogram(
                spec_data_list,
                title=f"Averaged: {ch_names_str}",
            )
            st.plotly_chart(fig_avg, use_container_width=True)
        else:
            st.warning("Could not load spectrogram data.")

# ---------------------------------------------------------------------------
# Section 2: Envelope Spectrograms (2nd-order, Eq. 1 — Loe et al. 2022)
# ---------------------------------------------------------------------------

st.subheader("Envelope Spectrograms")
st.caption(
    "2nd-order spectrograms of the band-limited power envelope S̄(t) (Eq. 1). "
    "Computed with Tᵥᵥ = 30 s window, 6 s step. Y-axis: modulation frequency (mHz)."
)

if not spec_channels:
    st.info("No spectrogram data available. Enable spectrogram saving when creating the study.")
elif not selected_bands:
    st.info("Select at least one band in the sidebar.")
else:
    # Load envelopes for the selected channels and average across them
    ch_to_load = [ci for ci in (selected_channels or all_ch_indices) if ci in spec_channels]
    if not ch_to_load:
        st.warning("No spectrogram data for the selected channels.")
    else:
        # Collect per-band averaged envelopes
        # spec_data_list may already be loaded above; reload here to keep sections independent
        env_spec_data: dict = {}  # band -> (T, avg_envelope)
        for band in selected_bands:
            env_key = f"env_{band}"
            envs, T_ref = [], None
            for ci in ch_to_load:
                d = load_full_spectrogram(selected_sid, ci)
                if d is not None and env_key in d:
                    envs.append(d[env_key])
                    if T_ref is None:
                        T_ref = d["T"]
            if envs and T_ref is not None:
                avg_env = np.mean(np.stack(envs, axis=0), axis=0)
                env_spec_data[band] = (T_ref, avg_env)

        if not env_spec_data:
            st.info("No envelope data found. Ensure the study was processed with spectrogram saving enabled.")
        else:
            cols_per_row = 2
            band_list = list(env_spec_data.keys())
            for row_start in range(0, len(band_list), cols_per_row):
                row_bands = band_list[row_start:row_start + cols_per_row]
                cols = st.columns(len(row_bands))
                for col, band in zip(cols, row_bands):
                    with col:
                        T_env, avg_env = env_spec_data[band]
                        fig_env = create_envelope_spectrogram(
                            T_env, avg_env,
                            title=f"Envelope Spectrogram — {BAND_DISPLAY.get(band, band)}",
                        )
                        st.plotly_chart(fig_env, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 3: Q-Score Heatmaps (one per channel)
# ---------------------------------------------------------------------------

st.subheader("Q-Score Heatmaps")
st.caption(f"Gray cells: p >= {p_cutoff:.3f} (not significant). Vivid cells: p < {p_cutoff:.3f}.")

if not selected_bands:
    st.info("Select at least one band in the sidebar.")
elif not study_channels:
    q_data, p_data = fetch_channel_q_p(selected_sid, None)
    fig_q = create_q_heatmap_with_significance(
        ts_ref, q_data, p_data, p_cutoff=p_cutoff,
        title="Q-Score Heatmap",
    )
    st.plotly_chart(fig_q, use_container_width=True)
else:
    cols_per_row = 2
    ch_list = study_channels
    for row_start in range(0, len(ch_list), cols_per_row):
        row_chs = ch_list[row_start:row_start + cols_per_row]
        cols = st.columns(len(row_chs))
        for col, ch_info in zip(cols, row_chs):
            with col:
                ci = ch_info["index"]
                label = ch_label_map.get(ci, f"Ch {ci}")
                q_data_ch, p_data_ch = fetch_channel_q_p(selected_sid, ci)
                ts_ch, _ = study_service.get_feature_timeseries_by_channel(
                    selected_sid, "mo_count", ci)
                if ts_ch:
                    fig_ch = create_q_heatmap_with_significance(
                        ts_ch, q_data_ch, p_data_ch,
                        p_cutoff=p_cutoff,
                        title=label,
                    )
                    st.plotly_chart(fig_ch, use_container_width=True)
                else:
                    st.info(f"No data for {label}")

# ---------------------------------------------------------------------------
# Section 3: Dominant Frequency Tracking
# ---------------------------------------------------------------------------

st.subheader("Dominant Modulation Frequency")

dom_freq_data = {}
for band in selected_bands:
    ts_df, vals_df = study_service.get_feature_timeseries_by_channel(
        selected_sid, f"mo_dom_freq_{band}", ref_ch)
    dom_freq_data[band] = (ts_df, vals_df)

if any(ts for ts, _ in dom_freq_data.values()):
    fig_dom = create_dominant_freq_chart(dom_freq_data)
    st.plotly_chart(fig_dom, use_container_width=True)
else:
    st.info("No dominant frequency data available.")

# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------

alerts = study_service.get_alerts(selected_sid)
if alerts:
    st.subheader(f"Alerts ({len(alerts)})")
    for a in alerts:
        mins = a.timestamp / 60
        st.warning(
            f"**{a.alert_type.upper()}** at {mins:.0f} min "
            f"({a.bucket_time}) — value {a.actual_value:.0f} "
            f"exceeded threshold {a.threshold:.0f}"
        )
