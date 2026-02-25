"""
Per-Channel Page — Full and band-limited spectrograms, envelope spectrograms,
Q-score heatmap, and dominant frequency tracking for a single channel.
"""

import sys
from pathlib import Path

_src = Path(__file__).resolve().parent.parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import streamlit as st
import numpy as np

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from models import init_db
from services import patient_service, study_service
from services.spectrogram_service import list_available_channels, load_full_spectrogram
from app.viz_helpers import (
    create_spectrogram_heatmap,
    create_band_limited_spectrogram,
    create_envelope_spectrogram,
    create_q_heatmap_with_significance,
    create_dominant_freq_chart,
    BAND_DISPLAY,
    BAND_COLORS,
    BAND_FREQS,
)

MO_BANDS = ["0.5_3hz", "3_8hz", "8_15hz", "15_30hz"]
ENV_KEY_PREFIX = "env_"

init_db()

st.set_page_config(page_title="Per-Channel", page_icon="📈", layout="wide")
st.title("Per-Channel View")

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
    st.info(f"No studies for **{patient.name}**.")
    st.stop()

study_labels = {s.id: f"{s.source_file or s.source} — {s.status}" for s in studies}
selected_sid = st.sidebar.selectbox(
    "Select Study",
    options=[s.id for s in studies],
    format_func=lambda sid: study_labels[sid],
)

study = study_service.get_study(selected_sid)

st.sidebar.divider()
st.sidebar.markdown(f"**Patient:** {patient.name}")
st.sidebar.markdown(f"**Study:** {study.source_file or study.source}")
st.sidebar.markdown(f"**Status:** {study.status}")
if study.duration_sec:
    st.sidebar.markdown(f"**Duration:** {study.duration_sec / 3600:.1f} hours")

# ---------------------------------------------------------------------------
# Feature set menu
# ---------------------------------------------------------------------------

st.sidebar.divider()
feature_sets = st.sidebar.multiselect(
    "Feature Sets",
    options=["MOs", "CAP (coming soon)"],
    default=["MOs"],
)
show_mos = "MOs" in feature_sets

# ---------------------------------------------------------------------------
# Channel selection
# ---------------------------------------------------------------------------

study_channels = study_service.get_study_channels(selected_sid)
spec_channels = list_available_channels(selected_sid)
ch_label_map = {c["index"]: c.get("label") or f"Ch {c['index']}" for c in study_channels}

all_ch_indices = [c["index"] for c in study_channels] if study_channels else spec_channels

st.sidebar.divider()
if not all_ch_indices:
    st.warning("No channels found for this study.")
    st.stop()

selected_ch = st.sidebar.selectbox(
    "Channel",
    options=all_ch_indices,
    format_func=lambda ci: ch_label_map.get(ci, f"Ch {ci}"),
)
ch_label = ch_label_map.get(selected_ch, f"Ch {selected_ch}")

# Band toggle
st.sidebar.divider()
selected_bands = st.sidebar.multiselect(
    "Bands",
    options=MO_BANDS,
    default=MO_BANDS,
    format_func=lambda b: BAND_DISPLAY.get(b, b),
)

# Envelope spectrogram window control
# Fs_env is fixed at ~1/6 Hz (one sample per 6-second spectrogram step).
# Larger window → finer frequency resolution, coarser time resolution.
st.sidebar.divider()
st.sidebar.markdown("**Envelope Spectrogram Window**")
_FS_ENV = 1.0 / 6.0  # Hz — fixed by first-order spectrogram step
env_window_min = st.sidebar.slider(
    "Window length (min)",
    min_value=5,
    max_value=60,
    value=40,
    step=5,
)
env_window_samples = round(env_window_min * 60 * _FS_ENV)
_env_step_samples = max(1, env_window_samples // 4)
_freq_res_mhz = (_FS_ENV / env_window_samples) * 1000.0
_time_res_min = (_env_step_samples / _FS_ENV) / 60.0
st.sidebar.caption(
    f"{env_window_samples} samples · "
    f"Freq. resolution: **{_freq_res_mhz:.1f} mHz** · "
    f"Time resolution: **{_time_res_min:.1f} min**"
)

# P-value cutoff
st.sidebar.divider()
p_cutoff = st.sidebar.slider(
    "P-value cutoff",
    min_value=0.001,
    max_value=0.5,
    value=0.05,
    step=0.001,
    format="%.3f",
)

# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

if not show_mos:
    st.info("Select a feature set in the sidebar to display data.")
    st.stop()

if not HAS_PLOTLY:
    st.warning("Install plotly for interactive charts.")
    st.stop()

# ---------------------------------------------------------------------------
# Load spectrogram data for selected channel
# ---------------------------------------------------------------------------

spec_data = None
if selected_ch in spec_channels:
    spec_data = load_full_spectrogram(selected_sid, selected_ch)

# ---------------------------------------------------------------------------
# Fetch Q / P / dominant freq data for selected channel
# ---------------------------------------------------------------------------

q_data, p_data, dom_freq_data = {}, {}, {}
for band in selected_bands:
    ts_q, vals_q = study_service.get_feature_timeseries_by_channel(
        selected_sid, f"mo_q_{band}", selected_ch)
    ts_p, vals_p = study_service.get_feature_timeseries_by_channel(
        selected_sid, f"mo_p_{band}", selected_ch)
    ts_df, vals_df = study_service.get_feature_timeseries_by_channel(
        selected_sid, f"mo_dom_freq_{band}", selected_ch)
    q_data[band] = (ts_q, vals_q)
    p_data[band] = (ts_p, vals_p)
    dom_freq_data[band] = (ts_df, vals_df)

ts_count, _ = study_service.get_feature_timeseries_by_channel(
    selected_sid, "mo_count", selected_ch)

# ---------------------------------------------------------------------------
# Section 1: Full-signal spectrogram
# ---------------------------------------------------------------------------

st.subheader(f"Channel: {ch_label}")

if spec_data is None:
    st.info("No spectrogram data for this channel. Enable spectrogram saving when creating the study.")
else:
    S = spec_data["S"]
    T = spec_data["T"]
    F = spec_data["F"]

    with st.expander("Full-Signal Spectrogram", expanded=True):
        fig_full = create_spectrogram_heatmap(S, T, F, title="Full-Signal Spectrogram")
        st.plotly_chart(fig_full, use_container_width=True)

    # ---------------------------------------------------------------------------
    # Section 2: Band-limited power spectrograms
    # ---------------------------------------------------------------------------

    if selected_bands:
        st.subheader("Band-Limited Power Spectrograms")
        cols_per_row = 2
        for row_start in range(0, len(selected_bands), cols_per_row):
            row_bands = selected_bands[row_start:row_start + cols_per_row]
            cols = st.columns(len(row_bands))
            for col, band in zip(cols, row_bands):
                with col:
                    fig_band = create_band_limited_spectrogram(S, T, F, band)
                    st.plotly_chart(fig_band, use_container_width=True)

    # ---------------------------------------------------------------------------
    # Section 3: Envelope spectrograms (2nd-order)
    # ---------------------------------------------------------------------------

    if selected_bands:
        st.subheader("Envelope Spectrograms (2nd-order)")
        for row_start in range(0, len(selected_bands), cols_per_row):
            row_bands = selected_bands[row_start:row_start + cols_per_row]
            cols = st.columns(len(row_bands))
            for col, band in zip(cols, row_bands):
                with col:
                    env_key = f"{ENV_KEY_PREFIX}{band}"
                    env = spec_data.get(env_key)
                    if env is not None:
                        fig_env = create_envelope_spectrogram(
                            T, env,
                            title=f"Envelope Spectrogram — {BAND_DISPLAY.get(band, band)}",
                            window_samples=env_window_samples,
                        )
                        st.plotly_chart(fig_env, use_container_width=True)
                    else:
                        st.info(f"No envelope data for {BAND_DISPLAY.get(band, band)}")

# ---------------------------------------------------------------------------
# Section 4: Q-Score Heatmap with significance
# ---------------------------------------------------------------------------

if ts_count and selected_bands:
    st.subheader("Q-Score Heatmap")
    st.caption(f"Gray cells: p >= {p_cutoff:.3f}. Vivid cells: p < {p_cutoff:.3f}.")
    fig_q = create_q_heatmap_with_significance(
        ts_count, q_data, p_data,
        p_cutoff=p_cutoff,
        title=f"Q-Score — {ch_label}",
    )
    st.plotly_chart(fig_q, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 5: Dominant Frequency Tracking
# ---------------------------------------------------------------------------

if any(ts for ts, _ in dom_freq_data.values()):
    st.subheader("Dominant Modulation Frequency")
    fig_dom = create_dominant_freq_chart(dom_freq_data, title=f"Dominant Freq — {ch_label}")
    st.plotly_chart(fig_dom, use_container_width=True)
