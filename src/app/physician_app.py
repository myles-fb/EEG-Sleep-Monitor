"""
Physician Dashboard — Home Page

Patient list, create new patient, and navigation to study views.
Run with:  streamlit run src/app/physician_app.py
"""

import sys
from pathlib import Path

# Ensure src/ is importable
_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import streamlit as st
import pandas as pd

from models import init_db, Patient, Device  # noqa: F401 — Device import ensures table creation
from services import patient_service

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

st.set_page_config(page_title="EEG Sleep Monitor", page_icon="🧠", layout="wide")
init_db()

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------

st.title("EEG Sleep Monitor")
st.caption("Physician dashboard for MO detection and sleep EEG analysis")

st.divider()

# ---------------------------------------------------------------------------
# Create New Patient
# ---------------------------------------------------------------------------

with st.expander("Create New Patient Profile", expanded=False):
    with st.form("new_patient", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Patient Name *", placeholder="e.g. Patient 001")
            age = st.number_input("Age", min_value=0, max_value=120, value=0, help="0 = not specified")
        with col2:
            study_type = st.selectbox("Study Type", ["single_night", "multi_night"])
            window_sec = st.selectbox(
                "Algorithm Window",
                [120, 300, 600],
                index=1,
                format_func=lambda x: f"{x // 60} min",
            )

        col3, col4 = st.columns(2)
        with col3:
            bucket_sec = st.selectbox(
                "Dashboard Bucket",
                [1800, 3600, 7200],
                index=1,
                format_func=lambda x: f"{x // 3600} hr" if x >= 3600 else f"{x // 60} min",
            )
            mo_threshold = st.number_input(
                "MO Count Alert Threshold",
                min_value=0,
                value=0,
                help="0 = no alerts",
            )
        with col4:
            enable_mo = st.checkbox("MO Detection", value=True)
            enable_env = st.checkbox("Envelope Spectrogram", value=True)
            enable_full = st.checkbox("Full-signal Spectrogram", value=True)
            save_raw = st.checkbox("Save Raw EEG", value=False)

        submitted = st.form_submit_button("Create Patient", use_container_width=True)
        if submitted:
            if not name.strip():
                st.error("Patient name is required.")
            else:
                patient_service.create_patient(
                    name=name.strip(),
                    age=age if age > 0 else None,
                    study_type=study_type,
                    window_size_seconds=window_sec,
                    bucket_size_seconds=bucket_sec,
                    enable_mo_detection=enable_mo,
                    enable_envelope_spectrogram=enable_env,
                    enable_full_spectrogram=enable_full,
                    save_raw_eeg=save_raw,
                    mo_count_threshold=mo_threshold if mo_threshold > 0 else None,
                )
                st.success(f"Patient **{name}** created.")
                st.rerun()

# ---------------------------------------------------------------------------
# Patient List
# ---------------------------------------------------------------------------

st.subheader("Patients")

patients = patient_service.list_patients()

if not patients:
    st.info("No patients yet. Create one above to get started.")
else:
    for p in patients:
        n_studies = len(p.studies) if p.studies else 0
        status_label = f"{n_studies} {'study' if n_studies == 1 else 'studies'}"
        with st.container(border=True):
            col_name, col_info, col_actions = st.columns([3, 4, 2])
            with col_name:
                st.markdown(f"**{p.name}**")
                st.caption(f"ID: `{p.id[:8]}...`")
            with col_info:
                age_str = f"Age {p.age}" if p.age else "Age N/A"
                st.text(f"{age_str}  |  {p.study_type}  |  {status_label}")
                features = []
                if p.enable_mo_detection:
                    features.append("MO")
                if p.enable_envelope_spectrogram:
                    features.append("Env")
                if p.enable_full_spectrogram:
                    features.append("Full")
                st.caption(f"Features: {', '.join(features) if features else 'None'}")
            with col_actions:
                if st.button("Delete", key=f"del_{p.id}", type="secondary"):
                    patient_service.delete_patient(p.id)
                    st.rerun()
