"""
New Study — Load an EDF file, run MOs processing, and store results.
"""

import sys
import tempfile
from pathlib import Path

_src = Path(__file__).resolve().parent.parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

# Also ensure project root is on path (for processing.mos)
_project_root = _src.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import streamlit as st

import requests

from models import init_db
from services import patient_service, study_service, config_service, device_service

init_db()

st.set_page_config(page_title="New Study", page_icon="🔬", layout="wide")
st.title("Start New Study")

FASTAPI_URL = "http://localhost:8765"

# ---------------------------------------------------------------------------
# Patient selection
# ---------------------------------------------------------------------------

patients = patient_service.list_patients()
if not patients:
    st.info("No patients found. Create one on the Home page first.")
    st.stop()

patient_names = {p.id: p.name for p in patients}
selected_pid = st.selectbox(
    "Select Patient",
    options=[p.id for p in patients],
    format_func=lambda pid: patient_names[pid],
)

patient = patient_service.get_patient(selected_pid)
if patient is None:
    st.stop()

# Show patient config summary
with st.expander("Patient Configuration"):
    config = config_service.generate_pi_config(patient)
    st.json(config)

st.divider()

# ---------------------------------------------------------------------------
# Data source selection
# ---------------------------------------------------------------------------

st.subheader("Data Source")

data_source = st.radio(
    "How to acquire EEG data?",
    ["EDF File", "Raspberry Pi (Live)"],
    horizontal=True,
)

if data_source == "Raspberry Pi (Live)":
    # -----------------------------------------------------------------------
    # Pi live streaming mode
    # -----------------------------------------------------------------------
    devices = device_service.list_devices()
    if not devices:
        st.warning("No Pi devices registered. Go to the **Devices** page to register one.")
        st.stop()

    dev_names = {d.id: f"{d.name} ({d.device_key})" for d in devices}
    selected_dev_id = st.selectbox(
        "Select Device",
        options=[d.id for d in devices],
        format_func=lambda did: dev_names[did],
    )

    notes = st.text_area("Study Notes (optional)", placeholder="e.g. Overnight monitoring")

    st.divider()

    if st.button("Start Live Study", type="primary", use_container_width=True):
        study = study_service.create_live_study(
            patient_id=patient.id,
            device_id=selected_dev_id,
            notes=notes.strip() if notes else None,
        )
        device_service.assign_patient(selected_dev_id, patient.id, study.id)

        # Push config to the Pi via FastAPI
        try:
            requests.post(f"{FASTAPI_URL}/api/devices/{selected_dev_id}/config", timeout=3)
            st.success(
                f"Live study started! Study ID: `{study.id[:8]}...`\n\n"
                "Config pushed to Pi. Go to the **Dashboard** to view incoming data."
            )
        except requests.ConnectionError:
            st.warning(
                f"Live study created (ID: `{study.id[:8]}...`) but could not reach "
                f"WebSocket server at `{FASTAPI_URL}`. Ensure it is running."
            )

else:
    # -----------------------------------------------------------------------
    # EDF file mode (existing workflow)
    # -----------------------------------------------------------------------
    st.subheader("EDF Data Source")

    source_mode = st.radio(
        "How to load EEG data?",
        ["Select existing EDF file", "Upload EDF file"],
        horizontal=True,
    )

    edf_path = None

    if source_mode == "Select existing EDF file":
        data_dir = Path(__file__).resolve().parent.parent.parent.parent / "data"
        edf_files = sorted(data_dir.glob("*.edf"))
        if edf_files:
            selected_edf = st.selectbox(
                "Available EDF files",
                options=edf_files,
                format_func=lambda p: p.name,
            )
            edf_path = str(selected_edf)
        else:
            st.warning(f"No EDF files found in `{data_dir}`")
    else:
        uploaded = st.file_uploader("Upload an EDF file", type=["edf"])
        if uploaded:
            tmp = tempfile.NamedTemporaryFile(suffix=".edf", delete=False)
            tmp.write(uploaded.getbuffer())
            tmp.flush()
            edf_path = tmp.name
            st.success(f"Uploaded: {uploaded.name}")

    # -----------------------------------------------------------------------
    # Processing configuration
    # -----------------------------------------------------------------------

    st.divider()
    st.subheader("Processing Settings")

    channel_index = st.number_input(
        "Bipolar Channel Index",
        min_value=0,
        max_value=17,
        value=0,
        help="0-17 for LB-18 bipolar montage channels",
    )

    notes = st.text_area("Study Notes (optional)", placeholder="e.g. Baseline night study")

    # -----------------------------------------------------------------------
    # Run processing
    # -----------------------------------------------------------------------

    st.divider()

    if edf_path is None:
        st.info("Select or upload an EDF file to begin.")
        st.stop()

    if st.button("Run MOs Processing", type="primary", use_container_width=True):
        # Create study record
        source_file = Path(edf_path).name
        study = study_service.create_study(
            patient_id=patient.id,
            source="edf",
            source_file=source_file,
            notes=notes.strip() if notes else None,
        )

        progress = st.progress(0, text="Initializing...")
        status_text = st.empty()

        def on_progress(current, total):
            pct = current / total
            progress.progress(pct, text=f"Processing window {current}/{total}")

        try:
            summary = study_service.process_edf(
                study_id=study.id,
                edf_path=edf_path,
                patient=patient,
                n_surrogates=1,
                channel_index=channel_index,
                progress_callback=on_progress,
            )

            progress.progress(1.0, text="Complete!")

            st.success("Study processing complete!")
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                st.metric("Windows Processed", summary["n_windows"])
            with col_b:
                duration_hrs = summary["duration_sec"] / 3600
                st.metric("Recording Duration", f"{duration_hrs:.1f} hrs")
            with col_c:
                st.metric("Feature Records", summary["total_records"])
            with col_d:
                st.metric("Alerts", summary["total_alerts"])

            st.info("Go to the **Dashboard** page to view results.")

        except Exception as e:
            st.error(f"Processing failed: {e}")
            # Mark study as failed
            study_service.complete_study(study.id)
