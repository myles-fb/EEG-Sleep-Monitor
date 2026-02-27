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

from app.auth import require_auth, show_user_sidebar
user_id = require_auth()
show_user_sidebar()

st.title("Start New Study")

from app.config import FASTAPI_URL

# ---------------------------------------------------------------------------
# Patient selection
# ---------------------------------------------------------------------------

patients = patient_service.list_patients(user_id=user_id)
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
    devices = device_service.list_devices(user_id=user_id)
    if not devices:
        st.warning("No Pi devices registered. Go to the **Devices** page to register one.")
        st.stop()

    dev_names = {d.id: f"{d.name} ({d.device_key})" for d in devices}
    selected_dev_id = st.selectbox(
        "Select Device",
        options=[d.id for d in devices],
        format_func=lambda did: dev_names[did],
    )

    # Board type and channel selection
    st.divider()
    st.subheader("Board & Channel Configuration")

    board_type = st.selectbox(
        "Board Type",
        ["OpenBCI Cyton (8 ch)"],
    )

    pi_all_channels = list(range(8))
    pi_active_channels = st.multiselect(
        "Active Channels",
        options=pi_all_channels,
        default=pi_all_channels,
        format_func=lambda ch: f"Channel {ch + 1}",
    )
    if not pi_active_channels:
        st.warning("Select at least one channel.")
        st.stop()

    notes = st.text_area("Study Notes (optional)", placeholder="e.g. Overnight monitoring")

    st.divider()

    if st.button("Start Live Study", type="primary", use_container_width=True):
        study = study_service.create_live_study(
            patient_id=patient.id,
            device_id=selected_dev_id,
            notes=notes.strip() if notes else None,
        )
        device_service.assign_patient(selected_dev_id, patient.id, study.id)

        # Store channel info on study
        channels_info = [{"index": ch, "label": f"Ch {ch + 1}"} for ch in pi_active_channels]
        study_service.update_study_channels(study.id, channels_info)

        # Push config to the Pi via FastAPI
        try:
            requests.post(f"{FASTAPI_URL}/api/devices/{selected_dev_id}/config", timeout=3)
            st.success(
                f"Live study started! Study ID: `{study.id[:8]}...`\n\n"
                f"Board: {board_type}, Channels: {pi_active_channels}\n\n"
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
    # Processing configuration — channel selection
    # -----------------------------------------------------------------------

    st.divider()
    st.subheader("Processing Settings")

    # LB-18 bipolar montage labels
    LB18_LABELS = [
        "Fp1-F7", "F7-T7", "T7-P7", "P7-O1",
        "Fp1-F3", "F3-C3", "C3-P3", "P3-O1",
        "Fp2-F8", "F8-T8", "T8-P8", "P8-O2",
        "Fp2-F4", "F4-C4", "C4-P4", "P4-O2",
        "Fz-Cz", "Cz-Pz",
    ]

    channel_mode = st.radio(
        "Channel Selection",
        ["All channels (18)", "Select specific channels"],
        horizontal=True,
    )

    if channel_mode == "All channels (18)":
        selected_channels = list(range(18))
        selected_labels = LB18_LABELS[:]
    else:
        options = [f"{i}: {LB18_LABELS[i]}" for i in range(18)]
        chosen = st.multiselect(
            "Select bipolar channels",
            options=options,
            default=[options[0]],
        )
        selected_channels = [int(c.split(":")[0]) for c in chosen]
        selected_labels = [LB18_LABELS[i] for i in selected_channels]

    if not selected_channels:
        st.warning("Select at least one channel.")
        st.stop()

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
                channel_indices=selected_channels,
                channel_labels=selected_labels,
                progress_callback=on_progress,
            )

            progress.progress(1.0, text="Complete!")

            st.success("Study processing complete!")
            cols = st.columns(5)
            with cols[0]:
                st.metric("Windows Processed", summary["n_windows"])
            with cols[1]:
                st.metric("Channels", summary.get("n_channels", 1))
            with cols[2]:
                duration_hrs = summary["duration_sec"] / 3600
                st.metric("Recording Duration", f"{duration_hrs:.1f} hrs")
            with cols[3]:
                st.metric("Feature Records", summary["total_records"])
            with cols[4]:
                st.metric("Alerts", summary["total_alerts"])

            st.info("Go to the **Dashboard** page to view results.")

        except Exception as e:
            st.error(f"Processing failed: {e}")
            # Mark study as failed
            study_service.complete_study(study.id)
