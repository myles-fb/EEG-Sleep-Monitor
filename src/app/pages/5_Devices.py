"""
Device Management — Register Pi devices, assign patients, manage live streaming.
"""

import sys
from pathlib import Path

_src = Path(__file__).resolve().parent.parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import streamlit as st
import requests

from models import init_db
from services import patient_service, device_service, study_service, config_service

init_db()

st.set_page_config(page_title="Devices", page_icon="📡", layout="wide")
st.title("Device Management")

from app.config import FASTAPI_URL

# ---------------------------------------------------------------------------
# Helper: fetch live device status from FastAPI
# ---------------------------------------------------------------------------

def get_live_status():
    """Fetch live connection status from the WebSocket server."""
    try:
        resp = requests.get(f"{FASTAPI_URL}/api/devices", timeout=3)
        if resp.status_code == 200:
            return resp.json()
    except requests.ConnectionError:
        pass
    return None


# ---------------------------------------------------------------------------
# Register New Device
# ---------------------------------------------------------------------------

with st.expander("Register New Pi Device", expanded=False):
    with st.form("register_device", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            dev_name = st.text_input("Device Name *", placeholder="e.g. Pi Lab-01")
            dev_key = st.text_input(
                "Device Key *",
                placeholder="e.g. pi-lab-01",
                help="Unique hardware identifier. Must match --device-id on the Pi.",
            )
        with col2:
            dev_serial = st.text_input(
                "Serial Port",
                placeholder="/dev/ttyUSB0",
                help="Serial port on the Pi (for reference only)",
            )
            dev_ip = st.text_input("IP Address", placeholder="192.168.1.100")

        submitted = st.form_submit_button("Register Device", use_container_width=True)
        if submitted:
            if not dev_name.strip() or not dev_key.strip():
                st.error("Device name and key are required.")
            elif device_service.get_device_by_key(dev_key.strip()):
                st.error(f"Device key `{dev_key}` is already registered.")
            else:
                device_service.register_device(
                    name=dev_name.strip(),
                    device_key=dev_key.strip(),
                    serial_port=dev_serial.strip() or None,
                    ip_address=dev_ip.strip() or None,
                )
                st.success(f"Device **{dev_name}** registered.")
                st.rerun()

st.divider()

# ---------------------------------------------------------------------------
# Device List
# ---------------------------------------------------------------------------

st.subheader("Registered Devices")

devices = device_service.list_devices()
live_status = get_live_status()

# Build a lookup from device_key -> live info
live_map = {}
if live_status:
    for info in live_status:
        live_map[info.get("device_key", "")] = info

if not devices:
    st.info("No devices registered. Register one above.")
else:
    patients = patient_service.list_patients()
    patient_map = {p.id: p.name for p in patients}

    for dev in devices:
        live = live_map.get(dev.device_key, {})
        conn_status = live.get("status", dev.status)

        # Status indicator color
        if conn_status == "streaming":
            indicator = "🟢"
        elif conn_status == "connected":
            indicator = "🟡"
        else:
            indicator = "🔴"

        with st.container(border=True):
            col_name, col_info, col_actions = st.columns([3, 4, 3])

            with col_name:
                st.markdown(f"{indicator} **{dev.name}**")
                st.caption(f"Key: `{dev.device_key}`")
                if dev.ip_address:
                    st.caption(f"IP: {dev.ip_address}")

            with col_info:
                st.text(f"Status: {conn_status}")
                if dev.patient:
                    st.text(f"Patient: {dev.patient.name}")
                else:
                    st.text("Patient: Not assigned")
                if dev.last_heartbeat:
                    st.caption(f"Last heartbeat: {dev.last_heartbeat}")

            with col_actions:
                # Assign patient
                if patients:
                    new_pid = st.selectbox(
                        "Assign Patient",
                        options=[""] + [p.id for p in patients],
                        format_func=lambda pid: patient_map.get(pid, "-- None --"),
                        key=f"assign_{dev.id}",
                        label_visibility="collapsed",
                    )
                    if new_pid and new_pid != (dev.patient_id or ""):
                        if st.button("Assign", key=f"btn_assign_{dev.id}"):
                            device_service.assign_patient(dev.id, new_pid)
                            st.rerun()

                # Start live study
                if dev.patient_id and conn_status in ("connected", "streaming"):
                    if st.button("Start Live Study", key=f"start_{dev.id}"):
                        study = study_service.create_live_study(
                            patient_id=dev.patient_id,
                            device_id=dev.id,
                        )
                        device_service.assign_patient(dev.id, dev.patient_id, study.id)
                        # Push config to Pi
                        try:
                            requests.post(
                                f"{FASTAPI_URL}/api/devices/{dev.id}/config",
                                timeout=3,
                            )
                        except requests.ConnectionError:
                            pass
                        st.success(f"Live study started (ID: {study.id[:8]}...)")
                        st.rerun()

                # Stop command
                if conn_status == "streaming":
                    if st.button("Stop", key=f"stop_{dev.id}", type="secondary"):
                        try:
                            requests.post(
                                f"{FASTAPI_URL}/api/devices/{dev.id}/command",
                                params={"action": "stop"},
                                timeout=3,
                            )
                            st.info("Stop command sent.")
                        except requests.ConnectionError:
                            st.warning("Could not reach WebSocket server.")

                # Delete
                if st.button("Delete", key=f"del_{dev.id}", type="secondary"):
                    device_service.delete_device(dev.id)
                    st.rerun()

# ---------------------------------------------------------------------------
# Server Status
# ---------------------------------------------------------------------------

st.divider()
st.subheader("WebSocket Server")

if live_status is not None:
    n_connected = sum(1 for d in live_status if d.get("status") != "offline")
    st.success(f"Server running at `{FASTAPI_URL}` — {n_connected} device(s) connected")
else:
    st.warning(
        f"Cannot reach WebSocket server at `{FASTAPI_URL}`. "
        "Start it with: `cd src && uvicorn server.ws_server:app --host 0.0.0.0 --port 8765`"
    )
