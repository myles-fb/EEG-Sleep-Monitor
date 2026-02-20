"""
FastAPI WebSocket server for Pi-to-server communication.

Run with:
    cd src && uvicorn server.ws_server:app --host 0.0.0.0 --port 8765
"""

import sys
import logging
from pathlib import Path

# Ensure src/ is importable
_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from models import init_db
from services import device_service, config_service, patient_service, study_service
from server.device_manager import DeviceManager
from server.ingestion_service import ingest_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="EEG Pi Gateway")
manager = DeviceManager()


@app.on_event("startup")
async def startup():
    init_db()
    logger.info("WebSocket server started")


# ---------------------------------------------------------------------------
# WebSocket endpoint for Pi devices
# ---------------------------------------------------------------------------

@app.websocket("/ws/pi/{device_id}")
async def pi_websocket(websocket: WebSocket, device_id: str):
    await websocket.accept()
    logger.info("Pi device %s connecting", device_id)

    # Look up device in DB
    device = device_service.get_device_by_key(device_id)
    if not device:
        await websocket.send_json({"type": "error", "message": "Unknown device"})
        await websocket.close(code=4001)
        return

    conn = await manager.register(device.id, device_id, websocket)
    device_service.update_status(device.id, "connected")

    # Restore patient/study assignment from DB
    if device.patient_id and device.current_study_id:
        await manager.assign_patient(device.id, device.patient_id, device.current_study_id)

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "heartbeat":
                await manager.update_heartbeat(device.id)
                device_service.update_heartbeat(device_id)

            elif msg_type == "config_request":
                await _handle_config_request(device, websocket)

            elif msg_type == "feature_update":
                await _handle_feature_update(device, data)

            elif msg_type == "register":
                logger.info("Device %s registered at %s", device_id, data.get("timestamp"))

    except WebSocketDisconnect:
        logger.info("Pi device %s disconnected", device_id)
    except Exception as e:
        logger.error("Error with device %s: %s", device_id, e)
    finally:
        await manager.unregister(device.id)
        device_service.update_status(device.id, "offline")


async def _handle_config_request(device, websocket):
    """Send patient config to the Pi."""
    if not device.patient_id:
        await websocket.send_json({
            "type": "config_response",
            "config": None,
            "message": "No patient assigned",
        })
        return

    patient = patient_service.get_patient(device.patient_id)
    if not patient:
        await websocket.send_json({
            "type": "config_response",
            "config": None,
            "message": "Patient not found",
        })
        return

    config = config_service.generate_pi_config(
        patient,
        study_id=device.current_study_id,
    )
    await websocket.send_json({"type": "config_response", "config": config})


async def _handle_feature_update(device, data):
    """Store feature data from Pi in the database."""
    conn = manager.get_connection(device.id)
    if not conn or not conn.study_id:
        logger.warning("Feature update from device %s with no active study", device.id)
        return

    ingest_features(
        study_id=conn.study_id,
        timestamp=data.get("timestamp", 0),
        window_start=data.get("window_start", 0),
        window_end=data.get("window_end", 0),
        features=data.get("features", {}),
        alerts=data.get("alerts", []),
        channel_index=data.get("channel_index"),
        channel_label=data.get("channel_label"),
    )


# ---------------------------------------------------------------------------
# REST endpoints for Streamlit dashboard
# ---------------------------------------------------------------------------

@app.get("/api/devices")
async def list_devices():
    """List all connected Pi devices with live status."""
    live = manager.get_all_status()
    # Merge with DB records for offline devices
    all_devices = device_service.list_devices()
    device_map = {d["device_key"]: d for d in live}

    result = []
    for dev in all_devices:
        if dev.device_key in device_map:
            info = device_map[dev.device_key]
            info["db_id"] = dev.id
            info["name"] = dev.name
            info["patient_name"] = dev.patient.name if dev.patient else None
            result.append(info)
        else:
            result.append({
                "device_id": dev.id,
                "device_key": dev.device_key,
                "db_id": dev.id,
                "name": dev.name,
                "patient_id": dev.patient_id,
                "patient_name": dev.patient.name if dev.patient else None,
                "study_id": dev.current_study_id,
                "status": dev.status,
                "last_heartbeat": dev.last_heartbeat.isoformat() if dev.last_heartbeat else None,
            })

    return JSONResponse(content=result)


@app.post("/api/devices/{device_id}/config")
async def push_config(device_id: str):
    """Push updated config to a connected Pi."""
    device = device_service.get_device(device_id)
    if not device:
        return JSONResponse(status_code=404, content={"error": "Device not found"})

    if not manager.is_connected(device.id):
        return JSONResponse(status_code=400, content={"error": "Device not connected"})

    if not device.patient_id:
        return JSONResponse(status_code=400, content={"error": "No patient assigned"})

    patient = patient_service.get_patient(device.patient_id)

    # Extract active channels from study channels_json if available
    active_channels = None
    if device.current_study_id:
        channels = study_service.get_study_channels(device.current_study_id)
        if channels:
            active_channels = [c["index"] for c in channels]

    config = config_service.generate_pi_config(
        patient,
        study_id=device.current_study_id,
        active_channels=active_channels,
    )
    await manager.push_config(device.id, config)
    return JSONResponse(content={"status": "config_pushed"})


@app.post("/api/devices/{device_id}/command")
async def send_command(device_id: str, action: str = "stop"):
    """Send a command to a connected Pi (stop, restart)."""
    device = device_service.get_device(device_id)
    if not device:
        return JSONResponse(status_code=404, content={"error": "Device not found"})

    if not manager.is_connected(device.id):
        return JSONResponse(status_code=400, content={"error": "Device not connected"})

    await manager.send_command(device.id, action)
    return JSONResponse(content={"status": f"command_{action}_sent"})
