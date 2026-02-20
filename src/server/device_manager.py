"""In-memory registry of active WebSocket connections from Pi devices."""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional

from fastapi import WebSocket

logger = logging.getLogger(__name__)


@dataclass
class DeviceConnection:
    device_id: str
    device_key: str
    websocket: WebSocket
    patient_id: Optional[str] = None
    study_id: Optional[str] = None
    status: str = "connected"
    connected_at: datetime = field(default_factory=datetime.utcnow)
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)


class DeviceManager:
    """Manages active Pi WebSocket connections."""

    def __init__(self):
        self._connections: Dict[str, DeviceConnection] = {}
        self._lock = asyncio.Lock()

    async def register(
        self, device_id: str, device_key: str, websocket: WebSocket
    ) -> DeviceConnection:
        async with self._lock:
            conn = DeviceConnection(
                device_id=device_id,
                device_key=device_key,
                websocket=websocket,
            )
            self._connections[device_id] = conn
            logger.info("Device %s registered", device_id)
            return conn

    async def unregister(self, device_id: str):
        async with self._lock:
            self._connections.pop(device_id, None)
            logger.info("Device %s unregistered", device_id)

    async def update_heartbeat(self, device_id: str):
        async with self._lock:
            conn = self._connections.get(device_id)
            if conn:
                conn.last_heartbeat = datetime.utcnow()

    async def assign_patient(
        self, device_id: str, patient_id: str, study_id: str
    ):
        async with self._lock:
            conn = self._connections.get(device_id)
            if conn:
                conn.patient_id = patient_id
                conn.study_id = study_id
                conn.status = "streaming"

    def get_connection(self, device_id: str) -> Optional[DeviceConnection]:
        return self._connections.get(device_id)

    async def push_config(self, device_id: str, config: dict):
        conn = self._connections.get(device_id)
        if conn and conn.websocket:
            await conn.websocket.send_json({
                "type": "config",
                "config": config,
            })

    async def send_command(self, device_id: str, action: str):
        conn = self._connections.get(device_id)
        if conn and conn.websocket:
            await conn.websocket.send_json({
                "type": "command",
                "action": action,
            })

    def get_all_status(self) -> list:
        result = []
        for device_id, conn in self._connections.items():
            result.append({
                "device_id": conn.device_id,
                "device_key": conn.device_key,
                "patient_id": conn.patient_id,
                "study_id": conn.study_id,
                "status": conn.status,
                "connected_at": conn.connected_at.isoformat(),
                "last_heartbeat": conn.last_heartbeat.isoformat(),
            })
        return result

    def is_connected(self, device_id: str) -> bool:
        return device_id in self._connections
