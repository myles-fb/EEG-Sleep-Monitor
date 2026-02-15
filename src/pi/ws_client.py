"""WebSocket client for Pi-to-server communication."""

import asyncio
import json
import logging
import time
from typing import Callable, Optional

import websockets

logger = logging.getLogger(__name__)


class PiWebSocketClient:
    """Async WebSocket client that connects to the FastAPI server."""

    def __init__(
        self,
        server_url: str,
        device_id: str,
        on_config_update: Optional[Callable[[dict], None]] = None,
        on_command: Optional[Callable[[str], None]] = None,
        reconnect_interval: float = 5.0,
        heartbeat_interval: float = 30.0,
    ):
        self.server_url = server_url
        self.device_id = device_id
        self.on_config_update = on_config_update
        self.on_command = on_command
        self.reconnect_interval = reconnect_interval
        self.heartbeat_interval = heartbeat_interval
        self._ws = None
        self._running = False
        self._connected = asyncio.Event()

    async def connect(self):
        """Connect with auto-reconnect loop."""
        self._running = True
        while self._running:
            try:
                async with websockets.connect(self.server_url) as ws:
                    self._ws = ws
                    self._connected.set()
                    logger.info("Connected to %s", self.server_url)

                    # Send registration message
                    await self._send({
                        "type": "register",
                        "device_id": self.device_id,
                        "timestamp": time.time(),
                    })

                    # Run listener and heartbeat concurrently
                    await asyncio.gather(
                        self._listen(),
                        self._heartbeat_loop(),
                    )

            except (websockets.ConnectionClosed, OSError) as e:
                self._ws = None
                self._connected.clear()
                if self._running:
                    logger.warning(
                        "Connection lost (%s), reconnecting in %.0fs",
                        e, self.reconnect_interval,
                    )
                    await asyncio.sleep(self.reconnect_interval)

    async def _listen(self):
        """Listen for server messages."""
        while self._ws and self._running:
            try:
                raw = await self._ws.recv()
                data = json.loads(raw)
                msg_type = data.get("type")

                if msg_type in ("config", "config_response"):
                    config = data.get("config")
                    if config and self.on_config_update:
                        self.on_config_update(config)

                elif msg_type == "command":
                    action = data.get("action")
                    logger.info("Received command: %s", action)
                    if self.on_command:
                        self.on_command(action)

            except websockets.ConnectionClosed:
                break

    async def _heartbeat_loop(self):
        """Send periodic heartbeats."""
        while self._ws and self._running:
            await asyncio.sleep(self.heartbeat_interval)
            try:
                await self._send({
                    "type": "heartbeat",
                    "device_id": self.device_id,
                    "timestamp": time.time(),
                })
            except websockets.ConnectionClosed:
                break

    async def _send(self, data: dict):
        if self._ws:
            await self._ws.send(json.dumps(data))

    async def send_features(
        self,
        timestamp: float,
        window_start: float,
        window_end: float,
        features: dict,
        alerts: list,
    ):
        """Send processed feature data to the server."""
        await self._connected.wait()
        await self._send({
            "type": "feature_update",
            "device_id": self.device_id,
            "timestamp": timestamp,
            "window_start": window_start,
            "window_end": window_end,
            "features": features,
            "alerts": alerts,
        })

    async def request_config(self):
        """Ask server for current patient config."""
        await self._connected.wait()
        await self._send({
            "type": "config_request",
            "device_id": self.device_id,
        })

    def stop(self):
        """Signal the client to disconnect."""
        self._running = False
        self._connected.set()  # Unblock any waiters
