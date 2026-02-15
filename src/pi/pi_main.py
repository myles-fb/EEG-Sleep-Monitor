"""
Raspberry Pi EEG Streaming Controller.

Entry point for live EEG acquisition, on-device processing, and
WebSocket streaming to the physician dashboard server.

Usage:
    cd src && python -m pi.pi_main \
        --serial-port /dev/ttyUSB0 \
        --server ws://HOST:8765/ws/pi/DEVICE_ID \
        --device-id pi-lab-01

For testing without a physical Cyton board, use BrainFlow's synthetic board:
    python -m pi.pi_main --synthetic --server ws://localhost:8765/ws/pi/pi-lab-01 --device-id pi-lab-01
"""

import argparse
import asyncio
import logging
import signal
import sys
import time
import warnings
from pathlib import Path
from threading import Thread, Event

import numpy as np

from acquisition.brainflow_stream import CytonStream
from processing.ring_buffer import RingBuffer
from processing.filters import preprocess_eeg
from processing.metrics import extract_features, FREQ_BANDS
from processing.mos import compute_mos_for_bucket
from pi.pi_config import PiConfig
from pi.ws_client import PiWebSocketClient

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_RATE = 250
DEFAULT_NUM_CHANNELS = 8
SYNTHETIC_BOARD_ID = -1  # BrainFlow synthetic board


class PiStreamingController:
    """Orchestrates EEG acquisition, processing, and WebSocket streaming."""

    def __init__(
        self,
        serial_port: str,
        server_url: str,
        device_id: str,
        config_path: str = None,
        board_id: int = 0,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        num_channels: int = DEFAULT_NUM_CHANNELS,
    ):
        self.serial_port = serial_port
        self.server_url = server_url
        self.device_id = device_id
        self.board_id = board_id
        self.sample_rate = sample_rate
        self.num_channels = num_channels

        self._stop_event = Event()
        self._config = PiConfig(config_path)

        # Will be initialized on start
        self._stream: CytonStream = None
        self._buffer: RingBuffer = None
        self._ws_client: PiWebSocketClient = None
        self._acq_thread: Thread = None

    def _on_config_update(self, config: dict):
        """Handle config pushed from server."""
        logger.info("Received config update from server")
        self._config.update_from_server(config)

    def _on_command(self, action: str):
        """Handle command from server."""
        logger.info("Received command: %s", action)
        if action == "stop":
            self._stop_event.set()
        elif action == "restart":
            self._stop_event.set()
            # A supervisor (systemd) will restart the process

    def _start_acquisition(self):
        """Connect to Cyton board and start streaming into ring buffer."""
        window_sec = self._config.algorithm_window
        buffer_samples = int(window_sec * self.sample_rate * 2)  # 2x window for safety

        self._buffer = RingBuffer(
            num_channels=self.num_channels,
            buffer_size_samples=buffer_samples,
            sample_rate=self.sample_rate,
        )

        self._stream = CytonStream(
            serial_port=self.serial_port,
            board_id=self.board_id,
            ring_buffer=self._buffer,
        )

        if not self._stream.connect():
            raise RuntimeError("Failed to connect to Cyton board")

        if not self._stream.start_stream():
            raise RuntimeError("Failed to start stream")

        # Run acquisition loop in background thread
        self._acq_thread = Thread(
            target=self._stream.run_stream_loop,
            daemon=True,
        )
        self._acq_thread.start()
        logger.info("Acquisition thread started")

    def _process_window(self, window_data: np.ndarray, timestamp: float) -> dict:
        """Process a single window of EEG data and return features + alerts."""
        features = {}
        alerts = []

        window_sec = self._config.algorithm_window
        ts_end = timestamp + window_sec

        # Bandpower features
        if self._config.is_feature_enabled("bandpower"):
            preprocessed = preprocess_eeg(
                window_data,
                self.sample_rate,
                remove_dc=True,
                apply_notch=True,
                apply_bandpass=True,
            )
            eeg_features = extract_features(
                preprocessed[0],  # first channel
                self.sample_rate,
                channel_index=0,
                timestamp=timestamp,
            )
            features["bandpower"] = {
                band: float(eeg_features.bandpower.get(band, 0))
                for band in FREQ_BANDS
            }

        # MO detection
        if self._config.is_feature_enabled("mo_detection"):
            lasso_win = min(120.0, window_sec)
            lasso_step = lasso_win / 4.0

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = compute_mos_for_bucket(
                    window_data,
                    self.sample_rate,
                    timestamp=timestamp,
                    n_surrogates=self._config.n_surrogates,
                    channel_index=0,
                    wintime_sec=lasso_win,
                    winjump_sec=lasso_step,
                )

            # Q-values per band
            features["mo_q_per_band"] = {
                band: float(q) if np.isfinite(q) else None
                for band, q in result.q_per_band.items()
            }

            # P-values per band
            features["mo_p_per_band"] = {
                band: float(p) if np.isfinite(p) else None
                for band, p in result.p_per_band.items()
            }

            # MO count (significant bands)
            sig_bands = sum(1 for p in result.p_per_band.values() if p < 0.05)
            features["mo_count"] = sig_bands

            # Dominant frequency per band
            features["mo_dom_freq_per_band"] = {
                band: float(f) if np.isfinite(f) else None
                for band, f in result.dominant_freq_hz_per_band.items()
            }

            # Per-window detail
            features["mo_window_detail"] = {
                "q_per_window": {
                    b: arr.tolist() for b, arr in result.q_per_window_per_band.items()
                },
                "p_per_window": {
                    b: arr.tolist() for b, arr in result.p_per_window_per_band.items()
                },
            }

            # Check MO count threshold
            mo_threshold = self._config.notification_thresholds.get("mo_count")
            if mo_threshold and sig_bands >= mo_threshold:
                hours = int(timestamp // 3600)
                alerts.append({
                    "alert_type": "mo_count",
                    "threshold": float(mo_threshold),
                    "actual_value": float(sig_bands),
                    "bucket_time": f"{hours:02d}:00-{hours + 1:02d}:00",
                })

        return {
            "features": features,
            "alerts": alerts,
            "timestamp": timestamp,
            "window_start": timestamp,
            "window_end": ts_end,
        }

    async def _processing_loop(self):
        """Main processing loop: extract windows, process, send features."""
        window_sec = self._config.algorithm_window
        window_samples = int(window_sec * self.sample_rate)

        logger.info(
            "Processing loop started (window=%ds, %d samples)",
            window_sec, window_samples,
        )

        # Request config from server on startup
        await self._ws_client.request_config()

        # Wait for enough data to accumulate
        recording_start = time.time()

        while not self._stop_event.is_set():
            # Wait for a full window of data
            await asyncio.sleep(window_sec)

            if self._stop_event.is_set():
                break

            # Extract window from ring buffer
            window_data, complete = self._buffer.get_latest(window_samples)
            if not complete:
                logger.warning("Incomplete window, skipping")
                continue

            timestamp = time.time() - recording_start
            logger.info("Processing window at t=%.1fs", timestamp)

            try:
                result = self._process_window(window_data, timestamp)

                await self._ws_client.send_features(
                    timestamp=result["timestamp"],
                    window_start=result["window_start"],
                    window_end=result["window_end"],
                    features=result["features"],
                    alerts=result["alerts"],
                )
                logger.info(
                    "Sent features for t=%.1fs (mo_count=%s)",
                    timestamp,
                    result["features"].get("mo_count", "N/A"),
                )
            except Exception as e:
                logger.error("Error processing window: %s", e)

    async def run(self):
        """Main entry point: connect, acquire, process, stream."""
        # Setup WebSocket client
        self._ws_client = PiWebSocketClient(
            server_url=self.server_url,
            device_id=self.device_id,
            on_config_update=self._on_config_update,
            on_command=self._on_command,
        )

        # Start acquisition
        self._start_acquisition()

        # Run WebSocket connection and processing loop concurrently
        try:
            await asyncio.gather(
                self._ws_client.connect(),
                self._processing_loop(),
            )
        except asyncio.CancelledError:
            pass
        finally:
            self._stop()

    def _stop(self):
        """Clean shutdown."""
        self._stop_event.set()
        if self._ws_client:
            self._ws_client.stop()
        if self._stream:
            self._stream.stop_stream()
            self._stream.disconnect()
        logger.info("Controller stopped")


def main():
    parser = argparse.ArgumentParser(
        description="Pi EEG Streaming Controller",
    )
    parser.add_argument(
        "--serial-port", type=str, default="/dev/ttyUSB0",
        help="Serial port for Cyton board",
    )
    parser.add_argument(
        "--server", type=str, required=True,
        help="WebSocket server URL (e.g. ws://host:8765/ws/pi/device-id)",
    )
    parser.add_argument(
        "--device-id", type=str, required=True,
        help="Unique device identifier (must match registered device_key)",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to local config cache file (JSON)",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use BrainFlow synthetic board for testing (no hardware needed)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    board_id = SYNTHETIC_BOARD_ID if args.synthetic else 0

    controller = PiStreamingController(
        serial_port=args.serial_port,
        server_url=args.server,
        device_id=args.device_id,
        config_path=args.config,
        board_id=board_id,
    )

    # Handle SIGINT/SIGTERM gracefully
    loop = asyncio.new_event_loop()

    def shutdown(sig):
        logger.info("Received signal %s, shutting down", sig)
        controller._stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown, sig)

    try:
        loop.run_until_complete(controller.run())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
