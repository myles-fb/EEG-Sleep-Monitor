"""Configuration management for Pi streaming client."""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Map feature names to config keys
_FEATURE_MAP = {
    "mo_detection": ["mo_q_score", "mo_count"],
    "bandpower": [
        "bandpower_delta",
        "bandpower_theta",
        "bandpower_alpha",
        "bandpower_beta",
    ],
    "envelope": ["envelope_summary"],
}


class PiConfig:
    """Loads, caches, and manages Pi configuration from server or local file."""

    def __init__(self, config_path: Optional[str] = None):
        self._config_path = Path(config_path) if config_path else None
        self._config: dict = {}
        if self._config_path and self._config_path.exists():
            self._load_from_file()

    def _load_from_file(self):
        try:
            self._config = json.loads(self._config_path.read_text())
            logger.info("Loaded config from %s", self._config_path)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load config from %s: %s", self._config_path, e)

    def _save_to_file(self):
        if self._config_path:
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            self._config_path.write_text(json.dumps(self._config, indent=2))
            logger.info("Saved config to %s", self._config_path)

    def update_from_server(self, data: dict):
        """Apply server-pushed config and cache locally."""
        self._config = data
        self._save_to_file()
        logger.info("Config updated from server (patient=%s)", data.get("patient_id"))

    @property
    def patient_id(self) -> Optional[str]:
        return self._config.get("patient_id")

    @property
    def study_id(self) -> Optional[str]:
        return self._config.get("study_id")

    @property
    def algorithm_window(self) -> int:
        return self._config.get("algorithm_window", 300)

    @property
    def n_surrogates(self) -> int:
        return self._config.get("n_surrogates", 5)

    @property
    def notification_thresholds(self) -> dict:
        return self._config.get("notification_thresholds", {})

    def is_feature_enabled(self, feature_name: str) -> bool:
        """Check if a high-level feature is enabled.

        feature_name can be: "mo_detection", "bandpower", "envelope"
        """
        features = self._config.get("features", {})
        config_keys = _FEATURE_MAP.get(feature_name, [feature_name])
        return any(features.get(k) for k in config_keys)

    @property
    def raw(self) -> dict:
        return dict(self._config)
