"""Ingest feature data from Pi devices and store in the database."""

import json
import logging

import numpy as np

from models import FeatureRecord, Alert, get_db

logger = logging.getLogger(__name__)


def ingest_features(
    study_id: str,
    timestamp: float,
    window_start: float,
    window_end: float,
    features: dict,
    alerts: list,
    channel_index: int = None,
    channel_label: str = None,
):
    """
    Store feature data sent by a Pi device.

    Mirrors the storage pattern from study_service.process_edf() so that
    Dashboard queries work identically for EDF and Pi data sources.

    Parameters
    ----------
    features : dict
        Expected keys: mo_q_per_band, mo_p_per_band, mo_count,
        mo_dom_freq_per_band, mo_window_detail, bandpower (optional).
    alerts : list
        List of dicts with keys: alert_type, threshold, actual_value, bucket_time.
    """
    total_records = 0

    with get_db() as db:
        # Q-values per band
        for band, q_val in features.get("mo_q_per_band", {}).items():
            val = float(q_val) if q_val is not None and np.isfinite(q_val) else None
            db.add(FeatureRecord(
                study_id=study_id,
                timestamp=timestamp,
                bucket_start=window_start,
                bucket_end=window_end,
                feature_key=f"mo_q_{band}",
                feature_value=val,
                channel_index=channel_index,
                channel_label=channel_label,
            ))
            total_records += 1

        # P-values per band
        for band, p_val in features.get("mo_p_per_band", {}).items():
            val = float(p_val) if p_val is not None and np.isfinite(p_val) else None
            db.add(FeatureRecord(
                study_id=study_id,
                timestamp=timestamp,
                bucket_start=window_start,
                bucket_end=window_end,
                feature_key=f"mo_p_{band}",
                feature_value=val,
                channel_index=channel_index,
                channel_label=channel_label,
            ))
            total_records += 1

        # MO count
        mo_count = features.get("mo_count", 0)
        db.add(FeatureRecord(
            study_id=study_id,
            timestamp=timestamp,
            bucket_start=window_start,
            bucket_end=window_end,
            feature_key="mo_count",
            feature_value=float(mo_count),
            channel_index=channel_index,
            channel_label=channel_label,
        ))
        total_records += 1

        # Dominant modulation frequency per band
        for band, freq in features.get("mo_dom_freq_per_band", {}).items():
            val = float(freq) if freq is not None and np.isfinite(freq) else None
            db.add(FeatureRecord(
                study_id=study_id,
                timestamp=timestamp,
                bucket_start=window_start,
                bucket_end=window_end,
                feature_key=f"mo_dom_freq_{band}",
                feature_value=val,
                channel_index=channel_index,
                channel_label=channel_label,
            ))
            total_records += 1

        # Per-window detail JSON
        window_detail = features.get("mo_window_detail")
        if window_detail:
            db.add(FeatureRecord(
                study_id=study_id,
                timestamp=timestamp,
                bucket_start=window_start,
                bucket_end=window_end,
                feature_key="mo_window_detail",
                feature_value=None,
                metadata_json=json.dumps(window_detail, default=str),
                channel_index=channel_index,
                channel_label=channel_label,
            ))
            total_records += 1

        # Bandpower features (not stored by EDF pipeline)
        for band, bp_val in features.get("bandpower", {}).items():
            val = float(bp_val) if bp_val is not None and np.isfinite(bp_val) else None
            db.add(FeatureRecord(
                study_id=study_id,
                timestamp=timestamp,
                bucket_start=window_start,
                bucket_end=window_end,
                feature_key=f"bandpower_{band}",
                feature_value=val,
                channel_index=channel_index,
                channel_label=channel_label,
            ))
            total_records += 1

        # Alerts
        for alert_data in alerts:
            db.add(Alert(
                study_id=study_id,
                timestamp=timestamp,
                alert_type=alert_data["alert_type"],
                threshold=float(alert_data["threshold"]),
                actual_value=float(alert_data["actual_value"]),
                bucket_time=alert_data.get("bucket_time"),
            ))

    logger.info(
        "Ingested %d records for study %s at t=%.1f",
        total_records, study_id, timestamp,
    )
    return total_records
