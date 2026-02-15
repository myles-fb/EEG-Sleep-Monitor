"""Export study data as CSV or JSON."""

import csv
import io
import json
from typing import Optional

from models import FeatureRecord, Alert, Study, Patient, get_db


def export_csv(study_id: str, feature_key: Optional[str] = None) -> str:
    """
    Export feature records for a study as CSV text.

    Columns: timestamp, bucket_start, bucket_end, feature_key, feature_value
    """
    with get_db() as db:
        q = db.query(FeatureRecord).filter_by(study_id=study_id)
        if feature_key:
            q = q.filter_by(feature_key=feature_key)
        records = q.order_by(FeatureRecord.timestamp).all()

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["timestamp", "bucket_start", "bucket_end", "feature_key", "feature_value"])
    for r in records:
        if r.feature_key == "mo_window_detail":
            continue  # skip raw JSON detail rows in CSV
        writer.writerow([r.timestamp, r.bucket_start, r.bucket_end, r.feature_key, r.feature_value])
    return buf.getvalue()


def export_json(study_id: str) -> str:
    """
    Export full study data as JSON, matching the feature template schema
    from the architecture spec.
    """
    with get_db() as db:
        study = db.query(Study).filter_by(id=study_id).first()
        patient = db.query(Patient).filter_by(id=study.patient_id).first() if study else None
        records = (
            db.query(FeatureRecord)
            .filter_by(study_id=study_id)
            .order_by(FeatureRecord.timestamp)
            .all()
        )
        alerts = (
            db.query(Alert)
            .filter_by(study_id=study_id)
            .order_by(Alert.timestamp)
            .all()
        )

    if not study:
        return json.dumps({"error": "Study not found"})

    # Group records by timestamp into feature-template buckets
    buckets = {}
    for r in records:
        ts = r.timestamp
        if ts not in buckets:
            buckets[ts] = {
                "timestamp": ts,
                "patient_id": study.patient_id,
                "bucket_start": r.bucket_start,
                "bucket_end": r.bucket_end,
                "features": {},
            }
        if r.feature_key == "mo_window_detail":
            buckets[ts]["features"]["mo_window_detail"] = (
                json.loads(r.metadata_json) if r.metadata_json else None
            )
        else:
            buckets[ts]["features"][r.feature_key] = r.feature_value

    payload = {
        "study_id": study.id,
        "patient_id": study.patient_id,
        "patient_name": patient.name if patient else None,
        "source": study.source,
        "source_file": study.source_file,
        "started_at": study.started_at.isoformat() if study.started_at else None,
        "ended_at": study.ended_at.isoformat() if study.ended_at else None,
        "duration_sec": study.duration_sec,
        "sample_rate": study.sample_rate,
        "status": study.status,
        "buckets": list(buckets.values()),
        "alerts": [
            {
                "timestamp": a.timestamp,
                "alert_type": a.alert_type,
                "threshold": a.threshold,
                "actual_value": a.actual_value,
                "bucket_time": a.bucket_time,
            }
            for a in alerts
        ],
    }
    return json.dumps(payload, indent=2, default=str)
