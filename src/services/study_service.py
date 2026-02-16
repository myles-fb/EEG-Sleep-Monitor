"""Study management and EEG processing service."""

import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Dict, List, Tuple

import numpy as np

from models import Study, FeatureRecord, Alert, Patient, get_db

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LB-18 bipolar montage (copied from scripts/run_mos_edf_pipeline.py to avoid
# cross-package import issues).
# ---------------------------------------------------------------------------

LB18_PAIRS = [
    ("EEGFp1Ref", "EEGF7Ref"),
    ("EEGF7Ref", "EEGT7Ref"),
    ("EEGT7Ref", "EEGP7Ref"),
    ("EEGP7Ref", "EEGO1Ref"),
    ("EEGFp1Ref", "EEGF3Ref"),
    ("EEGF3Ref", "EEGC3Ref"),
    ("EEGC3Ref", "EEGP3Ref"),
    ("EEGP3Ref", "EEGO1Ref"),
    ("EEGFp2Ref", "EEGF8Ref"),
    ("EEGF8Ref", "EEGT8Ref"),
    ("EEGT8Ref", "EEGP8Ref"),
    ("EEGP8Ref", "EEGO2Ref"),
    ("EEGFp2Ref", "EEGF4Ref"),
    ("EEGF4Ref", "EEGC4Ref"),
    ("EEGC4Ref", "EEGP4Ref"),
    ("EEGP4Ref", "EEGO2Ref"),
    ("EEGFzRef", "EEGCzRef"),
    ("EEGCzRef", "EEGPzRef"),
]


def _lb18_name_to_edf_style(name: str) -> str:
    if name.startswith("EEG") and name.endswith("Ref") and len(name) > 6:
        return "EEG " + name[3:-3] + "-Ref"
    return name


def _build_lb18_bpm_mask(channel_names: list) -> np.ndarray:
    """Build (18, 2) bipolar montage index array from EDF channel labels."""
    name_to_idx = {name.strip(): i for i, name in enumerate(channel_names)}
    bpm = np.zeros((18, 2), dtype=np.intp)
    for i, (left, right) in enumerate(LB18_PAIRS):
        left_idx = name_to_idx.get(left) or name_to_idx.get(
            _lb18_name_to_edf_style(left)
        )
        right_idx = name_to_idx.get(right) or name_to_idx.get(
            _lb18_name_to_edf_style(right)
        )
        if left_idx is None or right_idx is None:
            raise ValueError(f"Channel not found for pair ({left}, {right})")
        bpm[i, 0] = left_idx
        bpm[i, 1] = right_idx
    return bpm


def _load_edf(edf_path: Path) -> Tuple[np.ndarray, float, list]:
    """Load EDF file via MNE.  Returns (data, sample_rate, channel_names)."""
    import mne

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    data = raw.get_data()  # (n_channels, n_samples)
    return np.asarray(data, dtype=np.float64), float(raw.info["sfreq"]), list(raw.ch_names)


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

def create_study(patient_id, source="edf", source_file=None, notes=None):
    with get_db() as db:
        study = Study(
            patient_id=patient_id,
            source=source,
            source_file=source_file,
            notes=notes,
        )
        db.add(study)
        db.flush()
        return study


def create_live_study(patient_id, device_id, notes=None):
    """Create a study for live Pi streaming."""
    with get_db() as db:
        study = Study(
            patient_id=patient_id,
            device_id=device_id,
            source="pi",
            status="active",
            notes=notes,
        )
        db.add(study)
        db.flush()
        return study


def get_study(study_id):
    with get_db() as db:
        return db.query(Study).filter_by(id=study_id).first()


def list_studies(patient_id):
    with get_db() as db:
        return (
            db.query(Study)
            .filter_by(patient_id=patient_id)
            .order_by(Study.started_at.desc())
            .all()
        )


def complete_study(study_id):
    with get_db() as db:
        study = db.query(Study).filter_by(id=study_id).first()
        if study:
            study.status = "completed"
            study.ended_at = datetime.utcnow()
        return study


# ---------------------------------------------------------------------------
# EDF Processing
# ---------------------------------------------------------------------------

def process_edf(
    study_id: str,
    edf_path: str,
    patient: Patient,
    n_surrogates: int = 1,
    channel_index: int = 0,
    channel_indices: Optional[List[int]] = None,
    channel_labels: Optional[List[str]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Dict:
    """
    Process an EDF file for a study:
      1. Split into algorithm windows (patient.window_size_seconds)
      2. Run MOs pipeline on each window (single or multi-channel)
      3. Store FeatureRecords + Alerts in the database
      4. Save spectrogram NPZ files when enabled

    Returns a summary dict.
    """
    from processing.mos import compute_mos_for_bucket
    from services.spectrogram_service import save_window_spectrogram

    data, Fs, ch_names = _load_edf(Path(edf_path))

    # Try bipolar montage; fall back to raw channels
    bpm_mask = None
    try:
        bpm_mask = _build_lb18_bpm_mask(ch_names)
    except ValueError:
        logger.info("LB-18 montage not available; using raw channels")

    # Resolve channel list
    multi_channel = channel_indices is not None and len(channel_indices) > 1
    if channel_indices is None:
        channel_indices = [channel_index]
    if channel_labels is None:
        channel_labels = [None] * len(channel_indices)

    # Whether to capture spectrogram data
    want_spectrogram = bool(
        patient.enable_full_spectrogram or patient.enable_envelope_spectrogram
    )

    window_sec = patient.window_size_seconds or 300
    window_samples = int(window_sec * Fs)
    n_total = data.shape[1]
    n_windows = max(1, n_total // window_samples)

    lasso_win = min(120.0, window_sec)
    lasso_step = lasso_win / 4.0

    total_records = 0
    total_alerts = 0

    # Store channels_json on study
    channels_info = [
        {"index": ci, "label": cl}
        for ci, cl in zip(channel_indices, channel_labels)
    ]
    update_study_channels(study_id, channels_info)

    with get_db() as db:
        for i in range(n_windows):
            start = i * window_samples
            end = min(start + window_samples, n_total)
            bucket = data[:, start:end]
            ts = start / Fs
            ts_end = end / Fs

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw_result = compute_mos_for_bucket(
                    bucket,
                    Fs,
                    timestamp=ts,
                    bpm_mask=bpm_mask,
                    n_surrogates=n_surrogates,
                    channel_indices=channel_indices,
                    wintime_sec=lasso_win,
                    winjump_sec=lasso_step,
                    store_spectrogram=want_spectrogram,
                )

            # Normalize to list for uniform handling
            results_list = raw_result if isinstance(raw_result, list) else [raw_result]

            for res_idx, result in enumerate(results_list):
                ch_idx = channel_indices[res_idx] if res_idx < len(channel_indices) else result.channel_index
                ch_label = channel_labels[res_idx] if res_idx < len(channel_labels) else None

                # Save spectrogram NPZ
                if want_spectrogram and result.spectrogram_S is not None:
                    spec_T = result.spectrogram_T + ts  # offset to global time
                    save_window_spectrogram(
                        study_id=study_id,
                        window_index=i,
                        channel_index=ch_idx,
                        S=result.spectrogram_S,
                        T=spec_T,
                        F=result.spectrogram_F,
                        band_envelopes=result.band_envelopes,
                    )

                # Store q-values per band
                for band, q_val in result.q_per_band.items():
                    db.add(FeatureRecord(
                        study_id=study_id,
                        timestamp=ts,
                        bucket_start=ts,
                        bucket_end=ts_end,
                        feature_key=f"mo_q_{band}",
                        feature_value=float(q_val) if np.isfinite(q_val) else None,
                        channel_index=ch_idx,
                        channel_label=ch_label,
                    ))
                    total_records += 1

                # Store p-values per band
                for band, p_val in result.p_per_band.items():
                    db.add(FeatureRecord(
                        study_id=study_id,
                        timestamp=ts,
                        bucket_start=ts,
                        bucket_end=ts_end,
                        feature_key=f"mo_p_{band}",
                        feature_value=float(p_val) if np.isfinite(p_val) else None,
                        channel_index=ch_idx,
                        channel_label=ch_label,
                    ))
                    total_records += 1

                # MO count: number of bands with p < 0.05
                sig_bands = sum(
                    1 for p in result.p_per_band.values() if p < 0.05
                )
                db.add(FeatureRecord(
                    study_id=study_id,
                    timestamp=ts,
                    bucket_start=ts,
                    bucket_end=ts_end,
                    feature_key="mo_count",
                    feature_value=float(sig_bands),
                    channel_index=ch_idx,
                    channel_label=ch_label,
                ))
                total_records += 1

                # Dominant modulation frequency per band
                for band, freq in result.dominant_freq_hz_per_band.items():
                    db.add(FeatureRecord(
                        study_id=study_id,
                        timestamp=ts,
                        bucket_start=ts,
                        bucket_end=ts_end,
                        feature_key=f"mo_dom_freq_{band}",
                        feature_value=float(freq) if np.isfinite(freq) else None,
                        channel_index=ch_idx,
                        channel_label=ch_label,
                    ))
                    total_records += 1

                # Per-LASSO-window detail as JSON
                q_window_data = {
                    b: arr.tolist() for b, arr in result.q_per_window_per_band.items()
                }
                p_window_data = {
                    b: arr.tolist() for b, arr in result.p_per_window_per_band.items()
                }
                db.add(FeatureRecord(
                    study_id=study_id,
                    timestamp=ts,
                    bucket_start=ts,
                    bucket_end=ts_end,
                    feature_key="mo_window_detail",
                    feature_value=None,
                    metadata_json=json.dumps(
                        {"q_per_window": q_window_data, "p_per_window": p_window_data},
                        default=str,
                    ),
                    channel_index=ch_idx,
                    channel_label=ch_label,
                ))
                total_records += 1

                # Threshold-based alerts
                if patient.mo_count_threshold and sig_bands >= patient.mo_count_threshold:
                    hours = int(ts // 3600)
                    db.add(Alert(
                        study_id=study_id,
                        timestamp=ts,
                        alert_type="mo_count",
                        threshold=float(patient.mo_count_threshold),
                        actual_value=float(sig_bands),
                        bucket_time=f"{hours:02d}:00-{hours+1:02d}:00",
                    ))
                    total_alerts += 1

            if progress_callback:
                progress_callback(i + 1, n_windows)

        # Update study metadata
        study = db.query(Study).filter_by(id=study_id).first()
        if study:
            study.status = "completed"
            study.ended_at = datetime.utcnow()
            study.duration_sec = n_total / Fs
            study.sample_rate = Fs

    return {
        "n_windows": n_windows,
        "n_channels": len(channel_indices),
        "window_sec": window_sec,
        "total_records": total_records,
        "total_alerts": total_alerts,
        "duration_sec": n_total / Fs,
        "sample_rate": Fs,
    }


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def get_feature_records(study_id, feature_key=None) -> List[FeatureRecord]:
    """Get feature records, optionally filtered by key."""
    with get_db() as db:
        q = db.query(FeatureRecord).filter_by(study_id=study_id)
        if feature_key:
            q = q.filter_by(feature_key=feature_key)
        return q.order_by(FeatureRecord.timestamp).all()


def get_feature_timeseries(study_id, feature_key) -> Tuple[List[float], List[float]]:
    """Return (timestamps, values) for a single feature key."""
    records = get_feature_records(study_id, feature_key)
    timestamps = [r.timestamp for r in records]
    values = [r.feature_value for r in records]
    return timestamps, values


def get_alerts(study_id) -> List[Alert]:
    with get_db() as db:
        return (
            db.query(Alert)
            .filter_by(study_id=study_id)
            .order_by(Alert.timestamp)
            .all()
        )


def get_hourly_summary(study_id, bucket_size_seconds=3600) -> List[Dict]:
    """
    Aggregate feature records into dashboard buckets (default 1 hour).

    Returns list of dicts, one per bucket, with mean/max q-scores and total MO count.
    """
    with get_db() as db:
        records = (
            db.query(FeatureRecord)
            .filter_by(study_id=study_id)
            .filter(FeatureRecord.feature_key != "mo_window_detail")
            .order_by(FeatureRecord.timestamp)
            .all()
        )

    if not records:
        return []

    # Group by hourly bucket
    buckets: Dict[int, Dict[str, list]] = {}
    for r in records:
        idx = int(r.timestamp // bucket_size_seconds)
        if idx not in buckets:
            buckets[idx] = {}
        key = r.feature_key
        if key not in buckets[idx]:
            buckets[idx][key] = []
        if r.feature_value is not None:
            buckets[idx][key].append(r.feature_value)

    result = []
    for idx in sorted(buckets.keys()):
        entry = {
            "bucket_idx": idx,
            "hour_start": idx * bucket_size_seconds,
            "hour_end": (idx + 1) * bucket_size_seconds,
            "label": f"Hour {idx}",
        }
        for key, vals in buckets[idx].items():
            if not vals:
                continue
            if key == "mo_count":
                entry["mo_count_total"] = sum(vals)
                entry["mo_count_mean"] = float(np.mean(vals))
            else:
                entry[f"{key}_mean"] = float(np.mean(vals))
                entry[f"{key}_max"] = float(np.max(vals))
        result.append(entry)

    return result


# ---------------------------------------------------------------------------
# Channel-aware query helpers
# ---------------------------------------------------------------------------

def get_feature_timeseries_by_channel(
    study_id: str,
    feature_key: str,
    channel_index: Optional[int] = None,
) -> Tuple[List[float], List[float]]:
    """Return (timestamps, values) filtered by channel_index.

    If channel_index is None, returns all records (backward compat for
    single-channel studies where channel_index column is NULL).
    """
    with get_db() as db:
        q = (
            db.query(FeatureRecord)
            .filter_by(study_id=study_id, feature_key=feature_key)
        )
        if channel_index is not None:
            q = q.filter(FeatureRecord.channel_index == channel_index)
        records = q.order_by(FeatureRecord.timestamp).all()
    return [r.timestamp for r in records], [r.feature_value for r in records]


def get_study_channels(study_id: str) -> List[Dict]:
    """Parse Study.channels_json into a list of channel dicts.

    Returns [] for legacy single-channel studies.
    """
    with get_db() as db:
        study = db.query(Study).filter_by(id=study_id).first()
    if not study or not study.channels_json:
        return []
    try:
        return json.loads(study.channels_json)
    except (json.JSONDecodeError, TypeError):
        return []


def update_study_channels(study_id: str, channels: List[Dict]) -> None:
    """Write channels_json on a study."""
    with get_db() as db:
        study = db.query(Study).filter_by(id=study_id).first()
        if study:
            study.channels_json = json.dumps(channels)
