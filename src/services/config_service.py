"""Generate Pi configuration JSON from a patient profile."""

from models import Patient


def generate_pi_config(
    patient: Patient,
    study_id: str = None,
) -> dict:
    """
    Build the JSON config that would be sent to the Raspberry Pi
    to control which features are computed and at what cadence.

    The Pi reads this config and adjusts:
      - Active feature computation
      - Bucket / window timing
      - Notification thresholds
      - Raw EEG storage toggle
    """
    features = {
        "mo_q_score": None,
        "mo_count": None,
        "bandpower_delta": None,
        "bandpower_theta": None,
        "bandpower_alpha": None,
        "bandpower_beta": None,
        "envelope_summary": None,
    }

    # Enable features based on patient profile
    if patient.enable_mo_detection:
        features["mo_q_score"] = True
        features["mo_count"] = True
    if patient.enable_envelope_spectrogram:
        features["envelope_summary"] = True
    if patient.enable_full_spectrogram:
        for band in ("delta", "theta", "alpha", "beta"):
            features[f"bandpower_{band}"] = True

    config = {
        "patient_id": patient.id,
        "study_id": study_id,
        "study_type": patient.study_type,
        "bucket_duration": patient.bucket_size_seconds,
        "algorithm_window": patient.window_size_seconds,
        "features": features,
        "notification_thresholds": {},
        "save_raw_eeg": patient.save_raw_eeg,
    }

    if patient.mo_count_threshold is not None:
        config["notification_thresholds"]["mo_count"] = patient.mo_count_threshold

    return config
