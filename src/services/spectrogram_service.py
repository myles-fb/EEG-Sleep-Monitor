"""Save and load per-window spectrogram data as compressed NPZ files."""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "spectrograms"
DATA_ROOT = Path(os.environ.get("SPECTROGRAM_DATA_DIR", str(_DEFAULT_DATA_ROOT)))


def _window_path(study_id: str, window_index: int, channel_index: int) -> Path:
    return DATA_ROOT / study_id / f"w{window_index:04d}_ch{channel_index:03d}.npz"


def save_window_spectrogram(
    study_id: str,
    window_index: int,
    channel_index: int,
    S: np.ndarray,
    T: np.ndarray,
    F: np.ndarray,
    band_envelopes: Optional[Dict[str, np.ndarray]] = None,
) -> Path:
    """Save a single window's spectrogram and optional band envelopes as compressed NPZ.

    Args:
        S: (n_times, n_freqs) power spectrogram for one channel.
        T: (n_times,) time vector (global, offset by window start).
        F: (n_freqs,) frequency vector.
        band_envelopes: dict mapping band label -> (n_times,) envelope.

    Returns:
        Path to the saved file.
    """
    path = _window_path(study_id, window_index, channel_index)
    path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {"S": S, "T": T, "F": F}
    if band_envelopes:
        for label, env in band_envelopes.items():
            save_dict[f"env_{label}"] = env

    np.savez_compressed(str(path), **save_dict)
    return path


def load_window_spectrogram(
    study_id: str,
    window_index: int,
    channel_index: int,
) -> Optional[Dict[str, np.ndarray]]:
    """Load a single window's NPZ. Returns dict with keys S, T, F, env_*."""
    path = _window_path(study_id, window_index, channel_index)
    if not path.exists():
        return None
    data = np.load(str(path))
    return dict(data)


def load_full_spectrogram(
    study_id: str,
    channel_index: int,
) -> Optional[Dict[str, np.ndarray]]:
    """Concatenate all windows for a channel into a single spectrogram.

    Returns dict with keys: S (n_times_total, n_freqs), T (n_times_total,),
    F (n_freqs,), and env_<band> (n_times_total,) for each band.
    """
    study_dir = DATA_ROOT / study_id
    if not study_dir.exists():
        return None

    pattern = f"w*_ch{channel_index:03d}.npz"
    files = sorted(study_dir.glob(pattern))
    if not files:
        return None

    all_S, all_T = [], []
    env_accum: Dict[str, list] = {}
    F = None

    for f in files:
        d = np.load(str(f))
        all_S.append(d["S"])
        all_T.append(d["T"])
        if F is None:
            F = d["F"]
        for key in d.files:
            if key.startswith("env_"):
                env_accum.setdefault(key, []).append(d[key])

    result: Dict[str, np.ndarray] = {
        "S": np.concatenate(all_S, axis=0),
        "T": np.concatenate(all_T, axis=0),
        "F": F,
    }
    for key, arrs in env_accum.items():
        result[key] = np.concatenate(arrs, axis=0)

    return result


def list_available_channels(study_id: str) -> List[int]:
    """Return sorted list of channel indices that have stored spectrogram data."""
    study_dir = DATA_ROOT / study_id
    if not study_dir.exists():
        return []

    channels = set()
    for f in study_dir.glob("w*_ch*.npz"):
        # filename: w0000_ch000.npz
        parts = f.stem.split("_ch")
        if len(parts) == 2:
            try:
                channels.add(int(parts[1]))
            except ValueError:
                pass
    return sorted(channels)
