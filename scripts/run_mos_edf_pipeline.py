#!/usr/bin/env python3
"""
MOs testing pipeline: process 10 EDF files (1 per patient) and save results for visualization.

- Applies the 18-channel longitudinal bipolar montage (LB-18) from Bipolar_Montage_Code.md.
- Runs the MOs pipeline (mos.py) with:
  - 2 minute envelope window, 30 s step (1/4 * window)
  - 5 minute envelope window, 75 s step (1/4 * window)
- Writes one JSON result file per EDF per window config to an output directory.

Usage:
  python scripts/run_mos_edf_pipeline.py --input-dir /path/to/edfs --output-dir /path/to/results [--n-surrogates 50]
  python scripts/run_mos_edf_pipeline.py --input-dir /path/to/edfs --output-dir /path/to/results --patient-number 3   # only 3.edf

Requires: mne, numpy. Install with: pip install mne
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np

# LB-18 longitudinal bipolar pairs (left_name, right_name) for (left - right)
# From documentation/Bipolar_Montage_Code.md
LB18_PAIRS = [
    ("EEGFp1Ref", "EEGF7Ref"),   # (1)  Fp1-F7
    ("EEGF7Ref", "EEGT7Ref"),    # (2)  F7-T7
    ("EEGT7Ref", "EEGP7Ref"),    # (3)  T7-P7
    ("EEGP7Ref", "EEGO1Ref"),    # (4)  P7-O1
    ("EEGFp1Ref", "EEGF3Ref"),   # (5)  Fp1-F3
    ("EEGF3Ref", "EEGC3Ref"),    # (6)  F3-C3
    ("EEGC3Ref", "EEGP3Ref"),    # (7)  C3-P3
    ("EEGP3Ref", "EEGO1Ref"),    # (8)  P3-O1
    ("EEGFp2Ref", "EEGF8Ref"),   # (9)  Fp2-F8
    ("EEGF8Ref", "EEGT8Ref"),    # (10) F8-T8
    ("EEGT8Ref", "EEGP8Ref"),    # (11) T8-P8
    ("EEGP8Ref", "EEGO2Ref"),    # (12) P8-O2
    ("EEGFp2Ref", "EEGF4Ref"),   # (13) Fp2-F4
    ("EEGF4Ref", "EEGC4Ref"),    # (14) F4-C4
    ("EEGC4Ref", "EEGP4Ref"),    # (15) C4-P4
    ("EEGP4Ref", "EEGO2Ref"),    # (16) P4-O2
    ("EEGFzRef", "EEGCzRef"),    # (17) Fz-Cz
    ("EEGCzRef", "EEGPzRef"),    # (18) Cz-Pz
]


def _lb18_name_to_edf_style(name: str) -> str:
    """Convert MATLAB-style label (e.g. EEGFp1Ref) to EDF-style (e.g. EEG Fp1-Ref)."""
    if name.startswith("EEG") and name.endswith("Ref") and len(name) > 6:
        return "EEG " + name[3:-3] + "-Ref"
    return name


def build_lb18_bpm_mask(channel_names: list[str]) -> np.ndarray:
    """
    Build (18, 2) bipolar montage indices for LB-18 from EDF channel labels.

    Tries exact match first, then EDF-style names (e.g. EEG Fp1-Ref for EEGFp1Ref).
    Returns indices into the channel list for (left - right) for each pair.

    Raises:
        ValueError: if any channel in LB18_PAIRS is not found in channel_names.
    """
    name_to_idx = {name.strip(): i for i, name in enumerate(channel_names)}
    bpm = np.zeros((18, 2), dtype=np.intp)
    for i, (left, right) in enumerate(LB18_PAIRS):
        left_idx = name_to_idx.get(left) or name_to_idx.get(_lb18_name_to_edf_style(left))
        right_idx = name_to_idx.get(right) or name_to_idx.get(_lb18_name_to_edf_style(right))
        if left_idx is None:
            raise ValueError(f"Channel not found: {left!r} (tried EDF-style {_lb18_name_to_edf_style(left)!r}). Available: {list(name_to_idx.keys())[:8]}...")
        if right_idx is None:
            raise ValueError(f"Channel not found: {right!r} (tried EDF-style {_lb18_name_to_edf_style(right)!r}). Available: {list(name_to_idx.keys())[:8]}...")
        bpm[i, 0] = left_idx
        bpm[i, 1] = right_idx
    return bpm


def load_edf_data(edf_path: Path):
    """
    Load EDF file and return (data, sample_rate_hz, channel_names).

    data: (n_channels, n_samples) float64 in volts or µV depending on file.
    Suppresses MNE RuntimeWarnings for non-EEG channels (scaling/physical range).
    """
    try:
        import mne
    except ImportError as e:
        raise ImportError("EDF loading requires mne. Install with: pip install mne") from e
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    data, times = raw.get_data(return_times=True)
    # get_data returns (n_channels, n_samples)
    sfreq = raw.info["sfreq"]
    ch_names = list(raw.ch_names)
    return np.asarray(data, dtype=np.float64), float(sfreq), ch_names


def _nan_to_none(x):
    """Recursively convert NaN to None for JSON."""
    if isinstance(x, (float, np.floating)) and (x != x or np.isnan(x)):
        return None
    if isinstance(x, np.ndarray):
        return [_nan_to_none(v) for v in x.tolist()]
    if isinstance(x, dict):
        return {k: _nan_to_none(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_nan_to_none(v) for v in x]
    return x


def result_to_dict(result) -> dict:
    """Convert MOsResult to a JSON-serializable dict (NaN -> None)."""
    return {
        "timestamp": result.timestamp,
        "bucket_length_seconds": result.bucket_length_seconds,
        "sample_rate": result.sample_rate,
        "q_per_band": result.q_per_band,
        "p_per_band": result.p_per_band,
        "q_per_window_per_band": {k: _nan_to_none(v.tolist()) for k, v in result.q_per_window_per_band.items()},
        "p_per_window_per_band": {k: _nan_to_none(v.tolist()) for k, v in result.p_per_window_per_band.items()},
        "dominant_freq_hz_per_window_per_band": {
            k: _nan_to_none(v.tolist()) for k, v in result.dominant_freq_hz_per_window_per_band.items()
        },
        "dominant_freq_hz_per_band": _nan_to_none(result.dominant_freq_hz_per_band),
        "n_surrogates": result.n_surrogates,
        "channel_index": result.channel_index,
    }


def run_pipeline(
    input_dir: Path,
    output_dir: Path,
    n_surrogates: int = 50,
    channel_index: int = 0,
    patient_number: Optional[Union[str, int]] = None,
    all_channels: bool = False,
) -> None:
    """
    Find all .edf files in input_dir, run MOs with 2 min and 5 min envelope windows,
    and save one JSON per (file, window_config) into output_dir.

    If patient_number is set, only the EDF named "{patient_number}.edf" is processed.
    If all_channels is True, process all 18 bipolar channels in one call (shared surrogates).
    """
    # Ensure project root is on path for src.processing.mos
    _script_dir = Path(__file__).resolve().parent
    _project_root = _script_dir.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))
    try:
        from src.processing.mos import compute_mos_for_bucket
    except ImportError:
        from processing.mos import compute_mos_for_bucket

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    edf_files = sorted(input_dir.glob("*.edf"))
    if patient_number is not None:
        target_name = f"{patient_number}.edf"
        edf_files = [p for p in edf_files if p.name == target_name]
        if not edf_files:
            print(f"No .edf file named {target_name!r} in {input_dir}", file=sys.stderr)
            return
    if not edf_files:
        print(f"No .edf files found in {input_dir}", file=sys.stderr)
        return

    configs = [
        ("2min", 120.0, 30.0),   # 2 min window, 1/4 * window step
        ("5min", 300.0, 75.0),   # 5 min window, 1/4 * window step
    ]

    for edf_path in edf_files:
        patient_id = edf_path.stem
        print(f"Loading {edf_path.name} ...")
        try:
            data, Fs, ch_names = load_edf_data(edf_path)
        except Exception as e:
            print(f"  Skip {edf_path.name}: {e}", file=sys.stderr)
            continue
        try:
            bpm_mask = build_lb18_bpm_mask(ch_names)
        except ValueError as e:
            print(f"  Skip {edf_path.name}: {e}", file=sys.stderr)
            continue
        # (n_ch_edf, n_samples) -> apply bipolar -> (18, n_samples)
        for label, wintime_sec, winjump_sec in configs:
            out_name = f"{patient_id}_envelope_{label}.json"
            out_path = output_dir / out_name
            if all_channels:
                print(f"  Running MOs all 18 channels ({label}: window={wintime_sec}s, step={winjump_sec}s) ...")
            else:
                print(f"  Running MOs ({label}: window={wintime_sec}s, step={winjump_sec}s) ...")
            try:
                result = compute_mos_for_bucket(
                    data,
                    Fs,
                    timestamp=0.0,
                    bpm_mask=bpm_mask,
                    n_surrogates=n_surrogates,
                    channel_index=channel_index,
                    channel_indices=list(range(18)) if all_channels else None,
                    wintime_sec=wintime_sec,
                    winjump_sec=winjump_sec,
                )
                if isinstance(result, list):
                    result_data = [result_to_dict(r) for r in result]
                else:
                    result_data = result_to_dict(result)
                payload = {
                    "patient_id": patient_id,
                    "edf_file": edf_path.name,
                    "envelope_window_sec": wintime_sec,
                    "envelope_step_sec": winjump_sec,
                    "n_samples": int(data.shape[1]),
                    "duration_sec": float(data.shape[1] / Fs),
                    "result": result_data,
                }
                with open(out_path, "w") as f:
                    json.dump(payload, f, indent=2)
                print(f"  Wrote {out_path}")
            except Exception as e:
                print(f"  Failed {label}: {e}", file=sys.stderr)
    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Run MOs pipeline on EDF files (bipolar LB-18, 2 min and 5 min envelope windows), save results for visualization."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing .edf files (e.g. 10 files, 1 per patient)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write JSON result files",
    )
    parser.add_argument(
        "--n-surrogates",
        type=int,
        default=50,
        help="Number of surrogates for p-value (default 50)",
    )
    parser.add_argument(
        "--channel-index",
        type=int,
        default=0,
        help="Bipolar channel index to report (0..17, default 0)",
    )
    parser.add_argument(
        "--patient-number",
        type=str,
        default=None,
        metavar="N",
        help="If set, run only on the EDF named N.edf (e.g. 3 for 3.edf)",
    )
    parser.add_argument(
        "--all-channels",
        action="store_true",
        default=False,
        help="Process all 18 bipolar channels in one call (shared surrogates). "
             "Overrides --channel-index.",
    )
    args = parser.parse_args()
    run_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        n_surrogates=args.n_surrogates,
        channel_index=args.channel_index,
        patient_number=args.patient_number,
        all_channels=args.all_channels,
    )


if __name__ == "__main__":
    main()
