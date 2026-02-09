#!/usr/bin/env python3
"""
MOs Q-Value Heatmap Pipeline: process a full EDF recording and produce
mobile-ready outputs (structured JSON + heatmap PNGs).

Splits the recording into fixed-length time buckets, runs the MOs pipeline
on all 18 LB-18 bipolar channels per bucket, then stitches q-values across
buckets into a q_tensor of shape [n_bands][n_channels][n_total_windows].

Outputs:
  <patient_id>_mos_data.json   — structured data for a mobile API
  <patient_id>_qmap_<band>.png — one heatmap per MO frequency band

Usage:
  python scripts/generate_mos_heatmaps.py \
    --edf /path/to/3.edf \
    --output-dir /path/to/results \
    --n-surrogates 1 \
    --bucket-length 300 \
    --envelope-window 120 \
    --envelope-step 30
"""

from __future__ import annotations

import argparse
import json
import sys
from math import ceil
from pathlib import Path

import numpy as np

# Ensure project root is on sys.path so we can import from src/ and scripts/
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.run_mos_edf_pipeline import (
    LB18_PAIRS,
    build_lb18_bpm_mask,
    load_edf_data,
)
from src.processing.mos import (
    MO_FREQ_BANDS,
    SPECTROGRAM_STEP_SEC,
    SPECTROGRAM_WINDOW_SEC,
    compute_mos_for_bucket,
)

# Display labels for the 18 LB-18 bipolar channels
CHANNEL_LABELS = [
    f"{left[3:-3]}-{right[3:-3]}" for left, right in LB18_PAIRS
]

BAND_LABELS = [f"{lo}_{hi}hz" for lo, hi in MO_FREQ_BANDS]
BAND_DISPLAY_NAMES = [f"{lo}-{hi} Hz" for lo, hi in MO_FREQ_BANDS]


def compute_window_times(
    n_samples: int,
    Fs: float,
    wintime_sec: float,
    winjump_sec: float,
) -> np.ndarray:
    """Return center-time (seconds) of each LASSO window for one bucket.

    Mirrors the windowing logic in mos._lasso_mo_q_single_channel, but
    operates on the spectrogram time axis to produce absolute seconds.
    """
    # Spectrogram time points
    win_spec = int(round(SPECTROGRAM_WINDOW_SEC * Fs))
    step_spec = int(round(SPECTROGRAM_STEP_SEC * Fs))
    if win_spec > n_samples:
        win_spec = n_samples
        step_spec = max(1, n_samples // 4)
    starts_spec = np.arange(0, n_samples - win_spec + 1, step_spec)
    T = (starts_spec + win_spec / 2) / Fs
    n_env = len(T)

    # Envelope sample rate
    Fs_env = 1.0 / (T[1] - T[0]) if len(T) > 1 else 1.0

    # LASSO sliding windows (same logic as _lasso_mo_q_single_channel)
    win_samples = max(1, int(round(wintime_sec * Fs_env)))
    jump_samples = max(1, int(round(winjump_sec * Fs_env)))
    if win_samples > n_env:
        win_samples = n_env
        jump_samples = n_env
    num_funs = max(0, (n_env - win_samples) // jump_samples)
    lasso_starts = [i * jump_samples for i in range(num_funs)]
    if n_env >= win_samples and (
        not lasso_starts or lasso_starts[-1] != n_env - win_samples
    ):
        lasso_starts.append(n_env - win_samples)

    centers = np.array(
        [T[s + win_samples // 2] for s in lasso_starts], dtype=np.float64
    )
    return centers


def build_qp_tensors(
    all_results: list[list],
    band_labels: list[str],
    window_times_per_bucket: list[np.ndarray],
    bucket_offsets_sec: list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate per-bucket MOsResults into full-recording tensors.

    Returns:
        q_tensor: (n_bands, n_channels, n_total_windows) q-value array
        p_tensor: (n_bands, n_channels, n_total_windows) p-value array
        time_centers: (n_total_windows,) absolute seconds from recording start
    """
    n_bands = len(band_labels)
    n_channels = len(all_results[0])  # 18

    # Build time_centers with bucket offsets
    time_parts = []
    for wt, offset in zip(window_times_per_bucket, bucket_offsets_sec):
        time_parts.append(wt + offset)
    time_centers = np.concatenate(time_parts)
    n_total_windows = len(time_centers)

    q_tensor = np.full((n_bands, n_channels, n_total_windows), np.nan)
    p_tensor = np.full((n_bands, n_channels, n_total_windows), np.nan)

    for b, band in enumerate(band_labels):
        col = 0
        for bucket_results in all_results:
            # bucket_results is a list of 18 MOsResult (one per channel)
            n_win = len(bucket_results[0].q_per_window_per_band[band])
            for ch in range(n_channels):
                q_arr = bucket_results[ch].q_per_window_per_band[band]
                q_tensor[b, ch, col : col + n_win] = q_arr
                p_arr = bucket_results[ch].p_per_window_per_band[band]
                p_tensor[b, ch, col : col + n_win] = p_arr
            col += n_win

    return q_tensor, p_tensor, time_centers


def render_band_heatmap(
    matrix: np.ndarray,
    time_centers: np.ndarray,
    channel_labels: list[str],
    band_display_name: str,
    out_path: Path,
) -> None:
    """Render and save one heatmap PNG for a single MO band."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_channels, n_windows = matrix.shape

    fig, ax = plt.subplots(figsize=(max(8, n_windows * 0.25), 6))

    # Mask NaN values so they appear as white
    masked = np.ma.masked_invalid(matrix)

    im = ax.imshow(
        masked,
        aspect="auto",
        cmap="YlOrRd",
        interpolation="nearest",
        vmin=0,
    )

    # Y-axis: channel labels
    ax.set_yticks(range(n_channels))
    ax.set_yticklabels(channel_labels, fontsize=8)

    # X-axis: time labels
    duration = time_centers[-1] if len(time_centers) else 0
    if duration >= 3600:
        fmt = lambda s: f"{int(s // 3600)}:{int((s % 3600) // 60):02d}:{int(s % 60):02d}"
    else:
        fmt = lambda s: f"{int(s // 60)}:{int(s % 60):02d}"

    # Show ~8-12 tick labels
    n_ticks = min(12, n_windows)
    if n_ticks > 1:
        tick_indices = np.linspace(0, n_windows - 1, n_ticks, dtype=int)
    else:
        tick_indices = np.array([0])
    ax.set_xticks(tick_indices)
    ax.set_xticklabels([fmt(time_centers[i]) for i in tick_indices], fontsize=7, rotation=45)
    ax.set_xlabel("Time", fontsize=9)

    ax.set_title(f"MO q-value: {band_display_name}", fontsize=11)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("q-value", fontsize=9)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_json_payload(
    patient_id: str,
    edf_file: str,
    duration_sec: float,
    sample_rate: float,
    bucket_length_sec: float,
    envelope_window_sec: float,
    envelope_step_sec: float,
    n_surrogates: int,
    n_buckets: int,
    channel_labels: list[str],
    band_labels: list[str],
    band_display_names: list[str],
    time_centers: np.ndarray,
    q_tensor: np.ndarray,
    p_tensor: np.ndarray,
    image_files: dict[str, str],
) -> dict:
    """Assemble the full JSON-serializable dict."""
    n_bands = len(band_labels)
    n_channels = len(channel_labels)

    def _nan_safe(v):
        """Convert NaN/inf to None for JSON serialization."""
        if isinstance(v, float) and (v != v or np.isinf(v)):
            return None
        return v

    # q_max and p_min per band per channel (aggregate across all windows)
    q_max_per_band_per_channel = []
    p_per_band_per_channel = []
    for b in range(n_bands):
        q_max_row = []
        p_row = []
        for ch in range(n_channels):
            q_vals = q_tensor[b, ch, :]
            finite_q = q_vals[np.isfinite(q_vals)]
            q_max_row.append(float(np.max(finite_q)) if len(finite_q) else 0.0)
            p_vals = p_tensor[b, ch, :]
            finite_p = p_vals[np.isfinite(p_vals)]
            p_row.append(float(np.min(finite_p)) if len(finite_p) else 0.5)
        q_max_per_band_per_channel.append(q_max_row)
        p_per_band_per_channel.append(p_row)

    # Convert q_tensor and p_tensor to nested lists with NaN -> None
    q_tensor_list = []
    p_tensor_list = []
    for b in range(n_bands):
        q_band_data = []
        p_band_data = []
        for ch in range(n_channels):
            q_band_data.append([_nan_safe(float(x)) for x in q_tensor[b, ch, :]])
            p_band_data.append([_nan_safe(float(x)) for x in p_tensor[b, ch, :]])
        q_tensor_list.append(q_band_data)
        p_tensor_list.append(p_band_data)

    return {
        "patient_id": patient_id,
        "edf_file": edf_file,
        "duration_sec": duration_sec,
        "sample_rate": sample_rate,
        "bucket_length_sec": bucket_length_sec,
        "envelope_window_sec": envelope_window_sec,
        "envelope_step_sec": envelope_step_sec,
        "n_surrogates": n_surrogates,
        "n_buckets": n_buckets,
        "channels": channel_labels,
        "band_labels": band_labels,
        "band_display_names": band_display_names,
        "time_centers_sec": [_nan_safe(float(t)) for t in time_centers],
        "q_tensor": q_tensor_list,
        "p_tensor": p_tensor_list,
        "q_max_per_band_per_channel": q_max_per_band_per_channel,
        "p_per_band_per_channel": p_per_band_per_channel,
        "image_files": image_files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate MOs q-value heatmaps and structured JSON from an EDF file."
    )
    parser.add_argument(
        "--edf", type=Path, required=True, help="Path to the EDF file"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Output directory"
    )
    parser.add_argument(
        "--n-surrogates", type=int, default=1, help="Number of surrogates (default 1)"
    )
    parser.add_argument(
        "--bucket-length", type=float, default=300.0, help="Bucket length in seconds (default 300 = 5 min)"
    )
    parser.add_argument(
        "--envelope-window", type=float, default=120.0, help="LASSO envelope window in seconds (default 120)"
    )
    parser.add_argument(
        "--envelope-step", type=float, default=30.0, help="LASSO envelope step in seconds (default 30)"
    )
    args = parser.parse_args()

    edf_path = args.edf.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    patient_id = edf_path.stem
    bucket_length_sec = args.bucket_length
    n_surrogates = args.n_surrogates
    wintime_sec = args.envelope_window
    winjump_sec = args.envelope_step

    # 1. Load EDF
    print(f"Loading {edf_path.name} ...")
    data, Fs, ch_names = load_edf_data(edf_path)
    n_total_samples = data.shape[1]
    duration_sec = n_total_samples / Fs
    print(f"  {len(ch_names)} channels, {n_total_samples} samples, {duration_sec:.1f}s @ {Fs} Hz")

    # 2. Build bipolar montage mask
    bpm_mask = build_lb18_bpm_mask(ch_names)

    # 3. Split into buckets and process
    bucket_samples = int(bucket_length_sec * Fs)
    n_buckets = ceil(n_total_samples / bucket_samples)
    print(f"Processing {n_buckets} bucket(s) of {bucket_length_sec}s each ...")

    all_results: list[list] = []
    window_times_per_bucket: list[np.ndarray] = []
    bucket_offsets: list[float] = []

    for i in range(n_buckets):
        start = i * bucket_samples
        end = min(start + bucket_samples, n_total_samples)
        bucket_data = data[:, start:end]
        bucket_offset = start / Fs
        print(f"  Bucket {i + 1}/{n_buckets}: samples {start}-{end} ({bucket_offset:.0f}s - {end / Fs:.0f}s)")

        results = compute_mos_for_bucket(
            bucket_data,
            Fs,
            timestamp=bucket_offset,
            bpm_mask=bpm_mask,
            channel_indices=list(range(18)),
            n_surrogates=n_surrogates,
            wintime_sec=wintime_sec,
            winjump_sec=winjump_sec,
        )
        all_results.append(results)

        wt = compute_window_times(
            end - start, Fs, wintime_sec, winjump_sec
        )
        window_times_per_bucket.append(wt)
        bucket_offsets.append(bucket_offset)

    # 4. Build q_tensor and p_tensor
    print("Building q_tensor and p_tensor ...")
    q_tensor, p_tensor, time_centers = build_qp_tensors(
        all_results, BAND_LABELS, window_times_per_bucket, bucket_offsets
    )
    print(f"  q_tensor shape: {q_tensor.shape}  (bands, channels, windows)")
    print(f"  p_tensor shape: {p_tensor.shape}  (bands, channels, windows)")

    # 5. Render heatmaps
    print("Rendering heatmaps ...")
    image_files: dict[str, str] = {}
    for b, (band_label, band_display) in enumerate(zip(BAND_LABELS, BAND_DISPLAY_NAMES)):
        fname = f"{patient_id}_qmap_{band_label}.png"
        out_path = output_dir / fname
        render_band_heatmap(
            q_tensor[b], time_centers, CHANNEL_LABELS, band_display, out_path
        )
        image_files[band_label] = fname
        print(f"  Wrote {out_path.name}")

    # 6. Write JSON
    payload = build_json_payload(
        patient_id=patient_id,
        edf_file=edf_path.name,
        duration_sec=duration_sec,
        sample_rate=Fs,
        bucket_length_sec=bucket_length_sec,
        envelope_window_sec=wintime_sec,
        envelope_step_sec=winjump_sec,
        n_surrogates=n_surrogates,
        n_buckets=n_buckets,
        channel_labels=CHANNEL_LABELS,
        band_labels=BAND_LABELS,
        band_display_names=BAND_DISPLAY_NAMES,
        time_centers=time_centers,
        q_tensor=q_tensor,
        p_tensor=p_tensor,
        image_files=image_files,
    )
    json_path = output_dir / f"{patient_id}_mos_data.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Wrote {json_path.name}")

    print("Done.")


if __name__ == "__main__":
    main()
