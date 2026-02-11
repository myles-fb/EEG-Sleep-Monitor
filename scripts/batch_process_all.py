#!/usr/bin/env python3
"""
Batch process all EDF files: generate spectrograms and MOs heatmaps.

For each .edf file in the input directory, this script:
1. Generates channel-averaged full-signal spectrograms
2. Generates channel-averaged envelope spectrograms (for configurable bands)
3. Generates MOs q-value heatmaps and structured JSON data

All outputs are organized by patient ID in the output directory:

  output/
    {patient_id}/
      {patient_id}_spectrogram_full.npz
      {patient_id}_spectrogram_full.png
      {patient_id}_spectrogram_envelope_{band}.npz
      {patient_id}_spectrogram_envelope_{band}.png
      {patient_id}_mos_data.json
      {patient_id}_qmap_{band}.png  (one per MO band)

Usage:
  python scripts/batch_process_all.py \
    --input-dir data \
    --output-dir output \
    --n-surrogates 1 \
    --bucket-length 300 \
    --envelope-window 120 \
    --envelope-step 30

  # Process only specific patients
  python scripts/batch_process_all.py \
    --input-dir data \
    --output-dir output \
    --patients 3 7 10

  # Skip spectrograms (only generate MOs heatmaps)
  python scripts/batch_process_all.py \
    --input-dir data \
    --output-dir output \
    --skip-spectrograms

  # Skip MOs heatmaps (only generate spectrograms)
  python scripts/batch_process_all.py \
    --input-dir data \
    --output-dir output \
    --skip-mos
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np

# Ensure project root is on sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.compute_spectrograms import (
    compute_averaged_spectrogram,
    compute_envelope_spectrogram,
)
from scripts.generate_mos_heatmaps import (
    BAND_DISPLAY_NAMES,
    BAND_LABELS,
    CHANNEL_LABELS,
    build_json_payload,
    build_qp_tensors,
    compute_window_times,
    render_band_heatmap,
)
from scripts.run_mos_edf_pipeline import (
    build_lb18_bpm_mask,
    load_edf_data,
)
from src.processing.mos import compute_mos_for_bucket


def process_spectrograms(
    patient_id: str,
    data: np.ndarray,
    Fs: float,
    bpm_mask: np.ndarray,
    output_dir: Path,
    envelope_bands: List[tuple[float, float]],
    skip_plots: bool = False,
) -> None:
    """
    Generate and save full-signal and envelope spectrograms for one patient.

    Args:
        patient_id: patient identifier (e.g., "3")
        data: (n_channels, n_samples) EEG data
        Fs: sampling rate in Hz
        bpm_mask: (18, 2) bipolar montage mask
        output_dir: patient-specific output directory
        envelope_bands: list of (f_low, f_high) tuples for envelope extraction
        skip_plots: if True, skip PNG generation (only save .npz)
    """
    print(f"  [Spectrograms] Computing full-signal spectrogram ...")
    S_avg, T, F = compute_averaged_spectrogram(data, Fs, bpm_mask)

    # Save full-signal spectrogram
    np.savez(
        output_dir / f"{patient_id}_spectrogram_full.npz",
        S=S_avg, T=T, F=F, Fs=Fs,
    )
    print(f"    Saved {patient_id}_spectrogram_full.npz")

    # Plot full-signal spectrogram
    if not skip_plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 6))
        S_db = 10 * np.log10(S_avg.T + 1e-12)
        im = ax.imshow(
            S_db,
            aspect="auto",
            origin="lower",
            extent=[T[0], T[-1], F[0], F[-1]],
            cmap="viridis",
            interpolation="bilinear",
        )
        ax.set_xlabel("Time (s)", fontsize=11)
        ax.set_ylabel("Frequency (Hz)", fontsize=11)
        ax.set_title(f"Full-Signal Spectrogram: {patient_id}", fontsize=12)
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
        cbar.set_label("Power (dB)", fontsize=10)
        fig.savefig(
            output_dir / f"{patient_id}_spectrogram_full.png",
            dpi=150, bbox_inches="tight"
        )
        plt.close(fig)
        print(f"    Saved {patient_id}_spectrogram_full.png")

    # Envelope spectrograms for each band
    for f_low, f_high in envelope_bands:
        print(f"  [Spectrograms] Computing envelope spectrogram ({f_low}-{f_high} Hz) ...")
        S_env, T_env, F_env = compute_envelope_spectrogram(
            data, Fs, bpm_mask, envelope_band=(f_low, f_high)
        )

        # Save envelope spectrogram
        band_label = f"{f_low}_{f_high}hz"
        np.savez(
            output_dir / f"{patient_id}_spectrogram_envelope_{band_label}.npz",
            S_env=S_env, T_env=T_env, F_env=F_env,
            envelope_band=(f_low, f_high), Fs=Fs,
        )
        print(f"    Saved {patient_id}_spectrogram_envelope_{band_label}.npz")

        # Plot envelope spectrogram
        if not skip_plots:
            fig, ax = plt.subplots(figsize=(12, 6))
            S_env_db = 10 * np.log10(S_env.T + 1e-12)
            im = ax.imshow(
                S_env_db,
                aspect="auto",
                origin="lower",
                extent=[T_env[0], T_env[-1], F_env[0], F_env[-1]],
                cmap="plasma",
                interpolation="bilinear",
            )
            ax.set_xlabel("Time (s)", fontsize=11)
            ax.set_ylabel("Modulation Frequency (Hz)", fontsize=11)
            ax.set_title(
                f"Envelope Spectrogram ({f_low}-{f_high} Hz): {patient_id}",
                fontsize=12
            )
            cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
            cbar.set_label("Envelope Power (dB)", fontsize=10)
            fig.savefig(
                output_dir / f"{patient_id}_spectrogram_envelope_{band_label}.png",
                dpi=150, bbox_inches="tight"
            )
            plt.close(fig)
            print(f"    Saved {patient_id}_spectrogram_envelope_{band_label}.png")


def process_mos_heatmaps(
    patient_id: str,
    edf_filename: str,
    data: np.ndarray,
    Fs: float,
    bpm_mask: np.ndarray,
    output_dir: Path,
    n_surrogates: int,
    bucket_length_sec: float,
    envelope_window_sec: float,
    envelope_step_sec: float,
) -> None:
    """
    Generate and save MOs q-value heatmaps and structured JSON for one patient.

    Args:
        patient_id: patient identifier (e.g., "3")
        edf_filename: original EDF filename
        data: (n_channels, n_samples) EEG data
        Fs: sampling rate in Hz
        bpm_mask: (18, 2) bipolar montage mask
        output_dir: patient-specific output directory
        n_surrogates: number of surrogates for p-value
        bucket_length_sec: bucket length in seconds
        envelope_window_sec: LASSO envelope window in seconds
        envelope_step_sec: LASSO envelope step in seconds
    """
    from math import ceil
    import json

    n_total_samples = data.shape[1]
    duration_sec = n_total_samples / Fs

    print(f"  [MOs] Processing with {n_surrogates} surrogate(s), "
          f"bucket_length={bucket_length_sec}s, "
          f"envelope_window={envelope_window_sec}s ...")

    # Split into buckets
    bucket_samples = int(bucket_length_sec * Fs)
    n_buckets = ceil(n_total_samples / bucket_samples)

    all_results: list[list] = []
    window_times_per_bucket: list[np.ndarray] = []
    bucket_offsets: list[float] = []

    for i in range(n_buckets):
        start = i * bucket_samples
        end = min(start + bucket_samples, n_total_samples)
        bucket_data = data[:, start:end]
        bucket_offset = start / Fs
        print(f"    Bucket {i + 1}/{n_buckets}: samples {start}-{end} "
              f"({bucket_offset:.0f}s - {end / Fs:.0f}s)")

        results = compute_mos_for_bucket(
            bucket_data,
            Fs,
            timestamp=bucket_offset,
            bpm_mask=bpm_mask,
            channel_indices=list(range(18)),
            n_surrogates=n_surrogates,
            wintime_sec=envelope_window_sec,
            winjump_sec=envelope_step_sec,
        )
        all_results.append(results)

        wt = compute_window_times(
            end - start, Fs, envelope_window_sec, envelope_step_sec
        )
        window_times_per_bucket.append(wt)
        bucket_offsets.append(bucket_offset)

    # Build q_tensor and p_tensor
    print(f"  [MOs] Building q_tensor and p_tensor ...")
    q_tensor, p_tensor, time_centers = build_qp_tensors(
        all_results, BAND_LABELS, window_times_per_bucket, bucket_offsets
    )
    print(f"    q_tensor shape: {q_tensor.shape}  (bands, channels, windows)")

    # Render heatmaps
    print(f"  [MOs] Rendering heatmaps ...")
    image_files: dict[str, str] = {}
    for b, (band_label, band_display) in enumerate(zip(BAND_LABELS, BAND_DISPLAY_NAMES)):
        fname = f"{patient_id}_qmap_{band_label}.png"
        out_path = output_dir / fname
        render_band_heatmap(
            q_tensor[b], time_centers, CHANNEL_LABELS, band_display, out_path
        )
        image_files[band_label] = fname
        print(f"    Saved {fname}")

    # Write JSON
    payload = build_json_payload(
        patient_id=patient_id,
        edf_file=edf_filename,
        duration_sec=duration_sec,
        sample_rate=Fs,
        bucket_length_sec=bucket_length_sec,
        envelope_window_sec=envelope_window_sec,
        envelope_step_sec=envelope_step_sec,
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
    print(f"    Saved {patient_id}_mos_data.json")


def process_one_patient(
    edf_path: Path,
    output_root: Path,
    n_surrogates: int,
    bucket_length_sec: float,
    envelope_window_sec: float,
    envelope_step_sec: float,
    envelope_bands: List[tuple[float, float]],
    skip_spectrograms: bool,
    skip_mos: bool,
    skip_plots: bool,
    skip_existing: bool,
) -> None:
    """Process one EDF file: generate spectrograms and MOs heatmaps."""
    patient_id = edf_path.stem
    print(f"\n{'=' * 70}")
    print(f"Processing: {edf_path.name} (patient ID: {patient_id})")
    print(f"{'=' * 70}")

    # Create patient-specific output directory
    output_dir = output_root / patient_id

    # Skip if directory exists and skip_existing flag is set
    if skip_existing and output_dir.exists():
        print(f"⏭️  SKIPPED: Output directory already exists: {output_dir}")
        print(f"   Use --force to overwrite existing outputs")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load EDF
    print(f"Loading {edf_path.name} ...")
    try:
        data, Fs, ch_names = load_edf_data(edf_path)
    except Exception as e:
        print(f"  ERROR: Failed to load EDF: {e}")
        return

    n_samples = data.shape[1]
    duration_sec = n_samples / Fs
    print(f"  {len(ch_names)} channels, {n_samples} samples, "
          f"{duration_sec:.1f}s @ {Fs} Hz")

    # Build bipolar montage
    try:
        bpm_mask = build_lb18_bpm_mask(ch_names)
        print(f"  Built LB-18 bipolar montage (18 channels)")
    except ValueError as e:
        print(f"  ERROR: Bipolar montage failed: {e}")
        return

    # Process spectrograms
    if not skip_spectrograms:
        try:
            process_spectrograms(
                patient_id, data, Fs, bpm_mask, output_dir,
                envelope_bands, skip_plots
            )
        except Exception as e:
            print(f"  ERROR: Spectrogram processing failed: {e}")
            import traceback
            traceback.print_exc()

    # Process MOs heatmaps
    if not skip_mos:
        try:
            process_mos_heatmaps(
                patient_id, edf_path.name, data, Fs, bpm_mask, output_dir,
                n_surrogates, bucket_length_sec,
                envelope_window_sec, envelope_step_sec
            )
        except Exception as e:
            print(f"  ERROR: MOs processing failed: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nCompleted: {patient_id} -> {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch process all EDF files: generate spectrograms and MOs heatmaps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dir", type=Path, default=Path("data"),
        help="Directory containing .edf files"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output"),
        help="Root output directory (patient subdirs will be created)"
    )
    parser.add_argument(
        "--patients", type=str, nargs="+",
        help="Process only specific patient IDs (e.g., --patients 3 7 10)"
    )
    parser.add_argument(
        "--n-surrogates", type=int, default=1,
        help="Number of surrogates for MOs p-value (default 1)"
    )
    parser.add_argument(
        "--bucket-length", type=float, default=300.0,
        help="MOs bucket length in seconds (default 300 = 5 min)"
    )
    parser.add_argument(
        "--envelope-window", type=float, default=120.0,
        help="MOs LASSO envelope window in seconds (default 120 = 2 min)"
    )
    parser.add_argument(
        "--envelope-step", type=float, default=30.0,
        help="MOs LASSO envelope step in seconds (default 30)"
    )
    parser.add_argument(
        "--envelope-bands", type=str, default="0.5-3,3-8,8-15,15-30",
        help="Comma-separated envelope bands for spectrograms (e.g., '0.5-3,3-8,8-15')"
    )
    parser.add_argument(
        "--skip-spectrograms", action="store_true",
        help="Skip spectrogram generation (only MOs heatmaps)"
    )
    parser.add_argument(
        "--skip-mos", action="store_true",
        help="Skip MOs heatmap generation (only spectrograms)"
    )
    parser.add_argument(
        "--skip-plots", action="store_true",
        help="Skip PNG generation (only save .npz and .json)"
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip patients whose output directories already exist"
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_root = args.output_dir.resolve()

    if not input_dir.exists():
        print(f"ERROR: Input directory does not exist: {input_dir}")
        sys.exit(1)

    # Parse envelope bands
    envelope_bands = []
    for band_str in args.envelope_bands.split(","):
        try:
            low, high = map(float, band_str.strip().split("-"))
            envelope_bands.append((low, high))
        except ValueError:
            print(f"ERROR: Invalid envelope band format: {band_str!r}")
            print("  Expected format: 'low-high' (e.g., '0.5-3')")
            sys.exit(1)

    # Find EDF files
    edf_files = sorted(input_dir.glob("*.edf"))
    if not edf_files:
        print(f"ERROR: No .edf files found in {input_dir}")
        sys.exit(1)

    # Filter by patient IDs if specified
    if args.patients:
        patient_set = set(args.patients)
        edf_files = [p for p in edf_files if p.stem in patient_set]
        if not edf_files:
            print(f"ERROR: No matching .edf files for patients: {args.patients}")
            sys.exit(1)

    print(f"Found {len(edf_files)} EDF file(s) to process:")
    for edf_path in edf_files:
        print(f"  - {edf_path.name}")

    # Process each patient
    for edf_path in edf_files:
        process_one_patient(
            edf_path,
            output_root,
            n_surrogates=args.n_surrogates,
            bucket_length_sec=args.bucket_length,
            envelope_window_sec=args.envelope_window,
            envelope_step_sec=args.envelope_step,
            envelope_bands=envelope_bands,
            skip_spectrograms=args.skip_spectrograms,
            skip_mos=args.skip_mos,
            skip_plots=args.skip_plots,
            skip_existing=args.skip_existing,
        )

    print(f"\n{'=' * 70}")
    print(f"Batch processing complete!")
    print(f"Output directory: {output_root}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
