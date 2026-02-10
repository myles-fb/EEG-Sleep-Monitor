#!/usr/bin/env python3
"""
Standalone spectrogram computation utilities.

Provides functions to compute:
1. Channel-averaged full-signal (first-order) spectrograms
2. Channel-averaged second-order (envelope) spectrograms

These functions extract the core spectrogram computation logic from mos.py
without running the full MOs pipeline (no GESD, LASSO, surrogates, etc.).

Usage:
  from scripts.compute_spectrograms import compute_averaged_spectrogram, compute_envelope_spectrogram

  # Load your EEG data
  data, Fs, ch_names = load_edf_data(edf_path)
  bpm_mask = build_lb18_bpm_mask(ch_names)

  # Channel-averaged full-signal spectrogram
  S_avg, T, F = compute_averaged_spectrogram(data, Fs, bpm_mask)

  # Channel-averaged envelope spectrogram (second-order)
  S_env, T_env, F_env = compute_envelope_spectrogram(
      data, Fs, bpm_mask,
      envelope_band=(0.5, 3.0)  # frequency band for envelope extraction
  )
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# Ensure project root is on sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.run_mos_edf_pipeline import (
    build_lb18_bpm_mask,
    load_edf_data,
)
from src.processing.mos import (
    FPASS,
    SPECTROGRAM_STEP_SEC,
    SPECTROGRAM_WINDOW_SEC,
    TAPERS_K,
    TAPERS_NW,
    apply_bipolar_montage,
    extract_band_envelope,
    multitaper_spectrogram,
)


def compute_averaged_spectrogram(
    eeg_data: np.ndarray,
    Fs: float,
    bpm_mask: Optional[np.ndarray] = None,
    window_sec: float = SPECTROGRAM_WINDOW_SEC,
    step_sec: float = SPECTROGRAM_STEP_SEC,
    nw: float = TAPERS_NW,
    k: int = TAPERS_K,
    fpass: Tuple[float, float] = FPASS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute channel-averaged full-signal (first-order) spectrogram.

    This computes the multi-taper spectrogram for all channels and averages
    the power across channels to produce a single 2D time-frequency representation.

    Args:
        eeg_data: (channels, samples) raw EEG data
        Fs: sampling rate in Hz
        bpm_mask: optional (n_bipolar, 2) bipolar montage indices;
                  if None, uses eeg_data as-is
        window_sec: spectrogram window length in seconds (default 30)
        step_sec: spectrogram step in seconds (default 6)
        nw: time-bandwidth product for DPSS tapers (default 3)
        k: number of tapers (default 5)
        fpass: (f_low, f_high) frequency range in Hz (default (0, 100))

    Returns:
        S_avg: (n_times, n_freqs) channel-averaged power spectrogram
        T: (n_times,) time vector in seconds (center of each window)
        F: (n_freqs,) frequency vector in Hz
    """
    # Apply bipolar montage if provided
    if bpm_mask is not None:
        eeg = apply_bipolar_montage(eeg_data, bpm_mask)
    else:
        eeg = np.asarray(eeg_data, dtype=np.float64)

    # Compute multi-taper spectrogram: (n_times, n_freqs, n_channels)
    S, T, F = multitaper_spectrogram(
        eeg, Fs,
        window_sec=window_sec,
        step_sec=step_sec,
        nw=nw,
        k=k,
        fpass=fpass,
    )

    # Average power across channels
    S_avg = np.mean(S, axis=2)  # (n_times, n_freqs)

    return S_avg, T, F


def compute_envelope_spectrogram(
    eeg_data: np.ndarray,
    Fs: float,
    bpm_mask: Optional[np.ndarray] = None,
    envelope_band: Tuple[float, float] = (0.5, 3.0),
    window_sec: float = SPECTROGRAM_WINDOW_SEC,
    step_sec: float = SPECTROGRAM_STEP_SEC,
    nw: float = TAPERS_NW,
    k: int = TAPERS_K,
    fpass: Tuple[float, float] = FPASS,
    envelope_window_sec: Optional[float] = None,
    envelope_step_sec: Optional[float] = None,
    envelope_nw: float = 2.0,
    envelope_k: int = 3,
    envelope_fpass: Tuple[float, float] = (0, 1.0),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute channel-averaged second-order (envelope) spectrogram.

    This is a two-stage process:
    1. Compute the full-signal spectrogram
    2. Extract band-limited envelope (time series of power in a frequency band)
    3. Compute another spectrogram on the envelope itself
    4. Average across channels

    The envelope spectrogram reveals slow modulations (e.g., 0.5-3 Hz oscillations)
    in the amplitude of a faster frequency band.

    Args:
        eeg_data: (channels, samples) raw EEG data
        Fs: sampling rate in Hz
        bpm_mask: optional (n_bipolar, 2) bipolar montage indices;
                  if None, uses eeg_data as-is
        envelope_band: (f_low, f_high) frequency band for envelope extraction in Hz
                       (default (0.5, 3.0) for delta/slow-wave envelope)
        window_sec: first-stage spectrogram window length in seconds (default 30)
        step_sec: first-stage spectrogram step in seconds (default 6)
        nw: time-bandwidth product for first-stage tapers (default 3)
        k: number of first-stage tapers (default 5)
        fpass: first-stage frequency range in Hz (default (0, 100))
        envelope_window_sec: second-stage spectrogram window length in seconds;
                             if None, uses ~60s or 1/4 of envelope duration
        envelope_step_sec: second-stage spectrogram step in seconds;
                           if None, uses envelope_window_sec / 5
        envelope_nw: time-bandwidth product for second-stage tapers (default 2.0)
        envelope_k: number of second-stage tapers (default 3)
        envelope_fpass: second-stage frequency range in Hz (default (0, 1.0))
                        to capture slow modulations

    Returns:
        S_env_avg: (n_times_env, n_freqs_env) channel-averaged envelope power spectrogram
        T_env: (n_times_env,) time vector in seconds for envelope spectrogram
        F_env: (n_freqs_env,) frequency vector in Hz for envelope spectrogram
    """
    # Apply bipolar montage if provided
    if bpm_mask is not None:
        eeg = apply_bipolar_montage(eeg_data, bpm_mask)
    else:
        eeg = np.asarray(eeg_data, dtype=np.float64)

    # Stage 1: Compute full-signal spectrogram
    S, T, F = multitaper_spectrogram(
        eeg, Fs,
        window_sec=window_sec,
        step_sec=step_sec,
        nw=nw,
        k=k,
        fpass=fpass,
    )  # (n_times, n_freqs, n_channels)

    # Stage 2: Extract band-limited envelope per channel
    f_low, f_high = envelope_band
    env = extract_band_envelope(S, F, f_low, f_high)  # (n_times, n_channels)

    # Determine envelope sample rate
    Fs_env = 1.0 / (T[1] - T[0]) if T.size > 1 else 1.0

    # Auto-determine envelope spectrogram parameters if not provided
    envelope_duration_sec = env.shape[0] / Fs_env
    if envelope_window_sec is None:
        # Use ~60s or 1/4 of duration, whichever is smaller
        envelope_window_sec = min(60.0, envelope_duration_sec / 4.0)
        envelope_window_sec = max(envelope_window_sec, 10.0)  # at least 10s
    if envelope_step_sec is None:
        envelope_step_sec = envelope_window_sec / 5.0

    # Stage 3: Compute spectrogram on envelope for each channel
    # Reshape env to (n_channels, n_times) for multitaper_spectrogram
    env_transposed = env.T  # (n_channels, n_times)

    S_env, T_env, F_env = multitaper_spectrogram(
        env_transposed, Fs_env,
        window_sec=envelope_window_sec,
        step_sec=envelope_step_sec,
        nw=envelope_nw,
        k=envelope_k,
        fpass=envelope_fpass,
    )  # (n_times_env, n_freqs_env, n_channels)

    # Average across channels
    S_env_avg = np.mean(S_env, axis=2)  # (n_times_env, n_freqs_env)

    # Adjust T_env to be in absolute seconds (relative to original signal)
    # T_env is currently relative to the envelope; we need to map it back to original time
    # The envelope starts at T[0], so add offset
    T_env_absolute = T_env + T[0]

    return S_env_avg, T_env_absolute, F_env


def main():
    """
    Command-line interface for computing and visualizing spectrograms.

    Usage:
      python scripts/compute_spectrograms.py --edf /path/to/file.edf --output-dir /path/to/output
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute channel-averaged spectrograms from EDF files"
    )
    parser.add_argument(
        "--edf", type=Path, required=True, help="Path to EDF file"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Output directory for spectrograms"
    )
    parser.add_argument(
        "--envelope-band-low", type=float, default=0.5,
        help="Envelope band lower frequency in Hz (default 0.5)"
    )
    parser.add_argument(
        "--envelope-band-high", type=float, default=3.0,
        help="Envelope band upper frequency in Hz (default 3.0)"
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip plotting (only save numpy arrays)"
    )
    args = parser.parse_args()

    edf_path = args.edf.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    patient_id = edf_path.stem

    # Load EEG data
    print(f"Loading {edf_path.name} ...")
    data, Fs, ch_names = load_edf_data(edf_path)
    print(f"  {len(ch_names)} channels, {data.shape[1]} samples, {data.shape[1]/Fs:.1f}s @ {Fs} Hz")

    # Build bipolar montage
    bpm_mask = build_lb18_bpm_mask(ch_names)
    print(f"  Built LB-18 bipolar montage (18 channels)")

    # Compute full-signal spectrogram
    print("Computing channel-averaged full-signal spectrogram ...")
    S_avg, T, F = compute_averaged_spectrogram(data, Fs, bpm_mask)
    print(f"  Shape: {S_avg.shape} (n_times={len(T)}, n_freqs={len(F)})")

    # Save full-signal spectrogram
    np.savez(
        output_dir / f"{patient_id}_spectrogram_full.npz",
        S=S_avg, T=T, F=F,
        Fs=Fs,
    )
    print(f"  Saved {patient_id}_spectrogram_full.npz")

    # Compute envelope spectrogram
    envelope_band = (args.envelope_band_low, args.envelope_band_high)
    print(f"Computing channel-averaged envelope spectrogram ({envelope_band[0]}-{envelope_band[1]} Hz) ...")
    S_env_avg, T_env, F_env = compute_envelope_spectrogram(
        data, Fs, bpm_mask, envelope_band=envelope_band
    )
    print(f"  Shape: {S_env_avg.shape} (n_times={len(T_env)}, n_freqs={len(F_env)})")

    # Save envelope spectrogram
    np.savez(
        output_dir / f"{patient_id}_spectrogram_envelope_{envelope_band[0]}_{envelope_band[1]}hz.npz",
        S_env=S_env_avg, T_env=T_env, F_env=F_env,
        envelope_band=envelope_band,
        Fs=Fs,
    )
    print(f"  Saved {patient_id}_spectrogram_envelope_{envelope_band[0]}_{envelope_band[1]}hz.npz")

    # Plot if requested
    if not args.no_plot:
        print("Generating plots ...")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Plot full-signal spectrogram
        fig, ax = plt.subplots(figsize=(12, 6))
        S_db = 10 * np.log10(S_avg.T + 1e-12)  # Convert to dB, transpose for imshow
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
        ax.set_title(f"Channel-Averaged Full-Signal Spectrogram: {patient_id}", fontsize=12)
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
        cbar.set_label("Power (dB)", fontsize=10)
        fig.savefig(output_dir / f"{patient_id}_spectrogram_full.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {patient_id}_spectrogram_full.png")

        # Plot envelope spectrogram
        fig, ax = plt.subplots(figsize=(12, 6))
        S_env_db = 10 * np.log10(S_env_avg.T + 1e-12)
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
            f"Channel-Averaged Envelope Spectrogram ({envelope_band[0]}-{envelope_band[1]} Hz): {patient_id}",
            fontsize=12
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
        cbar.set_label("Envelope Power (dB)", fontsize=10)
        fig.savefig(
            output_dir / f"{patient_id}_spectrogram_envelope_{envelope_band[0]}_{envelope_band[1]}hz.png",
            dpi=150, bbox_inches="tight"
        )
        plt.close(fig)
        print(f"  Saved {patient_id}_spectrogram_envelope_{envelope_band[0]}_{envelope_band[1]}hz.png")

    print("Done.")


if __name__ == "__main__":
    main()