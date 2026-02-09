"""
EEG signal processing and feature extraction.

Computes bandpower, PSD, and other metrics from EEG data windows.
Designed for single-channel EEG data.
"""

import numpy as np
from scipy import signal
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


# Frequency bands for EEG analysis
FREQ_BANDS = {
    'delta': (0.5, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'beta': (13.0, 30.0),
    'gamma': (30.0, 100.0)
}


@dataclass
class EEGFeatures:
    """Container for computed EEG features."""
    timestamp: float
    bandpower: Dict[str, float]  # Band name -> power value
    relative_bandpower: Dict[str, float]  # Normalized bandpower
    psd_freqs: np.ndarray  # Frequency axis for PSD
    psd_power: np.ndarray  # Power spectral density
    channel_index: int  # Which channel these features are for
    window_size_seconds: float
    # Modulatory Oscillations (MOs): q-value and p-value per band when MOs is run on a bucket
    mo_q_per_band: Optional[Dict[str, float]] = None
    mo_p_per_band: Optional[Dict[str, float]] = None
    mo_p_per_window_per_band: Optional[Dict[str, np.ndarray]] = None

def compute_bandpower(
    data: np.ndarray,
    sample_rate: float,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
    method: str = 'welch'
) -> Dict[str, float]:
    """
    Compute band-limited power for specified frequency bands.
    
    Args:
        data: 1D array of EEG samples
        sample_rate: Sampling rate in Hz
        bands: Dictionary of band_name -> (low_freq, high_freq)
               If None, uses default FREQ_BANDS
        method: Method for PSD estimation ('welch' or 'fft')
    
    Returns:
        Dictionary mapping band names to power values
    """
    if bands is None:
        bands = FREQ_BANDS
    
    if method == 'welch':
        # Welch's method for PSD estimation
        freqs, psd = signal.welch(
            data,
            fs=sample_rate,
            nperseg=min(len(data), int(2 * sample_rate)),  # 2 second segments
            noverlap=None,
            nfft=None
        )
    else:
        # Simple FFT-based PSD
        fft_vals = np.fft.rfft(data)
        freqs = np.fft.rfftfreq(len(data), 1.0 / sample_rate)
        psd = np.abs(fft_vals) ** 2
    
    # Compute power in each band
    bandpower = {}
    for band_name, (low_freq, high_freq) in bands.items():
        # Find frequency indices
        freq_mask = (freqs >= low_freq) & (freqs <= high_freq)
        if np.any(freq_mask):
            # Integrate PSD over the band (trapezoidal integration)
            band_power = np.trapz(psd[freq_mask], freqs[freq_mask])
            bandpower[band_name] = band_power
        else:
            bandpower[band_name] = 0.0
    
    return bandpower


def compute_relative_bandpower(bandpower: Dict[str, float]) -> Dict[str, float]:
    """
    Compute relative bandpower (normalized by total power).
    
    Args:
        bandpower: Dictionary of absolute bandpower values
    
    Returns:
        Dictionary of relative bandpower (sums to 1.0)
    """
    total_power = sum(bandpower.values())
    
    if total_power > 0:
        return {band: power / total_power for band, power in bandpower.items()}
    else:
        return {band: 0.0 for band in bandpower.keys()}


def compute_psd(
    data: np.ndarray,
    sample_rate: float,
    method: str = 'welch'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute power spectral density.
    
    Args:
        data: 1D array of EEG samples
        sample_rate: Sampling rate in Hz
        method: Method for PSD estimation ('welch' or 'fft')
    
    Returns:
        Tuple of (frequencies, power_spectral_density)
    """
    if method == 'welch':
        freqs, psd = signal.welch(
            data,
            fs=sample_rate,
            nperseg=min(len(data), int(2 * sample_rate)),
            noverlap=None,
            nfft=None
        )
    else:
        fft_vals = np.fft.rfft(data)
        freqs = np.fft.rfftfreq(len(data), 1.0 / sample_rate)
        psd = np.abs(fft_vals) ** 2
    
    return freqs, psd


def extract_features(
    data: np.ndarray,
    sample_rate: float,
    channel_index: int = 0,
    window_size_seconds: float = 2.0,
    timestamp: Optional[float] = None
) -> EEGFeatures:
    """
    Extract all features from a window of EEG data.
    
    Args:
        data: 1D array of EEG samples for a single channel
        sample_rate: Sampling rate in Hz
        channel_index: Index of the channel
        window_size_seconds: Size of the window in seconds
        timestamp: Optional timestamp for the features
    
    Returns:
        EEGFeatures object with all computed features
    """
    if timestamp is None:
        import time
        timestamp = time.time()
    
    # Remove DC offset
    data = data - np.mean(data)
    
    # Compute bandpower
    bandpower = compute_bandpower(data, sample_rate)
    
    # Compute relative bandpower
    relative_bandpower = compute_relative_bandpower(bandpower)
    
    # Compute PSD
    psd_freqs, psd_power = compute_psd(data, sample_rate)
    
    return EEGFeatures(
        timestamp=timestamp,
        bandpower=bandpower,
        relative_bandpower=relative_bandpower,
        psd_freqs=psd_freqs,
        psd_power=psd_power,
        channel_index=channel_index,
        window_size_seconds=window_size_seconds
    )
