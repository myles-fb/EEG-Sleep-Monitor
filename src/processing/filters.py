"""
Signal filtering utilities for EEG preprocessing.
"""

import numpy as np
from scipy import signal
from typing import Optional


def remove_dc_offset(data: np.ndarray) -> np.ndarray:
    """
    Remove DC offset by subtracting the mean.
    
    Args:
        data: Input signal (1D or 2D array)
    
    Returns:
        Data with DC offset removed
    """
    if data.ndim == 1:
        return data - np.mean(data)
    else:
        # For multi-channel data, remove DC per channel
        return data - np.mean(data, axis=1, keepdims=True)


def detrend(data: np.ndarray) -> np.ndarray:
    """
    Remove linear trend from signal.
    
    Args:
        data: Input signal (1D or 2D array)
    
    Returns:
        Detrended data
    """
    if data.ndim == 1:
        return signal.detrend(data)
    else:
        # Detrend each channel separately
        return np.apply_along_axis(signal.detrend, axis=1, arr=data)


def apply_notch_filter(
    data: np.ndarray,
    sample_rate: float,
    notch_freq: float = 60.0,
    quality_factor: float = 30.0
) -> np.ndarray:
    """
    Apply notch filter to remove line noise (e.g., 60 Hz).
    
    Args:
        data: Input signal (1D or 2D array)
        sample_rate: Sampling rate in Hz
        notch_freq: Frequency to notch out (default: 60 Hz)
        quality_factor: Quality factor for the filter (higher = narrower notch)
    
    Returns:
        Filtered data
    """
    # Design notch filter
    b, a = signal.iirnotch(notch_freq, quality_factor, sample_rate)
    
    if data.ndim == 1:
        return signal.filtfilt(b, a, data)
    else:
        # Apply filter to each channel
        filtered = np.zeros_like(data)
        for i in range(data.shape[0]):
            filtered[i, :] = signal.filtfilt(b, a, data[i, :])
        return filtered


def apply_bandpass_filter(
    data: np.ndarray,
    sample_rate: float,
    low_freq: float = 0.5,
    high_freq: float = 40.0,
    order: int = 4
) -> np.ndarray:
    """
    Apply bandpass filter to EEG data.
    
    Args:
        data: Input signal (1D or 2D array)
        sample_rate: Sampling rate in Hz
        low_freq: Low cutoff frequency in Hz
        high_freq: High cutoff frequency in Hz
        order: Filter order (default: 4)
    
    Returns:
        Filtered data
    """
    # Design Butterworth bandpass filter
    nyquist = sample_rate / 2.0
    low = low_freq / nyquist
    high = high_freq / nyquist
    
    # Ensure frequencies are within valid range
    low = max(0.01, low)
    high = min(0.99, high)
    
    if low >= high:
        # Invalid filter parameters, return original data
        return data
    
    b, a = signal.butter(order, [low, high], btype='band')
    
    if data.ndim == 1:
        return signal.filtfilt(b, a, data)
    else:
        # Apply filter to each channel
        filtered = np.zeros_like(data)
        for i in range(data.shape[0]):
            filtered[i, :] = signal.filtfilt(b, a, data[i, :])
        return filtered


def preprocess_eeg(
    data: np.ndarray,
    sample_rate: float,
    remove_dc: bool = True,
    detrend_signal: bool = False,
    apply_notch: bool = False,
    notch_freq: float = 60.0,
    apply_bandpass: bool = False,
    bandpass_low: float = 0.5,
    bandpass_high: float = 40.0
) -> np.ndarray:
    """
    Apply a sequence of preprocessing steps to EEG data.
    
    Args:
        data: Input signal (1D or 2D array)
        sample_rate: Sampling rate in Hz
        remove_dc: Whether to remove DC offset
        detrend_signal: Whether to detrend the signal
        apply_notch: Whether to apply notch filter
        notch_freq: Notch frequency in Hz
        apply_bandpass: Whether to apply bandpass filter
        bandpass_low: Low cutoff frequency in Hz
        bandpass_high: High cutoff frequency in Hz
    
    Returns:
        Preprocessed data
    """
    processed = data.copy()
    
    if remove_dc:
        processed = remove_dc_offset(processed)
    
    if detrend_signal:
        processed = detrend(processed)
    
    if apply_notch:
        processed = apply_notch_filter(processed, sample_rate, notch_freq)
    
    if apply_bandpass:
        processed = apply_bandpass_filter(
            processed, sample_rate, bandpass_low, bandpass_high
        )
    
    return processed
