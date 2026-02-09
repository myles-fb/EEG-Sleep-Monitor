"""
Processing worker that reads from ring buffer and computes features.

This module runs as a separate process/thread, continuously reading from
the ring buffer and computing EEG features (bandpower, PSD) for visualization.
"""

import time
import logging
import numpy as np
from typing import Optional, Dict, Any, List
from multiprocessing import Manager

try:
    from .ring_buffer import RingBuffer
    from .metrics import extract_features, EEGFeatures
    from .filters import preprocess_eeg
    from .mos import compute_mos_for_bucket, MOsResult
except ImportError:
    # Fallback for absolute imports
    from processing.ring_buffer import RingBuffer
    from processing.metrics import extract_features, EEGFeatures
    from processing.filters import preprocess_eeg
    from processing.mos import compute_mos_for_bucket, MOsResult


class ProcessingWorker:
    """
    Worker that processes EEG data from ring buffer and computes features.
    
    Runs continuously, reading windows from the ring buffer and computing
    features at a fixed cadence.
    """
    
    def __init__(
        self,
        ring_buffer: RingBuffer,
        shared_state: Dict[str, Any],
        sample_rate: float = 250.0,
        window_size_seconds: float = 2.0,
        update_interval_seconds: float = 1.0,
        channel_index: int = 0,
        apply_notch: bool = False,
        apply_bandpass: bool = False,
        enable_mos: bool = False,
        mo_bucket_seconds: float = 120.0,
        mo_n_surrogates: int = 50,
    ):
        """
        Initialize the processing worker.
        
        Args:
            ring_buffer: Ring buffer to read from
            shared_state: Shared dictionary for storing computed features
            sample_rate: Sampling rate in Hz
            window_size_seconds: Size of window for feature computation
            update_interval_seconds: How often to compute new features
            channel_index: Which channel to process
            apply_notch: Whether to apply 60 Hz notch filter
            apply_bandpass: Whether to apply bandpass filter
            enable_mos: Whether to run Modulatory Oscillations (MOs) detection on time buckets
            mo_bucket_seconds: Length of time bucket for MOs (e.g. 120 or 300 seconds)
            mo_n_surrogates: Number of phase-randomized surrogates for MOs p-values
        """
        self.ring_buffer = ring_buffer
        self.shared_state = shared_state
        self.sample_rate = sample_rate
        self.window_size_seconds = window_size_seconds
        self.update_interval_seconds = update_interval_seconds
        self.channel_index = channel_index
        self.apply_notch = apply_notch
        self.apply_bandpass = apply_bandpass
        self.enable_mos = enable_mos
        self.mo_bucket_seconds = mo_bucket_seconds
        self.mo_n_surrogates = mo_n_surrogates
        self._last_mo_bucket_time: Optional[float] = None
        
        self.is_running = False
        self.logger = logging.getLogger(__name__)
    
    def run(self):
        """Main processing loop."""
        self.is_running = True
        self.logger.info("Processing worker started")
        
        last_update_time = time.time()
        
        try:
            while self.is_running:
                current_time = time.time()
                
                # Check if it's time to update features
                if current_time - last_update_time >= self.update_interval_seconds:
                    # Optionally run MOs on a full time bucket (e.g. 2 or 5 min)
                    if self.enable_mos and self.mo_bucket_seconds > 0:
                        self._maybe_run_mos(current_time)
                    # Get window from ring buffer
                    window, is_valid = self.ring_buffer.get_window_seconds(
                        self.window_size_seconds,
                        channel_indices=[self.channel_index]
                    )
                    
                    if is_valid and window.size > 0:
                        # Extract single channel data
                        # window shape is (num_channels, num_samples)
                        if window.ndim == 2 and window.shape[0] > 0:
                            channel_data = window[0, :]
                        elif window.ndim == 1:
                            channel_data = window
                        else:
                            continue
                        
                        # Ensure it's 1D
                        if channel_data.ndim > 1:
                            channel_data = channel_data.flatten()
                        
                        # Preprocess
                        processed_data = preprocess_eeg(
                            channel_data,
                            self.sample_rate,
                            remove_dc=True,
                            apply_notch=self.apply_notch,
                            apply_bandpass=self.apply_bandpass
                        )
                        
                        # Ensure processed data is 1D
                        if processed_data.ndim > 1:
                            processed_data = processed_data.flatten()
                        
                        # Extract features
                        features = extract_features(
                            processed_data,
                            self.sample_rate,
                            channel_index=self.channel_index,
                            window_size_seconds=self.window_size_seconds,
                            timestamp=current_time
                        )
                        
                        # Store in shared state
                        self._update_shared_state(features)
                    
                    last_update_time = current_time
                
                # Small sleep to avoid busy-waiting
                time.sleep(0.1)
        
        except KeyboardInterrupt:
            self.logger.info("Processing worker interrupted")
        except Exception as e:
            self.logger.error(f"Error in processing worker: {e}", exc_info=True)
        finally:
            self.is_running = False
            self.logger.info("Processing worker stopped")
    
    def _maybe_run_mos(self, current_time: float):
        """If buffer has at least mo_bucket_seconds of data, run MOs and store q-values."""
        bucket_window, is_valid = self.ring_buffer.get_window_seconds(
            self.mo_bucket_seconds,
            channel_indices=None,
        )
        if not is_valid or bucket_window.size == 0:
            return
        # Throttle: don't run MOs again on the same bucket (e.g. run at most once per bucket)
        if self._last_mo_bucket_time is not None:
            if current_time - self._last_mo_bucket_time < self.mo_bucket_seconds * 0.5:
                return
        self._last_mo_bucket_time = current_time
        try:
            result = compute_mos_for_bucket(
                bucket_window,
                self.sample_rate,
                timestamp=current_time,
                bucket_length_seconds=self.mo_bucket_seconds,
                bpm_mask=None,
                n_surrogates=self.mo_n_surrogates,
                channel_index=self.channel_index,
            )
            lock = self.shared_state.get("_lock")
            if lock:
                with lock:
                    self._store_mos_result(result)
            else:
                self._store_mos_result(result)
        except Exception as e:
            self.logger.warning("MOs detection failed: %s", e)

    def _store_mos_result(self, result: MOsResult):
        """Store MOs result in shared state and append to history for finetuning."""
        q_per_window = {
            band: arr.tolist() for band, arr in result.q_per_window_per_band.items()
        }
        p_per_window = {
            band: arr.tolist() for band, arr in result.p_per_window_per_band.items()
        }
        dom_freq_per_window = {
            band: arr.tolist() for band, arr in result.dominant_freq_hz_per_window_per_band.items()
        }
        self.shared_state["mo_result"] = {
            "timestamp": result.timestamp,
            "bucket_length_seconds": result.bucket_length_seconds,
            "q_per_band": result.q_per_band,
            "p_per_band": result.p_per_band,
            "q_per_window_per_band": q_per_window,
            "p_per_window_per_band": p_per_window,
            "dominant_freq_hz_per_window_per_band": dom_freq_per_window,
            "dominant_freq_hz_per_band": result.dominant_freq_hz_per_band.copy(),
            "n_surrogates": result.n_surrogates,
        }
        history = self.shared_state.get("mo_history")
        if history is None:
            history = []
            self.shared_state["mo_history"] = history
        history.append({
            "timestamp": result.timestamp,
            "q_per_band": result.q_per_band.copy(),
            "p_per_band": result.p_per_band.copy(),
            "q_per_window_per_band": {
                b: arr.tolist() for b, arr in result.q_per_window_per_band.items()
            },
            "p_per_window_per_band": {
                b: arr.tolist() for b, arr in result.p_per_window_per_band.items()
            },
            "dominant_freq_hz_per_window_per_band": {
                b: arr.tolist() for b, arr in result.dominant_freq_hz_per_window_per_band.items()
            },
            "dominant_freq_hz_per_band": result.dominant_freq_hz_per_band.copy(),
        })
        # Keep a bounded history (e.g. last 1000 buckets) to avoid unbounded growth
        max_history = 1000
        if len(history) > max_history:
            self.shared_state["mo_history"] = history[-max_history:]

    def _update_shared_state(self, features: EEGFeatures):
        """
        Update shared state with computed features.
        
        Args:
            features: Computed EEG features
        """
        # Thread-safe update using lock if available
        lock = self.shared_state.get('_lock')
        if lock:
            with lock:
                self._do_update_shared_state(features)
        else:
            self._do_update_shared_state(features)
    
    def _do_update_shared_state(self, features: EEGFeatures):
        """Internal method to update shared state (called with lock held)."""
        # Convert features to dictionary for sharing
        state = {
            'timestamp': features.timestamp,
            'bandpower': features.bandpower,
            'relative_bandpower': features.relative_bandpower,
            'psd_freqs': features.psd_freqs.tolist(),
            'psd_power': features.psd_power.tolist(),
            'channel_index': features.channel_index,
            'window_size_seconds': features.window_size_seconds
        }
        if features.mo_q_per_band is not None:
            state['mo_q_per_band'] = features.mo_q_per_band
        if features.mo_p_per_band is not None:
            state['mo_p_per_band'] = features.mo_p_per_band
        self.shared_state['features'] = state
        
        # Also store raw data window for visualization
        window, _ = self.ring_buffer.get_window_seconds(
            self.window_size_seconds,
            channel_indices=[self.channel_index]
        )
        if window.size > 0:
            self.shared_state['raw_data'] = window[0, :].tolist()
            self.shared_state['raw_data_timestamp'] = time.time()
    
    def stop(self):
        """Stop the processing worker."""
        self.is_running = False


def create_shared_state() -> Dict[str, Any]:
    """
    Create a shared state dictionary for threading.
    
    Note: Uses a regular dict with threading.Lock for thread safety
    instead of multiprocessing.Manager, which is better for Streamlit.
    
    Returns:
        Dictionary that can be shared between threads
    """
    import threading
    shared_dict = {
        'features': None,
        'raw_data': None,
        'raw_data_timestamp': None,
        'stream_status': None,
        'is_streaming': False,
        'mo_result': None,   # Latest MOs result (q_per_band, p_per_band) for current bucket
        'mo_history': [],   # List of past MOs results for finetuning
        '_lock': threading.Lock()  # Lock for thread safety
    }
    return shared_dict
