"""
Processing worker that reads from ring buffer and computes features.

This module runs as a separate process/thread, continuously reading from
the ring buffer and computing EEG features (bandpower, PSD) for visualization.
"""

import time
import logging
import numpy as np
from typing import Optional, Dict, Any
from multiprocessing import Manager

try:
    from .ring_buffer import RingBuffer
    from .metrics import extract_features, EEGFeatures
    from .filters import preprocess_eeg
except ImportError:
    # Fallback for absolute imports
    from processing.ring_buffer import RingBuffer
    from processing.metrics import extract_features, EEGFeatures
    from processing.filters import preprocess_eeg


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
        apply_bandpass: bool = False
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
        """
        self.ring_buffer = ring_buffer
        self.shared_state = shared_state
        self.sample_rate = sample_rate
        self.window_size_seconds = window_size_seconds
        self.update_interval_seconds = update_interval_seconds
        self.channel_index = channel_index
        self.apply_notch = apply_notch
        self.apply_bandpass = apply_bandpass
        
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
        self.shared_state['features'] = {
            'timestamp': features.timestamp,
            'bandpower': features.bandpower,
            'relative_bandpower': features.relative_bandpower,
            'psd_freqs': features.psd_freqs.tolist(),
            'psd_power': features.psd_power.tolist(),
            'channel_index': features.channel_index,
            'window_size_seconds': features.window_size_seconds
        }
        
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
        '_lock': threading.Lock()  # Lock for thread safety
    }
    return shared_dict
