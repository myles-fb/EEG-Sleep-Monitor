"""
Thread-safe ring buffer for EEG data streaming.

This ring buffer is designed for real-time EEG data acquisition where:
- Writer (acquisition) adds data continuously
- Reader (processing) consumes data in windows
- Data is stored in a circular buffer to avoid memory growth
"""

import numpy as np
from threading import Lock
from typing import Optional, Tuple
from collections import deque


class RingBuffer:
    """
    Thread-safe ring buffer for continuous EEG data streaming.
    
    The buffer stores data in a circular fashion, overwriting oldest data
    when full. Supports concurrent read/write operations.
    """
    
    def __init__(
        self,
        num_channels: int,
        buffer_size_samples: int,
        sample_rate: float = 250.0
    ):
        """
        Initialize the ring buffer.
        
        Args:
            num_channels: Number of EEG channels
            buffer_size_samples: Maximum number of samples to store
            sample_rate: Sampling rate in Hz (for time calculations)
        """
        self.num_channels = num_channels
        self.buffer_size_samples = buffer_size_samples
        self.sample_rate = sample_rate
        
        # Data storage: shape (num_channels, buffer_size_samples)
        self.buffer = np.zeros((num_channels, buffer_size_samples), dtype=np.float64)
        
        # Current write position (circular index)
        self.write_pos = 0
        
        # Total samples written (for absolute indexing)
        self.total_samples_written = 0
        
        # Lock for thread safety
        self.lock = Lock()
        
        # Track if buffer has wrapped around
        self.has_wrapped = False
    
    def write(self, data: np.ndarray):
        """
        Write new data to the ring buffer.
        
        Args:
            data: numpy array of shape (num_channels, num_samples)
                 Should contain ONLY EEG channels (already extracted from BrainFlow data)
                 Data should be pre-filtered to contain only EEG channels using get_eeg_channels()
        """
        if data.size == 0:
            return
        
        with self.lock:
            # Verify data shape matches expected number of channels
            if data.shape[0] != self.num_channels:
                raise ValueError(
                    f"Data has {data.shape[0]} channels but buffer expects {self.num_channels} channels. "
                    "Ensure EEG channels are extracted using get_eeg_channels() before writing."
                )
            
            num_samples = data.shape[1]
            
            if num_samples == 0:
                return
            
            # Write samples one by one (handles wrapping)
            for i in range(num_samples):
                self.buffer[:, self.write_pos] = data[:, i]
                self.write_pos = (self.write_pos + 1) % self.buffer_size_samples
                
                if self.write_pos == 0:
                    self.has_wrapped = True
            
            self.total_samples_written += num_samples
    
    def get_window(
        self,
        window_size_samples: int,
        channel_indices: Optional[list] = None
    ) -> Tuple[np.ndarray, bool]:
        """
        Get a window of recent data.
        
        Args:
            window_size_samples: Number of samples to retrieve
            channel_indices: Optional list of channel indices to return.
                           If None, returns all channels.
        
        Returns:
            Tuple of (data_array, is_valid)
            - data_array: shape (num_channels, window_size_samples)
            - is_valid: True if window contains valid data (enough samples)
        """
        with self.lock:
            # Check if we have enough data
            if not self.has_wrapped and self.total_samples_written < window_size_samples:
                return np.array([]), False
            
            # Determine which channels to return
            if channel_indices is None:
                channels = slice(None)
                num_channels_out = self.num_channels
            else:
                channels = channel_indices
                num_channels_out = len(channel_indices)
            
            # Allocate output array
            window = np.zeros((num_channels_out, window_size_samples), dtype=np.float64)
            
            # Calculate start position (going backwards from write_pos)
            start_pos = (self.write_pos - window_size_samples) % self.buffer_size_samples
            
            # Copy data (handles wrapping)
            if start_pos + window_size_samples <= self.buffer_size_samples:
                # No wrap needed
                window = self.buffer[channels, start_pos:start_pos + window_size_samples].copy()
            else:
                # Handle wrap around
                samples_before_wrap = self.buffer_size_samples - start_pos
                samples_after_wrap = window_size_samples - samples_before_wrap
                
                window[:, :samples_before_wrap] = self.buffer[channels, start_pos:].copy()
                window[:, samples_before_wrap:] = self.buffer[channels, :samples_after_wrap].copy()
            
            return window, True
    
    def get_window_seconds(
        self,
        window_size_seconds: float,
        channel_indices: Optional[list] = None
    ) -> Tuple[np.ndarray, bool]:
        """
        Get a window of recent data specified in seconds.
        
        Args:
            window_size_seconds: Window size in seconds
            channel_indices: Optional list of channel indices to return
        
        Returns:
            Tuple of (data_array, is_valid)
        """
        window_size_samples = int(window_size_seconds * self.sample_rate)
        return self.get_window(window_size_samples, channel_indices)
    
    def get_latest(self, num_samples: int, channel_indices: Optional[list] = None) -> Tuple[np.ndarray, bool]:
        """
        Get the latest N samples (alias for get_window).
        
        Args:
            num_samples: Number of samples to retrieve
            channel_indices: Optional list of channel indices
        
        Returns:
            Tuple of (data_array, is_valid)
        """
        return self.get_window(num_samples, channel_indices)
    
    def get_status(self) -> dict:
        """
        Get buffer status information.
        
        Returns:
            Dictionary with status info
        """
        with self.lock:
            return {
                'num_channels': self.num_channels,
                'buffer_size_samples': self.buffer_size_samples,
                'buffer_size_seconds': self.buffer_size_samples / self.sample_rate,
                'total_samples_written': self.total_samples_written,
                'write_pos': self.write_pos,
                'has_wrapped': self.has_wrapped,
                'samples_available': (
                    self.buffer_size_samples if self.has_wrapped
                    else self.total_samples_written
                ),
                'sample_rate': self.sample_rate
            }
    
    def clear(self):
        """Clear the buffer (reset to empty state)."""
        with self.lock:
            self.buffer.fill(0)
            self.write_pos = 0
            self.total_samples_written = 0
            self.has_wrapped = False
