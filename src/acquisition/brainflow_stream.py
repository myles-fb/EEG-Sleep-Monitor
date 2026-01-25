"""
BrainFlow stream for Cyton board with OpenBCI RFduino dongle.

This module handles the acquisition of EEG data from a Cyton board using BrainFlow.
Data is streamed to a shared ring buffer and can be consumed by the processing layer.
The stream can be configured via command-line arguments or defaults for Cyton board.

Usage:
    python brainflow_stream.py --serial-port /dev/ttyUSB0
    python brainflow_stream.py --serial-port COM3  # Windows
    python brainflow_stream.py --serial-port /dev/ttyUSB0 --log-dir ./logs
"""

import argparse
import time
import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

import numpy as np
import platform
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets

# Default Cyton board configuration
DEFAULT_BOARD_ID = BoardIds.CYTON_BOARD.value  # 0
DEFAULT_SAMPLE_RATE = 250  # Hz
DEFAULT_NUM_CHANNELS = 8  # EEG channels
DEFAULT_CHUNK_SIZE_MS = 100  # milliseconds between data pulls (polling interval)
# Note: This controls how often we poll for data, not the number of samples.
# At 250 Hz, 100ms ≈ 25 samples, but we pull ALL available samples each poll.


class CytonStream:
    """
    Manages streaming from a Cyton board via BrainFlow.
    
    Handles connection, data acquisition, and writing to a ring buffer.
    Provides status information and optional logging.
    """
    
    def __init__(
        self,
        serial_port: str,
        board_id: int = DEFAULT_BOARD_ID,
        ring_buffer: Optional[object] = None,
        log_dir: Optional[Path] = None,
        chunk_size_ms: int = DEFAULT_CHUNK_SIZE_MS
    ):
        """
        Initialize the Cyton stream.
        
        Args:
            serial_port: Serial port path (e.g., '/dev/ttyUSB0' or 'COM3')
            board_id: BrainFlow board ID (default: CYTON_BOARD = 0)
            ring_buffer: Optional ring buffer object to write data to
            log_dir: Optional directory to log raw data
            chunk_size_ms: Time between data pulls in milliseconds
        """
        self.serial_port = serial_port
        self.board_id = board_id
        self.ring_buffer = ring_buffer
        self.log_dir = log_dir
        self.chunk_size_ms = chunk_size_ms
        
        self.board: Optional[BoardShim] = None
        self.is_streaming = False
        self.sample_count = 0
        self.dropped_packets = 0
        self.last_timestamp = None
        self.start_time = None
        self.eeg_channel_indices: Optional[list] = None  # EEG channel row indices from BrainFlow
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Setup file logging if requested
        self.log_file = None
        if self.log_dir:
            self.log_dir = Path(log_dir)
            self.log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = self.log_dir / f"eeg_raw_{timestamp}.csv"
            self.logger.info(f"Raw data logging enabled: {self.log_file}")
    
    def _normalize_port_name(self, port: str) -> str:
        """
        Normalize serial port name for the current OS.
        
        On macOS, converts /dev/tty.* to /dev/cu.* for serial communication.
        The cu.* ports don't require carrier signal and are preferred for serial devices.
        
        Args:
            port: Original port name
            
        Returns:
            Normalized port name
        """
        if platform.system() == 'Darwin' and port.startswith('/dev/tty.'):
            # Convert tty.* to cu.* on macOS
            cu_port = port.replace('/dev/tty.', '/dev/cu.')
            self.logger.info(f"macOS detected: converting {port} to {cu_port}")
            return cu_port
        return port
    
    def connect(self) -> bool:
        """
        Connect to the Cyton board.
        
        Returns:
            True if connection successful, False otherwise
        """
        # Normalize port name for current OS
        normalized_port = self._normalize_port_name(self.serial_port)
        
        if normalized_port != self.serial_port:
            self.logger.info(f"Using normalized port: {normalized_port} (original: {self.serial_port})")
            self.serial_port = normalized_port
        
        try:
            # Configure BrainFlow input parameters
            params = BrainFlowInputParams()
            params.serial_port = self.serial_port
            
            self.logger.info(f"Attempting to connect to {self.serial_port}...")
            
            # Create board instance
            self.board = BoardShim(self.board_id, params)
            
            # Prepare session (connects to board)
            self.board.prepare_session()
            
            # Get board info (DEFAULT_PRESET)
            sample_rate = BoardShim.get_sampling_rate(self.board_id, preset=BrainFlowPresets.DEFAULT_PRESET)
            eeg_channels = BoardShim.get_eeg_channels(self.board_id, preset=BrainFlowPresets.DEFAULT_PRESET)
            num_channels = len(eeg_channels)
            
            # Store EEG channel indices for proper data extraction
            self.eeg_channel_indices = eeg_channels
            
            self.logger.info(f"Connected to Cyton board on {self.serial_port}")
            self.logger.info(f"Sample rate: {sample_rate} Hz")
            self.logger.info(f"EEG channels: {num_channels}")
            
            return True
            
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Failed to connect to board: {e}")
            
            # Provide helpful error messages
            if "UNABLE_TO_OPEN_PORT" in error_msg or "unable to prepare" in error_msg.lower():
                self.logger.error("\nTroubleshooting tips:")
                self.logger.error("1. On macOS, try using /dev/cu.* instead of /dev/tty.*")
                self.logger.error("   Example: /dev/cu.usbserial-DM02583G")
                self.logger.error("2. Check if the port exists: ls -la /dev/cu.*")
                self.logger.error("3. Verify permissions: ls -l /dev/cu.usbserial-DM02583G")
                self.logger.error("4. Try running with sudo (if permissions issue)")
                self.logger.error("5. Ensure the Cyton board is powered on and RFduino is connected")
                self.logger.error("6. Check if another process is using the port:")
                self.logger.error("   lsof | grep usbserial")
            
            return False
    
    def disconnect(self):
        """Disconnect from the board and clean up."""
        if self.board is not None:
            try:
                if self.is_streaming:
                    self.stop_stream()
                self.board.release_session()
                self.logger.info("Disconnected from board")
            except Exception as e:
                self.logger.error(f"Error during disconnect: {e}")
            finally:
                self.board = None
        
        if self.log_file:
            #TODO: Shouldn't happen if failed to connect to board
            self.logger.info(f"Logging complete. Data saved to {self.log_file}")
    
    def start_stream(self) -> bool:
        """
        Start streaming data from the board.
        
        Returns:
            True if streaming started successfully, False otherwise
        """
        if self.board is None:
            self.logger.error("Board not connected. Call connect() first.")
            return False
        
        if self.is_streaming:
            self.logger.warning("Stream already running")
            return True
        
        try:
            # Start streaming
            self.board.start_stream()
            self.is_streaming = True
            self.start_time = time.time()
            self.sample_count = 0
            self.dropped_packets = 0
            
            self.logger.info("Streaming started")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start stream: {e}")
            self.is_streaming = False
            return False
    
    def stop_stream(self):
        """Stop streaming data from the board."""
        if self.board is None or not self.is_streaming:
            return
        
        try:
            self.board.stop_stream()
            self.is_streaming = False
            
            duration = time.time() - self.start_time if self.start_time else 0
            self.logger.info(f"Streaming stopped. Duration: {duration:.1f}s, Samples: {self.sample_count}")
            
        except Exception as e:
            self.logger.error(f"Error stopping stream: {e}")
    
    def get_data_chunk(self) -> Optional[np.ndarray]:
        """
        Get a chunk of data from the board.
        
        Note: This pulls ALL available samples from BrainFlow's internal buffer,
        not a fixed number. The number of samples depends on:
        - Sample rate (250 Hz for Cyton)
        - Time since last poll (controlled by chunk_size_ms)
        - Processing speed
        
        At 250 Hz with 100ms polling: expect ~25 samples per chunk on average.
        
        Returns:
            numpy array of shape (num_rows, num_samples) or None if error
        """
        if not self.is_streaming or self.board is None:
            return None
        
        try:
            # Get number of samples available in BrainFlow's buffer (DEFAULT_PRESET)
            num_samples = self.board.get_board_data_count(preset=BrainFlowPresets.DEFAULT_PRESET)
            
            if num_samples == 0:
                return None
            
            # Pull ALL available samples from DEFAULT_PRESET (not a fixed chunk size)
            # This ensures we don't fall behind the stream
            # Returns 2D array [num_rows x num_samples] where rows are channels
            data = self.board.get_board_data(num_samples, preset=BrainFlowPresets.DEFAULT_PRESET)
            
            if data.size == 0:
                return None
            
            # Update statistics
            self.sample_count += num_samples
            
            # Get timestamp channel (DEFAULT_PRESET)
            timestamp_channel = BoardShim.get_timestamp_channel(self.board_id, preset=BrainFlowPresets.DEFAULT_PRESET)
            if timestamp_channel >= 0 and timestamp_channel < data.shape[0]:
                self.last_timestamp = data[timestamp_channel, -1]
            
            # Check for dropped packets (simplified - could be improved)
            # This is a basic check; BrainFlow may provide better indicators
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error getting data chunk: {e}")
            self.dropped_packets += 1
            return None
    
    def write_to_buffer(self, data: np.ndarray):
        """
        Write data to the ring buffer if available.
        
        Args:
            data: numpy array of shape (num_rows, num_samples) from BrainFlow
                 Contains all channels (EEG, timestamp, etc.) in DEFAULT_PRESET format
        """
        if self.ring_buffer is not None:
            try:
                # Extract only EEG channels using the correct channel indices
                if self.eeg_channel_indices is not None:
                    # BrainFlow data format: rows are channels, columns are samples
                    # Extract only the EEG channel rows
                    eeg_data = data[self.eeg_channel_indices, :]
                    # Write EEG data to ring buffer
                    self.ring_buffer.write(eeg_data)
                else:
                    # Fallback: if EEG channels not yet determined, use first num_channels
                    # This should not happen if connect() was called first
                    self.logger.warning("EEG channel indices not set, using first N channels")
                    num_channels = self.ring_buffer.num_channels if hasattr(self.ring_buffer, 'num_channels') else 8
                    eeg_data = data[:num_channels, :]
                    self.ring_buffer.write(eeg_data)
            except Exception as e:
                self.logger.error(f"Error writing to buffer: {e}")
    
    def log_to_file(self, data: np.ndarray):
        """
        Log raw EEG data to CSV file (only EEG channels from DEFAULT_PRESET).
        
        Args:
            data: numpy array of shape (num_rows, num_samples) from BrainFlow
                 Contains all channels, but we extract only EEG channels
        """
        if self.log_file is None:
            return
        
        try:
            # Extract only EEG channels for logging
            if self.eeg_channel_indices is not None:
                eeg_data = data[self.eeg_channel_indices, :]
            else:
                # Fallback if EEG channels not determined
                num_channels = len(BoardShim.get_eeg_channels(self.board_id, preset=BrainFlowPresets.DEFAULT_PRESET))
                eeg_data = data[:num_channels, :]
            
            # Append data to CSV file
            # Format: each column is a channel, each row is a time point
            # Transpose data so time is rows and channels are columns
            with open(self.log_file, 'ab') as f:
                np.savetxt(f, eeg_data.T, delimiter=',', fmt='%.6f')
        except Exception as e:
            self.logger.error(f"Error logging to file: {e}")
    
    def get_status(self) -> dict:
        """
        Get current stream status.
        
        Returns:
            Dictionary with status information
        """
        status = {
            'connected': self.board is not None,
            'streaming': self.is_streaming,
            'sample_rate': BoardShim.get_sampling_rate(self.board_id, preset=BrainFlowPresets.DEFAULT_PRESET) if self.board else 0,
            'sample_count': self.sample_count,
            'dropped_packets': self.dropped_packets,
            'last_timestamp': self.last_timestamp,
            'serial_port': self.serial_port,
            'board_id': self.board_id,
            'eeg_channels': self.eeg_channel_indices if self.eeg_channel_indices else []
        }
        
        if self.start_time:
            status['duration'] = time.time() - self.start_time
        
        return status
    
    def run_stream_loop(self):
        """
        Main streaming loop. Continuously pulls data and writes to buffer/logs.
        Run this in a separate thread or process.
        """
        if not self.is_streaming:
            self.logger.error("Stream not started. Call start_stream() first.")
            return
        
        self.logger.info("Starting stream loop...")
        
        try:
            while self.is_streaming:
                # Get data chunk
                data = self.get_data_chunk()
                
                if data is not None and data.size > 0:
                    # Write to ring buffer
                    self.write_to_buffer(data)
                    
                    # Log to file if enabled
                    if self.log_file:
                        self.log_to_file(data)
                
                # Sleep for chunk size
                time.sleep(self.chunk_size_ms / 1000.0)
                
        except KeyboardInterrupt:
            self.logger.info("Stream loop interrupted by user")
        except Exception as e:
            self.logger.error(f"Error in stream loop: {e}")
        finally:
            self.stop_stream()


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Stream EEG data from Cyton board via BrainFlow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Connect to Cyton on Linux/macOS
  python brainflow_stream.py --serial-port /dev/ttyUSB0
  
  # Connect on Windows
  python brainflow_stream.py --serial-port COM3
  
  # Enable logging
  python brainflow_stream.py --serial-port /dev/ttyUSB0 --log-dir ./logs
  
  # Custom chunk size
  python brainflow_stream.py --serial-port /dev/ttyUSB0 --chunk-size-ms 200
        """
    )
    
    parser.add_argument(
        '--serial-port',
        type=str,
        default='/dev/cu.usbserial-DM02583G',
        help='Serial port path (e.g., /dev/cu.usbserial-DM02583G on macOS, /dev/ttyUSB0 on Linux, COM3 on Windows)'
    )
    
    parser.add_argument(
        '--board-id',
        type=int,
        default=DEFAULT_BOARD_ID,
        help=f'BrainFlow board ID (default: {DEFAULT_BOARD_ID} for Cyton)'
    )
    
    parser.add_argument(
        '--log-dir',
        type=str,
        default=None,
        help='Directory to log raw EEG data (optional)'
    )
    
    parser.add_argument(
        '--chunk-size-ms',
        type=int,
        default=DEFAULT_CHUNK_SIZE_MS,
        help=f'Time between data pulls in milliseconds (default: {DEFAULT_CHUNK_SIZE_MS})'
    )
    
    parser.add_argument(
        '--duration',
        type=float,
        default=None,
        help='Stream duration in seconds (default: run until interrupted)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger = logging.getLogger(__name__)
    
    # Create stream instance
    stream = CytonStream(
        serial_port=args.serial_port,
        board_id=args.board_id,
        log_dir=args.log_dir,
        chunk_size_ms=args.chunk_size_ms
    )
    
    try:
        # Connect to board
        logger.info(f"Connecting to Cyton board on {args.serial_port}...")
        if not stream.connect():
            logger.error("Failed to connect to board")
            sys.exit(1)
        
        # Start streaming
        logger.info("Starting stream...")
        if not stream.start_stream():
            logger.error("Failed to start stream")
            sys.exit(1)
        
        # Print status periodically
        def print_status():
            status = stream.get_status()
            logger.info(
                f"Status - Samples: {status['sample_count']}, "
                f"Dropped: {status['dropped_packets']}, "
                f"Duration: {status.get('duration', 0):.1f}s"
            )
        
        # Run stream loop
        start_time = time.time()
        last_status_time = start_time
        
        logger.info("Streaming started. Press Ctrl+C to stop.")
        
        while stream.is_streaming:
            # Get data chunk
            data = stream.get_data_chunk()
            
            if data is not None and data.size > 0:
                # Write to buffer (if ring buffer was provided)
                stream.write_to_buffer(data)
                
                # Log to file if enabled
                if stream.log_file:
                    stream.log_to_file(data)
            
            # Print status every 5 seconds
            current_time = time.time()
            if current_time - last_status_time >= 5.0:
                print_status()
                last_status_time = current_time
            
            # Check duration limit
            if args.duration and (current_time - start_time) >= args.duration:
                logger.info(f"Duration limit ({args.duration}s) reached")
                break
            
            # Sleep to control polling frequency
            # This prevents busy-waiting and allows samples to accumulate.
            # At 250 Hz, sleeping 100ms allows ~25 samples to accumulate.
            # Note: We pull ALL available samples, so chunk size is variable.
            time.sleep(args.chunk_size_ms / 1000.0)
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        # Cleanup
        stream.disconnect()
        logger.info("Streaming complete")


if __name__ == '__main__':
    main()
