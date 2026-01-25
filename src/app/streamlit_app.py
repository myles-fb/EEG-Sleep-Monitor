"""
Streamlit dashboard for real-time EEG monitoring.

Displays live EEG data, bandpower metrics, and PSD visualization.
"""

import streamlit as st
import numpy as np
import time
from pathlib import Path
import sys
import threading

# Try to import plotly, fallback to matplotlib if not available
try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from processing.ring_buffer import RingBuffer
from processing.processor import ProcessingWorker, create_shared_state
from processing.metrics import FREQ_BANDS
from acquisition.brainflow_stream import CytonStream, DEFAULT_BOARD_ID, DEFAULT_SAMPLE_RATE


# Page configuration
st.set_page_config(
    page_title="EEG Monitoring Dashboard",
    page_icon="🧠",
    layout="wide"
)

# Initialize session state
if 'shared_state' not in st.session_state:
    st.session_state.shared_state = None
if 'ring_buffer' not in st.session_state:
    st.session_state.ring_buffer = None
if 'stream' not in st.session_state:
    st.session_state.stream = None
if 'processing_worker' not in st.session_state:
    st.session_state.processing_worker = None
if 'is_streaming' not in st.session_state:
    st.session_state.is_streaming = False
if 'stream_thread' not in st.session_state:
    st.session_state.stream_thread = None
if 'is_streaming_flag' not in st.session_state:
    st.session_state.is_streaming_flag = None


def initialize_streaming(serial_port: str, buffer_size_seconds: float = 30.0):
    """Initialize the streaming setup."""
    try:
        # Create shared state
        if st.session_state.shared_state is None:
            st.session_state.shared_state = create_shared_state()
        
        # Create ring buffer (30 seconds of data at 250 Hz)
        buffer_size_samples = int(buffer_size_seconds * DEFAULT_SAMPLE_RATE)
        if st.session_state.ring_buffer is None:
            st.session_state.ring_buffer = RingBuffer(
                num_channels=8,  # Cyton has 8 channels
                buffer_size_samples=buffer_size_samples,
                sample_rate=DEFAULT_SAMPLE_RATE
            )
        
        # Create stream instance
        if st.session_state.stream is None:
            st.session_state.stream = CytonStream(
                serial_port=serial_port,
                board_id=DEFAULT_BOARD_ID,
                ring_buffer=st.session_state.ring_buffer
            )
        
        return True
    except Exception as e:
        st.error(f"Failed to initialize streaming: {e}")
        return False


def stream_loop(stream, ring_buffer, shared_state, is_streaming_flag):
    """
    Background thread that continuously reads data from the stream.
    
    Note: Cannot access st.session_state from background threads.
    Pass objects directly instead.
    """
    try:
        while is_streaming_flag[0] and stream.is_streaming:
            # Get data chunk
            data = stream.get_data_chunk()
            
            if data is not None and data.size > 0:
                # Write to ring buffer
                stream.write_to_buffer(data)
            
            # Update stream status in shared state (thread-safe)
            if shared_state:
                status = stream.get_status()
                lock = shared_state.get('_lock')
                if lock:
                    with lock:
                        shared_state['stream_status'] = status
                else:
                    shared_state['stream_status'] = status
            
            # Sleep for chunk size
            time.sleep(stream.chunk_size_ms / 1000.0)
    
    except Exception as e:
        # Log error but can't use st.error from thread
        print(f"Error in stream loop: {e}")
        import traceback
        traceback.print_exc()


def start_streaming():
    """Start the EEG stream."""
    if st.session_state.stream is None:
        st.error("Stream not initialized")
        return False
    
    try:
        # Connect to board
        if not st.session_state.stream.connect():
            st.error("Failed to connect to board")
            return False
        
        # Start streaming
        if not st.session_state.stream.start_stream():
            st.error("Failed to start stream")
            return False
        
        # Start processing worker in a thread
        if st.session_state.processing_worker is None:
            st.session_state.processing_worker = ProcessingWorker(
                ring_buffer=st.session_state.ring_buffer,
                shared_state=st.session_state.shared_state,
                sample_rate=DEFAULT_SAMPLE_RATE,
                window_size_seconds=2.0,
                update_interval_seconds=1.0,
                channel_index=st.session_state.get('channel_index', 0),
                apply_notch=st.session_state.get('apply_notch', False),
                apply_bandpass=st.session_state.get('apply_bandpass', False)
            )
            
            worker_thread = threading.Thread(target=st.session_state.processing_worker.run, daemon=True)
            worker_thread.start()
        
        # Start stream loop in background thread
        # Use a list to pass mutable flag (thread-safe)
        is_streaming_flag = [True]
        stream_thread = threading.Thread(
            target=stream_loop,
            args=(
                st.session_state.stream,
                st.session_state.ring_buffer,
                st.session_state.shared_state,
                is_streaming_flag
            ),
            daemon=True
        )
        stream_thread.start()
        st.session_state['stream_thread'] = stream_thread
        st.session_state['is_streaming_flag'] = is_streaming_flag
        
        st.session_state.is_streaming = True
        st.session_state.shared_state['is_streaming'] = True
        return True
    
    except Exception as e:
        st.error(f"Error starting stream: {e}")
        return False


def stop_streaming():
    """Stop the EEG stream."""
    try:
        # Stop processing worker
        if st.session_state.processing_worker:
            st.session_state.processing_worker.stop()
            st.session_state.processing_worker = None
        
        # Stop stream
        if st.session_state.stream:
            st.session_state.stream.stop_stream()
            st.session_state.stream.disconnect()
            st.session_state.stream = None
        
        # Update flags
        st.session_state.is_streaming = False
        if 'is_streaming_flag' in st.session_state:
            st.session_state.is_streaming_flag[0] = False
        
        if st.session_state.shared_state:
            lock = st.session_state.shared_state.get('_lock')
            if lock:
                with lock:
                    st.session_state.shared_state['is_streaming'] = False
            else:
                st.session_state.shared_state['is_streaming'] = False
    
    except Exception as e:
        st.error(f"Error stopping stream: {e}")


# Main UI
st.title("🧠 Real-Time EEG Monitoring Dashboard")

# Sidebar controls
with st.sidebar:
    st.header("Configuration")
    
    serial_port = st.text_input(
        "Serial Port",
        value="/dev/cu.usbserial-DM02583G",
        help="Serial port for Cyton board (e.g., /dev/cu.usbserial-DM02583G on macOS)"
    )
    
    buffer_size_seconds = st.slider(
        "Buffer Size (seconds)",
        min_value=10,
        max_value=60,
        value=30,
        help="Size of the ring buffer in seconds"
    )
    
    st.session_state['apply_notch'] = st.checkbox("Apply 60 Hz Notch Filter", value=False)
    st.session_state['apply_bandpass'] = st.checkbox("Apply Bandpass Filter (0.5-40 Hz)", value=False)
    
    channel_index = st.selectbox(
        "Channel to Display",
        options=list(range(8)),
        index=0,
        help="Which EEG channel to visualize"
    )
    st.session_state['channel_index'] = channel_index
    
    st.divider()
    
    # Stream control buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("▶️ Start", disabled=st.session_state.is_streaming, use_container_width=True):
            if initialize_streaming(serial_port, buffer_size_seconds):
                if start_streaming():
                    st.success("Streaming started!")
                    st.rerun()
    
    with col2:
        if st.button("⏹️ Stop", disabled=not st.session_state.is_streaming, use_container_width=True):
            stop_streaming()
            st.success("Streaming stopped!")
            st.rerun()

# Main dashboard
if st.session_state.is_streaming and st.session_state.shared_state:
    # Status panel
    with st.container():
        st.subheader("📊 Status")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Status", "🟢 Streaming")
        
        with col2:
            if st.session_state.stream:
                status = st.session_state.stream.get_status()
                st.metric("Sample Rate", f"{status['sample_rate']} Hz")
        
        with col3:
            if st.session_state.stream:
                status = st.session_state.stream.get_status()
                st.metric("Samples", f"{status['sample_count']:,}")
        
        with col4:
            if st.session_state.stream:
                status = st.session_state.stream.get_status()
                st.metric("Dropped Packets", status['dropped_packets'])
    
    st.divider()
    
    # Get latest features from shared state
    features_data = st.session_state.shared_state.get('features')
    raw_data = st.session_state.shared_state.get('raw_data')
    
    if features_data and raw_data:
        # Raw EEG trace
        st.subheader("📈 Raw EEG Trace")
        
        # Convert to numpy array
        raw_array = np.array(raw_data)
        time_axis = np.arange(len(raw_array)) / DEFAULT_SAMPLE_RATE
        
        if HAS_PLOTLY:
            fig_raw = go.Figure()
            fig_raw.add_trace(go.Scatter(
                x=time_axis,
                y=raw_array,
                mode='lines',
                name=f'Channel {channel_index}',
                line=dict(color='blue', width=1)
            ))
            fig_raw.update_layout(
                title=f"Raw EEG - Channel {channel_index} (Last {len(raw_array)/DEFAULT_SAMPLE_RATE:.1f}s)",
                xaxis_title="Time (seconds)",
                yaxis_title="Amplitude (μV)",
                height=300,
                showlegend=False
            )
            st.plotly_chart(fig_raw, use_container_width=True)
        else:
            # Fallback to matplotlib
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(time_axis, raw_array, 'b-', linewidth=1)
            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("Amplitude (μV)")
            ax.set_title(f"Raw EEG - Channel {channel_index} (Last {len(raw_array)/DEFAULT_SAMPLE_RATE:.1f}s)")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)
        
        # Bandpower tiles
        st.subheader("⚡ Bandpower Metrics")
        
        bandpower = features_data.get('bandpower', {})
        relative_bandpower = features_data.get('relative_bandpower', {})
        
        cols = st.columns(4)
        band_names = ['delta', 'theta', 'alpha', 'beta']
        band_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, (band, color) in enumerate(zip(band_names, band_colors)):
            with cols[i]:
                abs_power = bandpower.get(band, 0.0)
                rel_power = relative_bandpower.get(band, 0.0)
                
                st.metric(
                    label=band.upper(),
                    value=f"{abs_power:.2f}",
                    delta=f"{rel_power*100:.1f}%"
                )
        
        # PSD visualization
        st.subheader("📊 Power Spectral Density")
        
        psd_freqs = np.array(features_data.get('psd_freqs', []))
        psd_power = np.array(features_data.get('psd_power', []))
        
        if len(psd_freqs) > 0 and len(psd_power) > 0:
            if HAS_PLOTLY:
                fig_psd = go.Figure()
                fig_psd.add_trace(go.Scatter(
                    x=psd_freqs,
                    y=10 * np.log10(psd_power + 1e-10),  # Convert to dB
                    mode='lines',
                    name='PSD',
                    line=dict(color='purple', width=2),
                    fill='tozeroy'
                ))
                
                # Add frequency band markers
                for band_name, (low, high) in FREQ_BANDS.items():
                    if high <= psd_freqs[-1]:
                        fig_psd.add_vrect(
                            x0=low, x1=high,
                            fillcolor=band_colors[band_names.index(band_name) if band_name in band_names else 0],
                            opacity=0.1,
                            layer="below",
                            line_width=0,
                        )
                
                fig_psd.update_layout(
                    title="Power Spectral Density",
                    xaxis_title="Frequency (Hz)",
                    yaxis_title="Power (dB)",
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(fig_psd, use_container_width=True)
            else:
                # Fallback to matplotlib
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(psd_freqs, 10 * np.log10(psd_power + 1e-10), 'purple', linewidth=2)
                ax.fill_between(psd_freqs, 10 * np.log10(psd_power + 1e-10), alpha=0.3)
                ax.set_xlabel("Frequency (Hz)")
                ax.set_ylabel("Power (dB)")
                ax.set_title("Power Spectral Density")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close(fig)
    
    else:
        st.info("Waiting for data...")
    
    # Auto-refresh
    time.sleep(0.5)
    st.rerun()

else:
    st.info("👈 Configure settings and click 'Start' to begin streaming")
    
    # Show instructions
    with st.expander("📖 Instructions"):
        st.markdown("""
        1. **Connect your Cyton board** to the computer via RFduino dongle
        2. **Enter the serial port** in the sidebar (e.g., `/dev/cu.usbserial-DM02583G` on macOS)
        3. **Configure filters** if needed (60 Hz notch, bandpass)
        4. **Click 'Start'** to begin streaming
        5. **Monitor** the real-time EEG data and metrics
        
        The dashboard will automatically update with:
        - Raw EEG trace (last 2 seconds)
        - Bandpower metrics (delta, theta, alpha, beta)
        - Power spectral density visualization
        """)
