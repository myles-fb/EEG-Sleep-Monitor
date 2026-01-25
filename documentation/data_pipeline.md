# EEG Data Pipeline Architecture

This document describes the complete data pipeline from Cyton board to Streamlit dashboard.

## Architecture Overview

```
Cyton Board → BrainFlow → Ring Buffer → Processing Worker → Shared State → Streamlit Dashboard
```

## Components

### 1. Acquisition Layer (`src/acquisition/brainflow_stream.py`)

- **CytonStream**: Connects to Cyton board via BrainFlow
- Continuously pulls data from the board
- Writes data to ring buffer
- Handles connection, streaming, and error recovery

### 2. Ring Buffer (`src/processing/ring_buffer.py`)

- **RingBuffer**: Thread-safe circular buffer for EEG data
- Stores continuous stream of data (configurable size, default 30 seconds)
- Supports concurrent read/write operations
- Provides windowed data access for processing

### 3. Processing Layer

#### Filters (`src/processing/filters.py`)
- DC offset removal
- Detrending
- 60 Hz notch filter
- Bandpass filtering

#### Metrics (`src/processing/metrics.py`)
- Bandpower computation (delta, theta, alpha, beta, gamma)
- Relative bandpower
- Power Spectral Density (PSD) via Welch's method
- Feature extraction pipeline

#### Processor (`src/processing/processor.py`)
- **ProcessingWorker**: Background worker that:
  - Reads windows from ring buffer
  - Applies preprocessing filters
  - Computes features (bandpower, PSD)
  - Updates shared state for visualization

### 4. Visualization Layer (`src/app/streamlit_app.py`)

- Streamlit dashboard with:
  - Real-time raw EEG trace
  - Bandpower metrics (delta, theta, alpha, beta)
  - Power Spectral Density visualization
  - Stream status indicators
  - Configuration controls

## Data Flow

1. **Acquisition**: `CytonStream` pulls data from board every 100ms
2. **Buffering**: Data is written to `RingBuffer` continuously
3. **Processing**: `ProcessingWorker` reads 2-second windows every 1 second
4. **Feature Extraction**: Computes bandpower and PSD
5. **Sharing**: Features stored in shared state (multiprocessing.Manager)
6. **Visualization**: Streamlit reads from shared state and displays

## Usage

### Start the Streamlit Dashboard

```bash
streamlit run src/app/streamlit_app.py
```

### Configuration

1. **Serial Port**: Enter your Cyton board's serial port (e.g., `/dev/cu.usbserial-DM02583G`)
2. **Filters**: Enable/disable 60 Hz notch and bandpass filters
3. **Channel**: Select which EEG channel to visualize (0-7)
4. **Buffer Size**: Configure ring buffer size (10-60 seconds)

### Starting Streaming

1. Click "Start" button in sidebar
2. The system will:
   - Connect to Cyton board
   - Start data acquisition
   - Begin processing features
   - Display real-time visualizations

### Features Displayed

- **Raw EEG Trace**: Last 2 seconds of raw signal
- **Bandpower Tiles**: Absolute and relative power for each frequency band
- **PSD Plot**: Power spectral density with frequency band markers
- **Status Panel**: Connection status, sample rate, sample count, dropped packets

## Technical Details

### Thread Safety

- Ring buffer uses `threading.Lock` for thread-safe operations
- Shared state uses `multiprocessing.Manager` for process-safe sharing
- Multiple threads: acquisition, processing, visualization

### Performance Considerations

- Ring buffer size: 30 seconds at 250 Hz = 7,500 samples per channel
- Processing window: 2 seconds (500 samples)
- Update rate: Features computed every 1 second
- UI refresh: Streamlit auto-refreshes every 0.5 seconds

### Memory Usage

- Ring buffer: ~8 channels × 7,500 samples × 8 bytes ≈ 480 KB
- Shared state: Minimal (only latest features)
- Total: < 1 MB for data structures

## Future Enhancements

- Multi-channel visualization
- Event detection and alerts
- Data logging to disk
- Replay mode for recorded sessions
- Real-time spectrogram (waterfall plot)
