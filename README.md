# EEG Sleep Monitor

This is a rapid prototype of at a real-time EEG monitoring dashboard for sleep analysis using OpenBCI Cyton boards. This application provides live visualization of EEG signals, bandpower metrics, and power spectral density for sleep monitoring applications.

## Overview

This project implements a complete data pipeline from EEG acquisition to real-time visualization:

```
Cyton Board → BrainFlow → Ring Buffer → Processing → Streamlit Dashboard
```

The system continuously streams EEG data from an OpenBCI Cyton board, processes it in real-time to extract frequency band metrics, and displays the results in an interactive web dashboard.

## Features

### Real-Time Data Acquisition
- **OpenBCI Cyton Support**: Direct integration with Cyton boards via BrainFlow
- **Continuous Streaming**: Stable data acquisition at 250 Hz sample rate
- **Ring Buffer**: Thread-safe circular buffer for continuous data storage
- **Error Handling**: Robust connection management and error recovery

### Signal Processing
- **Preprocessing Filters**:
  - DC offset removal
  - Detrending
  - 60 Hz notch filter (optional)
  - Bandpass filtering (0.5-40 Hz, optional)
- **Feature Extraction**:
  - Bandpower computation (Delta, Theta, Alpha, Beta, Gamma)
  - Relative bandpower metrics
  - Power Spectral Density (PSD) via Welch's method

### Interactive Dashboard
- **Real-Time Visualization**:
  - Live raw EEG trace (last 2 seconds)
  - Bandpower metrics with absolute and relative values
  - Power Spectral Density plot with frequency band markers
- **Status Monitoring**:
  - Connection status
  - Sample rate and count
  - Dropped packet tracking
  - Stream duration
- **Configuration Controls**:
  - Serial port selection
  - Filter toggles (notch, bandpass)
  - Channel selection
  - Buffer size adjustment

## Architecture

The application is organized into three main layers:

### 1. Acquisition Layer (`src/acquisition/`)
- **`brainflow_stream.py`**: Manages Cyton board connection and data streaming
  - Handles serial port communication
  - Extracts EEG channels from BrainFlow data
  - Writes data to ring buffer
  - Optional raw data logging

### 2. Processing Layer (`src/processing/`)
- **`ring_buffer.py`**: Thread-safe circular buffer for continuous data storage
- **`filters.py`**: Signal preprocessing utilities (DC removal, notch, bandpass)
- **`metrics.py`**: Feature extraction (bandpower, PSD computation)
- **`processor.py`**: Background worker that processes data and computes features

### 3. Visualization Layer (`src/app/`)
- **`streamlit_app.py`**: Interactive web dashboard
  - Real-time data visualization
  - User controls and configuration
  - Status monitoring

## Installation

### Prerequisites
- Python 3.9 or higher
- OpenBCI Cyton board with RFduino dongle
- macOS, Linux, or Windows

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/myles-fb/EEG-Sleep-Monitor.git
   cd EEG-Sleep-Monitor
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify your serial port**:
   - **macOS**: `/dev/cu.usbserial-*` (use `cu.*` not `tty.*`)
   - **Linux**: `/dev/ttyUSB*` or `/dev/ttyACM*`
   - **Windows**: `COM3`, `COM4`, etc.

## Usage

### Running the Streamlit Dashboard

1. **Start the dashboard**:
   ```bash
   streamlit run src/app/streamlit_app.py
   ```

2. **Configure in the sidebar**:
   - Enter your serial port (e.g., `/dev/cu.usbserial-DM02583G`)
   - Select channel to visualize (0-7)
   - Enable filters if needed (60 Hz notch, bandpass)
   - Adjust buffer size (10-60 seconds)

3. **Start streaming**:
   - Click "▶️ Start" button
   - Wait for connection (may take 2-3 seconds)
   - View real-time EEG data and metrics

4. **Stop streaming**:
   - Click "⏹️ Stop" button
   - Stream will disconnect cleanly

## Future Enhancements

- [ ] Multi-channel visualization
- [ ] Real-time spectrogram (waterfall plot)
- [ ] Event detection and alerts
- [ ] Data replay mode
- [ ] Multi-patient support
- [ ] Database integration
- [ ] CAP (Cyclic Alternating Pattern) detection
- [ ] Modulatory oscillation analysis

## Documentation

- [Data Pipeline Architecture](documentation/data_pipeline.md)
- [Project Plan](documentation/plan.md)
- [Git Workflow](documentation/sprints.md)

## License

Apache-2.0 License

## Acknowledgments

- [BrainFlow](https://github.com/brainflow-dev/brainflow) for EEG board integration
- [OpenBCI](https://openbci.com/) for the Cyton board
- [Streamlit](https://streamlit.io/) for the dashboard framework

## contact

For questions or support, please open an issue on the GitHub repository.

---

**Note**: This is an MVP (Minimum Viable Product) for local, single-patient EEG monitoring. Future versions may include multi-patient support, cloud integration, and advanced sleep analysis features.
