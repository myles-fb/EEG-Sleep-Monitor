# 🧠 EEG Sleep Monitor

A real-time EEG monitoring dashboard for sleep analysis using OpenBCI Cyton boards. This application provides live visualization of EEG signals, bandpower metrics, and power spectral density for sleep monitoring applications.

## 📋 Overview

This project implements a complete data pipeline from EEG acquisition to real-time visualization:

```
Cyton Board → BrainFlow → Ring Buffer → Processing → Streamlit Dashboard
```

The system continuously streams EEG data from an OpenBCI Cyton board, processes it in real-time to extract frequency band metrics, and displays the results in an interactive web dashboard.

## ✨ Features

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

## 🏗️ Architecture

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

## 📦 Installation

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

## 🚀 Usage

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

### Running the Standalone Stream Script

For testing or logging without the dashboard:

```bash
python src/acquisition/brainflow_stream.py --serial-port /dev/cu.usbserial-DM02583G --log-dir ./logs
```

Options:
- `--serial-port`: Serial port path (required)
- `--log-dir`: Directory to save raw EEG data (optional)
- `--chunk-size-ms`: Polling interval in milliseconds (default: 100)
- `--duration`: Stream duration in seconds (default: run until interrupted)
- `--verbose`: Enable verbose logging

## 📁 Project Structure

```
EEG-Sleep-Monitor/
├── src/
│   ├── acquisition/
│   │   └── brainflow_stream.py      # Cyton board streaming
│   ├── processing/
│   │   ├── ring_buffer.py           # Thread-safe data buffer
│   │   ├── filters.py               # Signal preprocessing
│   │   ├── metrics.py               # Feature extraction
│   │   └── processor.py             # Background processing worker
│   ├── app/
│   │   └── streamlit_app.py         # Web dashboard
│   └── utils/
│       └── logging.py               # Logging utilities
├── documentation/
│   ├── plan.md                      # Project plan and architecture
│   ├── data_pipeline.md            # Data flow documentation
│   └── sprints.md                   # Git workflow guide
├── scripts/
│   ├── run_stream.py                # Helper scripts
│   └── replay_session.py
├── tests/
│   ├── test_ring_buffer.py
│   └── test_metrics.py
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## 🔧 Configuration

### Serial Port Configuration

The application automatically converts macOS `tty.*` ports to `cu.*` ports for serial communication. On macOS, always use `/dev/cu.*` ports.

### Frequency Bands

Default EEG frequency bands:
- **Delta**: 0.5 - 4 Hz (deep sleep)
- **Theta**: 4 - 8 Hz (light sleep, meditation)
- **Alpha**: 8 - 13 Hz (relaxed wakefulness)
- **Beta**: 13 - 30 Hz (active thinking)
- **Gamma**: 30 - 100 Hz (high-level processing)

### Processing Parameters

- **Window Size**: 2 seconds (default)
- **Update Interval**: 1 second (default)
- **Sample Rate**: 250 Hz (Cyton standard)
- **Buffer Size**: 30 seconds (configurable)

## 🧪 Testing

Run tests (when implemented):
```bash
pytest tests/
```

## 📊 Data Format

### BrainFlow Integration
- Uses `DEFAULT_PRESET` for EEG data
- Extracts only EEG channels using `get_eeg_channels()`
- Data format: `[num_channels x num_samples]` numpy array

### Logged Data
When logging is enabled, raw EEG data is saved as CSV files:
- Format: Each row is a time point, each column is a channel
- Location: `src/acquisition/logs/` (or specified directory)
- Filename: `eeg_raw_YYYYMMDD_HHMMSS.csv`

## 🐛 Troubleshooting

### Connection Issues

**"UNABLE_TO_OPEN_PORT_ERROR"**:
- Verify the serial port exists: `ls /dev/cu.*` (macOS) or `ls /dev/ttyUSB*` (Linux)
- Ensure no other application is using the port
- On macOS, use `/dev/cu.*` not `/dev/tty.*`
- Check permissions (may need `sudo` on Linux)

**"Board not found"**:
- Verify Cyton board is powered on
- Check RFduino dongle connection
- Try unplugging and replugging the USB connection

### Performance Issues

**High CPU usage**:
- Reduce update interval in processing worker
- Increase chunk size (poll less frequently)
- Disable unnecessary filters

**Streaming lag**:
- Check buffer size (may be too small)
- Verify no other processes are using the serial port
- Reduce visualization update frequency

## 🔮 Future Enhancements

- [ ] Multi-channel visualization
- [ ] Real-time spectrogram (waterfall plot)
- [ ] Event detection and alerts
- [ ] Data replay mode
- [ ] Multi-patient support
- [ ] Database integration
- [ ] CAP (Cyclic Alternating Pattern) detection
- [ ] Modulatory oscillation analysis

## 📚 Documentation

- [Data Pipeline Architecture](documentation/data_pipeline.md)
- [Project Plan](documentation/plan.md)
- [Git Workflow](documentation/sprints.md)

## 🤝 Contributing

This is a capstone project. For questions or issues, please open an issue on GitHub.

## 📄 License

Apache-2.0 License

## 🙏 Acknowledgments

- [BrainFlow](https://github.com/brainflow-dev/brainflow) for EEG board integration
- [OpenBCI](https://openbci.com/) for the Cyton board
- [Streamlit](https://streamlit.io/) for the dashboard framework

## 📧 Contact

For questions or support, please open an issue on the GitHub repository.

---

**Note**: This is an MVP (Minimum Viable Product) for local, single-patient EEG monitoring. Future versions may include multi-patient support, cloud integration, and advanced sleep analysis features.