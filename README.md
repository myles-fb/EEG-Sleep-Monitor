# EEG Sleep Monitor

Real-time EEG monitoring and Modulatory Oscillation (MO) detection for sleep analysis using OpenBCI Cyton boards. Supports two data paths: **EDF file upload** (batch processing) and **Raspberry Pi live streaming** (edge compute over WebSocket).

## Overview

```
EDF Files ─────────────────────────────────────┐
                                                ▼
Cyton Board → BrainFlow → Ring Buffer → Processing → SQLite DB → Streamlit Dashboard
                (Pi)                     (Pi edge)       ▲
                                                         │
                                            WebSocket ───┘
                                          (Pi → Server)
```

The system runs MO detection (multi-taper spectrogram, phase-randomized surrogates, LASSO, entropy-based q-score, statistical p-value) on 5-minute algorithm windows. Results are stored as time-series feature records and displayed on a multi-page physician dashboard.

## Features

- **MO Detection Pipeline**: Full MATLAB-equivalent MOs pipeline in Python (708 lines) — spectrogram, surrogates, band envelopes, GESD outlier removal, LASSO with sinusoid dictionary, q-value, p-value
- **Dual Data Sources**: Upload EDF files for batch processing, or stream live from Raspberry Pi
- **Multi-Patient Dashboard**: Patient profiles, per-patient feature toggles, configurable algorithm windows and dashboard buckets
- **Pi Edge Computing**: On-device processing on Raspberry Pi 4 — only features sent over the network, not raw EEG
- **WebSocket Gateway**: FastAPI server accepts multiple Pi connections, stores features in shared SQLite DB
- **Alerts**: Threshold-based MO count alerts with per-hour bucketing

## Architecture

```
src/
├── acquisition/          # Cyton board streaming via BrainFlow
│   └── brainflow_stream.py
├── processing/           # Signal processing (runs on both server and Pi)
│   ├── ring_buffer.py    # Thread-safe circular buffer
│   ├── filters.py        # DC removal, notch, bandpass
│   ├── metrics.py        # Bandpower, PSD, EEGFeatures dataclass
│   ├── processor.py      # Background processing worker
│   └── mos.py            # MO detection pipeline
├── models/               # SQLAlchemy ORM
│   ├── database.py       # SQLite engine, get_db() context manager
│   └── models.py         # Patient, Study, Device, FeatureRecord, Alert
├── services/             # Business logic
│   ├── patient_service.py
│   ├── study_service.py  # EDF processing + live study creation
│   ├── device_service.py # Pi device CRUD
│   ├── config_service.py # Generate Pi config from patient profile
│   └── export_service.py # CSV/JSON export
├── server/               # WebSocket server (FastAPI)
│   ├── ws_server.py      # WS endpoint + REST API
│   ├── device_manager.py # Active connection registry
│   └── ingestion_service.py  # Pi features → DB
├── pi/                   # Raspberry Pi client
│   ├── pi_main.py        # Entry point + streaming controller
│   ├── pi_config.py      # Config loader/cache
│   └── ws_client.py      # Async WS client with auto-reconnect
└── app/                  # Streamlit physician dashboard
    ├── physician_app.py  # Home page (patient list + creation)
    └── pages/
        ├── 1_Dashboard.py    # Charts: Q-score, P-value, MO count, hourly summary
        ├── 2_New_Study.py    # EDF upload or Pi live study
        ├── 3_Export.py       # CSV/JSON export
        └── 4_Devices.py      # Pi device management
```

## Installation

```bash
# Clone and setup
git clone <repo-url>
cd capstone
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requirements: Python 3.9+, OpenBCI Cyton board (for live streaming).

## Usage

### Physician Dashboard (EDF mode — no hardware needed)

```bash
streamlit run src/app/physician_app.py
```

Create a patient, upload an EDF file on the New Study page, view results on the Dashboard.

### Full System (Pi live streaming)

Start both the WebSocket server and Streamlit dashboard:

```bash
./scripts/start_server.sh
```

Or separately:

```bash
# Terminal 1: WebSocket server
cd src && uvicorn server.ws_server:app --host 0.0.0.0 --port 8765

# Terminal 2: Streamlit dashboard
streamlit run src/app/physician_app.py
```

On the Raspberry Pi:

```bash
cd src && python -m pi.pi_main \
    --serial-port /dev/ttyUSB0 \
    --server ws://SERVER_IP:8765/ws/pi/DEVICE_ID \
    --device-id DEVICE_ID
```

For testing without hardware (synthetic EEG):

```bash
cd src && python -m pi.pi_main \
    --synthetic \
    --server ws://localhost:8765/ws/pi/pi-test \
    --device-id pi-test
```

### Standalone Acquisition

```bash
python src/acquisition/brainflow_stream.py --serial-port /dev/cu.usbserial-DM02583G
```

### Batch MO Processing

```bash
python scripts/run_mos_edf_pipeline.py --input-dir /path/to/edfs --output-dir /path/to/results --n-surrogates 50
```

## Testing

```bash
python -m pytest tests/test_mos.py -v
```

10 tests covering the MO pipeline end-to-end (bipolar montage, spectrogram, surrogates, band envelopes, GESD, LASSO, q-value, p-value, multi-channel).

## Pi Deployment

Copy the systemd unit file for auto-start on boot:

```bash
sudo cp pi/eeg_streamer.service /etc/systemd/system/
# Edit the file to set SERVER_IP and DEVICE_ID
sudo systemctl enable eeg_streamer
sudo systemctl start eeg_streamer
```

## Configuration

- **Algorithm window**: 2, 5, or 10 minutes (per patient, default 5 min)
- **Dashboard bucket**: 30 min, 1 hr, or 2 hr aggregation (default 1 hr)
- **Sample rate**: 250 Hz (Cyton standard)
- **Frequency bands**: delta (0.5-4), theta (4-8), alpha (8-13), beta (13-30), gamma (30-100) Hz
- **MO bands**: 0.5-3, 3-8, 8-15, 15-30 Hz

## License

Apache-2.0

## Acknowledgments

- [BrainFlow](https://github.com/brainflow-dev/brainflow) — EEG board integration
- [OpenBCI](https://openbci.com/) — Cyton hardware
- [Streamlit](https://streamlit.io/) — Dashboard framework
- [FastAPI](https://fastapi.tiangolo.com/) — WebSocket server