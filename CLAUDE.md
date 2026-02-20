# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EEG Sleep Monitor: a real-time EEG monitoring dashboard for sleep analysis using OpenBCI Cyton boards. Two data paths: **EDF upload** (batch processing) and **Raspberry Pi live streaming** (edge compute + WebSocket). Pipeline: **Cyton Board -> BrainFlow -> Ring Buffer -> Processing -> WebSocket/DB -> Streamlit Dashboard**.

## Common Commands

```bash
# Install dependencies (Python 3.9+, use a virtualenv)
pip install -r requirements.txt

# Run the Physician Dashboard (multi-page app)
streamlit run src/app/physician_app.py

# Run the real-time EEG streaming dashboard (requires Cyton board)
streamlit run src/app/streamlit_app.py

# Run standalone EEG stream acquisition
python src/acquisition/brainflow_stream.py --serial-port /dev/cu.usbserial-DM02583G

# Run MOs batch processing on EDF files
python scripts/run_mos_edf_pipeline.py --input-dir /path/to/edfs --output-dir /path/to/results

# Start WebSocket server (Pi gateway) + Streamlit dashboard together
./scripts/start_server.sh

# Start WebSocket server alone (for Pi connections)
cd src && uvicorn server.ws_server:app --host 0.0.0.0 --port 8765

# Start Pi streaming client (on the Raspberry Pi)
cd src && python -m pi.pi_main --serial-port /dev/ttyUSB0 --server ws://HOST:8765/ws/pi/DEVICE_ID --device-id DEVICE_ID

# Pi streaming with synthetic board (no hardware, for testing)
cd src && python -m pi.pi_main --synthetic --server ws://localhost:8765/ws/pi/pi-test --device-id pi-test

# Run all tests
pytest tests/

# Run a specific test file
pytest tests/test_mos.py -v

# Run a single test function
pytest tests/test_mos.py::test_compute_mos_for_bucket_smoke
```

## Architecture

Three-layer architecture with thread-based concurrency:

### Acquisition Layer (`src/acquisition/`)
- `brainflow_stream.py` — CytonStream class manages board connection, polls data at 250 Hz, writes to RingBuffer

### Processing Layer (`src/processing/`)
- `ring_buffer.py` — Thread-safe circular buffer storing `(num_channels, num_samples)` arrays. Core data structure connecting acquisition to processing.
- `filters.py` — Preprocessing: DC removal, detrending, 60 Hz notch filter, bandpass (0.5-40 Hz)
- `metrics.py` — Feature extraction: bandpower, relative bandpower, PSD via Welch's method. Uses `EEGFeatures` dataclass as the feature container.
- `processor.py` — `ProcessingWorker` runs on a background thread, reads windows from RingBuffer, applies filters, computes metrics, stores results in `shared_state` dict (guarded by `threading.Lock`)
- `mos.py` — Modulatory Oscillations (MOs) detection pipeline: multi-taper spectrogram -> phase-randomized surrogates -> band envelope extraction -> GESD outlier removal -> LASSO with sinusoid dictionary -> entropy-based q-value -> statistical p-value

### Data Layer (`src/models/`)
- `database.py` — SQLAlchemy engine (SQLite at `data/physician.db`), session management via `get_db()` context manager
- `models.py` — ORM models: `Patient`, `Study`, `FeatureRecord`, `Alert`, `Device`

### Service Layer (`src/services/`)
- `patient_service.py` — Patient CRUD operations
- `study_service.py` — Study management, EDF processing pipeline, live study creation, feature queries, hourly aggregation
- `config_service.py` — Generate Pi configuration JSON from patient profile (includes `study_id`, `n_surrogates`)
- `device_service.py` — Device CRUD: register, assign patient, update status/heartbeat
- `export_service.py` — Export study data as CSV or JSON

### WebSocket Server Layer (`src/server/`)
- `ws_server.py` — FastAPI app: WebSocket endpoint for Pi connections (`/ws/pi/{device_id}`), REST endpoints for dashboard (`/api/devices`, config push, commands)
- `device_manager.py` — In-memory registry of active WebSocket connections, push config/commands to Pis
- `ingestion_service.py` — Stores Pi feature data into DB (mirrors `study_service.process_edf()` storage pattern)

### Pi Client Layer (`src/pi/`)
- `pi_main.py` — Entry point: connects Cyton board + WebSocket, runs processing loop per algorithm window, sends features to server
- `pi_config.py` — Loads/caches config from server or local JSON, feature toggle checks
- `ws_client.py` — Async WebSocket client with auto-reconnect, heartbeat, feature sending

### Visualization Layer (`src/app/`)
- `physician_app.py` — Multi-page physician dashboard (home page: patient list + creation)
- `pages/1_Dashboard.py` — Patient dashboard: Q-score trends, MO count, p-values, alerts; live Pi status for streaming studies
- `pages/2_New_Study.py` — Start study from EDF files or live Pi streaming
- `pages/3_Export.py` — Export study results as CSV or JSON
- `pages/4_Devices.py` — Device management: register Pi devices, assign patients, start/stop live studies
- `streamlit_app.py` — Real-time web dashboard reading from `shared_state`. Thread communication uses mutable containers (lists/dicts with locks), not `st.session_state` directly.

## Key Conventions

- **EEG data shape**: Always `(num_channels, num_samples)` — 2D arrays. Single channel extracted as `data[ch, :]`.
- **Frequency bands**: delta (0.5-4), theta (4-8), alpha (8-13), beta (13-30), gamma (30-100) Hz — defined in `metrics.py` as `FREQ_BANDS`.
- **Thread safety**: RingBuffer and ProcessingWorker use `threading.Lock`. Shared state accessed via `shared_state['_lock']`.
- **Sample rate**: 250 Hz (Cyton standard), referenced throughout as `sample_rate` parameter.
- **macOS serial ports**: Code auto-converts `/dev/tty.*` to `/dev/cu.*` in brainflow_stream.py.
- **Logging**: Standard `logging.getLogger(__name__)` pattern throughout.
- **MOs numpy serialization**: MOs results convert numpy arrays to lists via `.tolist()` before storing in shared state.

## Key Conventions (continued)

- **Database**: SQLite via SQLAlchemy, stored at `data/physician.db`. Sessions managed via `get_db()` context manager (auto-commit, auto-rollback).
- **Feature storage**: Key-value time series in `FeatureRecord` table. Keys like `mo_q_0.5_3hz`, `mo_p_3_8hz`, `mo_count`, `mo_dom_freq_*`.
- **Time bucketing**: Algorithm window (default 5 min) for MOs processing; dashboard bucket (default 1 hr) for aggregated display.
- **Imports**: `src/` added to `sys.path` by each Streamlit page. Use `from models import ...` and `from services import ...`.

## Testing

Only `tests/test_mos.py` has implemented tests (10 test functions covering the MOs pipeline end-to-end). `test_ring_buffer.py` and `test_metrics.py` are placeholder files.
