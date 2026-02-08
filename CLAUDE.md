# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EEG Sleep Monitor: a real-time EEG monitoring dashboard for sleep analysis using OpenBCI Cyton boards. The pipeline flows: **Cyton Board -> BrainFlow -> Ring Buffer -> Processing -> Streamlit Dashboard**.

## Common Commands

```bash
# Install dependencies (Python 3.9+, use a virtualenv)
pip install -r requirements.txt

# Run the Streamlit dashboard
streamlit run src/app/streamlit_app.py

# Run standalone EEG stream acquisition
python src/acquisition/brainflow_stream.py --serial-port /dev/cu.usbserial-DM02583G

# Run MOs batch processing on EDF files
python scripts/run_mos_edf_pipeline.py --input-dir /path/to/edfs --output-dir /path/to/results --n-surrogates 50

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

### Visualization Layer (`src/app/`)
- `streamlit_app.py` — Real-time web dashboard reading from `shared_state`. Thread communication uses mutable containers (lists/dicts with locks), not `st.session_state` directly.

## Key Conventions

- **EEG data shape**: Always `(num_channels, num_samples)` — 2D arrays. Single channel extracted as `data[ch, :]`.
- **Frequency bands**: delta (0.5-4), theta (4-8), alpha (8-13), beta (13-30), gamma (30-100) Hz — defined in `metrics.py` as `FREQ_BANDS`.
- **Thread safety**: RingBuffer and ProcessingWorker use `threading.Lock`. Shared state accessed via `shared_state['_lock']`.
- **Sample rate**: 250 Hz (Cyton standard), referenced throughout as `sample_rate` parameter.
- **macOS serial ports**: Code auto-converts `/dev/tty.*` to `/dev/cu.*` in brainflow_stream.py.
- **Logging**: Standard `logging.getLogger(__name__)` pattern throughout.
- **MOs numpy serialization**: MOs results convert numpy arrays to lists via `.tolist()` before storing in shared state.

## Testing

Only `tests/test_mos.py` has implemented tests (11 test functions covering the MOs pipeline end-to-end). `test_ring_buffer.py` and `test_metrics.py` are placeholder files.
