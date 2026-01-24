# Prototype #1 Streamlit MVP Proposal  
**Local, Single-Patient EEG Monitoring Dashboard**

---

## 1. Purpose and Success Criteria

**Purpose**  
Build a local, single-patient real-time EEG monitoring dashboard to validate the full data pipeline from acquisition to visualization:
Cyton → BrainFlow → signal processing → live dashboard.

**Success Criteria (Demo-Ready)**
- Continuous streaming for ≥30 minutes without crashes
- Live raw EEG visualization (5–10 Hz UI refresh)
- Rolling band-limited power metrics (1–2 s windows)
- Rolling PSD or basic spectrogram visualization
- Clear device + stream status indicators (connected, sample rate, dropped packets)

---

## 2. Scope

### In Scope
- Single Cyton board connected to one local machine
- One EEG channel (or fixed small set) with documented reference/ground
- Basic preprocessing:
  - DC removal / detrending
  - Optional 60 Hz notch filter
  - Optional bandpass filtering
- Metrics (v1):
  - Band-limited power (delta, theta, alpha, beta)
  - PSD (Welch) or rolling spectrogram
- Simple event flag placeholder (rule-based)
- Local session logging (raw + metrics)

### Out of Scope (for MVP1)
- Multi-patient dashboards
- User accounts / authentication
- Cloud backend or database
- Notification system (email/SMS)
- Full CAP / modulatory oscillation detectors

---

## 3. System Architecture (Local Only)

### Acquisition Layer
- BrainFlow manages Cyton connection
- Samples pulled in chunks (e.g., 50–250 ms)
- Data written to an in-memory ring buffer
- Optional raw data logging to disk

### Processing Layer
- Consumes rolling windows from ring buffer
- Computes metrics at fixed cadence (e.g., every 1 s)
- Publishes latest metric snapshot for visualization

### Visualization Layer (Streamlit)
- Reads buffer snapshots and metrics
- Renders live plots and numeric summaries
- Handles user controls (start/stop, filters, window sizes)

**Data Sharing Mechanism**
- Python `multiprocessing`
- Separate acquisition process
- Shared ring buffer abstraction

---

## 4. User Experience (Physician Mock Workflow)

### Status Panel
- Device connection status
- Stream running / stopped
- Sample rate
- Last-sample timestamp
- Dropped packet counter

### Main Dashboard Panels
1. **Raw EEG Trace**
   - Last 10–20 seconds
2. **Bandpower Tiles**
   - Delta / Theta / Alpha / Beta
   - Optional short-term trend lines
3. **Frequency View**
   - Rolling PSD or basic spectrogram
4. **Event Flag Panel**
   - Placeholder “Concerning Event Detected” indicator

### Controls
- Start / Stop streaming
- Notch filter on/off (60 Hz)
- Bandpass range selection
- Window length and update interval
- Enable / disable session logging

---

## 5. Signal Processing & Metrics (V1)

### Preprocessing
- Remove DC offset / detrend
- Optional 60 Hz notch filter
- Optional bandpass (e.g., 0.5–40 Hz)

### Bandpower
- Compute PSD via Welch on rolling windows
- Integrate PSD over bands:
  - Delta: 0.5–4 Hz
  - Theta: 4–8 Hz
  - Alpha: 8–13 Hz
  - Beta: 13–30 Hz
- Output absolute and optional relative power

### Frequency Visualization
- MVP: Rolling PSD updated every 1 s
- Stretch: Rolling spectrogram (waterfall)

### Event Flag Placeholder
- Simple rule-based trigger (e.g., power threshold, artifact detection)
- Output:
  - Boolean flag
  - Timestamp
  - Short description

---

## 6. Data Logging & Replay

### Logging
- Raw EEG: timestamped CSV or Parquet
- Metrics: bandpower + PSD summaries per update

### Optional Replay Mode
- Load recorded session
- Drive dashboard without live hardware
- Improves demo stability and debugging

---

## 7. Tech Stack & Repo Structure

### Stack
- Language: Python
- Libraries:
  - BrainFlow
  - NumPy / SciPy
  - Streamlit
  - (Optional) MNE, PyFFTW

## 8. Risks & Mitigations

| Risk | Mitigation |
|-----|-----------|
| Streamlit reruns / state loss | Separate acquisition process |
| RF dongle instability | Reconnect logic + status UI |
| High CPU from FFTs | Limit cadence, downsample |
| Noisy electrodes | Show signal quality indicators |

## 9. Implementation Plan (Sprint)

### Milestone A – Data Streaming
- Connect to Cyton via BrainFlow
- Stream into ring buffer
- Display raw EEG trace

### Milestone B – Metrics
- Implement Welch PSD
- Add bandpower tiles
- Validate windowing + cadence

### Milestone C – Demo Polish
- Status indicators
- Start/stop controls
- Session logging
- Event flag placeholder

### Stretch
- Replay mode
- Rolling spectrogram

---

## 10. Acceptance Test (Demo Script)

1. Launch Streamlit app → shows “Disconnected”
2. Click “Connect & Start” → status turns green
3. Raw EEG trace updates continuously
4. Bandpower values respond to user actions (eyes closed, jaw clench)
5. PSD view reflects frequency changes
6. Stop streaming → clean shutdown, logs saved

---

## 11. Post-MVP Path

- Replace Streamlit UI with React
- Add FastAPI backend and persistent database
- Introduce multi-patient abstraction
- Implement CAP and modulatory oscillation detectors
- Add alerting and notification system