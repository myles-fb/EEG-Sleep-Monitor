# Capstone App Architecture – LLM Implementation Specification

---

## System Philosophy

We are building a distributed system with two layers:

### 1. Raspberry Pi (C++): "The Kitchen"
- Ingests EEG data from OpenBCI Cyton board
- Performs all computationally heavy processing
- Extracts features
- Packages results as structured JSON
- Optionally stores raw EEG locally
- Serves feature outputs to the app

### 2. Physician App (Python-based UI): "The Restaurant"
- Allows physician configuration
- Displays features and trends
- Stores results (time-series database)
- Handles exports and notifications
- Does NOT perform heavy signal processing

All core EEG processing must remain on the Pi for scalability and cost control.

---

# Core Design Decisions

## 1. Where Should Features Be Computed?

### Option A – Compute All Features on Pi
- Pi computes every possible feature
- Sends everything to database
- UI decides what to display

**Pros**
- No reconfiguration latency
- No synchronization issues
- Simpler pipeline logic

**Cons**
- Wasteful compute
- Larger data storage
- Less scalable

---

### Option B – Profile-Driven Compute (Preferred)
- Physician selects features in patient profile
- Profile config is sent to Pi
- Pi computes only requested features

**Pros**
- Efficient
- Scalable
- Reduces storage
- Better aligned with embedded system philosophy

**Cons**
- Requires configuration sync mechanism
- More complexity in profile update handling

**Decision:** Implement profile-driven computation with a stable feature template.

---

# Patient Profile System

## Patient Profile Fields (Initial Version)

Minimal first iteration:

- Patient ID
- Age (optional)
- Study Type:
  - Single night
  - Multi-night
- Feature Selection:
  - MO detection (q-values) (default)
  - Envelope spectrogram (default)
  - Full-signal spectrogram (default)
- Bucket size:
  - Default = 1 hour
- Window size for MOs algorithm:
  - Default = 5 minutes
- Notification thresholds (optional)
- Save raw EEG? (boolean)

---

## Profile → Pi Configuration Flow

1. Physician configures patient profile in app.
2. App generates a JSON config.
3. Config is sent to Pi.
4. Pi:
   - Adjusts active feature computation
   - Adjusts bucket time
   - Adjusts notification thresholds
   - Enables/disables raw data storage

---

# Feature Template Design

Even if features are not active, use a consistent schema.

Example:

```json
{
  "timestamp": "...",
  "patient_id": "...",
  "bucket_duration": 3600,
  "features": {
    "mo_q_score": null,
    "mo_count": null,
    "bandpower_delta": null,
    "bandpower_theta": null,
    "envelope_summary": null
  }
}
```

If a feature is disabled:
- Keep key
- Set value = null

This ensures:
- Schema consistency
- Easier database indexing
- Easier front-end rendering

---

# Time Bucketing System

We must define two distinct time concepts:

### 1. Algorithm Window
- Internal processing window
- Example: 5-minute window
- Used for envelope / MO detection

### 2. Dashboard Bucket
- Aggregation period for display
- Example: 1 hour
- Aggregates:
  - Mean Q-score
  - MO count
  - Bandpower averages

---

# First Iteration Scope (Single Night MVP)

## MVP Capabilities

- Create patient profile
- Select MO detection
- Default:
  - 5-minute algorithm windows
  - 1-hour dashboard buckets
- Compute and display:
  - Hourly Q-score
  - Hourly MO count
- Display live-updating graph
- Store results in time-series DB
- Export results as:
  - CSV
  - JSON

Not included in MVP:
- Multi-night aggregation
- Advanced ML recommendations
- Dashboard widget editing

---

# Dashboard Requirements

## Main Screen

- Button: "Create New Patient Profile"
- List of active patients

---

## Patient Dashboard View

Display:

- Time-series graph of Q-score
- MO count per bucket
- Study progress (if live)
- Export button

Future additions:
- Envelope spectrogram display
- Widget customization
- Threshold-triggered alerts

---

# Notification System (MVP)

Configurable in profile:

- Feature threshold
- Time interval check

Pi evaluates thresholds.
If triggered:
- Sends event to app
- App logs and displays alert

Example:

```json
{
  "alert_type": "mo_count",
  "threshold": 15,
  "bucket_time": "02:00-03:00"
}
```

---

# Data Storage Model

## On Pi

- Temporary buffer
- Optional raw EEG file storage
- Feature JSON logs

## On App Backend

Time-series database structure:

```
patient_id
night_id
timestamp
feature_key
feature_value
```

Future:
- Multi-night aggregation layer
- Trend detection

---

# Multi-Night (Phase 2)

Planned features:

- Stacked nightly graphs
- Simple trend visualization
- Export of multi-night dataset
- No complex averaging layer initially

---

# Export Requirements

Physicians must be able to export:

- Feature summaries (CSV)
- Full feature JSON
- Optional raw EEG

Must be compatible with:

- MATLAB
- Python
- External EEG analysis tools

---

# Architecture Summary

```
Cyton Board
    ↓
Raspberry Pi (C++)
    - Signal Processing
    - Feature Extraction
    - Threshold Detection
    - JSON Packaging
    ↓
App Backend (Python)
    - Profile Management
    - Data Storage
    - Visualization
    - Export
    - Alerts
```

---

# Critical Constraints

- Heavy processing must remain on Pi
- Minimize cloud dependency
- Minimize database bloat
- Keep feature schema stable
- Maintain modularity for new features
- Ensure reproducibility of metrics

---

# LLM Implementation Guidance

When modifying the existing Python codebase:

1. Do NOT rewrite EEG processing logic.
2. Wrap existing logic into modular feature extractors.
3. Create a configuration layer that:
   - Reads patient profile
   - Generates Pi configuration JSON
4. Create database models for:
   - Patient
   - Night
   - FeatureRecord
5. Implement:
   - Hourly aggregation
   - Graph building
6. Build export utilities.
7. Stub multi-night logic but do not fully implement yet.

---

# What We Are NOT Building Yet

- ML-based feature recommendation
- Automatic feature selection
- Multi-night trend inference engine
- Fully customizable dashboard widgets
- Cloud-heavy architecture

---

# Final Guiding Principle

The Pi is the embedded signal processing engine.

The App is the structured visualization and orchestration layer.

Processing lives in C++.
Configuration and visualization live in Python.

Keep the boundary clean.

