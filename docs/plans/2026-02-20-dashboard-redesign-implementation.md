# Dashboard Redesign Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restructure the physician dashboard into Summary and Per-Channel pages, fix known bugs, and add significance-encoded Q-heatmaps, envelope spectrograms, and dominant frequency tracking.

**Architecture:** The redesign is purely in the visualization/query layer — no pipeline (mos.py) or storage schema (FeatureRecord) changes. New viz helper functions are added to `viz_helpers.py`, `1_Dashboard.py` is rewritten as the Summary page, and a new `2_Per_Channel.py` page is created. Existing pages (New Study, Export, Devices) are renumbered to make room.

**Tech Stack:** Streamlit, Plotly, NumPy, SciPy (scipy.signal.spectrogram for envelope spectrograms), SQLAlchemy. Python 3.9.6 via `.venv/bin/python`. Run tests with `.venv/bin/python -m pytest`.

---

## Task 1: Create git branch

**Files:**
- No file changes

**Step 1: Create and switch to branch**

```bash
git checkout -b dashboard-redesign
```

**Step 2: Verify**

```bash
git branch --show-current
```
Expected output: `dashboard-redesign`

**Step 3: Commit**

```bash
git commit --allow-empty -m "chore: start dashboard-redesign branch"
```

---

## Task 2: Fix off-by-one windowing bug

**Root cause:** For a 30-min EDF loaded by MNE, `n_total` may be slightly less than `n_windows * window_samples` due to integer rounding (e.g., 449,999 instead of 450,000). `n_total // window_samples` then returns 5 instead of 6, dropping the nearly-complete final window.

**Fix:** After computing `n_windows` via floor division, check if there are remaining samples ≥ 50% of a window and include them as one more window. The existing `end = min(start + window_samples, n_total)` already truncates the final window correctly.

**Files:**
- Modify: `src/services/study_service.py:184`
- Test: `tests/test_study_service.py` (create new)

**Step 1: Write the failing test**

Create `tests/test_study_service.py`:

```python
"""Tests for study_service windowing logic."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def test_n_windows_includes_near_complete_final_window():
    """A recording with n_total just under 6 * window_samples should still yield 6 windows."""
    from services.study_service import _compute_n_windows

    window_samples = 75_000  # 5 min @ 250 Hz
    # Simulates MNE loading 30-min recording 1 sample short
    n_total = 6 * window_samples - 1  # 449_999

    n = _compute_n_windows(n_total, window_samples)
    assert n == 6, f"Expected 6 windows, got {n}"


def test_n_windows_exact_multiple():
    """Exact multiple should give the exact count."""
    from services.study_service import _compute_n_windows

    window_samples = 75_000
    n_total = 6 * window_samples  # 450_000

    n = _compute_n_windows(n_total, window_samples)
    assert n == 6


def test_n_windows_less_than_half_dropped():
    """A trailing chunk smaller than 50% of a window should NOT add an extra window."""
    from services.study_service import _compute_n_windows

    window_samples = 75_000
    n_total = 5 * window_samples + 10_000  # 385_010 — only 13% of a window left

    n = _compute_n_windows(n_total, window_samples)
    assert n == 5
```

**Step 2: Run test to confirm it fails**

```bash
.venv/bin/python -m pytest tests/test_study_service.py -v
```
Expected: ImportError — `_compute_n_windows` does not exist yet.

**Step 3: Add `_compute_n_windows` helper and update `process_edf`**

In `src/services/study_service.py`, add this function right above `process_edf` (before line 138):

```python
def _compute_n_windows(n_total: int, window_samples: int) -> int:
    """Compute the number of algorithm windows, including a nearly-complete final window.

    A trailing chunk is included if it is >= 50% of window_samples. This prevents
    an off-by-one drop when MNE loads the EDF with 1-2 samples fewer than expected.
    """
    n = max(1, n_total // window_samples)
    if n_total - n * window_samples >= window_samples // 2:
        n += 1
    return n
```

Then in `process_edf`, replace line 184:
```python
    n_windows = max(1, n_total // window_samples)
```
with:
```python
    n_windows = _compute_n_windows(n_total, window_samples)
```

**Step 4: Run tests to confirm they pass**

```bash
.venv/bin/python -m pytest tests/test_study_service.py -v
```
Expected: 3 PASSED

**Step 5: Run full test suite to confirm no regressions**

```bash
.venv/bin/python -m pytest tests/ -v
```
Expected: 13 PASSED

**Step 6: Commit**

```bash
git add tests/test_study_service.py src/services/study_service.py
git commit -m "fix: include near-complete final algorithm window (off-by-one)"
```

---

## Task 3: Fix spectrogram X-axis positioning

**Root cause:** `create_spectrogram_heatmap` in `viz_helpers.py` does not set an explicit y-axis range. Plotly adds autorange padding that pushes the x-axis away from the bottom.

**Files:**
- Modify: `src/app/viz_helpers.py:69-74`

**Step 1: Update `create_spectrogram_heatmap` layout**

In `src/app/viz_helpers.py`, replace the `fig.update_layout` call inside `create_spectrogram_heatmap` (lines 69-75):

```python
    fig.update_layout(
        title=title,
        xaxis=dict(title="Time (min)", rangeslider=dict(visible=True)),
        yaxis=dict(title="Frequency (Hz)"),
        height=400,
    )
```

with:

```python
    fig.update_layout(
        title=title,
        xaxis=dict(title="Time (min)", rangeslider=dict(visible=True)),
        yaxis=dict(title="Frequency (Hz)", range=[0, f_max], autorange=False),
        height=400,
        margin=dict(l=60, r=20, t=40, b=60),
    )
```

**Step 2: Verify visually**

Launch the app and open the Spectrograms tab. The x-axis should now sit at the bottom of the spectrogram.

```bash
streamlit run src/app/physician_app.py
```

**Step 3: Commit**

```bash
git add src/app/viz_helpers.py
git commit -m "fix: pin spectrogram y-axis range to prevent x-axis float"
```

---

## Task 4: Add significance-encoded Q-heatmap to viz_helpers

**Design:** Cells where `p >= p_cutoff` are set to `NaN` so Plotly renders them as neutral gray. Cells where `p < p_cutoff` keep their Q-score value and show in the YlOrRd colorscale. A note below the chart explains the gray encoding.

**Files:**
- Modify: `src/app/viz_helpers.py` (append new function)

**Step 1: Append `create_q_heatmap_with_significance` to `viz_helpers.py`**

Add after the last function in `src/app/viz_helpers.py`:

```python
def create_q_heatmap_with_significance(
    timestamps: List[float],
    q_data: Dict[str, tuple],
    p_data: Dict[str, tuple],
    p_cutoff: float = 0.05,
    title: str = "Q-Score Heatmap",
) -> "go.Figure":
    """Q-score heatmap with p-value significance encoding.

    Cells where p >= p_cutoff are rendered as NaN (gray).
    Cells where p < p_cutoff are shown vivid on the YlOrRd colorscale.

    Args:
        timestamps: common window start times in seconds (x-axis).
        q_data: band_label -> (ts_list, q_vals).
        p_data: band_label -> (ts_list, p_vals).
        p_cutoff: significance threshold (default 0.05).
        title: chart title.
    """
    bands = list(q_data.keys())
    T_min = [t / 60.0 for t in timestamps]

    z = []
    y_labels = []
    for band in bands:
        ts_q, vals_q = q_data[band]
        ts_p, vals_p = p_data[band]
        y_labels.append(BAND_DISPLAY.get(band, band))
        q_map = dict(zip(ts_q, vals_q))
        p_map = dict(zip(ts_p, vals_p))
        row = []
        for t in timestamps:
            p_val = p_map.get(t)
            q_val = q_map.get(t)
            if p_val is not None and p_val < p_cutoff:
                row.append(q_val)
            else:
                row.append(None)  # NaN → gray in Plotly
        z.append(row)

    fig = go.Figure(go.Heatmap(
        z=z,
        x=T_min,
        y=y_labels,
        colorscale="YlOrRd",
        colorbar=dict(title="Q-Score"),
        zmin=0,
    ))
    fig.update_layout(
        title=title,
        xaxis=dict(title="Time (min)", rangeslider=dict(visible=True)),
        height=300,
    )
    return fig
```

**Step 2: Confirm the module imports cleanly**

```bash
.venv/bin/python -c "from app.viz_helpers import create_q_heatmap_with_significance; print('OK')"
```
Expected: `OK`

**Step 3: Commit**

```bash
git add src/app/viz_helpers.py
git commit -m "feat: add significance-encoded Q-heatmap to viz_helpers"
```

---

## Task 5: Add envelope spectrogram function to viz_helpers

**Design:** Takes a concatenated band envelope (already stored in NPZ at spectrogram time resolution; step = 6 s, so Fs_env = 1/6 Hz) and computes a 2nd-order spectrogram using `scipy.signal.spectrogram`. Displays as a heatmap with modulation frequency on y-axis and time on x-axis.

**Files:**
- Modify: `src/app/viz_helpers.py` (append new function)

**Step 1: Append `create_envelope_spectrogram` to `viz_helpers.py`**

```python
def create_envelope_spectrogram(
    T: np.ndarray,
    envelope: np.ndarray,
    title: str = "Envelope Spectrogram",
    f_max_hz: float = 0.05,
) -> "go.Figure":
    """2nd-order spectrogram: spectrogram of a band envelope signal.

    The envelope is stored at spectrogram time resolution (step = 6 s,
    Fs_env ≈ 1/6 Hz). The output shows modulation frequencies up to
    f_max_hz (default 0.05 Hz = 20-second period oscillations).

    Args:
        T: (n_times,) time vector in seconds (global, from NPZ).
        envelope: (n_times,) band envelope values.
        title: chart title.
        f_max_hz: max modulation frequency to display (Hz).
    """
    from scipy.signal import spectrogram as sp_spectrogram

    if T.size < 4:
        fig = go.Figure()
        fig.update_layout(title=f"{title} (insufficient data)", height=250)
        return fig

    dt = float(T[1] - T[0]) if T.size > 1 else 6.0
    Fs_env = 1.0 / dt

    # Use a Hann window spanning ~25% of the signal length (minimum 4 samples)
    nperseg = max(4, min(len(envelope) // 4, 64))
    noverlap = nperseg // 2

    f_env, t_env, Sxx = sp_spectrogram(
        envelope,
        fs=Fs_env,
        nperseg=nperseg,
        noverlap=noverlap,
        window="hann",
    )

    # Crop to f_max_hz
    freq_mask = f_env <= f_max_hz
    f_display = f_env[freq_mask]
    Sxx_display = Sxx[freq_mask, :]

    # Convert spectrogram time offsets to global seconds, then minutes
    t_global_min = (T[0] + t_env) / 60.0

    z = np.log10(Sxx_display + 1e-30)

    fig = go.Figure(go.Heatmap(
        z=z,
        x=t_global_min,
        y=f_display,
        colorscale="Viridis",
        colorbar=dict(title="log10(Power)"),
    ))
    fig.update_layout(
        title=title,
        xaxis=dict(title="Time (min)", rangeslider=dict(visible=True)),
        yaxis=dict(title="Modulation Freq (Hz)", autorange=False, range=[0, f_max_hz]),
        height=300,
        margin=dict(l=60, r=20, t=40, b=60),
    )
    return fig
```

**Step 2: Confirm import**

```bash
.venv/bin/python -c "from app.viz_helpers import create_envelope_spectrogram; print('OK')"
```
Expected: `OK`

**Step 3: Commit**

```bash
git add src/app/viz_helpers.py
git commit -m "feat: add 2nd-order envelope spectrogram to viz_helpers"
```

---

## Task 6: Add band-limited and channel-averaged spectrogram functions to viz_helpers

**Design:**
- `create_band_limited_spectrogram`: slices the full S matrix to a band's frequency range.
- `create_channel_averaged_spectrogram`: averages S matrices from multiple channels.

**Files:**
- Modify: `src/app/viz_helpers.py` (append two new functions)

**Step 1: Append both functions to `viz_helpers.py`**

```python
# Frequency range for each band
BAND_FREQS = {
    "0.5_3hz": (0.5, 3.0),
    "3_8hz": (3.0, 8.0),
    "8_15hz": (8.0, 15.0),
    "15_30hz": (15.0, 30.0),
}


def create_band_limited_spectrogram(
    S: np.ndarray,
    T: np.ndarray,
    F: np.ndarray,
    band_key: str,
    title: str = "",
    log_scale: bool = True,
) -> "go.Figure":
    """Spectrogram cropped to a single frequency band's range.

    Args:
        S: (n_times, n_freqs) full power array.
        T: (n_times,) time in seconds.
        F: (n_freqs,) frequency in Hz.
        band_key: one of the BAND_FREQS keys (e.g. "0.5_3hz").
        title: chart title; defaults to band display name.
        log_scale: apply log10 to power values.
    """
    f_low, f_high = BAND_FREQS.get(band_key, (0.0, 40.0))
    if not title:
        title = f"Power Spectrogram — {BAND_DISPLAY.get(band_key, band_key)}"
    mask = (F >= f_low) & (F <= f_high)
    return create_spectrogram_heatmap(S[:, mask], T, F[mask], title=title,
                                      f_max=f_high, log_scale=log_scale)


def create_channel_averaged_spectrogram(
    spec_data_list: list,
    title: str = "Channel-Averaged Spectrogram",
    f_max: float = 40.0,
    log_scale: bool = True,
) -> "go.Figure":
    """Average power spectrograms across multiple channels.

    Args:
        spec_data_list: list of dicts, each with keys S (n_times, n_freqs),
            T (n_times,), F (n_freqs,). All must share the same T and F grids.
        title: chart title.
        f_max: crop display to this frequency.
        log_scale: apply log10 to averaged power values.
    """
    if not spec_data_list:
        fig = go.Figure()
        fig.update_layout(title="No spectrogram data", height=400)
        return fig

    S_avg = np.mean(np.stack([d["S"] for d in spec_data_list], axis=0), axis=0)
    T = spec_data_list[0]["T"]
    F = spec_data_list[0]["F"]
    return create_spectrogram_heatmap(S_avg, T, F, title=title,
                                      f_max=f_max, log_scale=log_scale)
```

**Step 2: Confirm import**

```bash
.venv/bin/python -c "from app.viz_helpers import create_band_limited_spectrogram, create_channel_averaged_spectrogram; print('OK')"
```
Expected: `OK`

**Step 3: Commit**

```bash
git add src/app/viz_helpers.py
git commit -m "feat: add band-limited and channel-averaged spectrogram helpers"
```

---

## Task 7: Add dominant frequency chart to viz_helpers

**Files:**
- Modify: `src/app/viz_helpers.py` (append new function)

**Step 1: Append `create_dominant_freq_chart` to `viz_helpers.py`**

```python
def create_dominant_freq_chart(
    dom_freq_data: Dict[str, tuple],
    title: str = "Dominant Modulation Frequency",
) -> "go.Figure":
    """Line chart showing dominant MO frequency per band over time.

    Args:
        dom_freq_data: band_label -> (ts_list, freq_vals_list) in seconds / Hz.
        title: chart title.
    """
    fig = go.Figure()
    for band, (ts, vals) in dom_freq_data.items():
        if not ts:
            continue
        T_min = [t / 60.0 for t in ts]
        fig.add_trace(go.Scatter(
            x=T_min,
            y=vals,
            mode="lines+markers",
            name=BAND_DISPLAY.get(band, band),
            line=dict(color=BAND_COLORS.get(band)),
            marker=dict(size=4),
        ))
    fig.update_layout(
        title=title,
        xaxis_title="Time (min)",
        yaxis_title="Frequency (Hz)",
        height=350,
        legend=dict(orientation="h", y=-0.2),
    )
    return fig
```

**Step 2: Confirm import**

```bash
.venv/bin/python -c "from app.viz_helpers import create_dominant_freq_chart; print('OK')"
```
Expected: `OK`

**Step 3: Commit**

```bash
git add src/app/viz_helpers.py
git commit -m "feat: add dominant frequency tracking chart to viz_helpers"
```

---

## Task 8: Renumber existing pages to make room for Per-Channel page

**Design:** Insert `2_Per_Channel.py` between Dashboard and New Study. Shift other pages up by one.

**Files:**
- Rename: `src/app/pages/2_New_Study.py` → `src/app/pages/3_New_Study.py`
- Rename: `src/app/pages/3_Export.py` → `src/app/pages/4_Export.py`
- Rename: `src/app/pages/4_Devices.py` → `src/app/pages/5_Devices.py`

**Step 1: Rename the files**

```bash
git mv src/app/pages/2_New_Study.py src/app/pages/3_New_Study.py
git mv src/app/pages/3_Export.py src/app/pages/4_Export.py
git mv src/app/pages/4_Devices.py src/app/pages/5_Devices.py
```

**Step 2: Verify Streamlit still finds all pages**

```bash
streamlit run src/app/physician_app.py &
sleep 5 && kill %1
```
(Just confirm no import errors in the output; manual inspection is sufficient.)

**Step 3: Commit**

```bash
git add -A
git commit -m "refactor: renumber pages to insert Per-Channel at position 2"
```

---

## Task 9: Rewrite 1_Dashboard.py as Summary page

**Design (from spec):**
- Channel-averaged spectrogram with channel toggle (sidebar multi-select)
- 4 Q-score heatmaps — one per channel — with significance encoding
- Band toggle (sidebar multi-select, filters Q-heatmap bands shown)
- Dominant frequency tracking line chart
- Feature set menu (sidebar; MOs active, CAP as placeholder)
- P-value cutoff slider (default 0.05)

Remove: P-values timeline, MO count bar chart, Hourly aggregation table, Diagnostic Comparison.

**Files:**
- Modify: `src/app/pages/1_Dashboard.py` (full rewrite)

**Step 1: Write the new 1_Dashboard.py**

Replace all content of `src/app/pages/1_Dashboard.py` with:

```python
"""
Summary Dashboard — Channel-averaged spectrogram, per-channel Q-heatmaps,
and dominant modulation frequency tracking.
"""

import sys
from pathlib import Path

_src = Path(__file__).resolve().parent.parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import streamlit as st
import numpy as np
import requests

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from models import init_db
from services import patient_service, study_service
from services.spectrogram_service import list_available_channels, load_full_spectrogram
from app.viz_helpers import (
    create_q_heatmap_with_significance,
    create_channel_averaged_spectrogram,
    create_dominant_freq_chart,
    BAND_DISPLAY,
    BAND_COLORS,
)

FASTAPI_URL = "http://localhost:8765"
MO_BANDS = ["0.5_3hz", "3_8hz", "8_15hz", "15_30hz"]

init_db()

st.set_page_config(page_title="Summary", page_icon="📊", layout="wide")
st.title("Summary")

# ---------------------------------------------------------------------------
# Patient & study selection
# ---------------------------------------------------------------------------

patients = patient_service.list_patients()
if not patients:
    st.info("No patients found. Create one on the Home page.")
    st.stop()

patient_names = {p.id: p.name for p in patients}
selected_pid = st.sidebar.selectbox(
    "Select Patient",
    options=[p.id for p in patients],
    format_func=lambda pid: patient_names[pid],
)

patient = patient_service.get_patient(selected_pid)
if patient is None:
    st.error("Patient not found.")
    st.stop()

studies = study_service.list_studies(patient.id)
if not studies:
    st.info(f"No studies for **{patient.name}**. Start one on the New Study page.")
    st.stop()

study_labels = {s.id: f"{s.source_file or s.source} — {s.status}" for s in studies}
selected_sid = st.sidebar.selectbox(
    "Select Study",
    options=[s.id for s in studies],
    format_func=lambda sid: study_labels[sid],
)

study = study_service.get_study(selected_sid)

# ---------------------------------------------------------------------------
# Study info & Pi status
# ---------------------------------------------------------------------------

st.sidebar.divider()
st.sidebar.markdown(f"**Patient:** {patient.name}")
st.sidebar.markdown(f"**Study:** {study.source_file or study.source}")
st.sidebar.markdown(f"**Status:** {study.status}")
if study.duration_sec:
    st.sidebar.markdown(f"**Duration:** {study.duration_sec / 3600:.1f} hours")

if study.source == "pi" and study.device_id:
    st.sidebar.divider()
    try:
        resp = requests.get(f"{FASTAPI_URL}/api/devices", timeout=3)
        if resp.status_code == 200:
            device_info = next(
                (d for d in resp.json() if d.get("db_id") == study.device_id), None
            )
            if device_info:
                dev_status = device_info.get("status", "offline")
                if dev_status == "streaming":
                    st.sidebar.success(f"Pi: {device_info.get('name', 'device')} — streaming")
                elif dev_status == "connected":
                    st.sidebar.warning(f"Pi: {device_info.get('name', 'device')} — connected (idle)")
                else:
                    st.sidebar.error(f"Pi: {device_info.get('name', 'device')} — offline")
    except requests.ConnectionError:
        st.sidebar.warning("WebSocket server unreachable")

if study.source == "pi" and study.status == "active":
    st.sidebar.divider()
    if st.sidebar.checkbox("Auto-refresh (30s)", value=False):
        import time as _time
        _time.sleep(30)
        st.rerun()

# ---------------------------------------------------------------------------
# Feature set menu
# ---------------------------------------------------------------------------

st.sidebar.divider()
st.sidebar.markdown("**Feature Sets**")
feature_sets = st.sidebar.multiselect(
    "Active feature sets",
    options=["MOs", "CAP (coming soon)"],
    default=["MOs"],
    help="Select which feature sets to display.",
)
show_mos = "MOs" in feature_sets

# ---------------------------------------------------------------------------
# Channel & band controls
# ---------------------------------------------------------------------------

study_channels = study_service.get_study_channels(selected_sid)
spec_channels = list_available_channels(selected_sid)

ch_label_map = {c["index"]: c.get("label") or f"Ch {c['index']}" for c in study_channels}

# Channel toggle: which channels to include in averaged spectrogram
all_ch_indices = [c["index"] for c in study_channels] if study_channels else spec_channels
ch_options_fmt = {ci: ch_label_map.get(ci, f"Ch {ci}") for ci in all_ch_indices}

if all_ch_indices:
    st.sidebar.divider()
    selected_channels = st.sidebar.multiselect(
        "Channels (averaged spectrogram)",
        options=all_ch_indices,
        default=all_ch_indices,
        format_func=lambda ci: ch_options_fmt.get(ci, f"Ch {ci}"),
    )
else:
    selected_channels = []

# Band toggle
st.sidebar.divider()
selected_bands = st.sidebar.multiselect(
    "Bands",
    options=MO_BANDS,
    default=MO_BANDS,
    format_func=lambda b: BAND_DISPLAY.get(b, b),
)

# P-value cutoff
st.sidebar.divider()
p_cutoff = st.sidebar.slider(
    "P-value cutoff",
    min_value=0.001,
    max_value=0.5,
    value=0.05,
    step=0.001,
    format="%.3f",
    help="Cells where p >= cutoff are shown in gray.",
)

# ---------------------------------------------------------------------------
# Early exit if no feature sets selected
# ---------------------------------------------------------------------------

if not show_mos:
    st.info("Select a feature set in the sidebar to display data.")
    st.stop()

# ---------------------------------------------------------------------------
# Fetch MOs data
# ---------------------------------------------------------------------------

def fetch_channel_q_p(study_id, channel_index=None):
    """Return q_data and p_data dicts for selected bands."""
    q_data, p_data = {}, {}
    for band in selected_bands:
        ts_q, vals_q = study_service.get_feature_timeseries_by_channel(
            study_id, f"mo_q_{band}", channel_index)
        ts_p, vals_p = study_service.get_feature_timeseries_by_channel(
            study_id, f"mo_p_{band}", channel_index)
        q_data[band] = (ts_q, vals_q)
        p_data[band] = (ts_p, vals_p)
    return q_data, p_data


# Use first channel's timestamps as the common x-axis reference
ref_ch = all_ch_indices[0] if all_ch_indices else None
ts_ref, _ = study_service.get_feature_timeseries_by_channel(
    selected_sid, "mo_count", ref_ch)

if not ts_ref:
    st.warning("No feature data for this study yet.")
    st.stop()

if not HAS_PLOTLY:
    st.warning("Install plotly for interactive charts.")
    st.stop()

# ---------------------------------------------------------------------------
# Section 1: Channel-Averaged Spectrogram
# ---------------------------------------------------------------------------

st.subheader("Channel-Averaged Spectrogram")

if not spec_channels:
    st.info("No spectrogram data available. Enable spectrogram saving when creating the study.")
elif not selected_channels:
    st.info("Select at least one channel in the sidebar.")
else:
    ch_to_load = [ci for ci in selected_channels if ci in spec_channels]
    if not ch_to_load:
        st.warning("No spectrogram data for the selected channels.")
    else:
        spec_data_list = []
        for ci in ch_to_load:
            d = load_full_spectrogram(selected_sid, ci)
            if d is not None:
                spec_data_list.append(d)
        if spec_data_list:
            ch_names_str = ", ".join(ch_label_map.get(ci, f"Ch {ci}") for ci in ch_to_load)
            fig_avg = create_channel_averaged_spectrogram(
                spec_data_list,
                title=f"Averaged: {ch_names_str}",
            )
            st.plotly_chart(fig_avg, use_container_width=True)
        else:
            st.warning("Could not load spectrogram data.")

# ---------------------------------------------------------------------------
# Section 2: Q-Score Heatmaps (one per channel)
# ---------------------------------------------------------------------------

st.subheader("Q-Score Heatmaps")
st.caption(f"Gray cells: p ≥ {p_cutoff:.3f} (not significant). Vivid cells: p < {p_cutoff:.3f}.")

if not selected_bands:
    st.info("Select at least one band in the sidebar.")
elif not study_channels:
    # Single-channel or legacy study: one heatmap
    q_data, p_data = fetch_channel_q_p(selected_sid, None)
    fig_q = create_q_heatmap_with_significance(
        ts_ref, q_data, p_data, p_cutoff=p_cutoff,
        title="Q-Score Heatmap",
    )
    st.plotly_chart(fig_q, use_container_width=True)
else:
    # One heatmap per channel; lay out in a 2-column grid
    cols_per_row = 2
    ch_list = study_channels
    for row_start in range(0, len(ch_list), cols_per_row):
        row_chs = ch_list[row_start:row_start + cols_per_row]
        cols = st.columns(len(row_chs))
        for col, ch_info in zip(cols, row_chs):
            with col:
                ci = ch_info["index"]
                label = ch_label_map.get(ci, f"Ch {ci}")
                q_data_ch, p_data_ch = fetch_channel_q_p(selected_sid, ci)
                ts_ch, _ = study_service.get_feature_timeseries_by_channel(
                    selected_sid, "mo_count", ci)
                if ts_ch:
                    fig_ch = create_q_heatmap_with_significance(
                        ts_ch, q_data_ch, p_data_ch,
                        p_cutoff=p_cutoff,
                        title=label,
                    )
                    st.plotly_chart(fig_ch, use_container_width=True)
                else:
                    st.info(f"No data for {label}")

# ---------------------------------------------------------------------------
# Section 3: Dominant Frequency Tracking
# ---------------------------------------------------------------------------

st.subheader("Dominant Modulation Frequency")

dom_freq_data = {}
for band in selected_bands:
    ch_idx = ref_ch
    ts_df, vals_df = study_service.get_feature_timeseries_by_channel(
        selected_sid, f"mo_dom_freq_{band}", ch_idx)
    dom_freq_data[band] = (ts_df, vals_df)

if any(ts for ts, _ in dom_freq_data.values()):
    fig_dom = create_dominant_freq_chart(dom_freq_data)
    st.plotly_chart(fig_dom, use_container_width=True)
else:
    st.info("No dominant frequency data available.")

# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------

alerts = study_service.get_alerts(selected_sid)
if alerts:
    st.subheader(f"Alerts ({len(alerts)})")
    for a in alerts:
        mins = a.timestamp / 60
        st.warning(
            f"**{a.alert_type.upper()}** at {mins:.0f} min "
            f"({a.bucket_time}) — value {a.actual_value:.0f} "
            f"exceeded threshold {a.threshold:.0f}"
        )
```

**Step 2: Verify Streamlit starts without import errors**

```bash
streamlit run src/app/physician_app.py &
sleep 5 && kill %1
```

**Step 3: Commit**

```bash
git add src/app/pages/1_Dashboard.py
git commit -m "feat: rewrite Dashboard as Summary page with averaged spectrogram and significance Q-heatmaps"
```

---

## Task 10: Create Per-Channel page (2_Per_Channel.py)

**Design (from spec):**
- Sidebar: channel select (single), band toggle, feature set menu, p-value cutoff
- Full-signal spectrogram
- Band-limited power spectrograms (one per selected band)
- Band envelope spectrograms / 2nd-order spectrograms (one per selected band)
- Q-score heatmap with significance encoding
- Dominant frequency tracking

**Files:**
- Create: `src/app/pages/2_Per_Channel.py`

**Step 1: Create the Per-Channel page**

Create `src/app/pages/2_Per_Channel.py` with:

```python
"""
Per-Channel Page — Full and band-limited spectrograms, envelope spectrograms,
Q-score heatmap, and dominant frequency tracking for a single channel.
"""

import sys
from pathlib import Path

_src = Path(__file__).resolve().parent.parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import streamlit as st
import numpy as np

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from models import init_db
from services import patient_service, study_service
from services.spectrogram_service import list_available_channels, load_full_spectrogram
from app.viz_helpers import (
    create_spectrogram_heatmap,
    create_band_limited_spectrogram,
    create_envelope_spectrogram,
    create_q_heatmap_with_significance,
    create_dominant_freq_chart,
    BAND_DISPLAY,
    BAND_COLORS,
    BAND_FREQS,
)

MO_BANDS = ["0.5_3hz", "3_8hz", "8_15hz", "15_30hz"]
ENV_KEY_PREFIX = "env_"

init_db()

st.set_page_config(page_title="Per-Channel", page_icon="📈", layout="wide")
st.title("Per-Channel View")

# ---------------------------------------------------------------------------
# Patient & study selection
# ---------------------------------------------------------------------------

patients = patient_service.list_patients()
if not patients:
    st.info("No patients found. Create one on the Home page.")
    st.stop()

patient_names = {p.id: p.name for p in patients}
selected_pid = st.sidebar.selectbox(
    "Select Patient",
    options=[p.id for p in patients],
    format_func=lambda pid: patient_names[pid],
)

patient = patient_service.get_patient(selected_pid)
if patient is None:
    st.error("Patient not found.")
    st.stop()

studies = study_service.list_studies(patient.id)
if not studies:
    st.info(f"No studies for **{patient.name}**.")
    st.stop()

study_labels = {s.id: f"{s.source_file or s.source} — {s.status}" for s in studies}
selected_sid = st.sidebar.selectbox(
    "Select Study",
    options=[s.id for s in studies],
    format_func=lambda sid: study_labels[sid],
)

study = study_service.get_study(selected_sid)

st.sidebar.divider()
st.sidebar.markdown(f"**Patient:** {patient.name}")
st.sidebar.markdown(f"**Study:** {study.source_file or study.source}")
st.sidebar.markdown(f"**Status:** {study.status}")
if study.duration_sec:
    st.sidebar.markdown(f"**Duration:** {study.duration_sec / 3600:.1f} hours")

# ---------------------------------------------------------------------------
# Feature set menu
# ---------------------------------------------------------------------------

st.sidebar.divider()
feature_sets = st.sidebar.multiselect(
    "Feature Sets",
    options=["MOs", "CAP (coming soon)"],
    default=["MOs"],
)
show_mos = "MOs" in feature_sets

# ---------------------------------------------------------------------------
# Channel selection
# ---------------------------------------------------------------------------

study_channels = study_service.get_study_channels(selected_sid)
spec_channels = list_available_channels(selected_sid)
ch_label_map = {c["index"]: c.get("label") or f"Ch {c['index']}" for c in study_channels}

all_ch_indices = [c["index"] for c in study_channels] if study_channels else spec_channels

st.sidebar.divider()
if not all_ch_indices:
    st.warning("No channels found for this study.")
    st.stop()

selected_ch = st.sidebar.selectbox(
    "Channel",
    options=all_ch_indices,
    format_func=lambda ci: ch_label_map.get(ci, f"Ch {ci}"),
)
ch_label = ch_label_map.get(selected_ch, f"Ch {selected_ch}")

# Band toggle
st.sidebar.divider()
selected_bands = st.sidebar.multiselect(
    "Bands",
    options=MO_BANDS,
    default=MO_BANDS,
    format_func=lambda b: BAND_DISPLAY.get(b, b),
)

# P-value cutoff
st.sidebar.divider()
p_cutoff = st.sidebar.slider(
    "P-value cutoff",
    min_value=0.001,
    max_value=0.5,
    value=0.05,
    step=0.001,
    format="%.3f",
)

# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

if not show_mos:
    st.info("Select a feature set in the sidebar to display data.")
    st.stop()

if not HAS_PLOTLY:
    st.warning("Install plotly for interactive charts.")
    st.stop()

# ---------------------------------------------------------------------------
# Load spectrogram data for selected channel
# ---------------------------------------------------------------------------

spec_data = None
if selected_ch in spec_channels:
    spec_data = load_full_spectrogram(selected_sid, selected_ch)

# ---------------------------------------------------------------------------
# Fetch Q / P / dominant freq data for selected channel
# ---------------------------------------------------------------------------

q_data, p_data, dom_freq_data = {}, {}, {}
for band in selected_bands:
    ts_q, vals_q = study_service.get_feature_timeseries_by_channel(
        selected_sid, f"mo_q_{band}", selected_ch)
    ts_p, vals_p = study_service.get_feature_timeseries_by_channel(
        selected_sid, f"mo_p_{band}", selected_ch)
    ts_df, vals_df = study_service.get_feature_timeseries_by_channel(
        selected_sid, f"mo_dom_freq_{band}", selected_ch)
    q_data[band] = (ts_q, vals_q)
    p_data[band] = (ts_p, vals_p)
    dom_freq_data[band] = (ts_df, vals_df)

ts_count, _ = study_service.get_feature_timeseries_by_channel(
    selected_sid, "mo_count", selected_ch)

# ---------------------------------------------------------------------------
# Section 1: Full-signal spectrogram
# ---------------------------------------------------------------------------

st.subheader(f"Channel: {ch_label}")

if spec_data is None:
    st.info("No spectrogram data for this channel. Enable spectrogram saving when creating the study.")
else:
    S = spec_data["S"]
    T = spec_data["T"]
    F = spec_data["F"]

    with st.expander("Full-Signal Spectrogram", expanded=True):
        fig_full = create_spectrogram_heatmap(S, T, F, title="Full-Signal Spectrogram")
        st.plotly_chart(fig_full, use_container_width=True)

    # ---------------------------------------------------------------------------
    # Section 2: Band-limited power spectrograms
    # ---------------------------------------------------------------------------

    if selected_bands:
        st.subheader("Band-Limited Power Spectrograms")
        cols_per_row = 2
        for row_start in range(0, len(selected_bands), cols_per_row):
            row_bands = selected_bands[row_start:row_start + cols_per_row]
            cols = st.columns(len(row_bands))
            for col, band in zip(cols, row_bands):
                with col:
                    fig_band = create_band_limited_spectrogram(S, T, F, band)
                    st.plotly_chart(fig_band, use_container_width=True)

    # ---------------------------------------------------------------------------
    # Section 3: Envelope spectrograms (2nd-order)
    # ---------------------------------------------------------------------------

    if selected_bands:
        st.subheader("Envelope Spectrograms (2nd-order)")
        for row_start in range(0, len(selected_bands), cols_per_row):
            row_bands = selected_bands[row_start:row_start + cols_per_row]
            cols = st.columns(len(row_bands))
            for col, band in zip(cols, row_bands):
                with col:
                    env_key = f"{ENV_KEY_PREFIX}{band}"
                    env = spec_data.get(env_key)
                    if env is not None:
                        fig_env = create_envelope_spectrogram(
                            T, env,
                            title=f"Envelope Spectrogram — {BAND_DISPLAY.get(band, band)}",
                        )
                        st.plotly_chart(fig_env, use_container_width=True)
                    else:
                        st.info(f"No envelope data for {BAND_DISPLAY.get(band, band)}")

# ---------------------------------------------------------------------------
# Section 4: Q-Score Heatmap with significance
# ---------------------------------------------------------------------------

if ts_count and selected_bands:
    st.subheader("Q-Score Heatmap")
    st.caption(f"Gray cells: p ≥ {p_cutoff:.3f}. Vivid cells: p < {p_cutoff:.3f}.")
    fig_q = create_q_heatmap_with_significance(
        ts_count, q_data, p_data,
        p_cutoff=p_cutoff,
        title=f"Q-Score — {ch_label}",
    )
    st.plotly_chart(fig_q, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 5: Dominant Frequency Tracking
# ---------------------------------------------------------------------------

if any(ts for ts, _ in dom_freq_data.values()):
    st.subheader("Dominant Modulation Frequency")
    fig_dom = create_dominant_freq_chart(dom_freq_data, title=f"Dominant Freq — {ch_label}")
    st.plotly_chart(fig_dom, use_container_width=True)
```

**Step 2: Verify Streamlit starts without errors**

```bash
streamlit run src/app/physician_app.py &
sleep 5 && kill %1
```

**Step 3: Run full test suite**

```bash
.venv/bin/python -m pytest tests/ -v
```
Expected: 13 PASSED

**Step 4: Commit**

```bash
git add src/app/pages/2_Per_Channel.py
git commit -m "feat: add Per-Channel page with 9 spectrograms and significance Q-heatmap"
```

---

## Task 11: Final verification

**Step 1: Run all tests**

```bash
.venv/bin/python -m pytest tests/ -v
```
Expected: all PASSED

**Step 2: Launch the app**

```bash
streamlit run src/app/physician_app.py
```

Manually verify:
- [ ] Sidebar shows: Summary (1), Per-Channel (2), New Study (3), Export (4), Devices (5)
- [ ] Summary page: averaged spectrogram renders for selected channels
- [ ] Summary page: one Q-heatmap per channel, gray for non-significant cells
- [ ] Summary page: dominant frequency chart shows one trace per band
- [ ] Summary page: p-value cutoff slider changes which cells are gray
- [ ] Summary page: band toggle hides/shows bands in Q-heatmaps and dom-freq chart
- [ ] Summary page: channel toggle changes which channels are averaged in spectrogram
- [ ] Per-Channel page: channel selector and band toggle work
- [ ] Per-Channel page: full spectrogram renders
- [ ] Per-Channel page: band-limited spectrograms render (2 per row)
- [ ] Per-Channel page: envelope spectrograms render (2 per row)
- [ ] Per-Channel page: Q-heatmap with significance renders
- [ ] Per-Channel page: dominant frequency chart renders
- [ ] Spectrogram x-axis sits at bottom of chart (no float)
- [ ] New Study, Export, Devices pages still load correctly

**Step 3: Commit**

```bash
git add -A
git commit -m "chore: dashboard redesign complete"
```
