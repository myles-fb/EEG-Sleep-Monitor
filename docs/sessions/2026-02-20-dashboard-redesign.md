# Session: Dashboard Redesign — 2026-02-20

## Goal
Restructure physician dashboard into two pages (Summary + Per-Channel), fix known bugs, add significance-encoded Q-heatmaps, envelope spectrograms, dominant freq tracking.

## Plan File
`docs/plans/2026-02-20-dashboard-redesign-implementation.md` — 11 tasks

## Design Spec
`docs/plans/2026-02-19-dashboard-redesign-design.md` — approved spec

## Branch
`dashboard-redesign` (branched from main)

## Key Findings

### Windowing Bug Root Cause
`study_service.py:184` — `n_windows = max(1, n_total // window_samples)`. MNE loads EDF with slightly fewer samples than expected (e.g. 449,999 vs 450,000). Fix: add helper `_compute_n_windows()` that includes trailing chunk if >= 50% of window_samples.

### Spectrogram X-axis Bug
`viz_helpers.py:69` — `create_spectrogram_heatmap` has no explicit y-axis range. Fix: add `range=[0, f_max], autorange=False` to y-axis layout.

### Envelope Sampling Rate
Envelopes stored at spectrogram time resolution: step=6s → Fs_env=1/6 Hz. Envelope spectrograms (2nd-order) use `scipy.signal.spectrogram` showing modulation frequencies up to ~0.05 Hz.

## Existing viz_helpers Functions (before this session)
- `create_spectrogram_heatmap(S, T, F, title, f_max, log_scale)`
- `create_band_envelope_plot(T, band_envelopes, title)` — REMOVED from new pages
- `create_q_heatmap(...)` — REPLACED by `create_q_heatmap_with_significance`
- `create_side_by_side(...)` — only used by Diagnostic tab (REMOVED)

## New viz_helpers Functions Added This Session
- `create_q_heatmap_with_significance(timestamps, q_data, p_data, p_cutoff, title)` — NaN for p>=cutoff
- `create_envelope_spectrogram(T, envelope, title, f_max_hz)` — 2nd-order spectrogram via scipy
- `create_band_limited_spectrogram(S, T, F, band_key, title, log_scale)` — sliced to band freq range
- `create_channel_averaged_spectrogram(spec_data_list, title, f_max, log_scale)` — averages S matrices
- `create_dominant_freq_chart(dom_freq_data, title)` — line chart per band
- `BAND_FREQS` dict — maps band_key to (f_low, f_high) tuple

## Page Structure After Redesign
1. `pages/1_Dashboard.py` → Summary page (averaged spectrogram, per-ch Q-heatmaps, dom-freq)
2. `pages/2_Per_Channel.py` → Per-Channel page (9 spectrograms, Q-heatmap, dom-freq) — NEW
3. `pages/3_New_Study.py` → renamed from 2_
4. `pages/4_Export.py` → renamed from 3_
5. `pages/5_Devices.py` → renamed from 4_

## Removed from Dashboard
- P-values timeline chart
- Significant MO Count per Window bar chart
- Hourly Aggregation table
- Diagnostic Comparison tab
- Envelope Power line graph

## Feature Key Naming
- `mo_q_{band}`, `mo_p_{band}`, `mo_count`, `mo_dom_freq_{band}`, `mo_window_detail`
- Bands: `0.5_3hz`, `3_8hz`, `8_15hz`, `15_30hz`

## NPZ Structure
`data/spectrograms/{study_id}/w{window:04d}_ch{ch:03d}.npz`
Keys: `S` (n_times, n_freqs), `T` (n_times,), `F` (n_freqs,), `env_0.5_3hz`, `env_3_8hz`, `env_8_15hz`, `env_15_30hz`

## Test Commands
```bash
.venv/bin/python -m pytest tests/ -v
streamlit run src/app/physician_app.py
```

## Tasks Completed This Session
- [x] Plan written
- [x] Task 1: Create branch (dashboard-redesign)
- [x] Task 2: Fix windowing bug (ffabb42)
- [x] Task 3: Fix spectrogram x-axis (973b338)
- [x] Task 4-7: Add all new viz helpers (cf14b8f)
- [x] Task 8: Renumber pages (6042d4d)
- [x] Task 9: Rewrite 1_Dashboard.py as Summary (42f4089)
- [x] Task 10: Create 2_Per_Channel.py (ea3adb3)
- [x] Task 11: Final verification — 25/25 tests passing

## Post-Merge Fix: Rangeslider X-axis Interruption (commit 472ef87)
- **Root cause:** `rangeslider=dict(visible=True)` on all heatmap `xaxis` dicts. Plotly adds a secondary mini-panel below with a dividing line that looks like an x-axis in the middle.
- **Fix:** Removed `rangeslider` from `create_spectrogram_heatmap`, `create_q_heatmap_with_significance`, `create_envelope_spectrogram`, `create_band_envelope_plot`, `create_q_heatmap`. Added `automargin=True` to Q-heatmap y-axis.
- **Dominant freq chart was unaffected** — it's a Scatter (no rangeslider), which confirmed the root cause.
- **Rule:** Never add `rangeslider` to Heatmap figures. Plotly's built-in zoom/pan toolbar handles navigation.

## 2nd-Order Envelope Spectrograms (commit 5bcb0cb)

### What Eq (1) means
S̄_{k,ρ}(t) = (1/|F_ρ|) Σ_{f ∈ F_ρ} S_k(f,t) — mean spectrogram power over band ρ at each time step.
This IS the `env_{band}` stored in NPZ files (computed by `extract_band_envelope` in mos.py).

### 2nd-order spectrogram parameters (from paper Sec. 4)
- Window: Tw = 30s → 5 samples at Fs_env = 1/6 Hz
- Step: 6s → 1 sample (noverlap = 4)
- nfft = 64 for smoother display (zero-padded; true resolution = 33 mHz)
- Y-axis: millihertz (mHz), range 0–100 mHz
- `create_envelope_spectrogram(T, envelope, title, f_max_mhz=100.0)`

### f_min fix for band-limited spectrograms
- `create_spectrogram_heatmap` now accepts `f_min=0.0` param
- `range=[f_min, f_max]` on y-axis — band spectrograms no longer show empty space below band
- `create_band_limited_spectrogram` passes `f_min=f_low` automatically

### Summary page additions (Section 2 after channel-averaged spectrogram)
- Envelope spectrograms section: averages envelopes across selected channels, 2x2 grid
- Caption explains Eq (1) and paper parameters

## Final State
All commits on `main` (5bcb0cb). 25 tests pass.
Pages: 1_Dashboard (Summary), 2_Per_Channel, 3_New_Study, 4_Export, 5_Devices.
