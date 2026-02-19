# Dashboard Redesign Design

**Date**: 2026-02-19
**Status**: Approved (pending implementation plan)

## Summary

Restructure the physician dashboard into two focused pages — **Summary** and **Per-Channel** — with bug fixes, visualization improvements, and a feature-set menu for future extensibility.

## Page Structure

### Summary Page
- **Channel-averaged spectrogram**: Average power values across user-selected channels into one spectrogram (same time/frequency axes, averaged power).
- **4 Q-score heatmaps**: One per channel. Each heatmap has rows = frequency bands, columns = time windows, color = q-score strength. Significance encoding (see below).
- **Channel toggle**: Sidebar multi-select to choose which channels are included in the averaged spectrogram calculation.
- **Band toggle**: Sidebar multi-select to choose which frequency bands are displayed.
- **Dominant frequency tracking**: Line chart, one trace per band, showing dominant modulation frequency over time.
- **Feature set menu**: Sidebar multi-select for feature sets (MOs now, CAP later). Multiple feature sets can be viewed simultaneously — each selected feature renders its own chart group on the page.

### Per-Channel Page
- **9 spectrograms per channel**:
  - 4 band-limited 1st-order (power) spectrograms — one per frequency band
  - 4 band-limited 2nd-order (envelope) spectrograms — spectrogram of the envelope signal for each band
  - 1 full-signal spectrogram
- **Q-score heatmap** with significance encoding (see below).
- **Band toggle**: Sidebar multi-select to choose which bands are displayed (filters which of the 8 band-limited spectrograms appear).
- **Dominant frequency tracking**: Same line chart as summary but for this channel only.
- **Feature set menu**: Same sidebar multi-select as summary page.

## Visualization Changes

### Q-value + P-value Integration
- **Remove** the separate P-values timeline chart.
- **Encode significance directly on the Q-heatmap**: Cells where p >= cutoff are faded/grayed out; cells where p < cutoff are vivid/saturated.
- **Configurable p-value cutoff**: Slider or input (default 0.05) so physicians can adjust the significance threshold.

### Removals
- **Hourly aggregation table** — removed entirely.
- **"Envelope Power" line graph** — replaced by envelope spectrograms (2nd-order spectrograms on per-channel page).
- **"Diagnostic Comparison" page** — removed entirely.
- **"Significant MO Count per Window" bar chart** — removed entirely.

## Bug Fixes

### Off-by-one: Only 5 windows for 30-min recording (should be 6)
- **Symptom**: 30-minute EDF produces windows at 0, 5, 10, 15, 20 (5 windows) instead of 0, 5, 10, 15, 20, 25 (6 windows).
- **Root cause**: Likely the windowing loop drops the last complete window. Investigate `study_service.process_edf()` slicing logic.
- **Fix**: Ensure all complete algorithm windows are processed, including the final one.

### Timestamps represent window start
- Timestamps should show the start of each algorithm window (0, 5, 10, ...), not the center or end.

### X-axis appears in middle of spectrogram graphs
- **Symptom**: X-axis renders just shy of the bottom on spectrogram/diagnostics pages, consistently.
- **Root cause**: Likely a Plotly layout issue — y-axis range not explicitly set, or margin/autorange problem in `create_spectrogram_heatmap`.
- **Fix**: Explicitly set y-axis range starting at 0 and ensure proper margin configuration.

## Data Flow

No changes to the processing pipeline (`mos.py`) or storage schema (`FeatureRecord`). All required data (q-scores, p-values, dominant frequencies, spectrograms, envelopes) is already computed and stored. Changes are purely in the visualization/query layer.

### New computations needed at display time:
- **Channel-averaged spectrogram**: Load per-channel spectrograms from `.npz` files, average `S` matrices across selected channels.
- **Envelope spectrograms (2nd-order)**: Compute spectrogram of the band envelope signals. The envelopes are already stored in the `.npz` files (`env_{band}`). Need to compute a spectrogram of each envelope at display time (or pre-compute during processing).
- **Band-limited power spectrograms**: Filter the full spectrogram to each band's frequency range at display time.

## Out of Scope
- New feature ideas (LLM integration, video integration, custom features, etc.) — deferred.
- Time window scroller fix — deferred.
- CAP implementation — future version; only the menu placeholder is needed now.
- Spectrogram resolution configuration — use sensible defaults.
