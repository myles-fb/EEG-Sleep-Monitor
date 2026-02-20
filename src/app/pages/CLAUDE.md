# src/app/pages — Dashboard Pages

> **Start here:** Read `docs/sessions/2026-02-20-dashboard-redesign.md` for the latest session context.
> Also read: `src/app/CLAUDE.md` for viz_helpers API and common patterns.

## Page Order (after dashboard-redesign)
1. `1_Dashboard.py` — Summary page (channel-averaged spectrogram, per-ch Q-heatmaps, dom-freq)
2. `2_Per_Channel.py` — Per-Channel page (9 spectrograms, Q-heatmap, dom-freq)
3. `3_New_Study.py` — Start study from EDF or Pi streaming
4. `4_Export.py` — Export study data as CSV/JSON
5. `5_Devices.py` — Pi device management

## What Was Removed (dashboard-redesign)
- P-values timeline chart
- MO Count bar chart
- Hourly aggregation table
- Diagnostic Comparison tab
- Envelope Power line graph (replaced by envelope spectrograms)

## Summary Page Layout (`1_Dashboard.py`)
1. Channel-Averaged Spectrogram (channel toggle = sidebar multi-select)
2. Q-Score Heatmaps: one per channel, 2-column grid, significance-encoded (gray = p >= cutoff)
3. Dominant Modulation Frequency line chart

## Per-Channel Page Layout (`2_Per_Channel.py`)
1. Full-signal spectrogram
2. Band-limited power spectrograms (2 per row)
3. Envelope spectrograms / 2nd-order (2 per row)
4. Q-Score Heatmap with significance encoding
5. Dominant Frequency chart

## Sidebar Controls (both pages)
- Patient / Study selectors
- Feature Sets menu (MOs active, CAP placeholder)
- Channel toggle (Summary) / Channel selector (Per-Channel)
- Band toggle (multi-select)
- P-value cutoff slider (0.001–0.5, default 0.05)

## Key Service Calls
```python
study_service.get_study_channels(study_id)      # -> [{index, label}, ...]
study_service.get_feature_timeseries_by_channel(study_id, feature_key, channel_index)  # -> (ts, vals)
list_available_channels(study_id)               # -> [int, ...]
load_full_spectrogram(study_id, channel_index)  # -> {S, T, F, env_*} or None
```
