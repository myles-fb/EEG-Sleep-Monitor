# src/services — Service Layer

> **Start here:** Read `docs/sessions/2026-02-20-dashboard-redesign.md` for the latest session context.

## Key Files
- `study_service.py` — study CRUD, EDF processing, feature queries
- `spectrogram_service.py` — NPZ save/load for spectrogram data
- `patient_service.py` — patient CRUD
- `config_service.py` — Pi config JSON generation
- `device_service.py` — device CRUD
- `export_service.py` — CSV/JSON export

## study_service.py API

### Windowing (fixed in dashboard-redesign)
```python
_compute_n_windows(n_total, window_samples)  # includes near-complete trailing window
```
Fix for off-by-one: trailing chunk included if >= 50% of window_samples.

### Query helpers
```python
get_feature_timeseries_by_channel(study_id, feature_key, channel_index=None)
# -> (List[float] timestamps, List[float] values)
# channel_index=None returns all (legacy single-channel)

get_study_channels(study_id)  # -> [{index, label}, ...]
get_alerts(study_id)          # -> List[Alert]
get_hourly_summary(study_id, bucket_size_seconds=3600)  # -> List[dict]
```

### Feature key naming
```
mo_q_{band}        Q-score per band
mo_p_{band}        P-value per band
mo_count           Number of significant bands (p < 0.05)
mo_dom_freq_{band} Dominant modulation frequency (Hz)
mo_window_detail   JSON with per-LASSO-window arrays
```
Bands: `0.5_3hz`, `3_8hz`, `8_15hz`, `15_30hz`

## spectrogram_service.py API
```python
load_full_spectrogram(study_id, channel_index)
# -> dict with S (n_times, n_freqs), T (n_times,), F (n_freqs,),
#    env_0.5_3hz, env_3_8hz, env_8_15hz, env_15_30hz

list_available_channels(study_id)  # -> List[int]
```

### NPZ file location
`data/spectrograms/{study_id}/w{window:04d}_ch{channel:03d}.npz`

### Envelope sampling rate
Envelopes stored at spectrogram time resolution (step=6s → Fs_env≈1/6 Hz).

## DB Session Pattern
```python
from models import get_db
with get_db() as db:
    records = db.query(FeatureRecord).filter_by(study_id=study_id).all()
    # auto-commit on context exit
```
