# src/app — Streamlit Dashboard Layer

> **Start here:** Read `docs/sessions/2026-02-20-dashboard-redesign.md` for the latest session context.

## Key Files
- `physician_app.py` — home page (patient list/creation), multi-page app entry point
- `viz_helpers.py` — all Plotly figure builders (import from here, never inline in pages)
- `pages/` — numbered pages; Streamlit orders them alphabetically/numerically

## viz_helpers.py API

### Constants
```python
BAND_COLORS   # dict: band_key -> hex color
BAND_DISPLAY  # dict: band_key -> human label
BAND_FREQS    # dict: band_key -> (f_low, f_high) Hz
```

### Figure builders
```python
create_spectrogram_heatmap(S, T, F, title, f_max=40.0, log_scale=True)
create_q_heatmap_with_significance(timestamps, q_data, p_data, p_cutoff=0.05, title)
create_envelope_spectrogram(T, envelope, title, f_max_hz=0.05)
create_band_limited_spectrogram(S, T, F, band_key, title, log_scale=True)
create_channel_averaged_spectrogram(spec_data_list, title, f_max=40.0, log_scale=True)
create_dominant_freq_chart(dom_freq_data, title)
```

All functions accept numpy arrays and return `go.Figure`.

## Patterns

### sys.path setup (required at top of every page)
```python
import sys
from pathlib import Path
_src = Path(__file__).resolve().parent.parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))
```

### Standard imports
```python
from models import init_db
from services import patient_service, study_service
from services.spectrogram_service import list_available_channels, load_full_spectrogram
from app.viz_helpers import (...)
```

### Page boilerplate
```python
init_db()
st.set_page_config(page_title="...", page_icon="...", layout="wide")
```

## MO Bands
```python
MO_BANDS = ["0.5_3hz", "3_8hz", "8_15hz", "15_30hz"]
```

## Environment
- Run: `streamlit run src/app/physician_app.py`
- Python: `.venv/bin/python`
