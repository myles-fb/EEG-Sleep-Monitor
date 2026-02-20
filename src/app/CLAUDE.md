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
create_spectrogram_heatmap(S, T, F, title, f_max=40.0, f_min=0.0, log_scale=True)
# f_min sets the y-axis lower bound — pass band f_low for band-limited views

create_q_heatmap_with_significance(timestamps, q_data, p_data, p_cutoff=0.05, title)
create_envelope_spectrogram(T, envelope, title, f_max_mhz=100.0)
# 2nd-order spectrogram per Eq (1) Loe et al. 2022; y-axis in mHz

create_band_limited_spectrogram(S, T, F, band_key, title, log_scale=True)
# Automatically sets f_min=f_low so y-axis starts at band lower bound, not 0

create_channel_averaged_spectrogram(spec_data_list, title, f_max=40.0, log_scale=True)
create_dominant_freq_chart(dom_freq_data, title)
```

All functions accept numpy arrays and return `go.Figure`.

**No rangesliders on heatmaps** — rangesliders were removed from all heatmap/spectrogram figures because they create a secondary panel with a dividing line that visually interrupts the chart (looks like x-axis in the middle). Use Plotly's built-in zoom/pan toolbar instead. Scatter/line charts may still use rangesliders.

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
