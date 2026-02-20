"""Reusable Plotly figure builders for the physician dashboard."""

from typing import Dict, List, Optional

import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    go = None
    make_subplots = None

# Consistent band colors
BAND_COLORS = {
    "0.5_3hz": "#1f77b4",
    "3_8hz": "#ff7f0e",
    "8_15hz": "#2ca02c",
    "15_30hz": "#d62728",
}

BAND_DISPLAY = {
    "0.5_3hz": "Delta (0.5-3 Hz)",
    "3_8hz": "Theta (3-8 Hz)",
    "8_15hz": "Alpha (8-15 Hz)",
    "15_30hz": "Beta (15-30 Hz)",
}

# Envelope key prefixes used in NPZ files
_ENV_KEY_MAP = {
    "env_0.5_3hz": "0.5_3hz",
    "env_3_8hz": "3_8hz",
    "env_8_15hz": "8_15hz",
    "env_15_30hz": "15_30hz",
}


def create_spectrogram_heatmap(
    S: np.ndarray,
    T: np.ndarray,
    F: np.ndarray,
    title: str = "Spectrogram",
    f_max: float = 40.0,
    log_scale: bool = True,
) -> "go.Figure":
    """Create a scrollable spectrogram heatmap with rangeslider.

    Args:
        S: (n_times, n_freqs) power array.
        T: (n_times,) time in seconds.
        F: (n_freqs,) frequency in Hz.
        f_max: crop display to this frequency.
        log_scale: apply log10 to power values.
    """
    freq_mask = F <= f_max
    S_plot = S[:, freq_mask]
    F_plot = F[freq_mask]
    T_min = T / 60.0  # convert to minutes

    z = np.log10(S_plot + 1e-12).T if log_scale else S_plot.T

    fig = go.Figure(go.Heatmap(
        z=z,
        x=T_min,
        y=F_plot,
        colorscale="Viridis",
        colorbar=dict(title="log10(Power)" if log_scale else "Power"),
    ))
    fig.update_layout(
        title=title,
        xaxis=dict(title="Time (min)", rangeslider=dict(visible=True)),
        yaxis=dict(title="Frequency (Hz)"),
        height=400,
    )
    return fig


def create_band_envelope_plot(
    T: np.ndarray,
    band_envelopes: Dict[str, np.ndarray],
    title: str = "Band Envelopes",
) -> "go.Figure":
    """Overlay band envelopes with rangeslider.

    Args:
        T: (n_times,) time in seconds.
        band_envelopes: mapping band_label -> (n_times,) envelope.
            Keys may be raw labels ("0.5_3hz") or NPZ-style ("env_0.5_3hz").
    """
    T_min = T / 60.0

    fig = go.Figure()
    for raw_key, env in band_envelopes.items():
        band_key = _ENV_KEY_MAP.get(raw_key, raw_key)
        color = BAND_COLORS.get(band_key, None)
        label = BAND_DISPLAY.get(band_key, band_key)
        fig.add_trace(go.Scatter(
            x=T_min,
            y=env,
            mode="lines",
            name=label,
            line=dict(color=color),
        ))
    fig.update_layout(
        title=title,
        xaxis=dict(title="Time (min)", rangeslider=dict(visible=True)),
        yaxis=dict(title="Envelope Power"),
        height=350,
        legend=dict(orientation="h", y=-0.3),
    )
    return fig


def create_q_heatmap(
    timestamps: List[float],
    q_data: Dict[str, tuple],
    title: str = "Q-Score Heatmap",
    highlight_time: Optional[float] = None,
) -> "go.Figure":
    """Q-score heatmap: rows = bands, columns = time windows.

    Args:
        timestamps: time values in seconds (for x-axis).
        q_data: dict mapping band_label -> (ts_list, vals_list).
        highlight_time: optional time (seconds) to draw a vrect highlight.
    """
    bands = list(q_data.keys())
    T_min = [t / 60.0 for t in timestamps]

    z = []
    y_labels = []
    for band in bands:
        ts, vals = q_data[band]
        y_labels.append(BAND_DISPLAY.get(band, band))
        # Align vals to the common timestamps
        val_map = dict(zip(ts, vals))
        row = [val_map.get(t) for t in timestamps]
        z.append(row)

    fig = go.Figure(go.Heatmap(
        z=z,
        x=T_min,
        y=y_labels,
        colorscale="YlOrRd",
        colorbar=dict(title="Q-Score"),
    ))

    if highlight_time is not None:
        ht_min = highlight_time / 60.0
        fig.add_vrect(
            x0=ht_min - 0.5, x1=ht_min + 0.5,
            line_width=2, line_color="blue",
            fillcolor="rgba(0,0,255,0.1)",
        )

    fig.update_layout(
        title=title,
        xaxis=dict(title="Time (min)", rangeslider=dict(visible=True)),
        height=300,
    )
    return fig


def create_side_by_side(
    T_spec: np.ndarray,
    band_envelopes: Dict[str, np.ndarray],
    timestamps_q: List[float],
    q_data: Dict[str, tuple],
    highlight_time: Optional[float] = None,
) -> "go.Figure":
    """Side-by-side (stacked) envelope + Q-heatmap with shared x-axis.

    Args:
        T_spec: (n_times,) time in seconds for envelopes.
        band_envelopes: band_label -> (n_times,) envelope values.
        timestamps_q: common timestamps for Q-score data.
        q_data: band_label -> (ts_list, vals_list).
        highlight_time: optional seconds to highlight with vrect.
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=["Band Envelopes", "Q-Score Heatmap"],
        row_heights=[0.55, 0.45],
    )

    T_min = T_spec / 60.0

    # Top: band envelopes
    for raw_key, env in band_envelopes.items():
        band_key = _ENV_KEY_MAP.get(raw_key, raw_key)
        color = BAND_COLORS.get(band_key, None)
        label = BAND_DISPLAY.get(band_key, band_key)
        fig.add_trace(
            go.Scatter(x=T_min, y=env, mode="lines", name=label,
                       line=dict(color=color), legendgroup=band_key),
            row=1, col=1,
        )

    # Bottom: Q-score heatmap
    bands = list(q_data.keys())
    T_q_min = [t / 60.0 for t in timestamps_q]
    z, y_labels = [], []
    for band in bands:
        ts, vals = q_data[band]
        y_labels.append(BAND_DISPLAY.get(band, band))
        val_map = dict(zip(ts, vals))
        z.append([val_map.get(t) for t in timestamps_q])

    fig.add_trace(
        go.Heatmap(z=z, x=T_q_min, y=y_labels, colorscale="YlOrRd",
                   colorbar=dict(title="Q", len=0.4, y=0.2),
                   showlegend=False),
        row=2, col=1,
    )

    # Highlight vrect on both subplots
    if highlight_time is not None:
        ht_min = highlight_time / 60.0
        for row in [1, 2]:
            fig.add_vrect(
                x0=ht_min - 0.5, x1=ht_min + 0.5,
                line_width=2, line_color="blue",
                fillcolor="rgba(0,0,255,0.1)",
                row=row, col=1,
            )

    fig.update_layout(
        height=650,
        xaxis2=dict(title="Time (min)"),
        yaxis=dict(title="Envelope Power"),
        legend=dict(orientation="h", y=-0.15),
    )
    return fig
