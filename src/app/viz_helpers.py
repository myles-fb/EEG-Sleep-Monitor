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
    f_min: float = 0.0,
    log_scale: bool = True,
) -> "go.Figure":
    """Create a spectrogram heatmap.

    Args:
        S: (n_times, n_freqs) power array.
        T: (n_times,) time in seconds.
        F: (n_freqs,) frequency in Hz.
        f_max: crop display to this frequency (Hz).
        f_min: y-axis lower bound (Hz). Defaults to 0. Set to band f_low
            for band-limited spectrograms so the axis starts at the band edge.
        log_scale: apply log10 to power values.
    """
    freq_mask = (F >= f_min) & (F <= f_max)
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
        xaxis=dict(title="Time (min)"),
        yaxis=dict(title="Frequency (Hz)", range=[f_min, f_max], autorange=False),
        height=400,
        margin=dict(l=60, r=20, t=40, b=60),
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
        xaxis=dict(title="Time (min)"),
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
        xaxis=dict(title="Time (min)"),
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


# Frequency range for each band (Hz)
BAND_FREQS = {
    "0.5_3hz": (0.5, 3.0),
    "3_8hz": (3.0, 8.0),
    "8_15hz": (8.0, 15.0),
    "15_30hz": (15.0, 30.0),
}


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
                row.append(None)  # NaN -> gray in Plotly
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
        xaxis=dict(title="Time (min)"),
        yaxis=dict(automargin=True),
        height=300,
        margin=dict(l=60, r=20, t=40, b=50),
    )
    return fig


def create_envelope_spectrogram(
    T: np.ndarray,
    envelope: np.ndarray,
    title: str = "Envelope Spectrogram",
    f_max_mhz: float = 100.0,
    window_samples: int = 400,
) -> "go.Figure":
    """2nd-order spectrogram of the band-limited spectral envelope (Eq. 1, Loe et al. 2022).

    The band-limited spectral envelope S̄_{k,ρ}(t) is the mean spectrogram power
    over frequency band ρ at each time step (stored as env_{band} in NPZ files,
    sampled at f_s = 1/6 Hz due to the 6 s spectrogram step).

    To resolve mHz-scale MO modulation, the window must be long relative to the
    envelope sampling rate. With Fs_env ≈ 1/6 Hz:
      - frequency resolution = Fs_env / window_samples (Hz)
      - e.g. window_samples=400 (40 min) → 0.42 mHz resolution

    Step is set to window_samples // 4 (75 % overlap) to balance time resolution
    and computation. Frequency axis is in millihertz (mHz) to match Fig. 1F/G.

    Args:
        T: (n_times,) global time vector in seconds (from NPZ).
        envelope: (n_times,) band-limited power envelope (Eq. 1).
        title: chart title.
        f_max_mhz: upper bound for displayed modulation frequency (mHz). Default 100.
        window_samples: number of envelope samples per spectrogram window.
            Larger → finer frequency resolution, coarser time resolution.
            Default 400 (≈40 min at Fs_env=1/6 Hz, matching the paper's LASSO window).
    """
    from scipy.signal import spectrogram as sp_spectrogram

    if T.size < 5:
        fig = go.Figure()
        fig.update_layout(title=f"{title} (insufficient data)", height=250)
        return fig

    dt = float(T[1] - T[0]) if T.size > 1 else 6.0
    Fs_env = 1.0 / dt  # ~1/6 Hz

    nperseg = min(window_samples, max(4, len(envelope)))
    step_samples = max(1, nperseg // 4)  # 75 % overlap
    noverlap = nperseg - step_samples

    # Zero-pad for smoother frequency display (does not improve spectral resolution)
    nfft = max(nperseg, 256)

    f_env, t_env, Sxx = sp_spectrogram(
        envelope,
        fs=Fs_env,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        window="hann",
    )

    # Convert to mHz and crop to display range
    f_mhz = f_env * 1000.0
    freq_mask = f_mhz <= f_max_mhz
    f_display_mhz = f_mhz[freq_mask]
    Sxx_display = Sxx[freq_mask, :]

    t_global_min = (T[0] + t_env) / 60.0
    z = np.log10(Sxx_display + 1e-30)

    fig = go.Figure(go.Heatmap(
        z=z,
        x=t_global_min,
        y=f_display_mhz,
        colorscale="Viridis",
        colorbar=dict(title="log10(Power)"),
    ))
    fig.update_layout(
        title=title,
        xaxis=dict(title="Time (min)"),
        yaxis=dict(title="Modulation Freq (mHz)", autorange=False, range=[0, f_max_mhz]),
        height=300,
        margin=dict(l=60, r=20, t=40, b=50),
    )
    return fig


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
    return create_spectrogram_heatmap(
        S[:, mask], T, F[mask], title=title, f_max=f_high, f_min=f_low, log_scale=log_scale
    )


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
    return create_spectrogram_heatmap(S_avg, T, F, title=title, f_max=f_max, log_scale=log_scale)


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
