"""
Modulatory Oscillations (MOs) detection pipeline.

Replicates the MATLAB pipeline for detecting MOs in EEG: bipolar montage,
multi-taper spectrogram, phase-randomized surrogates, band-limited envelopes,
GESD outlier removal, LASSO-based MO detection, and significance vs surrogate.

Designed to run on a single time bucket (e.g. 2 or 5 minutes) from the ring
buffer and produce a q-value (likelihood of modulation) per frequency band.
"""

from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import signal as scipy_signal
from scipy import stats
from scipy.signal.windows import dpss
from typing import Optional, Tuple, List, Dict, Any, Union
from dataclasses import dataclass, field

# Optional sklearn for LASSO path (MATLAB lasso-style); fallback if missing
try:
    from sklearn.linear_model import lasso_path
    HAS_SKLEARN = True
except ImportError:
    lasso_path = None
    HAS_SKLEARN = False


# Frequency bands for MO envelope extraction (Hz)
MO_FREQ_BANDS = [
    (0.5, 3),
    (3, 8),
    (8, 15),
    (15, 30),
]

# Spectrogram parameters (Chronux-style)
TAPERS_NW = 3
TAPERS_K = 5
FPASS = (0, 100)
SPECTROGRAM_WINDOW_SEC = 30
SPECTROGRAM_STEP_SEC = 6

# LASSO parameters (MATLAB lasso default: geometric λ path, no λ₂)
LASSO_WINTIME_SEC = 120
LASSO_WINJUMP_SEC = 30

# GESD
GESD_ALPHA = 0.05
GESD_MAX_OUTLIERS = 10

# LASSO HIE (from MATLAB lassoHIE): sinusoid dictionary
LASSO_ALPH = 2  # sin/cos-like basis per frequency
MOVMEAN_WIN = 10  # moving mean window on envelope before LASSO


def _movmean(x: np.ndarray, w: int, axis: int = 0) -> np.ndarray:
    """Moving mean along axis (MATLAB movmean). Uses uniform filter; mode='nearest' at edges."""
    if w <= 1 or x.size == 0:
        return np.asarray(x, dtype=np.float64).copy()
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(
        np.asarray(x, dtype=np.float64), size=w, axis=axis, mode="nearest"
    )


def _build_sinusoid_dictionary(
    t0: np.ndarray,
    wintime_sec: float,
    Fs_env: float,
    alph: int = LASSO_ALPH,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build design matrix A of sinusoids (MATLAB lassoHIE dictionary).

    a_freqs = (2/wintime) : (0.5/wintime) : (Fs_env/4) Hz.
    For each frequency, alph columns: sin(omega*t + 2*pi/alph*j), j=1..alph.

    Returns:
        A: (len(t0), alph * len(a_freqs))
        a_freqs: (n_freqs,) candidate modulation frequencies in Hz
        thetas: (alph,) phase shifts used (2*pi/alph, 4*pi/alph, ...)
    """
    f_min = 2.0 / wintime_sec
    f_step = 0.5 / wintime_sec
    f_max = Fs_env / 4.0
    if f_max <= f_min:
        a_freqs = np.array([f_min])
    else:
        a_freqs = np.arange(f_min, f_max + f_step * 0.5, f_step)
    thetas = (2 * np.pi / alph) * np.arange(1, alph + 1)
    n_freqs = len(a_freqs)
    A = np.zeros((len(t0), alph * n_freqs), dtype=np.float64)
    for k in range(n_freqs):
        omega = a_freqs[k] * 2 * np.pi
        for j in range(alph):
            A[:, k * alph + j] = np.sin(t0 * omega + thetas[j])
    return A, a_freqs, thetas


def take_entropy_lasso(x: np.ndarray) -> np.ndarray:
    """
    Entropy of strength distribution over frequencies (MATLAB take_entropy_lasso).

    x: (num_freqs, num_lambdas) or (num_freqs,) — absolute strengths.
    Returns: (num_lambdas,) or scalar — entropy per lambda, normalized by log(num_freqs).
    """
    x = np.abs(np.asarray(x, dtype=np.float64))
    if x.ndim == 1:
        x = x[:, np.newaxis]
    a, b = x.shape[0], x.shape[1]
    s = np.sum(x, axis=0, keepdims=True)
    s[s == 0] = 1
    prob_dist = x / s
    prob_dist = np.clip(prob_dist, 1e-15, 1.0)
    y = -np.nansum(prob_dist * np.log(prob_dist), axis=0) / np.log(a)
    return y[0] if y.size == 1 else y


@dataclass
class MOsResult:
    """Result of MOs detection for one time bucket."""
    timestamp: float
    bucket_length_seconds: float
    sample_rate: float
    q_per_band: Dict[str, float]  # band label -> q (max over windows, for p-value)
    p_per_band: Dict[str, float]  # band label -> p-value vs surrogate (min over windows)
    q_per_window_per_band: Dict[str, np.ndarray]  # band label -> (n_windows,) q per window
    p_per_window_per_band: Dict[str, np.ndarray]  # band label -> (n_windows,) p-value per window
    dominant_freq_hz_per_window_per_band: Dict[str, np.ndarray]  # band label -> (n_windows,) dominant modulation freq in Hz
    dominant_freq_hz_per_band: Dict[str, float]  # band label -> dominant freq at window where q is max (Hz, NaN if none)
    n_surrogates: int
    channel_index: int = 0


def apply_bipolar_montage(
    eeg: np.ndarray,
    bpm_mask: np.ndarray,
) -> np.ndarray:
    """
    Compute bipolar EEG from montage definition.

    Args:
        eeg: (channels, samples)
        bpm_mask: (num_bipolar, 2) each row [i, j] -> channel i - channel j

    Returns:
        d1: (num_bipolar, samples)
    """
    return eeg[bpm_mask[:, 0], :] - eeg[bpm_mask[:, 1], :]


def _dpss_windows(n: int, nw: float, k: int) -> np.ndarray:
    """Return K DPSS (Slepian) tapers of length n, time-bandwidth nw. Periodic for spectral analysis."""
    return dpss(n, nw, Kmax=k, sym=False)


def multitaper_spectrogram(
    data: np.ndarray,
    Fs: float,
    window_sec: float = SPECTROGRAM_WINDOW_SEC,
    step_sec: float = SPECTROGRAM_STEP_SEC,
    nw: float = TAPERS_NW,
    k: int = TAPERS_K,
    fpass: Tuple[float, float] = FPASS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Multi-taper spectrogram (Chronux-style).

    Args:
        data: (channels, samples)
        Fs: sampling rate Hz
        window_sec: window length in seconds
        step_sec: step between windows in seconds
        nw: time-bandwidth product
        k: number of tapers
        fpass: (f_low, f_high) Hz

    Returns:
        S: (n_times, n_freqs, n_channels) power
        T: (n_times,) time vector (center of each window)
        F: (n_freqs,) frequency vector
    """
    n_channels = data.shape[0]
    n_samples = data.shape[1]
    win_samples = int(round(window_sec * Fs))
    step_samples = int(round(step_sec * Fs))

    if win_samples > n_samples:
        win_samples = n_samples
        step_samples = max(1, n_samples // 4)

    tapers = _dpss_windows(win_samples, nw, k)
    n_fft = max(win_samples, 2 ** int(np.ceil(np.log2(win_samples))))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / Fs)
    f_low, f_high = fpass
    freq_mask = (freqs >= f_low) & (freqs <= f_high)
    F = freqs[freq_mask]
    n_freqs = len(F)

    starts = np.arange(0, n_samples - win_samples + 1, step_samples)
    n_times = len(starts)
    T = (starts + win_samples / 2) / Fs

    S = np.zeros((n_times, n_freqs, n_channels), dtype=np.float64)

    for ch in range(n_channels):
        x = np.asarray(data[ch, :], dtype=np.float64)
        x = x - np.mean(x)
        # Extract all segments at once: (n_times, win_samples)
        segments = np.array([x[start : start + win_samples] for start in starts])
        # Apply all tapers to all segments: (n_times, k, win_samples)
        tapered = segments[:, np.newaxis, :] * tapers[np.newaxis, :, :]
        # Batch FFT across all times and tapers
        specs = np.fft.rfft(tapered, n=n_fft, axis=2)  # (n_times, k, n_fft//2+1)
        # Average power across tapers
        power_avg = np.mean(np.abs(specs) ** 2, axis=1)  # (n_times, n_fft//2+1)
        S[:, :, ch] = power_avg[:, freq_mask]

    return S, T, F


def phase_randomize_surrogate(
    data: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Phase-randomized surrogate: preserve magnitude spectrum, randomize phases.

    Args:
        data: (channels, samples) real-valued
        rng: random generator (for reproducibility)

    Returns:
        surrogate: (channels, samples) same shape
    """
    if rng is None:
        rng = np.random.default_rng()
    out = np.zeros_like(data)
    n = data.shape[1]
    for ch in range(data.shape[0]):
        x = np.asarray(data[ch, :], dtype=np.complex128)
        X = np.fft.fft(x)
        mag = np.abs(X)
        phase = np.angle(X)
        n_pos = (n + 1) // 2
        phase_new = phase.copy()
        phase_new[1:n_pos] = rng.uniform(-np.pi, np.pi, n_pos - 1)
        if n % 2 == 0:
            phase_new[n_pos] = phase_new[n_pos]
        phase_new[n_pos:] = -phase_new[1 : n - n_pos + 1][::-1]
        Y = mag * np.exp(1j * phase_new)
        out[ch, :] = np.fft.ifft(Y).real
    return out


def extract_band_envelope(
    S: np.ndarray,
    F: np.ndarray,
    f_low: float,
    f_high: float,
) -> np.ndarray:
    """
    Band-limited power envelope: mean power in [f_low, f_high] per time and channel.

    Args:
        S: (n_times, n_freqs, n_channels)
        F: (n_freqs,) frequency vector
        f_low, f_high: band bounds (Hz)

    Returns:
        env: (n_times, n_channels)
    """
    idx1 = np.searchsorted(F, f_low, side="right") # find the first frequency index greater than f_low
    idx2 = np.searchsorted(F, f_high, side="left") + 1 # find the first frequency index greater than or equal to f_high
    idx2 = min(idx2, S.shape[1]) # if idx2 is greater than the number of frequencies, set it to the number of frequencies
    if idx1 >= idx2:
        return np.mean(S, axis=1)
    return np.mean(S[:, idx1:idx2, :], axis=1) # average power across the frequency band (Equation 1 from MO.pdf)


def gesd_outlier_mask(
    x: np.ndarray,
    alpha: float = GESD_ALPHA,
    max_outliers: int = GESD_MAX_OUTLIERS,
) -> np.ndarray:
    """
    Generalized Extreme Studentized Deviate test; return boolean mask True = outlier.

    Args:
        x: 1D array
        alpha: significance level
        max_outliers: maximum number of outliers to consider

    Returns:
        mask: True where x is an outlier
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    if n < 3 or max_outliers >= n:
        return np.zeros(n, dtype=bool)

    mask = np.zeros(n, dtype=bool)
    remaining = np.arange(n)
    for _ in range(max_outliers):
        if remaining.size < 2:
            break
        vals = x[remaining]
        mean = np.mean(vals)
        std = np.std(vals, ddof=0)
        if std <= 0:
            break
        r = np.abs(vals - mean) / std
        i_max = np.argmax(r)
        # Critical value from t-distribution (NIST approximation)
        # lambda_i = t_{n-i-1, 1-alpha/(2(n-i))} * (n-i-1) / sqrt((n-i-1 + t^2)(n-i))
        i_removed = np.sum(mask)
        dof = n - i_removed - 1
        if dof < 1:
            break
        p_crit = alpha / (2 * (n - i_removed))
        t_crit = stats.t.ppf(1 - p_crit, dof)
        lambda_i = t_crit * (dof) / np.sqrt((dof + t_crit ** 2) * (n - i_removed))
        if r[i_max] <= lambda_i:
            break
        idx_out = remaining[i_max]
        mask[idx_out] = True
        remaining = np.delete(remaining, i_max)
    return mask


def interpolate_outliers(
    # TODO: improve documentation for this function
    # This function is used to interpolate over outliers in the envelope signal.
    # May need to add a parameter for the maximum number of outliers to interpolate over.
    #May also need a parameter for alpha value for the GESD test.
    env: np.ndarray,
    outlier_mask: np.ndarray,
) -> np.ndarray:
    """Replace outlier positions with linear interpolation. env can be 1D or (n_times, n_ch)."""
    env = np.asarray(env, dtype=np.float64)
    if env.ndim == 1:
        env = env[:, np.newaxis]
        squeeze = True
    else:
        squeeze = False
    out = env.copy()
    n_times, n_ch = env.shape
    if outlier_mask.ndim == 1:
        outlier_mask = outlier_mask[:, np.newaxis]
    for ch in range(n_ch):
        m = outlier_mask[:, ch] if outlier_mask.shape[1] > 1 else outlier_mask.ravel()
        if not np.any(m):
            continue
        x = np.arange(n_times)
        good = ~m
        if np.sum(good) < 2:
            continue
        out[m, ch] = np.interp(x[m], x[good], env[good, ch])
    if squeeze:
        out = out[:, 0]
    return out


def _lasso_mo_q_single_channel(
    envelope: np.ndarray,
    T: np.ndarray,
    Fs_env: float,
    wintime_sec: float = LASSO_WINTIME_SEC,
    winjump_sec: float = LASSO_WINJUMP_SEC,
    alph: int = LASSO_ALPH,
    lasso_path_n_alphas: int = 100,
    lasso_path_eps: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    LASSO-based MO strength (qm) and dominant modulation frequency per window (MATLAB lassoHIE).

    Uses full λ path via sklearn lasso_path (MATLAB lasso default behavior). Per window:
    scale envelope to [-1,1], demean, fit LASSO path; per λ compute J = corr(A*bb, s0),
    per-frequency polar strength, entropy; q = J*(1-entropy); qm = max(q). Dominant
    frequency = frequency of max-magnitude coefficient in β at the λ that maximizes q.

    Returns:
        qm_per_window: (n_windows,) q value per window
        dominant_freq_hz_per_window: (n_windows,) dominant modulation frequency in Hz (NaN if not computed)
    """
    n = len(envelope)
    win_samples = max(1, int(round(wintime_sec * Fs_env)))
    jump_samples = max(1, int(round(winjump_sec * Fs_env)))
    if win_samples > n:
        win_samples = n
        jump_samples = n
    num_funs = max(0, (n - win_samples) // jump_samples)
    starts = [i * jump_samples for i in range(num_funs)]
    if n >= win_samples and (not starts or starts[-1] != n - win_samples):
        starts.append(n - win_samples)
    qm_per_window = []
    dominant_freq_per_window = []
    if T.size < win_samples:
        t0 = np.arange(win_samples, dtype=np.float64) / Fs_env
    else:
        t0 = T[:win_samples].astype(np.float64)
    A, a_freqs, thetas = _build_sinusoid_dictionary(t0, wintime_sec, Fs_env, alph)
    n_freqs = len(a_freqs)

    for start in starts:
        end = start + win_samples
        s0 = envelope[start:end].astype(np.float64)
        rng_val = np.ptp(s0)
        if rng_val > 0:
            s0 = 2.0 * (s0 - np.min(s0)) / rng_val - 1.0
        s0 = s0 - np.mean(s0)
        if not HAS_SKLEARN or lasso_path is None or A.shape[1] == 0:
            qm_per_window.append(0.0)
            dominant_freq_per_window.append(np.nan)
            continue
        alphas, coefs, _ = lasso_path(
            A, s0, eps=lasso_path_eps, n_alphas=lasso_path_n_alphas
        )
        # coefs shape (n_features, n_alphas)
        n_alphas = coefs.shape[1]

        # Vectorized correlation: compute fitted for all lambdas at once
        fitted_all = A @ coefs  # (win_samples, n_alphas)
        std_fitted = np.std(fitted_all, axis=0)  # (n_alphas,)
        std_s0 = np.std(s0)
        if std_s0 > 0:
            # Pearson correlation: corr = cov(X,Y) / (std_X * std_Y)
            s0_centered = s0 - np.mean(s0)
            fitted_centered = fitted_all - np.mean(fitted_all, axis=0, keepdims=True)
            cov_vals = (fitted_centered.T @ s0_centered) / len(s0)  # (n_alphas,)
            denom = std_fitted * std_s0
            valid = denom > 0
            J_arr = np.zeros(n_alphas)
            J_arr[valid] = cov_vals[valid] / denom[valid]
            J_arr = np.nan_to_num(J_arr, nan=0.0)
        else:
            J_arr = np.zeros(n_alphas)

        # Vectorized polar magnitude: reshape coefs to (n_freqs, alph, n_alphas)
        cos_thetas = np.cos(thetas)  # (alph,)
        sin_thetas = np.sin(thetas)  # (alph,)
        coefs_reshaped = coefs.reshape(n_freqs, alph, n_alphas)  # (n_freqs, alph, n_alphas)
        # Sum of (weight * cos(theta)) and (weight * sin(theta)) across alph basis functions
        xy_x = np.einsum('fai,a->fi', coefs_reshaped, cos_thetas)  # (n_freqs, n_alphas)
        xy_y = np.einsum('fai,a->fi', coefs_reshaped, sin_thetas)  # (n_freqs, n_alphas)
        f_hold = np.sqrt(xy_x ** 2 + xy_y ** 2)  # (n_freqs, n_alphas)

        q_entropy = take_entropy_lasso(f_hold)
        q = J_arr * (1.0 - q_entropy)
        qm = float(np.max(q)) if q.size else 0.0
        qi = int(np.argmax(q)) if q.size else 0
        qm_per_window.append(qm)
        # Dominant frequency = frequency of max-magnitude coefficient at chosen λ
        f_hold_best = f_hold[:, qi] if qi < n_alphas else np.zeros(n_freqs)
        if n_freqs > 0 and np.any(f_hold_best > 0):
            dom_idx = int(np.argmax(f_hold_best))
            dominant_freq_per_window.append(float(a_freqs[dom_idx]))
        else:
            dominant_freq_per_window.append(np.nan)
    return (
        np.array(qm_per_window, dtype=np.float64),
        np.array(dominant_freq_per_window, dtype=np.float64),
    )


def lasso_mo_q(
    env: np.ndarray,
    T: np.ndarray,
    wintime_sec: float = LASSO_WINTIME_SEC,
    winjump_sec: float = LASSO_WINJUMP_SEC,
) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """
    LASSO-based MO strength (MATLAB lassoHIE): qm and dominant modulation freq per window per channel.

    Uses full λ path (no λ₂). No temporal or spatial priors.

    Args:
        env: (n_times, n_channels) band-limited envelope (after GESD/interp)
        T: (n_times,) time vector
        wintime_sec, winjump_sec: LASSO window and step

    Returns:
        q_aggregated: (n_channels,) max over windows (for p-value)
        q_per_window_per_ch: list of length n_channels, each (n_windows,)
        dominant_freq_hz_per_window_per_ch: list of length n_channels, each (n_windows,) in Hz (NaN if not computed)
    """
    if T.size < 2:
        Fs_env = 1.0
    else:
        Fs_env = 1.0 / (T[1] - T[0])
    env_smooth = _movmean(env, MOVMEAN_WIN, axis=0)
    n_ch = env_smooth.shape[1]
    q_aggregated = np.zeros(n_ch)
    q_per_window_per_ch = []
    dominant_freq_per_window_per_ch = []
    for ch in range(n_ch):
        qm_w, dom_freq_w = _lasso_mo_q_single_channel(
            env_smooth[:, ch], T, Fs_env, wintime_sec, winjump_sec
        )
        q_per_window_per_ch.append(qm_w)
        dominant_freq_per_window_per_ch.append(dom_freq_w)
        q_aggregated[ch] = float(np.max(qm_w)) if qm_w.size else 0.0
    return q_aggregated, q_per_window_per_ch, dominant_freq_per_window_per_ch


def _compute_one_surrogate(args: Tuple) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Process one surrogate: phase-randomize, compute spectrogram. Returns (S, T, F)."""
    eeg, Fs, rng_seed = args
    rng = np.random.default_rng(rng_seed)
    surr = phase_randomize_surrogate(eeg, rng=rng)
    Ss, Ts, Fs_arr = multitaper_spectrogram(
        surr, Fs,
        window_sec=SPECTROGRAM_WINDOW_SEC,
        step_sec=SPECTROGRAM_STEP_SEC,
        nw=TAPERS_NW,
        k=TAPERS_K,
        fpass=FPASS,
    )
    return Ss, Ts, Fs_arr


def compute_mos_for_bucket(
    eeg_bucket: np.ndarray,
    Fs: float,
    timestamp: float = 0.0,
    bucket_length_seconds: Optional[float] = None,
    bpm_mask: Optional[np.ndarray] = None,
    freq_bands: Optional[List[Tuple[float, float]]] = None,
    n_surrogates: int = 1,
    rng: Optional[np.random.Generator] = None,
    channel_index: int = 0,
    channel_indices: Optional[List[int]] = None,
    wintime_sec: float = LASSO_WINTIME_SEC,
    winjump_sec: float = LASSO_WINJUMP_SEC,
    n_workers: Optional[int] = None,
) -> Union[MOsResult, List[MOsResult]]:
    """
    Run full MOs pipeline on a single time bucket.

    Args:
        eeg_bucket: (channels, samples) EEG for one bucket (e.g. 2 or 5 min)
        Fs: sampling rate Hz
        timestamp: optional timestamp for the bucket
        bucket_length_seconds: optional, for logging
        bpm_mask: optional (n_bipolar, 2) bipolar montage; if None, use eeg as-is
        freq_bands: list of (f_low, f_high); default MO_FREQ_BANDS
        n_surrogates: number of phase-randomized surrogates for p-value
        rng: random generator
        channel_index: which channel index to process (single channel, backward compat)
        channel_indices: list of channel indices to process. Overrides channel_index when
            provided. Surrogates are shared across all channels for efficiency.
        wintime_sec: LASSO envelope window length (s). Use 120 for 2 min, 300 for 5 min.
        winjump_sec: LASSO window step (s). Typically 1/4 * wintime_sec.
        n_workers: number of parallel workers for surrogate computation. None = serial (1).

    Returns:
        MOsResult if single channel, List[MOsResult] if multiple channels via channel_indices.
    """
    if rng is None:
        rng = np.random.default_rng()
    if freq_bands is None:
        freq_bands = MO_FREQ_BANDS
    if eeg_bucket.ndim == 1:
        eeg_bucket = eeg_bucket[np.newaxis, :]
    if bpm_mask is not None:
        eeg = apply_bipolar_montage(eeg_bucket, bpm_mask)
    else:
        eeg = np.asarray(eeg_bucket, dtype=np.float64)

    # Resolve which channels to process
    if channel_indices is not None:
        ch_list = [min(ci, eeg.shape[0] - 1) for ci in channel_indices]
    else:
        ch_list = [min(channel_index, eeg.shape[0] - 1)]
    multi_channel = len(ch_list) > 1

    eeg = eeg[ch_list, :]  # (len(ch_list), samples)
    n_ch = eeg.shape[0]
    if bucket_length_seconds is None:
        bucket_length_seconds = eeg.shape[1] / Fs

    # Real spectrogram
    S, T, F = multitaper_spectrogram(
        eeg, Fs,
        window_sec=SPECTROGRAM_WINDOW_SEC,
        step_sec=SPECTROGRAM_STEP_SEC,
        nw=TAPERS_NW,
        k=TAPERS_K,
        fpass=FPASS,
    )
    Fs_env = 1.0 / (T[1] - T[0]) if T.size > 1 else 1.0

    # Pre-compute surrogate spectrograms once, reuse across all bands and channels
    surr_seeds = [int(rng.integers(0, 2**63)) for _ in range(n_surrogates)]
    if n_workers is not None and n_workers > 1 and n_surrogates > 1:
        surr_args = [(eeg, Fs, seed) for seed in surr_seeds]
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            surr_spectrograms = list(pool.map(_compute_one_surrogate, surr_args))
    else:
        surr_spectrograms: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for seed in surr_seeds:
            surr_spectrograms.append(_compute_one_surrogate((eeg, Fs, seed)))

    # Per-channel result accumulators
    per_ch_q_per_band: List[Dict[str, float]] = [{} for _ in range(n_ch)]
    per_ch_p_per_band: List[Dict[str, float]] = [{} for _ in range(n_ch)]
    per_ch_q_per_window_per_band: List[Dict[str, np.ndarray]] = [{} for _ in range(n_ch)]
    per_ch_p_per_window_per_band: List[Dict[str, np.ndarray]] = [{} for _ in range(n_ch)]
    per_ch_dom_freq_per_window_per_band: List[Dict[str, np.ndarray]] = [{} for _ in range(n_ch)]
    per_ch_dom_freq_per_band: List[Dict[str, float]] = [{} for _ in range(n_ch)]

    for (f_low, f_high) in freq_bands:
        band_label = f"{f_low}_{f_high}hz"
        env = extract_band_envelope(S, F, f_low, f_high)
        for ch in range(n_ch):
            mask = gesd_outlier_mask(env[:, ch], alpha=GESD_ALPHA, max_outliers=GESD_MAX_OUTLIERS)
            env[:, ch] = interpolate_outliers(env[:, ch], mask)
        q_real, q_per_window_per_ch, dom_freq_per_window_per_ch = lasso_mo_q(
            env, T, wintime_sec=wintime_sec, winjump_sec=winjump_sec
        )
        # Surrogate per-window q-values (for temporal null distribution)
        # Each entry: list of n_ch arrays, each (n_windows,)
        q_surr_per_window_list = []
        for (Ss, Ts, Fs_arr) in surr_spectrograms:
            env_surr = extract_band_envelope(Ss, Fs_arr, f_low, f_high)
            for ch in range(n_ch):
                mask = gesd_outlier_mask(env_surr[:, ch], alpha=GESD_ALPHA, max_outliers=GESD_MAX_OUTLIERS)
                env_surr[:, ch] = interpolate_outliers(env_surr[:, ch], mask)
            _, q_per_window_surr, _ = lasso_mo_q(
                env_surr, Ts, wintime_sec=wintime_sec, winjump_sec=winjump_sec
            )
            q_surr_per_window_list.append(q_per_window_surr)

        for ch in range(n_ch):
            per_ch_q_per_window_per_band[ch][band_label] = q_per_window_per_ch[ch]
            dom_freq_w = dom_freq_per_window_per_ch[ch]
            per_ch_dom_freq_per_window_per_band[ch][band_label] = dom_freq_w
            q_w = q_per_window_per_ch[ch]
            if q_w.size > 0 and np.any(np.isfinite(q_w)):
                idx_max = int(np.argmax(q_w))
                per_ch_dom_freq_per_band[ch][band_label] = float(dom_freq_w[idx_max])
            else:
                per_ch_dom_freq_per_band[ch][band_label] = np.nan

            # Pool all surrogate per-window q-values for this channel (temporal null)
            all_surr_q = np.concatenate([s[ch] for s in q_surr_per_window_list])
            mu0 = float(np.mean(all_surr_q))
            sigma0 = float(np.std(all_surr_q, ddof=0))  # ddof=0 matches MATLAB std(x,1)
            if sigma0 <= 0:
                sigma0 = 1e-10

            # Per-window p-values (matching MATLAB: normcdf per window)
            p_windows = 1.0 - stats.norm.cdf(q_w, loc=mu0, scale=sigma0)
            p_windows = np.clip(p_windows, 0.0, 1.0)
            per_ch_p_per_window_per_band[ch][band_label] = p_windows

            # Aggregate p and q (backward compat: min p across windows, max q)
            per_ch_p_per_band[ch][band_label] = float(np.min(p_windows)) if p_windows.size else 0.5
            per_ch_q_per_band[ch][band_label] = float(np.max(q_w)) if q_w.size else 0.0

    # Build MOsResult per channel
    results = []
    for i, ch_idx in enumerate(ch_list):
        results.append(MOsResult(
            timestamp=timestamp,
            bucket_length_seconds=bucket_length_seconds,
            sample_rate=Fs,
            q_per_band=per_ch_q_per_band[i],
            p_per_band=per_ch_p_per_band[i],
            q_per_window_per_band=per_ch_q_per_window_per_band[i],
            p_per_window_per_band=per_ch_p_per_window_per_band[i],
            dominant_freq_hz_per_window_per_band=per_ch_dom_freq_per_window_per_band[i],
            dominant_freq_hz_per_band=per_ch_dom_freq_per_band[i],
            n_surrogates=n_surrogates,
            channel_index=ch_idx,
        ))

    if multi_channel:
        return results
    return results[0]
