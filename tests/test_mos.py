"""Tests for Modulatory Oscillations (MOs) detection pipeline."""

import numpy as np
import pytest

try:
    from src.processing.mos import (
        apply_bipolar_montage,
        multitaper_spectrogram,
        phase_randomize_surrogate,
        extract_band_envelope,
        gesd_outlier_mask,
        interpolate_outliers,
        take_entropy_lasso,
        lasso_mo_q,
        compute_mos_for_bucket,
        MO_FREQ_BANDS,
        MOsResult,
    )
except ImportError:
    from processing.mos import (
        apply_bipolar_montage,
        multitaper_spectrogram,
        phase_randomize_surrogate,
        extract_band_envelope,
        gesd_outlier_mask,
        interpolate_outliers,
        take_entropy_lasso,
        lasso_mo_q,
        compute_mos_for_bucket,
        MO_FREQ_BANDS,
        MOsResult,
    )


def test_bipolar_montage():
    eeg = np.random.randn(4, 100)
    bpm_mask = np.array([[0, 1], [2, 3]])
    d1 = apply_bipolar_montage(eeg, bpm_mask)
    assert d1.shape == (2, 100)
    np.testing.assert_allclose(d1[0], eeg[0] - eeg[1])
    np.testing.assert_allclose(d1[1], eeg[2] - eeg[3])


def test_multitaper_spectrogram():
    Fs = 250.0
    data = np.random.randn(1, 5000).astype(np.float64) * 0.5
    S, T, F = multitaper_spectrogram(data, Fs, window_sec=2, step_sec=0.5)
    assert S.ndim == 3
    assert S.shape[1] == len(F)
    assert S.shape[2] == 1
    assert len(T) == S.shape[0]
    assert F[0] >= 0 and F[-1] <= 100


def test_phase_randomize_surrogate():
    data = np.random.randn(2, 1000).astype(np.float64)
    rng = np.random.default_rng(42)
    surr = phase_randomize_surrogate(data, rng=rng)
    assert surr.shape == data.shape
    assert np.all(np.isreal(surr))
    np.testing.assert_allclose(np.abs(np.fft.fft(surr[0])), np.abs(np.fft.fft(data[0])), atol=1e-5)


def test_extract_band_envelope():
    n_t, n_f, n_ch = 10, 50, 2
    S = np.random.rand(n_t, n_f, n_ch)
    F = np.linspace(0, 100, n_f)
    env = extract_band_envelope(S, F, 8, 15)
    assert env.shape == (n_t, n_ch)


def test_gesd_outlier_mask():
    x = np.array([1.0, 2.0, 2.1, 2.0, 1.9, 2.2, 10.0, 2.0])
    mask = gesd_outlier_mask(x, alpha=0.05, max_outliers=2)
    assert np.sum(mask) >= 1
    assert mask[6]


def test_interpolate_outliers():
    env = np.array([1.0, 2.0, 100.0, 2.0, 3.0])
    mask = np.array([False, False, True, False, False])
    out = interpolate_outliers(env, mask)
    assert out[2] != 100.0
    assert 1.0 <= out[2] <= 3.0


def test_take_entropy_lasso():
    x = np.abs(np.random.rand(5, 3))
    y = take_entropy_lasso(x)
    assert y.shape == (3,)
    assert np.all(y >= 0) and np.all(y <= 1)


def test_lasso_mo_q_returns_per_window():
    n_times, n_ch = 60, 2
    T = np.arange(n_times, dtype=np.float64) * 6.0
    env = np.random.rand(n_times, n_ch).astype(np.float64)
    q_agg, q_per_window, dom_freq_per_window = lasso_mo_q(
        env, T, wintime_sec=120, winjump_sec=30
    )
    assert q_agg.shape == (n_ch,)
    assert len(q_per_window) == n_ch
    assert len(dom_freq_per_window) == n_ch
    for ch in range(n_ch):
        assert q_per_window[ch].ndim == 1
        assert np.max(q_per_window[ch]) == q_agg[ch]
        assert dom_freq_per_window[ch].ndim == 1
        assert dom_freq_per_window[ch].shape == q_per_window[ch].shape


def test_compute_mos_for_bucket_smoke():
    #Bucket smoke is a test that the function runs without errors.
    #It does not check the results.
    Fs = 250.0
    bucket_sec = 30
    n_samples = int(bucket_sec * Fs)
    eeg = np.random.randn(2, n_samples).astype(np.float64) * 0.5
    result = compute_mos_for_bucket(
        eeg, Fs, timestamp=0.0, n_surrogates=2, channel_index=0
    )
    assert result.q_per_band
    assert result.p_per_band
    assert result.q_per_window_per_band
    assert result.p_per_window_per_band
    assert result.dominant_freq_hz_per_window_per_band
    assert result.dominant_freq_hz_per_band
    for band in result.q_per_band:
        assert band in result.p_per_band
        assert band in result.q_per_window_per_band
        assert result.q_per_window_per_band[band].ndim == 1
        assert band in result.p_per_window_per_band
        assert result.p_per_window_per_band[band].ndim == 1
        assert result.p_per_window_per_band[band].shape == result.q_per_window_per_band[band].shape
        assert np.all(result.p_per_window_per_band[band] >= 0.0)
        assert np.all(result.p_per_window_per_band[band] <= 1.0)
        assert band in result.dominant_freq_hz_per_window_per_band
        assert result.dominant_freq_hz_per_window_per_band[band].ndim == 1
        assert result.dominant_freq_hz_per_window_per_band[band].shape == result.q_per_window_per_band[band].shape
        assert band in result.dominant_freq_hz_per_band
    assert result.bucket_length_seconds == bucket_sec
    assert result.sample_rate == Fs


def test_compute_mos_for_bucket_multi_channel():
    """Multi-channel path returns List[MOsResult] with correct channel_index fields."""
    Fs = 250.0
    bucket_sec = 30
    n_samples = int(bucket_sec * Fs)
    eeg = np.random.randn(4, n_samples).astype(np.float64) * 0.5
    results = compute_mos_for_bucket(
        eeg, Fs, timestamp=0.0, n_surrogates=2, channel_indices=[0, 1]
    )
    assert isinstance(results, list)
    assert len(results) == 2
    for i, r in enumerate(results):
        assert isinstance(r, MOsResult)
        assert r.channel_index == i
        assert r.q_per_band
        assert r.p_per_band
        assert r.p_per_window_per_band
        for band in r.q_per_band:
            assert band in r.p_per_band
            assert band in r.q_per_window_per_band
            assert r.q_per_window_per_band[band].ndim == 1
            assert band in r.p_per_window_per_band
            assert r.p_per_window_per_band[band].ndim == 1
            assert r.p_per_window_per_band[band].shape == r.q_per_window_per_band[band].shape
            assert np.all(r.p_per_window_per_band[band] >= 0.0)
            assert np.all(r.p_per_window_per_band[band] <= 1.0)
