"""Tests for spectrogram storage service."""

import shutil
import numpy as np
import pytest

try:
    from src.services.spectrogram_service import (
        save_window_spectrogram,
        load_window_spectrogram,
        load_full_spectrogram,
        list_available_channels,
        DATA_ROOT,
    )
except ImportError:
    from services.spectrogram_service import (
        save_window_spectrogram,
        load_window_spectrogram,
        load_full_spectrogram,
        list_available_channels,
        DATA_ROOT,
    )

TEST_STUDY_ID = "__test_spec_svc__"


@pytest.fixture(autouse=True)
def cleanup():
    """Remove test spectrogram files after each test."""
    yield
    test_dir = DATA_ROOT / TEST_STUDY_ID
    if test_dir.exists():
        shutil.rmtree(test_dir)


def _make_spectrogram(n_times=10, n_freqs=50):
    S = np.random.rand(n_times, n_freqs)
    T = np.arange(n_times, dtype=np.float64) * 6.0
    F = np.linspace(0, 100, n_freqs)
    envs = {
        "0.5_3hz": np.random.rand(n_times),
        "3_8hz": np.random.rand(n_times),
    }
    return S, T, F, envs


def test_save_load_roundtrip():
    S, T, F, envs = _make_spectrogram()
    save_window_spectrogram(TEST_STUDY_ID, 0, 0, S, T, F, envs)
    loaded = load_window_spectrogram(TEST_STUDY_ID, 0, 0)
    assert loaded is not None
    np.testing.assert_allclose(loaded["S"], S, atol=1e-6)
    np.testing.assert_allclose(loaded["T"], T, atol=1e-6)
    np.testing.assert_allclose(loaded["F"], F, atol=1e-6)
    np.testing.assert_allclose(loaded["env_0.5_3hz"], envs["0.5_3hz"], atol=1e-6)
    np.testing.assert_allclose(loaded["env_3_8hz"], envs["3_8hz"], atol=1e-6)


def test_load_nonexistent():
    result = load_window_spectrogram(TEST_STUDY_ID, 99, 99)
    assert result is None


def test_multi_window_concatenation():
    n_times, n_freqs = 10, 50
    F = np.linspace(0, 100, n_freqs)
    for w in range(3):
        S = np.random.rand(n_times, n_freqs)
        T = np.arange(n_times, dtype=np.float64) * 6.0 + w * n_times * 6.0
        envs = {"0.5_3hz": np.ones(n_times) * (w + 1)}
        save_window_spectrogram(TEST_STUDY_ID, w, 0, S, T, F, envs)

    full = load_full_spectrogram(TEST_STUDY_ID, 0)
    assert full is not None
    assert full["S"].shape == (30, n_freqs)
    assert full["T"].shape == (30,)
    assert full["F"].shape == (n_freqs,)
    assert full["env_0.5_3hz"].shape == (30,)
    # Check concatenation order
    np.testing.assert_allclose(full["env_0.5_3hz"][:10], 1.0)
    np.testing.assert_allclose(full["env_0.5_3hz"][10:20], 2.0)
    np.testing.assert_allclose(full["env_0.5_3hz"][20:30], 3.0)


def test_list_available_channels():
    S, T, F, envs = _make_spectrogram()
    for ch in [0, 2, 5]:
        save_window_spectrogram(TEST_STUDY_ID, 0, ch, S, T, F, envs)
    channels = list_available_channels(TEST_STUDY_ID)
    assert channels == [0, 2, 5]


def test_list_channels_empty():
    channels = list_available_channels("__nonexistent__")
    assert channels == []
