# Modulatory Oscillations (MOs) Algorithm: Build Documentation

This document describes the steps taken to build the MOs (Modulatory Oscillations) detection pipeline in Python, including the LASSO-based modulation-strength (q) computation. The implementation replicates the original MATLAB pipeline and is designed to run on **time buckets** from a ring buffer (e.g. 2 or 5 minutes) and to produce **one q-value per window** per band, with bucket-level q obtained by **max over windows** (not mean).

---

## 1. Design Goals

The pipeline was built with these requirements (from the Python replication spec):

- **Time-bucket input**: MOs detection runs on a single time bucket (e.g. 2 or 5 minutes) received from the ring buffer, not on a full recording.
- **Bucket length**: Configurable (e.g. 120 s or 300 s).
- **Integration**: MOs detection is an EEG feature alongside PSD and band-limited power; results are stored for downstream use and finetuning.
- **q-value storage**: All q-values (per window and per bucket) are stored; per-window q supports analysis and algorithm tuning.

References: `documentation/MOs_Algo.md`, `src/processing/MOs_Algo.md`.

---

## 2. Pipeline Overview

The MOs pipeline consists of:

1. **Bipolar montage** (optional)
2. **Multi-taper spectrogram** (real data)
3. **Phase-randomized surrogate** generation
4. **Surrogate spectrogram**
5. **Per frequency band**:
   - Band-limited envelope extraction
   - Outlier removal (GESD) and linear interpolation
   - **LASSO-based MO detection** (lassoHIE) → q per window, then max over windows
   - Surrogate LASSO → distribution of q
   - **Statistical significance**: p-value from real q vs surrogate (Gaussian)

Implementation: `src/processing/mos.py`; entry point `compute_mos_for_bucket()`.

---

## 3. Step-by-Step Pipeline

### 3.1 Bipolar Montage

- **Input**: EEG `(channels, samples)`; optional `bpm_mask` of shape `(num_bipolar, 2)`.
- **Operation**: For each bipolar pair `(i, j)`, compute `d1[k, :] = EEG[i, :] - EEG[j, :]`.
- **Output**: Bipolar EEG `(num_bipolar, samples)`; if no mask, raw EEG is used.

```python
d1 = eeg[bpm_mask[:, 0], :] - eeg[bpm_mask[:, 1], :]
```

---

### 3.2 Multi-Taper Spectrogram (Chronux-Style)

- **Parameters**: Time–bandwidth product NW = 3, number of tapers K = 5; frequency range 0–100 Hz; window 30 s, step 6 s.
- **Method**: DPSS (Slepian) tapers via `scipy.signal.windows.dpss`; for each window, taper the segment, FFT, average power across tapers.
- **Output**: `S` (n_times × n_freqs × n_channels), time vector `T`, frequency vector `F`.
- **Envelope sampling rate**: `Fs_env = 1 / (T[1] - T[0])` (e.g. 1/6 Hz when step = 6 s).

---

### 3.3 Phase-Randomized Surrogate

- **Purpose**: Generate null distribution for significance testing.
- **Steps**: FFT → keep magnitude → randomize phase of positive frequencies → enforce conjugate symmetry → inverse FFT.
- **Output**: Surrogate time series same shape as input; multiple surrogates (e.g. 50) are used per band.

---

### 3.4 Band-Limited Envelope

- **Input**: Spectrogram `S`, frequency vector `F`, band `[f_low, f_high]` (e.g. 0.5–3, 3–8, 8–15, 15–30 Hz).
- **Operation**: For each time and channel, average power over frequency bins in the band:  
  `env[t, ch] = mean(S[t, freqidx1:freqidx2, ch])`.
- **Output**: Envelope `(n_times, n_channels)`.

---

### 3.5 Outlier Removal (GESD) and Interpolation

- **Method**: Generalized Extreme Studentized Deviate (GESD) test; α = 0.05, max outliers = 10.
- **Operation**: Identify outlier time points per channel; replace with **linear interpolation** (equivalent in spirit to MATLAB `filloutliers(env(:,j), 'linear', 'gesd')`).
- **Output**: Cleaned envelope, same shape.

---

### 3.6 Statistical Significance (Per Band)

- **Real q**: One scalar per channel per band = **max over windows** of the per-window q (see LASSO section).
- **Surrogate q**: Same LASSO procedure on each surrogate envelope → distribution of q per channel.
- **p-value**: `p = 1 - norm.cdf(q_real, mu0, sigma0)` with `mu0 = mean(q_surr)`, `sigma0 = std(q_surr, ddof=0)`.

---

## 4. LASSO Algorithm (lassoHIE)

The LASSO step follows the MATLAB function `lassoHIE` (see `documentation/Lasso_Matlab_Code.md`). It produces a **modulation strength (q)** per window; q is then **max over λ** per window, and **max over windows** per bucket.

### 4.1 Parameters

| Parameter    | Default | Description |
|-------------|---------|-------------|
| `wintime`   | 120 s   | LASSO window length |
| `winjump`   | 30 s    | Step between windows (wintime/4) |
| `winsize`   | round(wintime × Fs_env) | Window length in envelope samples |
| `lambda2`   | 2^0..2^4 × 1e-2 | LASSO regularization grid |
| `alph`      | 2       | Basis functions per frequency (sin, −sin) |
| `movmean`   | 10      | Moving-average window on envelope before LASSO |

### 4.2 Envelope Preprocessing

1. **Smoothing**: `s = movmean(env, 10, axis=0)` along time (MATLAB `movmean(env, 10, 1)`).
2. **Per window**: For the segment `s0 = s[start:start+winsize, ch]`:
   - Scale to [−1, 1]: `s0 = 2*(s0 - min(s0))/range(s0) - 1` (if range > 0).
   - Demean: `s0 = s0 - mean(s0)`.

### 4.3 Sinusoid Dictionary (Design Matrix A)

- **Candidate modulation frequencies** (Hz):  
  `a_freqs = (2/wintime) : (0.5/wintime) : (Fs_env/4)`  
  So the dictionary spans from 2/wintime to Fs_env/4 in steps of 0.5/wintime.
- **Time vector**: First window’s time steps `t0 = T[0:winsize]` (same A used for all windows).
- **Basis**: For each frequency `f` in `a_freqs`, `ω = 2πf`, and for `j = 1, 2`:  
  column = `sin(ω*t0 + 2π/alph*j)` → with `alph = 2` this gives `sin(ωt + π)` and `sin(ωt + 2π)` (i.e. −sin(ωt) and sin(ωt)).
- **Matrix**: `A` has shape `(winsize, alph × len(a_freqs))`; each pair of columns is the sin/−sin basis for one modulation frequency.
- **Phase vector**: `thetas = [2π/alph, 2π]` = [π, 2π] for converting coefficients to polar strength later.

### 4.4 Window Schedule

- **Number of windows**: `num_funs = floor((length(env) - winsize) / winjump)`.
- **Window starts**: `0, winjump, 2*winjump, ..., (num_funs-1)*winjump`, plus a **last window** at `length(env) - winsize` (aligned to end of envelope).
- So there are **num_funs + 1** windows per channel per band.

### 4.5 Per-Window LASSO and q

For each window and channel:

1. **LASSO**: Fit `s0 ≈ A @ b` for each λ in `lambda2` (e.g. sklearn `Lasso(alpha=lam, fit_intercept=False)`).  
   Obtain coefficient vector `bb` (length = number of columns of A) for each λ.

2. **Fit quality (J)**: For each λ,  
   `J[λ] = corr(A @ bb[λ], s0)`  
   (Pearson correlation between fitted and observed envelope).

3. **Per-frequency strength (polar magnitude)**:
   - For each λ and each frequency index `fr`:
     - Indices for that freq: `v = [alph*fr, alph*fr+1]` (the two coefficients).
     - `fitted_weights = bb[v]`.
     - `grid_coords = [weights * cos(thetas), weights * sin(thetas)]` (2×2).
     - `xy_vec = sum(grid_coords)` (over rows) → 2-vector.
     - **Polar magnitude** = `sqrt(sum(xy_vec.^2))` → stored as `f_hold[fr, λ]`.
   - This measures how strongly that modulation frequency is present in the fit for that λ.

4. **Entropy over frequencies**:  
   `q_entropy[λ] = take_entropy_lasso(f_hold[:, λ])`.  
   **take_entropy_lasso**: `x = abs(x)`, `prob = x / sum(x)`,  
   `y = -sum(prob * log(prob)) / log(num_freqs)`  
   (entropy of the strength distribution over candidate frequencies, normalized).

5. **q per λ**:  
   `q[λ] = J[λ] * (1 - q_entropy[λ])`.  
   High correlation and low entropy (one dominant modulation frequency) give high q.

6. **qm for this window**:  
   `qm_window = max(q)` over λ.  
   So **q is chosen by max over λ**, not mean.

### 4.6 Aggregation Over Windows and Bucket

- **Per window**: Each window yields one `qm_window` per channel (and per band).
- **Per channel**: `q_per_window[ch]` = array of length (num_funs + 1).
- **Bucket-level q (for p-value)**:  
  `q_bucket[ch] = max(q_per_window[ch])`.  
  So the scalar used for significance is the **max over windows**, not the mean.

### 4.7 Outputs

- **q_per_window_per_band**: For each band, array of q-values, one per window (for the selected channel in the API).
- **q_per_band**: For each band, one scalar = `max(q_per_window)` for that channel (used for p-value and reporting).
- **p_per_band**: p-value from real q vs surrogate q distribution (Gaussian CDF).

---

## 5. Implementation Summary

| Component            | Module / function              | Notes |
|---------------------|---------------------------------|-------|
| Bipolar montage     | `apply_bipolar_montage()`      | Optional; mask shape (n_bipolar, 2). |
| Spectrogram         | `multitaper_spectrogram()`     | DPSS, 30 s / 6 s, 0–100 Hz. |
| Surrogate           | `phase_randomize_surrogate()`  | FFT, random phase, IFFT. |
| Envelope            | `extract_band_envelope()`      | Mean power in band. |
| GESD + interpolate   | `gesd_outlier_mask()`, `interpolate_outliers()` | Per channel. |
| Dictionary          | `_build_sinusoid_dictionary()` | A, a_freqs, thetas. |
| Entropy             | `take_entropy_lasso()`         | Over frequencies per λ. |
| LASSO per channel   | `_lasso_mo_q_single_channel()`| qm per window. |
| LASSO all channels  | `lasso_mo_q()`                 | Returns (q_aggregated, q_per_window_per_ch). |
| Full bucket         | `compute_mos_for_bucket()`     | Returns `MOsResult`. |

### 5.1 MOsResult

- `timestamp`, `bucket_length_seconds`, `sample_rate`
- `q_per_band`: band → scalar q (max over windows)
- `p_per_band`: band → p-value
- `q_per_window_per_band`: band → 1D array of q per window
- `n_surrogates`, `channel_index`

### 5.2 Integration

- **Processor** (`src/processing/processor.py`): Optional `enable_mos`, `mo_bucket_seconds`. When enabled and the ring buffer has at least `mo_bucket_seconds` of data, the worker runs `compute_mos_for_bucket()` and stores the result (and history) in shared state (`mo_result`, `mo_history`), including `q_per_window_per_band` for finetuning and analysis. The number of surrogates is hardcoded to 1.

---

## 6. Critical Replication Details

- **Bipolar montage**: Indices must match MATLAB if comparing to original data.
- **Phase randomization**: Conjugate symmetry preserved for real surrogates.
- **Envelope rate**: `Fs_env` from spectrogram time step (e.g. 1/6 Hz).
- **Outliers**: GESD + linear interpolation (not z-score clipping).
- **Surrogate stats**: Gaussian q; `std(..., ddof=0)` to match MATLAB `std(X, 1)`.
- **q selection**: **Max over λ** per window; **max over windows** per bucket (not mean).

---

## 7. References

- **Pipeline spec**: `documentation/MOs_Algo.md`, `src/processing/MOs_Algo.md`
- **LASSO MATLAB code**: `documentation/Lasso_Matlab_Code.md`
- **Implementation**: `src/processing/mos.py`
- **Tests**: `tests/test_mos.py`
