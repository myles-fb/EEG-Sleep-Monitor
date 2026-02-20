**Python Replication Specification (from MATLAB)**

This document specifies, step by step, how to replicate an existing MATLAB pipeline for detecting **Modulatory Oscillations (MOs)** in EEG data using Python. It is written as a deterministic implementation guide suitable for an LLM or a human developer.

IMPORTANT: Modify this pipeline to have the following characteristics:
- Performs MOs detection to generate a q-value (likelihood that modulation is present) for a single time bucket received from the ring buffer, rather than a whole .edf recording
- The length of the time bucket will be a parameter -- in this pipeline, it was either 2 or 5 minutes
- Designed with compability in mind for the rest of this codebase: we want MOs detection to be an EEG feature like PSD or band-limited power
- Ultimately, I want to store all these q-values because, unlike PSD, MOs detection is a new and largely untested algorithm, and keeping results of the algorithm will help us finetune its performance.

---

## Goal

For each patient EEG recording:

1. Load EEG data (EDF or cached MAT)
2. Compute a **bipolar montage**
3. Compute a **multi-taper spectrogram** for:

   * real EEG
   * phase-randomized surrogate EEG
4. For each frequency band:

   * extract a **band-limited power envelope**
   * remove outliers
   * run **LASSO-based MO detection**
   * compute **statistical significance** using surrogate distributions

All results are saved per patient and per frequency band.

---

## Directory Structure

```
datadir/
  ├── 1.edf
  ├── 2.edf
  ├── ...
  └── 10.edf

savedir/
  └── Patient X/
      ├── spectrograms
      ├── envelopes
      ├── lasso outputs
      └── p-values

codedir/
  ├── montage definitions
  ├── LASSO routines
  └── spectrogram utilities
```

---

## Inputs

### Patients

```python
ptlist = ["1","2","3","4","5","6","7","8","9","10"]
```

### Frequency Bands (Envelope Extraction)

```python
freq_list1 = [0.5, 3, 8, 15]   # lower bounds (Hz)
freq_list2 = [3,   8, 15, 30]  # upper bounds (Hz)
```

Bands are paired by index:

* 0.5–3 Hz
* 3–8 Hz
* 8–15 Hz
* 15–30 Hz

---

## Data Format Assumptions

* EEG array shape: `(channels, samples)`
* Sampling rate: `Fs_raw = header.frequency[0]`
* Bipolar montage defined by an integer array:

```python
bpmMask.shape == (num_bipolar_channels, 2)
```

Each row defines a channel subtraction:

```python
d1[i, :] = EEG[bpmMask[i,0], :] - EEG[bpmMask[i,1], :]
```

---

## Processing Pipeline

---

## 1. Load EEG Data

For each patient:

* If `{pt}.mat` exists:

  * Load `EEGData` and `header`
* Else:

  * Read `{pt}.edf`
  * Save `{pt}.mat` for caching

Ensure orientation:

```python
EEGData.shape == (channels, samples)
```

---

## 2. Bipolar Montage

* Load or define `bpmMask` once
* Compute bipolar EEG:

```python
d1 = EEGData[bpmMask[:,0], :] - EEGData[bpmMask[:,1], :]
```

---

## 3. Spectrogram Parameters (Chronux-style)

```python
tapers = (3, 5)      # time-bandwidth product, number of tapers
fpass = (0, 100)     # Hz
window = 30          # seconds
step = 6             # seconds
Fs = Fs_raw
```

---

## 4. Real Data Spectrogram

For each bipolar channel:

1. Demean signal
2. Compute multi-taper spectrogram using sliding windows
3. Store:

   * `S[t, f, ch]` → power
   * `T[t]` → time vector
   * `F[f]` → frequency vector

Save:

```
{pt}_spec_real_0_100
```

---

## 5. Surrogate Data Generation (Phase Randomization)

For each channel:

1. FFT along time axis
2. Preserve magnitude spectrum
3. Randomize phase of positive frequencies
4. Enforce conjugate symmetry
5. Inverse FFT → real-valued surrogate

```python
noise_series.shape == d1.shape
```

Save surrogate time series.

---

## 6. Surrogate Spectrogram

Repeat the exact spectrogram procedure on surrogate data.

Save:

```
{pt}_spec_surr_0_100
```

---

## 7. Envelope Extraction (Per Frequency Band)

For each band `[f_low, f_high]`:

### 7.1 Real Data Envelope

For each channel:

1. Identify frequency indices:

   ```python
   freqidx1 = first F > f_low
   freqidx2 = first F >= f_high
   ```
2. Average power across frequency bins:

   ```python
   env[:, ch] = mean(S[:, freqidx1:freqidx2+1, ch], axis=1)
   ```
3. Remove outliers using **GESD**
4. Fill removed points via **linear interpolation**

Save:

```
{pt}_env_{f_high}hz
```

---

### 7.2 Surrogate Envelope

Repeat the same steps using surrogate spectrogram `Ss`.

Save:

```
{pt}_env_surr_{f_high}hz
```

---

## 8. LASSO Parameters

Envelope sampling rate:

```python
Fs_env = 1 / (T[1] - T[0])
```

Parameters:

```python
wintime = 120      # seconds (2 minutes)
winjump = 30       # seconds
lambda2 = 2**np.arange(0,5) * 1e-2
```

---

## 9. Real LASSO-Q Analysis

Inputs:

```python
data.env = env
data.T1  = T
```

Run:

```python
spt_lasso_Q(data, params)
```

Output:

```
qmcube  # real MO strength estimates
```

Saved as:

```
{pt}_q_{f_high}hz
```

---

## 10. Surrogate LASSO Analysis

Compute:

```python
qvars = [
  wintime,
  Fs_env,
  wintime * Fs_env,
  winjump * Fs_env
]
```

Run:

```python
lassoHIE(env_surr, Ts, savename, qvars)
```

Output:

```
q_surr  # surrogate q-values
```

---

## 11. Statistical Significance

For each channel:

```python
mu0    = mean(q_surr, axis=0)
sigma0 = std(q_surr, axis=0, ddof=0)
p_val  = 1 - normcdf(q_real, mu0, sigma0)
```

Save:

```
{pt}_p_{f_high}hz
```

---

## Output Summary (Per Patient)

```
spec_real_0_100
spec_surr_0_100
env_3hz, env_8hz, env_15hz, env_30hz
env_surr_*
q_*
qsurr_*
p_*
```

---

## Critical Replication Constraints

* Bipolar montage **must match MATLAB indices**
* Phase randomization **must preserve conjugate symmetry**
* Envelope sampling rate comes from **spectrogram time bins**
* Outlier removal uses **GESD**, not z-score clipping
* Surrogate significance assumes **Gaussian q-distribution**
* MATLAB `std(X,1)` corresponds to Python `std(..., ddof=0)`

---

## Notes

This document intentionally mirrors MATLAB behavior exactly. Any deviation (e.g., different spectrogram method, different outlier detection) will alter MO statistics and invalidate comparisons.
