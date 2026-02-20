# MOs Algorithm Fidelity Verification Report

**Source:** `documentation/MOs_Checklist.md` (Loe et al., 2022, Eqs. 1–10, Sections 2.1, 3.1–3.3)  
**Implementation:** `src/processing/mos.py`  
**Date:** Verification against current codebase.

---

## Summary

| Section | Items | Pass | Partial | Fail |
|---------|-------|------|---------|------|
| A) Inputs & Preprocessing | 1–3 | 3 | 0 | 0 |
| B) Envelope Windowing | 4–6 | 2 | 0 | 1 |
| C) Sparse Spectral Estimation | 7–8 | 2 | 0 | 0 |
| D) Spatiotemporal Filtering | 9–10 | 0 | 0 | 2 |
| E) Augmented Regression | 11 | 0 | 0 | 1 |
| F) Sliding-Window Mechanics | 12 | 1 | 0 | 0 |
| G) Modulation Index q | 13–16 | 4 | 0 | 0 |
| H) λ₁ Path & Solver | 17–19 | 2 | 0 | 0 |
| I) Parameter Selection | 20 | 1 | 0 | 0 |

**Overall:** Implementation matches the checklist for **inputs, spectrogram, envelope definition, dictionary, base LASSO, per-window regression, modulation index formula, q-max selection, dominant frequency (item 16), and λ₁ path (item 17)**. It uses **sklearn lasso_path** (MATLAB lasso–style geometric path); **no λ₂** is used (item 19 N/A by design). It **deviates** on envelope window defaults (shorter windows) and **does not implement** the paper’s temporal prior (Eq. 4), spatial prior (Eq. 5), or augmented regression (Eq. 10).

---

## A) Inputs & Preprocessing (Paper Section 2.1)

### 1. Spectrogram parameters — **PASS**

- **Checklist:** Multi-taper spectral estimation; window 30 s, step 6 s (24 s overlap).
- **Code:** `multitaper_spectrogram()` uses `SPECTROGRAM_WINDOW_SEC = 30`, `SPECTROGRAM_STEP_SEC = 6`; DPSS tapers via `scipy.signal.windows.dpss`; power averaged across tapers.
- **Location:** `mos.py` lines 39–40, 154–218.

### 2. Band-limited envelope definition (Eq. 1) — **PASS**

- **Checklist:** Envelope = average spectrogram power across frequency bins per band: 0.5–3, 3–8, 8–15, 15–30 Hz.
- **Code:** `MO_FREQ_BANDS = [(0.5, 3), (3, 8), (8, 15), (15, 30)]`; `extract_band_envelope()` computes `np.mean(S[:, idx1:idx2, :], axis=1)` per band.
- **Location:** `mos.py` lines 29–34, 254–274.

### 3. Envelope smoothing — **PASS**

- **Checklist:** 1-minute moving average on envelope before sparse spectral analysis.
- **Code:** `_movmean(env, MOVMEAN_WIN, axis=0)` with `MOVMEAN_WIN = 10`. At envelope rate Fs_env = 1/6 Hz (one sample per 6 s), 10 samples = 60 s = 1 minute.
- **Location:** `mos.py` lines 53, 355, 463.

---

## B) Envelope Windowing (Paper Eq. 2)

### 4. Correct windowed envelope construction — **PASS**

- **Checklist:** Γ_{k,ρ}(i) = [S_{k,ρ}(iT_m), …, S_{k,ρ}(iT_m + N_m)] (windowed envelope vectors).
- **Code:** Per-window segments built as `s0 = envelope[start:start+win_samples]` with `win_samples = round(wintime_sec * Fs_env)` and starts at `i * jump_samples` (and last window at end). Same structure as windowed envelope.
- **Location:** `mos.py` lines 378–398.

### 5. Correct interpretation of parameters — **PASS**

- **Checklist:** Tm and Nm apply to **envelope** windows, not spectrogram; do **not** set Tm = 30 s, Nm = 6 s.
- **Code:** Window/step are in envelope time: `wintime_sec` and `winjump_sec` are converted to envelope samples via `Fs_env`. Spectrogram uses separate `SPECTROGRAM_WINDOW_SEC`/`SPECTROGRAM_STEP_SEC` (30 s / 6 s). No confusion with 30 s / 6 s for LASSO.
- **Location:** `mos.py` lines 39–40 vs 43–44, 378–384.

### 6. Paper-default envelope windowing — **FAIL**

- **Checklist:** At f_s = 1/6 Hz, Nm = 400 samples (~40 min), Tm = 100 samples (~10 min).
- **Code:** `LASSO_WINTIME_SEC = 120`, `LASSO_WINJUMP_SEC = 30`. At 1/6 Hz that is 20 envelope samples per window and 5-sample step (~10 min step, ~20 min window), not 400 and 100.
- **Location:** `mos.py` lines 43–44.
- **Note:** Implementation follows MATLAB `lassoHIE` (e.g. 300 s window in vars). To match paper Section 4 exactly, Nm and Tm (in envelope samples or equivalent seconds) would need to be 400 and 100 at 1/6 Hz.

---

## C) Sparse Spectral Estimation (Paper Eq. 3)

### 7. Dictionary construction (X) — **PASS**

- **Checklist:** Sinusoidal / inverse-DFT-style dictionary; f_min = 2 f_s / N_m, f_max = f_s / 4, harmonic spacing f_min/4.
- **Code:** `_build_sinusoid_dictionary()`: `f_min = 2.0 / wintime_sec`, `f_max = Fs_env / 4.0`, `f_step = 0.5 / wintime_sec`. With Nm = win_samples = wintime_sec * Fs_env, 2*Fs_env/Nm = 2/wintime_sec ✓; 0.5/wintime_sec = (2/wintime_sec)/4 = f_min/4 ✓. Columns are sin(ωt + phase) for each candidate frequency.
- **Location:** `mos.py` lines 66–98.

### 8. Base LASSO objective — **PASS**

- **Checklist:** min_b |Γ − Xb|² + λ₁|b|₁.
- **Code:** `Lasso(alpha=lam, fit_intercept=False)`; `model.fit(A, s0)` with s0 = windowed envelope, A = dictionary. sklearn Lasso minimizes ||y − Xb||² + α||b||₁.
- **Location:** `mos.py` lines 408–410.

---

## D) Spatiotemporal Filtering / Dynamic Prior (Paper Eqs. 4–5)

### 9. Temporal regularization term (λ₂) — **FAIL**

- **Checklist:** Objective includes λ₂|b − β_{k,ρ}(i−1)|² (temporal prior).
- **Code:** No temporal prior. The variable `lambda2` is used as the **L1 (alpha)** penalty in separate Lasso fits (grid over regularization), not as weight for a prior β(i−1). No state β(i−1) is maintained across windows.
- **Location:** `mos.py` 362–433; only `Lasso(alpha=lam, ...)` is used.

### 10. Spatial neighbor aggregation (Eq. 5) — **FAIL**

- **Checklist:** Prior β_{k,ρ}(i−1) = Σ_{l∈C_k} c_{l,k} β_{l,ρ}(i−1); neighbor set C_k and weights c_{l,k} defined.
- **Code:** No spatial prior or channel coupling; each channel is processed independently in `_lasso_mo_q_single_channel` / `lasso_mo_q`. No C_k or c_{l,k}.
- **Location:** `mos.py` 359–434, 439–471.

---

## E) Augmented Regression Formulation (Paper Eq. 10)

### 11. Correct regression reformulation — **FAIL**

- **Checklist:** Rewrite as standard LASSO with Γ̂ = [Γ; √λ₂ β(i−1)], X̂ = [X; √λ₂ I].
- **Code:** No augmented design; only single Lasso on (A, s0). No stacking with prior or identity block.
- **Location:** `mos.py` 408–410.

---

## F) Sliding-Window Mechanics (Paper Section 3.3)

### 12. Per-window regression — **PASS**

- **Checklist:** LASSO solved independently per window index i, per channel k, per band ρ; no global regression.
- **Code:** Loop over `starts` (window starts); for each start, `s0 = envelope[start:end]`, fit Lasso(A, s0). Separate loop over channels and bands. No cross-window or global design matrix.
- **Location:** `mos.py` 398–431, 451–470, 415–424.

---

## G) Modulation Index q (Paper Eqs. 6–9)

### 13. Pearson correlation (Eq. 6) — **PASS**

- **Checklist:** r_{k,ρ}(i) = corr(Γ_{k,ρ}(i), X β_{k,ρ}(i)).
- **Code:** `J = np.corrcoef(fitted, s0)[0, 1]` with `fitted = A @ bb`, s0 = windowed envelope. Same quantity.
- **Location:** `mos.py` 411–414.

### 14. Pseudo-entropy (Eqs. 7–8) — **PASS**

- **Checklist:** Pseudo-entropy from absolute coefficient magnitudes with paper’s normalization.
- **Code:** `take_entropy_lasso()`: `x = np.abs(x)`, prob = x/sum(x), entropy = −Σ p log p normalized by log(num_freqs). Uses absolute strengths (polar magnitudes per frequency).
- **Location:** `mos.py` 100–116, 315–324, 422–424.

### 15. Modulation index definition (Eq. 9) — **PASS**

- **Checklist:** q_{k,ρ}(i) = r_{k,ρ}(i) · (1 − h_{k,ρ}(i)).
- **Code:** `q = J_arr * (1.0 - q_entropy)` then `qm = max(q)` per window.
- **Location:** `mos.py` 425–428.

### 16. Dominant frequency — **PASS**

- **Checklist:** If implemented, dominant modulation frequency = frequency of maximum-magnitude coefficient in β.
- **Code:** For the λ that maximizes q per window, dominant frequency = `a_freqs[argmax(f_hold)]` (frequency of max polar magnitude). Exposed in `MOsResult.dominant_freq_hz_per_window_per_band` (per-window Hz) and `MOsResult.dominant_freq_hz_per_band` (dominant freq at the window where q is max).
- **Location:** `mos.py` `_lasso_mo_q_single_channel` (dominant_freq_per_window), `MOsResult`, `compute_mos_for_bucket`, `processor._store_mos_result`.

---

## H) λ₁ Path & Solver Method (Paper Section 3.3)

### 17. λ₁ regularization path — **PASS**

- **Checklist:** Solutions for λ₁ ∈ [0, λ_max] with λ_max such that solution has one nonzero coefficient.
- **Code:** Uses **sklearn.linear_model.lasso_path** (MATLAB lasso–style): geometric sequence of alphas, default `n_alphas=100`, `eps=1e-3` (alpha_min/alpha_max). One path per window; λ₁ selected by maximizing q along the path.
- **Location:** `mos.py` `_lasso_mo_q_single_channel` (lasso_path call), no λ₂.

### 18. Coordinate descent solver — **PASS**

- **Checklist:** LASSO solved with (simultaneous) coordinate descent (Friedman et al. 2010).
- **Code:** `sklearn.linear_model.lasso_path` uses coordinate descent internally.
- **Location:** `mos.py` (lasso_path).

### 19. Discrete λ₂ scan — **N/A (by design)**

- **Checklist:** λ₂ ∈ {0, 0.01, …}; for each λ₂, full λ₁ path.
- **Code:** **No λ₂** is used (no temporal prior). Single λ₁ path per window, as in MATLAB `lasso(A,s0)`.
- **Location:** `mos.py` (no λ₂ parameter).

---

## I) Parameter Selection Criterion

### 20. Objective-based selection — **PASS**

- **Checklist:** λ₁ and λ₂ chosen to maximize modulation index q for each window i, channel k, band ρ.
- **Code:** For each window, q is computed for each λ on the path; `qm_window = max(q)` (max over λ). λ₁ is chosen to maximize q; no λ₂.
- **Location:** `mos.py` (q = J * (1 - q_entropy), qm = max(q), qi = argmax(q)).

---

## Recommendations

1. **Envelope window defaults (item 6):** To align with paper Section 4, add an option (e.g. paper_defaults=True) that sets window and step in envelope samples to Nm=400, Tm=100 at Fs_env=1/6 Hz (~40 min window, ~10 min step), or document that current defaults follow the MATLAB pipeline.
2. **Temporal and spatial priors (items 9–11):** Not implemented by design (per user). To match Eqs. 4–5 and 10 in future work, implement per-window state β(i−1), temporal term λ₂|b − β(i−1)|², optional spatial prior, and augmented LASSO formulation.

---

**End of verification report.**
