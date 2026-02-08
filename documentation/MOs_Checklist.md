# MOs Algorithm Fidelity Verification Checklist

**Goal:** Verify that the implemented MOs (Modulatory Oscillations) algorithm remains faithful to *Detecting slow narrowband modulation in EEG signals* (Loe et al., 2022), with particular attention to Equations (1–10) and Sections 2.1, 3.1–3.3.

---

## A) Inputs & Preprocessing (Paper Section 2.1)

1. **Spectrogram parameters**

   * Confirm the real EEG spectrogram is computed using **multi-taper spectral estimation** with:

     * Window length: **30 s**
     * Step size: **6 s** (24 s overlap)

2. **Band-limited envelope definition (Eq. 1)**

   * Confirm envelope signals are computed by **averaging spectrogram power across frequency bins** for each band:

     * 0.5–3 Hz
     * 3–8 Hz
     * 8–15 Hz
     * 15–30 Hz

3. **Envelope smoothing**

   * Confirm each envelope time series is smoothed using a **1-minute moving average** prior to sparse spectral analysis.

---

## B) Envelope Windowing (Paper Eq. 2)

4. **Correct windowed envelope construction**

   * Confirm the algorithm constructs windowed envelope vectors:

     [ \Gamma_{k,\rho}(i) = [S_{k,\rho}(iT_m), \dots, S_{k,\rho}(iT_m + N_m)] ]

5. **Correct interpretation of parameters**

   * Verify **Tm and Nm apply to envelope windows**, *not* to the spectrogram.
   * Ensure the implementation does **not** incorrectly set:

     * Tm = 30 s
     * Nm = 6 s

6. **Paper-default envelope windowing**

   * At envelope sampling rate (f_s = 1/6) Hz, confirm defaults match Section 4:

     * **Nm = 400 samples** (~40 min)
     * **Tm = 100 samples** (~10 min)

---

## C) Sparse Spectral Estimation (Paper Eq. 3)

7. **Dictionary construction (X)**

   * Confirm X is a **sinusoidal / inverse-DFT-style dictionary** spanning modulation frequencies defined as:

     * (f_{\min} = 2 f_s / N_m)
     * (f_{\max} = f_s / 4)
     * Harmonic spacing of (f_{\min}/4)

8. **Base LASSO objective**

   * Confirm the sparse estimate solves:

     [ \min_b |\Gamma_{k,\rho}(i) - Xb|_2^2 + \lambda_1 |b|_1 ]

---

## D) Spatiotemporal Filtering / Dynamic Prior (Paper Eqs. 4–5)

9. **Temporal regularization term (λ₂)**

   * Confirm the objective includes:

     [ \lambda_2 |b - \beta_{k,\rho}(i-1)|_2^2 ]

10. **Spatial neighbor aggregation (Eq. 5)**

    * Confirm the prior is computed as:

      [ \beta_{k,\rho}(i-1) = \sum_{l \in C_k} c_{l,k} , \beta_{l,\rho}(i-1) ]
    * Ensure:

      * Neighbor set (C_k) is explicitly defined
      * Weights (c_{l,k}) are explicitly specified

---

## E) Augmented Regression Formulation (Paper Eq. 10)

11. **Correct regression reformulation**

    * Confirm the problem is rewritten as a standard LASSO with:

      * (\hat{\Gamma} = [\Gamma;; \lambda_2 \beta(i-1)])
      * (\hat{X} = [X;; \lambda_2 I])

---

## F) Sliding-Window Mechanics (Paper Section 3.3)

12. **Per-window regression**

    * Confirm the LASSO is solved **independently for each window index i**, for:

      * Each channel k
      * Each frequency band ρ
    * Verify no global (across-window) regression is used.

---

## G) Modulation Index q (Paper Eqs. 6–9)

13. **Pearson correlation (Eq. 6)**

    * Confirm computation of:

      [ r_{k,\rho}(i) = \mathrm{corr}(\Gamma_{k,\rho}(i), X\beta_{k,\rho}(i)) ]

14. **Pseudo-entropy (Eqs. 7–8)**

    * Confirm pseudo-entropy is computed from **absolute coefficient magnitudes** with the paper’s normalization.

15. **Modulation index definition (Eq. 9)**

    * Confirm:

      [ q_{k,\rho}(i) = r_{k,\rho}(i) \cdot (1 - h_{k,\rho}(i)) ]

16. **Dominant frequency (optional but typical)**

    * If implemented, confirm the dominant modulation frequency is the **frequency of the maximum-magnitude coefficient** in β.

---

## H) λ₁ Path & Solver Method (Paper Section 3.3)

17. **λ₁ regularization path**

    * Confirm solutions are computed for:

      [ \lambda_1 \in [0, \lambda_{\max}] ]
    * Where (\lambda_{\max}) yields a solution with **one nonzero coefficient**.

18. **Coordinate descent solver**

    * Confirm the LASSO is solved using **(simultaneous) coordinate descent**, consistent with Friedman et al. (2010).

19. **Discrete λ₂ scan**

    * Confirm λ₂ is iterated over:

      [ {0, 0.01, 0.02, 0.04, 0.08, 0.16} ]
    * And for each λ₂, a full λ₁ path is solved.

---

## I) Parameter Selection Criterion

20. **Objective-based selection**

    * Confirm (\lambda_1) and (\lambda_2) are selected to **maximize the modulation index q** for each:

      * Window i
      * Channel k
      * Frequency band ρ

---

**End of checklist.**
