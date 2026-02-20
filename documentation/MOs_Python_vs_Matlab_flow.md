# Python vs MATLAB MOs pipeline: flow comparison and runtime

## Flow comparison

### MATLAB (from Lasso_Matlab_Code.md + inferred)

1. **Upstream (not in lassoHIE):** Spectrogram and envelope are computed elsewhere; `lassoHIE` receives **precomputed** `env` (n_times × 18) and `T1`.
2. **Per band:** One call to `lassoHIE(env, T1, fname, vars)`.
3. **Inside lassoHIE:**
   - `s = movmean(env, 10, 1)` once for full envelope.
   - **`parfor t = 1:num_funs`** — windows are parallelized.
   - Per window: `[bb, ss] = lasso(A, s0)` — **one** MATLAB `lasso()` call per window; MATLAB returns the **entire λ path** in one shot (number of λs is MATLAB’s default, often ~100).
   - Post-process path (J, f_hold, q_entropy, q) from the single `bb` matrix.
4. **Surrogates:** Done outside `lassoHIE`. How many and whether spectrogram is redone per surrogate is not in the snippet; typically: generate surrogate EEG → spectrogram → envelope → call `lassoHIE(env_surr, ...)`.

So in MATLAB: **one lasso() call per (window, channel)**; that call returns the full path; **parfor over windows**.

---

### Python (mos.py + run_mos_edf_pipeline.py)

1. **Full EDF as one bucket:** The script passes the **entire** loaded EDF (`data`) to `compute_mos_for_bucket`. So one huge “bucket” (e.g. 30–60 min of data).
2. **Per band** (4 bands):
   - **Real:** One multitaper spectrogram on full bucket → envelope → GESD/interp → `lasso_mo_q(env, T)`.
   - **Per surrogate** (e.g. 1 or 50): **New surrogate EEG** → **full multitaper spectrogram again** → envelope → GESD/interp → `lasso_mo_q(env_surr, Ts)`.
3. **Spectrogram:** `multitaper_spectrogram` uses **nested Python loops**: for each channel, for each time bin, for each taper → FFT. No `parfor`; pure Python loops.
4. **lasso_mo_q:** For **all 18 channels** (even though only `channel_index` is used later):
   - For **each window** (e.g. 100+ for long data): **one `lasso_path(A, s0, n_alphas=100)`** call (sklearn).
   - After the path: Python loops over **every** λ (100) to compute J and f_hold (per-frequency polar magnitude), then entropy and q.
5. **No parallelism:** Everything is single-threaded (no parfor, no joblib).

So in Python: **one lasso_path() per (window, channel)**; each call computes 100 alphas; **no parallelization**; and **full spectrogram repeated (1 + n_surrogates) times per band**.

---

## Why Python is much slower (even with 1 surrogate)

### 1. **Spectrogram is redone per band and per surrogate**

- **Real:** 1 spectrogram per band → 4 spectrograms.
- **Surrogates:** 1 spectrogram per surrogate **per band** → with 1 surrogate, 4 more (8 total); with 50 surrogates, 200.
- Each spectrogram: **double loop** (channels × time bins) in Python, and per (channel, time bin) a loop over 5 tapers + FFT. For a long EDF (e.g. 1 hr at 250 Hz) that’s on the order of **18 × ~600 × 5** FFTs per spectrogram → **~54k FFTs per spectrogram**, all from Python loops (no vectorization over time).

**MATLAB:** Likely one spectrogram per “segment” or per full file, then reuse; and/or optimized/vectorized/compiled routines.

---

### 2. **LASSO: many more path evaluations and post-processing in Python**

- **MATLAB:** `lasso(A, s0)` returns the full path in one optimized (often compiled) call; post-processing is over the returned `bb` columns.
- **Python:** `lasso_path(A, s0, n_alphas=100)` also returns a path, but:
  - **100 alphas** are explicitly requested; similar to MATLAB in count.
  - For **each** of the 100 solutions we then do: `A @ bb`, `corrcoef`, and a **loop over n_freqs** to compute polar magnitude (f_hold). So 100 matrix-vector products + 100 correlation computations + 100 × n_freqs small ops **per window per channel**.
- **Scale:** e.g. 116 windows × 18 channels × 4 bands × (1 real + 1 surr) = **16,704** `lasso_path` calls per run, each with 100 alphas and the post-processing above. So the total number of “path + post-process” steps is large.

---

### 3. **Processing the whole EDF as one bucket**

- Python pipeline runs on the **entire** EDF at once → long envelope → **many windows** (e.g. (T_envelope - win_samples) / jump ≈ 100+).
- If MATLAB runs on **shorter segments** (e.g. 5–10 min), it would have fewer windows per call and less work per run.

---

### 4. **No parallelism**

- MATLAB uses **`parfor t=1:num_funs`** (parallel over windows).
- Python: no equivalent; everything is sequential. So all 16k+ lasso_path calls and all spectrogram FFTs are single-threaded.

---

### 5. **Redundant work: all 18 channels**

- We only need `channel_index` (e.g. 0) for the saved result, but `lasso_mo_q` runs LASSO for **all 18 channels** and we only use one. So we do **18×** the LASSO work needed for the reported channel.

---

### 6. **Per-band surrogate loop**

- For each band we do: real run + **n_surrogates** × (surrogate EEG → spectrogram → envelope → full `lasso_mo_q`).
- So with 1 surrogate we still do **2 full spectrograms per band** (real + 1 surr) and **2 full lasso_mo_q** per band (each over all channels and all windows). That’s 8 spectrograms and 8 “full envelope → LASSO” passes for 4 bands.

---

## Summary table

| Aspect | MATLAB | Python |
|--------|--------|--------|
| Spectrogram | Presumably once (or per segment); may be vectorized/compiled | 4 × (1 + n_surrogates) times; nested Python loops |
| LASSO | One `lasso()` per (window, channel); path in one call; parfor over windows | One `lasso_path(..., n_alphas=100)` per (window, channel); sequential |
| Data length | Possibly chunked (shorter segments) | Full EDF as one bucket → many windows |
| Channels | Same 18 | Same 18, but only 1 used in output (work not pruned) |
| Parallelism | parfor over windows | None |

---

## Recommended changes (to reduce runtime)

1. **Reduce n_alphas:** Use e.g. `lasso_path_n_alphas=50` or 30 (or make it configurable) to cut path size.
2. **Option to run LASSO for one channel only:** If only `channel_index` is needed, run `lasso_mo_q` on `env[:, channel_index:channel_index+1]` (or add a flag to skip other channels).
3. **Vectorize or speed up spectrogram:** Replace the double loop over (channel, time) with batched FFTs or a single call to a routine that does multitaper spectrogram in one go (e.g. a scipy/chronux-style API if available).
4. **Reuse spectrogram for surrogates when possible:** e.g. generate all surrogate envelopes for a band (from one set of surrogate EEGs), then run LASSO on each; avoid recomputing the same surrogate spectrogram in a more expensive way.
5. **Process in shorter buckets:** e.g. 5- or 10-minute chunks instead of the full EDF, then aggregate (fewer windows per run).
6. **Parallelize:** Use `joblib` or `multiprocessing` over windows (or over surrogates) to mimic MATLAB’s parfor.
7. **Fewer surrogates for testing:** Keep 1 (or 5) for quick runs; use 50 only when you need stable p-values.

Implementing (2) and (1) plus (7) will give the largest immediate gains with minimal change to the algorithm.
