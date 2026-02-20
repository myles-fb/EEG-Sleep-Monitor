We need to test the MOs algorithm here against the MOs algorithms from Matlab.

To do this, we'll use the same data set: 10 .edf files, 1 per patient.

## Pipeline (implemented)

- **Bipolar montage:** 18-channel longitudinal bipolar (LB-18) from `documentation/Bipolar_Montage_Code.md` is applied to the EDF channel list; channel names must match (e.g. `EEGFp1Ref`, `EEGF7Ref`, …).
- **MOs:** Data are run through `src/processing/mos.py` via `compute_mos_for_bucket()`.
- **Envelope windows:** Two configurations:
  - **2 minute:** window = 120 s, step = 30 s (1/4 × window).
  - **5 minute:** window = 300 s, step = 75 s (1/4 × window).
- **Output:** One JSON file per EDF per config is written (e.g. `{patient_id}_envelope_2min.json`, `{patient_id}_envelope_5min.json`) so results can be used for visuals later.

## How to run

1. Install dependencies: `pip install -r requirements.txt` (includes `mne` for EDF loading).
2. Place the 10 `.edf` files in a directory (e.g. `data/edfs/`).
3. From the project root:
   ```bash
   python scripts/run_mos_edf_pipeline.py --input-dir data/edfs --output-dir data/mos_results
   ```
4. Optional: `--channel-index 0` (which bipolar channel to report).

Output JSON files include `q_per_band`, `p_per_band`, `q_per_window_per_band`, `dominant_freq_hz_per_window_per_band`, and `dominant_freq_hz_per_band` for downstream visualization and comparison with MATLAB.

